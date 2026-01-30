# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import uuid
import asyncio
import csv
import os
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm
import ray

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.utils.profiler import marked_timer


class RayTranierTest(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.yita = 0.25
    def _get_next_batch(self):
        try:
            if not hasattr(self, '_train_iterator'):
                self._train_iterator = iter(self.train_dataloader)
            return next(self._train_iterator)
        except StopIteration:
            if hasattr(self.train_dataloader, 'next_iter_state'):
                self.train_dataloader.next_iter_state = None
            self._train_iterator = iter(self.train_dataloader)
            return next(self._train_iterator)
        
    def rollout(self, id):
        batch_dict = self._get_next_batch()
        new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                        
        # pop those keys for generation
        if "multi_modal_data" in new_batch.non_tensor_batch.keys():
            gen_batch = new_batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
        )
        else:
            gen_batch = new_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids","raw_prompt"],
            )
        gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
        gen_batch = gen_batch.chunk(int(len(gen_batch)/self.config.actor_rollout_ref.rollout.n))
        new_batch = new_batch.chunk(int(len(new_batch)))
        for i, batch in enumerate(gen_batch):
            task = asyncio.create_task(self.async_rollout_manager.async_generate_sequences_rllab(batch, id))
            self.wait_rollout_list[task] = (id, new_batch[i])
            self.workers_cnt[id] += 1

    def _compute_reward_and_scores(self, batch: DataProto):
        metrics = {}
        if self.use_rm:
            # we first compute reward model score
            reward_tensor = self.rm_wg.compute_rm_score(batch)
            batch = batch.union(reward_tensor)

        # we combine with rule-based rm
        reward_extra_infos_dict: dict[str, list] = {}
        try:
            reward_result = self.reward_fn(batch, return_dict=True)
            reward_tensor = reward_result["reward_tensor"]
            reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
        except Exception as e:
            print(f"Error in reward_fn: {e}")
            reward_tensor = self.reward_fn(batch)
            reward_extra_infos_dict = {}

        batch.batch["token_level_scores"] = reward_tensor

        if reward_extra_infos_dict:
            batch.non_tensor_batch.update(
                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
            )

        # compute rewards. apply_kl_penalty if available
        if self.config.algorithm.use_kl_in_reward:
            batch, kl_metrics = apply_kl_penalty(
                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
            )
            metrics.update(kl_metrics)
        else:
            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
        
        return batch, metrics

    def _apply_dapo_filter(self, batch: DataProto):
        # Calculate metrics if enabled, but do NOT filter samples
        if self.config.algorithm.filter_groups.enable:
            metric_name = self.config.algorithm.filter_groups.metric
            if metric_name == "seq_final_reward":
                # Turn to numpy for easier filtering
                batch.non_tensor_batch["seq_final_reward"] = (
                    batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                )
            elif metric_name == "seq_reward":
                batch.non_tensor_batch["seq_reward"] = (
                    batch.batch["token_level_scores"].sum(dim=-1).numpy()
                )

        # Modified: Keep all samples, do not filter based on std or anything else
        valid_count = len(batch) // self.config.actor_rollout_ref.rollout.n
        return batch, valid_count

    async def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.gen_steps = 0
        
        # Initialize CSV file for logging lengths
        self.length_log_file = "response_lengths2.csv"
        with open(self.length_log_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["global_step", "uid", "response_length", "reward"])

        self.wait_rollout_list = {}
        self.wait_reward_list = {}
        self.workers_num = len(self.async_rollout_manager.agent_loop_workers)
        self.workers_cnt = {i: 0 for i in range(self.workers_num)}
        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None

        timing_raw = defaultdict(float)
        
        # Pre-dispatch tasks to all workers
        print("Initializing rollout workers...")
        signal_actor = ray.get_actor("signal_actor")

        for epoch in range(self.config.trainer.total_epochs):
            while self.global_steps <= self.total_training_steps:
                metrics = {}
                do_profile = (
                    self.global_steps in self.config.trainer.profile_steps
                    if self.config.trainer.profile_steps is not None
                    else False
                )
                
                with marked_timer("start_profile", timing_raw):
                    if do_profile:
                        self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
                        if self.use_reference_policy:
                            self.ref_policy_wg.start_profile()
                        if self.use_critic:
                            self.critic_wg.start_profile()
                        if self.use_rm:
                            self.rm_wg.start_profile()

                # Data Collection Loop
                collected_batch = DataProto()
                num_valid_prompts_collected = 0
                num_gen_batches = 0
                
                with marked_timer("step", timing_raw):
                    if self.config.actor_rollout_ref.rollout.free_cache_engine:
                        self.async_rollout_manager.wake_up()
                    self.async_rollout_manager.reset_prefix_cache()
                    for i in range(self.workers_num):
                        # Avoid over-dispatching if we already have enough pending tasks
                        if sum(self.workers_cnt.values()) >= self.config.data.train_batch_size:
                            break
                        self.rollout(i)
                    while self.wait_rollout_list:
                        # Wait for at least one task to complete
                        done, pending = await asyncio.wait(self.wait_rollout_list.keys(), return_when=asyncio.FIRST_COMPLETED)

                        for task in done:
                            # 1. Get result
                            gen_batch_output = await task
                            worker_id, input_batch = self.wait_rollout_list.pop(task)
                            
                            # 2. Dispatch new task immediately (Pipeline)
                            # Only dispatch if we need more data AND haven't reached max batches
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            max_gen_reached = (max_num_gen_batches > 0 and num_gen_batches >= max_num_gen_batches)
                            self.workers_cnt[worker_id] -= 1
                            if num_valid_prompts_collected < self.config.data.train_batch_size and not max_gen_reached: 
                
                                if self.workers_cnt[worker_id] == 0:  # keep some tasks in the pipeline
                                    #self.rollout(worker_id)
                                    num_gen_batches += 1

                                # Try to find an existing ID in the batch
                                id_keys = ["uid", "id", "question_id", "problem_id", "sample_id"]
                                found_id = False
                                for key in id_keys:
                                    if key in input_batch.non_tensor_batch:
                                        input_batch.non_tensor_batch["uid"] = input_batch.non_tensor_batch[key]
                                        found_id = True
                                        break
                                
                                if not found_id:
                                    input_batch.non_tensor_batch["uid"] = np.array(
                                        [str(uuid.uuid4()) for _ in range(len(input_batch.batch))], dtype=object
                                    )

                                input_batch = input_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                                if "timing" in gen_batch_output.meta_info:
                                    metrics.update(gen_batch_output.meta_info["timing"])
                                    gen_batch_output.meta_info.pop("timing")
                                batch_chunk = input_batch.union(gen_batch_output)
                                
                                with marked_timer("reward", timing_raw, "yellow"):
                                    batch_chunk, reward_metrics = self._compute_reward_and_scores(batch_chunk)
                                    metrics.update(reward_metrics)
                                
                                batch_chunk, valid_count = self._apply_dapo_filter(batch_chunk)
                                
                                if batch_chunk is not None and len(batch_chunk) > 0:
                                    collected_batch = DataProto.concat([collected_batch, batch_chunk]) if collected_batch.batch is not None else batch_chunk
                                    num_valid_prompts_collected += valid_count
                                    print(f"Collected {num_valid_prompts_collected} valid prompts so far.")
                                
                                if num_valid_prompts_collected >= self.config.data.train_batch_size:
                                    ray.get(signal_actor.set_abort.remote(True))
                                    break
                                if max_gen_reached and num_valid_prompts_collected < self.config.data.train_batch_size:
                                    print(f"Warning: Reached max_num_gen_batches ({max_num_gen_batches}) but only collected {num_valid_prompts_collected} valid prompts.")

                    if self.config.actor_rollout_ref.rollout.free_cache_engine:
                        self.async_rollout_manager.sleep()

                    # If we didn't collect enough data and stopped due to max_num_gen_batches, we might want to skip update or proceed
                    if collected_batch.batch is None or len(collected_batch) == 0:
                        print("No valid data collected. Skipping step.")
                        continue

                    # Align the batch size
                    batch = collected_batch
                    prompt_bsz = self.config.data.train_batch_size
                    if num_valid_prompts_collected > prompt_bsz:
                         # We might have collected slightly more than needed, truncate
                         # Note: This truncation might be tricky if we want to keep n_rollouts together.
                         # But since we concat chunks which are already n_rollouts aligned, we just need to truncate by n_rollouts * N
                         traj_bsz = prompt_bsz * self.config.actor_rollout_ref.rollout.n
                         batch = batch[:traj_bsz]

                    # === Analysis Only (No Training) ===
                    ray.get(signal_actor.set_abort.remote(False))
                    batch.batch["response_mask"] = compute_response_mask(batch)
                    
                    # Calculate length distribution
                    response_length = batch.batch["response_mask"].sum(dim=-1).float()

                    # Save lengths to CSV
                    uids = batch.non_tensor_batch["uid"]
                    lengths = response_length.tolist()
                    rewards = batch.batch["token_level_rewards"].sum(dim=-1).tolist()
                    with open(self.length_log_file, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        for uid, length, reward in zip(uids, lengths, rewards):
                            writer.writerow([self.global_steps, uid, length, reward])

                    length_dist = {
                        "min": response_length.min().item(),
                        "max": response_length.max().item(),
                        "mean": response_length.mean().item(),
                        "std": response_length.std().item(),
                        "p50": torch.quantile(response_length, 0.5).item(),
                        "p90": torch.quantile(response_length, 0.9).item(),
                        "p99": torch.quantile(response_length, 0.99).item(),
                    }
                    print(f"Step {self.global_steps} Response Length Distribution: {length_dist}")
                    metrics.update({f"length_dist/{k}": v for k, v in length_dist.items()})

                with marked_timer("stop_profile", timing_raw):
                    if do_profile:
                        self.actor_rollout_wg.stop_profile()
                        if self.use_reference_policy:
                            self.ref_policy_wg.stop_profile()
                        if self.use_critic:
                            self.critic_wg.stop_profile()
                        if self.use_rm:
                            self.rm_wg.stop_profile()

                # collect metrics
                # metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)


                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1
