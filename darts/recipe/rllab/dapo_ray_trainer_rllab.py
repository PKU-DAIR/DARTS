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


class RayDAPOTrainerRLLab(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.yita = 0.25
    def _get_next_batch(self):
        if not hasattr(self, '_train_iterator'):
            self._train_iterator = iter(self.train_dataloader)
            
        try:
            return next(self._train_iterator)
        except StopIteration:
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
        gen_batch = gen_batch.chunk(int(len(gen_batch)/8))
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
        if not self.config.algorithm.filter_groups.enable:
            # If filtering is disabled, all prompts are valid
            valid_count = len(batch) // self.config.actor_rollout_ref.rollout.n
            return batch, valid_count

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

        # Collect the sequence reward for each trajectory
        prompt_uid2metric_vals = defaultdict(list)
        for uid, metric_val in zip(
            batch.non_tensor_batch["uid"], batch.non_tensor_batch[metric_name], strict=True
        ):
            prompt_uid2metric_vals[uid].append(metric_val)

        prompt_uid2metric_std = {}
        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

        kept_prompt_uids = [
            uid
            for uid, std in prompt_uid2metric_std.items()
            if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
        ]
        
        valid_count = len(kept_prompt_uids)

        kept_traj_idxs = []
        for idx, traj_from_prompt_uid in enumerate(batch.non_tensor_batch["uid"]):
            if traj_from_prompt_uid in kept_prompt_uids:
                kept_traj_idxs.append(idx)

        if not kept_traj_idxs:
            return DataProto(), 0
            
        new_batch = batch[kept_traj_idxs]
        return new_batch, valid_count

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
                                    self.rollout(worker_id)
                                    num_gen_batches += 1

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

                    # === Updating ===
                    ray.get(signal_actor.set_abort.remote(False))
                    batch.batch["response_mask"] = compute_response_mask(batch)

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, "blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, "olive"):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, "brown"):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, "red"):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # validate
                    is_last_step = self.global_steps >= self.total_training_steps
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, "green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with marked_timer("save_checkpoint", timing_raw, "green"):
                            self._save_checkpoint()

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
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
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

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1
