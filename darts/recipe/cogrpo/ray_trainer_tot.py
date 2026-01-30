import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Optional
import copy
import numpy as np
import ray
import torch
from tensordict import TensorDict
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    ResourcePoolManager,
    Role,
    WorkerType,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.debug import marked_timer
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.tracking import ValidationGenerationsLogger
import asyncio

WorkerType = type[Worker]
from recipe.one_step_off_policy.ray_trainer_stream import OneStepOffRayTrainer
from recipe.one_step_off_policy.save_metrics import _save_metrics_to_csv 

def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.reweight_method,
                config.pf_ppo.weight_pow,
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class ToTRayTrainer(OneStepOffRayTrainer):

    @staticmethod
    def _safe_copy_dataproto_no_pickle(dp: DataProto) -> DataProto:
        """Copy a DataProto without triggering DataProto.__getstate__.

        `deepcopy(DataProto)` may go through pickle and call `DataProto.__getstate__`, which
        (for tensordict>=0.5) calls `TensorDict.consolidate()`. If the TensorDict is empty
        (common after popping all tensor keys for generation), consolidate can crash with:
        `torch.cat(): expected a non-empty list of Tensors`.
        """
        if dp.batch is None:
            batch_copy = None
        else:
            batch_copy = TensorDict({}, batch_size=dp.batch.batch_size, device=dp.batch.device)
            for k, v in dp.batch.items():
                batch_copy[k] = v.clone()

        return type(dp)(
            batch=batch_copy,
            non_tensor_batch=copy.deepcopy(dp.non_tensor_batch),
            meta_info=copy.deepcopy(dp.meta_info),
        )

    def _entropy_rebranch_enabled(self, batch: DataProto) -> bool:
        """Enable entropy-based rebranching for a subset of data.

        Priority:
        1) batch.non_tensor_batch["entropy_rebranch"] if present
        2) env var ENABLE_ENTROPY_REBRANCH=1
        3) config.trainer.entropy_rebranch.enable (if exists)
        """
        return True
        try:
            if "entropy_rebranch" in batch.non_tensor_batch:
                v = batch.non_tensor_batch["entropy_rebranch"]
                if isinstance(v, (np.ndarray, list)):
                    v = v[0]
                return bool(v)
        except Exception:
            pass

        if os.environ.get("ENABLE_ENTROPY_REBRANCH", "0") in ("1", "true", "True"):
            return True

        try:
            cfg = self.config.trainer.get("entropy_rebranch", None)
            if cfg is None:
                return False
            return bool(cfg.get("enable", False))
        except Exception:
            return False

    def _safe_argmax_entropy_pos(
        self,
        entropys: torch.Tensor,
        response_mask: torch.Tensor,
        min_pos: int = 0,
    ) -> torch.Tensor:
        """Return per-sample position of max entropy token (within mask and >= min_pos).

        entropys/response_mask shape: (bs, resp_len)
        """
        bs, resp_len = entropys.shape
        device = entropys.device
        pos_idx = torch.arange(resp_len, device=device).unsqueeze(0).expand(bs, -1)
        valid = (response_mask > 0) & (pos_idx >= int(min_pos))

        # Use -inf for invalid positions.
        masked_ent = entropys.masked_fill(~valid, float("-inf"))
        best_pos = torch.argmax(masked_ent, dim=-1)

        # If a row had no valid tokens, fall back to the last masked token (or 0).
        no_valid = torch.isinf(masked_ent).all(dim=-1)
        if torch.any(no_valid):
            last_valid = torch.clamp(torch.sum((response_mask > 0).to(torch.long), dim=-1) - 1, min=0)
            best_pos = torch.where(no_valid, last_valid, best_pos)
        return best_pos

    def _build_rebranch_gen_batch(
        self,
        raw_prompts: np.ndarray,
        responses: torch.Tensor,
        cut_pos: torch.Tensor,
        n: int,
        keep_per_prompt: int,
        rollout_per_kept: int = 4,
        append_mode: str = "append_assistant",
        rng: Optional[np.random.Generator] = None,
    ) -> DataProto:
        """Create a new generation batch by truncating each trajectory at `cut_pos` and re-rolling out.

        We randomly keep `keep_per_prompt` trajectories per original prompt, and replicate each kept
        prompt `rollout_per_kept` times, so total per prompt becomes keep_per_prompt * rollout_per_kept.

        append_mode:
          - "append_assistant": append a new assistant message with prefix text.
          - "append_to_last_assistant": append prefix text onto the last assistant message.
        """

        prompt_length = int(self.config.actor_rollout_ref.rollout.prompt_length)
        if rng is None:
            rng = np.random.default_rng()

        if raw_prompts is None:
            raise RuntimeError("raw_prompt is required for entropy rebranching, but it's missing.")

        total = responses.size(0)
        if total % n != 0:
            raise ValueError(f"Entropy rebranch expects total trajectories divisible by n. Got total={total}, n={n}")
        batch_size = total // n

        if keep_per_prompt <= 0:
            keep_per_prompt = max(1, n // 4)
        if keep_per_prompt * rollout_per_kept != n:
            # Keep it runnable: adjust keep_per_prompt if needed.
            if n % rollout_per_kept == 0:
                keep_per_prompt = n // rollout_per_kept
            else:
                raise ValueError(
                    f"Need keep_per_prompt*{rollout_per_kept}==n to keep batch size stable. Got keep={keep_per_prompt}, n={n}"
                )

        msgs_list = []
        for q in range(batch_size):
            group_start = q * n
            group_indices = np.arange(group_start, group_start + n)
            chosen = rng.choice(group_indices, size=keep_per_prompt, replace=False)

            for idx in chosen.tolist():
                prefix_ids = responses[idx, : int(cut_pos[idx].item())]
                base = raw_prompts[idx]
                # In agent_loop, each element is usually a numpy array of message dicts.
                # Normalize to list[dict] first.
                if isinstance(base, np.ndarray):
                    base_msgs = base.tolist()
                elif isinstance(base, list):
                    base_msgs = base
                elif isinstance(base, dict):
                    base_msgs = [base]
                else:
                    base_msgs = [{"role": "user", "content": str(base)}]
                base_msgs = copy.deepcopy(base_msgs)

                def _build_msgs_with_prefix(prefix_text: str) -> list[dict]:
                    msgs = copy.deepcopy(base_msgs)
                    if append_mode == "append_assistant":
                        msgs.append({"role": "assistant", "content": prefix_text})
                    elif append_mode == "append_to_last_assistant":
                        if (
                            len(msgs) > 0
                            and isinstance(msgs[-1], dict)
                            and msgs[-1].get("role") == "assistant"
                        ):
                            msgs[-1]["content"] = str(msgs[-1].get("content", "")) + prefix_text
                        else:
                            msgs.append({"role": "assistant", "content": prefix_text})
                    else:
                        raise ValueError(f"Unknown append_mode={append_mode}")
                    return msgs

                # Ensure prompt token length <= prompt_length for AgentLoopWorker.tokenizer.pad(max_length=prompt_length)
                # by truncating the appended prefix (in token space) via binary search.
                def _token_len(msgs: list[dict]) -> int:
                    return len(
                        self.tokenizer.apply_chat_template(
                            msgs,
                            add_generation_prompt=True,
                            tokenize=True,
                        )
                    )

                # Fast path: if base itself is already too long, fall back to a single user message.
                if _token_len(base_msgs) > prompt_length:
                    base_ids = self.tokenizer.apply_chat_template(
                        base_msgs,
                        add_generation_prompt=True,
                        tokenize=True,
                    )
                    # Keep tail tokens to fit; decode as a user-only prompt.
                    keep = max(1, prompt_length - 64)
                    tail_text = self.tokenizer.decode(base_ids[-keep:], skip_special_tokens=True)
                    base_msgs = [{"role": "user", "content": tail_text}]

                # If no prefix, keep messages as-is.
                max_prefix = int(prefix_ids.numel()) if isinstance(prefix_ids, torch.Tensor) else len(prefix_ids)
                best = 0
                lo, hi = 0, max_prefix
                while lo <= hi:
                    mid = (lo + hi) // 2
                    prefix_sub = prefix_ids[:mid]
                    prefix_text_mid = self.tokenizer.decode(prefix_sub, skip_special_tokens=True)
                    msgs_mid = _build_msgs_with_prefix(prefix_text_mid)
                    if _token_len(msgs_mid) <= prompt_length:
                        best = mid
                        lo = mid + 1
                    else:
                        hi = mid - 1

                prefix_text = self.tokenizer.decode(prefix_ids[:best], skip_special_tokens=True)
                final_msgs = _build_msgs_with_prefix(prefix_text)
                # IMPORTANT: AgentLoopWorker expects each messages object to have `.tolist()`.
                # Store per-sample messages as np.ndarray(dtype=object).
                msg_arr = np.array(final_msgs, dtype=object)
                for _ in range(rollout_per_kept):
                    msgs_list.append(msg_arr.copy())

        # Tokenize
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        prompt_texts = [
            self.tokenizer.apply_chat_template(m.tolist(), tokenize=False, add_generation_prompt=True) for m in msgs_list
        ]
        encodings = self.tokenizer(prompt_texts, padding=True, return_tensors="pt", add_special_tokens=False)
        position_ids = (
            torch.arange(encodings["input_ids"].size(1), device=encodings["input_ids"].device)
            .unsqueeze(0)
            .expand(encodings["input_ids"].size(0), -1)
        )

        gen_dict = {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "position_ids": position_ids,
            "raw_prompt": np.array(msgs_list, dtype=object),
        }
        gen_batch = DataProto.from_single_dict(gen_dict)
        # Ensure raw_prompt_ids does not override our new prompts
        if "raw_prompt_ids" in gen_batch.non_tensor_batch:
            del gen_batch.non_tensor_batch["raw_prompt_ids"]
        return gen_batch

    def _prepare_mixed_prompt_batch(self, gen_batch, n):
        raw_prompts = gen_batch.non_tensor_batch.get("raw_prompt", None)
        if raw_prompts is None:
            print(f"[WARNING] 'raw_prompt' not found in batch keys: {gen_batch.non_tensor_batch.keys()}. Skipping mixed prompt generation.")
            return gen_batch.repeat(repeat_times=n, interleave=True)
        
        if n % 2 != 0:
            print(f"[WARNING] n={n} is not even. Skipping mixed prompt generation.")
            return gen_batch.repeat(repeat_times=n, interleave=True)

        print(f"[INFO] Applying mixed prompts for batch size {len(gen_batch.batch)} with n={n}...")
        
        # 1. Expand the batch first (repeats metadata like raw_prompt, uids etc.)
        gen_batch = gen_batch.repeat(repeat_times=n, interleave=True)
        
        # 2. Prepare new prompts with different system prompts
        sys_prompt_1 = "You are a dedicated math teacher explaining your reasoning step-by-step to help students understand not just how to solve math problems, but why the methods work, fostering a deep and lasting comprehension of mathematical concepts."
        sys_prompt_2 = "Today, I plan to lead my students into solving a complex mathematical problem that requires careful consideration and logical deduction. We will focus on the significance of maintaining a systematic approach and using clear, precise language throughout our analysis."
        
        expanded_raw_prompts = gen_batch.non_tensor_batch["raw_prompt"]
        msgs_list = []

        prompt_length = int(self.config.actor_rollout_ref.rollout.prompt_length)
        
        # We want first n/2 of each sample to be Sys1, and next n/2 to be Sys2.
        for i, raw in enumerate(expanded_raw_prompts):
            use_sys_2 = (i % n) >= (n // 2)
            sys_prompt = sys_prompt_2 if use_sys_2 else sys_prompt_1

            # Normalize raw prompt to list[dict]
            if isinstance(raw, np.ndarray):
                base_msgs = raw.tolist()
            elif isinstance(raw, list):
                base_msgs = raw
            elif isinstance(raw, dict):
                base_msgs = [raw]
            else:
                base_msgs = [{"role": "user", "content": str(raw)}]

            msgs = [{"role": "system", "content": sys_prompt}] + base_msgs
            # If system prompt makes it too long for AgentLoopWorker.pad(max_length=prompt_length), fall back.
            try:
                token_len = len(
                    self.tokenizer.apply_chat_template(
                        msgs,
                        add_generation_prompt=True,
                        tokenize=True,
                    )
                )
            except Exception:
                token_len = prompt_length + 1

            if token_len > prompt_length:
                # Minimal, safe system prompt to avoid overflow.
                msgs = [{"role": "system", "content": "You are a helpful assistant."}] + base_msgs

            # Keep per-sample messages as np.ndarray(dtype=object) for AgentLoopWorker
            msgs_list.append(np.array(msgs, dtype=object))

        # 3. Use tokenizer to handle formatting and padding globally
        # Enforce left-padding for generation
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # First apply template to get strings
        prompt_texts = [self.tokenizer.apply_chat_template(m.tolist(), tokenize=False, add_generation_prompt=True) for m in msgs_list]
        
        # Then tokenize as a batch (handles padding automatically)
        encodings = self.tokenizer(prompt_texts, padding=True, return_tensors='pt', add_special_tokens=False)
        
        # 4. Update the batch tensors
        gen_batch.batch['input_ids'] = encodings['input_ids']
        gen_batch.batch['attention_mask'] = encodings['attention_mask']
        gen_batch.batch['position_ids'] = torch.arange(encodings['input_ids'].size(1), device=encodings['input_ids'].device).unsqueeze(0).expand(encodings['input_ids'].size(0), -1)
        
        # 4.5. CRITICAL: Update raw_prompt in non_tensor_batch
        # This is required because AgentLoopWorker (and vllm_async_server integration) uses 'raw_prompt' 
        # to re-tokenize/apply templates, ignoring our 'input_ids' changes.
        gen_batch.non_tensor_batch["raw_prompt"] = np.array(msgs_list, dtype=object)

        # 5. CRITICAL: Remove raw_prompt_ids from non_tensor_batch
        # The gen_batch inherited 'raw_prompt_ids' from the original batch. 
        # Since we modified input_ids, the old raw_prompt_ids are invalid.
        # Removing them forces vllm_rollout to regenerate them from our new input_ids.
        if "raw_prompt_ids" in gen_batch.non_tensor_batch:
            del gen_batch.non_tensor_batch["raw_prompt_ids"]

        # 6. Verification: Decode and print the first prompt to ensure System Prompt is present
        debug_decoded = self.tokenizer.decode(gen_batch.batch['input_ids'][0], skip_special_tokens=False)
        print("="*20 + " DEBUG PROMPT INFO " + "="*20)
        print(f"[DEBUG] Raw prompt type: {type(expanded_raw_prompts[0])}")
        try:
            print(f"[DEBUG] Raw prompt content sample: {str(expanded_raw_prompts[0])[:500]}...")
        except:
            pass
        print("-" * 10)
        print(f"[DEBUG] Formatted prompt text (just before tokenization):\n{prompt_texts[0]}")
        print("-" * 10)
        print(f"[DEBUG] Decoded input_ids from gen_batch:\n{debug_decoded}")
        print("="*60)
        
        return gen_batch

    def fit(self):
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()
        self.sync_rollout_weights()

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
        last_val_metrics = None
        self.max_steps_duration = 0

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                do_profile = (
                    self.global_steps in self.config.trainer.profile_steps
                    if self.config.trainer.profile_steps is not None
                    else False
                )
                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(do_profile)

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                entropy_rebranch = self._entropy_rebranch_enabled(batch)

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "interaction_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("interaction_kwargs")
                if "index" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("index")
                if "agent_name" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("agent_name")

                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                # Use mixed prompts instead of uniform repeat
                n = int(self.config.actor_rollout_ref.rollout.n)
                #gen_batch = self._prepare_mixed_prompt_batch(gen_batch, n)
                gen_batch = gen_batch.repeat(repeat_times=int(self.config.actor_rollout_ref.rollout.n), interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    def _generate_sequences(gb: DataProto) -> DataProto:
                        if not self.async_rollout_mode:
                            out = self.rollout_wg.generate_sequences(gb)
                        else:
                            out = self.async_rollout_manager.generate_sequences(gb)
                        timing_raw.update(out.meta_info.get("timing", {}))
                        out.meta_info.pop("timing", None)
                        return out

                    def _train_from_gen_batch(gb: DataProto, stage: str) -> tuple[DataProto, torch.Tensor, dict]:
                        """Run one full RL update using generation batch gb.

                        Returns: (trained_batch, per_token_entropys, reward_extra_infos_dict)
                        """
                        # generate
                        with marked_timer(f"gen_{stage}", timing_raw, color="red"):
                            gb_out = _generate_sequences(gb)

                        # Build training batch
                        local_batch = self._safe_copy_dataproto_no_pickle(batch)
                        local_batch.non_tensor_batch["uid"] = np.array(
                            [str(uuid.uuid4()) for _ in range(len(local_batch.batch))], dtype=object
                        )
                        local_batch = local_batch.repeat(repeat_times=n, interleave=True)
                        local_batch = local_batch.union(gb_out)

                        if "response_mask" not in local_batch.batch.keys():
                            local_batch.batch["response_mask"] = compute_response_mask(local_batch)

                        if self.config.trainer.balance_batch:
                            self._balance_batch(local_batch, metrics=metrics)

                        local_batch.meta_info["global_token_num"] = torch.sum(
                            local_batch.batch["attention_mask"], dim=-1
                        ).tolist()

                        # reward
                        with marked_timer(f"reward_{stage}", timing_raw, color="yellow"):
                            if self.use_rm:
                                reward_tensor = self.rm_wg.compute_rm_score(local_batch)
                                local_batch = local_batch.union(reward_tensor)

                            if self.config.reward_model.launch_reward_fn_async:
                                future_reward = compute_reward_async.remote(local_batch, self.config, self.tokenizer)
                            else:
                                reward_tensor, reward_extra_infos_dict = compute_reward(local_batch, self.reward_fn)

                        # old log prob + entropys
                        with marked_timer(f"old_log_prob_{stage}", timing_raw, color="blue"):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(local_batch)
                            per_token_entropys = old_log_prob.batch["entropys"].detach()
                            response_masks = local_batch.batch["response_mask"]
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_agg = agg_loss(
                                loss_mat=per_token_entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                            )
                            metrics.update({f"actor/entropy_{stage}": entropy_agg.detach().item()})
                            old_log_prob.batch.pop("entropys")
                            local_batch = local_batch.union(old_log_prob)

                        if self.use_reference_policy:
                            with marked_timer(f"ref_{stage}", timing_raw, color="olive"):
                                if not self.ref_in_actor:
                                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(local_batch)
                                else:
                                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(local_batch)
                                local_batch = local_batch.union(ref_log_prob)

                        if self.use_critic:
                            with marked_timer(f"values_{stage}", timing_raw, color="cyan"):
                                values = self.critic_wg.compute_values(local_batch)
                                local_batch = local_batch.union(values)

                        with marked_timer(f"adv_{stage}", timing_raw, color="brown"):
                            if self.config.reward_model.launch_reward_fn_async:
                                reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                            local_batch.batch["token_level_scores"] = reward_tensor

                            if reward_extra_infos_dict:
                                local_batch.non_tensor_batch.update(
                                    {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                                )

                            if self.config.algorithm.use_kl_in_reward:
                                local_batch, kl_metrics = apply_kl_penalty(
                                    local_batch,
                                    kl_ctrl=self.kl_ctrl_in_reward,
                                    kl_penalty=self.config.algorithm.kl_penalty,
                                )
                                metrics.update({f"{k}_{stage}": v for k, v in kl_metrics.items()})
                            else:
                                local_batch.batch["token_level_rewards"] = local_batch.batch["token_level_scores"]

                            norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                            local_batch = compute_advantage(
                                local_batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=n,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                config=self.config.algorithm,
                            )

                        if self.use_critic:
                            with marked_timer(f"update_critic_{stage}", timing_raw, color="pink"):
                                critic_output = self.critic_wg.update_critic(local_batch)
                            critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                            metrics.update({f"{k}_{stage}": v for k, v in critic_output_metrics.items()})

                        if self.config.trainer.critic_warmup <= self.global_steps:
                            with marked_timer(f"update_actor_{stage}", timing_raw, color="red"):
                                local_batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                                actor_output = self.actor_rollout_wg.update_actor(local_batch)
                            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                            metrics.update({f"{k}_{stage}": v for k, v in actor_output_metrics.items()})

                        rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                        if rollout_data_dir:
                            with marked_timer(f"dump_rollout_generations_{stage}", timing_raw, color="green"):
                                inputs = self.tokenizer.batch_decode(local_batch.batch["prompts"], skip_special_tokens=True)
                                outputs = self.tokenizer.batch_decode(local_batch.batch["responses"], skip_special_tokens=True)
                                scores = local_batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                                self._dump_generations(
                                    inputs=inputs,
                                    outputs=outputs,
                                    scores=scores,
                                    reward_extra_infos_dict=reward_extra_infos_dict,
                                    dump_path=rollout_data_dir,
                                )

                        with marked_timer(f"sync_weights_{stage}", timing_raw):
                            self.sync_rollout_weights()

                        return local_batch, per_token_entropys, reward_extra_infos_dict

                    if entropy_rebranch:
                        metrics["entropy_rebranch/enabled"] = 1
                        rng = np.random.default_rng(self.global_steps)

                        # Stage 0: initial rollout to locate max-entropy token per trajectory
                        with marked_timer("gen_stage0", timing_raw, color="red"):
                            gen0 = _generate_sequences(gen_batch)

                        # Build a temp batch to compute per-token entropys (no training)
                        tmp0 = self._safe_copy_dataproto_no_pickle(batch)
                        tmp0.non_tensor_batch["uid"] = np.array(
                            [str(uuid.uuid4()) for _ in range(len(tmp0.batch))], dtype=object
                        )
                        tmp0 = tmp0.repeat(repeat_times=n, interleave=True)
                        tmp0 = tmp0.union(gen0)
                        if "response_mask" not in tmp0.batch.keys():
                            tmp0.batch["response_mask"] = compute_response_mask(tmp0)

                        with marked_timer("entropy_stage0", timing_raw, color="blue"):
                            lp0 = self.actor_rollout_wg.compute_log_prob(tmp0)
                            ent0 = lp0.batch["entropys"].detach()

                        cut0 = self._safe_argmax_entropy_pos(ent0, tmp0.batch["response_mask"], min_pos=0)
                        metrics["entropy_rebranch/cut0_mean"] = float(torch.mean(cut0.float()).item())

                        raw0 = gen_batch.non_tensor_batch.get("raw_prompt", None)
                        keep = max(1, n // 4)

                        gen1 = self._build_rebranch_gen_batch(
                            raw_prompts=raw0,
                            responses=tmp0.batch["responses"],
                            cut_pos=cut0,
                            n=n,
                            keep_per_prompt=keep,
                            rollout_per_kept=4,
                            append_mode="append_assistant",
                            rng=rng,
                        )
                        gen1.meta_info["global_steps"] = self.global_steps

                        # Stage 1: train on rebranched rollouts
                        batch1, ent1, _ = _train_from_gen_batch(gen1, stage="stage1")

                        cut1 = self._safe_argmax_entropy_pos(ent1, batch1.batch["response_mask"], min_pos=0)
                        metrics["entropy_rebranch/cut1_mean"] = float(torch.mean(cut1.float()).item())

                        raw1 = gen1.non_tensor_batch.get("raw_prompt", None)
                        gen2 = self._build_rebranch_gen_batch(
                            raw_prompts=raw1,
                            responses=batch1.batch["responses"],
                            cut_pos=cut1,
                            n=n,
                            keep_per_prompt=keep,
                            rollout_per_kept=4,
                            append_mode="append_to_last_assistant",
                            rng=rng,
                        )
                        gen2.meta_info["global_steps"] = self.global_steps

                        # Stage 2: train again on deeper rebranch
                        batch, _, reward_extra_infos_dict = _train_from_gen_batch(gen2, stage="stage2")

                    else:
                        # ===== Original single-step behavior =====
                        # generate a batch
                        with marked_timer("gen", timing_raw, color="red"):
                            gen_batch_output = _generate_sequences(gen_batch)

                        if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                            with marked_timer("gen_max", timing_raw, color="purple"):
                                gen_baseline_batch = deepcopy(gen_batch)
                                gen_baseline_batch.meta_info["do_sample"] = False
                                gen_baseline_output = self.rollout_wg.generate_sequences(gen_baseline_batch)

                                batch = batch.union(gen_baseline_output)
                                reward_baseline_tensor = self.reward_fn(batch)
                                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                                batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                                batch.batch["reward_baselines"] = reward_baseline_tensor

                                del gen_baseline_batch, gen_baseline_output

                        batch.non_tensor_batch["uid"] = np.array(
                            [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                        )
                        # repeat to align with repeated responses in rollout
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(gen_batch_output)

                        if "response_mask" not in batch.batch.keys():
                            batch.batch["response_mask"] = compute_response_mask(batch)

                        if self.config.trainer.balance_batch:
                            self._balance_batch(batch, metrics=metrics)

                        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                        with marked_timer("reward", timing_raw, color="yellow"):
                            # compute reward model score
                            if self.use_rm:
                                reward_tensor = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(reward_tensor)

                            if self.config.reward_model.launch_reward_fn_async:
                                future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                            else:
                                reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                        # recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)

                            if "rollout_log_probs" in batch.batch.keys():
                                rollout_old_log_probs = batch.batch["rollout_log_probs"]
                                actor_old_log_probs = batch.batch["old_log_probs"]
                                attention_mask = batch.batch["attention_mask"]
                                responses = batch.batch["responses"]
                                response_length = responses.size(1)
                                response_mask = attention_mask[:, -response_length:]

                                rollout_probs = torch.exp(rollout_old_log_probs)
                                actor_probs = torch.exp(actor_old_log_probs)
                                rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                                rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                                rollout_probs_diff_max = torch.max(rollout_probs_diff)
                                rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                                rollout_probs_diff_std = torch.std(rollout_probs_diff)
                                metrics.update(
                                    {
                                        "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                        "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                        "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                    }
                                )

                        if self.use_reference_policy:
                            with marked_timer("ref", timing_raw, color="olive"):
                                if not self.ref_in_actor:
                                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                else:
                                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)

                        if self.use_critic:
                            with marked_timer("values", timing_raw, color="cyan"):
                                values = self.critic_wg.compute_values(batch)
                                batch = batch.union(values)

                        with marked_timer("adv", timing_raw, color="brown"):
                            reward_extra_infos_dict: dict[str, list]
                            if self.config.reward_model.launch_reward_fn_async:
                                reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                            batch.batch["token_level_scores"] = reward_tensor

                            if reward_extra_infos_dict:
                                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                            if self.config.algorithm.use_kl_in_reward:
                                batch, kl_metrics = apply_kl_penalty(
                                    batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                                )
                                metrics.update(kl_metrics)
                            else:
                                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                            norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                config=self.config.algorithm,
                            )

                        if self.use_critic:
                            with marked_timer("update_critic", timing_raw, color="pink"):
                                critic_output = self.critic_wg.update_critic(batch)
                            critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                            metrics.update(critic_output_metrics)

                        if self.config.trainer.critic_warmup <= self.global_steps:
                            with marked_timer("update_actor", timing_raw, color="red"):
                                batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                                actor_output = self.actor_rollout_wg.update_actor(batch)
                            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                            metrics.update(actor_output_metrics)

                        rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                        if rollout_data_dir:
                            with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                                inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                                outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                                scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                                self._dump_generations(
                                    inputs=inputs,
                                    outputs=outputs,
                                    scores=scores,
                                    reward_extra_infos_dict=reward_extra_infos_dict,
                                    dump_path=rollout_data_dir,
                                )

                        with marked_timer("sync_weights", timing_raw):
                            self.sync_rollout_weights()

                    

                    # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                    # Save checkpoint once per outer step (for entropy_rebranch, after stage2 already synced)
                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                        or esi_close_to_expiration
                    ):
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()
                if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)
                        
                        
                with marked_timer("stop_profile", timing_raw):
                    self._stop_profiling(do_profile)

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                #_save_metrics_to_csv(metrics, self.global_steps, epoch, self.config.trainer.experiment_name)
                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
