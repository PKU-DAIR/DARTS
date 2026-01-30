import json
import os
import uuid
import csv
import asyncio
import math
import torch
import traceback
from collections import defaultdict
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Optional, Dict, List

import numpy as np
import ray
import torch
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

WorkerType = type[Worker]
from recipe.one_step_off_policy.ray_trainer import OneStepOffRayTrainer
from recipe.one_step_off_policy.save_metrics import _save_metrics_to_csv
from recipe.one_step_off_policy.solver import ResourceOptimizer,repeat_by_position

class ScoreFilter:
    """
    使用指数移动平均（EMA）追踪每个ID的分数
    """
    def __init__(self, alpha=0.3):
        """
        Args:
            alpha: EMA平滑系数，越大越重视新值
        """
        self.alpha = alpha
        self.scores = {}  # 存储每个ID的EMA分数
        self.counts = {}  # 可选：记录每个ID被更新的次数
    

    def update(self, id2mean):
        for key, new_val in id2mean.items():
            if key not in self.scores:
                self.scores[key] = float(new_val)
                self.counts[key] = 1
            else:
                # 递推均值：把这次 batch 的均值作为一次新观测
                n = self.counts[key]
                self.scores[key] = (self.scores[key] * n + float(new_val)) / (n + 1)
                self.counts[key] = n + 1
    
    def get_score(self, key):
        """获取某个ID的当前分数"""
        return self.scores.get(key, 0.0)   # 不存在返回0
    
    def get_all_scores(self):
        """获取所有分数"""
        return self.scores.copy()

class VarienceFilter:
    """
    使用指数移动平均（EMA）追踪每个ID的分数
    """
    def __init__(self, alpha=0.7):
        """
        Args:
            alpha: EMA平滑系数，越大越重视新值
        """
        self.alpha = alpha
        self.varience = {}  # 存储每个ID的EMA分
        self.mean_var = 1
        

    def update(self, id2mean):
        cnt = 0
        sum = 0 
        for key, new_val in id2mean.items():
            if key not in self.varience:
                self.varience[key] = float(new_val)
                
            else:
                # 递推均值：把这次 batch 的均值作为一次新观测
                self.varience[key] = self.alpha * float(new_val) + (1 - self.alpha) * self.varience[key]
            
        for key in self.varience:
            cnt += 1
            sum += self.varience[key]
        self.mean_var = sum/cnt if cnt!=0 else 1.0
    
    def get_varience(self, key):
        """获取某个ID的当前分数"""
        return self.varience.get(key, self.mean_var)   # 不存在返回0
    
    def get_all_varience(self):
        """获取所有分数"""
        return self.varience.copy()



def dynamic_batch_reorder(batch_dict, len_filter, trainer_instance=None, default_length=128):
    if not (hasattr(len_filter, 'scores') and len_filter.scores and 'index' in batch_dict):
        return batch_dict
    
    # Get rollout worker count for load balancing
    num_workers = 1
    if trainer_instance and hasattr(trainer_instance, 'actor_rollout_wg'):
        num_workers = trainer_instance.actor_rollout_wg.world_size
    
    try:
        indices = batch_dict['index']
        data_sources = batch_dict.get('data_source', ['unknown'] * len(indices))        
        length_predictions = []
        has_historical_data = False
        
        for i, idx in enumerate(indices):
            source = data_sources[i] if i < len(data_sources) else 'unknown'
            key = f"{source}|{idx}"
            if key in len_filter.scores:
                pred_len = len_filter.scores[key]
                has_historical_data = True
            else:
                pred_len = default_length
            length_predictions.append((i, pred_len))
        
        # Only perform load balancing if we have historical data
        if not has_historical_data:
            return batch_dict
        
        if len(length_predictions) >= num_workers:
            length_predictions.sort(key=lambda x: x[1], reverse=True)
            worker_buckets = [[] for _ in range(num_workers)]
            worker_loads = [0.0] * num_workers
            for orig_idx, pred_len in length_predictions:
                min_load_worker = min(range(num_workers), key=lambda w: worker_loads[w])
                worker_buckets[min_load_worker].append(orig_idx)
                worker_loads[min_load_worker] += pred_len
            reorder_indices = []
            for bucket in worker_buckets:
                reorder_indices.extend(bucket)
            if reorder_indices != list(range(len(reorder_indices))):
                reordered_dict = {}
                for key in batch_dict:
                    if (hasattr(batch_dict[key], '__len__') and 
                        len(batch_dict[key]) == len(reorder_indices)):
                        if isinstance(batch_dict[key], torch.Tensor):
                            reordered_dict[key] = batch_dict[key][reorder_indices]
                        elif isinstance(batch_dict[key], (list, tuple)):
                            reordered_dict[key] = [batch_dict[key][i] for i in reorder_indices]
                        elif isinstance(batch_dict[key], np.ndarray):
                            reordered_dict[key] = batch_dict[key][reorder_indices]
                        else:
                            reordered_dict[key] = batch_dict[key]  # Keep unchanged if unsure
                    else:
                        reordered_dict[key] = batch_dict[key] 
                worker_load_info = [f"W{i}:{worker_loads[i]:.1f}" for i in range(num_workers)]
                print(f"🚀 Dynamic load balancing: reordered {len(reorder_indices)} sequences across {num_workers} workers: [{', '.join(worker_load_info)}]")
                return reordered_dict
                
    except Exception as e:
        print(f"⚠️ Dynamic load balancing failed, using original order: {e}")
    
    return batch_dict


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


class RayTrainerStreamDAPO(OneStepOffRayTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_batch_threshold = None
        self.score_filter = ScoreFilter(alpha=0.3)
        self.len_filter = ScoreFilter(alpha=0.3)
        self.varience_ema = VarienceFilter(alpha=0.3)

    def _save_len_filter_data(self):
        """保存varience_ema数据到pickle文件"""
        try:
            import pickle
            
            # 确定保存路径
            checkpoint_dir = self.config.trainer.default_local_dir
            varience_dir = os.path.join(checkpoint_dir, "varience_data")
            os.makedirs(varience_dir, exist_ok=True)
            
            # 保存varience_ema数据
            varience_path = os.path.join(varience_dir, f"varience_ema_step_{self.global_steps}.pkl")
            
            # 准备要保存的数据
            varience_save_data = {
                'varience': self.varience_ema.varience.copy(),
                'alpha': self.varience_ema.alpha,
                'global_steps': self.global_steps
            }
            
            with open(varience_path, 'wb') as f:
                pickle.dump(varience_save_data, f)
            
            print(f"💾 保存varience_ema数据到: {varience_path}")
            print(f"   - varience_ema记录数: {len(self.varience_ema.varience)}")
            print(f"   - alpha值: {self.varience_ema.alpha}")
            
        except Exception as e:
            print(f"❌ 保存varience_ema数据失败: {e}")
            import traceback
            traceback.print_exc()

    def _load_len_filter_data(self, target_step=None):
        """从pickle文件加载varience_ema数据"""
        try:
            import pickle
            import glob
            
            checkpoint_dir = self.config.trainer.default_local_dir
            varience_dir = os.path.join(checkpoint_dir, "varience_data")
            
            if not os.path.exists(varience_dir):
                print(f"⚠️ varience_ema数据目录不存在: {varience_dir}")
                return False
            
            # 如果没有指定步数，找到最新的文件
            if target_step is None:
                # 查找所有varience_ema文件
                pattern = os.path.join(varience_dir, "varience_ema_step_*.pkl")
                varience_files = glob.glob(pattern)
                
                if not varience_files:
                    print("⚠️ 没有找到varience_ema数据文件")
                    return False
                
                # 提取步数并找到最新的
                step_files = []
                for file_path in varience_files:
                    filename = os.path.basename(file_path)
                    try:
                        # 从 "varience_ema_step_123.pkl" 中提取 123
                        step_str = filename.replace("varience_ema_step_", "").replace(".pkl", "")
                        step_num = int(step_str)
                        step_files.append((step_num, file_path))
                    except (ValueError, IndexError):
                        continue
                
                if not step_files:
                    print("⚠️ 没有找到有效的varience_ema数据文件")
                    return False
                
                # 使用最新的文件
                target_step, target_file = max(step_files)
                print(f"📂 自动选择最新的varience_ema数据，步数: {target_step}")
            else:
                target_file = os.path.join(varience_dir, f"varience_ema_step_{target_step}.pkl")
            
            # 加载varience_ema数据
            if os.path.exists(target_file):
                with open(target_file, 'rb') as f:
                    varience_save_data = pickle.load(f)
                
                # 恢复数据到varience_ema对象
                self.varience_ema.varience = varience_save_data['varience'].copy()
                self.varience_ema.alpha = varience_save_data['alpha']
                tmp = {}
                self.varience_ema.update(tmp)
                
                print(f"📂 成功加载varience_ema数据: {len(self.varience_ema.varience)}个记录")
                print(f"   - alpha值: {self.varience_ema.alpha}")
                print(f"   - 来源步数: {varience_save_data.get('global_steps', '未知')}")
                
                return True
            else:
                print(f"⚠️ varience_ema数据文件不存在: {target_file}")
                return False
            
        except Exception as e:
            print(f"❌ 加载varience_ema数据失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    
    def _save_step_responses_to_csv(self, batch: DataProto, step: int):
        """将当前步骤的所有回复保存到CSV文件"""
        try:
            output_path = "////step_responses.csv"
            
            # 提取数据
            responses_text = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            
            # 计算每个回复的长度（基于response_mask）
            response_lengths = batch.batch["response_mask"].sum(dim=1).cpu().numpy()
            
            # 获取奖励（token级别奖励的总和）
            if "token_level_scores" in batch.batch:
                rewards = batch.batch["token_level_scores"].sum(dim=1).cpu().numpy()
            elif "rewards" in batch.batch:
                rewards = batch.batch["rewards"].cpu().numpy()
            else:
                # 如果没有奖励信息，用0填充
                rewards = [0.0] * len(responses_text)
            
            # 检查文件是否存在，如果不存在则写入头部
            file_exists = os.path.exists(output_path)
            
            with open(output_path, 'a', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['step', 'response', 'length', 'reward']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # 如果文件不存在，写入头部
                if not file_exists:
                    writer.writeheader()
                
                # 写入每个回复的数据
                for i in range(len(responses_text)):
                    writer.writerow({
                        'step': step,
                        'response': responses_text[i],
                        'length': int(response_lengths[i]),
                        'reward': float(rewards[i])
                    })
            
            print(f"✅ 已将第{step}步的{len(responses_text)}条回复保存到 {output_path}")
            
        except Exception as e:
            print(f"❌ 保存第{step}步回复数据失败: {e}")
            import traceback
            traceback.print_exc()
    
    async def _generate_with_token_streaming_overlapped(self, gen_batch: DataProto, chunk_size: int = 100, timing_raw: dict = None, token_batch_threshold: Optional[int] = None):
        """
        使用token级流式生成,并重叠计算old_log_prob和ref_log_prob
        
        策略: 每积累到一定数量的tokens就进行log_prob计算，
        通过修改response_mask等方式实现增量计算
        
        Args:
            gen_batch: 输入批次
            chunk_size: 每次返回的token数量
            timing_raw: 用于记录时间信息的字典
            
        Returns:
            完整的生成结果，包含overlapped的log_prob计算结果
        """
        import asyncio
        import time
        
        print(f"开始token级流式生成+增量重叠计算,batch_size={len(gen_batch)}, chunk_size={chunk_size}")
        
        if hasattr(self.async_rollout_manager, 'async_generate_sequences_token_stream'):
            sample_accumulators = {}
            expected_samples = len(gen_batch)
            completed_samples = set()
            finished_samples = set()
            ftime = 0
            
            # 增量计算状态跟踪
            sample_processed_tokens = {}  # 记录每个样本已处理的token数量
            sample_compute_results = {}   # 存储每个样本的计算结果
            accumulated_tokens = 0        # 当前积累的总token数
            rollout_n = self.config.actor_rollout_ref.rollout.n
            global_bsz = self.config.data.train_batch_size
            global_n = rollout_n * global_bsz
            max_len = self.config.data.max_response_length

            #token_batch_threshold = rollout_n * global_bsz * 800
            if token_batch_threshold is None:
                token_batch_threshold = rollout_n * global_bsz * 500
            else:
                token_batch_threshold = token_batch_threshold / 2
            print(f"等待 {expected_samples} 个样本的流式生成+增量重叠计算...")
            print(f"设置token批次阈值: {token_batch_threshold}")
            cur_time = time.time()
            print("🚀gen开始:",cur_time)
            try:
                async for sample_idx, partial_result in self.async_rollout_manager.async_generate_sequences_token_stream(gen_batch, chunk_size=chunk_size):
                    # 更新sample accumulator
                    previous_tokens = 0
                    flag = False
                    if sample_idx in sample_accumulators:
                        # 计算之前已有的token数量
                        if 'response_mask' in sample_accumulators[sample_idx].batch:
                            previous_tokens = int(torch.sum(sample_accumulators[sample_idx].batch['response_mask'][0]).item())
                    
                    # 更新accumulator
                    sample_accumulators[sample_idx] = partial_result
                    completed_samples.add(sample_idx)

                    if partial_result.non_tensor_batch.get("is_finished", False):
                        finished_samples.add(sample_idx)
                        flag = True

                    # 计算当前新增的token数量
                    if 'response_mask' in partial_result.batch:
                        current_tokens = int(torch.sum(partial_result.batch['response_mask'][0]).item())
                        new_tokens = current_tokens - previous_tokens
                        accumulated_tokens += new_tokens
                        if sample_idx not in sample_processed_tokens:
                            sample_processed_tokens[sample_idx] = 0

                    # 检查是否达到批次处理阈值
                    #if accumulated_tokens >= token_batch_threshold and ftime <= 2:
                    if flag and ((len(finished_samples) % (global_n//4)) == 0):
                        cur_time = time.time()
                        print("🚀开始F:",cur_time)
                        batch_for_compute = self._build_incremental_batch(
                            sample_accumulators, 
                            sample_processed_tokens,
                            finished_samples,
                        )
                        print(f"🔥开始增量计算,当前完成sample{len(finished_samples)}个, 计算样本数: {len(batch_for_compute)}")

                        if batch_for_compute is not None:
                            print(f"构建增量计算批次, 包含{len(batch_for_compute)}个样本")
                            with marked_timer("old_log_prob", timing_raw, color="blue"):
                                computed_batch = self._submit_old_logprob_compute(batch_for_compute)                           
                            if self.use_reference_policy: 
                                with marked_timer("ref", timing_raw, color="olive"):
                                    computed_batch = self._submit_ref_logprob_compute(computed_batch)
                            self._update_sample_compute_results(
                                    computed_batch,
                                    sample_compute_results,
                                    sample_processed_tokens
                            )
                            accumulated_tokens = 0
                            print(f"✅ 增量计算完成，更新了{len(batch_for_compute)}个样本的结果")
                            cur_time = time.time()
                            print("🚀结束F:",cur_time)
                        
                cur_time = time.time()
                print("🚀gen结束:",cur_time)
                
                batch_for_compute = self._build_incremental_batch(
                    sample_accumulators, 
                    sample_processed_tokens,
                    finished_samples,
                    is_last=True
                )
                if batch_for_compute: 
                    print(f"🔥流式生成完成，开始最终整合, 完成样本数: {len(batch_for_compute)}")
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        computed_batch = self._submit_old_logprob_compute(batch_for_compute)                
                    if self.use_reference_policy:
                        with marked_timer("ref", timing_raw, color="olive"):
                            computed_batch = self._submit_ref_logprob_compute(computed_batch)
                    self._update_sample_compute_results(
                        computed_batch,
                        sample_compute_results,
                        sample_processed_tokens
                    )
                    print(f"⚠️ 增量计算完成，更新了{len(batch_for_compute)}个样本的结果")

                self.actor_rollout_wg.clear_kv_cache()
                if self.use_reference_policy:
                    self.ref_policy_wg.clear_kv_cache2()
                
                
                final_results = []
                
                for sample_idx in sorted(sample_accumulators.keys()):
                    if sample_idx not in finished_samples:
                        continue
                    sample = sample_accumulators[sample_idx]                    
                    if sample_idx in sample_compute_results and sample_compute_results[sample_idx]:
                        sample = self._merge_compute_result(sample, sample_compute_results[sample_idx])
                       
                        del sample_compute_results[sample_idx]
                    final_results.append(sample)
            
                if final_results:
                    final_batch = DataProto.concat(final_results)
                    print(f"🚀🚀🚀 总样本数: {len(final_batch)}")
                    return final_batch,timing_raw
                
                    
            except Exception as e:
                print(f"❌ 流式生成+增量重叠计算失败: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("❌ 流式生成不可用，回退到标准生成方法")        
        return None

    def _build_incremental_batch(self, sample_accumulators: Dict, sample_processed_tokens: Dict, finished_samples : set = {}, is_last: bool = False) -> Optional[DataProto]:
        """构建用于增量计算的批次数据"""
        try:
            incremental_samples = []
            
            for sample_idx, partial_result in sample_accumulators.items():
                if sample_idx in sample_processed_tokens:
                    if is_last and sample_idx not in finished_samples:
                        continue
                    if not is_last and sample_idx not in finished_samples:
                        continue
                    processed_tokens = sample_processed_tokens[sample_idx]
                    
                    incremental_sample = self._create_incremental_sample(
                        partial_result, processed_tokens, sample_idx
                    )
                    
                    if incremental_sample is not None:
                        incremental_samples.append(incremental_sample)
            
            
            if incremental_samples:
                batch = DataProto.concat(incremental_samples)
                
                # 确保批次大小能被worker数量整除
                num_workers = self.actor_rollout_wg.world_size
                print(f"🚀构建增量批次, 原始大小={len(batch)}, worker数量={num_workers}")
                current_size = len(batch)
                
                if current_size % num_workers != 0:
                    target_size = ((current_size + num_workers - 1) // num_workers) * num_workers
                    samples_to_add = target_size - current_size                    
                    if samples_to_add > 0:
                        last_sample = batch[-1:]  
                        additional_samples = []
                        for _ in range(samples_to_add):
                            additional_samples.append(last_sample)                        
                        if additional_samples:
                            additional_batch = DataProto.concat(additional_samples)
                            batch = DataProto.concat([batch, additional_batch])
                
                return batch
            else:
                return None
                
        except Exception as e:
            print(f"❌ 构建增量批次失败: {e}")
            return None
    
    def _create_incremental_sample(self, partial_result: DataProto, processed_tokens: int, sample_idx: int) -> Optional[DataProto]:
        """为单个样本创建增量计算数据，将已处理的tokens替换为pad tokens"""
        try:
            # 复制原始数据
            incremental_sample = DataProto(
                batch=partial_result.batch.clone(),
                non_tensor_batch=partial_result.non_tensor_batch.copy(),
                meta_info=partial_result.meta_info.copy()
            )
            response_mask = incremental_sample.batch['response_mask'].clone()
            current_valid_tokens = int(torch.sum(response_mask[0]).item())

            if processed_tokens == current_valid_tokens:
                # 如果没有新增token，直接返回None表示不需要计算
                return None

            if processed_tokens != 0:
                pad_token_id = getattr(self.tokenizer, 'pad_token_id', 0)
                if pad_token_id is None:
                    pad_token_id = self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else 0            
                prompt_len = len(incremental_sample.batch['input_ids'][0]) - len(incremental_sample.batch['responses'][0])

            
                
                responses = incremental_sample.batch['responses'].clone()
                
                
                if current_valid_tokens > processed_tokens:
                    # 找到有效token的位置
                    valid_positions = torch.nonzero(response_mask[0], as_tuple=False).flatten()
                    
                    if len(valid_positions) > processed_tokens:
                        # 获取已处理的token位置和新增的token位置
                        processed_positions = valid_positions[:processed_tokens]
                        new_token_positions = valid_positions[processed_tokens:]
                        
                        # 将已处理的tokens替换为pad
                        responses[0, processed_positions] = pad_token_id
                        incremental_sample.batch['responses'] = responses
                        
                        # 更新response_mask：只保留新增的tokens
                        new_response_mask = torch.zeros_like(response_mask[0])
                        new_response_mask[new_token_positions] = 1
                        incremental_sample.batch['response_mask'] = new_response_mask.unsqueeze(0)

                        incremental_sample.batch['prompts'][0] = pad_token_id
                        
                        # 同时更新attention_mask和input_ids以保持一致性
                        if 'attention_mask' in incremental_sample.batch:
                            attention_mask = incremental_sample.batch['attention_mask'].clone()
                            response_length = responses.shape[1]
                            # 更新attention_mask中对应response部分
                            attention_mask[0, -response_length:] = new_response_mask
                            attention_mask[0, :prompt_len] = 0
                            incremental_sample.batch['attention_mask'] = attention_mask
                        
                        if 'input_ids' in incremental_sample.batch:
                            input_ids = incremental_sample.batch['input_ids'].clone()
                            # 更新input_ids中对应response部分
                            input_ids[0, -response_length:] = responses[0]
                            input_ids[0, :prompt_len] = pad_token_id
                            incremental_sample.batch['input_ids'] = input_ids

                        if 'position_ids' in incremental_sample.batch:
                            position_ids = incremental_sample.batch['position_ids'].clone()
                            
                            # 方法1：重新计算position_ids以保持与attention_mask一致
                            # 使用与agent_loop相同的计算方式
                            new_position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask
                            incremental_sample.batch['position_ids'] = new_position_ids
                        
                        
            return incremental_sample
                    
            
        except Exception as e:
            print(f"❌ 创建样本 {sample_idx} 的增量数据失败: {e}")
            return None
    
    def _update_sample_compute_results(self, computed_batch: DataProto, sample_compute_results: Dict, sample_processed_tokens: Dict):
        """更新样本的计算结果"""
        try:
            batch_size = len(computed_batch)
            
            for i in range(batch_size):
                if "sample_idx" in computed_batch.non_tensor_batch:
                    sample_idx = int(computed_batch.non_tensor_batch["sample_idx"][i])
                    
                    # 提取该样本的计算结果
                    sample_result = {
                        'old_log_probs': computed_batch.batch['old_log_probs'][i:i+1] if 'old_log_probs' in computed_batch.batch else None,
                        'ref_log_prob': computed_batch.batch['ref_log_prob'][i:i+1] if 'ref_log_prob' in computed_batch.batch else None,
                        'response_mask': computed_batch.batch['response_mask'][i:i+1] if 'response_mask' in computed_batch.batch else None,
                        'processed_tokens': len(computed_batch.batch['response_mask'][i].nonzero()) if 'response_mask' in computed_batch.batch else 0
                    }
                    
                    # 更新样本计算结果
                    if sample_idx not in sample_compute_results:
                        sample_compute_results[sample_idx] = []
                    sample_compute_results[sample_idx].append(sample_result)
                    
                    # 更新已处理的token数
                    sample_processed_tokens[sample_idx] = len(computed_batch.batch['response_mask'][i].nonzero()) if 'response_mask' in computed_batch.batch else 0
                    
        except Exception as e:
            print(f"❌ 更新样本计算结果失败: {e}")
    
    def _merge_compute_result(self, accumulator: DataProto, compute_results: List[Dict]) -> DataProto:
        """合并样本的累积数据和计算结果，优化后版本"""
        try:
            merged_result = DataProto(
                batch=accumulator.batch.clone() if accumulator.batch is not None else None,
                non_tensor_batch=accumulator.non_tensor_batch.copy(),
                meta_info=accumulator.meta_info.copy()
            )
            
            if 'responses' in merged_result.batch:
                response_shape = merged_result.batch['responses'].shape
                device = merged_result.batch['responses'].device  # 获取设备，确保后续张量在同一设备
                
                # 初始化log_probs时指定设备，避免后续设备不匹配
                if 'old_log_probs' not in merged_result.batch:
                    merged_result.batch['old_log_probs'] = torch.zeros(response_shape, dtype=torch.float, device=device)
                if 'ref_log_prob' not in merged_result.batch:
                    merged_result.batch['ref_log_prob'] = torch.zeros(response_shape, dtype=torch.float, device=device)
            
            if compute_results and merged_result.batch is not None and 'response_mask' in merged_result.batch:
                full_response_mask = merged_result.batch['response_mask'].bool()[0]  # [seq_len]
                # 过滤无效的compute_result，减少循环次数
                valid_compute_results = [
                    r for r in compute_results 
                    if r.get('response_mask') is not None 
                    and (r.get('old_log_probs') is not None or r.get('ref_log_prob') is not None)
                ]
                
                for result in valid_compute_results:
                    inc_mask = result['response_mask'][0].bool()  # 增量掩码 [seq_len]
                    # 只处理在full_response_mask内的位置（避免越界或无效位置）
                    inc_mask = inc_mask & full_response_mask
                    
                    # 批量更新old_log_probs
                    if result.get('old_log_probs') is not None:
                        inc_old_logits = result['old_log_probs'][0]  # [seq_len]
                        merged_result.batch['old_log_probs'][0][inc_mask] = inc_old_logits[inc_mask]
                    
                    if result.get('ref_log_prob') is not None:
                        inc_ref_logits = result['ref_log_prob'][0]  # [seq_len]
                        merged_result.batch['ref_log_prob'][0][inc_mask] = inc_ref_logits[inc_mask]
                merged_result.meta_info['has_incremental_compute'] = True
            
            return merged_result
        except Exception as e:
            print(f"❌ 合并计算结果失败: {e}")
            return accumulator
    
    def _submit_old_logprob_compute(self, batch: DataProto) -> DataProto:
        """提交old_log_prob计算"""
        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
        old_log_prob.batch.pop("entropys")
        batch.batch['old_log_probs'] = old_log_prob.batch['old_log_probs']
        return batch
            
    
    def _submit_ref_logprob_compute(self, batch: DataProto) -> DataProto:
        """提交ref_log_prob计算"""
        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
        batch.batch['ref_log_prob'] = ref_log_prob.batch['ref_log_prob']
        return batch
        
     
    async def _generate_with_token_streaming(self, gen_batch: DataProto, chunk_size: int = 100) -> DataProto:
        """
        使用token级流式生成,每生成chunk_size个token就返回一次,在这里收集并拼接
        
        Args:
            gen_batch: 输入批次
            chunk_size: 每次返回的token数量
            
        Returns:
            完整的生成结果
        """
        print(f"开始token级流式生成,batch_size={len(gen_batch)}, chunk_size={chunk_size}")
        
        if hasattr(self.async_rollout_manager, 'async_generate_sequences_token_stream'):
            sample_accumulators = {}
            expected_samples = len(gen_batch)
            completed_samples = set()
            
            print(f"等待 {expected_samples} 个样本的流式生成...")
            
            try:
                # 使用异步rollout管理器的真正流式生成
                async for sample_idx, partial_result in self.async_rollout_manager.async_generate_sequences_token_stream(gen_batch, chunk_size=chunk_size):
                    # 直接使用流式生成返回的partial_result，它已经是通过_postprocess_single_sample处理的DataProto
                    sample_accumulators[sample_idx] = partial_result
                    
                    # 计算实际生成的token数量（通过response_mask或attention_mask）
                    if 'response_mask' in partial_result.batch:
                        actual_tokens = int(torch.sum(partial_result.batch['response_mask'][0]).item())
                    elif 'responses' in partial_result.batch and 'attention_mask' in partial_result.batch:
                        response_length = partial_result.batch['responses'].shape[1]
                        response_attention_mask = partial_result.batch['attention_mask'][0, -response_length:]
                        actual_tokens = int(torch.sum(response_attention_mask).item())
                    else:
                        actual_tokens = 0
                    
                    print(f"  样本 {sample_idx}: 累积{actual_tokens}个实际token")
                    completed_samples.add(sample_idx)

                    
                
                # 等待一段时间确保所有样本都完成
                if len(completed_samples) < expected_samples:
                    print(f"⚠️ 部分样本未完成，已完成: {len(completed_samples)}")
                
                # 重新组装所有样本的最终结果
                final_results = []
                for sample_idx in sorted(sample_accumulators.keys()):
                    # 直接使用accumulator，因为它已经是通过_postprocess_single_sample处理的DataProto
                    accumulator = sample_accumulators[sample_idx]
                    
                    final_results.append(accumulator)
                    
                if final_results:
                    # 使用DataProto.concat合并所有结果
                    final_result = DataProto.concat(final_results)
                    print(f"流式生成重组完成，最终batch_size={len(final_result)}")
                    return final_result
                
                else:
                    print("❌ 没有有效的累积结果，回退到标准生成")
                    
            except Exception as e:
                print(f"❌ 流式生成失败: {e}")
        else:
            print("❌ 流式生成不可用，回退到标准生成方法")
        
        # 回退到标准生成
        return 0

    async def _generate_with_sample_streaming_overlapped(self, gen_batch: DataProto, chunk_size: int = 100, timing_raw: dict = None, token_batch_threshold: Optional[int] = None):
       
        import asyncio
        import time
        print(f"开始token级流式生成+增量重叠计算,batch_size={len(gen_batch)}, chunk_size={chunk_size}")
        if hasattr(self.async_rollout_manager, 'async_generate_sequences_token_stream'):
            expected_samples = len(gen_batch)
            finished_samples = set()
            batch_for_compute = []
            sample_idx_list = []
            flag = False
            final_results = []
            results = {}
            
            rollout_n = self.config.actor_rollout_ref.rollout.n
            global_bsz = self.config.data.train_batch_size
            global_n = rollout_n * global_bsz
            print(f"等待 {expected_samples} 个样本的流式生成+增量重叠计算...")
            cur_time = time.time()
            print("🚀gen开始:",cur_time)
            try:
                async for sample_idx, partial_result in self.async_rollout_manager.async_generate_sequences_token_stream(gen_batch, chunk_size=chunk_size):
                    if partial_result.non_tensor_batch.get("is_finished", False):
                        finished_samples.add(sample_idx)
                        batch_for_compute.append(partial_result)
                        sample_idx_list.append(sample_idx)
                        flag = True
                    else:
                        flag = False

                    if flag and ((len(finished_samples) % (global_n//4)) == 0):
                        cur_time = time.time()
                        print("🚀开始F:",cur_time)
                        if batch_for_compute is not None:
                            batch_for_compute = DataProto.concat(batch_for_compute)
                            with marked_timer("old_log_prob", timing_raw, color="blue"):
                                computed_batch = self._submit_old_logprob_compute(batch_for_compute)                           
                            if self.use_reference_policy: 
                                with marked_timer("ref", timing_raw, color="olive"):
                                    computed_batch = self._submit_ref_logprob_compute(computed_batch)
                            for i in range(len(computed_batch)):
                                sample_idx = sample_idx_list[i]
                                # 创建单个样本的DataProto
                                single_sample_batch = {}
                                for key, tensor in computed_batch.batch.items():
                                    single_sample_batch[key] = tensor[i:i+1]  # 保持batch维度
                                
                                single_sample_non_tensor = {}
                                for key, array in computed_batch.non_tensor_batch.items():
                                    single_sample_non_tensor[key] = array[i:i+1]  # 保持batch维度
                                
                                # 创建单样本DataProto
                                single_sample_dataproto = DataProto.from_dict(
                                    tensors=single_sample_batch,
                                    non_tensors=single_sample_non_tensor,
                                    meta_info=computed_batch.meta_info
                                )
                                
                                results[sample_idx] = single_sample_dataproto

                            batch_for_compute = []
                            sample_idx_list = []
                            print(f"✅ 增量计算完成，更新了{len(computed_batch)}个样本的结果")
                            cur_time = time.time()
                            print("🚀结束F:",cur_time)
                        
                cur_time = time.time()
                print("🚀gen结束:",cur_time)

                self.actor_rollout_wg.clear_kv_cache()
                if self.use_reference_policy:
                    self.ref_policy_wg.clear_kv_cache2()  
                print(len(results))          
                
                for sample_idx in sorted(finished_samples):
                    if sample_idx in results:
                        final_results.append(results[sample_idx])
                if final_results:   
                    final_results = DataProto.concat(final_results)
                    print(f"🚀🚀🚀 总样本数: {len(final_results)}")
                    return final_results,timing_raw
                
                    
            except Exception as e:
                print(f"❌ 流式生成+增量重叠计算失败: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("❌ 流式生成不可用，回退到标准生成方法")        
        return None
    

    def fit(self):
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking
        import time
        _elapsed = 0.0

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.gen_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()
        # 加载varience_ema历史数据
        self._load_len_filter_data()
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
        self.gen_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0
        
        # Variables for filtering logic
        accumulated_batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0

#var_step_q 记录过去十步的variance
#self.varience_ema 记录滑动平均
#self._resp_len_stats 每个step都会清除

        step_Queue_length = 10
        if not hasattr(self, "_var_step_q"):
            self._var_step_q = deque(maxlen=step_Queue_length)
            self._var_step_avg = 0.0  # 滑动平均

        def _new_stat():
            return {"n": 0, "mean": 0.0, "M2": 0.0}

        print("🚀 Using token-level streaming")
        from collections import defaultdict
        timing_raw = defaultdict(float)  # Initialize timing_raw outside the loop to accumulate across continues
        for epoch in range(self.config.trainer.total_epochs):
            avg_token_len = 0
            if not hasattr(self, "_resp_len_stats"):
               self._resp_len_stats = defaultdict(_new_stat)
            else:
               self._resp_len_stats.clear()


            for batch_dict in self.train_dataloader:
                # Only clear score_filter but keep len_filter for historical data
                num_gen_batches += 1

                self._resp_len_stats.clear()

                self.score_filter.scores.clear()
                self.score_filter.counts.clear()
                # 注释：保留len_filter的历史数据用于负载均衡
                self.len_filter.scores.clear()
                self.len_filter.counts.clear()


                from collections import defaultdict
                self._epoch_event_rows = [] 
                self._occ_counter = defaultdict(int)
                metrics = {}

                do_profile = (
                    self.global_steps in self.config.trainer.profile_steps
                    if self.config.trainer.profile_steps is not None
                    else False
                )
                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(do_profile)

                # Apply dynamic batching to reorder sequences for load balancing
                # batch_dict = dynamic_batch_reorder(batch_dict, self.len_filter, trainer_instance=self)

                batch: DataProto = DataProto.from_single_dict(batch_dict)

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
                #if "index" in batch.non_tensor_batch:
                    #non_tensor_batch_keys_to_pop.append("index")
                if "agent_name" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("agent_name")

                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )
                gen_batch.non_tensor_batch["index"] = np.array(
                    batch.non_tensor_batch["index"], dtype=object
                )
                '''
                #NOTE self._var_step_avg是过去十个step的variance平均
                #NOTE self.varience_ema是variance的滑动平均
                用法是def get_varience(self, key):
                key的获取
                '''
                gen_batch.meta_info["global_steps"] = self.global_steps
                counter = ray.get_actor("redundancy_counter")


                if self.global_steps <= 10:
                    self.config.algorithm.redundancy = 12
                    gen_batch = gen_batch.repeat(repeat_times=self.config.algorithm.redundancy, interleave=True)
                    ray.get(counter.reset.remote(self.config.actor_rollout_ref.rollout.n, self.config.algorithm.redundancy))
                else:
                    cur_variance = []
                    avg_r = 12 
                    idxs = batch.non_tensor_batch["index"]  
                    sources = batch.non_tensor_batch.get(
                        "data_source",
                        np.array(["unknown"] * len(idxs), dtype=object),
                    )
                    keys = [f"{str(sources[i])}|{str(idxs[i])}" for i in range(len(idxs))]
                    for k in keys:
                        cur_variance.append(self.varience_ema.get_varience(k))
                    optimizer = ResourceOptimizer(cur_variance, self.config.data.train_batch_size *avg_r, min_m=10, max_m=16)
                    cur_redundancy = optimizer.optimize()
                    gen_batch = gen_batch.custom_repeat(repeat_times=cur_redundancy)
                    cur_position  = repeat_by_position(cur_redundancy)
                    result = [max((16 - x)//2, 0) for x in cur_redundancy]
                    print(f"当前variance: {[round(v,2) for v in cur_variance]}")
                    print(f"当前冗余度: {cur_redundancy}")
                    print(f"当前补齐数: {result}")
                    ray.get(counter.reset_num.remote(self.config.actor_rollout_ref.rollout.n, cur_redundancy, cur_position, result))
                
                # 给一个数组，[B,]. 类似于[8,8,8,8,16,16,16]
                # pass global_steps to trace
                
                is_last_step = self.gen_steps >= self.total_training_steps

                if self.token_batch_threshold is None:
                    self.token_batch_threshold = self.config.actor_rollout_ref.rollout.n * self.config.data.train_batch_size * 1200
                else:
                    self.token_batch_threshold = self.config.actor_rollout_ref.rollout.n * self.config.data.train_batch_size * avg_token_len * 1.5

                with marked_timer("step", timing_raw):
                    # 使用token级流式生成
                    with marked_timer("gen", timing_raw, color="red"):
                        # 获取配置
                        chunk_size = self.config.trainer.get("token_stream_chunk_size", 100)
                        use_token_streaming = self.config.trainer.get("enable_token_streaming", True)
                        if use_token_streaming:
                            print(f"  使用token级流式生成+重叠计算 (chunk_size={chunk_size})")
                            import asyncio
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            try:
                                gen_batch_output,timing_raw = loop.run_until_complete(
                                    self._generate_with_sample_streaming_overlapped(gen_batch, chunk_size=chunk_size, timing_raw=timing_raw, token_batch_threshold=self.token_batch_threshold)
                                )
                                
                                # 检查是否成功生成
                                if gen_batch_output is not None:
                                    # 检查是否已经包含了overlapped的计算结果
                                    has_old_logprob = "old_log_probs" in gen_batch_output.batch or "log_probs" in gen_batch_output.batch
                                    has_ref_logprob = "ref_log_prob" in gen_batch_output.batch
                                    
                                    print(f"   - old_log_prob已计算: {has_old_logprob}")
                                    print(f"   - ref_log_prob已计算: {has_ref_logprob}")
                                else:
                                    print("⚠️ 流式生成+重叠计算失败，回退到标准生成")
                                    # 回退到标准生成
                                    if not self.async_rollout_mode:
                                        gen_batch_output = self.rollout_wg.generate_sequences(gen_batch)
                                    else:
                                        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                                    has_old_logprob = False
                                    has_ref_logprob = False
                                
                            finally:
                                loop.close() 
                        else:
                            # 使用标准生成方法
                            if not self.async_rollout_mode:
                                gen_batch_output = self.rollout_wg.generate_sequences(gen_batch)
                            else:
                                gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                            has_old_logprob = False
                            has_ref_logprob = False
                        # 处理timing信息
                        if hasattr(gen_batch_output, 'meta_info') and 'timing' in gen_batch_output.meta_info:
                            timing_raw.update(gen_batch_output.meta_info["timing"])
                            gen_batch_output.meta_info.pop("timing", None)

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
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
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
                        
                        # Process reward tensor
                        if not self.config.reward_model.launch_reward_fn_async:
                            batch.batch["token_level_scores"] = reward_tensor
                            if reward_extra_infos_dict:
                                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    # === Filtering logic from ray_trainer_dapo.py ===
                    if not self.config.algorithm.filter_groups.enable:
                        pass  # No filtering, continue with current batch
                    else:  # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            batch.non_tensor_batch["seq_final_reward"] = (
                                batch.batch["token_level_rewards"].sum(dim=-1).cpu().numpy()
                            )
                        elif metric_name == "seq_reward":
                            batch.non_tensor_batch["seq_reward"] = (
                                batch.batch["token_level_scores"].sum(dim=-1).cpu().numpy()
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
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)

                        batch = batch[kept_traj_idxs]
                        accumulated_batch = batch if accumulated_batch is None else DataProto.concat([accumulated_batch, batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_gen_batches=}. Keep generating...")
                                progress_bar.update(1)
                                self.gen_steps += 1
                                continue
                            else:
                                raise ValueError(
                                    f"{num_gen_batches=} >= {max_num_gen_batches=}."
                                    + " Generated too many. Please check if your data are too difficult."
                                    + " You could also try set max_num_gen_batches=0 to enable endless trials."
                                )
                        else:
                            # Align the batch
                            print(f"{num_prompt_in_batch=} >= {prompt_bsz=}. Proceeding to update...")
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            batch = accumulated_batch[:traj_bsz]
                    # === End of filtering logic ===

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        if not has_old_logprob:
                            print("🔄 计算old_log_prob (未在流式过程中计算)")
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)
                        else:
                            print("✅ 跳过old_log_prob计算 (已在流式过程中完成)")
                            # 从gen_batch_output中提取entropy信息
                            if "entropys" in gen_batch_output.batch:
                                entropys = gen_batch_output.batch["entropys"]
                                response_masks = batch.batch["response_mask"]
                                loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                                entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                                old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                                metrics.update(old_log_prob_metrics)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
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
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            # 检查是否已经在流式生成中计算了ref_log_prob
                            if not has_ref_logprob:
                                print("🔄 计算ref_log_prob (未在流式过程中计算)")
                                if not self.ref_in_actor:
                                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                else:
                                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)
                            else:
                                print("✅ 跳过ref_log_prob计算 (已在流式过程中完成)")

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm (for async case)
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                            batch.batch["token_level_scores"] = reward_tensor
                            if reward_extra_infos_dict:
                                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )
                    
                    # Reset accumulation variables for next iteration
                    accumulated_batch = None
                    num_prompt_in_batch = 0
                    num_gen_batches = 0

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            # Add temperature to meta_info if not present (needed for actor update)
                            if "temperature" not in batch.meta_info:
                                batch.meta_info["temperature"] = getattr(self.config.actor_rollout_ref.rollout, "temperature", 1.0)
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
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
                            
                    self.sync_rollout_weights()

                    idxs = batch.non_tensor_batch["index"]
                    
                    sources = batch.non_tensor_batch.get(
                        "data_source",
                        np.array(["unknown"] * len(idxs), dtype=object),
                    )
                    keys = [f"{str(sources[i])}|{str(idxs[i])}" for i in range(len(idxs))]

                    from collections import defaultdict

                    s = batch.batch["token_level_scores"].sum(-1)
                    scores_list = s.detach().cpu().tolist()
                    resp_mask = batch.batch["response_mask"]
                    resp_len_llm  = resp_mask.sum(dim=1) 
                    resp_len_llm_list = resp_len_llm.detach().cpu().tolist() 
                    # 记录本 epoch 的所有事件
                    for k, r, L in zip(keys, scores_list, resp_len_llm_list):
                        occ = self._occ_counter[k]         # 该 key 在本 epoch 的出现序号
                        self._occ_counter[k] = occ + 1
                        # 追加一条出现事件
                        self._epoch_event_rows.append((
                            epoch,
                            self.global_steps,
                            k,
                            occ,
                            float(r),
                            float(L),
                        ))

                        # === 按 key 在线更新 response_length 统计量（Welford）===
                        stat = self._resp_len_stats[k]
                        x = float(L)
                        stat["n"] += 1
                        delta = x - stat["mean"]
                        stat["mean"] += delta / stat["n"]
                        
                        stat["M2"] += delta * (x - stat["mean"])


                    acc, cnt = defaultdict(float), defaultdict(int)
                    with torch.no_grad():
                        for k, s in zip(keys, scores_list):
                            acc[k] += s
                            cnt[k] += 1
                    id2mean = {k: acc[k] / cnt[k] for k in acc}
                    self.score_filter.update(id2mean)

                    #记录response_length
                    # 像 reward 一样做本 batch 的按 key 聚合（平均）
                    acc, cnt = defaultdict(float), defaultdict(int)
                    for k, l in zip(keys, resp_len_llm_list):
                        acc[k] += float(l)
                        cnt[k] += 1
                    id2mean_len = {k: acc[k] / cnt[k] for k in acc}
                    self.len_filter.update(id2mean_len)
                    #varience_step是这一步所有的标准差之和
                    varience_step = 0.0

                    resp_len_std_by_key = {}
                    for k, v in self._resp_len_stats.items():
                        n = v["n"]
                        if n > 1:
                            var = v["M2"] / (n - 1)   # 无偏样本方差
                            resp_len_std_by_key[k] = math.sqrt(var)
                        else:
                            resp_len_std_by_key[k] = 0.0
                            
                        varience_step += resp_len_std_by_key[k]
                    
                    self._var_step_q.append(varience_step)
                    self._var_step_avg = sum(self._var_step_q) / len(self._var_step_q)

                    self.varience_ema.update(resp_len_std_by_key)


                    #此时 varience_step 是本 step 的 response length 标准差之和
                    #此时 resp_len_std_by_key 是各 key 的 response length 标准差字典


                    # validate
                

                    # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )
                    # Check if the conditions for saving a checkpoint are met.
                    # The conditions include a mandatory condition (1) and
                    # one of the following optional conditions (2/3/4):
                    # 1. The save frequency is set to a positive value.
                    # 2. It's the last training step.
                    # 3. The current step number is a multiple of the save frequency.
                    # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                        or esi_close_to_expiration
                    ):
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()
                            # 保存varience_ema数据
                            self._save_len_filter_data()
                            

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

                    
                avg_token_len = int(torch.mean(torch.sum(batch.batch["response_mask"], dim=1).float()).item())
                lengths = [int(torch.sum(mask).item()) for mask in batch.batch["response_mask"]]
                
                #with open("/len.txt", "a") as f:
                #    f.write(f"{lengths}\n")
                
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
                timing_raw = defaultdict(float)  # clear timing
                
                metrics["train/num_gen_batches"] = num_gen_batches
                #_save_metrics_to_csv(metrics, self.global_steps, epoch, self.config.trainer.experiment_name)


                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                # 保存当前步骤的回复数据到CSV
                #self._save_step_responses_to_csv(batch, self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)


                #record
                scores_epoch = self.score_filter.get_all_scores()
                len_epoch = self.len_filter.get_all_scores()
                

                # —— 写入到你指定的绝对目录 —— 
                if False:
                    base_dir = "///darts/logs"
                    os.makedirs(base_dir, exist_ok=True)

                    out_path = os.path.join(base_dir, "metrics_history.csv")
                    all_keys = set(scores_epoch) | set(len_epoch)
                    file_exists = os.path.exists(out_path) and os.path.getsize(out_path) > 0
                    with open(out_path, "a", encoding="utf-8", newline="") as f:
                        writer = csv.writer(f)
                        if not file_exists:
                            writer.writerow(["epoch","step",  "key", "mean_reward", "response_length"])
                        for k in sorted(all_keys):
                            mr = scores_epoch.get(k)          # 可能为 None
                            rl = len_epoch.get(k)             # 可能为 None
                            writer.writerow([
                                epoch,
                                self.global_steps,
                                k,
                                "" if mr is None else float(mr),
                                "" if rl is None else float(rl),
                            ])

                    events_path = os.path.join(base_dir, "metrics_events.csv")
                    file_exists = os.path.exists(events_path) and os.path.getsize(events_path) > 0
                    with open(events_path, "a", encoding="utf-8", newline="") as f:
                        writer = csv.writer(f)
                        if not file_exists:
                            writer.writerow(["epoch", "step", "key", "occurrence", "reward", "response_length"])
                        writer.writerows(self._epoch_event_rows)


