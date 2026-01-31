# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
Single Process Actor
"""
import random
import logging
import os
import asyncio
import inspect
from typing import Dict, List, Optional, Any

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers.cache_utils import DynamicCache

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

# 确保logger有handler，否则可能不会输出
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.kv_cache_dict = {}

    
    def clear_kv_cache(self):
        """清理KV cache字典 - 公共接口"""
        # 显式删除KV cache中的tensor引用
        for sample_idx, kv_data in self.kv_cache_dict.items():
            if 'past_key_values' in kv_data:
                for layer_kv in kv_data['past_key_values']:
                    if layer_kv and len(layer_kv) == 2:
                        k, v = layer_kv
                        del k, v
        
        self.kv_cache_dict.clear()
        
        # 强制GPU内存清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        print("KV cache dictionary cleared")

    def get_kv_cache_info(self):
        """获取KV cache信息用于调试"""
        info = {
            'num_samples': len(self.kv_cache_dict),
        }
        for sample_idx, kv_cache_dict in self.kv_cache_dict.items():
            if kv_cache_dict and 'past_key_values' in kv_cache_dict:
                kv_cache = kv_cache_dict['past_key_values']
                seq_len = kv_cache_dict.get('sequence_length', 0)
                actual_kv_len = kv_cache[0][0].shape[-2] if kv_cache and len(kv_cache) > 0 else 0
                info[f'sample_{sample_idx}_seq_len'] = seq_len
                info[f'sample_{sample_idx}_actual_kv_len'] = actual_kv_len
            else:
                info[f'sample_{sample_idx}_seq_len'] = 0
                info[f'sample_{sample_idx}_actual_kv_len'] = 0
        return info

    def _pad_kv_cache_to_batch(self, kv_list: List[Optional[Dict]], batch_size: int, device: torch.device, use_dynamic_cache: bool = True):
        """
        将多个样本的KV cache组合成batch形式，支持remove_padding + varlen模式
        
        在remove_padding模式下：
        - 使用sequence维度concatenation而非batch维度stacking
        - 生成cu_seqlens用于Flash Attention varlen
        - KV cache格式: [1, num_heads, total_seq_len, head_dim]
        
        
        Args:
            kv_list: List of KV cache dict for each sample, format: {'past_key_values': List[Tuple], 'sequence_length': int}
            batch_size: batch size  
            device: 目标设备
            use_dynamic_cache: 是否使用DynamicCache
            
        Returns:
            tuple: (kv_cache, cu_seqlens) where:
                - kv_cache: DynamicCache或tuple list
                - cu_seqlens: varlen模式下的累积序列长度 (remove_padding模式) 或 None (标准模式)
        """
        if all(kv is None for kv in kv_list):
            return None, None
        
        # 找到第一个非None的KV cache来确定层数和结构
        sample_kv_dict = next(kv for kv in kv_list if kv is not None)
        sample_kv = sample_kv_dict['past_key_values']
        num_layers = len(sample_kv)
        
        # 收集每个样本的实际序列长度
        actual_lengths = []
        for kv_dict in kv_list:
            if kv_dict is not None:
                actual_lengths.append(kv_dict['sequence_length'])
            else:
                actual_lengths.append(0)
        cu_seqlens = [0]
        for length in actual_lengths:
            cu_seqlens.append(cu_seqlens[-1] + length)
        cu_seqlens_tensor = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
        
        if self.use_remove_padding:
            total_seq_len = sum(actual_lengths)            
            batch_dynamic_cache = DynamicCache()
            for layer_idx in range(num_layers):
                layer_keys = []
                layer_values = []
                for i , kv_dict in enumerate(kv_list):
                    if kv_dict is not None and layer_idx < len(kv_dict['past_key_values']):
                        k, v = kv_dict['past_key_values'][layer_idx]
                        #actual_len = actual_lengths[i]  # 使用预处理后的实际长度           
                        #k_actual = k[:, :, :actual_len, :]  # [1, num_heads, actual_len, head_dim]
                        #v_actual = v[:, :, :actual_len, :]  # [1, num_heads, actual_len, head_dim]                          
                        layer_keys.append(k)
                        layer_values.append(v)

                if layer_keys and layer_values:
                    # 🔥 关键优化：先在CPU上拼接，减少GPU内存分配次数
                    concat_k_cpu = torch.cat(layer_keys, dim=-2)  # 在CPU上拼接
                    concat_v_cpu = torch.cat(layer_values, dim=-2)  # 在CPU上拼接
                    
                    # 然后一次性转移到GPU
                    concat_k = concat_k_cpu.to(device)  
                    concat_v = concat_v_cpu.to(device)  
                    
                    batch_dynamic_cache.key_cache.append(concat_k)
                    batch_dynamic_cache.value_cache.append(concat_v)
                    
                    # 🔥 立即释放所有临时张量和CPU缓存
                    del layer_keys, layer_values, concat_k_cpu, concat_v_cpu, concat_k, concat_v
                    
                
                # 设置seen_tokens为总token数
            batch_dynamic_cache._seen_tokens = total_seq_len
                
            return batch_dynamic_cache if len(batch_dynamic_cache.key_cache) > 0 else None, cu_seqlens_tensor
            
        
        #no remove padding
        else:
            max_seq_len = max(actual_lengths) if actual_lengths else 0
            # 标准模式：batch维度stacking + padding
            if use_dynamic_cache:
                batch_dynamic_cache = DynamicCache()
                
                for layer_idx in range(num_layers):
                    batch_keys = []
                    batch_values = []
                    
                    for sample_idx in range(batch_size):
                        kv_dict = kv_list[sample_idx]
                        if kv_dict is not None and layer_idx < len(kv_dict['past_key_values']):
                            k, v = kv_dict['past_key_values'][layer_idx]
                            actual_seq_len = kv_dict['sequence_length']
                            
                            # Pad到max_seq_len
                            if actual_seq_len < max_seq_len:
                                pad_len = max_seq_len - actual_seq_len
                                k_padded = torch.cat([
                                    k, 
                                    torch.zeros(*k.shape[:-2], pad_len, k.shape[-1], device=k.device, dtype=k.dtype)
                                ], dim=-2)
                                v_padded = torch.cat([
                                    v, 
                                    torch.zeros(*v.shape[:-2], pad_len, v.shape[-1], device=v.device, dtype=v.dtype)
                                ], dim=-2)
                            else:
                                k_padded = k
                                v_padded = v
                        else:
                            # 为没有cache的样本创建零cache
                            if sample_kv:
                                sample_k, sample_v = sample_kv[layer_idx]
                                k_padded = torch.zeros(*sample_k.shape[:-2], max_seq_len, sample_k.shape[-1], 
                                                     device=device, dtype=sample_k.dtype)
                                v_padded = torch.zeros(*sample_v.shape[:-2], max_seq_len, sample_v.shape[-1], 
                                                     device=device, dtype=sample_v.dtype)
                            else:
                                continue
                                
                        batch_keys.append(k_padded)
                        batch_values.append(v_padded)
                    
                    if batch_keys and batch_values:
                        # 在batch维度stack
                        batch_k = torch.cat(batch_keys, dim=0)  # [batch_size, num_heads, seq_len, head_dim]
                        batch_v = torch.cat(batch_values, dim=0)  # [batch_size, num_heads, seq_len, head_dim]
                        
                        batch_dynamic_cache.key_cache.append(batch_k)
                        batch_dynamic_cache.value_cache.append(batch_v)
                        
                        # 🔥 释放临时batch列表
                        del batch_keys, batch_values, batch_k, batch_v
                
                batch_dynamic_cache._seen_tokens = max_seq_len
                
                return batch_dynamic_cache if len(batch_dynamic_cache.key_cache) > 0 else None, None
            
            else:
                # 传统tuple list格式的标准模式
                batch_past_key_values = []
                
                for layer_idx in range(num_layers):
                    batch_keys = []
                    batch_values = []
                    
                    for sample_idx in range(batch_size):
                        kv_dict = kv_list[sample_idx]
                        if kv_dict is not None and layer_idx < len(kv_dict['past_key_values']):
                            k, v = kv_dict['past_key_values'][layer_idx]
                            actual_seq_len = kv_dict['sequence_length']
                            
                            # Pad到max_seq_len
                            if actual_seq_len < max_seq_len:
                                pad_len = max_seq_len - actual_seq_len
                                k_padded = torch.cat([
                                    k, 
                                    torch.zeros(*k.shape[:-2], pad_len, k.shape[-1], device=k.device, dtype=k.dtype)
                                ], dim=-2)
                                v_padded = torch.cat([
                                    v, 
                                    torch.zeros(*v.shape[:-2], pad_len, v.shape[-1], device=v.device, dtype=v.dtype)
                                ], dim=-2)
                            else:
                                k_padded = k
                                v_padded = v
                        else:
                            # 为没有cache的样本创建零cache
                            if sample_kv:
                                sample_k, sample_v = sample_kv[layer_idx]
                                k_padded = torch.zeros(*sample_k.shape[:-2], max_seq_len, sample_k.shape[-1], 
                                                     device=device, dtype=sample_k.dtype)
                                v_padded = torch.zeros(*sample_v.shape[:-2], max_seq_len, sample_v.shape[-1], 
                                                     device=device, dtype=sample_v.dtype)
                            else:
                                continue
                                
                        batch_keys.append(k_padded)
                        batch_values.append(v_padded)
                    
                    if batch_keys and batch_values:
                        # 在batch维度stack
                        batch_k = torch.cat(batch_keys, dim=0)  # [batch_size, num_heads, seq_len, head_dim]
                        batch_v = torch.cat(batch_values, dim=0)  # [batch_size, num_heads, seq_len, head_dim]
                        batch_past_key_values.append((batch_k, batch_v))
                        
                        # 🔥 释放临时batch列表
                        del batch_keys, batch_values
                
                return batch_past_key_values if batch_past_key_values else None, None

    def _extract_and_update_kv_cache_varlen(self, output_past_key_values, sample_indices: List[int], current_cu_seqlens=None):
        """
        只处理新生成tokens的varlen kv cache，并根据current_cu_seqlens追加到已有cache。
        
        Args:
            output_past_key_values: 新生成tokens的varlen格式KV cache，sequence维度已连接
            sample_indices: 新生成tokens分别属于哪个sample
            current_cu_seqlens: [0,2,5,6]，表示每个sample新生成token数
        """
        if output_past_key_values is None or current_cu_seqlens is None:
            return
        if len(output_past_key_values) == 0:
            return
        num_samples = len(sample_indices)
        # 处理每个样本的新生成tokens
        for sample_idx_in_batch in range(num_samples):
            sample_idx = sample_indices[sample_idx_in_batch]
            
            # 从current_cu_seqlens计算该sample新生成的token数量
            start_pos = current_cu_seqlens[sample_idx_in_batch].item()
            end_pos = current_cu_seqlens[sample_idx_in_batch + 1].item()
            new_token_count = end_pos - start_pos
            
            if new_token_count <= 0:
                continue            
            sample_kv_cache = []
            
            # 为每一层处理新生成tokens的KV cache
            for layer_idx, (layer_k, layer_v) in enumerate(output_past_key_values):
                try:
                    # 从varlen格式中提取该样本新生成tokens的部分
                    new_k = layer_k[:, :, start_pos:end_pos, :]  # [1, num_heads, new_token_count, head_dim]
                    new_v = layer_v[:, :, start_pos:end_pos, :]  # [1, num_heads, new_token_count, head_dim]
                    
                    # 🔥 立即转移到CPU，避免GPU显存累积
                    new_k_cpu = new_k.cpu()
                    new_v_cpu = new_v.cpu()
                    del new_k, new_v  # 立即释放GPU上的slice
                    
                    # 如果已有cache，则拼接；否则直接使用新的
                    if sample_idx in self.kv_cache_dict and self.kv_cache_dict[sample_idx] is not None:
                        old_k, old_v = self.kv_cache_dict[sample_idx]['past_key_values'][layer_idx]
                        combined_k = torch.cat([old_k, new_k_cpu], dim=-2)
                        combined_v = torch.cat([old_v, new_v_cpu], dim=-2)
                        del old_k, old_v, new_k_cpu, new_v_cpu
                    else:
                        combined_k = new_k_cpu
                        combined_v = new_v_cpu
                        del new_k_cpu, new_v_cpu
                    
                    sample_kv_cache.append((combined_k, combined_v))
                    
                except Exception as e:
                    logger.error(f"❌ Error processing KV cache for sample {sample_idx} layer {layer_idx}: {e}")
                    continue
            
            if sample_kv_cache:
                if sample_idx in self.kv_cache_dict and self.kv_cache_dict[sample_idx] is not None:
                    original_length = self.kv_cache_dict[sample_idx]['sequence_length']
                    total_length = original_length + new_token_count
                    # 🔥 删除旧的 KV cache dict 中的引用，避免 CPU 内存累积
                    old_past_kv = self.kv_cache_dict[sample_idx]['past_key_values']
                    for old_layer_kv in old_past_kv:
                        if old_layer_kv and len(old_layer_kv) == 2:
                            del old_layer_kv
                    del old_past_kv
                else:
                    total_length = new_token_count
                
                self.kv_cache_dict[sample_idx] = {
                    'past_key_values': sample_kv_cache,
                    'sequence_length': total_length
                }
                
                
        #logger.info(f"✅ Updated varlen KV cache.")
           
        
            
    def _extract_and_update_kv_cache(self, output_past_key_values, sample_indices: List[int], batch_size: int, current_cu_seqlens=None):
        """
        只处理新生成tokens的kv cache，并根据current_cu_seqlens追加到已有cache。
        
        Args:
            output_past_key_values: 新生成tokens的KV cache (可能是List或DynamicCache)
            sample_indices: 样本索引列表
            batch_size: batch size
            current_cu_seqlens: 累积序列长度，最后一个元素表示新生成tokens的总数量
        """
        # 处理 DynamicCache 对象
        if hasattr(output_past_key_values, 'to_legacy_cache'):
            legacy_cache = output_past_key_values.to_legacy_cache()
            output_past_key_values = legacy_cache
                    
        if current_cu_seqlens is not None:
            # Remove padding + varlen模式：根据current_cu_seqlens的最后一个元素只保留新生成的KV cache

            total_new_tokens = current_cu_seqlens[-1].item()

            if total_new_tokens <= 0:
                return

            if len(output_past_key_values) == 0:
                return
            # 从每一层的KV cache中只保留最后的新生成部分
            trimmed_kv_cache = []
            for layer_idx, (layer_k, layer_v) in enumerate(output_past_key_values):
                # 获取当前层的总长度
                total_seq_len = layer_k.shape[-2]
                
                # 只保留最后的新生成tokens
                start_pos = total_seq_len - total_new_tokens
                new_k = layer_k[:, :, start_pos:, :]  # [1, num_heads, total_new_tokens, head_dim]
                new_v = layer_v[:, :, start_pos:, :]  # [1, num_heads, total_new_tokens, head_dim]
                trimmed_kv_cache.append((new_k, new_v))
            self._extract_and_update_kv_cache_varlen(trimmed_kv_cache, sample_indices, current_cu_seqlens)
            # 🔥 释放trimmed cache
            del trimmed_kv_cache
        
       
    def _forward_micro_batch_with_cache(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            micro_batch: 包含模型输入的微批次数据，包括：
                - input_ids: 输入token序列
                - attention_mask: 注意力掩码
                - position_ids: 位置编码
                - responses: 响应token序列
                - sample_idx: 样本索引（用于KV cache管理）
            temperature: 采样温度
            calculate_entropy: 是否计算熵
            
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            if "image_bound" in micro_batch["multi_modal_inputs"][0]:  # minicpm-o logic
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = [inputs[key] for inputs in micro_batch["multi_modal_inputs"]]
            else:
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = torch.cat(
                        [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                    )

        # 从micro_batch中获取sample_idx来构建KV cache
        sample_indices = None
        batch_past_key_values = None
        cu_seqlens_kv = None  # varlen模式下的累积序列长度
        use_kv_cache = False
        
        if "sample_idx" in micro_batch:
            sample_indices = micro_batch["sample_idx"].cpu().tolist() if torch.is_tensor(micro_batch["sample_idx"]) else micro_batch["sample_idx"]
            # 收集每个样本的KV cache
            kv_list = []
            has_any_existing_cache = False
            for idx in sample_indices:
                if idx in self.kv_cache_dict:
                    kv_list.append(self.kv_cache_dict[idx])
                    has_any_existing_cache = True
                else:
                    kv_list.append(None)
            
            use_kv_cache = True
            
            # 将KV cache组成batch (如果有已有的cache)
            if has_any_existing_cache:
                device = micro_batch["input_ids"].device
                use_dynamic_cache = True
                # 新的方法返回tuple: (kv_cache, cu_seqlens)
                batch_past_key_values, cu_seqlens_kv = self._pad_kv_cache_to_batch(kv_list,
                                                                                   len(sample_indices),
                                                                                   device,
                                                                                   use_dynamic_cache)
            else:
                cu_seqlens_kv = torch.zeros((len(sample_indices)+1,), dtype=torch.int32)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)
                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch.keys()
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True
                
                # 使用新的KV cache逻辑
                use_cache = use_kv_cache
                
                if use_cache and batch_past_key_values is not None:
                    extra_args["past_key_values"] = batch_past_key_values
              
                past_len = 0
                if batch_past_key_values is not None:
                    past_len = batch_past_key_values.get_seq_length()
                
                
                num_seqs = len(cu_seqlens) - 1

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None, 
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=use_cache,  # 动态设置，避免参数冲突
                    cu_kvlen=cu_seqlens_kv,
                    **extra_args,  # 移动到最后，避免参数重复冲突
                )
             
                if use_cache and output.past_key_values.get_seq_length() != len(input_ids_rmpad[0])+past_len:
                    print(f"❌ Mismatch in KV cache length: expected {len(input_ids_rmpad[0])}, got {output.past_key_values.get_seq_length()}")
                    raise ValueError("quit")
                
                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True
                
                # 使用新的KV cache逻辑
                use_cache = use_kv_cache
                
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=use_cache,  # 动态设置，避免参数冲突
                    **extra_args,
                )

                

                if hasattr(output, 'past_key_values'):
                    print(f"  - output.past_key_values type: {type(output.past_key_values)}")
                    print(f"  - output.past_key_values len: {len(output.past_key_values) if output.past_key_values is not None else 'None'}")
                    

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)
        
        # 更新KV cache字典 (统一处理，避免重复)
        if output.past_key_values is not None and sample_indices is not None:
            current_cu_seqlens = cu_seqlens if self.use_remove_padding else None
            if hasattr(output.past_key_values, "to_legacy_cache"):
                past_kv = output.past_key_values.to_legacy_cache()
            
            
            self._extract_and_update_kv_cache(
                past_kv, 
                sample_indices, 
                batch_size, 
                current_cu_seqlens  # 当前输入的累积序列长度
            )
            
            del output.past_key_values
            if 'past_kv' in locals():
                del past_kv 
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        # 🔥 在所有处理完成后释放输入的batch_past_key_values
        if 'batch_past_key_values' in locals() and batch_past_key_values is not None:
            if hasattr(batch_past_key_values, 'key_cache'):
                # DynamicCache 对象，清空缓存列表
                batch_past_key_values.key_cache.clear()
                batch_past_key_values.value_cache.clear()
            del batch_past_key_values
       
        return entropy, log_probs
    
    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            if "image_bound" in micro_batch["multi_modal_inputs"][0]:  # minicpm-o logic
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = [inputs[key] for inputs in micro_batch["multi_modal_inputs"]]
            else:
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = torch.cat(
                        [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                    )

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch.keys()
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        has_sample_idx = "sample_idx" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        if has_sample_idx:
            non_tensor_select_keys.append("sample_idx")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # 如果有 sample_idx 且存在 KV cache，将 KV cache 长度信息添加到 data 中
        if has_sample_idx and use_dynamic_bsz:
            sample_indices = data.non_tensor_batch.get("sample_idx", [])
            kv_cache_lengths = []
            
            for sample_idx in sample_indices:
                if sample_idx in self.kv_cache_dict:
                    kv_cache_len = self.kv_cache_dict[sample_idx]['sequence_length']
                    kv_cache_lengths.append(kv_cache_len)
                else:
                    kv_cache_lengths.append(0)
            
            # 将 KV cache 长度信息添加到 non_tensor_batch 中
            data.non_tensor_batch["kv_cache_lengths"] = kv_cache_lengths
        
        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                # 🚀🚀🚀
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    if on_policy:
                        old_log_prob = log_prob.detach()
                    else:
                        old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla
                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)
                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                    )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    loss.backward()

                    micro_batch_metrics.update(
                        {
                            "actor/pg_loss": pg_loss.detach().item() * loss_scale_factor,
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        }
                    )
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
