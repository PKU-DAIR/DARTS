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
KV Cache Manager for Token-level Streaming with Infinite Pipeline optimization
"""

import torch
import asyncio
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from collections import OrderedDict
from dataclasses import dataclass
from verl import DataProto

logger = logging.getLogger(__name__)

@dataclass
class KVCacheState:
    """KV Cache状态管理 - 修正为多层格式"""
    past_key_values: List[Tuple[torch.Tensor, torch.Tensor]]  # 多层KV cache
    sequence_length: int
    last_access_time: float
    request_id: str
    sample_idx: int
    is_partial: bool = True
    reference_count: int = 0

@dataclass
class StreamingComputeTask:
    """流式计算任务"""
    task_id: str
    task_type: str  # "old_log_prob" or "ref_log_prob"
    partial_data: DataProto
    kv_state: KVCacheState
    future: asyncio.Future
    created_time: float

class InfinitePipelineKVManager:
    """无限流水线KV Cache管理器 - 支持Qwen2.5模型"""
    
    # 预定义的模型配置
    MODEL_CONFIGS = {
        "qwen2.5-0.5b": {
            "num_layers": 24,
            "num_heads": 14,
            "num_key_value_heads": 2,  # 对于GQA模型
            "head_dim": 64,
            "hidden_size": 896
        },
        "qwen2.5-1.5b": {
            "num_layers": 28, 
            "num_heads": 12,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "hidden_size": 1536
        },
        "gpt2": {
            "num_layers": 12,
            "num_heads": 12,
            "num_key_value_heads": 12,
            "head_dim": 64,
            "hidden_size": 768
        }
    }
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_cache_size = config.get("max_kv_cache_size", 1000)
        self.cache_ttl = config.get("kv_cache_ttl", 300.0)
        
        # 模型配置
        model_name = config.get("model_name", "qwen2.5-0.5b").lower()
        self.model_config = self.MODEL_CONFIGS.get(model_name, self.MODEL_CONFIGS["qwen2.5-0.5b"])
        
        # KV Cache存储
        self.kv_cache: Dict[str, KVCacheState] = OrderedDict()
        
        # 异步计算任务管理
        self.compute_tasks: Dict[str, StreamingComputeTask] = {}
        self.compute_overlaps = 0
        
        # 异步锁
        self._cache_lock = asyncio.Lock()
        self._compute_lock = asyncio.Lock()
        
        logger.info(f"KV Manager initialized with model: {model_name}, config: {self.model_config}")
        
    async def get_or_create_kv_state(self, 
                                   request_id: str,
                                   sample_idx: int,
                                   partial_data: DataProto) -> Optional[KVCacheState]:
        """获取或创建KV状态"""
        cache_key = f"{request_id}_{sample_idx}"
        
        async with self._cache_lock:
            if cache_key in self.kv_cache:
                kv_state = self.kv_cache[cache_key]
                kv_state.last_access_time = time.time()
                kv_state.reference_count += 1
                self.cache_hits += 1
                
                # 移到末尾(LRU)
                self.kv_cache.move_to_end(cache_key)
                logger.debug(f"KV Cache hit: {cache_key}")
                return kv_state
            else:
                # 创建新的KV状态
                self.cache_misses += 1
                kv_state = await self._extract_kv_from_data(request_id, sample_idx, partial_data)
                if kv_state:
                    self.kv_cache[cache_key] = kv_state
                    self._evict_old_entries()
                    logger.debug(f"Created new KV state: {cache_key}")
                return kv_state
    
    async def _extract_kv_from_data(self, 
                                  request_id: str, 
                                  sample_idx: int, 
                                  partial_data: DataProto) -> Optional[KVCacheState]:
        """从部分数据中提取KV状态 - 修正为多层格式"""
        try:
            # 这里需要根据模型实际情况提取KV Cache
            # 示例实现，实际需要根据具体模型调整
            
            batch_size = len(partial_data)
            if batch_size == 0:
                return None
                
            # 从数据中获取序列长度
            seq_len = partial_data.batch["input_ids"].shape[1]
            
            # 使用模型配置参数
            num_layers = self.model_config["num_layers"]
            num_heads = self.model_config["num_key_value_heads"]  # 使用KV heads而不是query heads
            head_dim = self.model_config["head_dim"]
            
            logger.debug(f"Creating KV cache: layers={num_layers}, heads={num_heads}, head_dim={head_dim}, seq_len={seq_len}")
            
            # 为多层模型创建KV cache
            past_key_values = []
            for _ in range(num_layers):
                # 实际实现应该从模型的forward过程中获取KV Cache
                # 这里先用占位符
                key_cache = torch.zeros(batch_size, num_heads, seq_len, head_dim, 
                                      dtype=torch.bfloat16, device=partial_data.batch["input_ids"].device)
                value_cache = torch.zeros(batch_size, num_heads, seq_len, head_dim, 
                                        dtype=torch.bfloat16, device=partial_data.batch["input_ids"].device)
                past_key_values.append((key_cache, value_cache))
            
            return KVCacheState(
                past_key_values=past_key_values,
                sequence_length=seq_len,
                last_access_time=time.time(),
                request_id=request_id,
                sample_idx=sample_idx,
                is_partial=True,
                reference_count=1
            )
            
        except Exception as e:
            logger.error(f"Failed to extract KV state: {e}")
            return None
    
    def _evict_old_entries(self):
        """淘汰旧的缓存条目"""
        current_time = time.time()
        
        # 移除过期条目
        expired_keys = []
        for key, kv_state in self.kv_cache.items():
            if (current_time - kv_state.last_access_time > self.cache_ttl or
                kv_state.reference_count == 0):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.kv_cache[key]
        
        # 如果仍然超出大小限制，按LRU淘汰
        while len(self.kv_cache) > self.max_cache_size:
            oldest_key = next(iter(self.kv_cache))
            del self.kv_cache[oldest_key]
    
    async def submit_compute_task(self,
                                request_id: str,
                                sample_idx: int,
                                task_type: str,
                                partial_data: DataProto,
                                compute_fn,
                                kv_state: Optional[KVCacheState] = None) -> str:
        """提交流式计算任务"""
        
        task_id = f"{request_id}_{sample_idx}_{task_type}_{int(time.time() * 1000)}"
        
        # 创建异步计算任务
        future = asyncio.create_task(
            self._execute_compute_with_kv(compute_fn, partial_data, kv_state)
        )
        
        task = StreamingComputeTask(
            task_id=task_id,
            task_type=task_type,
            partial_data=partial_data,
            kv_state=kv_state,
            future=future,
            created_time=time.time()
        )
        
        async with self._compute_lock:
            self.compute_tasks[task_id] = task
            self.compute_overlaps += 1
        
        logger.debug(f"Submitted compute task: {task_id} ({task_type})")
        return task_id
    
    async def _execute_compute_with_kv(self, 
                                     compute_fn,
                                     partial_data: DataProto,
                                     kv_state: Optional[KVCacheState]) -> DataProto:
        """使用KV状态执行计算 - 修正为多层格式"""
        try:
            # 如果有KV状态，将其注入到计算过程中
            if kv_state:
                # 这里需要修改partial_data，添加KV Cache信息
                # 具体实现需要根据模型结构调整
                if not hasattr(partial_data, 'meta_info'):
                    partial_data.meta_info = {}
                partial_data.meta_info['kv_cache'] = {
                    'past_key_values': kv_state.past_key_values,
                    'sequence_length': kv_state.sequence_length
                }
            
            # 执行实际计算
            result = await asyncio.to_thread(compute_fn, partial_data)
            return result
            
        except Exception as e:
            logger.error(f"Compute task failed: {e}")
            raise
    
    async def get_compute_result(self, task_id: str) -> Optional[DataProto]:
        """获取计算结果"""
        async with self._compute_lock:
            if task_id not in self.compute_tasks:
                return None
            
            task = self.compute_tasks[task_id]
        
        try:
            # 等待计算完成
            result = await task.future
            
            # 清理任务
            async with self._compute_lock:
                if task_id in self.compute_tasks:
                    del self.compute_tasks[task_id]
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get compute result {task_id}: {e}")
            return None
    
    async def wait_for_compute_results(self, task_ids: List[str]) -> Dict[str, DataProto]:
        """等待多个计算任务完成"""
        results = {}
        
        # 并行等待所有任务
        tasks = []
        for task_id in task_ids:
            tasks.append(self.get_compute_result(task_id))
        
        try:
            completed_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for task_id, result in zip(task_ids, completed_results):
                if not isinstance(result, Exception) and result is not None:
                    results[task_id] = result
                else:
                    logger.error(f"Task {task_id} failed: {result}")
            
        except Exception as e:
            logger.error(f"Failed to wait for compute results: {e}")
        
        return results
    
    async def update_kv_state(self,
                            request_id: str,
                            sample_idx: int,
                            new_partial_data: DataProto):
        """更新KV状态 - 修正为多层格式"""
        cache_key = f"{request_id}_{sample_idx}"
        
        async with self._cache_lock:
            if cache_key in self.kv_cache:
                kv_state = self.kv_cache[cache_key]
                
                # 更新KV状态
                new_kv_state = await self._extract_kv_from_data(request_id, sample_idx, new_partial_data)
                if new_kv_state:
                    kv_state.past_key_values = new_kv_state.past_key_values
                    kv_state.sequence_length = new_kv_state.sequence_length
                    kv_state.last_access_time = time.time()
                    
                    logger.debug(f"Updated KV state: {cache_key}, new_length: {new_kv_state.sequence_length}")
    
    def release_kv_reference(self, request_id: str, sample_idx: int):
        """释放KV引用"""
        cache_key = f"{request_id}_{sample_idx}"
        
        if cache_key in self.kv_cache:
            kv_state = self.kv_cache[cache_key]
            kv_state.reference_count = max(0, kv_state.reference_count - 1)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
        
        return {
            "cache_size": len(self.kv_cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "compute_overlaps": self.compute_overlaps,
            "active_compute_tasks": len(self.compute_tasks)
        }
    
    async def cleanup(self):
        """清理资源"""
        async with self._cache_lock:
            self.kv_cache.clear()
        
        async with self._compute_lock:
            # 取消所有未完成的计算任务
            for task in self.compute_tasks.values():
                if not task.future.done():
                    task.future.cancel()
            self.compute_tasks.clear()
