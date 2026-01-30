#!/usr/bin/env python3
"""
测试动态负载均衡重排序逻辑
"""

import torch
import numpy as np
from collections import defaultdict

# 模拟ScoreFilter类
class MockScoreFilter:
    def __init__(self):
        self.scores = {}
        self.counts = {}
    
    def add_score(self, key, score):
        """手动添加分数用于测试"""
        self.scores[key] = score

# 模拟trainer实例
class MockTrainer:
    def __init__(self, world_size):
        self.actor_rollout_wg = MockWorkerGroup(world_size)

class MockWorkerGroup:
    def __init__(self, world_size):
        self.world_size = world_size

def dynamic_batch_reorder(batch_dict, len_filter, trainer_instance=None, default_length=128):
    """
    复制的动态负载均衡函数用于测试
    """
    if not (hasattr(len_filter, 'scores') and len_filter.scores and 'index' in batch_dict):
        return batch_dict, None
    
    # Get rollout worker count for load balancing
    num_workers = 1
    if trainer_instance and hasattr(trainer_instance, 'actor_rollout_wg'):
        num_workers = trainer_instance.actor_rollout_wg.world_size
    
    try:
        # Extract previous response lengths from len_filter
        indices = batch_dict['index']
        data_sources = batch_dict.get('data_source', ['unknown'] * len(indices))
        
        # Build list of (original_index, predicted_length) pairs
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
            return batch_dict, None
        
        # Load balancing: distribute sequences across workers to balance total length per worker
        if len(length_predictions) >= num_workers:
            # Sort by length (descending) for better load balancing
            length_predictions.sort(key=lambda x: x[1], reverse=True)
            
            # Initialize worker buckets
            worker_buckets = [[] for _ in range(num_workers)]
            worker_loads = [0.0] * num_workers
            
            # Assign each sequence to the worker with the least current load
            for orig_idx, pred_len in length_predictions:
                # Find worker with minimum load
                min_load_worker = min(range(num_workers), key=lambda w: worker_loads[w])
                worker_buckets[min_load_worker].append(orig_idx)
                worker_loads[min_load_worker] += pred_len
            
            # Flatten worker buckets to get reorder indices
            reorder_indices = []
            for bucket in worker_buckets:
                reorder_indices.extend(bucket)
            
            # Only reorder if the order actually changes
            if reorder_indices != list(range(len(reorder_indices))):
                # Reorder all batch components according to load balancing
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
                        reordered_dict[key] = batch_dict[key]  # Keep unchanged if size doesn't match
                
                # Log load balancing results
                worker_load_info = [f"W{i}:{worker_loads[i]:.1f}" for i in range(num_workers)]
                print(f"🚀 Dynamic load balancing: reordered {len(reorder_indices)} sequences across {num_workers} workers: [{', '.join(worker_load_info)}]")
                
                # 返回详细信息用于测试验证
                return reordered_dict, {
                    'worker_buckets': worker_buckets,
                    'worker_loads': worker_loads,
                    'reorder_indices': reorder_indices,
                    'original_lengths': [pred_len for _, pred_len in length_predictions]
                }
                
    except Exception as e:
        print(f"⚠️ Dynamic load balancing failed, using original order: {e}")
    
    return batch_dict, None

def test_load_balancing():
    """测试负载均衡逻辑"""
    print("="*60)
    print("测试动态负载均衡重排序逻辑")
    print("="*60)
    
    # 测试用例1: 4个worker，8个序列，不同长度
    print("\n📋 测试用例1: 4个worker，8个序列")
    print("-"*40)
    
    # 创建模拟数据
    len_filter = MockScoreFilter()
    trainer = MockTrainer(world_size=4)
    
    # 设置不同的response长度
    lengths = [200, 50, 150, 80, 300, 100, 250, 120]
    indices = list(range(8))
    
    for i, length in enumerate(lengths):
        len_filter.add_score(f"unknown|{i}", length)
    
    # 创建batch_dict
    batch_dict = {
        'index': indices,
        'data_source': ['unknown'] * 8,
        'input_ids': torch.arange(8).unsqueeze(1),  # shape: [8, 1]
        'labels': list(range(8)),
        'metadata': np.array(['data_' + str(i) for i in range(8)], dtype=object)
    }
    
    print(f"原始序列长度: {lengths}")
    print(f"原始索引顺序: {indices}")
    
    # 执行负载均衡
    result, debug_info = dynamic_batch_reorder(batch_dict, len_filter, trainer)
    
    if debug_info:
        print(f"\n重排后索引顺序: {debug_info['reorder_indices']}")
        print(f"Worker分配情况:")
        for i, bucket in enumerate(debug_info['worker_buckets']):
            bucket_lengths = [lengths[idx] for idx in bucket]
            print(f"  Worker {i}: 索引{bucket} -> 长度{bucket_lengths} -> 总长度{debug_info['worker_loads'][i]:.1f}")
        
        # 验证负载均衡效果
        max_load = max(debug_info['worker_loads'])
        min_load = min(debug_info['worker_loads'])
        load_variance = max_load - min_load
        print(f"\n负载均衡评估:")
        print(f"  最大负载: {max_load:.1f}")
        print(f"  最小负载: {min_load:.1f}")
        print(f"  负载差异: {load_variance:.1f}")
        print(f"  负载方差: {np.var(debug_info['worker_loads']):.2f}")
        
        # 验证数据重排序正确性
        print(f"\n数据重排序验证:")
        original_labels = batch_dict['labels']
        reordered_labels = result['labels']
        expected_labels = [original_labels[i] for i in debug_info['reorder_indices']]
        
        print(f"  原始labels: {original_labels}")
        print(f"  重排后labels: {reordered_labels}")
        print(f"  预期labels: {expected_labels}")
        print(f"  重排序正确: {reordered_labels == expected_labels}")
        
        # 验证张量重排序
        original_tensor = batch_dict['input_ids'].flatten().tolist()
        reordered_tensor = result['input_ids'].flatten().tolist()
        expected_tensor = [original_tensor[i] for i in debug_info['reorder_indices']]
        
        print(f"  原始tensor: {original_tensor}")
        print(f"  重排后tensor: {reordered_tensor}")
        print(f"  张量重排序正确: {reordered_tensor == expected_tensor}")
    
    # 测试用例2: 边界情况 - 序列数少于worker数
    print("\n\n📋 测试用例2: 边界情况 - 2个序列，4个worker")
    print("-"*40)
    
    small_batch = {
        'index': [0, 1],
        'data_source': ['unknown', 'unknown'],
        'labels': [0, 1]
    }
    
    len_filter2 = MockScoreFilter()
    len_filter2.add_score("unknown|0", 100)
    len_filter2.add_score("unknown|1", 200)
    
    print(f"序列长度: [100, 200]")
    result2, debug_info2 = dynamic_batch_reorder(small_batch, len_filter2, trainer)
    
    if debug_info2:
        print("应该不会重排序，因为序列数 < worker数")
    else:
        print("✅ 正确：序列数少于worker数时，保持原始顺序")
    
    # 测试用例3: 没有历史数据
    print("\n\n📋 测试用例3: 没有历史数据")
    print("-"*40)
    
    empty_filter = MockScoreFilter()
    no_history_batch = {
        'index': [0, 1, 2],
        'data_source': ['unknown', 'unknown', 'unknown'],
        'labels': [0, 1, 2]
    }
    
    print(f"空的len_filter，没有历史数据")
    result3, debug_info3 = dynamic_batch_reorder(no_history_batch, empty_filter, trainer)
    
    if debug_info3 is None:
        print("✅ 正确：没有历史数据时，保持原始顺序")
    else:
        print("❌ 错误：应该保持原始顺序")

def test_load_balancing_algorithm():
    """专门测试负载均衡算法的正确性"""
    print("\n\n📋 负载均衡算法正确性测试")
    print("-"*40)
    
    # 测试极端情况：一个很大的任务和多个小任务
    lengths = [1000, 10, 20, 30, 40, 50, 60, 70]  # 总长度 1280
    num_workers = 4
    
    # 模拟分配过程
    length_predictions = [(i, lengths[i]) for i in range(len(lengths))]
    length_predictions.sort(key=lambda x: x[1], reverse=True)
    
    worker_buckets = [[] for _ in range(num_workers)]
    worker_loads = [0.0] * num_workers
    
    print(f"序列长度（降序）: {[x[1] for x in length_predictions]}")
    
    for orig_idx, pred_len in length_predictions:
        min_load_worker = min(range(num_workers), key=lambda w: worker_loads[w])
        worker_buckets[min_load_worker].append(orig_idx)
        worker_loads[min_load_worker] += pred_len
        print(f"分配序列{orig_idx}(长度{pred_len}) -> Worker{min_load_worker} (新负载: {worker_loads[min_load_worker]})")
    
    print(f"\n最终worker负载: {worker_loads}")
    print(f"负载标准差: {np.std(worker_loads):.2f}")
    print(f"平均负载: {np.mean(worker_loads):.1f}")
    
    # 与简单平均分配对比
    simple_avg = sum(lengths) / num_workers
    print(f"理想平均负载: {simple_avg:.1f}")

if __name__ == "__main__":
    test_load_balancing()
    test_load_balancing_algorithm()
    print("\n✅ 测试完成！")