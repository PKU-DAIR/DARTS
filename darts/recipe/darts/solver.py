import numpy as np
import math
from scipy.interpolate import interp1d

def repeat_by_position(numbers):
    """
    根据位置重复数字
    
    Args:
        numbers: 输入数字列表，如 [1, 3, 2]
        
    Returns:
        重复后的列表，如 [1, 3, 3, 3, 2, 2]
        (第0个数重复1次，第1个数重复2次，第2个数重复3次...)
    """
    result = []
    for i, num in enumerate(numbers):
        result.extend([i] * num)
    return result

class ResourceOptimizer:
    def __init__(self, v_list, total_sum, min_m=16, max_m=50):
        self.v = v_list
        self.N = len(v_list)
        self.total = total_sum
        self.min_m = min_m
        self.max_m = max_m
        self.m = None 
        self._check_feasibility()
        self.v = self.normalize(self.v)

    def normalize(self, arr):
        """归一化数组到[0, 1]范围"""
        arr = np.array(arr)
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val - min_val == 0:
            return np.ones_like(arr)
        return (arr - min_val + 1)  / (max_val - min_val)
    
    def _check_feasibility(self):
        """检查总资源是否在可行范围内"""
        min_possible = self.N * self.min_m
        max_possible = self.N * self.max_m
        if self.total < min_possible:
            raise ValueError(f"总资源不足：至少需要{min_possible}，但给定{self.total}")
        if self.total > max_possible:
            raise ValueError(f"总资源过剩：最多只能分配{max_possible}，但给定{self.total}")

    def _marginal_gain(self, i):
        """计算第i个变量增加1单位m_i的边际收益（时间减少量）"""
        current_m = self.m[i]
        if current_m >= self.max_m:
            return -np.inf  # 已达上界，无法再分配
        # 计算t(v_i, m_i)和t(v_i, m_i+1)的差值（边际收益）
        # 因t与m_i近似反比，此处以t = v_i / m_i为例（可替换为实际t函数）
        t_current = self._time_func(current_m, self.v[i])
        t_next = self._time_func(current_m + 1, self.v[i])
        return t_current - t_next  # 正值，越大说明收益越高

    def _time_func(self, m, v):
        """时间函数：t(v, m)，与m近似反比（可替换为实际表达式）"""
        return v / m


   
    def optimize(self):
        """执行贪心优化，返回最优m_i列表"""
        # 初始化：所有m_i先分配下界
        self.m = [self.min_m] * self.N
        remaining = self.total - sum(self.m)  # 剩余可分配资源
        self._check_feasibility()

        # 迭代分配剩余资源
        while remaining > 0:
            # 计算每个变量的边际收益
            gains = [self._marginal_gain(i) for i in range(self.N)]
            # 选择边际收益最大的变量（若多个相同，选索引最小的）
            best_i = np.argmax(gains)
            # 分配1单位资源
            self.m[best_i] += 1
            remaining -= 1

        return self.m

    def verify(self):
        """验证结果是否满足所有约束"""
        if self.m is None:
            raise RuntimeError("请先调用optimize()进行优化")
        # 检查总和
        #if sum(self.m) != self.total:
        #    raise ValueError(f"总和不匹配：计算得{sum(self.m)}，预期{self.total}")
        # 检查上下界
        for mi in self.m:
            if not (self.min_m <= mi <= self.max_m):
                raise ValueError(f"m_i={mi}超出范围[{self.min_m}, {self.max_m}]")
        print("结果验证通过：所有约束均满足")

    def total_time(self):
        """计算总时间Σt(v_i, m_i)"""
        if self.m is None:
            raise RuntimeError("请先调用optimize()进行优化")
        return sum(self._time_func(mi, vi) for mi, vi in zip(self.m, self.v))


if __name__ == "__main__":
    # 测试 repeat_by_position 函数
    a = [1, 3, 2]
    result = repeat_by_position(a)
    print(result)  # 输出: [0, 1, 1, 1, 2, 2]

    # 测试 ResourceOptimizer 类
    
    
    v_list = [70.3, 325.31, 97.38, 61.3, 165.26, 63.07, 43.01, 71.49, 242.19, 186.97, 321.57, 36.03, 143.56, 167.52, 67.6, 95.87, 172.7, 287.77, 113.49, 66.66, 134.76, 113.42, 170.93, 47.08, 97.79, 176.61, 231.01, 80.08, 228.06, 229.72, 209.06, 240.27, 125.13, 123.55, 254.21, 137.66, 84.23, 155.83, 212.54, 111.44, 43.01, 104.41, 84.74, 553.39, 133.87, 77.81, 157.65, 70.55, 122.77, 226.7, 178.22, 284.9, 134.93, 207.23, 148.56, 266.89, 221.4, 136.71, 311.76, 410.15, 178.07, 156.43, 163.88, 238.08, 182.7, 152.18, 196.0, 142.65, 179.61, 149.28, 288.59, 112.58, 185.05, 77.65, 124.58, 129.07, 321.59, 158.68, 77.95, 121.07, 76.44, 181.93, 138.29, 191.65, 100.13, 213.16, 121.69, 214.81, 510.12, 43.42, 144.63, 84.69, 167.26, 223.09, 162.48, 321.88, 85.4, 232.35, 156.4, 184.37, 186.25, 143.84, 228.35, 32.1, 256.76, 251.88, 198.98, 114.14, 235.46, 69.61, 308.43, 70.64, 147.99, 275.27, 134.31, 91.2, 97.91, 127.93, 146.24, 184.48, 246.36, 116.3, 122.69, 251.44, 48.32, 132.97, 100.68, 154.85, 147.34, 169.74, 89.9, 84.49, 63.04, 96.64, 99.87, 223.51, 332.71, 64.84, 212.12, 343.22, 201.17, 189.51, 175.08, 115.08, 149.82, 286.64, 177.69, 134.76, 42.03, 122.1, 171.33, 236.21, 171.64, 323.84, 70.24, 250.58, 132.21, 74.69, 249.96, 97.11, 172.81, 268.37, 156.84, 114.16, 102.5, 206.79, 227.5, 164.53, 223.75, 360.25, 197.96, 154.69, 244.92, 213.67, 103.21, 120.56, 53.23, 117.97, 179.86, 137.77, 139.3, 172.95, 117.71, 108.11, 114.1, 137.55, 127.82, 204.81, 235.11, 116.48, 107.05, 168.6, 129.2, 104.43, 95.77, 153.91, 204.64, 244.34, 195.82, 71.09, 125.39, 100.81, 128.88, 174.33, 154.0, 160.15, 247.86, 204.28, 202.6, 146.15, 111.86, 132.03, 195.95, 223.1, 107.05, 136.14, 33.33, 99.75, 186.06, 120.0, 139.03, 213.03, 174.21, 47.02, 174.51, 37.95, 151.98, 189.68, 69.23, 164.97, 60.75, 150.79, 259.33, 118.98, 183.79, 53.91, 205.29, 254.78, 63.03, 84.66, 110.54, 72.38, 286.48, 176.53, 290.26, 315.86, 148.74, 146.19, 36.37, 41.08, 179.92, 115.4, 124.89, 115.54, 147.19, 132.77]
    #v_list = [1 for _ in range(256)]
    v_list = sorted(v_list)
    #v_list[0]=100
    print(f"排序后的v_list: {v_list}")
    total_sum =256 * 12
    optimizer = ResourceOptimizer(v_list, total_sum, min_m=10, max_m=16)
    m_opt = optimizer.optimize()
    print(f"优化结果 m_i: {m_opt}")
    optimizer.verify()
    total_time = optimizer.total_time()
    print(f"总时间 Σt(v_i, m_i): {total_time}")