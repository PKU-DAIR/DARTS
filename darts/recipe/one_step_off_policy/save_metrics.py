def _save_metrics_to_csv(metrics, global_step, epoch, experiment_name):
    """将metrics保存到CSV文件"""
    import json
    try:
        # 转换metrics中的所有值为Python原生类型
        serializable_metrics = {}
        for key, value in metrics.items():
            # 处理numpy类型和torch类型
            if hasattr(value, 'item'):
                # 处理torch.Tensor和numpy标量
                try:
                    serializable_metrics[key] = value.item() if hasattr(value, 'numel') and value.numel() == 1 else float(value)
                except:
                    serializable_metrics[key] = float(value)
            elif hasattr(value, 'dtype'):
                # 处理numpy数组等
                try:
                    if hasattr(value, 'tolist'):
                        serializable_metrics[key] = value.tolist()
                    else:
                        serializable_metrics[key] = float(value)
                except:
                    serializable_metrics[key] = float(value)
            else:
                # 其他类型直接保存
                serializable_metrics[key] = value
        
        # 将metrics转换为JSON字符串
        metrics_json = json.dumps(serializable_metrics, ensure_ascii=False)
        
        # 直接追加到CSV文件
        with open("/log.csv", 'a', encoding='utf-8') as f:
            f.write(f'"{experiment_name}",{global_step},{epoch},"{metrics_json}"\n')
            
        # 每100步打印一次保存状态
        if global_step % 100 == 0:
            print(f"💾 Metrics已保存到CSV (步骤 {global_step})")
            
    except Exception as e:
        print(f"❌ 保存metrics到CSV失败: {e}")
        # 如果JSON方法失败，使用备选方案
        _save_metrics_to_csv_backup(metrics, global_step, epoch, experiment_name)

def _save_metrics_to_csv_backup(metrics, global_step, epoch, experiment_name):
    """备用的metrics保存方法"""
    try:
        # 直接将所有metrics转换为字符串格式
        metrics_str_parts = []
        for key, value in metrics.items():
            # 统一转换为字符串
            try:
                if hasattr(value, 'item'):
                    # 处理torch.Tensor和numpy标量
                    str_value = str(value.item()) if hasattr(value, 'numel') and value.numel() == 1 else str(float(value))
                else:
                    str_value = str(value)
                metrics_str_parts.append(f"{key}:{str_value}")
            except:
                metrics_str_parts.append(f"{key}:{repr(value)}")
        
        # 用分号连接所有metrics
        metrics_str = ";".join(metrics_str_parts)
        
        # 直接追加到CSV文件
        with open("/log.csv", 'a', encoding='utf-8') as f:
            f.write(f'"{experiment_name}",{global_step},{epoch},"{metrics_str}"\n')
            
        print(f"💾 Metrics已用备用方法保存到CSV (步骤 {global_step})")
            
    except Exception as e:
        print(f"❌ 备用保存方法也失败: {e}")