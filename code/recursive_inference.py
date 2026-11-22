import torch
import numpy as np
from data_utils import *

def denormalize_water_levels(Z_norm):
    """逆变换标准化水位到绝对水位 - 使用全局归一化"""
    return denormalize_water_global(Z_norm)

def denormalize_deltas(deltas_norm: np.ndarray) -> np.ndarray:
    """预测的归一化变化量 → 绝对变化量（米）"""
    return deltas_norm * WATER_GLOBAL_RANGE

class RecursiveInference:
    """递归水位预测推理器 - 使用全局归一化"""
    
    def __init__(self, model, device=None):
        self.model = model
        self.device = device if device else next(model.parameters()).device
        self.model.eval()
    
    def predict_full_sequence(self, initial_water_norm, preset_flow_norm, time_steps=1, initial_flow_norm=None):
        """
        递归预测完整水位序列 - 完全在归一化空间内累加，最后一次性逆归一化
        """
        # 处理输入格式
        if isinstance(initial_water_norm, list):
            initial_water_norm = np.array(initial_water_norm)
        
        if initial_water_norm.ndim == 1:
            initial_water_norm = np.tile(initial_water_norm, (time_steps, 1))
        
        pred_len = len(preset_flow_norm)
        
        # 转换为张量
        initial_water_tensor = torch.FloatTensor(initial_water_norm).unsqueeze(0).to(self.device)
        target_flow_tensor = torch.FloatTensor(preset_flow_norm).unsqueeze(0).to(self.device)
        
        # 处理初始流量
        if initial_flow_norm is None:
            initial_flow_norm = np.zeros((time_steps, 3))
        initial_flow_tensor = torch.FloatTensor(initial_flow_norm).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            self.model.reset_state()
            
            # 模型直接输出归一化后的变化量 delta_norm (移除 TF 参数)
            predicted_deltas_norm = self.model(
                initial_water=initial_water_tensor,
                initial_flow=initial_flow_tensor,
                full_flow_sequence=target_flow_tensor
            )
            
            predicted_deltas_norm = predicted_deltas_norm.cpu().numpy()[0]  # [pred_len, node_num]
        
        # 取初始序列的最后一个归一化水位作为起点
        current_water_norm = initial_water_norm[-1].copy()  # [node_num]
        predicted_water_norm_list = []
        
        for t in range(pred_len):
            # 在归一化空间直接累加归一化变化量
            current_water_norm = current_water_norm + predicted_deltas_norm[t]
            predicted_water_norm_list.append(current_water_norm.copy())
        
        predicted_water_norm = np.stack(predicted_water_norm_list, axis=0)  # [pred_len, node_num]
        
        # ==================== 最后一次性逆归一化到绝对水位 ====================
        from data_utils import denormalize_water_global
        predicted_water_abs = denormalize_water_global(predicted_water_norm)
        
        return predicted_water_abs
    
    def predict_single_condition(self, condition_index, all_conditions, time_steps=1):
        """
        预测单个工况的完整序列 - 使用全局归一化
        """
        condition = all_conditions[condition_index]
        if len(condition) < 6:
            raise ValueError(f"Invalid condition format at index {condition_index}")
        
        Z_norm, Q_norm, _, _, Z_abs, Q_abs = condition
        seq_len = len(Z_norm)
        
        if seq_len <= time_steps:
            return None, None
        
        # 初始标准化水位 Z_norm[0:time_steps]
        initial_water_norm = Z_norm[:time_steps]
        # 初始标准化流量 Q_norm[0:time_steps]
        initial_flow_norm = Q_norm[:time_steps]
        # 流量从 time_steps 开始 Q_norm[time_steps:]
        preset_flow_norm = Q_norm[time_steps:]
        
        # 预测绝对水位
        predicted_water_abs = self.predict_full_sequence(initial_water_norm, preset_flow_norm, time_steps, initial_flow_norm)
        
        # 真实绝对水位 Z_abs[time_steps:]
        true_water_abs = Z_abs[time_steps:]
        
        return predicted_water_abs, true_water_abs
    
    def evaluate_on_conditions(self, test_conditions, time_steps=1):
        """在测试工况上评估模型性能 - 使用全局归一化"""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        condition_metrics = []
        
        for i, condition in enumerate(test_conditions):
            Z_norm, Q_norm, _, _, Z_abs, Q_abs = condition
            seq_len = len(Z_norm)
            
            if seq_len <= time_steps + 1:
                continue
            
            # 预测
            predicted_water_abs, true_water_abs = self.predict_single_condition(i, test_conditions, time_steps)
            
            if predicted_water_abs is not None and true_water_abs is not None:
                # 计算指标 (绝对值)
                valid_mask = np.abs(true_water_abs) > 1e-8
                
                mse = np.mean((predicted_water_abs - true_water_abs) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(predicted_water_abs - true_water_abs))
                
                if np.any(valid_mask):
                    relative_error = np.mean(
                        np.abs((predicted_water_abs[valid_mask] - true_water_abs[valid_mask]) / 
                              np.abs(true_water_abs[valid_mask]))
                    ) * 100
                else:
                    relative_error = 0.0
                
                condition_metrics.append({
                    'condition_idx': i,
                    'sequence_length': seq_len,
                    'rmse': rmse,
                    'mae': mae,
                    'relative_error': relative_error
                })
                
                all_predictions.append(predicted_water_abs.flatten())
                all_targets.append(true_water_abs.flatten())
                
                print(f"工况 {i}: RMSE = {rmse:.6f}, MAE = {mae:.6f}, 相对误差 = {relative_error:.2f}%")
        
        # 总体指标
        if all_predictions:
            all_pred = np.concatenate(all_predictions)
            all_true = np.concatenate(all_targets)
            
            # 创建有效掩码
            valid_mask = np.abs(all_true) > 1e-8
            
            overall_rmse = np.sqrt(np.mean((all_pred - all_true) ** 2))
            overall_mae = np.mean(np.abs(all_pred - all_true))
            
            if np.any(valid_mask):
                overall_relative = np.mean(
                    np.abs((all_pred[valid_mask] - all_true[valid_mask]) / 
                          np.abs(all_true[valid_mask]))
                ) * 100
            else:
                overall_relative = 0.0
            
            print(f"\n总体性能:")
            print(f"RMSE: {overall_rmse:.6f}")
            print(f"MAE: {overall_mae:.6f}") 
            print(f"平均相对误差: {overall_relative:.2f}%")
            
            return {
                'condition_metrics': condition_metrics,
                'overall_rmse': overall_rmse,
                'overall_mae': overall_mae,
                'overall_relative_error': overall_relative
            }
        
        return None

