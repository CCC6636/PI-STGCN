import torch
import torch.nn as nn
import torch.nn.functional as F
from model_core import *
from data_utils import *
torch.set_printoptions(precision=6, sci_mode=False)
import torch
import torch.nn as nn
from model_core import *

class ImprovedWaterModel(nn.Module):
    def __init__(self, node_num, time_steps=1, hidden_dim=512,
                 up_matrix=None, down_matrix=None, influence_matrix=None, num_layers=2):
        super().__init__()
        self.node_num = node_num
        self.hidden_dim = hidden_dim
        self.time_steps = time_steps

        # 1. 物理驱动 (改为传入统一的 influence_matrix)
        self.flow_influence = FlowInfluenceModule(
            influence_matrix=influence_matrix,
            injection_nodes=[0, 9, 55],
            impact_scale=1.0 
        )

        # 2. 编码器 (不变)
        self.water_encoder = NodeWiseWaterEncoder(hidden_dim=hidden_dim)
        self.flow_effect_encoder = FlowEffectEncoder(hidden_dim=hidden_dim)

        # 3. GCN (使用你的新版 GCN)
        self.spatial_gcn = SpatialGraphConv(
            node_num=node_num,
            input_dim=hidden_dim * 2,
            hidden_dim=hidden_dim,
            up_matrix=up_matrix,      # ⬅️ 传入上游骨架
            down_matrix=down_matrix   # ⬅️ 传入下游骨架
        )

        # 4. 解码器 (使用上面的混合版 LSTMDecoder)
        self.decoder = LSTMDecoder(
            node_num=node_num,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )

        self.delta_scale = nn.Parameter(torch.tensor(0.15))

    def reset_state(self):
        self.decoder.reset_state()

    def forward(self, initial_water, initial_flow, full_flow_sequence):
        # A. 物理场计算
        flow_effects_map = self.flow_influence(full_flow_sequence)

        # B. 准备初始状态
        current_water = initial_water[:, -1, :]        
        current_flow_effect = flow_effects_map[:, 0, :] 

        # C. 编码
        w_feat = self.water_encoder(current_water)          
        f_feat = self.flow_effect_encoder(current_flow_effect) 

        # D. GCN 融合 (只运行一次，用于初始化)
        combined_feat = torch.cat([w_feat, f_feat], dim=-1) 
        spatial_context = self.spatial_gcn(combined_feat)   

        # E. 解码预测
        predicted_deltas = self.decoder(
            initial_context=spatial_context, # GCN 结果传进去
            initial_water_val=current_water,
            flow_effects_sequence=flow_effects_map,
            delta_scale=self.delta_scale # 0.15
        )

        return predicted_deltas


class LSTMDecoder(nn.Module):
    """
    [混合架构版 LSTM 解码器]
    - 初始化: 使用 GCN 提取的高级物理特征 (Smart Init)
    - 推演: 使用全量节点数据 [B, 112] 进行 Bottleneck 推演 (Robust Body)
    """
    def __init__(self, node_num, hidden_dim, num_layers=2):
        super().__init__()
        self.node_num = node_num
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 1. 状态初始化层 (保留新模型的优势)
        # 将 GCN 的 spatial_context [B, 56, H] 聚合后初始化 LSTM
        self.init_proj = nn.Linear(hidden_dim, hidden_dim * num_layers * 2) 

        # 2. 输入映射层 (关键修改: 回归旧模型逻辑)
        # 能够看到所有节点的水位(56) + 流量影响(56) = 112维
        self.input_proj = nn.Linear(node_num * 2, hidden_dim)

        # 3. LSTM 核心
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # 4. 输出映射层 (Bottleneck 机制)
        # 通过一个共享的 MLP 直接预测所有 56 个节点
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, node_num), # 输出 56 维向量
            nn.Tanh() # 限制范围，保证稳定性
        )

    def init_hidden(self, spatial_context):
        """利用 GCN 上下文初始化 LSTM (混合架构的精髓)"""
        # spatial_context: [B, 56, H]
        # 对空间特征取平均，作为全局初始状态
        ctx_mean = spatial_context.mean(dim=1) # [B, H]
        
        raw_init = self.init_proj(ctx_mean)
        raw_init = raw_init.view(spatial_context.size(0), self.num_layers, 2, self.hidden_dim)
        
        h0 = raw_init[:, :, 0, :].permute(1, 0, 2).contiguous()
        c0 = raw_init[:, :, 1, :].permute(1, 0, 2).contiguous()
        return h0, c0
        
    def reset_state(self): pass
        
    def forward(self, initial_context, initial_water_val, flow_effects_sequence, 
                teacher_forcing_ratio=0.0, targets=None, delta_scale=0.15):
        
        B, T, N = flow_effects_sequence.shape
        
        # Step 1: GCN 仅在此处发挥作用 (初始化)
        hidden = self.init_hidden(initial_context)
        
        curr_water = initial_water_val # [B, 56]
        all_deltas = []
        
        # Step 2: LSTM 循环 (GCN 退场，LSTM 接管)
        for t in range(T):
            # 获取当前流量影响 [B, 56]
            curr_flow_effect = flow_effects_sequence[:, t, :] 
            
            # --- 关键修改: 全量拼接 ---
            # input: [B, 56+56] = [B, 112]
            step_input = torch.cat([curr_water, curr_flow_effect], dim=1)
            
            # 映射 [B, 112] -> [B, 1, H]
            lstm_in = self.input_proj(step_input).unsqueeze(1) 
            
            # LSTM 前向
            lstm_out, hidden = self.lstm(lstm_in, hidden)
            
            # 预测 [B, 1, H] -> [B, 56]
            delta_raw = self.out_proj(lstm_out.squeeze(1)) 
            
            # 缩放 (使用传入的 delta_scale, 也就是 0.15)
            delta = delta_raw * delta_scale
            all_deltas.append(delta.unsqueeze(1))
            
            curr_water = curr_water + delta
                
        return torch.cat(all_deltas, dim=1)


# class ImprovedPhysicsAwareLoss(nn.Module):
#     def __init__(self, physics_weight=0.1, smooth_weight=0.05):
#         super().__init__()
#         self.physics_weight = physics_weight
#         self.smooth_weight = smooth_weight
#         self.mae_loss = nn.L1Loss()
        
#     def forward(self, predicted_deltas_norm, target_deltas_norm, initial_water_abs):
#         """
#         predicted_deltas_norm: [batch, pred_len, node_num] 预测的归一化变化量
#         target_deltas_norm: [batch, pred_len, node_num] 真实的归一化变化量  
#         initial_water_abs: [batch, node_num] 初始绝对水位
#         """
#         batch_size, pred_len, node_num = predicted_deltas_norm.shape
        
#         # 转换为绝对变化量
#         pred_deltas_abs = predicted_deltas_norm * WATER_GLOBAL_RANGE
#         target_deltas_abs = target_deltas_norm * WATER_GLOBAL_RANGE
        
#         # 计算累积绝对水位
#         initial_water = initial_water_abs.unsqueeze(1)  # [batch, 1, node_num]
        
#         # 累积预测水位
#         pred_water_abs = initial_water + torch.cumsum(pred_deltas_abs, dim=1)
#         # 累积真实水位  
#         true_water_abs = initial_water + torch.cumsum(target_deltas_abs, dim=1)
        
#         # 主要损失：绝对水位的MAE
#         mae_abs = self.mae_loss(pred_water_abs, true_water_abs)
        
#         # # 辅助损失：物理约束（可选）
#         # physics_loss = self.physics_constraint_loss(predicted_deltas_norm, target_deltas_norm)
        
#         # 平滑性约束（可选）
#         smooth_loss = self.smoothness_loss(predicted_deltas_norm)
        
#         # 组合损失
#         total_loss = mae_abs  + self.smooth_weight * smooth_loss
        
#         loss_details = {
#             'total': total_loss.item(),
#             'absolute_mae': mae_abs.item(),
#             'physics_constraint': physics_loss.item() if physics_loss is not None else 0.0,
#             'smoothness': smooth_loss.item() if smooth_loss is not None else 0.0
#         }
        
#         return total_loss, loss_details
    
#     # def physics_constraint_loss(self, pred_deltas_norm, target_deltas_norm):
#     #     """简化的物理约束"""
#     #     # 可以保留部分物理约束，但权重较低
#     #     return torch.tensor(0.0, device=pred_deltas_norm.device)
    
#     def smoothness_loss(self, pred_deltas_norm):
#         """变化量平滑性约束"""
#         if pred_deltas_norm.shape[1] <= 1:
#             return torch.tensor(0.0, device=pred_deltas_norm.device)
        
#         delta_changes = pred_deltas_norm[:, 1:, :] - pred_deltas_norm[:, :-1, :]
#         return torch.mean(torch.abs(delta_changes))
class ImprovedPhysicsAwareLoss(nn.Module):
    def __init__(self, smooth_weight=0.05):
        super().__init__()
        self.smooth_weight = smooth_weight
        self.mae_loss = nn.L1Loss()
        
    def forward(self, predicted_deltas_norm, target_deltas_norm, initial_water_abs):
        """
        predicted_deltas_norm: [batch, pred_len, node_num] 预测的归一化变化量
        target_deltas_norm: [batch, pred_len, node_num] 真实的归一化变化量  
        initial_water_abs: [batch, node_num] 初始绝对水位
        """
        batch_size, pred_len, node_num = predicted_deltas_norm.shape
        
        # 转换为绝对变化量
        pred_deltas_abs = predicted_deltas_norm * WATER_GLOBAL_RANGE
        target_deltas_abs = target_deltas_norm * WATER_GLOBAL_RANGE
        
        # 计算累积绝对水位
        initial_water = initial_water_abs.unsqueeze(1)  # [batch, 1, node_num]
        
        # 累积预测水位
        pred_water_abs = initial_water + torch.cumsum(pred_deltas_abs, dim=1)
        # 累积真实水位  
        true_water_abs = initial_water + torch.cumsum(target_deltas_abs, dim=1)
        
        # 主要损失：绝对水位的MAE
        mae_abs = self.mae_loss(pred_water_abs, true_water_abs)
        
        # 时间平滑约束：惩罚相邻预测水位增量的突变。
        smooth_loss = self.smoothness_loss(predicted_deltas_norm)

        # 实际优化目标：绝对水位 MAE + 加权增量平滑损失。
        # 本实现不额外计算质量守恒或动量守恒方程残差。
        total_loss = mae_abs + self.smooth_weight * smooth_loss

        loss_details = {
            'total': total_loss.item(),
            'absolute_mae': mae_abs.item(),
            'smoothness': smooth_loss.item()
        }
        
        return total_loss, loss_details
    
    def smoothness_loss(self, pred_deltas_norm):
        """变化量平滑性约束"""
        if pred_deltas_norm.shape[1] <= 1:
            return torch.tensor(0.0, device=pred_deltas_norm.device)
        
        delta_changes = pred_deltas_norm[:, 1:, :] - pred_deltas_norm[:, :-1, :]
        return torch.mean(torch.abs(delta_changes))