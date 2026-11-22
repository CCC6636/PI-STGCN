# DOCUMENT filename="model_core.py"
import torch
import torch.nn as nn
import torch.nn.functional as F

class FlowInfluenceModule(nn.Module):
    """
    模块1: 物理驱动计算 (极简矩阵乘法重构版)
    职责: 将泵站流量 [B, T, K] 通过统一的物理矩阵一步映射为全网空间影响 [B, T, N]
    """
    def __init__(self, influence_matrix, injection_nodes=[0, 9, 55], impact_scale=1.0):
        super().__init__()
        self.scale = impact_scale
        
        # 将传入的统一物理矩阵 (56x56) 转为 tensor
        if influence_matrix is not None:
            matrix_tensor = torch.tensor(influence_matrix, dtype=torch.float32)
        else:
            # 容错处理：如果没传矩阵，默认用单位矩阵
            matrix_tensor = torch.eye(56, dtype=torch.float32)
            
        K = len(injection_nodes)
        N = matrix_tensor.shape[1]
        
        # 构建 [K, N] 的权重矩阵 (例如 3 x 56)
        M_subset = torch.zeros((K, N), dtype=torch.float32)
        
        # 终极简化：直接提取注入点对应的那一行物理绝对值向量
        for i, node_idx in enumerate(injection_nodes):
            M_subset[i, :] = matrix_tensor[node_idx, :]
            
        # 注册为 buffer：跟随模型保存，自动转移到 GPU，不参与梯度更新
        self.register_buffer('influence_weights', M_subset)
        
    def forward(self, flow_sequence):
        # flow_sequence shape: [Batch, Time, K] 
        # influence_weights shape: [K, N]
        # 矩阵乘法直接完成所有注入点的物理扩散: [B, T, K] @ [K, N] -> [B, T, N]
        effects = torch.matmul(flow_sequence, self.influence_weights)
        
        return effects * self.scale


class NodeWiseWaterEncoder(nn.Module):
    """
    模块2: 水位状态编码器
    """
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, x):
        # x: [B, 56]
        x = x.unsqueeze(-1) 
        return self.mlp(x)


class FlowEffectEncoder(nn.Module):
    """
    模块3: 流量驱动编码器
    """
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
    
    def forward(self, x):
        x = x.unsqueeze(-1)
        return self.mlp(x)


# class SpatialGraphConv(nn.Module):
#     def __init__(self, node_num, input_dim, hidden_dim, adj_matrix=None):
#         super().__init__()
        
#         # 1. 固定物理矩阵
#         if adj_matrix is not None:
#             self.fixed_adj = nn.Parameter(torch.FloatTensor(adj_matrix), requires_grad=False)
#         else:
#             self.fixed_adj = nn.Parameter(torch.eye(node_num), requires_grad=False)
            
#         # 2. 动态可学习矩阵 (初始化为0)
#         self.learnable_adj = nn.Parameter(torch.zeros(node_num, node_num))
#         nn.init.normal_(self.learnable_adj, mean=0.0, std=0.01)
#         self.alpha = nn.Parameter(torch.tensor(0.1)) 
        
#         # 3. 特征投影
#         self.proj = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim)
#         )
        
#         # 4. 激活函数 (新增，增加非线性能力)
#         self.activation = nn.ReLU()

#     def get_current_adj(self):
#         """用于导出热力图"""
#         raw_adj = self.fixed_adj + self.alpha * torch.tanh(self.learnable_adj)
#         # 使用 Softmax 归一化便于可视化和理解
#         return F.softmax(raw_adj, dim=-1).detach().cpu().numpy()

#     def forward(self, x):
#         # x: [B, N, Input_Dim]
#         x_trans = self.proj(x)
        
#         # 1. 计算当前矩阵
#         raw_adj = self.fixed_adj + self.alpha * torch.tanh(self.learnable_adj)
        
#         # 2. Softmax 归一化 (关键修正: 处理负值，保证稳定性)
#         current_adj = F.softmax(raw_adj, dim=-1)
        
#         # 3. 矩阵乘法聚合
#         adj_expand = current_adj.unsqueeze(0).expand(x.size(0), -1, -1)
#         out = torch.bmm(adj_expand, x_trans)
        
#         # 4. 激活 (关键修正)
#         out = self.activation(out)
#         return out

class SpatialGraphConv(nn.Module):
    def __init__(self, node_num, input_dim, hidden_dim, up_matrix=None, down_matrix=None):
        super().__init__()
        
        # 1. 物理拓扑底座 (剥离了顺流和逆流的骨架)
        if up_matrix is not None and down_matrix is not None:
            self.register_buffer('up_matrix', torch.FloatTensor(up_matrix))
            self.register_buffer('down_matrix', torch.FloatTensor(down_matrix))
        else:
            self.register_buffer('up_matrix', torch.zeros(node_num, node_num))
            self.register_buffer('down_matrix', torch.zeros(node_num, node_num))
            
        # 定义两个可学习的物理标量！
        # 初始值设为我们的经验值 (1.0 和 0.2)，但 PyTorch 会在训练中自己微调它们
        self.w_up = nn.Parameter(torch.tensor(1.0))
        self.w_down = nn.Parameter(torch.tensor(0.2))
        
        # 2. 动态可学习残差矩阵 (保留，用于捕捉纯物理之外的隐含关系)
        self.learnable_adj = nn.Parameter(torch.zeros(node_num, node_num))
        nn.init.normal_(self.learnable_adj, mean=0.0, std=0.01)
        self.alpha = nn.Parameter(torch.tensor(0.1)) 
        
        # 3. 特征投影
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.activation = nn.ReLU()

    def get_current_adj(self):
        """用于导出热力图"""
        base_adj = (self.up_matrix * self.w_up) + (self.down_matrix * self.w_down)
        base_adj = base_adj + torch.eye(base_adj.size(0), device=base_adj.device)
        raw_adj = base_adj + self.alpha * torch.tanh(self.learnable_adj)
        return F.softmax(raw_adj, dim=-1).detach().cpu().numpy()

    def forward(self, x):
        x_trans = self.proj(x)
        
        # 1. 动态物理加权：用最新的 w_up 和 w_down 组合骨架
        base_adj = (self.up_matrix * self.w_up) + (self.down_matrix * self.w_down)
        
        # 2. 补上对角线 1.0 (自身状态延续)
        base_adj = base_adj + torch.eye(base_adj.size(0), device=base_adj.device)
        
        # 3. 加上非物理的隐式残差学习
        raw_adj = base_adj + self.alpha * torch.tanh(self.learnable_adj)
        
        # 4. Softmax 行归一化 (自带防止梯度爆炸和比例统一的功能)
        current_adj = F.softmax(raw_adj, dim=-1)
        
        # 5. 矩阵乘法聚合
        adj_expand = current_adj.unsqueeze(0).expand(x.size(0), -1, -1)
        out = torch.bmm(adj_expand, x_trans)
        
        out = self.activation(out)
        return out