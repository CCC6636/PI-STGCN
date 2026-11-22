import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# # # # ==================== 0-8063全局常量 ====================
WATER_GLOBAL_MIN = 2581
WATER_GLOBAL_MAX = 2613
WATER_GLOBAL_RANGE = WATER_GLOBAL_MAX - WATER_GLOBAL_MIN  

FLOW_GLOBAL_SCALE = 1700   # 全局最大绝对流量


def extract_columns_by_indices(csv_file, node_indices, q_indices):
    """根据索引提取列名"""
    df = pd.read_csv(csv_file, nrows=1)
    all_columns = df.columns.tolist()
    
    if isinstance(node_indices, str) and node_indices.startswith("range:"):
        range_str = node_indices.replace("range:", "")
        start, end = map(int, range_str.split('-'))
        node_columns = all_columns[start:end+1]
    else:
        node_indices_list = list(map(int, node_indices.split(',')))
        node_columns = [all_columns[i] for i in node_indices_list]
    
    q_indices_list = list(map(int, q_indices.split(',')))
    q_columns = [all_columns[i] for i in q_indices_list]
    
    return node_columns, q_columns


# ==================== 水位全局归一化 & 逆归一化 ====================
def normalize_water_global(Z_abs: np.ndarray) -> np.ndarray:
    """水位全局 Min-Max 归一化到 [0,1]"""
    return (Z_abs - WATER_GLOBAL_MIN) / WATER_GLOBAL_RANGE

def denormalize_water_global(Z_norm: np.ndarray) -> np.ndarray:
    """逆归一化回绝对水位"""
    return Z_norm * WATER_GLOBAL_RANGE + WATER_GLOBAL_MIN

# ==================== 流量全局归一化 & 逆归一化 ====================
def normalize_flow_global(Q_abs: np.ndarray) -> np.ndarray:
    """流量全局统一尺度归一化（带符号）"""
    return Q_abs / FLOW_GLOBAL_SCALE

def denormalize_flow_global(Q_norm: np.ndarray) -> np.ndarray:
    """逆归一化回绝对流量（推理时可能用到）"""
    return Q_norm * FLOW_GLOBAL_SCALE

# ==================== 变化量全局归一化（最关键！）===================
def compute_normalized_deltas_from_abs(
    target_water_abs: np.ndarray,      # [seq_len, node_num] 或 [1, seq_len, node_num]
    initial_water_abs: np.ndarray      # [time_steps, node_num] 或 [1, time_steps, node_num]
) -> np.ndarray:
    """
    计算归一化后的 target_deltas（训练目标）
    返回形状: [pred_len, node_num] 的 np.ndarray
    """
    # 确保是 numpy
    target = np.array(target_water_abs)
    initial = np.array(initial_water_abs)
    
    # 取初始序列最后一个时间步作为起点
    prev_water = initial[-1:]  # [1, node_num]
    
    # 计算每一时刻相对于上一步的绝对变化量
    abs_deltas = target - np.concatenate([prev_water, target[:-1]], axis=0)
    
    # 全局归一化变化量
    norm_deltas = abs_deltas / WATER_GLOBAL_RANGE
    
    return norm_deltas.astype(np.float32)

def load_water_flow_data(csv_file, node_columns, q_columns):
    """
    加载水位和流量数据 - 使用全局归一化
    """
    df = pd.read_csv(csv_file)
    Z_data = df[node_columns].values.astype(np.float32)
    Q_data = df[q_columns].values.astype(np.float32)
    
    # 应用全局归一化
    Z_norm = normalize_water_global(Z_data)
    Q_norm = normalize_flow_global(Q_data)
    
    # 创建stats结构（保持接口兼容）
    node_num = len(node_columns)
    z_stats = {i: {'type': 'global_minmax', 'min': WATER_GLOBAL_MIN, 'range': WATER_GLOBAL_RANGE} 
               for i in range(node_num)}
    
    q_num = len(q_columns)
    q_stats = {i: {'type': 'global_scale', 'scale': FLOW_GLOBAL_SCALE} 
               for i in range(q_num)}
    
    q_scalers = [{'type': 'global_scale', 'scale': FLOW_GLOBAL_SCALE, 'channel_idx': i} 
                for i in range(q_num)]
    
    # print(f"数据加载完成: 全局归一化水位形状 {Z_norm.shape}, 全局缩放流量形状 {Q_norm.shape}")
    
    return Z_norm, Q_norm, z_stats, q_scalers, Z_data, Q_data

def apply_normalization(conditions):
    """应用全局归一化到条件列表"""
    normalized_conditions = []
    for condition in conditions:
        Z_abs, Q_abs, _, _, _, _ = condition
        
        # 应用全局归一化
        Z_norm = normalize_water_global(Z_abs)
        Q_norm = normalize_flow_global(Q_abs)
        
        # 创建stats和scalers
        node_num = Z_abs.shape[1]
        z_stats = {i: {'type': 'global_minmax', 'min': WATER_GLOBAL_MIN, 'range': WATER_GLOBAL_RANGE} 
                  for i in range(node_num)}
        
        q_num = Q_abs.shape[1]
        q_stats = {i: {'type': 'global_scale', 'scale': FLOW_GLOBAL_SCALE} 
                  for i in range(q_num)}
        
        q_scalers = [{'type': 'global_scale', 'scale': FLOW_GLOBAL_SCALE, 'channel': i} 
                    for i in range(q_num)]
        
        normalized_conditions.append((Z_norm, Q_norm, z_stats, q_scalers, Z_abs, Q_abs))
    
    return normalized_conditions

class WaterFlowDataset(Dataset):
    """水位-流量数据集 - 使用全局归一化"""
    
    def __init__(self, conditions, time_steps=1):
        self.conditions = conditions
        self.time_steps = time_steps
        self.samples = self._create_samples()
    
    def _create_samples(self):
        """创建训练样本"""
        samples = []
        
        for condition in self.conditions:
            Z_norm, Q_norm, z_stats, q_scalers, Z_abs, Q_abs = condition
            seq_len = len(Z_norm)
            
            if seq_len <= self.time_steps + 1:
                continue
                 
            samples.append({
                'full_water': Z_norm,
                'full_water_abs': Z_abs,
                'full_flow': Q_norm,
                'full_flow_abs': Q_abs,
                'sequence_length': seq_len,
                'z_stats': z_stats,
                'q_scalers': q_scalers
            })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            torch.FloatTensor(sample['full_water']),
            torch.FloatTensor(sample['full_water_abs']),
            torch.FloatTensor(sample['full_flow']),
            sample['sequence_length']
        )

def collate_variable_length(batch):
    """处理变长序列的collate函数"""
    water_norm_sequences = [item[0] for item in batch]
    water_abs_sequences = [item[1] for item in batch]
    flow_sequences = [item[2] for item in batch]
    sequence_lengths = [item[3] for item in batch]
    
    # 找到最大长度
    max_len = max(sequence_lengths)
    batch_size = len(batch)
    node_num = water_norm_sequences[0].shape[1]
    flow_dim = flow_sequences[0].shape[1]
    
    # 初始化填充张量
    water_norm_padded = torch.zeros(batch_size, max_len, node_num)
    water_abs_padded = torch.zeros(batch_size, max_len, node_num)
    flow_padded = torch.zeros(batch_size, max_len, flow_dim)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    
    # 填充数据
    for i, (w_norm, w_abs, flow, seq_len) in enumerate(zip(water_norm_sequences, water_abs_sequences, flow_sequences, sequence_lengths)):
        actual_len = min(seq_len, max_len)
        water_norm_padded[i, :actual_len] = w_norm[:actual_len]
        water_abs_padded[i, :actual_len] = w_abs[:actual_len]
        flow_padded[i, :actual_len] = flow[:actual_len]
        mask[i, :actual_len] = True
    
    return water_norm_padded, water_abs_padded, flow_padded, mask, sequence_lengths

def create_data_loaders(all_conditions, time_steps=1, batch_size=8, val_ratio=0.111, random_seed=42):
    """创建数据加载器 - 使用全局归一化"""
    # 划分训练和验证条件
    train_conditions, val_conditions = train_test_split(
        all_conditions, test_size=val_ratio, random_state=random_seed
    )
    
    train_dataset = WaterFlowDataset(train_conditions, time_steps)
    val_dataset = WaterFlowDataset(val_conditions, time_steps)
    
    train_generator = torch.Generator()
    train_generator.manual_seed(random_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_variable_length,
        generator=train_generator
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate_variable_length
    )
    
    print(f"训练工况: {len(train_dataset)}, 验证工况: {len(val_dataset)}")
    return train_loader, val_loader