import torch
import glob
import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from data_utils import *
from recursive_model import ImprovedWaterModel, ImprovedPhysicsAwareLoss
from recursive_trainer import RecursiveTrainer
from recursive_inference import RecursiveInference


def set_random_seed(seed=42):
    """Set random seeds for repeatable runs on the same software/hardware stack."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_raw_matrix(matrix_file, node_columns):
    """加载影响矩阵"""
    try:
        df = pd.read_csv(matrix_file, index_col=0)
        
        # 清洁列名和索引
        def clean_name(name):
            name = name.strip()
            if '(' in name and ')' in name:
                name = name.split('(')[-1].split(')')[0].strip()
            elif ':' in name:
                name = name.split(':')[0].strip()
            return name
        
        df.columns = [clean_name(col) for col in df.columns]
        df.index = [clean_name(idx) for idx in df.index]
        node_columns_clean = [clean_name(col) for col in node_columns]
        
        # 检查节点匹配
        missing_columns = set(node_columns_clean) - set(df.columns)
        missing_rows = set(node_columns_clean) - set(df.index)
        if missing_columns or missing_rows:
            print(
                f"错误: 影响矩阵节点标签不完整。缺少列: {missing_columns}; "
                f"缺少行: {missing_rows}"
            )
            return None
        
        # 按node_columns顺序重新排列
        try:
            df_reordered = df.reindex(index=node_columns_clean, columns=node_columns_clean)
            influence_matrix = df_reordered.values.astype(float)
            
            print(f"成功加载影响矩阵，形状: {influence_matrix.shape}")
            print(f"原始矩阵范围: [{influence_matrix.min():.3f}, {influence_matrix.max():.3f}]")
            print(f"正影响数量: {(influence_matrix > 0).sum()}, 负影响数量: {(influence_matrix < 0).sum()}")
            return influence_matrix
        except Exception as e:
            print(f"重新排列影响矩阵失败: {e}；返回加载失败状态")
            return None
            
    except Exception as e:
        print(f"加载影响矩阵失败: {e}；返回加载失败状态")
        return None
        
def get_directional_physics_matrices(raw_matrix):
    """
    将源节点为行、目标节点为列的原始影响矩阵，转换为 GCN 使用的方向矩阵。

    原始矩阵约定：raw_matrix[source, target]。
    GCN 聚合约定：A[target, source]，以便执行 H_new = A @ H。
    因此先按原始符号拆分，再转置为目标行、源列。
    """
    node_num = raw_matrix.shape[0]
    if raw_matrix.ndim != 2 or raw_matrix.shape[0] != raw_matrix.shape[1]:
        raise ValueError(f"影响矩阵必须是方阵，当前形状为 {raw_matrix.shape}")

    # 原始矩阵：行=源节点，列=目标节点；正值表示下游影响，负值表示上游影响。
    upstream_source_target = np.maximum(-raw_matrix, 0) / 15.0
    downstream_source_target = np.maximum(raw_matrix, 0) / 15.0

    # GCN 使用目标节点为行、源节点为列的邻接矩阵。
    upstream_inf = upstream_source_target.T.copy()
    downstream_inf = downstream_source_target.T.copy()

    np.fill_diagonal(upstream_inf, 0.0)
    np.fill_diagonal(downstream_inf, 0.0)

    # 边界条件按原始矩阵中的源节点行定义：
    # Node 1 不保留不合理的逆向影响；Node 56 不保留不合理的顺向影响。
    # 转置后，原始源节点行对应 GCN 矩阵中的列。
    node1, node3 = 0, node_num - 1
    for j in range(node_num):
        if raw_matrix[node1, j] < 0 and j != node1:
            upstream_inf[j, node1] = 0.0
        if raw_matrix[node3, j] > 0 and j != node3:
            downstream_inf[j, node3] = 0.0

    return upstream_inf.astype(np.float32), downstream_inf.astype(np.float32)


def main():
    # 设备设置
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 参数设置
    config = {
        'data_dir': os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Sample', 'data'),
        'node_indices': 'range:7-62',
        'q_indices': '4,5,6',
        'time_steps': 1,
        'random_seed': 42,
        'hidden_dim': 1024,
        'batch_size': 8,
        'learning_rate': 5e-4,
        'epochs': 200,
        'save_dir': os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs'),
        'influence_matrix_file': os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Sample', 'section_distance.csv')
    }

    # 固定随机种子，确保数据划分、参数初始化和训练批次顺序可复现。
    set_random_seed(config['random_seed'])

    # 在保存归一化统计量、测试集数据和模型检查点之前创建输出目录。
    os.makedirs(config['save_dir'], exist_ok=True)

    # 加载CSV文件
    csv_files = sorted(glob.glob(os.path.join(config['data_dir'], '*.csv')))
    print(f"找到 {len(csv_files)} 个CSV文件")
    if not csv_files:
        raise FileNotFoundError(f"未在数据目录中找到CSV文件: {config['data_dir']}")
    
    # 提取列
    first_csv = csv_files[0]
    node_columns, q_columns = extract_columns_by_indices(first_csv, config['node_indices'], config['q_indices'])
    print(f"节点数量: {len(node_columns)}")
    print(f"流量列名: {q_columns}")
    
    # 加载所有条件（使用全局归一化）
    all_conditions = []
    all_csv_names = []
    
    for csv_file in csv_files:
        Z_norm, Q_norm, z_stats, q_scalers, Z_abs, Q_abs = load_water_flow_data(csv_file, node_columns, q_columns)
        condition = (Z_norm, Q_norm, z_stats, q_scalers, Z_abs, Q_abs)
        all_conditions.append(condition)
        all_csv_names.append(os.path.basename(csv_file))
    
    # 三路划分 (train: 0.8, val: 0.1, test: 0.1)
    train_conditions, test_conditions, train_names, test_names = train_test_split(
        all_conditions, all_csv_names, test_size=0.1, random_state=config['random_seed']  # 先分离出 10%（test）
    )
    
    
    # 创建全局归一化stats（用于保存）
    node_num = len(node_columns)
    z_stats = {i: {'type': 'global_minmax', 'min': WATER_GLOBAL_MIN, 'range': WATER_GLOBAL_RANGE} 
              for i in range(node_num)}
    
    q_num = len(q_columns)
    q_stats = {i: {'type': 'global_scale', 'scale': FLOW_GLOBAL_SCALE} 
              for i in range(q_num)}
    
    # 保存stats和test_conditions
    stats_save_path = os.path.join(config['save_dir'], 'normalization_stats.pkl')
    with open(stats_save_path, 'wb') as f:
        pickle.dump({'z_stats': z_stats, 'q_stats': q_stats}, f)
    print(f"归一化stats已保存: {stats_save_path}")
    
    test_save_path = os.path.join(config['save_dir'], 'test_conditions.pkl')
    test_conditions_with_names = []
    for i, condition in enumerate(test_conditions):
        test_conditions_with_names.append({
            'condition_data': condition,
            'csv_name': test_names[i],
            'condition_index': i
        })
    
    with open(test_save_path, 'wb') as f:
        pickle.dump(test_conditions_with_names, f)
    print(f"Test条件已保存: {test_save_path}")
    
     ## === 加载影响矩阵并处理为统一绝对值矩阵 ===
    if 'influence_matrix_file' in config and config['influence_matrix_file']:
        raw_matrix = load_raw_matrix(config['influence_matrix_file'], node_columns)
        
        if raw_matrix is None:
            raise ValueError(
                "物理影响矩阵加载失败或节点标签不匹配。请检查 section_distance.csv，"
                "不使用静默的单位矩阵替代真实物理先验。"
            )

        # 1. GCN 专用矩阵：从源行/目标列拆分后转置为目标行/源列。
        up_matrix, down_matrix = get_directional_physics_matrices(raw_matrix)

        # 2. 流量分布专用矩阵：保留源行/目标列，供 FlowInfluenceModule
        #    按 Q1、Q2、Q3 注入节点提取对应源节点行。
        unified_influence_matrix = np.abs(raw_matrix) / 15.0
        node1, node3 = 0, raw_matrix.shape[0] - 1
        for j in range(unified_influence_matrix.shape[1]):
            if raw_matrix[node1, j] < 0 and j != node1:
                unified_influence_matrix[node1, j] = 0.0
            if raw_matrix[node3, j] > 0 and j != node3:
                unified_influence_matrix[node3, j] = 0.0

        print("流量影响矩阵（源行/目标列）和 GCN 方向矩阵（目标行/源列）生成成功！")
    
    # 创建数据加载器
    train_loader, val_loader = create_data_loaders(
        train_conditions, 
        time_steps=config['time_steps'],
        batch_size=config['batch_size'],
        random_seed=config['random_seed']
    )
    
    print(f"数据集划分: Test={len(test_conditions)}")
    # 创建改进的递归预测模型
    model = ImprovedWaterModel(
        node_num=node_num,
        time_steps=config['time_steps'],
        hidden_dim=config['hidden_dim'],
        up_matrix=up_matrix,         
        down_matrix=down_matrix,      
        influence_matrix=unified_influence_matrix,
        num_layers=2
    ).to(device)
    
    print(f"改进模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 训练模型
    trainer = RecursiveTrainer(model, train_loader, val_loader, device, save_dir=config['save_dir'])
    trainer.setup_optimizer(learning_rate=config['learning_rate'])
    train_history = trainer.train(epochs=config['epochs'])
    
    # 创建预测器
    predictor = RecursiveInference(model, device)
    
    
    # 在测试集上快速评估
    print("\n=== 在测试工况上快速评估 ===")
    test_eval_results = predictor.evaluate_on_conditions(test_conditions, time_steps=config['time_steps'])
    

if __name__ == "__main__":
    main()