"""
递归预测模型训练模块。

该模块负责纯自回归模式下的模型训练、验证、学习率调度、
训练过程记录以及模型检查点保存与加载。
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
from torch.optim.lr_scheduler import LinearLR
from recursive_model import ImprovedWaterModel, ImprovedPhysicsAwareLoss
from data_utils import compute_normalized_deltas_from_abs, WATER_GLOBAL_RANGE

class RecursiveTrainer:
    """基于全局归一化的递归预测模型训练器。"""

    def __init__(self, model, train_loader, val_loader, device, save_dir='STGCN/best/models'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir

        os.makedirs(save_dir, exist_ok=True)

        # 1. 初始化训练记录
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }

        # 2. 初始化学习率预热状态
        self.warmup_complete = False
        self.warmup_steps = 0

    def setup_optimizer(self, learning_rate=1e-4, weight_decay=1e-4):
        """配置 AdamW 优化器、学习率调度器和训练损失函数。"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.warmup_scheduler = LinearLR(
            self.optimizer, start_factor=0.1, total_iters=5 * len(self.train_loader)
        )

        self.reduce_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=5,
            factor=0.5,
            min_lr=1e-6
        )

        self.total_warmup_steps = 5 * len(self.train_loader)
        self.warmup_complete = False
        self.criterion = ImprovedPhysicsAwareLoss()

    def compute_normalized_deltas(self, target_water_abs, initial_water_abs):
        """将绝对水位序列转换为全局归一化的逐步水位变化量。"""
        # 1. 获取批次、预测长度和节点数量
        batch_size = target_water_abs.shape[0]
        pred_len = target_water_abs.shape[1]
        node_num = target_water_abs.shape[2]

        deltas_norm = torch.zeros((batch_size, pred_len, node_num),
                                 dtype=torch.float32, device=self.device)

        # 2. 逐样本计算归一化水位变化量
        for b in range(batch_size):
            target_b = target_water_abs[b].cpu().numpy()
            initial_b = initial_water_abs[b].cpu().numpy()
            delta_b = compute_normalized_deltas_from_abs(target_b, initial_b)
            deltas_norm[b] = torch.from_numpy(delta_b).to(self.device)

        return deltas_norm

    def train_epoch(self, epoch):
        """执行一个训练轮次，并按有效预测时间步累计损失。"""
        self.model.train()
        total_timestep_loss = 0
        total_timesteps = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Training Epoch {epoch}",ncols=150)
        for batch_idx, (full_water_norm, full_water_abs, full_flow_norm, mask, seq_lens) in enumerate(pbar):
            batch_size, max_len, node_num = full_water_norm.shape

            full_water_norm = full_water_norm.to(self.device)
            full_water_abs = full_water_abs.to(self.device)
            full_flow_norm = full_flow_norm.to(self.device)
            mask = mask.to(self.device)

            batch_timestep_loss = 0
            batch_total_timesteps = 0

            # 1. 按工况逐条执行递归预测
            for i in range(batch_size):
                seq_len = seq_lens[i]
                if seq_len <= self.model.time_steps:
                    continue

                water_seq_norm = full_water_norm[i:i+1, :seq_len]
                water_seq_abs = full_water_abs[i:i+1, :seq_len]
                flow_seq_norm = full_flow_norm[i:i+1, :seq_len]

                initial_water_norm = water_seq_norm[:, :self.model.time_steps, :]
                initial_water_abs = water_seq_abs[:, :self.model.time_steps, :]
                initial_flow_norm = flow_seq_norm[:, :self.model.time_steps, :]

                pred_len = seq_len - self.model.time_steps
                target_water_abs = water_seq_abs[:, self.model.time_steps:, :]
                target_flow_norm = flow_seq_norm[:, self.model.time_steps:, :]

                target_deltas_norm = self.compute_normalized_deltas(target_water_abs, initial_water_abs)

                self.model.reset_state()

                try:
                    # 执行纯自回归前向传播，不使用 Teacher Forcing
                    predicted_deltas_norm = self.model(
                        initial_water=initial_water_norm,
                        initial_flow=initial_flow_norm,
                        full_flow_sequence=target_flow_norm
                    )

                    total_loss, loss_details = self.criterion(
                        predicted_deltas_norm,
                        target_deltas_norm,
                        initial_water_abs[:, -1, :]
                    )

                    batch_timestep_loss += total_loss * pred_len
                    batch_total_timesteps += pred_len

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"序列 {i} OOM，跳过")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
                except Exception as e:
                    print(f"训练序列 {i} 时出错: {e}")
                    continue

            if batch_total_timesteps > 0:
                avg_batch_loss = batch_timestep_loss / batch_total_timesteps

                # 2. 清除上一批次梯度并执行反向传播
                self.optimizer.zero_grad(set_to_none=True)
                avg_batch_loss.backward()

                # 3. 梯度裁剪并更新模型参数
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                total_timestep_loss += batch_timestep_loss.item()
                total_timesteps += batch_total_timesteps
                num_batches += 1

                pbar.set_postfix({
                    'Loss': f'{avg_batch_loss.item():.4f}',
                    'Seqs': f'{batch_size}'
                })

            if not self.warmup_complete:
                self.warmup_scheduler.step()
                self.warmup_steps += 1
                if self.warmup_steps >= self.total_warmup_steps:
                    self.warmup_complete = True
                    print(f"Epoch {epoch}: Warmup 阶段完成，切换到 ReduceLROnPlateau")

        return total_timestep_loss / total_timesteps if total_timesteps > 0 else 0

    def validate_epoch(self):
        """在验证集上执行纯自回归预测并计算平均损失。"""
        self.model.eval()
        total_timestep_loss = 0
        total_timesteps = 0
        num_batches = 0

        with torch.no_grad():
            for full_water_norm, full_water_abs, full_flow_norm, mask, seq_lens in self.val_loader:
                batch_size, max_len, node_num = full_water_norm.shape

                full_water_norm = full_water_norm.to(self.device)
                full_water_abs = full_water_abs.to(self.device)
                full_flow_norm = full_flow_norm.to(self.device)

                batch_timestep_loss = 0
                batch_total_timesteps = 0

                # 1. 按工况逐条执行递归验证
                for i in range(batch_size):
                    seq_len = seq_lens[i]
                    if seq_len <= self.model.time_steps:
                        continue

                    water_seq_norm = full_water_norm[i:i+1, :seq_len]
                    water_seq_abs = full_water_abs[i:i+1, :seq_len]
                    flow_seq_norm = full_flow_norm[i:i+1, :seq_len]

                    initial_water_norm = water_seq_norm[:, :self.model.time_steps, :]
                    initial_water_abs = water_seq_abs[:, :self.model.time_steps, :]
                    initial_flow_norm = flow_seq_norm[:, :self.model.time_steps, :]
                    target_water_abs = water_seq_abs[:, self.model.time_steps:, :]
                    target_flow_norm = flow_seq_norm[:, self.model.time_steps:, :]

                    target_deltas_norm = self.compute_normalized_deltas(target_water_abs, initial_water_abs)

                    self.model.reset_state()

                    try:
                        # 执行纯自回归验证，不使用 Teacher Forcing
                        predicted_deltas_norm = self.model(
                            initial_water=initial_water_norm,
                            initial_flow=initial_flow_norm,
                            full_flow_sequence=target_flow_norm
                        )

                        total_loss, _ = self.criterion(
                            predicted_deltas_norm,
                            target_deltas_norm,
                            initial_water_abs[:, -1, :]
                        )

                        batch_timestep_loss += total_loss.item() * (seq_len - self.model.time_steps)
                        batch_total_timesteps += (seq_len - self.model.time_steps)
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            print(f"验证序列 {i} OOM，跳过")
                            torch.cuda.empty_cache()
                            continue
                        else:
                            raise e
                    except Exception as e:
                        print(f"验证序列 {i} 时出错: {e}")
                        continue

                if batch_total_timesteps > 0:
                    total_timestep_loss += batch_timestep_loss
                    total_timesteps += batch_total_timesteps
                    num_batches += 1

        return total_timestep_loss / total_timesteps if total_timesteps > 0 else 0

    def train(self, epochs, save_interval=10):
        """执行完整训练流程，并保存最佳、阶段性及最终模型。"""
        print(f"开始递归训练 (纯自回归模式)，共 {epochs} 个epochs")

        # 1. 初始化最佳验证损失
        best_val_loss = float('inf')

        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")

            train_loss = self.train_epoch(epoch)
            val_loss = self.validate_epoch()

            current_lr = self.optimizer.param_groups[0]['lr']
            if self.warmup_complete:
                self.reduce_scheduler.step(val_loss)
                new_lr = self.optimizer.param_groups[0]['lr']
                if new_lr != current_lr:
                    print(f"LR 衰减: {current_lr:.2e} -> {new_lr:.2e}")

            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])

            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {self.optimizer.param_groups[0]['lr']:.2e}")

            # 2. 根据验证损失保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model('best_recursive_model.pth')
                print(f"保存最佳模型，验证损失: {val_loss:.6f}")

            if epoch % save_interval == 0:
                self.save_model(f'recursive_checkpoint_epoch_{epoch}.pth')

        self.save_model('final_recursive_model.pth')
        print("递归训练完成!")

        return self.train_history

    def save_model(self, filename):
        """保存模型参数、优化器状态和训练历史。"""
        save_path = os.path.join(self.save_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_history': self.train_history,
        }, save_path)

    def load_model(self, filename):
        """从检查点恢复模型参数及可用的训练状态。"""
        load_path = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'train_history' in checkpoint:
            self.train_history = checkpoint['train_history']

        print(f"加载模型: {filename}")
