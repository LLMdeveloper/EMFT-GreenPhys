import torch
import torch.nn as nn
import numpy as np

ENERGY_SCALE = 1000.0
LOG_DENOM = np.log1p(ENERGY_SCALE)

def transform_energy(E_raw):
    """对数变换：缓解能量标签的长尾分布"""
    return np.log1p(np.asarray(E_raw) * ENERGY_SCALE) / LOG_DENOM

class EnergyHead(nn.Module):
    """附加在隐状态上的轻量级能量投影头"""
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states):
        return self.linear(hidden_states).squeeze(-1)

def compute_emft_losses(loss_lm, energy_preds, energy_labels, labels, beta, gamma):
    """
    计算三项损失并返回总损失
    """
    device = energy_preds.device
    mask = (labels != -100).float().to(device)
    
    # 掩码序列池化（求 response 部分的均值）
    seq_len = mask.sum(dim=1, keepdim=True).clamp(min=1)
    seq_energy_pred = (energy_preds * mask).sum(dim=1) / seq_len.squeeze(1)
    seq_energy_label = (energy_labels * mask).sum(dim=1) / seq_len.squeeze(1)

    # 能量回归损失 (MSE)
    loss_energy = torch.nn.functional.mse_loss(seq_energy_pred, seq_energy_label)
    # 路径积分正则项 (Sequence Average Squared)
    loss_path = (seq_energy_pred ** 2).mean()

    # 复合总损失
    total_loss = loss_lm + beta * loss_energy + gamma * loss_path
    
    return total_loss, loss_energy, loss_path, seq_energy_pred, seq_energy_label