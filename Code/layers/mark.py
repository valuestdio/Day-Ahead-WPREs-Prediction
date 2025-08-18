import pandas as pd
import numpy as np
import torch

def generate_x_mark(x_enc, start_time, freq):
    """
    根据 x_enc 的时间维度和起始时间生成 x_mark_enc,用于 DataEmbedding_inverted。
    
    Args:
        x_enc: Tensor [B, L, N]，用于参考 batch_size 和 seq_len
        start_time: str,如 '2023-01-01 00:00:00'
        freq: str,时间频率,例如 'h' 表示每小时，'t' 表示每分钟
    
    Returns:
        x_mark_enc: Tensor [B, L, T]
    """
    B, L, _ = x_enc.shape

    # 生成时间序列
    times = pd.date_range(start=start_time, periods=L, freq=freq)

    # 提取时间特征（可选扩展）
    time_features = np.stack([
        times.hour / 23.0,         # hour of day, normalized
        times.dayofweek / 6.0,     # day of week, normalized
        times.day / 31.0,          # day of month
        times.month / 12.0,        # month
        np.sin(2 * np.pi * times.hour / 24.0),  # cyclic hour (sin)
        np.cos(2 * np.pi * times.hour / 24.0),  # cyclic hour (cos)
    ], axis=1)  # shape: [L, T]

    # 扩展到 batch 维度
    x_mark_enc = np.tile(time_features[None, :, :], (B, 1, 1))  # [B, L, T]
    x_mark_enc = torch.tensor(x_mark_enc, dtype=torch.float32, device=x_enc.device)

    return x_mark_enc

