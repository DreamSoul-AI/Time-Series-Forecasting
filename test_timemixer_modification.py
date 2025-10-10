import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np


# ---------------------------------------------------------------------
# 1. 配置超参数
# ---------------------------------------------------------------------
class Config:
    def __init__(self):
        # ===== 基本参数 =====
        self.d_model               = 32     # 隐藏特征维度
        self.seq_len               = 48     # 历史序列长度  S_in
        self.pred_len              = 24     # 预测序列长度  S_out

        # ===== 多尺度下采样（此简化版只用第 0 层） =====
        self.down_sampling_window  = 2
        self.down_sampling_layers  = 1

        # ===== 通道信息 =====
        self.enc_in   = 1                  # 历史数值维
        self.dec_in   = 1                  # 未来数值维
        self.c_out    = 1
        self.features = 'M'                # 这里无实际用处，仅保持接口一致

        # ===== 其他参数 =====
        self.e_layers = 4
        self.d_ff     = 32
        self.embed    = 'timeF'
        self.freq     = 'h'
        self.dropout  = 0.1


# ---------------------------------------------------------------------
# 2. SimplifiedTimeMixer 网络
# ---------------------------------------------------------------------
class SimplifiedTimeMixer(nn.Module):
    """
    关键改动：
    predict_layer 把 D → S_in, 生成时间权重 [B, S_out, S_in]
    
    bmm 的内积维都是 S_in, 输出 [B, S_out, D]
    """
    def __init__(self, configs: Config):
        super().__init__()
        self.configs = configs

        # ---------- 多尺度预测层 ----------
        self.predict_layers = nn.ModuleList([
            nn.Linear(configs.d_model,
                      configs.seq_len // (configs.down_sampling_window ** i))
            for i in range(configs.down_sampling_layers + 1)
        ])

        # ---------- 嵌入 ----------
        self.value_embedding    = nn.Linear(configs.enc_in, configs.d_model)
        self.temporal_embedding = nn.Linear(4, configs.d_model)   # (month, day, weekday, hour)

    # -----------------------------------------------------------------
    # forward
    # -----------------------------------------------------------------
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        x_enc      : [B, S_in,  enc_in]
        x_mark_enc : [B, S_in,  4]
        x_dec      : [B, S_out, dec_in]  (本简化模型未使用，可留作占位)
        x_mark_dec : [B, S_out, 4]
        """
        # 1) 历史数值嵌入  --------------------------
        enc_out = self.value_embedding(x_enc)           # [B, S_in, D]

        # 2) 未来时间特征嵌入 ------------------------
        y_mark_embed = self.temporal_embedding(x_mark_dec)  # [B, S_out, D]

        # 3) 生成时间权重  ---------------------------
        # 这里只用第 0 层（最高分辨率）：D → S_in
        query = self.predict_layers[0](y_mark_embed)    # [B, S_out, S_in]

        # 4) 与历史特征做批量矩阵乘法 ---------------
        key = enc_out                                   # [B, S_in, D]   ★ 不再 transpose
        dec_out = torch.bmm(query, key)                 # [B, S_out, D]

        return dec_out


# ---------------------------------------------------------------------
# 3. 简单单元测试
# ---------------------------------------------------------------------
def test_model():
    print("Running unit‑test ...")

    cfg   = Config()
    model = SimplifiedTimeMixer(cfg)

    B = 2
    x_enc      = torch.randn(B, cfg.seq_len,  cfg.enc_in)             # [2, 48, 1]
    x_dec      = torch.randn(B, cfg.pred_len, cfg.dec_in)             # [2, 24, 1]
    x_mark_enc = torch.randint(0, 12, (B, cfg.seq_len, 4)).float()    # [2, 48, 4]
    x_mark_dec = torch.randint(0, 12, (B, cfg.pred_len, 4)).float()   # [2, 24, 4]

    out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    expected = (B, cfg.pred_len, cfg.d_model)

    print(f"Output shape : {tuple(out.shape)}")
    assert out.shape == expected, \
        f"Expect {expected}, got {tuple(out.shape)}"

    print("✓ test passed.")


# ---------------------------------------------------------------------
# 4. CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    test_model()
