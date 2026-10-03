#!/usr/bin/env python3
"""
Block-wise rollout using absolute H prediction (no delta accumulation)
Each block predicts H directly — no error accumulation between blocks
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

# ================= CONFIG =================
BASE_DIR   = "/content/drive/MyDrive/onefreq1/Material_A1"
MODEL_PATH = "/content/drive/MyDrive/onefreq1/A_sequences_flat_debug/A_best_bh_transfourier.pth"
NORM_PATH  = "/content/drive/MyDrive/onefreq1/A_sequences_flat_debug/norm_stats.npz"

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN       = 100
INPUT_DIM     = 8
D_MODEL       = 128
N_LAYERS      = 3
N_HEADS       = 4
K_STEPS       = 100
FUTURE_POINTS = 900

# ================= LOAD NORMALIZATION =================
stats    = np.load(NORM_PATH)
B_mean   = stats["B_mean"];   B_std   = stats["B_std"]
H_mean   = stats["H_mean"];   H_std   = stats["H_std"]
T_mean   = stats["T_mean"];   T_std   = stats["T_std"]
dB_mean  = stats["dB_mean"];  dB_std  = stats["dB_std"]
d2B_mean = stats["d2B_mean"]; d2B_std = stats["d2B_std"]
dH_mean  = stats["dH_mean"];  dH_std  = stats["dH_std"]
d2H_mean = stats["d2H_mean"]; d2H_std = stats["d2H_std"]

print("✅ Normalization stats loaded")
print(f"  B   → mean: {B_mean:.4f},  std: {B_std:.4f}")
print(f"  H   → mean: {H_mean:.4f},  std: {H_std:.4f}")
print(f"  dB  → mean: {dB_mean:.6f}, std: {dB_std:.6f}")
print(f"  d2B → mean: {d2B_mean:.6f},std: {d2B_std:.6f}")
print(f"  dH  → mean: {dH_mean:.6f}, std: {dH_std:.6f}")
print(f"  d2H → mean: {d2H_mean:.6f},std: {d2H_std:.6f}")

# ================= MODEL =================
class MultiHeadFourier(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
        self.pre_conv = nn.Conv1d(d_model, d_model, 3, groups=d_model, bias=False)
        self.silu     = nn.SiLU()
        self.W_V      = nn.Linear(d_model, d_model)
        self.W_G1     = nn.Linear(d_model, d_model)
        self.W_G2     = nn.Conv1d(d_model, d_model, 1, groups=n_heads)
        self.linear   = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape
        x = x.permute(0, 2, 1)
        x = nn.functional.pad(x, (2, 0))
        x = self.pre_conv(x).permute(0, 2, 1)
        N   = 2 * T
        x_v = self.W_V(x).reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)
        x_g = self.W_G1(x).transpose(1, 2)
        x_g = self.W_G2(self.silu(x_g)).transpose(1, 2)
        x_g = x_g.reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)
        G_fft  = torch.fft.rfft(x_g.float(), n=N, dim=2)
        V_fft  = torch.fft.rfft(x_v.float(), n=N, dim=2)
        x_fft  = torch.fft.irfft(G_fft * V_fft, n=N, dim=2)
        x_fft  = x_fft[:, :, :T, :]
        x_fft  = x_fft.transpose(1, 2).contiguous().reshape(B, T, D)
        return self.linear(x_fft)

class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.fc2 = nn.Linear(4 * d_model, d_model)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class Block(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1     = nn.LayerNorm(d_model)
        self.fourier = MultiHeadFourier(d_model, n_heads)
        self.ln2     = nn.LayerNorm(d_model)
        self.mlp     = MLP(d_model)

    def forward(self, x):
        x = x + self.fourier(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class BHTransFourier(nn.Module):
    def __init__(self, input_dim, d_model, n_layers, n_heads, ksteps):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.blocks     = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f       = nn.LayerNorm(d_model)
        self.head       = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.SiLU(),
            nn.Linear(128, ksteps)
        )

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        x = x[:, -1, :]
        return self.head(x)

# ================= LOAD MODEL =================
model = BHTransFourier(INPUT_DIM, D_MODEL, N_LAYERS, N_HEADS, K_STEPS).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("✅ Model loaded")

# ================= LOAD DATA =================
B_all  = pd.read_csv(os.path.join(BASE_DIR, "A_True_B_seq.csv"),   header=None).values
H_true = pd.read_csv(os.path.join(BASE_DIR, "A_Padded_H_seq.csv"), header=None).values
T_now  = pd.read_csv(os.path.join(BASE_DIR, "A_True_T.csv"),       header=None).values.squeeze()

NUM_ROWS, TOTAL_B = B_all.shape
print(f"B_all shape:  {B_all.shape}")
print(f"H_true shape: {H_true.shape}")
print(f"Running on all {NUM_ROWS} rows")

# ================= BLOCK ROLLOUT =================
H_pred_all = np.zeros((NUM_ROWS, SEQ_LEN + FUTURE_POINTS))

with torch.no_grad():

    for r in tqdm(range(NUM_ROWS), desc="Block rollout"):

        H_pred_all[r, :SEQ_LEN] = H_true[r, :SEQ_LEN]

        for start in range(SEQ_LEN, SEQ_LEN + FUTURE_POINTS, K_STEPS):

            hist_start = start - SEQ_LEN
            fut_end    = min(start + K_STEPS, SEQ_LEN + FUTURE_POINTS)

            B_seq    = B_all[r, hist_start:start]
            H_seq    = H_pred_all[r, hist_start:start]
            B_future = B_all[r, start:fut_end]

            # ===== NORMALIZE =====
            B_seq_n    = (B_seq    - B_mean) / B_std
            H_seq_n    = (H_seq    - H_mean) / H_std
            B_future_n = (B_future - B_mean) / B_std
            T_val_n    = 1.0

            # ===== FORWARD dB, d2B =====
            B_combined   = np.concatenate([B_seq_n, B_future_n])
            dB_combined  = np.diff(B_combined)
            d2B_combined = np.diff(dB_combined)

            dB_feat  = dB_combined[99:99 + K_STEPS]
            d2B_feat = d2B_combined[98:98 + K_STEPS]
            dB_feat  = np.pad(dB_feat,  (0, SEQ_LEN - K_STEPS))
            d2B_feat = np.pad(d2B_feat, (0, SEQ_LEN - K_STEPS))

            # ===== BACKWARD dH, d2H =====
            dH_feat  = np.diff(H_seq_n)
            dH_feat  = np.pad(dH_feat,  (1, 0), 'edge')
            d2H_feat = np.diff(H_seq_n, n=2)
            d2H_feat = np.pad(d2H_feat, (2, 0), 'edge')

            # ===== NORMALIZE DERIVATIVES =====
            dB_feat  = (dB_feat  - dB_mean)  / dB_std
            d2B_feat = (d2B_feat - d2B_mean) / d2B_std
            dH_feat  = (dH_feat  - dH_mean)  / dH_std
            d2H_feat = (d2H_feat - d2H_mean) / d2H_std

            # ===== FUTURE B — K_STEPS first then zeros =====
            T_seq   = np.full(SEQ_LEN, T_val_n)
            Bf_full = np.zeros(SEQ_LEN)
            Bf_full[:K_STEPS] = B_future_n[:K_STEPS]

            # ===== STACK FEATURES =====
            X = np.stack([B_seq_n, H_seq_n, dB_feat, d2B_feat,
                          dH_feat, d2H_feat, T_seq, Bf_full], axis=1)
            X = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            # ===== MODEL PREDICTION =====
            H_pred_norm = model(X).cpu().numpy().squeeze()

            # ===== DENORMALIZE — direct absolute H, no accumulation =====
            H_block = np.atleast_1d(H_pred_norm * H_std + H_mean)

            H_pred_all[r, start:start + K_STEPS] = H_block[:K_STEPS]

# ================= SAVE CSV =================
out_csv = os.path.join(BASE_DIR, "A.csv")
pd.DataFrame(H_pred_all).to_csv(out_csv, index=False, header=False)

print("✅ Block prediction complete")
print(f"✅ Output shape: {H_pred_all.shape}")
print(f"✅ CSV saved at: {out_csv}")