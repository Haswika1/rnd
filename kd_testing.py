#!/usr/bin/env python3
"""
Student inference — per-freq rollout with absolute H prediction.
- B: full 1000 pts available for all freqs
- H: only first 100 pts real, remaining 900 are NaN
- Freqs 1,2,3 (window > 100): left-pad real H pts with zeros to reach window
- Freqs 4,5,6,7 (window <= 100): use first `window` real H pts directly as seed
- All freqs: full rollout across 1000 pts using freq-specific window+ksteps
- Merges all freq predictions back into one output CSV in original row order
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

# ================= CONFIG =================
PRED_DIR   = "/content/drive/MyDrive/onefreq1/Material_A1"
DATA_DIR   = "/content/drive/MyDrive/onefreq1/A_sequences_flat_debug"
MODEL_PATH = f"{DATA_DIR}/student_kd_maxk320.pth"
NORM_PATH  = f"{DATA_DIR}/norm_stats.npz"

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_DIM   = 8
D_MODEL     = 128
N_LAYERS    = 3
N_HEADS     = 4
MAX_KSTEPS  = 320

FREQS           = [1, 2, 3, 4, 5, 6, 7]
WINDOW_PER_FREQ = {1: 320, 2: 200, 3: 130, 4: 80, 5: 50, 6: 32, 7: 20}
KSTEPS_PER_FREQ = {1: 320, 2: 200, 3: 130, 4: 80, 5: 50, 6: 32, 7: 20}

TOTAL_PTS  = 1000   # total B pts per row
REAL_H_PTS = 100    # only first 100 H pts are real, rest are NaN

# ================= LOAD NORMALIZATION =================
stats    = np.load(NORM_PATH)
B_mean   = stats["B_mean"];   B_std   = stats["B_std"]
H_mean   = stats["H_mean"];   H_std   = stats["H_std"]
dB_mean  = stats["dB_mean"];  dB_std  = stats["dB_std"]
d2B_mean = stats["d2B_mean"]; d2B_std = stats["d2B_std"]
dH_mean  = stats["dH_mean"];  dH_std  = stats["dH_std"]
d2H_mean = stats["d2H_mean"]; d2H_std = stats["d2H_std"]
print("✅ Normalization stats loaded")

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
        N      = 2 * T
        x_v    = self.W_V(x).reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)
        x_g    = self.W_G1(x).transpose(1, 2)
        x_g    = self.W_G2(self.silu(x_g)).transpose(1, 2)
        x_g    = x_g.reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)
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
    def __init__(self, input_dim, d_model, n_layers, n_heads, max_ksteps):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.blocks     = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f       = nn.LayerNorm(d_model)
        self.head       = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.SiLU(),
            nn.Linear(128, max_ksteps)
        )

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        x = x[:, -1, :]
        return self.head(x)   # (B, MAX_KSTEPS)

# ================= LOAD MODEL =================
model = BHTransFourier(INPUT_DIM, D_MODEL, N_LAYERS, N_HEADS, MAX_KSTEPS).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("✅ Student model loaded")

# ================= FEATURE BUILDER =================
def build_features(B_seq_n, H_seq_n, B_future_n, window, ksteps):
    """
    B_seq_n    : (window,)  normalized
    H_seq_n    : (window,)  normalized
    B_future_n : (ksteps,)  normalized
    Returns    : (window, 8)
    """
    # Concatenate history + future B
    B_combined   = np.concatenate([B_seq_n, B_future_n])   # (window+ksteps,)
    dB_combined  = np.diff(B_combined)                      # (window+ksteps-1,)
    d2B_combined = np.diff(dB_combined)                     # (window+ksteps-2,)

    # Extract dB/d2B at the boundary (future side), length=ksteps
    dB_raw  = dB_combined[window - 1 : window - 1 + ksteps]   # (ksteps,)
    d2B_raw = d2B_combined[window - 2 : window - 2 + ksteps]  # (ksteps,)

    # Pad or trim to window length
    if ksteps < window:
        dB_feat  = np.pad(dB_raw,  (0, window - ksteps))   # (window,)
        d2B_feat = np.pad(d2B_raw, (0, window - ksteps))   # (window,)
    else:
        # ksteps >= window: trim to window
        dB_feat  = dB_raw[:window]    # (window,)
        d2B_feat = d2B_raw[:window]   # (window,)

    # Backward dH, d2H from history
    dH_feat  = np.diff(H_seq_n, prepend=H_seq_n[:1])          # (window,)
    d2H_feat = np.diff(H_seq_n, n=2, prepend=H_seq_n[:2])     # (window,)

    # Normalize derivatives
    dB_feat  = (dB_feat  - dB_mean)  / dB_std
    d2B_feat = (d2B_feat - d2B_mean) / d2B_std
    dH_feat  = (dH_feat  - dH_mean)  / dH_std
    d2H_feat = (d2H_feat - d2H_mean) / d2H_std

    # Future B channel: place B_future_n in first ksteps slots, rest zeros
    T_seq   = np.ones(window,  dtype=np.float32)
    Bf_full = np.zeros(window, dtype=np.float32)
    Bf_full[:ksteps] = B_future_n   # ksteps <= window always ✅

    # Stack all 8 features: (window, 8)
    X = np.stack([B_seq_n, H_seq_n, dB_feat, d2B_feat,
                  dH_feat, d2H_feat, T_seq, Bf_full], axis=1)
    return X.astype(np.float32)

# ================= PER-FREQ INFERENCE =================
all_results = {}   # original_row_idx -> H_pred (TOTAL_PTS,)

for freq in FREQS:
    window = WINDOW_PER_FREQ[freq]
    ksteps = KSTEPS_PER_FREQ[freq]

    print(f"\n{'='*55}")
    print(f"Freq {freq} | window={window} | ksteps={ksteps}")
    print(f"{'='*55}")

    # Load freq-specific test files
    B_test  = pd.read_csv(os.path.join(PRED_DIR, f"B_{freq}.csv"),
                          header=None).values.astype(np.float32)   # (N, TOTAL_PTS)
    H_test  = pd.read_csv(os.path.join(PRED_DIR, f"H_{freq}.csv"),
                          header=None).values.astype(np.float32)   # (N, TOTAL_PTS) — NaN after col 100
    row_idx = pd.read_csv(os.path.join(PRED_DIR, f"indices_{freq}.csv"),
                          header=None).values.squeeze().astype(int)

    NUM_ROWS = B_test.shape[0]
    assert B_test.shape[1] == TOTAL_PTS, \
        f"Freq {freq}: expected {TOTAL_PTS} B cols, got {B_test.shape[1]}"
    assert window <= TOTAL_PTS, \
        f"Freq {freq}: window={window} > TOTAL_PTS={TOTAL_PTS}"
    assert ksteps <= window, \
        f"Freq {freq}: ksteps={ksteps} > window={window}"

    print(f"  Rows: {NUM_ROWS} | B cols: {TOTAL_PTS} | Real H pts: {REAL_H_PTS}")
    print(f"  Seed padding: {max(0, window - REAL_H_PTS)} zeros left-padded")

    H_pred_all = np.zeros((NUM_ROWS, TOTAL_PTS), dtype=np.float32)

    with torch.no_grad():
        for r in tqdm(range(NUM_ROWS), desc=f"Freq {freq}"):

            B_row = B_test[r]                        # (TOTAL_PTS,) full B, no NaN
            H_real = H_test[r, :REAL_H_PTS]         # (100,) real H pts only

            # ===== SEED H_pred with real H pts =====
            if window <= REAL_H_PTS:
                # window fits within real H pts — no padding needed
                # e.g. freq4: window=80, use H_real[:80]
                H_pred_all[r, :window] = H_real[:window]
            else:
                # window > real H pts — left-pad with zeros
                # e.g. freq1: window=320, pad 220 zeros + 100 real pts
                pad_len = window - REAL_H_PTS
                H_pred_all[r, :window] = np.pad(H_real, (pad_len, 0))

            # ===== BLOCK ROLLOUT =====
            for start in range(window, TOTAL_PTS, ksteps):
                hist_start = start - window
                fut_end    = min(start + ksteps, TOTAL_PTS)
                actual_k   = fut_end - start   # may be < ksteps at last block

                # History: always exactly (window,) — from H_pred_all
                B_seq = B_row[hist_start : start]            # (window,)
                H_seq = H_pred_all[r, hist_start : start]   # (window,)

                # Future B: (actual_k,) → pad to ksteps if last block
                B_future = B_row[start : fut_end]            # (actual_k,)
                if actual_k < ksteps:
                    B_future = np.pad(B_future, (0, ksteps - actual_k))
                # B_future is now always (ksteps,)

                # Normalize
                B_seq_n    = (B_seq    - B_mean) / B_std   # (window,)
                H_seq_n    = (H_seq    - H_mean) / H_std   # (window,)
                B_future_n = (B_future - B_mean) / B_std   # (ksteps,)

                # Build features and predict
                X   = build_features(B_seq_n, H_seq_n, B_future_n, window, ksteps)
                # X: (window, 8)
                X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                # X_t: (1, window, 8)

                pred_norm = model(X_t).cpu().numpy().squeeze()   # (MAX_KSTEPS,)
                pred_norm = pred_norm[:ksteps]                   # (ksteps,) slice to freq ksteps
                H_block   = pred_norm * H_std + H_mean           # denormalize → (ksteps,)

                # Write only actual_k steps (handles last block correctly)
                H_pred_all[r, start : start + actual_k] = H_block[:actual_k]

    # Store with original row indices
    for i, idx in enumerate(row_idx):
        all_results[idx] = H_pred_all[i]

    print(f"  ✅ Freq {freq} done — {NUM_ROWS} rows predicted")

# ================= MERGE AND SAVE =================
print("\nMerging predictions in original row order...")
total_rows = max(all_results.keys()) + 1
H_final    = np.zeros((total_rows, TOTAL_PTS), dtype=np.float32)

for idx, pred in all_results.items():
    H_final[idx] = pred

out_csv = os.path.join(PRED_DIR, "A.csv")
pd.DataFrame(H_final).to_csv(out_csv, index=False, header=False)

print(f"✅ Inference complete")
print(f"✅ Output shape : {H_final.shape}")
print(f"✅ CSV saved at : {out_csv}")