#!/usr/bin/env python3
"""
Teacher training — absolute H prediction
Trains all 7 teachers with per-freq window sizes and ksteps
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ================= CONFIG =================
DATA_DIR       = "/content/drive/MyDrive/onefreq1/A_sequences_flat_debug"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE     = 256
EPOCHS         = 70
PATIENCE_LIMIT = 15
LR             = 0.001
FREQS          = [1, 2, 3, 4, 5, 6, 7]

WINDOW_PER_FREQ = {1: 320, 2: 200, 3: 130, 4: 80, 5: 50, 6: 32, 7: 20}
KSTEPS_PER_FREQ = {1: 320, 2: 200, 3: 130, 4: 80, 5: 50, 6: 32, 7: 20}

# ================= NORM STATS =================
# Compute norm stats from freq1 (largest window) as reference
print("Computing normalization statistics from freq1 data...")

B_all, H_all = [], []
for freq in FREQS:
    B_all.append(np.load(f"{DATA_DIR}/freq{freq}_seq_B.npy"))
    H_all.append(np.load(f"{DATA_DIR}/freq{freq}_seq_H.npy"))

B_concat = np.concatenate([b.ravel() for b in B_all])
H_concat = np.concatenate([h.ravel() for h in H_all])

B_mean, B_std = B_concat.mean(), B_concat.std()
H_mean, H_std = H_concat.mean(), H_concat.std()
T_mean, T_std = 1.0, 1.0

# Compute derivative stats from freq1 (largest window=320, ksteps=100)
B_seq1    = np.load(f"{DATA_DIR}/freq1_seq_B.npy")
H_seq1    = np.load(f"{DATA_DIR}/freq1_seq_H.npy")
B_fut1    = np.load(f"{DATA_DIR}/freq1_future_B.npy")

ksteps1   = KSTEPS_PER_FREQ[1]  # 100
window1   = WINDOW_PER_FREQ[1]  # 320

B_seq1_n  = (B_seq1 - B_mean) / B_std
H_seq1_n  = (H_seq1 - H_mean) / H_std
B_fut1_n  = (B_fut1 - B_mean) / B_std

B_comb1   = np.concatenate([B_seq1_n, B_fut1_n], axis=1)   # (N, 320+100)
dB_comb1  = np.diff(B_comb1,  axis=1)                       # (N, 419)
d2B_comb1 = np.diff(dB_comb1, axis=1)                       # (N, 418)

# dB at future boundary: indices [window1-1 : window1-1+ksteps1]
dB_ref    = dB_comb1[:,  window1 - 1 : window1 - 1 + ksteps1]   # (N, 100)
d2B_ref   = d2B_comb1[:, window1 - 2 : window1 - 2 + ksteps1]   # (N, 100)

dH_ref    = np.diff(H_seq1_n, axis=1, prepend=H_seq1_n[:, :1])
d2H_ref   = np.diff(H_seq1_n, n=2,   axis=1, prepend=H_seq1_n[:, :2])

dB_mean,  dB_std  = dB_ref.mean(),  dB_ref.std()
d2B_mean, d2B_std = d2B_ref.mean(), d2B_ref.std()
dH_mean,  dH_std  = dH_ref.mean(),  dH_ref.std()
d2H_mean, d2H_std = d2H_ref.mean(), d2H_ref.std()

print(f"B   mean/std: {B_mean:.4f} / {B_std:.4f}")
print(f"H   mean/std: {H_mean:.4f} / {H_std:.4f}")
print(f"dB  mean/std: {dB_mean:.6f} / {dB_std:.6f}")
print(f"d2B mean/std: {d2B_mean:.6f} / {d2B_std:.6f}")
print(f"dH  mean/std: {dH_mean:.6f} / {dH_std:.6f}")
print(f"d2H mean/std: {d2H_mean:.6f} / {d2H_std:.6f}")

np.savez(
    f"{DATA_DIR}/norm_stats.npz",
    B_mean=B_mean,     B_std=B_std,
    H_mean=H_mean,     H_std=H_std,
    T_mean=T_mean,     T_std=T_std,
    dB_mean=dB_mean,   dB_std=dB_std,
    d2B_mean=d2B_mean, d2B_std=d2B_std,
    dH_mean=dH_mean,   dH_std=dH_std,
    d2H_mean=d2H_mean, d2H_std=d2H_std,
)
print(f"✅ Norm stats saved\n")


# ================= DATASET =================
class BHDataset(Dataset):

    def __init__(self, data_dir, freq):
        window = WINDOW_PER_FREQ[freq]
        ksteps = KSTEPS_PER_FREQ[freq]

        B_seq    = np.load(f"{data_dir}/freq{freq}_seq_B.npy")     # (N, window)
        H_seq    = np.load(f"{data_dir}/freq{freq}_seq_H.npy")     # (N, window)
        B_future = np.load(f"{data_dir}/freq{freq}_future_B.npy")  # (N, ksteps)
        T_now    = np.load(f"{data_dir}/freq{freq}_T_now.npy")     # (N,)
        H_target = np.load(f"{data_dir}/freq{freq}_target_H.npy")  # (N, ksteps)

        # -------- NORMALIZATION --------
        B_seq    = (B_seq    - B_mean) / B_std
        H_seq    = (H_seq    - H_mean) / H_std
        B_future = (B_future - B_mean) / B_std
        T_now    = np.ones_like(T_now)
        H_target = (H_target - H_mean) / H_std

        # -------- FORWARD dB, d2B --------
        # Concatenate history + future: (N, window + ksteps)
        B_combined   = np.concatenate([B_seq, B_future], axis=1)  # (N, window+ksteps)
        dB_combined  = np.diff(B_combined,  axis=1)               # (N, window+ksteps-1)
        d2B_combined = np.diff(dB_combined, axis=1)               # (N, window+ksteps-2)

        # Extract future dB/d2B at boundary: length = ksteps, then pad/trim to window
        dB_raw  = dB_combined[:,  window - 1 : window - 1 + ksteps]  # (N, ksteps)
        d2B_raw = d2B_combined[:, window - 2 : window - 2 + ksteps]  # (N, ksteps)

        # Pad or trim to match window length
        if ksteps < window:
            dB  = np.pad(dB_raw,  ((0, 0), (0, window - ksteps)))
            d2B = np.pad(d2B_raw, ((0, 0), (0, window - ksteps)))
        else:
            dB  = dB_raw[:,  :window]
            d2B = d2B_raw[:, :window]

        # -------- BACKWARD dH, d2H --------
        dH  = np.diff(H_seq, axis=1, prepend=H_seq[:, :1])
        d2H = np.diff(H_seq, n=2,   axis=1, prepend=H_seq[:, :2])

        # -------- NORMALIZE DERIVATIVES --------
        dB  = (dB  - dB_mean)  / dB_std
        d2B = (d2B - d2B_mean) / d2B_std
        dH  = (dH  - dH_mean)  / dH_std
        d2H = (d2H - d2H_mean) / d2H_std

        # -------- FUTURE B CHANNEL --------
        T_seq   = np.ones((len(B_seq), window), dtype=np.float32)
        Bf_full = np.zeros((len(B_seq), window), dtype=np.float32)
        Bf_full[:, :ksteps] = B_future[:, :ksteps]   # safe: ksteps <= window always

        # -------- STACK FEATURES --------
        X = np.stack([B_seq, H_seq, dB, d2B, dH, d2H, T_seq, Bf_full], axis=2)
        # (N, window, 8)

        self.x = torch.tensor(X,          dtype=torch.float32)
        self.y = torch.tensor(H_target,   dtype=torch.float32)  # (N, ksteps)

        print(f"  Freq {freq} | window={window} | ksteps={ksteps} | "
              f"X: {self.x.shape} | Y: {self.y.shape}")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


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


# ================= TRAIN ALL TEACHERS =================
criterion = nn.MSELoss()
results   = {}

for freq in FREQS:
    window = WINDOW_PER_FREQ[freq]
    ksteps = KSTEPS_PER_FREQ[freq]

    print(f"\n{'='*55}")
    print(f"  Training Teacher {freq} | window={window} | ksteps={ksteps}")
    print(f"{'='*55}")

    ds     = BHDataset(DATA_DIR, freq)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    model     = BHTransFourier(8, 128, 3, 4, ksteps).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_loss        = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for x, y in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(loader)
        scheduler.step()

        print(f"Epoch {epoch+1:03d} | Loss: {train_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        if train_loss < best_loss:
            best_loss        = train_loss
            patience_counter = 0
            save_path = f"{DATA_DIR}/teacher_freq{freq}_w{window}_k{ksteps}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"  ✅ Teacher {freq} saved!")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE_LIMIT:
                print(f"  🛑 Early stopping at epoch {epoch+1}")
                break

    results[freq] = best_loss
    print(f"✅ Teacher {freq} complete. Best loss: {best_loss:.6f}")

# ================= SUMMARY =================
print(f"\n{'='*55}")
print("  TEACHER TRAINING SUMMARY")
print(f"{'='*55}")
for freq, loss in results.items():
    print(f"  Teacher {freq} (window={WINDOW_PER_FREQ[freq]}, ksteps={KSTEPS_PER_FREQ[freq]}): "
          f"Best loss = {loss:.6f}")
print(f"\n🎯 All {len(FREQS)} teachers trained!")