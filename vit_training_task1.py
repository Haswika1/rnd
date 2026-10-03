#!/usr/bin/env python3
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
K_STEPS        = 100
SEQ_LEN        = 100

# ================= NORMALIZATION STATS =================
print("Computing normalization statistics from training data...")

train_B        = np.load(f"{DATA_DIR}/train_seq_B.npy")
train_H        = np.load(f"{DATA_DIR}/train_seq_H.npy")
train_T        = np.load(f"{DATA_DIR}/train_T_now.npy")
train_B_future = np.load(f"{DATA_DIR}/train_future_B.npy")

B_mean, B_std = train_B.mean(), train_B.std()
H_mean, H_std = train_H.mean(), train_H.std()
T_mean        = train_T.mean()
T_std         = 1.0

train_B_n        = (train_B        - B_mean) / B_std
train_H_n        = (train_H        - H_mean) / H_std
train_B_future_n = (train_B_future - B_mean) / B_std

# -------- FORWARD dB, d2B --------
B_combined_train   = np.concatenate([train_B_n, train_B_future_n[:, :K_STEPS]], axis=1)
dB_combined_train  = np.diff(B_combined_train,  axis=1)
d2B_combined_train = np.diff(dB_combined_train, axis=1)

dB_train  = dB_combined_train[:,  99: 99 + K_STEPS]
d2B_train = d2B_combined_train[:, 98: 98 + K_STEPS]
dB_train  = np.pad(dB_train,  ((0, 0), (0, SEQ_LEN - K_STEPS)))
d2B_train = np.pad(d2B_train, ((0, 0), (0, SEQ_LEN - K_STEPS)))

# -------- BACKWARD dH, d2H --------
dH_train  = np.diff(train_H_n, axis=1, prepend=train_H_n[:, :1])
d2H_train = np.diff(train_H_n, n=2, axis=1, prepend=train_H_n[:, :2])

dB_mean,  dB_std  = dB_train.mean(),  dB_train.std()
d2B_mean, d2B_std = d2B_train.mean(), d2B_train.std()
dH_mean,  dH_std  = dH_train.mean(),  dH_train.std()
d2H_mean, d2H_std = d2H_train.mean(), d2H_train.std()

print("B   mean/std:", B_mean,   B_std)
print("H   mean/std:", H_mean,   H_std)
print("T   mean/std:", T_mean,   T_std)
print("dB  mean/std:", dB_mean,  dB_std)
print("d2B mean/std:", d2B_mean, d2B_std)
print("dH  mean/std:", dH_mean,  dH_std)
print("d2H mean/std:", d2H_mean, d2H_std)

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
print("Normalization stats saved.")

# ================= DATASET =================
class BHDataset(Dataset):

    def __init__(self, data_dir):

        B_seq    = np.load(f"{data_dir}/train_seq_B.npy")
        H_seq    = np.load(f"{data_dir}/train_seq_H.npy")
        B_future = np.load(f"{data_dir}/train_future_B.npy")
        T_now    = np.load(f"{data_dir}/train_T_now.npy")
        H_target = np.load(f"{data_dir}/train_target_H.npy")

        # -------- NORMALIZATION --------
        B_seq    = (B_seq    - B_mean) / B_std
        H_seq    = (H_seq    - H_mean) / H_std
        B_future = (B_future - B_mean) / B_std
        T_now    = np.ones_like(T_now)
        H_target = (H_target - H_mean) / H_std   # normalize absolute H target

        # -------- FORWARD dB, d2B --------
        B_combined   = np.concatenate([B_seq, B_future[:, :K_STEPS]], axis=1)
        dB_combined  = np.diff(B_combined,  axis=1)
        d2B_combined = np.diff(dB_combined, axis=1)

        dB  = dB_combined[:,  99: 99 + K_STEPS]
        d2B = d2B_combined[:, 98: 98 + K_STEPS]
        dB  = np.pad(dB,  ((0, 0), (0, SEQ_LEN - K_STEPS)))
        d2B = np.pad(d2B, ((0, 0), (0, SEQ_LEN - K_STEPS)))

        # -------- BACKWARD dH, d2H --------
        dH  = np.diff(H_seq, axis=1, prepend=H_seq[:, :1])
        d2H = np.diff(H_seq, n=2, axis=1, prepend=H_seq[:, :2])

        # -------- NORMALIZE DERIVATIVES --------
        dB  = (dB  - dB_mean)  / dB_std
        d2B = (d2B - d2B_mean) / d2B_std
        dH  = (dH  - dH_mean)  / dH_std
        d2H = (d2H - d2H_mean) / d2H_std

        # -------- PRINT STATS --------
        print(f"\n[TRAIN] Normalized feature statistics:")
        print(f"  B_seq    → mean: {B_seq.mean():.4f},  std: {B_seq.std():.4f}")
        print(f"  H_seq    → mean: {H_seq.mean():.4f},  std: {H_seq.std():.4f}")
        print(f"  B_future → mean: {B_future.mean():.4f},  std: {B_future.std():.4f}")
        print(f"  T_now    → mean: {T_now.mean():.4f},  std: {T_now.std():.4f} (constant 1.0)")
        print(f"  H_target → mean: {H_target.mean():.4f},  std: {H_target.std():.4f}")
        print(f"  dB       → mean: {dB.mean():.4f},  std: {dB.std():.4f}")
        print(f"  d2B      → mean: {d2B.mean():.4f},  std: {d2B.std():.4f}")
        print(f"  dH       → mean: {dH.mean():.4f},  std: {dH.std():.4f}")
        print(f"  d2H      → mean: {d2H.mean():.4f},  std: {d2H.std():.4f}")

        # -------- TEMPERATURE SEQUENCE --------
        T_seq = np.repeat(T_now[:, None], SEQ_LEN, axis=1)

        # -------- FUTURE B FEATURE --------
        Bf_full = np.zeros_like(B_seq)
        Bf_full[:, :K_STEPS] = B_future[:, :K_STEPS]

        # -------- INPUT FEATURES --------
        X = np.stack([B_seq, H_seq, dB, d2B, dH, d2H, T_seq, Bf_full], axis=2)

        # -------- ABSOLUTE H TARGET (no delta) --------
        # predict absolute H directly — no accumulation error at inference
        abs_target = H_target[:, :K_STEPS]   # already normalized above

        self.x = torch.tensor(X,          dtype=torch.float32)
        self.y = torch.tensor(abs_target, dtype=torch.float32)

        print(f"X shape: {self.x.shape}")
        print(f"Y shape: {self.y.shape}")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# ================= LOAD DATA =================
train_ds     = BHDataset(DATA_DIR)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# ================= MODEL CONFIG =================
INPUT_DIM = 8
D_MODEL   = 128
N_LAYERS  = 3
N_HEADS   = 4

# ================= MODEL COMPONENTS =================
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

# ================= TRAIN SETUP =================
model     = BHTransFourier(INPUT_DIM, D_MODEL, N_LAYERS, N_HEADS, K_STEPS).to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_loss        = float("inf")
patience_counter = 0

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Training samples: {len(train_ds)}")
print(f"K_STEPS:          {K_STEPS}")
print(f"Device:           {DEVICE}\n")

# ================= TRAIN LOOP =================
for epoch in range(EPOCHS):

    model.train()
    train_loss = 0

    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    scheduler.step()

    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch+1:03d} | Train: {train_loss:.6f} | LR: {current_lr:.6f}")

    if train_loss < best_loss:
        best_loss        = train_loss
        patience_counter = 0
        torch.save(model.state_dict(), f"{DATA_DIR}/A_best_bh_transfourier.pth")
        print("✅ Model saved!")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE_LIMIT:
            print("🛑 Early stopping triggered.")
            break

print("🎯 Training complete.")
print(f"Best train loss: {best_loss:.6f}")