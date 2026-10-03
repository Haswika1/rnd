#!/usr/bin/env python3
"""
Student KD training — per-freq batching, variable window sizes
Each freq batch has its own shape. Teacher inputs use last-N-timesteps slicing.
L_KD = (1-beta) * MSE(actual, student_pred)
     + beta * sum_i(w_i * MSE(teacher_i_pred, student_pred))
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
BETA           = 0.1
FREQS          = [1, 2, 3, 4, 5, 6, 7]

WINDOW_PER_FREQ = {1: 320, 2: 200, 3: 130, 4: 80, 5: 50, 6: 32, 7: 20}
KSTEPS_PER_FREQ = {1: 320, 2: 200, 3: 130, 4: 80, 5: 50, 6: 32, 7: 20}

# Max ksteps across all freqs (for unified output head of student)
MAX_KSTEPS = max(KSTEPS_PER_FREQ.values())  # 100

# ================= LOAD NORM STATS =================
stats    = np.load(f"{DATA_DIR}/norm_stats.npz")
B_mean   = stats["B_mean"];   B_std   = stats["B_std"]
H_mean   = stats["H_mean"];   H_std   = stats["H_std"]
T_mean   = stats["T_mean"];   T_std   = stats["T_std"]
dB_mean  = stats["dB_mean"];  dB_std  = stats["dB_std"]
d2B_mean = stats["d2B_mean"]; d2B_std = stats["d2B_std"]
dH_mean  = stats["dH_mean"];  dH_std  = stats["dH_std"]
d2H_mean = stats["d2H_mean"]; d2H_std = stats["d2H_std"]
print("✅ Norm stats loaded")


# ================= DATASET =================
class BHDataset(Dataset):

    def __init__(self, data_dir, freq):
        window = WINDOW_PER_FREQ[freq]
        ksteps = KSTEPS_PER_FREQ[freq]

        B_seq    = np.load(f"{data_dir}/freq{freq}_seq_B.npy")     # (N, window)
        H_seq    = np.load(f"{data_dir}/freq{freq}_seq_H.npy")     # (N, window)
        B_future = np.load(f"{data_dir}/freq{freq}_future_B.npy")  # (N, ksteps)
        T_now    = np.load(f"{data_dir}/freq{freq}_T_now.npy")
        H_target = np.load(f"{data_dir}/freq{freq}_target_H.npy")  # (N, ksteps)

        B_seq    = (B_seq    - B_mean) / B_std
        H_seq    = (H_seq    - H_mean) / H_std
        B_future = (B_future - B_mean) / B_std
        T_now    = np.ones_like(T_now)
        H_target = (H_target - H_mean) / H_std

        # -------- FORWARD dB, d2B --------
        B_combined   = np.concatenate([B_seq, B_future], axis=1)  # (N, window+ksteps)
        dB_combined  = np.diff(B_combined,  axis=1)
        d2B_combined = np.diff(dB_combined, axis=1)

        dB_raw  = dB_combined[:,  window - 1 : window - 1 + ksteps]
        d2B_raw = d2B_combined[:, window - 2 : window - 2 + ksteps]

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
        Bf_full[:, :ksteps] = B_future[:, :ksteps]

        X = np.stack([B_seq, H_seq, dB, d2B, dH, d2H, T_seq, Bf_full], axis=2)
        # (N, window, 8)

        # Pad H_target to MAX_KSTEPS so all freqs have same Y shape
        H_target_padded = np.zeros((len(H_target), MAX_KSTEPS), dtype=np.float32)
        H_target_padded[:, :ksteps] = H_target

        self.x      = torch.tensor(X,              dtype=torch.float32)
        self.y      = torch.tensor(H_target_padded, dtype=torch.float32)
        self.ksteps = ksteps
        self.window = window
        self.freq   = freq

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
    """Student: outputs MAX_KSTEPS always; we slice to freq-specific ksteps in loss."""
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


# ================= LOAD ALL TEACHERS =================
print("\nLoading teachers...")
teachers = {}
for freq in FREQS:
    window = WINDOW_PER_FREQ[freq]
    ksteps = KSTEPS_PER_FREQ[freq]

    # Teacher head outputs ksteps for its own freq
    class TeacherModel(nn.Module):
        def __init__(self, k):
            super().__init__()
            self.input_proj = nn.Linear(8, 128)
            self.blocks     = nn.ModuleList([Block(128, 4) for _ in range(3)])
            self.ln_f       = nn.LayerNorm(128)
            self.head       = nn.Sequential(
                nn.Linear(128, 128),
                nn.SiLU(),
                nn.Linear(128, k)
            )
        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = block(x)
            x = self.ln_f(x)
            x = x[:, -1, :]
            return self.head(x)

    t = TeacherModel(ksteps).to(DEVICE)
    t.load_state_dict(torch.load(
        f"{DATA_DIR}/teacher_freq{freq}_w{window}_k{ksteps}.pth",
        map_location=DEVICE
    ))
    t.eval()
    teachers[freq] = t
    print(f"  ✅ Teacher {freq} loaded (window={window}, ksteps={ksteps})")


# ================= KD LOSS =================
def kd_loss_for_freq(student_pred_full, teacher_preds, target_padded, freq, beta=BETA):
    """
    student_pred_full : (B, MAX_KSTEPS)
    teacher_preds     : dict freq -> (B, ksteps_for_that_freq) — already padded to MAX_KSTEPS
    target_padded     : (B, MAX_KSTEPS) — target padded, valid only [:ksteps]
    freq              : current freq being trained
    """
    ksteps = KSTEPS_PER_FREQ[freq]

    # Slice to valid ksteps
    s_pred  = student_pred_full[:, :ksteps]   # (B, ksteps)
    target  = target_padded[:,   :ksteps]     # (B, ksteps)

    hard_loss = nn.functional.mse_loss(s_pred, target)

    # Soft loss: compare student vs each teacher (slice to min shared ksteps)
    mse_list = []
    for f in FREQS:
        k_f    = KSTEPS_PER_FREQ[f]
        k_min  = min(ksteps, k_f)
        t_pred = teacher_preds[f][:, :k_min]
        s_sl   = s_pred[:, :k_min]
        mse_i  = nn.functional.mse_loss(t_pred, s_sl, reduction='none').mean(dim=1)
        mse_list.append(mse_i)

    mse_stack = torch.stack(mse_list, dim=0)       # (7, B)
    weights   = torch.softmax(-mse_stack, dim=0)   # (7, B)

    soft_loss = 0
    for i, f in enumerate(FREQS):
        k_f    = KSTEPS_PER_FREQ[f]
        k_min  = min(ksteps, k_f)
        t_pred = teacher_preds[f][:, :k_min]
        s_sl   = s_pred[:, :k_min]
        mse_i  = nn.functional.mse_loss(t_pred, s_sl, reduction='none').mean(dim=1)
        soft_loss += (weights[i] * mse_i).mean()

    return (1 - beta) * hard_loss + beta * soft_loss


def get_teacher_input(x_student, teacher_freq):
    """
    x_student   : (B, student_window, 8)
    teacher_freq: which teacher we're feeding
    Returns     : (B, teacher_window, 8)
    Uses last teacher_window timesteps of student input (or pads if student shorter).
    """
    t_window = WINDOW_PER_FREQ[teacher_freq]
    s_window = x_student.shape[1]

    if s_window >= t_window:
        # Slice last t_window steps
        return x_student[:, -t_window:, :]
    else:
        # Left-pad with zeros
        pad = torch.zeros(x_student.shape[0], t_window - s_window, x_student.shape[2],
                          device=x_student.device)
        return torch.cat([pad, x_student], dim=1)


# ================= BUILD PER-FREQ DATALOADERS =================
print("\nBuilding per-freq datasets...")
datasets    = {}
dataloaders = {}
for freq in FREQS:
    ds = BHDataset(DATA_DIR, freq)
    datasets[freq]    = ds
    dataloaders[freq] = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                                   num_workers=2, pin_memory=True)

# ================= STUDENT MODEL =================
student   = BHTransFourier(8, 128, 3, 4, MAX_KSTEPS).to(DEVICE)
optimizer = torch.optim.Adam(student.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_loss        = float("inf")
patience_counter = 0

total_params = sum(p.numel() for p in student.parameters())
print(f"\nStudent parameters: {total_params:,}")
print(f"MAX_KSTEPS (student output): {MAX_KSTEPS}")
print(f"BETA: {BETA}  |  Device: {DEVICE}\n")

# ================= TRAINING LOOP =================
for epoch in range(EPOCHS):
    student.train()
    epoch_loss   = 0
    total_batches = 0

    # Cycle through all freq dataloaders each epoch
    iterators = {freq: iter(dl) for freq, dl in dataloaders.items()}
    max_batches_per_freq = max(len(dl) for dl in dataloaders.values())

    for _ in tqdm(range(max_batches_per_freq), desc=f"Epoch {epoch+1}/{EPOCHS}"):
        for freq in FREQS:
            try:
                x, y = next(iterators[freq])
            except StopIteration:
                iterators[freq] = iter(dataloaders[freq])
                x, y = next(iterators[freq])

            x = x.to(DEVICE)   # (B, window_freq, 8)
            y = y.to(DEVICE)   # (B, MAX_KSTEPS)

            # Get all teacher predictions for this batch
            teacher_preds = {}
            with torch.no_grad():
                for tf in FREQS:
                    x_t = get_teacher_input(x, tf)          # (B, teacher_window, 8)
                    teacher_preds[tf] = teachers[tf](x_t)   # (B, ksteps_tf)

            optimizer.zero_grad()
            student_pred = student(x)                        # (B, MAX_KSTEPS)
            loss = kd_loss_for_freq(student_pred, teacher_preds, y, freq)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            epoch_loss    += loss.item()
            total_batches += 1

    epoch_loss /= total_batches
    scheduler.step()

    print(f"Epoch {epoch+1:03d} | KD Loss: {epoch_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    if epoch_loss < best_loss:
        best_loss        = epoch_loss
        patience_counter = 0
        torch.save(student.state_dict(), f"{DATA_DIR}/student1_kd_maxk{MAX_KSTEPS}.pth")
        print("  ✅ Student saved!")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE_LIMIT:
            print("  🛑 Early stopping triggered.")
            break

print(f"\n🎯 Student KD training complete.")
print(f"   Best KD loss : {best_loss:.6f}")
print(f"   Saved        : student1_kd_maxk{MAX_KSTEPS}.pth")