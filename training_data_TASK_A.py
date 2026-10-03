#!/usr/bin/env python3
"""
Preprocessing: saves both per-frequency datasets (for teachers)
and combined dataset (for student)
"""

import os
import numpy as np
import h5py

# ================= CONFIG =================
RAW_H5  = "/content/drive/MyDrive/onefreq1/MaterialA_raw.h5"
OUT_DIR = "/content/drive/MyDrive/onefreq1/A_sequences_flat_debug"

WINDOW  = 100
STRIDE  = 2
FREQS   = [1,2,3,4,5,6,7]
ksteps  = 100   # max — teachers will slice as needed

os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(1)

# ================= WINDOW FUNCTION =================
def create_windows_flat(B_1d, H_1d, T_scalar, freq):
    seq_B_list    = []
    seq_H_list    = []
    future_B_list = []
    target_H_list = []
    T_list        = []
    F_list        = []

    total_len  = len(B_1d)
    block_size = WINDOW + ksteps

    for start in range(0, total_len - block_size + 1, STRIDE):
        seq_B_list.append(B_1d[start : start + WINDOW])
        seq_H_list.append(H_1d[start : start + WINDOW])
        future_B_list.append(B_1d[start + WINDOW : start + WINDOW + ksteps])
        target_H_list.append(H_1d[start + WINDOW : start + WINDOW + ksteps])
        T_list.append(T_scalar)
        F_list.append(freq)

    print(f"  Windows: {len(seq_B_list)}")
    return seq_B_list, seq_H_list, future_B_list, target_H_list, T_list, F_list

# ================= COMBINED STORAGE =================
all_B, all_H, all_futB, all_tarH, all_T, all_F = [], [], [], [], [], []

# ================= MAIN =================
with h5py.File(RAW_H5, "r") as f:
    for freq in FREQS:
        print(f"\nProcessing frequency {freq}")

        B = f[f"B_{freq}"][:]
        H = f[f"H_{freq}"][:]
        T = f[f"T_{freq}"][:].reshape(-1)

        good_rows = np.isfinite(B).all(axis=1) & np.isfinite(H).all(axis=1)
        B = B[good_rows]
        H = H[good_rows]
        T = T[good_rows]

        print(f"  After filtering: B {B.shape}")

        B_flat = B.ravel()
        H_flat = H.ravel()
        T_mean = T.mean()

        print(f"  Flattened: {B_flat.shape}")

        windows = create_windows_flat(B_flat, H_flat, T_mean, freq)

        # ===== SAVE PER FREQUENCY =====
        prefix = f"{OUT_DIR}/freq{freq}"
        np.save(f"{prefix}_seq_B.npy",    np.array(windows[0], dtype=np.float32))
        np.save(f"{prefix}_seq_H.npy",    np.array(windows[1], dtype=np.float32))
        np.save(f"{prefix}_future_B.npy", np.array(windows[2], dtype=np.float32))
        np.save(f"{prefix}_target_H.npy", np.array(windows[3], dtype=np.float32))
        np.save(f"{prefix}_T_now.npy",    np.array(windows[4], dtype=np.float32))
        np.save(f"{prefix}_freq.npy",     np.array(windows[5], dtype=np.float32))
        print(f"  Freq {freq} saved: {len(windows[0])} windows")

        # ===== ADD TO COMBINED =====
        all_B.extend(windows[0])
        all_H.extend(windows[1])
        all_futB.extend(windows[2])
        all_tarH.extend(windows[3])
        all_T.extend(windows[4])
        all_F.extend(windows[5])

# ================= SAVE COMBINED =================
print("\nSaving combined dataset...")
np.save(f"{OUT_DIR}/train_seq_B.npy",    np.array(all_B,    dtype=np.float32))
np.save(f"{OUT_DIR}/train_seq_H.npy",    np.array(all_H,    dtype=np.float32))
np.save(f"{OUT_DIR}/train_future_B.npy", np.array(all_futB, dtype=np.float32))
np.save(f"{OUT_DIR}/train_target_H.npy", np.array(all_tarH, dtype=np.float32))
np.save(f"{OUT_DIR}/train_T_now.npy",    np.array(all_T,    dtype=np.float32))
np.save(f"{OUT_DIR}/train_freq.npy",     np.array(all_F,    dtype=np.float32))

print(f"Combined: {len(all_B)} windows")
print("\n[DEBUG] Final shapes:")
for freq in FREQS:
    d = np.load(f"{OUT_DIR}/freq{freq}_seq_B.npy")
    print(f"  Freq {freq}: {d.shape[0]} windows")
print(f"  Combined: {len(all_B)} windows")
print("\n✅ Preprocessing complete")