#!/usr/bin/env python3
"""
Preprocessing: saves per-frequency datasets with their own window sizes and ksteps.
No combined student dataset — student uses per-freq files directly.
"""

import os
import numpy as np
import h5py

# ================= CONFIG =================
RAW_H5  = "/content/drive/MyDrive/onefreq1/MaterialA_raw.h5"
OUT_DIR = "/content/drive/MyDrive/onefreq1/A_sequences_flat_debug"

STRIDE = 2
FREQS  = [1, 2, 3, 4, 5, 6, 7]

WINDOW_PER_FREQ = {1: 320, 2: 200, 3: 130, 4: 80, 5: 50, 6: 32, 7: 20}
KSTEPS_PER_FREQ = {1: 320, 2: 200, 3: 130, 4: 80, 5: 50, 6: 32, 7: 20}

os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(1)

# ================= WINDOW FUNCTION =================
def create_windows_flat(B_1d, H_1d, T_scalar, freq, window, ksteps):
    seq_B_list    = []
    seq_H_list    = []
    future_B_list = []
    target_H_list = []
    T_list        = []
    F_list        = []

    total_len  = len(B_1d)
    block_size = window + ksteps

    for start in range(0, total_len - block_size + 1, STRIDE):
        seq_B_list.append(B_1d[start : start + window])
        seq_H_list.append(H_1d[start : start + window])
        future_B_list.append(B_1d[start + window : start + window + ksteps])
        target_H_list.append(H_1d[start + window : start + window + ksteps])
        T_list.append(T_scalar)
        F_list.append(freq)

    print(f"  Windows (window={window}, ksteps={ksteps}): {len(seq_B_list)}")
    return seq_B_list, seq_H_list, future_B_list, target_H_list, T_list, F_list

# ================= MAIN =================
with h5py.File(RAW_H5, "r") as f:
    for freq in FREQS:
        window = WINDOW_PER_FREQ[freq]
        ksteps = KSTEPS_PER_FREQ[freq]

        print(f"\n{'='*50}")
        print(f"Processing frequency {freq}  |  window={window}  |  ksteps={ksteps}")
        print(f"{'='*50}")

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

        windows = create_windows_flat(B_flat, H_flat, T_mean, freq, window, ksteps)

        prefix = f"{OUT_DIR}/freq{freq}"
        np.save(f"{prefix}_seq_B.npy",    np.array(windows[0], dtype=np.float32))
        np.save(f"{prefix}_seq_H.npy",    np.array(windows[1], dtype=np.float32))
        np.save(f"{prefix}_future_B.npy", np.array(windows[2], dtype=np.float32))
        np.save(f"{prefix}_target_H.npy", np.array(windows[3], dtype=np.float32))
        np.save(f"{prefix}_T_now.npy",    np.array(windows[4], dtype=np.float32))
        np.save(f"{prefix}_freq.npy",     np.array(windows[5], dtype=np.float32))
        print(f"  Freq {freq} saved: {len(windows[0])} windows")

# ================= SUMMARY =================
print(f"\n{'='*50}")
print(f"[SUMMARY]")
for freq in FREQS:
    t = np.load(f"{OUT_DIR}/freq{freq}_seq_B.npy")
    print(f"  Freq {freq} (window={WINDOW_PER_FREQ[freq]}, ksteps={KSTEPS_PER_FREQ[freq]}): "
          f"{t.shape[0]} windows  shape={t.shape}")
print(f"{'='*50}")
print("\n✅ Preprocessing complete")