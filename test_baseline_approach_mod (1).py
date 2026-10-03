#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =====================================================
# Remove rows with invalid incremental permeability
# =====================================================
def remove_invalid_mu_rows(B, H):
    mu0 = 4 * np.pi * 1e-7
    Bp = np.concatenate([B[:, -1:], B], axis=1)
    Hp = np.concatenate([H[:, -1:], H], axis=1)
    dB = Bp[:, 1:] - Bp[:, :-1]
    dH = Hp[:, 1:] - Hp[:, :-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = dB / dH
    mu_r = mu / mu0
    bad_rows = np.any(~np.isfinite(mu_r), axis=1)
    keep_idx = np.where(~bad_rows)[0]
    print(f"    Removed {bad_rows.sum()} invalid rows "
          f"({len(keep_idx)} kept / {B.shape[0]})")
    return keep_idx

# =====================================================
# Base paths
# =====================================================
BASE     = "/content/drive/MyDrive/onefreq1"
PRED_DIR = "/content/drive/MyDrive/onefreq1/Material_A1"
GT_DIR   = os.path.join(BASE, "Ground_Truth_Data")

# =====================================================
# Material configuration
# =====================================================
materials = {
    "A": {"code": "3C92", "folder": "MaterialA-3C92", "pred_file": "Af.csv", "testSeqLen": 1260},
}

# =====================================================
# Frequency values per freq index (in kHz -> converted to Hz below)
# freq 1..7 correspond to these values, as provided by user
# =====================================================
FREQ_KHZ = {
    1: 320,
    2: 200,
    3: 130,
    4: 80,
    5: 50,
    6: 32,
    7: 20,
}
FREQ_HZ = {k: v * 1e3 for k, v in FREQ_KHZ.items()}

# Time window to plot, in seconds
T_MAX = 60e-6  # 60 microseconds

# =====================================================
# Accumulators
# =====================================================
seq_errup_all = []
ene_errup_all = []
global_seq_errors = []
global_ene_errors = []

# =====================================================
# Evaluation loop
# =====================================================
for key, m in materials.items():

    print(f"\nEvaluating Baseline Material {key} ({m['code']})")

    pred   = np.loadtxt(os.path.join(PRED_DIR, m["pred_file"]), delimiter=",")
    assert pred.shape[0] == m["testSeqLen"], "Prediction length mismatch"

    gt_path = os.path.join(GT_DIR, m["folder"])
    B_true  = np.loadtxt(os.path.join(gt_path, f"{m['code']}_Testing_True_B_seq.csv"), delimiter=",")
    H_true  = np.loadtxt(os.path.join(gt_path, f"{m['code']}_Testing_True_H_seq.csv"), delimiter=",")
    Loss    = np.loadtxt(os.path.join(gt_path, f"{m['code']}_Testing_True_Loss.csv"),  delimiter=",")

    ttl_segs = 3
    ratios   = [0.9, 0.5, 0.1]

    if m["code"] == "FEC007":
        ttl_segs = 1
        ratios   = [0.999]

    seq_errup = []
    ene_errup = []

    for n in range(ttl_segs):
        s   = int(n * H_true.shape[0] / ttl_segs)
        e   = int((n + 1) * H_true.shape[0] / ttl_segs)
        cut = round((1 - ratios[n]) * H_true.shape[1])

        B_block = B_true[s:e]
        H_block = H_true[s:e]
        P_block = pred[s:e]
        L_block = Loss[s:e]

        valid_idx = remove_invalid_mu_rows(B_block, H_block)
        B_block = B_block[valid_idx]
        H_block = H_block[valid_idx]
        P_block = P_block[valid_idx]
        L_block = L_block[valid_idx]

        B           = B_block[:, cut:]
        prediction  = P_block[:, cut:]
        measurement = H_block[:, cut:]
        Core_loss   = L_block

        rms_H   = np.sqrt(np.mean((prediction - measurement) ** 2, axis=1))
        rmse_pm = rms_H / np.sqrt(np.mean(measurement ** 2, axis=1)) * 100

        dB        = np.diff(B, axis=1)
        H_mid_p   = (prediction[:, :-1] + prediction[:, 1:]) / 2
        H_mid_m   = (measurement[:, :-1] + measurement[:, 1:]) / 2
        diff_ene  = (H_mid_p - H_mid_m) * dB
        ene_error = np.sum(diff_ene, axis=1) / Core_loss * 100

        seq_errup.append(np.percentile(np.abs(rmse_pm), 95))
        ene_errup.append(np.percentile(np.abs(ene_error), 95))
        global_seq_errors.extend(np.abs(rmse_pm))
        global_ene_errors.extend(np.abs(ene_error))

    seq_errup_all.append(seq_errup)
    ene_errup_all.append(ene_errup)
    print("  ✓ Done")

# =====================================================
# Metrics
# =====================================================
material_seq_avg_95 = np.array([np.mean(v) for v in seq_errup_all])
material_ene_avg_95 = np.array([np.mean(v) for v in ene_errup_all])
overall_seq_avg_95  = np.mean(material_seq_avg_95)
overall_ene_avg_95  = np.mean(material_ene_avg_95)
global_seq_95 = np.percentile(np.array(global_seq_errors), 95)
global_ene_95 = np.percentile(np.array(global_ene_errors), 95)

print("\n================ MATERIAL-WISE AVERAGES ================")
for i, key in enumerate(materials.keys()):
    print(f"Material {key}: Seq95 = {material_seq_avg_95[i]:.4f} %, "
          f"Ene95 = {material_ene_avg_95[i]:.4f} %")

print("\n================ OVERALL AVERAGES ======================")
print(f"Overall Avg 95% Sequence Error (%) : {overall_seq_avg_95:.4f}")
print(f"Overall Avg 95% Energy Error (%)   : {overall_ene_avg_95:.4f}")

print("\n================ TRUE GLOBAL METRICS ===================")
print(f"Global 95% Sequence Error (%)      : {global_seq_95:.4f}")
print(f"Global 95% Energy Error (%)        : {global_ene_95:.4f}")
print("=======================================================")

# =====================================================
# FREQ-WISE EVALUATION
# =====================================================
print("\n================ FREQ-WISE METRICS ===================")

for freq in range(1, 8):
    print(f"\n--- Frequency {freq} ---")
    idx_file = os.path.join(PRED_DIR, f"indices_{freq}.csv")
    if not os.path.exists(idx_file):
        print(f"  Missing indices file: {idx_file}")
        continue

    idx = pd.read_csv(idx_file, header=None).values.squeeze().astype(int)
    print(f"  Rows: {len(idx)}")

    B_f = B_true[idx]; H_f = H_true[idx]
    P_f = pred[idx];   L_f = Loss[idx]

    seq_errors = []; ene_errors = []

    for n in range(3):
        s   = int(n * B_f.shape[0] / 3)
        e   = int((n + 1) * B_f.shape[0] / 3)
        cut = round((1 - [0.9, 0.5, 0.1][n]) * B_f.shape[1])

        B_block = B_f[s:e]; H_block = H_f[s:e]
        P_block = P_f[s:e]; L_block = L_f[s:e]

        valid_idx = remove_invalid_mu_rows(B_block, H_block)
        B_block = B_block[valid_idx]; H_block = H_block[valid_idx]
        P_block = P_block[valid_idx]; L_block = L_block[valid_idx]

        if len(B_block) == 0:
            continue

        B           = B_block[:, cut:]
        prediction  = P_block[:, cut:]
        measurement = H_block[:, cut:]
        Core_loss   = L_block

        rms_H    = np.sqrt(np.mean((prediction - measurement) ** 2, axis=1))
        rmse_pm  = rms_H / np.sqrt(np.mean(measurement ** 2, axis=1)) * 100
        dB       = np.diff(B, axis=1)
        H_mid_p  = (prediction[:, :-1] + prediction[:, 1:]) / 2
        H_mid_m  = (measurement[:, :-1] + measurement[:, 1:]) / 2
        diff_ene = (H_mid_p - H_mid_m) * dB
        ene_error = np.sum(diff_ene, axis=1) / Core_loss * 100

        seq_errors.extend(np.abs(rmse_pm))
        ene_errors.extend(np.abs(ene_error))

    if len(seq_errors) > 0:
        print(f"  Seq 95% Error : {np.percentile(seq_errors, 95):.4f}%")
        print(f"  Ene 95% Error : {np.percentile(ene_errors, 95):.4f}%")
    else:
        print("  No valid rows after filtering")


# =====================================================
# PLOTS PER FREQ (no "best row" selection — just picks
# a representative valid row per frequency, plotted as
# dual-axis H (left) / B (right) vs Time, cropped to
# T_MAX using each frequency's own period)
# =====================================================
PLOT_DIR_FREQ = os.path.join(PRED_DIR, "freq_plots")
H_AXIS_COLOR  = "#1f77b4"   # blue
B_AXIS_COLOR  = "#ff7f0e"   # orange
os.makedirs(PLOT_DIR_FREQ, exist_ok=True)

print("\n================ PLOTS PER FREQ (0-60us) ===================")

for freq in range(1, 8):
    print(f"--- Frequency {freq} ({FREQ_KHZ[freq]} kHz) ---")

    idx_file = os.path.join(PRED_DIR, f"indices_{freq}.csv")
    if not os.path.exists(idx_file):
        print(f"  Missing: {idx_file} — skipping\n")
        continue

    idx = pd.read_csv(idx_file, header=None).values.squeeze().astype(int)
    if idx.ndim == 0:
        idx = idx.reshape(1)

    B_f = B_true[idx]
    H_f = H_true[idx]

    valid_local = remove_invalid_mu_rows(B_f, H_f)
    if len(valid_local) == 0:
        print("  No valid rows — skipping\n")
        continue

    # pick a random valid row for this frequency (not "best")
    global_idx = idx[np.random.choice(valid_local)]

    H_row = H_true[global_idx]
    B_row = B_true[global_idx]
    P_row = pred[global_idx]

    N = H_row.shape[0]                       # samples per period for this row
    period = 1.0 / FREQ_HZ[freq]              # seconds
    dt = period / N                           # seconds per sample
    time_s = np.arange(N) * dt
    time_us = time_s * 1e6

    mask = time_s <= T_MAX
    if not np.any(mask):
        print(f"  Warning: period ({period*1e6:.3f} us) shorter than first sample step "
              f"vs T_MAX={T_MAX*1e6:.0f}us — plotting full period instead.\n")
        mask = np.ones(N, dtype=bool)

    time_us_cut = time_us[mask]
    H_row_cut   = H_row[mask]
    P_row_cut   = P_row[mask]
    B_row_cut   = B_row[mask]

    print(f"  Global row: {global_idx} | period: {period*1e6:.3f} us | "
          f"dt: {dt*1e9:.2f} ns | samples in window: {mask.sum()}/{N}")

    fig, ax_h = plt.subplots(figsize=(9, 4.5))
    ax_b = ax_h.twinx()

    l1, = ax_h.plot(time_us_cut, H_row_cut, color=H_AXIS_COLOR, lw=2.0,
                     label="Measured H")
    l2, = ax_h.plot(time_us_cut, P_row_cut, color=H_AXIS_COLOR, lw=1.5, ls="--",
                     label="Predicted H")
    l3, = ax_b.plot(time_us_cut, B_row_cut, color=B_AXIS_COLOR, lw=2.0,
                     label="Measured B")

    ax_h.set_xlabel("Time (µs)", fontsize=12)
    ax_h.set_ylabel("H  (A/m)", fontsize=12, color=H_AXIS_COLOR)
    ax_b.set_ylabel("B  (T)", fontsize=12, color=B_AXIS_COLOR)

    ax_h.tick_params(axis="y", labelcolor=H_AXIS_COLOR)
    ax_b.tick_params(axis="y", labelcolor=B_AXIS_COLOR)

    ax_h.set_xlim(0, min(T_MAX * 1e6, time_us_cut.max()))

    ax_h.set_title(
        f"Freq {freq} ({FREQ_KHZ[freq]} kHz) — H & B (0-{T_MAX*1e6:.0f}us)  |  row {global_idx}",
        fontsize=11, fontweight="bold"
    )

    lines = [l1, l2, l3]
    ax_h.legend(lines, [ln.get_label() for ln in lines], fontsize=11, loc="best")
    ax_h.grid(True, alpha=0.35)
    fig.tight_layout()
    fname = f"freq{freq}_H_B_waveform_0-60us.png"
    fig.savefig(os.path.join(PLOT_DIR_FREQ, fname), dpi=200)
    plt.close(fig)
    print(f"  Saved: {fname}\n")

print(f"All plots saved in: {PLOT_DIR_FREQ}")