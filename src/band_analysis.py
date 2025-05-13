"""
Frequency Band Analysis for Motor Imagery EEG.

Maps CWT scales 1–50 to EEG frequency bands, then evaluates how much
discriminative information each band contributes by training logistic
probe classifiers on band-restricted CWT features.

Scale → frequency mapping (Morlet, fs_eff = 93.75 Hz):
  f ≈ (fc_morlet * fs_eff) / scale   where fc_morlet ≈ 0.8125 (pywt 'morl')

Bands:
  Delta  : 0.5–4   Hz
  Theta  : 4–8     Hz
  Alpha  : 8–13    Hz
  Beta   : 13–30   Hz
  Gamma  : 30+     Hz

Does NOT modify any existing metrics files.
Saves metrics/band_analysis.csv and outputs/figures/band_*.png + freq_scale_map.png

Usage
-----
  python src/band_analysis.py                   # all subjects
  python src/band_analysis.py --subjects 1 2    # specific subjects
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import pywt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CLASS_NAMES, N_CLASSES, METRICS_DIR, FIGURES_DIR, CWT_SCALES, RANDOM_SEED,
)
from preprocessing import preprocess_subject


def scales_to_freqs(scales: list, fs_eff: float = 93.75) -> np.ndarray:
    fc = pywt.central_frequency("morl")   # ≈ 0.8125
    return np.array([fc * fs_eff / s for s in scales])


FREQS = scales_to_freqs(CWT_SCALES) 

BANDS = {
    "Delta (0.5-4 Hz)": (0.5,  4.0),
    "Theta (4-8 Hz)":   (4.0,  8.0),
    "Alpha (8-13 Hz)":  (8.0, 13.0),
    "Beta (13-30 Hz)": (13.0, 30.0),
    "Gamma (30+ Hz)":  (30.0, 200.0),
}


def band_scale_indices(lo: float, hi: float) -> np.ndarray:
    return np.where((FREQS >= lo) & (FREQS < hi))[0]


def probe_band_accuracy(X: np.ndarray, y: np.ndarray,
                        scale_idx: np.ndarray, n_splits: int = 5) -> float:
    if len(scale_idx) == 0:
        return float("nan")
    X_band = X[:, scale_idx, :, :].mean(axis=(2, 3))   # (N, n_scales)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_band)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    accs = []
    for train_idx, val_idx in cv.split(X_scaled, y):
        clf = LogisticRegression(max_iter=500, random_state=RANDOM_SEED,
                                 C=1.0, solver="lbfgs")
        clf.fit(X_scaled[train_idx], y[train_idx])
        accs.append(accuracy_score(y[val_idx], clf.predict(X_scaled[val_idx])))
    return float(np.mean(accs))


def analyse_subject(subject: int) -> dict:
    print(f"  Subject {subject:02d} …", flush=True)
    X, y = preprocess_subject(subject, session="T")
    row: dict = {"subject": subject}
    for band_name, (lo, hi) in BANDS.items():
        idx = band_scale_indices(lo, hi)
        acc = probe_band_accuracy(X, y, idx)
        print(f"    {band_name:22s}  scales={len(idx):2d}  acc={acc:.3f}")
        row[band_name] = round(acc, 4)
    return row


def plot_band_accuracy_heatmap(df: pd.DataFrame) -> None:
    band_cols = [c for c in df.columns if c != "subject"]
    heat = df.set_index("subject")[band_cols]
    heat.index = [f"S{s:02d}" for s in heat.index]
    heat.columns = [c.split(" ")[0] for c in heat.columns]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(heat, annot=True, fmt=".2f", cmap="YlGn",
                vmin=0.25, vmax=1.0, linewidths=0.5, ax=ax)
    ax.set_title("Band Discriminability — Logistic Probe Accuracy\n(chance = 0.25)")
    ax.set_xlabel("Frequency Band"); ax.set_ylabel("Subject")
    fig.tight_layout()
    fig.savefig(str(FIGURES_DIR / "band_accuracy_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"  → band_accuracy_heatmap.png")


def plot_band_accuracy_lines(df: pd.DataFrame) -> None:
    band_cols = [c for c in df.columns if c != "subject"]
    short = [c.split(" ")[0] for c in band_cols]
    fig, ax = plt.subplots(figsize=(9, 5))
    palette = sns.color_palette("tab10", len(df))
    for i, (_, row) in enumerate(df.iterrows()):
        ax.plot(short, [row[b] for b in band_cols],
                marker="o", label=f"S{int(row['subject']):02d}",
                color=palette[i], alpha=0.8)
    ax.axhline(0.25, ls="--", color="gray", alpha=0.5, label="Chance")
    ax.set_xlabel("Frequency Band"); ax.set_ylabel("CV Accuracy (logistic probe)")
    ax.set_title("Frequency Band Discriminability per Subject")
    ax.legend(ncol=3, fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(FIGURES_DIR / "band_accuracy_lines.png"), dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"  → band_accuracy_lines.png")


def plot_mean_band_bar(df: pd.DataFrame) -> None:
    band_cols = [c for c in df.columns if c != "subject"]
    means = df[band_cols].mean()
    stds  = df[band_cols].std()
    short = [c.split(" ")[0] for c in band_cols]
    fig, ax = plt.subplots(figsize=(8, 4))
    palette = sns.color_palette("viridis", len(band_cols))
    ax.bar(short, means, yerr=stds, capsize=4, color=palette, alpha=0.85, ecolor="black")
    ax.axhline(0.25, ls="--", color="gray", alpha=0.6, label="Chance (25%)")
    ax.set_xlabel("Frequency Band")
    ax.set_ylabel("Mean CV Accuracy ± std (9 subjects)")
    ax.set_title("Average Discriminability by Frequency Band")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(FIGURES_DIR / "band_accuracy_mean.png"), dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"  → band_accuracy_mean.png")


def plot_freq_scale_map() -> None:
    fig, ax = plt.subplots(figsize=(10, 3))
    colors = {"Delta": "#3498DB", "Theta": "#2ECC71",
              "Alpha": "#E67E22", "Beta": "#9B59B6", "Gamma": "#E74C3C"}
    for band_name, (lo, hi) in BANDS.items():
        idx = band_scale_indices(lo, hi)
        label = band_name.split(" ")[0]
        if len(idx):
            ax.barh(0, len(idx), left=idx[0], height=0.5,
                    color=colors.get(label, "gray"),
                    label=f"{label} (scales {idx[0]+1}–{idx[-1]+1})",
                    alpha=0.8)
    ax.set_xlabel("CWT Scale Index (0-based)")
    ax.set_yticks([]); ax.set_title("CWT Scale → Frequency Band Mapping")
    ax.legend(loc="upper right", fontsize=8); ax.set_xlim(0, 50)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(FIGURES_DIR / "freq_scale_map.png"), dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"  → freq_scale_map.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Frequency band analysis")
    parser.add_argument("--subjects", type=int, nargs="+",
                        default=list(range(1, 10)))
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"  Frequency Band Analysis  │  {len(args.subjects)} subjects")
    print(f"\n  Scale → frequency mapping (fs_eff = 93.75 Hz):")
    for band_name, (lo, hi) in BANDS.items():
        idx = band_scale_indices(lo, hi)
        if len(idx):
            print(f"    {band_name:22s}  scales {idx[0]+1:2d}–{idx[-1]+1:2d}")
    print(f"{'='*55}\n")

    rows = []
    for s in args.subjects:
        row = analyse_subject(s)
        if row:
            rows.append(row)

    if not rows:
        print("  No data found."); return

    df = pd.DataFrame(rows)
    csv_path = METRICS_DIR / "band_analysis.csv"
    df.to_csv(str(csv_path), index=False)
    print(f"\n  Saved → {csv_path.name}")

    print(f"\n  Generating figures …")
    plot_freq_scale_map()
    plot_band_accuracy_heatmap(df)
    plot_band_accuracy_lines(df)
    plot_mean_band_bar(df)
    print(f"\n  Done.\n")


if __name__ == "__main__":
    main()
