#!/usr/bin/env python3
"""
Plot validation WPE and SI-SDR curves from a CSV produced by bsd_td.py
Usage:   python plot_val_curve.py /path/to/val_curve_crash_backup.csv
"""

import sys, os
import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    print("Usage: python plot_training_curve.py <csv_file>")
    sys.exit(1)

csv_file = sys.argv[1]
df = pd.read_csv(csv_file)

# Expect columns: epoch, val_wpe, val_si_sdr  (case-sensitive)
fig, axes = plt.subplots(2, 1, figsize=(6, 8))

# subplot 1 – Validation STOI                ########### to replace with STOI
axes[0].plot(df["epoch"], df["val_stoi_bsd"])
axes[0].set_xlabel("epoch")
axes[0].set_ylabel("Validation STOI")
axes[0].set_title("STOI vs #ephoch")

# --- subplot 2: SI-SDR ---
axes[1].plot(df["epoch"], df["val_si_sdr"])
axes[1].set_xlabel("epoch")
axes[1].set_ylabel("Validation SI-SDR (dB)")
axes[1].set_title("SI-SDR vs #ephoch")

plt.tight_layout()

png_file = os.path.splitext(csv_file)[0] + ".png"
plt.savefig(png_file)
print(f"Saved plot → {png_file}")
plt.show()          # comment out if running on headless server
