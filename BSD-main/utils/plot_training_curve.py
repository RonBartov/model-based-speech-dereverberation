#!/usr/bin/env python3
"""
Plot validation STOI / SI-SDR  curves from a CSV produced by bsd_td.py

Usage examples
--------------
# default (no label in title)
python plot_training_curve.py <csv_file>

# custom label
python plot_training_curve.py --csv_file <csv_file> --label BSD
python plot_training_curve.py --csv_file <csv_file> --label BSSD
"""

import sys, os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
    description="Plot validation curves from a val_curve*.csv file")
parser.add_argument("--csv_file", help="path to CSV produced during training")
parser.add_argument("--label", default="BSD", help="model label to prefix titles")  # <-- NEW
args = parser.parse_args()

df = pd.read_csv(args.csv_file)

# optional prefix for all subplot titles
prefix = (args.label + ": ") if args.label else ""

fig, axes = plt.subplots(2, 1, figsize=(6, 8))
            
# subplot 1 – Validation STOI   
axes[0].plot(df["epoch"], df["train_stoi"], label="train")
axes[0].plot(df["epoch"], df["val_stoi_bsd"], label="valid")
axes[0].set_xlabel("epoch")
axes[0].set_ylabel("STOI")
axes[0].set_title(f"{prefix}Train vs Validation STOI")
axes[0].legend()

# subplot 2 – Validation SI-SDR
axes[1].plot(df["epoch"], df["train_si_sdr"], label="train")
axes[1].plot(df["epoch"], df["val_si_sdr"],   label="valid")        
axes[1].set_xlabel("epoch")
axes[1].set_ylabel("SI-SDR (dB)")
axes[1].set_title(f"{prefix}Train vs Validation SI-SDR")
axes[1].legend()

# # subplot 3 – Validation PESQ
# axes[2].plot(df["epoch"], df["train_pesq"], label="train")
# axes[2].plot(df["epoch"], df["val_pesq_bsd"],   label="valid")
# axes[2].set_xlabel("epoch")
# axes[2].set_ylabel("PESQ")
# axes[2].set_title(f"{prefix}Train vs Validation PESQ")
# axes[2].legend()

# axes[3].axis("off")
plt.tight_layout()

png_file = os.path.splitext(args.csv_file)[0] + ".png"
plt.savefig(png_file)
print(f"Saved plot → {png_file}")
plt.show()          # comment out if running on headless server