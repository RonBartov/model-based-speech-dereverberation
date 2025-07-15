#!/usr/bin/env python
"""
batch_auto_bench.py
-------------------
Run a short benchmark (100 train_on_batch updates) for several batch sizes.
The script:

1. Loads your full config file (`shoebox_c2.json`).
2. Overrides only the fields needed for a quick memory–speed test:
      • batch_size
      • epochs = 1
      • no weight/state/audio I/O
3. Measures seconds / mini-batch and writes:
      batch, sec_per_step, sec_per_sample
   to `batch_bench.csv`.

Edit BATCHES to the values you want to test.
"""

import csv
import json
import os
import time
import tensorflow as tf
from bsd_td import bssd       # your model class

# ---------- batch sizes you want to profile -------------------------------
BATCHES = [32, 64, 96, 128, 192, 256, 320]     # edit as needed
STEPS   = 100                                  # mini-batches per test
CONFIG  = "quick_mem.json"      # ← use the lightweight version
CSV_OUT = "batch_bench.csv"                    # output file
# --------------------------------------------------------------------------

# Load the full config once and reuse
base_cfg = json.load(open(CONFIG))

def run_one(batch: int) -> float:
    """Run STEPS train_on_batch calls and return seconds per step."""
    cfg = base_cfg.copy()
    cfg.update(
        batch_size=batch,
        epochs=1,
        is_load_weights=False,
        is_save_weights=False,
    )
    model = bssd(cfg)
    spk_per_batch = getattr(model, "speakers_per_batch", 20)

    # Pre-generate a single batch to avoid data-generator overhead
    sid = model.fgen.generate_triplet_indices(speakers=spk_per_batch,
                                              utterances_per_speaker=3)
    z, r, pid, sid = model.fgen.generate_multichannel_mixtures(model.nsrc, sid)

    start = time.time()
    for _ in range(STEPS):
        model.model.train_on_batch([z, r, pid[:, 0], sid[:, 0]], None)
    sec_per_step = (time.time() - start) / STEPS
    print(f"batch {batch:>4} -> {sec_per_step:.4f} s/step")
    return sec_per_step

# Create / append to CSV
new_file = not os.path.exists(CSV_OUT)
with open(CSV_OUT, "a", newline="") as f:
    writer = csv.writer(f)
    if new_file:
        writer.writerow(["batch", "sec_per_step", "sec_per_sample"])

    for b in BATCHES:
        sps = run_one(b)
        writer.writerow([b, f"{sps:.6f}", f"{sps / b:.6f}"])

print(f"\nResults written to {CSV_OUT}")
