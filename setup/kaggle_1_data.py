"""
================================================================================
KAGGLE NOTEBOOK 1 — DATA  (Accelerator: None / CPU,  Internet: ON)
================================================================================
PURPOSE
    Download + format the four datasets (MedQA, Samsum, Alpaca, Dolly) ONCE into
    data/*.json, then Save a Version so the data persists as notebook output.
    The real training notebook can later attach that saved output read-only
    instead of re-downloading.

HOW TO USE
    1. Create a Kaggle notebook, Accelerator = None (CPU), Internet = ON.
    2. Paste each CELL below into its own notebook cell, in order.
    3. Run All, check the counts in the last cell, then Save Version.
    Re-run only when the dataset formatting (data/download_datasets.py) changes.

NOTE
    Stopping a Kaggle session wipes installed libs + /kaggle/working. Saved
    Versions/Datasets and W&B runs DO persist. This downloads cloud->cloud
    (Kaggle's network), so it does NOT use your laptop's data.
================================================================================
"""

# ───────────────────────────── CELL 1 — get the code ─────────────────────────
!rm -rf qlora-forgetting-replay
!git clone https://github.com/AnsariUsaid/qlora-forgetting-replay.git
%cd qlora-forgetting-replay


# ──────────────── CELL 2 — datasets library (Parquet sources need current) ────
!pip install -q -U datasets


# ───────────────── CELL 3 — download + format into data/ (~1 min) ─────────────
!python data/download_datasets.py


# ─────────────────────────── CELL 4 — verify counts ──────────────────────────
import json
for f in ["medqa_train", "samsum_train", "alpaca_cleaned", "dolly_15k"]:
    d = json.load(open(f"data/{f}.json"))
    print(f"{f:18s} {len(d):>7,} samples | keys: {list(d[0].keys())}")
# Expected: medqa_train ~10,178 · samsum_train 14,731 · alpaca_cleaned 51,760 · dolly_15k 15,011
# If the counts look right -> Save Version (data persists in the notebook output).
