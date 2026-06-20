"""
================================================================================
KAGGLE NOTEBOOK 2 — TRAINING + EVAL  (Accelerator: GPU T4,  Internet: ON)
================================================================================
PURPOSE
    Install the stack, log in to W&B, (re)build data/, then train one QLoRA
    adapter and evaluate it two ways:
        • forgetting  (evaluate_forgetting.py) — did it lose general knowledge?
        • task        (evaluate_task.py)       — did it learn the task?
    Base model = unsloth/Llama-3.2-3B. (Gemma-2 was dropped: it needs bf16, the
    free T4 has none, so fp16 corrupted training — see results/failed-gemma/.)

HOW TO USE
    1. Create a Kaggle notebook, Accelerator = GPU T4, Internet = ON.
    2. Add the Kaggle Secret WANDB_API_KEY  (Add-ons -> Secrets) — Cell 3 needs it.
    3. Paste each CELL into its own notebook cell, run top to bottom.
       Order matters on a cold start.
    Cells 1-5 = setup + sanity. Cells 6-10 = one real run + its evals (the
    Llama re-validation: 4-bit / MedQA / 0% replay). Vary the flags for the campaign.

LONG RUNS
    Use Save Version -> "Save & Run All (Commit)" (GPU + Internet on): headless,
    ~12h wall per commit. Keep --limit ~100 on the evals so you don't hit the wall.

PERSISTENCE
    Re-cloning wipes /kaggle/working. After a run, copy results/all_results.csv,
    results/task_results.csv and results/baselines.json back into the repo and push.
    Adapters stay in the Kaggle version output (too big for git); curves stay in W&B.
================================================================================
"""

# ───────────────────────────── CELL 1 — get the code ─────────────────────────
!rm -rf qlora-forgetting-replay
!git clone https://github.com/AnsariUsaid/qlora-forgetting-replay.git
%cd qlora-forgetting-replay


# ────────────────────────── CELL 2 — install the stack (~4 min) ───────────────
# Deps are inlined here (the old setup/requirements.txt was removed so this file
# is self-contained). Versions are loose; to lock the env, run `pip freeze` in a
# WORKING session and pin the lines below.
!pip install -q unsloth transformers peft trl bitsandbytes accelerate datasets
!pip install -q lm-eval evaluate rouge-score sacrebleu scikit-learn pandas wandb


# ───────────────────── CELL 3 — log in to Weights & Biases ────────────────────
# Needs the Kaggle Secret WANDB_API_KEY enabled for THIS notebook.
from kaggle_secrets import UserSecretsClient
import wandb
wandb.login(key=UserSecretsClient().get_secret("WANDB_API_KEY"))


# ──────────────── CELL 4 — (re)build data/ cloud-side (~1 min) ────────────────
# Later this can be replaced by attaching Notebook 1's saved output read-only.
!python data/download_datasets.py


# ───────────────── CELL 5 — SANITY run (tiny: 500 samples, 1 epoch) ───────────
# Confirms the loop works end to end. WATCH THE LOSS: Llama-3.2-3B starts well
# below 1 and descends. If it sits ~20+ (above the random ceiling), something is
# wrong — stop and investigate before burning a full run.
!python train.py --quant 4 --replay 0.0 --task medqa --max_samples 500 --epochs 1


# ════════════════════════════════════════════════════════════════════════════
#   REAL RUN — Llama re-validation: 4-bit / MedQA / 0% replay
#   Campaign knobs:  --quant 4|8|16   --replay 0.0 0.05 0.10 0.20 0.30   --task medqa|samsum
#   If you hit CUDA OOM (most likely on 16-bit): add  --batch_size 1
# ════════════════════════════════════════════════════════════════════════════

# ───────── CELL 6 — TRAIN (full: 3 epochs default; batch up for 4-bit speed) ──
!python train.py --quant 4 --replay 0.0 --task medqa --batch_size 8


# ───────── CELL 7 — FORGETTING baseline (per quant, ONCE, NO adapter) ─────────
# No Llama baselines exist yet — run this once per quant level before its
# fine-tuned runs. --limit 100 keeps it ~30 min (full MMLU ≈ 3h). batch_size 1
# to stay safe on GPU memory.
!python evaluate_forgetting.py --quant 4 --run baseline_4bit --batch_size 1 --limit 100


# ─────── CELL 8 — FORGETTING of the fine-tuned adapter (reads baseline -> FS) ──
!python evaluate_forgetting.py --quant 4 --run llama3b_4bit_replay0pct_medqa \
    --adapter outputs/llama3b_4bit_replay0pct_medqa/adapter --batch_size 1 --limit 100


# ──────────────── CELL 9 — TASK performance (did it learn MedQA?) ─────────────
# Uses the new letter-logit scoring. CONFIRM: predictions spread across A/B/C/D
# (not always one letter) and accuracy is real (well above 25% chance).
!python evaluate_task.py --task medqa --quant 4 --run llama3b_4bit_replay0pct_medqa \
    --adapter outputs/llama3b_4bit_replay0pct_medqa/adapter --limit 100


# ─────────────── CELL 10 — show results (then copy out + push) ────────────────
!cat results/all_results.csv
!echo "---"
!cat results/baselines.json
# Copy results/*.csv + baselines.json back into the repo and push from your laptop.
