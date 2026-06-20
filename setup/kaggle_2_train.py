"""
================================================================================
KAGGLE NOTEBOOK 2 — TRAINING + EVAL  (Accelerator: GPU T4,  Internet: ON)
================================================================================
PURPOSE
    Run ONE grid cell end to end: train a QLoRA adapter on a task, then measure
        • forgetting  — did the model lose general knowledge?  (evaluate_forgetting.py)
        • task skill  — did it learn the task, and how much did fine-tuning ADD?
                        (evaluate_task.py, vs the base model's own task score)
    Base model = unsloth/Llama-3.2-3B. (Gemma-2 was dropped — fp16 corruption on
    T4; see results/failed-gemma/.)

HOW THIS NOTEBOOK WORKS
    Cells 1-4 = setup (always run). Cell 5 = CONFIG: set QUANT / REPLAY / TASK once
    and every command below fills itself in (so the run name and the adapter path
    can never drift apart). One Kaggle commit = one grid cell.

WHICH CELLS TO RUN FOR A COMMIT
    • First run at a NEW quant level (e.g. first 4-bit run):
        1,2,3,4, 5(config), 7(general baseline), 8(task baseline), 9,10,11, 12
    • Later run at a quant you've ALREADY baselined AND pushed baselines.json for:
        1,2,3,4, 5(config), 9,10,11, 12   (skip 7-8; the cloned repo already has them)
    • Cell 6 (sanity) is OPTIONAL — only to debug the training loop. DELETE it for
      a real commit so "Run All" doesn't waste time on it.

BASELINES (read this once)
    - General-knowledge baseline (cell 7) is per-QUANT and is SAVED to
      results/baselines.json. Cell 10 READS it to compute the Forgetting Score, so
      it MUST exist. Re-cloning wipes /kaggle/working, so after the first run at a
      quant, push baselines.json to the repo — then later same-quant commits clone
      it and can skip cell 7.
    - Task baseline (cell 8) is per-QUANT+TASK, base model with NO adapter. It's
      informational (a CSV row showing the base score) — it doesn't gate anything,
      so run it once per quant+task and keep the row.

EVAL DEPTH (the "slow down, do it right" settings)
    MMLU: --limit 100  (per subject × 57 ≈ 5,700 Qs, reliable).
    hellaswag/winogrande/arc: --limit_general 1000  (kills small-sample noise).
    Keep these the SAME on baselines AND fine-tuned runs or the numbers don't compare.

LONG RUNS + PERSISTENCE
    Use Save Version -> "Save & Run All (Commit)" (GPU + Internet on, ~12h wall).
    After it finishes, download results/all_results.csv, results/task_results.csv,
    results/baselines.json from the version Output, drop them into the repo, push.
================================================================================
"""

# ───────────────────────────── CELL 1 — get the code ─────────────────────────
!rm -rf qlora-forgetting-replay
!git clone https://github.com/AnsariUsaid/qlora-forgetting-replay.git
%cd qlora-forgetting-replay


# ────────────────────────── CELL 2 — install the stack (~4 min) ───────────────
# Deps inlined (no requirements.txt). Versions loose; pin from a working pip freeze.
!pip install -q unsloth transformers peft trl bitsandbytes accelerate datasets
!pip install -q lm-eval evaluate rouge-score sacrebleu scikit-learn pandas wandb


# ───────────────────── CELL 3 — log in to Weights & Biases ────────────────────
# Needs the Kaggle Secret WANDB_API_KEY enabled for THIS notebook.
from kaggle_secrets import UserSecretsClient
import wandb
wandb.login(key=UserSecretsClient().get_secret("WANDB_API_KEY"))


# ──────────────── CELL 4 — (re)build data/ cloud-side (~1 min) ────────────────
!python data/download_datasets.py


# ════════════════════════════════════════════════════════════════════════════
# CELL 5 — CONFIG  ◀── the ONLY cell you edit between runs. Set 3 values.
# ════════════════════════════════════════════════════════════════════════════
QUANT  = 4          # 4 | 8 | 16
REPLAY = 0.0        # 0.0 | 0.05 | 0.10 | 0.20 | 0.30
TASK   = "medqa"    # medqa | samsum

RUN = f"llama3b_{QUANT}bit_replay{int(REPLAY*100)}pct_{TASK}"
print("This run  :", RUN)
print("Adapter   :", f"outputs/{RUN}/adapter")
print("Eval depth: MMLU --limit 100 | general --limit_general 1000")


# ───── CELL 6 — OPTIONAL sanity (tiny: 500 samples, 1 epoch). DELETE for a commit ─────
# Only to confirm the training loop works. Watch the loss DROP below ~1 (Llama is
# healthy; a stuck ~20 would mean trouble). Overwrites the real adapter path, so
# never keep this in the same commit as cell 9.
!python train.py --quant {QUANT} --replay {REPLAY} --task {TASK} --max_samples 500 --epochs 1


# ════════════════════════ BASELINES (base model, NO adapter) ════════════════════════

# ── CELL 7 — GENERAL-KNOWLEDGE baseline — run ONCE per QUANT, then push baselines.json ──
# Saves MMLU/hellaswag/winogrande/arc for this quant as the reference for FS.
!python evaluate_forgetting.py --quant {QUANT} --run baseline_{QUANT}bit --batch_size 1 --limit 100 --limit_general 1000

# ── CELL 8 — TASK baseline (base model on the task) — run ONCE per QUANT+TASK ──
# How well the UN-fine-tuned model already does the task → fine-tuning's lift =
# (cell 11 score − this). MedQA: letter-logit accuracy/F1. Samsum: ROUGE-L.
!python evaluate_task.py --task {TASK} --quant {QUANT} --run base_{QUANT}bit_{TASK} --limit 100


# ════════════════════════════════ THE RUN ════════════════════════════════════

# ───────────────── CELL 9 — TRAIN the adapter (~3h for a full task) ───────────
# 4-bit has GPU headroom → batch 8 for speed. Add --batch_size 1 if 16-bit OOMs.
!python train.py --quant {QUANT} --replay {REPLAY} --task {TASK} --batch_size 8

# ──────────── CELL 10 — FORGETTING of the fine-tuned adapter (reads baseline -> FS) ──
!python evaluate_forgetting.py --quant {QUANT} --run {RUN} --adapter outputs/{RUN}/adapter --batch_size 1 --limit 100 --limit_general 1000

# ─────────────── CELL 11 — TASK performance of the fine-tuned adapter ─────────
!python evaluate_task.py --task {TASK} --quant {QUANT} --run {RUN} --adapter outputs/{RUN}/adapter --limit 100


# ─────────────── CELL 12 — show results (then download + push to repo) ────────
!echo "=== all_results.csv (forgetting) ===" ; cat results/all_results.csv
!echo "" ; echo "=== task_results.csv (task skill + base baseline) ===" ; cat results/task_results.csv
!echo "" ; echo "=== baselines.json (per-quant general baseline) ===" ; cat results/baselines.json
# Persist: download these 3 files from the version Output -> repo -> commit -> push.
