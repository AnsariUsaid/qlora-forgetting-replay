# QLoRA-Forgetting-Replay — Project Roadmap

**Goal (one sentence):** prove that lower-bit quantization makes a model forget
more general knowledge during fine-tuning, that mixing in "replay" data fixes it,
and — the novel contribution (H1) — that 4-bit needs *more* replay than 8-bit to
recover the same capability.

## The experiment grid (33 runs)
- 3 quant levels (FP16, INT8, NF4 4-bit) × 5 replay ratios (0, 5, 10, 20, 30%)
  × 2 tasks (MedQA, Samsum) = **30 runs**
- \+ 3 baseline evals (pre-fine-tune reference) = **33 total**
- \+ 6 ablation runs later (Alpaca vs Dolly replay source)

## Datasets and their roles
| Dataset | HF source (Parquet, datasets-v4 safe) | Role | Train rows |
|---|---|---|---|
| MedQA  | `GBaker/MedQA-USMLE-4-options` | task (fine-tune target) | 10,178 |
| Samsum | `knkarthick/samsum` | task (fine-tune target) | 14,731 |
| Alpaca | `yahma/alpaca-cleaned` | replay buffer (fights forgetting) | 51,760 |
| Dolly  | `databricks/databricks-dolly-15k` | alternate replay buffer (C4 ablation) | 15,011 |

> Note: report's original `bigbio/med_qa` + `Samsung/samsum` are loading-script
> datasets that the new `datasets` v4 refuses — swapped for the Parquet mirrors above.

All four are normalized to `{instruction, input, output}` so any can be mixed.
**Mixing only ever happens as: one task + a slice of replay buffer.** Never task+task.

---

## Phases (one by one)

### [x] Phase 0 — Setup
- Kaggle env + GPU + W&B verified.
- GitHub repo created, public.

### [x] Phase 1 — Code in place
- Starter code §8.1-8.5 committed, tagged `starter-asis`.
- `download_datasets.py` fixed (B1/B2/C3/D5), committed, pushed.

### [x] Phase 2 — Get the data
- Ran the Kaggle cells (DataSets-QLora notebook): download + format all 4 datasets. ✅
- Verified counts live: MedQA 10,178 · Samsum 14,731 · Alpaca 51,760 · Dolly 15,011. ✅
- B1/C3 schema confirmed (MedQA hybrid output "D) Nitrofurantoin"). ✅
- Data saved via Quick Save (notebook output Version). A standalone Kaggle Dataset
  is optional/later; the real 30 runs will attach saved data instead of re-downloading.

### [x] Phase 3 — Fix training-path code (core done)
1. `build_replay_dataset.py` — 0%-replay-unshuffled fixed. ✅ committed
2. `train.py` — C1 (FP16/INT8/NF4 from one base) ✅, 8-bit verify ✅, C4 (GPU-mem) ✅,
   D1 (prompt) ✅, T4-safe dtype ✅, sanity flags (--max_samples/--epochs) ✅. committed
   - Deferred: B6 run-name ↔ `run_experiment.sh` alignment → Phase 7.
   - Deferred: D2 trl version pin → goes in the notebook install cell (Phase 4 setup).
3. Standardize adapter save/load paths (train ↔ eval) → Phase 5 (needs eval code).

### [x] Phase 4 — Sanity check (one tiny run)
- 4-bit, 0% replay, MedQA, `--max_samples 500 --epochs 1`. PASSED ✅: quant check ✓,
  loss dropped 27→21, W&B logged, adapter saved, peak GPU 3430 MB.
- Surfaced + fixed D2: migrated SFTTrainer to the new trl API (SFTConfig, max_length,
  processing_class) — the old API silently dropped batch size → CUDA OOM on T4.

### [x] Phase 5 — Fix evaluation code
- `evaluate_forgetting.py` rewritten ✅ — argparse (B3), base-vs-adapter load (B4),
  CSV first-write (B5), per-task few-shot (C2), baseline wiring (C5), lm-eval API (D3).
  GPU fixes: build `BitsAndBytesConfig` ourselves (transformers v5 dropped `load_in_4bit`),
  load model once, `--batch_size 1` (Gemma-2's 256k-vocab logits OOM otherwise).
- `evaluate_task.py` written from scratch ✅ — MedQA accuracy + macro-F1, Samsum ROUGE-L,
  on the held-out test split. ⚠️ Known bug: crashes on >512-token prompts (Unsloth
  fast-inference path) — length fix pending.

### [ ] Phase 6 — Baseline forgetting (report "Week 2", most important result)  ← CURRENT
- Run the 6 zero-replay runs (3 quant × 2 tasks, replay=0%).
- Eval base model (pre-fine-tune) on MMLU/HellaSwag/ARC = reference point.
- Compute Forgetting Score each. Expected: 4-bit FS > 8-bit FS > FP16 FS.
- **Progress (2026-06-19):** 4-bit baseline ✅ (MMLU 49.97, saved to `baselines.json`).
  Run 001 (4-bit MedQA 0%) training + forgetting ✅ → **FS 54.07** (general caps
  collapsed to ~random across all 4 benchmarks). Task-eval pending the length fix.
  Remaining: 2 baselines (8/16-bit) + 5 more zero-replay runs + their task scores.
- Results live in `results/all_results.csv`, `results/baselines.json`, `results/LOGS.md`.
- Speed lever for the campaign: add `--limit ~100` to the evals (~3 hrs → ~30 min each).

### [ ] Phase 7 — Replay experiments (report "Week 3")
- Run remaining 24 runs (3 quant × 4 replay ratios × 2 tasks).
- Split across Kaggle accounts for GPU quota.

### [ ] Phase 8 — Analysis + ablations (report "Week 4")
- Aggregate to one CSV. Plot FS vs replay ratio (3 curves).
- Run 6 ablation runs (Alpaca vs Dolly).
- Find the "crossover point" (replay ratio where 4-bit FS == 8-bit FS) = headline number.

### [ ] Phase 9 — Paper (Weeks 5-8)
- Results, discussion, "practical recipe" table. Submit (IEEE Access / EMNLP Findings).

---

## Metrics per run (6)
Forgetting Score (primary) · MMLU accuracy · Task performance (F1/ROUGE-L) ·
Tradeoff Index · Peak GPU memory · Training time.

**Forgetting Score** = (MMLU_before − MMLU_after) / MMLU_before × 100

---

## Bug/fix tracker (from the starter-code audit)
Severity: 🔴 blocker (crashes) · 🟠 correctness (wrong results) · 🟡 design/version.

| ID | File | Issue | Status |
|----|------|-------|--------|
| B1 | download_datasets.py | script-based MedQA/Samsum → Parquet mirrors (datasets v4) | ✅ fixed |
| B2 | download_datasets.py | missing `os.makedirs("data")` | ✅ fixed |
| C3 | download_datasets.py | Dolly downloaded but never saved | ✅ fixed |
| D5 | download_datasets.py | MedQA hybrid output "A) text" | ✅ fixed |
| — | build_replay_dataset.py | 0%-replay branch returns unshuffled | ✅ fixed |
| C1 | train.py | prequantized base; FP16/INT8 impossible | ✅ fixed |
| 8b | train.py | verify model is really at requested bits (Unsloth #2679) | ✅ added |
| C4 | train.py | no peak-GPU-memory logging | ✅ fixed |
| D1 | train.py | prompt template drift vs download | ✅ fixed |
| B6 | train.py / run_experiment.sh | adapter path/name mismatch | ⬜ Phase 7 (shell side) |
| D2 | train.py | trl API drift → migrated to SFTConfig/max_length/processing_class | ✅ fixed |
| B3 | evaluate_forgetting.py | no argparse/__main__ block | ✅ fixed |
| B4 | evaluate_forgetting.py | `pretrained=` must be base, `peft=` adapter | ✅ fixed |
| B5 | evaluate_forgetting.py | CSV first-write crashes (getsize) | ✅ fixed |
| C2 | evaluate_forgetting.py | global few-shot breaks 0-shot tasks | ✅ fixed |
| C5 | evaluate_forgetting.py | baseline MMLU never wired up | ✅ fixed |
| D3 | evaluate_forgetting.py | lm-eval keys/version not pinned | ✅ fixed |
| E1 | evaluate_forgetting.py | transformers v5 dropped `load_in_4bit` → build BitsAndBytesConfig; batch_size 1 (vocab-logit OOM) | ✅ fixed |
| — | evaluate_task.py | did not exist — written from scratch (F1 / ROUGE-L) | ✅ done |
| E2 | evaluate_task.py | crashes on >512-token prompts (Unsloth fast-inference) | ⬜ length fix pending |
| D4 | run_experiment.sh | 30 runs > Kaggle session wall; no resume/`set -e` | ⬜ Phase 7 |
