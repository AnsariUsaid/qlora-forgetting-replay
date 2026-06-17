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
| Dataset | Role |
|---|---|
| MedQA  | task (fine-tune target) |
| Samsum | task (fine-tune target) |
| Alpaca | replay buffer (mixed into task runs to fight forgetting) |
| Dolly  | alternate replay buffer (C4 ablation: does replay source matter?) |

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

### [ ] Phase 2 — Get the data  ← CURRENT
- Run the Kaggle cells: download + format MedQA / Samsum / Alpaca / Dolly.
- Verify sample counts (MedQA ~12,723 · Samsum ~14,732 · Alpaca ~51,760 · Dolly ~15,011).
- Confirm B1/C3 schema assumptions held on the live data.
- Save `data/` as a Kaggle Dataset (no re-download ever).

### [ ] Phase 3 — Fix training-path code (before any training)
1. `build_replay_dataset.py` — fix 0%-replay-unshuffled bug (the task+replay mixer).
2. `train.py` — **C1 (the big one):** FP16/INT8/NF4 all load from one base model.
   Also: run-naming (B6), GPU-memory logging (C4), canonical prompt (D1), trl pin (D2).
3. Standardize adapter save/load paths so train ↔ eval agree.

### [ ] Phase 4 — Sanity check (one tiny run)
- 4-bit, 0% replay, MedQA, ~500 samples, 1 epoch.
- Confirm only: (a) train loss drops, (b) MMLU drops vs baseline, (c) W&B logs it.
- Gate before spending GPU hours on the full grid.

### [ ] Phase 5 — Fix evaluation code (before Phase 6)
- `evaluate_forgetting.py` — argparse (B3), base-vs-adapter load (B4),
  CSV first-write (B5), per-task few-shot (C2), baseline wiring (C5), lm-eval pin (D3).
- Write `evaluate_task.py` from scratch (F1 for MedQA, ROUGE-L for Samsum).

### [ ] Phase 6 — Baseline forgetting (report "Week 2", most important result)
- Run the 6 zero-replay runs (3 quant × 2 tasks, replay=0%).
- Eval base model (pre-fine-tune) on MMLU/HellaSwag/ARC = reference point.
- Compute Forgetting Score each. Expected: 4-bit FS > 8-bit FS > FP16 FS.

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
| B1 | download_datasets.py | MedQA `options` is a list, not dict | ✅ fixed |
| B2 | download_datasets.py | missing `os.makedirs("data")` | ✅ fixed |
| C3 | download_datasets.py | Dolly downloaded but never saved | ✅ fixed |
| D5 | download_datasets.py | MedQA hybrid output "A) text" | ✅ fixed |
| — | build_replay_dataset.py | 0%-replay branch returns unshuffled | ⬜ Phase 3 |
| C1 | train.py | prequantized base; FP16/INT8 impossible | ⬜ Phase 3 |
| B6 | train.py / run_experiment.sh | adapter path/name mismatch | ⬜ Phase 3 |
| C4 | train.py | no peak-GPU-memory logging | ⬜ Phase 3 |
| D1 | train.py | prompt template drift vs download | ⬜ Phase 3 |
| D2 | train.py | trl version not pinned (API drift) | ⬜ Phase 3 |
| B3 | evaluate_forgetting.py | no argparse/__main__ block | ⬜ Phase 5 |
| B4 | evaluate_forgetting.py | `pretrained=` must be base, `peft=` adapter | ⬜ Phase 5 |
| B5 | evaluate_forgetting.py | CSV first-write crashes (getsize) | ⬜ Phase 5 |
| C2 | evaluate_forgetting.py | global few-shot breaks 0-shot tasks | ⬜ Phase 5 |
| C5 | evaluate_forgetting.py | baseline MMLU never wired up | ⬜ Phase 5 |
| D3 | evaluate_forgetting.py | lm-eval keys/version not pinned | ⬜ Phase 5 |
| — | evaluate_task.py | does not exist — write from scratch | ⬜ Phase 5 |
| D4 | run_experiment.sh | 30 runs > Kaggle session wall; no resume/`set -e` | ⬜ Phase 7 |
