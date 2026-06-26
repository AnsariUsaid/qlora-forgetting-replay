# Run logs

Links to the full logs/artifacts for each run (numbers live in `all_results.csv`).

> Earlier Gemma-2 runs are archived in `failed-gemma/` — they were invalid
> (fp16 corruption on T4; see `failed-gemma/WHY_FAILED.md`). The study now uses
> `unsloth/Llama-3.2-3B`. Logs below are Llama runs only.

Per-run numbers live in `all_results.csv` / `task_results.csv` (rebuilt by
`aggregate_results.py` from `results/runs/<run>/`). W&B project: `qlora-forgetting`
(user ansari-usaid). Kaggle/W&B per-run links not individually tracked — runs are
identified by run name. The full narrative + findings are in `SESSION_HANDOFF.md`.

## Progress snapshot (2026-06-24)
Samsum replay sweep — FS% by quant × replay:
| replay | 4-bit | 8-bit | 16-bit |
|---|---|---|---|
| 0% | 4.32 | 3.43 | 1.77 |
| 10% | 2.54 | 1.98 | 1.04 |
| 20% | 1.82 | 1.58 | 0.83 |
| 30% | 2.63 | 2.18 | 1.44 |

## SQL snapshot (2026-06-25) — trained @25k
Adopted 25k after 10k trained too short to forget (4-bit 0% FS ~1.95 = noise). FS% by quant:
| replay | 4-bit | 8-bit | 16-bit |
|---|---|---|---|
| 0% | 2.88 | — | _(stale 10k 2.08 — redo @25k)_ |
Task exact-match: base 5.6 / 8.2 (4 / 16-bit) → tuned ~86–88. SQL task eval at `--limit 500`.
Critical findings are in `OBSERVATIONS.md` (repo root).

## Notes / caveats
- **Eval depth:** MMLU at `--limit 100` (per-subject ≈ 5,700 Qs); hellaswag/winogrande/
  arc at `--limit_general 1000`. Keep identical across all runs + baselines.
- **U-shape (optimal ≈ 20% replay):** forgetting minimizes at 20%, rises at 30% for ALL
  THREE quant levels (4-bit 1.82→2.63, 8-bit 1.58→2.18, 16-bit 0.83→1.44) — replicated
  3/3 (FS + avg_general). Likely real; still confounded by replay adding training steps
  (higher replay = more steps). Disentangle with seed-repeats + a fixed-total-steps control.
- **Forgetting is task-dependent:** Samsum (no MMLU overlap) → real forgetting. MedQA
  (overlaps MMLU's medical subjects) → FS −1.32% (MMLU nudged up). Samsum is the clean probe.
- **MedQA 4-bit secondaries are at the OLD `--limit_general 100`** (MMLU/FS still valid).
  TODO: `base_4bit_medqa` task baseline + optional re-eval at 1000.
- **Task lift (base → fine-tuned), Samsum ROUGE-L:** 4-bit 17.66→45.03, 8-bit 15.30→45.63,
  16-bit 15.28→46.57.
