# Run logs

Links to the full logs/artifacts for each run (numbers live in `all_results.csv`).

> Earlier Gemma-2 runs are archived in `failed-gemma/` — they were invalid
> (fp16 corruption on T4; see `failed-gemma/WHY_FAILED.md`). The study now uses
> `unsloth/Llama-3.2-3B`. Logs below are Llama runs only.

| Run | Kaggle log (console + adapter) | W&B (training) |
|---|---|---|
| llama3b · 4-bit · MedQA · 0% | _(add link)_ | _(add link)_ |
| llama3b · 4-bit · Samsum · 0% | _(add link)_ | _(add link)_ |

## Notes / caveats
- **Eval depth:** MMLU at `--limit 100` (per-subject ≈ 5,700 Qs); hellaswag/winogrande/
  arc at `--limit_general 1000`. Keep identical across all runs + baselines.
- **MedQA 4-bit secondaries are at the OLD `--limit_general 100`** (measured before the
  noise fix). Its MMLU and Forgetting Score (−1.32%) are valid (MMLU was always limit-100),
  but its hellaswag/winogrande/arc values are noisier than the rest. TODO: re-score the
  MedQA adapter at limit_general 1000 + add its task baseline (`base_4bit_medqa`).
- **Forgetting is task-dependent:** Samsum (no MMLU overlap) → FS +4.32% (real forgetting).
  MedQA (overlaps MMLU's medical subjects) → FS −1.32% (MMLU nudged up). Samsum is the
  clean forgetting probe.
- **Task lift (base → fine-tuned):** Samsum ROUGE-L 17.66 → 45.03 (+27.4).
