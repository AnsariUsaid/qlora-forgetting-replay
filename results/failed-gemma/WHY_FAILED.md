# Why these Gemma-2 results are invalid (archived, do not use)

These files hold every result we produced on `unsloth/gemma-2-2b`. **All of the
numbers here are wrong** and must not be cited, plotted, or compared against the
Llama runs. They are kept only as a record of the dead end.

## Root cause: Gemma-2 + fp16 on a T4 corrupts training

Gemma-2 uses attention/final-logit **soft-capping** and produces large activations
that overflow fp16's ~65504 ceiling → inf/NaN → garbage gradients. Google/HF state
Gemma-2 must run in **bf16 or fp32**. The free Kaggle/Colab **T4 GPU has no bf16**
(training log prints `Bfloat16 = FALSE`), so `train.py` fell back to fp16 →
numerical corruption.

### The smoking gun
Training loss started at **~21–27**. Cross-entropy is capped at
ln(vocab) = ln(256000) ≈ **12.45** for a *uniform-random* model. A loss of 27 is
**worse than random** — only possible with corrupted logits, not undertraining.

### Controlled proof (20-step micro-train, identical data/config, only model/dtype differ)
- Gemma-2-2B fp16: loss **21.85 → 21.08** (flat, above the random ceiling) ❌ corrupted
- Llama-3.2-3B fp16: loss **0.77 → 0.14** (smooth healthy descent) ✅ fine

This explains every downstream symptom: MMLU collapsing to ~random, empty/newline
generation, and "always-one-letter" task predictions. None of it was the code,
recipe, or learning rate — it was the model × dtype × hardware combination.

Note: Gemma-2 *inference* of the base model on a T4 was fine (baseline MMLU ~50%);
only *training* corrupted. So the failure was always training-specific.

## What was wrong in each file
- `all_results.csv` — `baseline_4bit` (FS 0) and `gemma2b_4bit_replay0pct_medqa`
  (FS 54.07). The FS 54.07 looked like a dramatic forgetting result but is just
  the corrupted model collapsing to ~random on every benchmark.
- `baselines.json` — the 4-bit Gemma MMLU baseline (49.97). Valid as a base-model
  inference number, but useless because no valid fine-tuned Gemma run exists to
  compare it to.
- `LOGS.md` — links to the Kaggle/W&B logs for the corrupted production run.

## Resolution
Switched the whole study to `unsloth/Llama-3.2-3B` (no soft-capping → numerically
fine in fp16 on a free T4). Fresh results are collected in the parent `results/`
directory starting from the Llama re-validation run.
