# Observations

Brief, critical findings only — one takeaway each. Newest first. Full numbers live in
`results/`; full narrative in `SESSION_HANDOFF.md`.

## 2026-06-26

- **Quant ladder HOLDS for SQL — H1 replicated on a 2nd forgetting task.** At 25k, 4-bit 0%
  FS 2.88 > 16-bit 0% FS 1.56 (gap 1.32, well above ±0.6% noise) — same direction as Samsum
  (4.32 > 1.77). At 10k it was invisible/inverted (1.95 vs 2.08), so **enough training
  pressure is a PREREQUISITE for the quant effect to appear.** Task score is quant-independent
  (88.0 vs 87.8) → the cost of low-bit quant shows up in FORGETTING, not task ability.
- **SQL 0% ladder COMPLETE & monotonic: 4-bit 2.88 ≥ 8-bit 2.70 > 16-bit 1.56.** Same shape
  as Samsum. The 4-vs-8 gap (0.18) is within noise (as on Samsum); the real, clear separation
  is **quantized (4/8 ≈ 2.7–2.9) vs full-precision (16 ≈ 1.6)**. Task score quant-independent
  at all three (~88%). Next: SQL replay sweep to test if the U-shape replicates too.

## 2026-06-25

- **Forgetting scales with training COMPUTE, not just task type.** SQL 4-bit 0%: at 10k
  (~75 min) FS was 1.95 (noise); at 25k (~222 min, ≈Samsum's compute) FS rose to 2.88 (real
  signal). An "easy/short" task isn't immune to forgetting — it just needs enough training
  to show it. → Adopted 25k for the whole SQL sweep; 10k SQL runs retired.

- **Cross-task forgetting is compute-confounded → only compare at MATCHED compute.** At
  ~equal training time, SQL forgets LESS than Samsum (2.88 vs 4.32). So the task-type story
  is "structured/easy task (SQL) forgets less than open-ended (Samsum)" — the opposite of the
  early guess that SQL (more divergent) would forget more.

- **Never compare runs of different training sizes.** Mid-migration the CSV showed 4-bit SQL
  (25k, 2.88) > 16-bit SQL (10k, 2.08) — looks like the quant ladder, but it's a size
  artifact (4-bit trained 3× longer). The real ladder needs both quants at the same size.

- **Task metric is per-task; forgetting is the common currency.** SQL → exact-match,
  Samsum → ROUGE-L, MedQA → accuracy (different units, NOT cross-comparable). Compare tasks
  only via Forgetting Score (same MMLU-drop metric for all).

## Earlier (Samsum sweep + MedQA)

- **U-shape: optimal replay ≈ 20%, forgetting rises again at 30%** — replicated across all
  3 quant levels (4/8/16-bit), in FS and avg_general. Likely real, but confounded by replay
  ADDING data (more replay = more steps). Confirm with seed-repeats + a fixed-step control.

- **Quant ladder (4 > 8 > 16 forgets more) holds for Samsum** at every replay level tested —
  H1's backbone. Replay recovers a roughly constant FRACTION of forgetting at every bit-width.

- **MedQA is a poor forgetting probe** — it overlaps MMLU's medical subjects, so fine-tuning
  nudges MMLU UP (FS −1.32). Samsum is the clean probe. A per-subject MMLU split (medical vs
  rest) would expose the selective forgetting MedQA hides.
