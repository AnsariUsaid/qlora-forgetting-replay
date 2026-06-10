#!/bin/bash
# Run all 30 experiments. Split this across 3 Kaggle accounts.

TASKS=("medqa" "samsum")
QUANT_LEVELS=(4 8 16)
REPLAY_RATIOS=(0.0 0.05 0.10 0.20 0.30)

for TASK in "${TASKS[@]}"; do
  for Q in "${QUANT_LEVELS[@]}"; do
    for R in "${REPLAY_RATIOS[@]}"; do
      echo "=== Running: task=$TASK quant=${Q}bit replay=${R} ==="
      python train.py --quant $Q --replay $R --task $TASK
      python evaluate_forgetting.py \
        --adapter "outputs/gemma2b_${Q}bit_replay${R}_${TASK}/adapter" \
        --run "gemma2b_${Q}bit_replay${R}_${TASK}" \
        --csv "results/all_results.csv"
    done
  done
done
echo "All experiments complete."
