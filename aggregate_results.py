"""
Rebuild the master result tables from the per-run folders.

Each run writes its own results/runs/<run>/forgetting.json and/or task.json
(by evaluate_forgetting.py / evaluate_task.py). This script is the ONLY thing
that writes the master CSVs, so they're always a faithful, de-duplicated view
of the per-run files and never need hand-merging. Re-running a run overwrites
its own folder; re-running this rebuilds the tables.

    python aggregate_results.py
"""
import os, json, csv, glob

RUNS_DIR    = "results/runs"
FORGET_CSV  = "results/all_results.csv"
TASK_CSV    = "results/task_results.csv"
BASELINES   = "results/baselines.json"

FORGET_COLS = ["run", "quant_bits", "mode", "mmlu", "hellaswag", "winogrande",
               "arc_easy", "avg_general", "forgetting_score"]
TASK_COLS   = ["run", "task", "quant_bits", "n_eval", "accuracy", "macro_f1", "rougeL", "exact_match"]
# Fields evaluate_forgetting.py reads back from baselines.json to compute FS.
BASELINE_FIELDS = ["mmlu", "hellaswag", "winogrande", "arc_easy", "avg_general"]


def _collect(filename):
    rows = []
    for path in sorted(glob.glob(os.path.join(RUNS_DIR, "*", filename))):
        with open(path) as f:
            rows.append(json.load(f))
    return rows


def _write_csv(path, cols, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def main():
    forget = _collect("forgetting.json")
    task   = _collect("task.json")
    # Readable order: baselines first, then by quant level, then run name.
    forget.sort(key=lambda r: (r.get("mode") != "baseline",
                               r.get("quant_bits", 0), r.get("run", "")))
    task.sort(key=lambda r: (r.get("quant_bits", 0),
                             r.get("task", ""), r.get("run", "")))
    _write_csv(FORGET_CSV, FORGET_COLS, forget)
    _write_csv(TASK_CSV, TASK_COLS, task)

    # Rebuild baselines.json from the baseline_* runs so it's always complete (one
    # entry per quant level). evaluate_forgetting.py reads this to compute FS, so a
    # missing entry would crash a fine-tuned run. Deriving it here removes that gap.
    store = {str(r["quant_bits"]): {k: r[k] for k in BASELINE_FIELDS}
             for r in forget if r.get("mode") == "baseline"}
    with open(BASELINES, "w") as f:
        json.dump(store, f, indent=2)

    print(f"Aggregated {len(forget)} forgetting rows -> {FORGET_CSV}")
    print(f"Aggregated {len(task)} task rows -> {TASK_CSV}")
    print(f"Rebuilt {len(store)} baselines ({', '.join(sorted(store))}-bit) -> {BASELINES}")


if __name__ == "__main__":
    main()
