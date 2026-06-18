"""
Measure catastrophic forgetting with lm-evaluation-harness (report §8.4).

What "forgetting" means here: how much general capability the model loses from
fine-tuning. We compare the model's MMLU score BEFORE vs AFTER fine-tuning, at
the SAME quantization level (quantization itself costs some capability, so the
baseline must be measured per quant level — that's why there are 3 baselines).

    Forgetting Score (FS) = (MMLU_baseline − MMLU_finetuned) / MMLU_baseline × 100

Two modes (chosen by whether --adapter is given):
  • baseline  (no --adapter): eval the un-fine-tuned base model at this quant
                level, SAVE its scores as the reference for this quant level.
  • finetuned (with --adapter): eval base+adapter, READ the matching baseline,
                compute FS, append everything to the results CSV.

Fixes over starter-asis:
  B3 real argparse/__main__ so the script actually runs when called.
  B4 pretrained=<base model>, peft=<adapter> (was: both pointed at the adapter).
  B5 CSV header write no longer crashes when the file doesn't exist yet.
  C2 MMLU at 5-shot, the rest at 0-shot — two eval calls (one call can't mix).
  C5 baseline scores are saved and read back, so FS can actually be computed.
  D3 use lm-eval's Python API (returns scores directly) instead of parsing a
     results file whose name/location changes between versions; read metric keys
     defensively.
"""
import os, json, csv, argparse

# General-capability probes. MMLU is the headline (world knowledge); the other
# three are commonsense/reasoning. Each maps to the metric key lm-eval reports.
ZERO_SHOT_TASKS = {
    "hellaswag":  "acc_norm,none",
    "winogrande": "acc,none",
    "arc_easy":   "acc_norm,none",
}
MMLU_METRIC = "acc,none"


def build_model_args(quant_bits: int, base_model: str, adapter: str = None) -> str:
    """lm-eval model_args string. Loads the base model at the requested bit-width
    (matching training), with the LoRA adapter on top if one is given."""
    parts = [f"pretrained={base_model}", "dtype=float16"]   # float16 = T4-safe
    if quant_bits == 4:
        parts.append("load_in_4bit=True")
    elif quant_bits == 8:
        parts.append("load_in_8bit=True")
    if adapter:
        parts.append(f"peft={adapter}")
    return ",".join(parts)


def _get_metric(results: dict, task: str, preferred: str) -> float:
    """Read a task's accuracy defensively — metric key names drift across
    lm-eval versions, so fall back to any 'acc*' key if the preferred one moves."""
    d = results["results"][task]
    if preferred in d:
        return d[preferred]
    for k, v in d.items():
        if k.startswith("acc") and isinstance(v, (int, float)):
            return v
    raise KeyError(f"No accuracy metric found for task '{task}' in {list(d)}")


def _run_eval(model_args: str, tasks, num_fewshot: int, batch_size: int, limit):
    """One lm-eval pass over `tasks`; returns the raw results dict."""
    import lm_eval
    out = lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=list(tasks),
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        limit=limit,          # None = full; a number = quick sanity subset
    )
    return out


def evaluate_forgetting(quant_bits: int, run_name: str, csv_path: str,
                        base_model: str, adapter: str = None,
                        baseline_file: str = "results/baselines.json",
                        batch_size: int = 4, limit=None) -> dict:
    is_baseline = adapter is None
    model_args = build_model_args(quant_bits, base_model, adapter)

    # C2: MMLU is 5-shot; the commonsense tasks are 0-shot → two separate calls.
    mmlu_res  = _run_eval(model_args, ["mmlu"], 5, batch_size, limit)
    other_res = _run_eval(model_args, list(ZERO_SHOT_TASKS), 0, batch_size, limit)

    scores = {"mmlu": _get_metric(mmlu_res, "mmlu", MMLU_METRIC)}
    for task, metric in ZERO_SHOT_TASKS.items():
        scores[task] = _get_metric(other_res, task, metric)
    scores["avg_general"] = sum(scores.values()) / len(scores)

    # As percentages, rounded.
    pct = {k: round(v * 100, 2) for k, v in scores.items()}

    if is_baseline:
        # C5: persist this quant level's reference scores for later FS math.
        os.makedirs(os.path.dirname(baseline_file) or ".", exist_ok=True)
        store = {}
        if os.path.exists(baseline_file):
            with open(baseline_file) as f:
                store = json.load(f)
        store[str(quant_bits)] = pct
        with open(baseline_file, "w") as f:
            json.dump(store, f, indent=2)
        fs = 0.0
        print(f"[Baseline {quant_bits}-bit] MMLU {pct['mmlu']}% saved as reference.")
    else:
        # C5: read the matching baseline and compute the Forgetting Score.
        with open(baseline_file) as f:
            store = json.load(f)
        if str(quant_bits) not in store:
            raise RuntimeError(
                f"No {quant_bits}-bit baseline in {baseline_file}. "
                f"Run the baseline (no --adapter) for {quant_bits}-bit first.")
        baseline_mmlu = store[str(quant_bits)]["mmlu"]
        fs = compute_forgetting_score(baseline_mmlu, pct["mmlu"])
        print(f"[Forgetting] {run_name} — MMLU {pct['mmlu']}% "
              f"(baseline {baseline_mmlu}%) → FS {fs:.2f}%")

    row = {
        "run": run_name,
        "quant_bits": quant_bits,
        "mode": "baseline" if is_baseline else "finetuned",
        "mmlu": pct["mmlu"],
        "hellaswag": pct["hellaswag"],
        "winogrande": pct["winogrande"],
        "arc_easy": pct["arc_easy"],
        "avg_general": pct["avg_general"],
        "forgetting_score": round(fs, 2),
    }

    # B5: only write the header when the file is new/empty (don't getsize a
    # file that doesn't exist yet).
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    file_ready = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_ready:
            writer.writeheader()
        writer.writerow(row)

    return row


def compute_forgetting_score(baseline_mmlu: float, finetuned_mmlu: float) -> float:
    """FS = (before − after) / before × 100. Positive = capability was lost."""
    return (baseline_mmlu - finetuned_mmlu) / baseline_mmlu * 100


if __name__ == "__main__":                                     # B3
    p = argparse.ArgumentParser()
    p.add_argument("--quant", type=int, required=True, choices=[4, 8, 16])
    p.add_argument("--run", type=str, required=True, help="Name for the CSV row.")
    p.add_argument("--adapter", type=str, default=None,
                   help="Adapter path. OMIT for a baseline (no fine-tuning) eval.")
    p.add_argument("--base_model", type=str, default="unsloth/gemma-2-2b")
    p.add_argument("--csv", type=str, default="results/all_results.csv")
    p.add_argument("--baseline_file", type=str, default="results/baselines.json")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--limit", type=int, default=None,
                   help="Examples per task (e.g. 50) for a quick sanity eval.")
    args = p.parse_args()

    evaluate_forgetting(
        quant_bits=args.quant, run_name=args.run, csv_path=args.csv,
        base_model=args.base_model, adapter=args.adapter,
        baseline_file=args.baseline_file, batch_size=args.batch_size,
        limit=args.limit,
    )
