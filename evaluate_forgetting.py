import subprocess, json, csv, os

def evaluate_forgetting(adapter_path: str, run_name: str, results_csv: str):
    """
    Uses lm-evaluation-harness to run MMLU (5-shot), HellaSwag,
    WinoGrande, and ARC-Easy. Appends results to a shared CSV.
    """
    tasks = "mmlu,hellaswag,winogrande,arc_easy"
    output_path = f"eval_outputs/{run_name}_forgetting.json"
    os.makedirs("eval_outputs", exist_ok=True)

    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={adapter_path},peft={adapter_path}",
        "--tasks", tasks,
        "--num_fewshot", "5",        # 5-shot for MMLU, 0-shot for rest
        "--batch_size", "4",
        "--output_path", output_path,
        "--device", "cuda"
    ]

    subprocess.run(cmd, check=True)

    with open(output_path) as f:
        results = json.load(f)

    mmlu_acc    = results["results"]["mmlu"]["acc,none"]
    hellaswag   = results["results"]["hellaswag"]["acc_norm,none"]
    winogrande  = results["results"]["winogrande"]["acc,none"]
    arc_easy    = results["results"]["arc_easy"]["acc_norm,none"]
    avg_general = (mmlu_acc + hellaswag + winogrande + arc_easy) / 4

    row = {
        "run": run_name,
        "mmlu_5shot": round(mmlu_acc * 100, 2),
        "hellaswag": round(hellaswag * 100, 2),
        "winogrande": round(winogrande * 100, 2),
        "arc_easy": round(arc_easy * 100, 2),
        "avg_general": round(avg_general * 100, 2)
    }

    with open(results_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if os.path.getsize(results_csv) == 0: writer.writeheader()
        writer.writerow(row)

    print(f"[Forgetting Eval] {run_name} — MMLU: {row['mmlu_5shot']}% | "
          f"Avg: {row['avg_general']}%")
    return row


def compute_forgetting_score(
    baseline_mmlu: float, finetuned_mmlu: float
) -> float:
    """FS = (before - after) / before * 100. Positive = forgot."""
    return (baseline_mmlu - finetuned_mmlu) / baseline_mmlu * 100
