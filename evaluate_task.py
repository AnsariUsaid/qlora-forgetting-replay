"""
Measure TASK performance of a fine-tuned adapter (report §10 "Task Performance").

This answers "did the model actually learn the task?" — the complement to
evaluate_forgetting.py ("did it forget general knowledge?"). A good result needs
BOTH: high task score AND low forgetting.

  • MedQA  (multiple-choice): generate an answer, extract the chosen letter A-D,
            score accuracy + macro-F1 vs the gold letter.
  • Samsum (summarization):    generate a summary, score ROUGE-L vs the reference.

Evaluated on the HELD-OUT TEST split (data/{task}_test.json) — never the training
data, or the score would be meaningless (memorisation, not learning).

Loads the SAME base model at the SAME quant level as training, with the adapter
on top, so the measured model matches what was trained.
"""
import os, json, csv, re, argparse

# Inference context length. Set well above the training max (512) so long MedQA
# vignettes / Samsum dialogues are never trimmed into the model — an over-length
# prompt triggers an Unsloth fast-inference crash (bug E2). 2048 covers all prompts,
# and it's a ceiling (not a per-step cost), so it does not slow generation.
MAX_SEQ_LEN = 2048

# Inference prompt = the training prompt up to "### Response:" (the model fills
# in the rest). Must match train.py's ALPACA_PROMPT or the model sees a format
# it wasn't trained on.
INFER_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""


def load_model(quant_bits: int, adapter: str, base_model: str, max_seq_len=MAX_SEQ_LEN):
    """Load base+adapter at the requested bit-width, in inference mode."""
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter if adapter else base_model,   # adapter dir resolves its base
        max_seq_length=max_seq_len,
        dtype=None,
        load_in_4bit=(quant_bits == 4),
        load_in_8bit=(quant_bits == 8),
    )
    FastLanguageModel.for_inference(model)               # 2x faster generation
    return model, tokenizer


def generate(model, tokenizer, instruction: str, inp: str, max_new_tokens: int) -> str:
    import torch
    prompt = INFER_PROMPT.format(instruction=instruction, input=inp)
    # Hard-truncate so input + generated tokens stay within the context — prevents the
    # >max-length Unsloth crash (bug E2). Only fires on rare over-length prompts; the
    # common case is untouched.
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=MAX_SEQ_LEN - max_new_tokens).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False,                              # greedy = deterministic
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode only the newly generated part (skip the prompt tokens).
    gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                           skip_special_tokens=True)
    return gen.strip()


def _extract_letter(text: str) -> str:
    """First A-D in the text (the model emits 'A) ...'); '' if none found."""
    m = re.search(r"[ABCD]", text.upper())
    return m.group(0) if m else ""


def score_medqa(data, model, tokenizer) -> dict:
    from sklearn.metrics import accuracy_score, f1_score
    gold, pred = [], []
    for ex in data:
        gold.append(_extract_letter(ex["output"]))               # "A) ..." → "A"
        out = generate(model, tokenizer, ex["instruction"], ex["input"], 16)
        pred.append(_extract_letter(out) or "?")                 # ? = unparseable
    acc = accuracy_score(gold, pred)
    f1 = f1_score(gold, pred, average="macro", labels=list("ABCD"), zero_division=0)
    return {"accuracy": round(acc * 100, 2), "macro_f1": round(f1 * 100, 2)}


def score_samsum(data, model, tokenizer) -> dict:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    total = 0.0
    for ex in data:
        out = generate(model, tokenizer, ex["instruction"], ex["input"], 128)
        total += scorer.score(ex["output"], out)["rougeL"].fmeasure
    return {"rougeL": round(total / len(data) * 100, 2)}


def evaluate_task(task: str, quant_bits: int, run_name: str, csv_path: str,
                  base_model: str, adapter: str,
                  data_file: str = None, limit: int = None) -> dict:
    data_file = data_file or f"data/{task}_test.json"
    with open(data_file) as f:
        data = json.load(f)
    if limit:
        data = data[:limit]                              # quick sanity subset

    model, tokenizer = load_model(quant_bits, adapter, base_model)

    if task == "medqa":
        metrics = score_medqa(data, model, tokenizer)
    elif task == "samsum":
        metrics = score_samsum(data, model, tokenizer)
    else:
        raise ValueError(f"Unknown task '{task}' (expected medqa or samsum).")

    # Unified CSV schema across tasks (irrelevant columns left blank).
    row = {
        "run": run_name,
        "task": task,
        "quant_bits": quant_bits,
        "n_eval": len(data),
        "accuracy": metrics.get("accuracy", ""),
        "macro_f1": metrics.get("macro_f1", ""),
        "rougeL": metrics.get("rougeL", ""),
    }

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    file_ready = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_ready:
            writer.writeheader()
        writer.writerow(row)

    print(f"[Task Eval] {run_name} ({task}, n={len(data)}) → {metrics}")
    return row


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, required=True, choices=["medqa", "samsum"])
    p.add_argument("--quant", type=int, required=True, choices=[4, 8, 16])
    p.add_argument("--run", type=str, required=True, help="Name for the CSV row.")
    p.add_argument("--adapter", type=str, required=True, help="Trained adapter path.")
    p.add_argument("--base_model", type=str, default="unsloth/gemma-2-2b")
    p.add_argument("--csv", type=str, default="results/task_results.csv")
    p.add_argument("--data_file", type=str, default=None,
                   help="Defaults to data/<task>_test.json (held-out split).")
    p.add_argument("--limit", type=int, default=None,
                   help="Eval only the first N test examples (quick sanity).")
    args = p.parse_args()

    evaluate_task(
        task=args.task, quant_bits=args.quant, run_name=args.run,
        csv_path=args.csv, base_model=args.base_model, adapter=args.adapter,
        data_file=args.data_file, limit=args.limit,
    )
