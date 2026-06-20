"""
Measure TASK performance of a fine-tuned adapter (report §10 "Task Performance").

This answers "did the model actually learn the task?" — the complement to
evaluate_forgetting.py ("did it forget general knowledge?"). A good result needs
BOTH: high task score AND low forgetting.

  • MedQA  (multiple-choice): read the model's A-D answer-token logits and take the
            argmax (no text generation); score accuracy + macro-F1 vs the gold letter.
  • Samsum (summarization):    generate a summary, score ROUGE-L vs the reference.

Evaluated on the HELD-OUT TEST split (data/{task}_test.json) — never the training
data, or the score would be meaningless (memorisation, not learning).

Loads the SAME base model at the SAME quant level as training. With --adapter the
fine-tuned model is measured; WITHOUT it the bare base model is measured (the task
baseline, so we can see how much fine-tuning added).
"""
import os, json, re, argparse

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


LETTERS = ["A", "B", "C", "D"]


def _extract_letter(text: str) -> str:
    """First A-D in the text (gold answers are 'A) ...'); '' if none found."""
    m = re.search(r"[ABCD]", text.upper())
    return m.group(0) if m else ""


def _letter_token_ids(tokenizer):
    """First token id of each option letter A-D. The trained completion starts
    with the letter ('A) ...'), so the model's whole answer is decided by the
    very first token it would emit after the prompt."""
    return [tokenizer.encode(L, add_special_tokens=False)[0] for L in LETTERS]


def score_medqa(data, model, tokenizer) -> dict:
    """Letter-logit scoring (replaces generate-then-regex): read the model's
    A/B/C/D logits at the answer position and take the argmax. Deterministic —
    no decoding fragility (empty output, wrong letter, 'always-A'), and every
    example yields a real A-D prediction so the accuracy actually discriminates."""
    import torch
    from sklearn.metrics import accuracy_score, f1_score

    letter_ids = _letter_token_ids(tokenizer)
    gold, pred = [], []
    for ex in data:
        gold.append(_extract_letter(ex["output"]))               # "A) ..." → "A"
        prompt = INFER_PROMPT.format(instruction=ex["instruction"], input=ex["input"])
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=MAX_SEQ_LEN).to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, :]            # next-token logits
        choice = int(torch.argmax(logits[letter_ids]))           # 0..3 → A..D
        pred.append(LETTERS[choice])
    acc = accuracy_score(gold, pred)
    f1 = f1_score(gold, pred, average="macro", labels=LETTERS, zero_division=0)
    return {"accuracy": round(acc * 100, 2), "macro_f1": round(f1 * 100, 2)}


def score_samsum(data, model, tokenizer) -> dict:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    total = 0.0
    for ex in data:
        out = generate(model, tokenizer, ex["instruction"], ex["input"], 128)
        total += scorer.score(ex["output"], out)["rougeL"].fmeasure
    return {"rougeL": round(total / len(data) * 100, 2)}


def evaluate_task(task: str, quant_bits: int, run_name: str, runs_dir: str,
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

    # Unified schema across tasks (irrelevant fields left blank).
    row = {
        "run": run_name,
        "task": task,
        "quant_bits": quant_bits,
        "n_eval": len(data),
        "accuracy": metrics.get("accuracy", ""),
        "macro_f1": metrics.get("macro_f1", ""),
        "rougeL": metrics.get("rougeL", ""),
    }

    # Per-run source of truth (see evaluate_forgetting.py): write to this run's own
    # folder; the master task_results.csv is rebuilt from these by aggregate_results.py.
    run_path = os.path.join(runs_dir, run_name)
    os.makedirs(run_path, exist_ok=True)
    with open(os.path.join(run_path, "task.json"), "w") as f:
        json.dump(row, f, indent=2)

    print(f"[Task Eval] {run_name} ({task}, n={len(data)}) → {metrics}")
    return row


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, required=True, choices=["medqa", "samsum"])
    p.add_argument("--quant", type=int, required=True, choices=[4, 8, 16])
    p.add_argument("--run", type=str, required=True,
                   help="Run name (also the per-run output folder name).")
    p.add_argument("--adapter", type=str, default=None,
                   help="Trained adapter path. OMIT to score the BASE (un-fine-tuned) "
                        "model — the task baseline that shows how much fine-tuning added.")
    p.add_argument("--base_model", type=str, default="unsloth/Llama-3.2-3B")
    p.add_argument("--runs_dir", type=str, default="results/runs",
                   help="Per-run output: <runs_dir>/<run>/task.json. "
                        "Rebuild results/task_results.csv with aggregate_results.py.")
    p.add_argument("--data_file", type=str, default=None,
                   help="Defaults to data/<task>_test.json (held-out split).")
    p.add_argument("--limit", type=int, default=None,
                   help="Eval only the first N test examples (quick sanity).")
    args = p.parse_args()

    evaluate_task(
        task=args.task, quant_bits=args.quant, run_name=args.run,
        runs_dir=args.runs_dir, base_model=args.base_model, adapter=args.adapter,
        data_file=args.data_file, limit=args.limit,
    )
