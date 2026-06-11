"""
Download and format all datasets for the QLoRA-forgetting study.

Fixes applied on top of starter-asis (report §8.1):
  B1  MedQA `options` is a LIST of {"key","value"} dicts, not a dict —
      the old `.items()` call would crash. Iterate the list instead.
  B2  Create `data/` before writing, else open(...,"w") raises.
  C3  Dolly was downloaded but never saved, and its columns differ
      (instruction/context/response) — format + save it for the C4 ablation.
  D5  MedQA output is the HYBRID format "A) <answer text>": the model
      learns the full concept, but scoring stays a trivial letter
      exact-match on the "A)" prefix (see evaluate_task.py, Week 2).
"""
from datasets import load_dataset
import json, os

# ── Download datasets ───────────────────────────────────────────
medqa  = load_dataset("bigbio/med_qa", name="med_qa_en_source",
                      trust_remote_code=True)
samsum = load_dataset("Samsung/samsum")
alpaca = load_dataset("yahma/alpaca-cleaned")
dolly  = load_dataset("databricks/databricks-dolly-15k")

# ── Format MedQA into instruction format ───────────────────────
# Source schema: question, answer (text), answer_idx (letter),
#                options = [{"key": "A", "value": "..."}, ...]
def format_medqa(example):
    options = " | ".join(
        f"{opt['key']}) {opt['value']}" for opt in example["options"]
    )
    # D5 hybrid target: letter prefix + full answer text.
    # Letter is the gold (answer_idx); text grounds the small model.
    output = f"{example['answer_idx']}) {example['answer']}"
    return {
        "instruction": (
            "Answer the following medical question by selecting "
            "the correct option.\n\n"
            f"Question: {example['question']}\n"
            f"Options: {options}"
        ),
        "input": "",
        "output": output,
    }

# ── Format Samsum into instruction format ──────────────────────
def format_samsum(example):
    return {
        "instruction": "Summarize the following conversation in 1-2 sentences.",
        "input": example["dialogue"],
        "output": example["summary"],
    }

# ── Format Dolly into the shared instruction/input/output schema ─
# Dolly columns are instruction / context / response (not in/out),
# so map them explicitly or replay formatting will break.
def format_dolly(example):
    return {
        "instruction": example["instruction"],
        "input": example.get("context", ""),
        "output": example["response"],
    }

# ── Format and save ─────────────────────────────────────────────
os.makedirs("data", exist_ok=True)  # B2

medqa_formatted  = [format_medqa(x)  for x in medqa["train"]]
samsum_formatted = [format_samsum(x) for x in samsum["train"]]
alpaca_data      = list(alpaca["train"])          # already instruction/input/output
dolly_formatted  = [format_dolly(x) for x in dolly["train"]]  # C3

with open("data/medqa_train.json", "w") as f:
    json.dump(medqa_formatted, f)
with open("data/samsum_train.json", "w") as f:
    json.dump(samsum_formatted, f)
with open("data/alpaca_cleaned.json", "w") as f:
    json.dump(alpaca_data, f)
with open("data/dolly_15k.json", "w") as f:                    # C3
    json.dump(dolly_formatted, f)

print(f"MedQA:  {len(medqa_formatted)} samples")
print(f"Samsum: {len(samsum_formatted)} samples")
print(f"Alpaca: {len(alpaca_data)} samples")
print(f"Dolly:  {len(dolly_formatted)} samples")

# Sanity peek — verify the hybrid MedQA target looks right
print("\nExample MedQA output:", medqa_formatted[0]["output"])
