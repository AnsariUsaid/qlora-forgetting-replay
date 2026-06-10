from datasets import load_dataset
import json, random

# ── Download task datasets ──────────────────────────────────────
medqa = load_dataset("bigbio/med_qa", name="med_qa_en_source",
                     trust_remote_code=True)
samsum = load_dataset("Samsung/samsum")
alpaca = load_dataset("yahma/alpaca-cleaned")
dolly  = load_dataset("databricks/databricks-dolly-15k")

# ── Format MedQA into instruction format ──────────────────────
def format_medqa(example):
    options = " | ".join(
        [f"{k}: {v}" for k, v in example["options"].items()]
    )
    return {
        "instruction": (
            "Answer the following medical question by selecting "
            "the correct option.\n\n"
            f"Question: {example['question']}\n"
            f"Options: {options}"
        ),
        "input": "",
        "output": example["answer"]
    }

# ── Format Samsum into instruction format ─────────────────────
def format_samsum(example):
    return {
        "instruction": "Summarize the following conversation in 1-2 sentences.",
        "input": example["dialogue"],
        "output": example["summary"]
    }

# ── Alpaca prompt template ─────────────────────────────────────
ALPACA_PROMPT = """Below is an instruction that describes a task.
Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

def to_alpaca_prompt(ex):
    return ALPACA_PROMPT.format(
        instruction=ex["instruction"],
        input=ex.get("input", ""),
        output=ex["output"]
    )

# Format and save
medqa_formatted = [format_medqa(x) for x in medqa["train"]]
samsum_formatted = [format_samsum(x) for x in samsum["train"]]
alpaca_data = list(alpaca["train"])

with open("data/medqa_train.json", "w") as f:
    json.dump(medqa_formatted, f)
with open("data/samsum_train.json", "w") as f:
    json.dump(samsum_formatted, f)
with open("data/alpaca_cleaned.json", "w") as f:
    json.dump(alpaca_data, f)

print(f"MedQA: {len(medqa_formatted)} samples")
print(f"Samsum: {len(samsum_formatted)} samples")
print(f"Alpaca: {len(alpaca_data)} samples")
