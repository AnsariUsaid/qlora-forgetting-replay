"""
Download and format all datasets for the QLoRA-forgetting study.

Fixes applied on top of starter-asis (report §8.1):
  B1  MedQA: the report's bigbio/med_qa is a *loading-script* dataset, which
      the new `datasets` (v4+) refuses ("trust_remote_code not supported").
      Switched to GBaker/MedQA-USMLE-4-options — a Parquet (script-free)
      mirror. Its `options` is a DICT {"A": "...", ...}, so iterate .items().
      (Likewise Samsung/samsum was script-based → use knkarthick/samsum,
       a Parquet mirror with identical dialogue/summary columns.)
  B2  Create `data/` before writing, else open(...,"w") raises.
  C3  Dolly was downloaded but never saved, and its columns differ
      (instruction/context/response) — format + save it for the C4 ablation.
  D5  MedQA output is the HYBRID format "A) <answer text>": the model
      learns the full concept, but scoring stays a trivial letter
      exact-match on the "A)" prefix (see evaluate_task.py, Week 2).
"""
from datasets import load_dataset
import json, os, random

# ── Download datasets (all Parquet / script-free, work on datasets v4+) ──
medqa  = load_dataset("GBaker/MedQA-USMLE-4-options")   # ~10,178 train
samsum = load_dataset("knkarthick/samsum")              # 14,732 train
alpaca = load_dataset("yahma/alpaca-cleaned")
dolly  = load_dataset("databricks/databricks-dolly-15k")
sql    = load_dataset("b-mc2/sql-create-context")       # ~78,577 train ONLY (no test split)

# ── Format MedQA into instruction format ───────────────────────
# Source schema (GBaker/MedQA-USMLE-4-options):
#   question (str), answer (full text), answer_idx (letter "A".."D"),
#   options = {"A": "...", "B": "...", "C": "...", "D": "..."}  (a dict)
def format_medqa(example):
    options = " | ".join(
        f"{k}) {v}" for k, v in example["options"].items()
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

# ── Format SQL (text-to-SQL) into instruction format ───────────
# Source schema (b-mc2/sql-create-context):
#   question (NL ask), context (CREATE TABLE schema), answer (the gold SQL query)
# A structured/code-like task → expected to be MORE domain-divergent than Samsum,
# so a stronger forgetting probe (and tests whether replay hurts rigid-syntax tasks).
def format_sql(example):
    return {
        "instruction": "Write a SQL query that answers the question, using the given database schema.",
        "input": f"Schema: {example['context']}\nQuestion: {example['question']}",
        "output": example["answer"],
    }

# ── Format and save ─────────────────────────────────────────────
os.makedirs("data", exist_ok=True)  # B2

medqa_formatted  = [format_medqa(x)  for x in medqa["train"]]
samsum_formatted = [format_samsum(x) for x in samsum["train"]]
alpaca_data      = list(alpaca["train"])          # already instruction/input/output
dolly_formatted  = [format_dolly(x) for x in dolly["train"]]  # C3

# Held-out TEST splits — used by evaluate_task.py to score task performance on
# data the model never trained on (MedQA and Samsum both ship a test split).
medqa_test  = [format_medqa(x)  for x in medqa["test"]]
samsum_test = [format_samsum(x) for x in samsum["test"]]

with open("data/medqa_train.json", "w") as f:
    json.dump(medqa_formatted, f)
with open("data/samsum_train.json", "w") as f:
    json.dump(samsum_formatted, f)
with open("data/alpaca_cleaned.json", "w") as f:
    json.dump(alpaca_data, f)
with open("data/dolly_15k.json", "w") as f:                    # C3
    json.dump(dolly_formatted, f)
with open("data/medqa_test.json", "w") as f:
    json.dump(medqa_test, f)
with open("data/samsum_test.json", "w") as f:
    json.dump(samsum_test, f)

# SQL ships ONLY a train split, so carve out our own held-out test (seed 42, like
# every other split here). Subsample to ~10k train to match MedQA/Samsum scale, so
# dataset size isn't an extra variable when comparing forgetting across tasks.
sql_all = [format_sql(x) for x in sql["train"]]
random.Random(42).shuffle(sql_all)
sql_test  = sql_all[:1000]
sql_train = sql_all[1000:11000]
with open("data/sql_train.json", "w") as f:
    json.dump(sql_train, f)
with open("data/sql_test.json", "w") as f:
    json.dump(sql_test, f)

print(f"MedQA:  {len(medqa_formatted)} train / {len(medqa_test)} test")
print(f"Samsum: {len(samsum_formatted)} train / {len(samsum_test)} test")
print(f"SQL:    {len(sql_train)} train / {len(sql_test)} test")
print(f"Alpaca: {len(alpaca_data)} samples")
print(f"Dolly:  {len(dolly_formatted)} samples")

# Sanity peek — verify the hybrid MedQA target looks right
print("\nExample MedQA output:", medqa_formatted[0]["output"])
