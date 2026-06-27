"""
QLoRA fine-tuning runner (report §8.3).

Fixes applied on top of starter-asis:
  C1  Load from ONE full-precision base (unsloth/gemma-2-2b) and quantize at
      load time to the level each run needs — FP16 / INT8 / NF4. The starter
      used unsloth/gemma-2-2b-bnb-4bit (already baked to 4-bit), so FP16/INT8
      were impossible, and load_in_8bit was computed but never passed.
      Using one base for all arms means bit-width is the ONLY variable.
  8b  verify_quantization() asserts the loaded model is really in the requested
      precision. Unsloth's 8-bit path has a known bug (issue #2679) where the
      8-bit config can be dropped — this check fails loudly in the sanity run
      instead of silently producing fake-INT8 results.
  C4  Log peak GPU memory (report requires it as a per-run metric).
  D1  Single canonical Alpaca prompt lives here (download_datasets.py no longer
      carries a competing template, so there's no drift).
  --max_samples / --epochs : flags so the sanity run (500 samples, 1 epoch) is
      a command-line option, not a code edit.
"""
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
import torch, json, argparse, wandb
from build_replay_dataset import build_replay_dataset


def verify_quantization(model, quant_bits):
    """Assert the model's linear layers are actually at the requested precision.
    Catches the Unsloth 8-bit bug (#2679) before we waste a run on fake INT8."""
    found = set()
    for m in model.modules():
        cls = m.__class__.__name__
        if cls == "Linear4bit":
            found.add(4)
        elif cls in ("Linear8bitLt", "Linear8bit"):
            found.add(8)

    if quant_bits == 4:
        assert 4 in found, f"Requested 4-bit but found no 4-bit layers (got {found or 'none'})"
    elif quant_bits == 8:
        assert 8 in found, (
            f"Requested 8-bit but found no 8-bit layers (got {found or 'none — loaded full precision?'}). "
            "Likely the Unsloth 8-bit bug (#2679) — INT8 arm is invalid, switch loaders."
        )
    elif quant_bits == 16:
        assert not found, f"Requested 16-bit (full precision) but found quantized layers: {found}"

    print(f"[Quant check] requested {quant_bits}-bit → found bnb layers: {found or 'none (full precision)'} ✓")


def run_experiment(quant_bits: int, replay_ratio: float, task: str,
                   max_samples: int = None, epochs: int = 3,
                   batch_size: int = 2, grad_accum: int = 8, seed: int = 42):
    # ── Config ─────────────────────────────────────────────────────
    # Effective batch = batch_size * grad_accum = 16 (T4-safe defaults).
    MODEL_NAME = "unsloth/Llama-3.2-3B"  # full-precision base; quantize at load.
    # Switched off gemma-2-2b: it needs bf16/fp32, but the free T4 has no bf16, so
    # Unsloth fell back to fp16 and Gemma-2's soft-capping overflowed fp16 →
    # corrupted training (loss stuck ~21, worse than random). Llama-3.2-3B has no
    # soft-capping → numerically fine in fp16 on a T4.
    MAX_SEQ_LEN = 512
    LORA_RANK = 16
    LORA_ALPHA = 32
    LR = 2e-4

    run_name = f"llama3b_{quant_bits}bit_replay{int(replay_ratio*100)}pct_{task}"
    if seed != 42:                       # seed-repeat runs get a distinct name/folder
        run_name += f"_seed{seed}"       # so repeats don't overwrite the seed-42 run
    wandb.init(project="qlora-forgetting", name=run_name,
               config=dict(quant_bits=quant_bits, replay_ratio=replay_ratio,
                            task=task, lora_rank=LORA_RANK, epochs=epochs,
                            max_samples=max_samples, seed=seed))

    # ── Load model at the requested quantization ───────────────────
    # One base model, quantized at load → bit-width is the only variable.
    #   4  → load_in_4bit (NF4)     8 → load_in_8bit     16 → neither (full precision)
    # dtype=None lets Unsloth auto-pick fp16 (T4) or bf16 (Ampere+).
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=(quant_bits == 4),
        load_in_8bit=(quant_bits == 8),
    )
    verify_quantization(model, quant_bits)

    # ── Attach LoRA adapters ───────────────────────────────────────
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj",
                         "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )

    # ── Build dataset with replay ──────────────────────────────────
    with open(f"data/{task}_train.json") as f:
        task_data = json.load(f)
    with open("data/alpaca_cleaned.json") as f:
        alpaca_data = json.load(f)

    # Sanity-run knob: truncate the task set to a small slice.
    if max_samples is not None:
        task_data = task_data[:max_samples]

    mixed_data = build_replay_dataset(task_data, alpaca_data, replay_ratio, seed=seed)

    # Split each example into PROMPT (what the model reads) and COMPLETION (what it
    # must learn to produce). The prompt ends exactly at "### Response:\n"; the answer
    # is the completion. With completion_only_loss=True (below), trl masks the prompt
    # from the loss so the model is graded ONLY on the answer — otherwise it spends its
    # learning memorizing the question, which scrambles general knowledge (forgetting)
    # without teaching the answer. evaluate_task.py's inference prompt ends at the same
    # point, so train and inference stay consistent.
    PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""

    EOS = tokenizer.eos_token

    def format_prompts(examples):
        prompts = [
            PROMPT_TEMPLATE.format(instruction=inst, input=inp)
            for inst, inp in zip(examples["instruction"], examples["input"])
        ]
        completions = [out + EOS for out in examples["output"]]
        return {"prompt": prompts, "completion": completions}

    ds = Dataset.from_list(mixed_data)
    ds = ds.map(format_prompts, batched=True, remove_columns=ds.column_names)

    # ── Train ─────────────────────────────────────────────────────
    # D2: new trl wants SFTConfig (not TrainingArguments), max_length (not
    # max_seq_length), and processing_class (not tokenizer). The old API silently
    # dropped per_device_train_batch_size → trl defaulted to 8 → CUDA OOM on T4.
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds,
        args=SFTConfig(
            completion_only_loss=True,   # mask the prompt → grade only the answer
            max_length=MAX_SEQ_LEN,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            warmup_steps=50,
            num_train_epochs=epochs,
            learning_rate=LR,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            optim="adamw_8bit",
            logging_steps=10,
            output_dir=f"outputs/{run_name}",
            report_to="wandb",
            save_strategy="epoch",
            seed=seed,
        )
    )

    import time
    torch.cuda.reset_peak_memory_stats()      # C4: measure this run's peak only
    t0 = time.time()
    trainer.train()
    train_time_mins = (time.time() - t0) / 60
    peak_mem_mb = torch.cuda.max_memory_allocated() / 1024**2   # C4

    # ── Save adapter ───────────────────────────────────────────────
    save_path = f"outputs/{run_name}/adapter"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    wandb.log({"train_time_minutes": train_time_mins,
               "peak_gpu_memory_mb": peak_mem_mb})
    wandb.finish()
    print(f"\n✓ Done: {run_name} in {train_time_mins:.1f} min "
          f"| peak GPU {peak_mem_mb:.0f} MB\n")
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant", type=int, default=4)        # 4, 8, or 16
    parser.add_argument("--replay", type=float, default=0.0)   # 0.0 to 0.3
    parser.add_argument("--task", type=str, default="medqa")
    parser.add_argument("--max_samples", type=int, default=None,  # sanity run
                        help="Truncate task data to this many samples (e.g. 500).")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2,      # T4-safe
                        help="Per-device batch size. Lower if you hit CUDA OOM.")
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42,          # 42 = all prior runs
                        help="Random seed (data order, LoRA init, trainer). "
                             "Change it for seed-repeat / noise-estimate runs.")
    args = parser.parse_args()
    run_experiment(args.quant, args.replay, args.task,
                   max_samples=args.max_samples, epochs=args.epochs,
                   batch_size=args.batch_size, grad_accum=args.grad_accum,
                   seed=args.seed)
