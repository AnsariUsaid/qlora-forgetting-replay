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
from trl import SFTTrainer
from transformers import TrainingArguments
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
                   max_samples: int = None, epochs: int = 3):
    # ── Config ─────────────────────────────────────────────────────
    MODEL_NAME = "unsloth/gemma-2-2b"   # full-precision (BF16) base; quantize at load
    MAX_SEQ_LEN = 512
    LORA_RANK = 16
    LORA_ALPHA = 32
    LR = 2e-4
    BATCH_SIZE = 4
    GRAD_ACCUM = 4

    run_name = f"gemma2b_{quant_bits}bit_replay{int(replay_ratio*100)}pct_{task}"
    wandb.init(project="qlora-forgetting", name=run_name,
               config=dict(quant_bits=quant_bits, replay_ratio=replay_ratio,
                            task=task, lora_rank=LORA_RANK, epochs=epochs,
                            max_samples=max_samples))

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
        random_state=42,
    )

    # ── Build dataset with replay ──────────────────────────────────
    with open(f"data/{task}_train.json") as f:
        task_data = json.load(f)
    with open("data/alpaca_cleaned.json") as f:
        alpaca_data = json.load(f)

    # Sanity-run knob: truncate the task set to a small slice.
    if max_samples is not None:
        task_data = task_data[:max_samples]

    mixed_data = build_replay_dataset(task_data, alpaca_data, replay_ratio)

    ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

    EOS = tokenizer.eos_token

    def format_prompts(examples):
        texts = [
            ALPACA_PROMPT.format(
                instruction=inst, input=inp, output=out
            ) + EOS
            for inst, inp, out in zip(
                examples["instruction"],
                examples["input"],
                examples["output"]
            )
        ]
        return {"text": texts}

    ds = Dataset.from_list(mixed_data)
    ds = ds.map(format_prompts, batched=True)

    # ── Train ─────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        args=TrainingArguments(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            warmup_steps=50,
            num_train_epochs=epochs,
            learning_rate=LR,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            output_dir=f"outputs/{run_name}",
            report_to="wandb",
            save_strategy="epoch",
            seed=42,
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
    args = parser.parse_args()
    run_experiment(args.quant, args.replay, args.task,
                   max_samples=args.max_samples, epochs=args.epochs)
