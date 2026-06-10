from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset
import torch, json, argparse, wandb
from build_replay_dataset import build_replay_dataset

def run_experiment(quant_bits: int, replay_ratio: float, task: str):
    # ── Config ─────────────────────────────────────────────────────
    MODEL_NAME = "unsloth/gemma-2-2b-bnb-4bit"  # Unsloth pre-quantized
    MAX_SEQ_LEN = 512
    LORA_RANK = 16
    LORA_ALPHA = 32
    LR = 2e-4
    EPOCHS = 3
    BATCH_SIZE = 4
    GRAD_ACCUM = 4

    run_name = f"gemma2b_{quant_bits}bit_replay{int(replay_ratio*100)}pct_{task}"
    wandb.init(project="qlora-forgetting", name=run_name,
               config=dict(quant_bits=quant_bits, replay_ratio=replay_ratio,
                            task=task, lora_rank=LORA_RANK))

    # ── Load model with correct quantization ──────────────────────
    load_in_4bit = (quant_bits == 4)
    load_in_8bit = (quant_bits == 8)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        dtype=torch.bfloat16 if quant_bits == 16 else None,
        load_in_4bit=load_in_4bit,
    )

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

    mixed_data = build_replay_dataset(task_data, alpaca_data, replay_ratio)

    ALPACA_PROMPT = """Below is an instruction that describes a task.

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
            num_train_epochs=EPOCHS,
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
    t0 = time.time()
    trainer.train()
    train_time_mins = (time.time() - t0) / 60

    # ── Save adapter ───────────────────────────────────────────────
    save_path = f"outputs/{run_name}/adapter"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    wandb.log({"train_time_minutes": train_time_mins})
    wandb.finish()
    print(f"\n✓ Done: {run_name} in {train_time_mins:.1f} min\n")
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant", type=int, default=4)      # 4, 8, or 16
    parser.add_argument("--replay", type=float, default=0.0) # 0.0 to 0.3
    parser.add_argument("--task", type=str, default="medqa")
    args = parser.parse_args()
    run_experiment(args.quant, args.replay, args.task)
