import json, random
from typing import List, Dict

def build_replay_dataset(
    task_data: List[Dict],
    alpaca_data: List[Dict],
    replay_ratio: float,           # e.g. 0.10 for 10%
    seed: int = 42
) -> List[Dict]:
    """
    Construct mixed training set:
    task_data + replay_ratio * len(task_data) alpaca samples.

    replay_ratio=0.0 → pure task fine-tuning (no replay)
    replay_ratio=0.1 → 10% of task size added from Alpaca
    """
    random.seed(seed)

    n_replay = int(len(task_data) * replay_ratio)

    if n_replay == 0:
        print(f"[Replay] ratio=0 → pure task training, {len(task_data)} samples")
        return task_data

    replay_samples = random.sample(alpaca_data, n_replay)

    # Combine and shuffle uniformly
    combined = task_data + replay_samples
    random.shuffle(combined)

    print(f"[Replay] ratio={replay_ratio:.0%} → "
          f"{len(task_data)} task + {n_replay} replay "
          f"= {len(combined)} total samples")
    return combined


def build_quality_filtered_replay(
    task_data: List[Dict],
    alpaca_data: List[Dict],
    replay_ratio: float,
    model,                         # pass loaded base model for perplexity scoring
    tokenizer,
    seed: int = 42
) -> List[Dict]:
    """
    Quality-filtered replay: sort Alpaca by perplexity under base model,
    take the N lowest-perplexity samples (most 'on-distribution' for general caps).
    Used for the Contribution C4 ablation.
    """
    import torch
    from tqdm import tqdm

    n_replay = int(len(task_data) * replay_ratio)
    random.seed(seed)

    # Score a subset of Alpaca (scoring all 52K is slow — use 5K random subset)
    candidate_pool = random.sample(alpaca_data, min(5000, len(alpaca_data)))
    scored = []

    model.eval()
    with torch.no_grad():
        for ex in tqdm(candidate_pool, desc="Scoring replay candidates"):
            text = ex.get("instruction", "") + " " + ex.get("output", "")
            inputs = tokenizer(text, return_tensors="pt",
                               truncation=True, max_length=256).to("cuda")
            loss = model(**inputs, labels=inputs["input_ids"]).loss.item()
            scored.append((loss, ex))

    scored.sort(key=lambda x: x[0])         # sort ascending — lowest PPL first
    top_k = [ex for _, ex in scored[:n_replay]]

    combined = task_data + top_k
    random.shuffle(combined)
    print(f"[Quality Replay] Selected {n_replay} lowest-perplexity Alpaca samples")
    return combined
