# Running on Kaggle

This project runs on free Kaggle notebooks. Use **two** notebooks:

| Notebook | Accelerator | Purpose |
|---|---|---|
| **Data** (e.g. `DataSets-QLora`) | **None (CPU)** | download + format the datasets once |
| **Training** (e.g. `QLORA-1`) | **GPU T4** | train adapters + evaluate |

Both clone this public repo, then run its scripts with `!python ...`.

> Reminder: a Kaggle session is temporary. When you **stop the session**, the
> installed libraries and the `/kaggle/working` files are wiped. On every fresh
> start you must re-run the install + clone cells. (Saved Versions/Datasets and
> your W&B runs DO persist.) GPU quota = 30h/week — only turn the GPU on for the
> Training notebook, and stop the session when you step away for a while.

---

## Notebook 1 — Data (CPU, Internet ON)

```python
# Cell 1 — get the code
!rm -rf qlora-forgetting-replay
!git clone https://github.com/AnsariUsaid/qlora-forgetting-replay.git
%cd qlora-forgetting-replay
```
```python
# Cell 2 — datasets library (Parquet sources need a current version)
!pip install -q -U datasets
```
```python
# Cell 3 — download + format MedQA / Samsum / Alpaca / Dolly into data/
!python data/download_datasets.py
```
```python
# Cell 4 — verify
import json
for f in ["medqa_train", "samsum_train", "alpaca_cleaned", "dolly_15k"]:
    d = json.load(open(f"data/{f}.json"))
    print(f"{f:18s} {len(d):>7,} samples | keys: {list(d[0].keys())}")
```
Expected: MedQA ~10,178 · Samsum 14,731 · Alpaca 51,760 · Dolly 15,011.
Then **Save Version** so the data persists in the notebook output.

---

## Notebook 2 — Training (GPU T4, Internet ON)

Run top to bottom (or **Run All**). Order matters on a cold start.

```python
# Cell 1 — get the code (do this BEFORE installing, so requirements.txt is present)
!rm -rf qlora-forgetting-replay
!git clone https://github.com/AnsariUsaid/qlora-forgetting-replay.git
%cd qlora-forgetting-replay
```
```python
# Cell 2 — install the full stack (~4 min)
!pip install -q -r setup/requirements.txt
```
```python
# Cell 3 — log in to Weights & Biases (uses a Kaggle Secret named WANDB_API_KEY)
from kaggle_secrets import UserSecretsClient
import wandb
wandb.login(key=UserSecretsClient().get_secret("WANDB_API_KEY"))
```
```python
# Cell 4 — regenerate data/ (cloud-side, ~1 min). Later this can be a Kaggle
# Dataset attached read-only instead of re-downloading.
!python data/download_datasets.py
```
```python
# Cell 5 — SANITY run (tiny: 500 samples, 1 epoch). Confirms the loop works.
!python train.py --quant 4 --replay 0.0 --task medqa --max_samples 500 --epochs 1
```

### Real training runs (Phase 6+)
Drop `--max_samples`/`--epochs`, pick quant level and replay ratio. GPU headroom
is large at 4-bit, so raise the batch size for speed:
```python
!python train.py --quant 4 --replay 0.10 --task medqa --batch_size 8
```
- `--quant` 4 | 8 | 16   ·   `--replay` 0.0 0.05 0.10 0.20 0.30   ·   `--task` medqa | samsum
- If you ever hit CUDA OOM (likely only on 16-bit), add `--batch_size 1`.

### Evaluation (Phase 6+)
First the per-quant baseline (no adapter), once per quant level:
```python
!python evaluate_forgetting.py --quant 4 --run baseline_4bit --limit 50
```
Then a fine-tuned run (reads the baseline, computes the Forgetting Score):
```python
!python evaluate_forgetting.py --quant 4 --run gemma2b_4bit_replay0pct_medqa \
    --adapter outputs/gemma2b_4bit_replay0pct_medqa/adapter --limit 50
```
`--limit` runs a quick subset; drop it for the full (slow) MMLU once verified.
Results append to `results/all_results.csv`; baselines live in `results/baselines.json`.
