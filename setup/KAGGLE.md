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
Always use `--batch_size 1` for evals — Gemma-2's ~256k-vocab logits OOM otherwise.

First the per-quant baseline (no adapter), once per quant level. **The 4-bit baseline
is already done and committed in `results/baselines.json`, so it never needs re-running.**
```python
!python evaluate_forgetting.py --quant 8 --run baseline_8bit --batch_size 1 --limit 100
```
Then a fine-tuned run (reads the baseline, computes the Forgetting Score):
```python
!python evaluate_forgetting.py --quant 4 --run gemma2b_4bit_replay0pct_medqa \
    --adapter outputs/gemma2b_4bit_replay0pct_medqa/adapter --batch_size 1 --limit 100
```
**Speed: keep `--limit ~100` on the real runs.** Full MMLU at batch 1 ≈ 3 hrs/eval;
`--limit 100` (≈100 Qs/subject) drops it to ~30 min and stays within ~1-2% of the full
score — negligible against the large forgetting signal. (Optionally re-run only the final
headline numbers at full precision by dropping `--limit`.)

Task performance (did it learn the task?) — accuracy/F1 for MedQA, ROUGE-L for Samsum:
```python
!python evaluate_task.py --task medqa --quant 4 --run gemma2b_4bit_replay0pct_medqa \
    --adapter outputs/gemma2b_4bit_replay0pct_medqa/adapter
```
Results append to `results/all_results.csv` (forgetting) and `results/task_results.csv`
(task); baselines live in `results/baselines.json`.

---

## Production workflow — background commits + persistence

For long runs (a full train+eval chain is many hours), don't run live. Use
**Save Version → "Save & Run All (Commit)"** (GPU T4 + Internet on): Kaggle runs the
notebook headless up to ~12 hrs, independent of your browser, and saves the output
when the **last cell finishes cleanly**. Caveats learned the hard way:
- The ~12 hr wall is **per commit**. A full eval-heavy chain can take ~11 hrs — keep
  `--limit` on so you don't risk the wall.
- If a cell **errors**, the commit is marked failed — but the output saved so far
  (adapters, CSVs) is usually still in the version. Keep that version until you've
  re-run the failed cell.
- To re-run just one cell (e.g. a fixed task-eval) cheaply: start a **fresh** commit
  that **attaches the previous version's output** (Add Input → your notebook output)
  so the saved adapter is at `/kaggle/input/...`, then run only that cell (~30-45 min).

**Persistence — results must leave the session.** Re-cloning wipes `/kaggle/working`.
The durable record is the git repo: after each run, copy the updated CSV rows + any new
baseline into `results/` and **push**. Adapters stay in the Kaggle version output (too
big for git); training curves stay in W&B. Per-run log links go in `results/LOGS.md`.
