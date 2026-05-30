# WSSIS Runbook — Remote GPU Machine

Step-by-step guide to clone, set up conda, download data, run P0 prep, and execute all experiments from [report/EXPERIMENT.md](../report/EXPERIMENT.md) and [report/PLAN.md](../report/PLAN.md).

---

## Prerequisites


| Requirement                | Notes                                              |
| -------------------------- | -------------------------------------------------- |
| Linux + NVIDIA GPU         | CUDA 12.4+ recommended (PyTorch cu124 wheels)      |
| Conda (Miniconda/Anaconda) | For env `wssis`                                    |
| Git                        | Clone this repo                                    |
| Kaggle account             | API token → `data/kaggle.json`                     |
| Disk space                 | ~200 GB (COCO images + ~23 GB SAM cache + outputs) |


**GPU policy** ([report/RANDOM_NOTE.md](../report/RANDOM_NOTE.md)): one GPU runs SAM teacher inference; remaining GPUs train the student. Set `CUDA_VISIBLE_DEVICES=0` for teacher and `WSSIS_NUM_GPUS` for Mask2Former.

---

## Step 0 — Clone repository

```bash
git clone <your-github-url>/wssis.git
cd wssis
export WSSIS_REPO_ROOT="$(pwd)"
export PYTHONPATH="$WSSIS_REPO_ROOT:$PYTHONPATH"
```

---

## Step 1 — Kaggle credentials

```bash
mkdir -p data
cp ~/.kaggle/kaggle.json data/kaggle.json
chmod 600 data/kaggle.json   # Linux only
```

Template: `data/kaggle.json.example`

---

## Step 2 — Create conda environment

```bash
bash scripts/setup/00_create_conda_env.sh
conda activate wssis
```

This will:

1. Create/update env `wssis` from `environment.yml` (Python 3.10)
2. Install PyTorch **2.6** + **cu124** wheels (default in setup script)
3. `pip install -r requirements.txt`
4. Install Detectron2 (prebuilt wheel when available, else editable `modules/detectron2`)
5. Compile Mask2Former `MSDeformAttn` ops
6. Download `checkpoints/sam_vit_b_01ec64.pth`

**Tested versions** (see `requirements.txt` header): torch `2.6.0+cu124`, torchvision `0.21.0+cu124`, detectron2 `0.6`, ultralytics `8.4.x`, `segment-anything` `1.0`.

**Manual fallback** if Detectron2 build fails:

```bash
pip install --no-build-isolation -e modules/detectron2
cd modules/mask2former/mask2former/modeling/pixel_decoder/ops && bash make.sh
```

Override PyTorch CUDA index (e.g. cu118 node): `WSSIS_PYTORCH_INDEX=https://download.pytorch.org/whl/cu118 bash scripts/setup/00_create_conda_env.sh`

---

## Step 3 — Download datasets (Kaggle → `data/`)

Datasets:

- [COCO 2017](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset) → `data/coco2017/`
- [coco-minitrain-10k](https://www.kaggle.com/datasets/banuprasadb/coco-minitrain-10k) → `data/coco_minitrain_10k/`

```bash
bash scripts/setup/01_download_data.sh
```

Verify:

```bash
ls data/coco2017/annotations/instances_train2017.json
ls data/coco_minitrain_10k/train2017.txt
```

Skip if already present:

```bash
python -m modules.wssis.prep.download_kaggle --skip-coco
python -m modules.wssis.prep.download_kaggle --skip-minitrain
```

---

## Step 4 — P0 preparation (once)

Pick a **run id** (all logs/checkpoints/viz go to one bundle):

```bash
export WSSIS_RUN_ID=wssis_main
bash scripts/prep/run_p0.sh --run-id $WSSIS_RUN_ID
# larger GPU: pass Stage-1 batch size
bash scripts/prep/run_p0.sh --run-id $WSSIS_RUN_ID --batch-size 16
```

**Resume after interrupt** (skips completed steps in `progress.json`):

```bash
bash scripts/prep/run_p0.sh --run-id $WSSIS_RUN_ID --resume --skip-splits --skip-embeddings
```

Or step-by-step:


| Step | Command                                                                            | Output                                                                               |
| ---- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| P0.1 | `python -m modules.wssis.prep.generate_splits`                                     | `data/splits/*`                                                                      |
| P0.2 | `python -m modules.wssis.prep.precompute_sam_embeddings`                           | `data/cache/sam_embeddings/` (~23 GB)                                                |
| P0.4 | `python -m modules.wssis.prep.train_stage1_gnn --run-id $WSSIS_RUN_ID --epochs 30 --batch-size 4` | `outputs/runs/<id>/checkpoints/best.pt` + legacy `checkpoints/gnn_refiner_stage1.pt` |


### Logging & checkpoints (Stage-1)


| Feature                                                                   | Location                                         |
| ------------------------------------------------------------------------- | ------------------------------------------------ |
| Per-epoch **loss components** (raw + weighted BCE, Dice, sym; train + val) + **AP** (`raw_sam_ap`, `val_refined_ap`, `delta_ap`) | `outputs/runs/<id>/logs/metrics.jsonl` |
| Stage-1 **train data** | `labeled_5pct_train.txt`; in-loop val = `labeled_5pct_val.txt`; final = `val_all.txt` |
| Text log                                                                  | `logs/train.log`                                 |
| TensorBoard                                                               | `logs/tensorboard/`                              |
| WandB                                                                     | Set `WANDB_PROJECT=wssis`                        |
| `last.pt` / `best.pt` / `epoch_XXX.pt`                                    | `checkpoints/`                                   |
| Early stopping (patience=10 on val_refined_ap)                            | enabled by default; `--no-early-stop` to disable |
| Resume training                                                           | `--resume` loads `checkpoints/last.pt`           |


**Visualizations:** Every epoch → `outputs/runs/<id>/visualizations/` (5 panels). Report bundle → `outputs/runs/<id>/report/`.

**Debug (small subset):**

```bash
python -m modules.wssis.prep.precompute_sam_embeddings --limit 32
python -m modules.wssis.prep.train_stage1_gnn --run-id debug --epochs 2 --max-instances 500
```

**Exp 2C (no symmetric loss GNN):**

```bash
python -m modules.wssis.prep.train_stage1_gnn --run-id $WSSIS_RUN_ID \
  --symmetric-weight 0 --output-name gnn_refiner_no_sym.pt
```

### Why Stage-1 train/val feels slow

Each batch runs **SAM mask decoder** (per instance) + **GNN**. Without P0.2 cache, it also ran **SAM ViT-B encoder** every step — often **many times per image** (one dataloader row per COCO instance).

| Factor | Effect |
|--------|--------|
| **Instance-level dataset** | ~500 labeled images → **thousands** of train steps/epoch (multiple objects per image) |
| **Val on full `val_all`** | Many more val batches than train |
| **Large `batch_size`** (e.g. 16) | More serial SAM decoder calls per batch |
| **No P0.2 cache** (old runs) | Re-encoded every instance; very slow |

**Speedups (current code):** `use_sam_embedding_cache=true` (default) loads `data/cache/sam_embeddings/{train,val}/*.fp16.npy` from P0.2 and deduplicates encoder work by `image_id` within a batch. Used in **Stage-1 train/val**, **teacher eval**, and **epoch visualizations**. **Run P0.2 before P0.4** for best speed.

**Intentional live encoder:** only `precompute_sam_embeddings` (P0.2) and cache-miss fallback in `modules/wssis/sam_cache.py`.

**Further knobs:** lower `--batch-size` if OOM; `--max-instances` for debugging; ensure P0.2 finished on `train_all` + `val_all`.

### Eval split policy (train fast, final full)

| Phase | Train data | In-loop / routine eval | Final eval |
|-------|------------|------------------------|------------|
| **P0.4 Stage-1 GNN** | `labeled_5pct_train.txt` (~80% of 5% pool) | `labeled_5pct_val.txt` (~20% holdout, still 5% pool) | **`val_all.txt` full** (auto after P0.4; `--no-final-eval` to skip) |
| **Stage-2 / experiments** | per exp (`train_all`, etc.) | **`val_sample_20pct.txt`** (~20% of val) | `python … evaluate_teacher --full-val` or `run_experiment --stage eval --full-val` |

Regenerate splits after changing fractions: `python -m modules.wssis.prep.generate_splits --force`

### Stage-1 data split (supervised 5% only)

P0.1 builds **coco-minitrain-10k** lists, then `labeled_5pct.txt` (full 5% pool), split into **train/val folds** inside that pool.  
**P0.4** trains on `labeled_5pct_train` only; early stopping uses `labeled_5pct_val` (not full `val_all`).  
The 95% weak split is **not** used until Stage-2.

### Known limitation (PLAN §0.5)

Stage-1 GNN now follows PLAN §2: SAM embed initializes graph nodes; inputs are image + 3 SAM masks + weak signal. Val logs **raw SAM AP**, **refined AP**, and **ΔAP**. Re-train P0.4 after pulling this change (old `sam_embed`-only checkpoints are incompatible).

---

## Step 5 — Run experiments

List experiments:

```bash
python -m modules.wssis.run_experiment --list
```

### Main result first (Exp 1C)

```bash
python -m modules.wssis.run_experiment --exp 1C --run-id $WSSIS_RUN_ID
# or
python scripts/experiments/run_exp_1c.py
```

### Upload Exp 1C best weights to Hugging Face (demo)

```bash
huggingface-cli login
python scripts/upload_exp_1c_hf.py --repo-id YOUR_USER/wssis-1c-demo --run-id $WSSIS_RUN_ID
```

### Single experiment


| Script                              | Experiment                     |
| ----------------------------------- | ------------------------------ |
| `scripts/experiments/run_exp_1a.py` | 1A — 5% supervised lower bound |
| `scripts/experiments/run_exp_1b.py` | 1B — raw SAM weak baseline     |
| `scripts/experiments/run_exp_1c.py` | **1C — full SWSIS (main)**     |
| `scripts/experiments/run_exp_1d.py` | 1D — 100% upper bound          |
| `scripts/experiments/run_exp_2a.py` | 2A — no GNN                    |
| `scripts/experiments/run_exp_2b.py` | 2B — no distillation           |
| `scripts/experiments/run_exp_2c.py` | 2C — no symmetric loss         |
| `scripts/experiments/run_exp_3a.py` | 3A — boxes only                |
| `scripts/experiments/run_exp_3b.py` | 3B — points only               |
| `scripts/experiments/run_exp_3c.py` | 3C — mixed signals             |
| `scripts/experiments/run_exp_4a.py` | 4A — YOLOv8-seg                |


Dry-run (print commands only):

```bash
python -m modules.wssis.run_experiment --exp 1C --stage train --dry-run
```

### Run all experiments (PLAN order)

```bash
export WSSIS_RUN_ID=wssis_main
bash scripts/experiments/run_all_experiments.sh --with-p0 --run-id $WSSIS_RUN_ID
# resume after interrupt:
bash scripts/experiments/run_all_experiments.sh --run-id $WSSIS_RUN_ID --resume
```

Order: **1C → 1A → 1B → 1D → 2A → 2B → 2C → 3A → 3B → 3C → 4A**

Outputs: `outputs/runs/<run_id>/experiments/<ID>/` + `**outputs/runs/<run_id>/report/`** for upload

---

## Step 6 — Logging & evaluation

**Primary metric: COCO instance-segmentation mask AP.** IoU/Dice are auxiliary (GNN training only). Student eval uses full COCO AP, AP50, AP75, AP_S, AP_M, AP_L.

Progress tracker: **[CHECKLIST.md](CHECKLIST.md)**

Log during training:

- `sup_loss`, `semi_loss`, `distill_loss` (Stage-2, when integrated)
- GNN `sym_loss`, `partial_ce`, agreement rate
- GPU memory, time/epoch

Use WandB (optional):

```bash
export WANDB_PROJECT=wssis
wandb login
```

### Teacher eval on val (raw SAM + GNN-refined, all 3 weak-signal types)

Fast eval (20% val sample, default for experiment `--stage eval`):

```bash
bash scripts/eval/run_teacher_eval.sh --run-id $WSSIS_RUN_ID
```

Full val (reporting / final numbers):

```bash
bash scripts/eval/run_teacher_eval.sh --run-id $WSSIS_RUN_ID --full-val
# or after P0.4 (automatic unless --no-final-eval):
# outputs/runs/<id>/eval/teacher_val_report_full.json
```
# or:
python -m modules.wssis.training.evaluate_teacher --run-id $WSSIS_RUN_ID
```

Output: `outputs/runs/<id>/eval/teacher_val_report.json` with AP/AP50 per signal type:
`boxes_only`, `points_only`, `scribbles_only` for both `raw_sam` and `gnn_refined`.

Weak-signal maps are 1×H×W tensors (stacked as 3 channels): point/scribble use Gaussian widening; box uses uniform fill inside bbox.

Or via experiment runner:

```bash
python -m modules.wssis.run_experiment --exp 1C --stage eval --run-id $WSSIS_RUN_ID
```

---

## Directory layout after setup

```
wssis/
├── data/
│   ├── kaggle.json          # NOT in git
│   ├── coco2017/
│   ├── coco_minitrain_10k/
│   ├── splits/              # P0.1
│   └── cache/sam_embeddings/  # P0.2
├── checkpoints/             # legacy symlinks / copies of best.pt
├── outputs/runs/<run_id>/   # ONE bundle for report (logs, viz, ckpt, report/)
├── outputs/experiments/     # legacy (prefer runs/<id>/experiments/)
├── modules/
│   ├── wssis/               # unified orchestration
│   ├── vig_refinenet/
│   ├── mask2former/
│   └── segment-anything/
├── scripts/setup/
├── scripts/prep/
├── scripts/experiments/
├── environment.yml
└── requirements.txt
```

---

## Troubleshooting


| Issue                          | Fix                                                        |
| ------------------------------ | ---------------------------------------------------------- |
| `_CONDA_PYTHON_SYSCONFIGDATA_NAME_USED: unbound variable` on `conda activate` | Fixed in repo scripts via `scripts/lib/activate_wssis.sh`; sync repo (include `scripts/lib/`) then re-run |
| `ModuleNotFoundError: modules` | `export PYTHONPATH=$PWD` from repo root                    |
| Kaggle 403                     | Check `data/kaggle.json` permissions and API token         |
| CUDA OOM on P0.2               | `--limit 100` for testing; reduce batch size               |
| Detectron2 import error        | Re-run `00_create_conda_env.sh` or `pip install --no-build-isolation -e modules/detectron2` |
| PyTorch / CUDA mismatch        | Reinstall with `WSSIS_PYTORCH_INDEX` + matching `WSSIS_TORCH_VERSION` env vars            |
| Mask2Former config missing     | Add COCO configs under `modules/mask2former/configs/coco/` |


---

## Quick reference (copy-paste)

```bash
conda activate wssis
export WSSIS_REPO_ROOT=$(pwd) PYTHONPATH=$(pwd)
export WSSIS_RUN_ID=wssis_main

bash scripts/setup/00_create_conda_env.sh
cp ~/.kaggle/kaggle.json data/kaggle.json
bash scripts/setup/01_download_data.sh
bash scripts/prep/run_p0.sh --run-id $WSSIS_RUN_ID
bash scripts/eval/run_teacher_eval.sh --run-id $WSSIS_RUN_ID
python -m modules.wssis.run_experiment --exp 1C --run-id $WSSIS_RUN_ID --stage all
# zip outputs/runs/$WSSIS_RUN_ID/report/ for submission
# Full checklist: scripts/CHECKLIST.md
```

