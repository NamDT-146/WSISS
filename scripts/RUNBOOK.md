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
| Early stopping (patience=3 on val_refined_ap)                             | enabled by default; `--no-early-stop` to disable |
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
| **Stage-2 / experiments** | per exp (`train_all`, etc.) | **`val_sample_20pct.txt`** (~20% of val); early stop **patience=3** on `segm/AP` | **full `val_all`** auto at end of training (Mask2Former); batch eval via `run_all_experiment_eval.sh --full-val` |

Regenerate splits after changing fractions: `python -m modules.wssis.prep.generate_splits --force`

### Stage-1 data split (supervised 5% only)

P0.1 builds **coco-minitrain-10k** lists, then `labeled_5pct.txt` (full 5% pool), split into **train/val folds** inside that pool.  
**P0.4** trains on `labeled_5pct_train` only; early stopping uses `labeled_5pct_val` (not full `val_all`).  
The 95% weak split is **not** used until Stage-2.

### Known limitation (PLAN §0.5)

Stage-1 GNN now follows PLAN §2: SAM embed initializes graph nodes; inputs are image + 3 SAM masks + weak signal. Val logs **raw SAM AP**, **refined AP**, and **ΔAP**. Re-train P0.4 after pulling this change (old `sam_embed`-only checkpoints are incompatible).

---

## Step 5 — Run experiments (train only; eval is separate)

**Run the smoke test first** (~10 min, 1 GPU) before full GPU training:

```bash
bash scripts/experiments/run_smoke_test.sh
# or: export WSSIS_SMOKE=1 WSSIS_RUN_ID=smoke_quick && bash scripts/experiments/run_smoke_test.sh
```

Smoke uses `outputs/runs/smoke_quick/` (separate from production `wssis_main`). Sets: 2 images, 10 M2F iters, 1 epoch Stage-1.

**Time-saving policy:** Teacher AP runs **once** during P0.4 (full val). Stage-2 sweeps **train only** by default — no repeated teacher eval per experiment.

List experiments:

```bash
python -m modules.wssis.run_experiment --list
```

### Execution plan (recommended — True SWSIS)

| Phase | What | Command |
| ----- | ---- | ------- |
| **0. Smoke** | Sanity check all paths | `bash scripts/experiments/run_smoke_test.sh` |
| **1. Prep** (once) | P0 + teacher AP | `bash scripts/prep/run_p0.sh --run-id $WSSIS_RUN_ID` |
| **2A. Train** | 1A (4 GPU) + P0.4 GNN (1 GPU) | parallel terminals — see below |
| **2B. Train** | 1C then 4A | `bash scripts/experiments/run_experiments_parallel.sh --run-id $WSSIS_RUN_ID` |
| **3. Eval** | Student AP batch (1A, 1C, 4A) | `bash scripts/eval/run_all_experiment_eval.sh --run-id $WSSIS_RUN_ID --full-val` |
| **4. Report item 2** | Teacher AP on 5% holdout | `bash scripts/eval/run_teacher_eval.sh --run-id $WSSIS_RUN_ID --stage1-holdout` |

**Report item 5 (upper bound):** reuse existing full-supervised Mask2Former run as **1D** — do not re-run.

### GPU phases (4-GPU node example)

```bash
export WSSIS_RUN_ID=wssis_main

# Terminal A — retrain image-level GNN (1 GPU)
python -m modules.wssis.prep.train_stage1_gnn --run-id $WSSIS_RUN_ID --resume

# Terminal B — 1A labeled-only baseline (3–4 GPUs)
export WSSIS_NUM_GPUS=4
python -m modules.wssis.run_experiment --exp 1A --stage train --run-id $WSSIS_RUN_ID

# After P0.4 — item 2 teacher report
bash scripts/eval/run_teacher_eval.sh --run-id $WSSIS_RUN_ID --stage1-holdout

# Phase B — true semi-weak
export WSSIS_NUM_GPUS=4
python -m modules.wssis.run_experiment --exp 1C --stage train --run-id $WSSIS_RUN_ID
export WSSIS_NUM_GPUS=2
python -m modules.wssis.run_experiment --exp 4A --stage train --run-id $WSSIS_RUN_ID
```

### Single experiment (active IDs)

| Command | Experiment |
| ------- | ---------- |
| `python -m modules.wssis.run_experiment --exp 1A --stage train` | **1A** — 5% supervised lower bound |
| `python -m modules.wssis.run_experiment --exp 1C --stage train` | **1C** — true semi-weak SWSIS (main) |
| `python -m modules.wssis.run_experiment --exp 4A --stage train` | **4A** — YOLOv8-seg semi-weak |

Archived scripts (`run_exp_1b.py`, `run_exp_2a.py`, …): see [report/ARCHIVED_EXPERIMENTS.md](../report/ARCHIVED_EXPERIMENTS.md).


Dry-run (print commands only):

```bash
python -m modules.wssis.run_experiment --exp 1C --stage train --dry-run
```

### Run all experiments — sequential (PLAN order)

```bash
export WSSIS_RUN_ID=wssis_main

# First time: P0 + train all (no eval)
bash scripts/experiments/run_all_experiments.sh --with-p0 --run-id $WSSIS_RUN_ID

# Resume training after interrupt (P0 skipped unless --with-p0)
bash scripts/experiments/run_all_experiments.sh --run-id $WSSIS_RUN_ID --resume

# Train + batch student eval at the end
bash scripts/experiments/run_all_experiments.sh --run-id $WSSIS_RUN_ID --with-eval
```

Order: **1A → 1C → 4A** (active registry). See `python -m modules.wssis.run_experiment --list`.

Outputs: `outputs/runs/<run_id>/experiments/<ID>/`

### Run all experiments — multi-GPU (recommended)

**Hybrid schedule:** Exp **1A** uses **all visible GPUs** first, then **1C** and **4A** on the pool.

```bash
export WSSIS_RUN_ID=wssis_main

# Auto-detect GPU count (e.g. 4 GPUs on your node):
bash scripts/experiments/run_all_experiments.sh --run-id $WSSIS_RUN_ID --parallel

# Or pin explicitly:
bash scripts/experiments/run_all_experiments.sh --run-id $WSSIS_RUN_ID --parallel 4
bash scripts/experiments/run_all_experiments.sh --run-id $WSSIS_RUN_ID --parallel --resume

# Or directly:
bash scripts/experiments/run_experiments_parallel.sh --run-id $WSSIS_RUN_ID
bash scripts/experiments/run_experiments_parallel.sh --jobs 4 --run-id $WSSIS_RUN_ID
```

Override detection: `export WSSIS_GPU_COUNT=4`

| Phase | Experiments | GPUs (example: 4-GPU node) |
| ----- | ----------- | ---------------------------- |
| **1** | **1A** (labeled baseline) | All 4 |
| **2** | **1C** (semi-weak M2F) | All 4 |
| **3** | **4A** (semi-weak YOLO) | 2 |

Logs: `outputs/runs/<id>/logs/parallel/` (`exp_1C.all_gpus.log`, `exp_*.gpuN.log`).

**Before phase 2 / Exp 2C**, compile Mask2Former ops (once per env) and train the no-sym GNN:

```bash
bash scripts/setup/03_compile_mask2former_ops.sh

python -m modules.wssis.prep.train_stage1_gnn --run-id $WSSIS_RUN_ID \
  --symmetric-weight 0 --output-name gnn_refiner_no_sym.pt
```

To run all 11 in the pool without the 1C multi-GPU phase:

```bash
bash scripts/experiments/run_experiments_parallel.sh --no-main-first --run-id $WSSIS_RUN_ID
```

**If a run failed before Mask2Former ops were compiled**, remove false `done` entries from `outputs/runs/<id>/progress.json` (`exp_1C`, `exp_1A`, …) before `--resume`.

---

## Step 6 — Evaluation (after training)

**Primary metric: COCO instance-segmentation mask AP.** IoU/Dice are auxiliary (GNN training only).

Progress tracker: **[CHECKLIST.md](CHECKLIST.md)**

### Teacher eval (once — not per experiment)

Runs automatically at end of P0.4 → `eval/teacher_val_report_full.json`.

Manual re-run only if GNN checkpoint changed:

```bash
bash scripts/eval/run_teacher_eval.sh --run-id $WSSIS_RUN_ID --full-val --skip-if-done
```

Per-signal eval (`boxes_only`, `points_only`, `scribbles_only`) — training-matched for GNN v2.

**Holdout report (matches Stage-1 val / `metrics.jsonl`):**

```bash
bash scripts/eval/run_teacher_eval.sh --run-id $WSSIS_RUN_ID --stage1-holdout
```

→ `eval/teacher_val_report_stage1_holdout.json` (per signal type).

**YOLO (4A):**

```bash
bash scripts/eval/run_yolo_eval.sh 4A --run-id $WSSIS_RUN_ID
```

### Student eval (per experiment — batch after all training)

Fast (20% val subset):

```bash
bash scripts/eval/run_all_experiment_eval.sh --run-id $WSSIS_RUN_ID
# resume skips eval_* steps already done in progress.json:
bash scripts/eval/run_all_experiment_eval.sh --run-id $WSSIS_RUN_ID --resume
```

Single experiment:

```bash
bash scripts/eval/run_experiment_eval.sh 1C --run-id $WSSIS_RUN_ID
```

Full val for report numbers:

```bash
bash scripts/eval/run_all_experiment_eval.sh --run-id $WSSIS_RUN_ID --full-val
```

Teacher eval is **not** run during student eval unless you explicitly pass `--with-teacher-eval` (rare).

### Package for submission (essential artifacts only)

Do **not** `zip -r result.zip outputs` — that pulls in huge training dumps (tensorboard, YOLO export datasets, periodic M2F checkpoints).

```bash
bash scripts/package_results.sh --run-id $WSSIS_RUN_ID -o result.zip
```

**Included:** `checkpoints/best.pt` (GNN), `experiments/*/mask2former/model_best.pth`, `experiments/4A/yolov8_seg/weights/best.pt`, `logs/metrics.jsonl`, `eval/*.json`, `visualizations/` (incl. representative inference grids), `progress.json`, experiment configs.

**Excluded:** `tensorboard/`, `yolo_export/`, `model_0*.pth` periodic checkpoints, other non-essential training cache.

### Logging during training

- `sup_loss`, `semi_loss`, `distill_loss` (Stage-2, when integrated)
- GNN `sym_loss`, `partial_ce`, agreement rate
- GPU memory, time/epoch

Use WandB (optional):

```bash
export WANDB_PROJECT=wssis
wandb login
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
│   ├── run_all_experiments.sh
│   └── run_experiments_parallel.sh
├── scripts/eval/
│   ├── run_teacher_eval.sh
│   ├── run_experiment_eval.sh
│   └── run_all_experiment_eval.sh
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
| `MultiScaleDeformableAttention` import error | `bash scripts/setup/03_compile_mask2former_ops.sh`; verify with `python -c "from modules.wssis.mask2former_ops import verify_msda_import; verify_msda_import()"` |
| Experiments marked `done` but never trained | Edit `outputs/runs/<id>/progress.json` — remove `exp_*` / set `"status": "failed"` — then re-run without `--resume` or delete those keys |
| `--parallel` used 5 GPUs on a 4-GPU node | Use `--parallel` (auto) or `--parallel 4`; scripts now clamp to visible GPU count |


---

## Quick reference (copy-paste)

```bash
conda activate wssis
export WSSIS_REPO_ROOT=$(pwd) PYTHONPATH=$(pwd)
export WSSIS_RUN_ID=wssis_main

# 1) Setup + data + prep (teacher AP included at end of P0.4)
bash scripts/prep/run_p0.sh --run-id $WSSIS_RUN_ID

# 2) Train: 1C on all GPUs, then 10 others @ N parallel (auto-detect, e.g. 4)
bash scripts/experiments/run_all_experiments.sh --run-id $WSSIS_RUN_ID --parallel

# 3) Student eval batch (after all training)
bash scripts/eval/run_all_experiment_eval.sh --run-id $WSSIS_RUN_ID

# Package essentials only (weights + metrics + viz — not full outputs/)
bash scripts/package_results.sh --run-id $WSSIS_RUN_ID -o result.zip

# Full checklist: scripts/CHECKLIST.md
```

