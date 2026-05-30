# WSSIS Runbook — Remote GPU Machine

Step-by-step guide to clone, set up conda, download data, run P0 prep, and execute all experiments from [report/EXPERIMENT.md](../report/EXPERIMENT.md) and [report/PLAN.md](../report/PLAN.md).

---

## Prerequisites

| Requirement | Notes |
|-------------|--------|
| Linux + NVIDIA GPU | CUDA 11.8+ recommended |
| Conda (Miniconda/Anaconda) | For env `wssis` |
| Git | Clone this repo |
| Kaggle account | API token → `data/kaggle.json` |
| Disk space | ~200 GB (COCO images + ~23 GB SAM cache + outputs) |

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
1. Create/update env `wssis` from `environment.yml`
2. `pip install -r requirements.txt`
3. Install Detectron2 (CUDA 11.8 / torch 2.1 wheel)
4. Compile Mask2Former `MSDeformAttn` ops
5. Download `checkpoints/sam_vit_b_01ec64.pth`

**Manual fallback** if Detectron2 wheel fails:

```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git'
cd modules/mask2former/mask2former/modeling/pixel_decoder/ops && bash make.sh
```

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

```bash
bash scripts/prep/run_p0.sh
```

Or step-by-step:

| Step | Command | Output |
|------|---------|--------|
| P0.1 | `python -m modules.wssis.prep.generate_splits` | `data/splits/*` |
| P0.2 | `python -m modules.wssis.prep.precompute_sam_embeddings` | `data/cache/sam_embeddings/` (~23 GB) |
| P0.4 | `python -m modules.wssis.prep.train_stage1_gnn --epochs 20` | `checkpoints/gnn_refiner_stage1.pt` |

**Visualizations:** Every epoch writes grids to `outputs/stage1/<run_name>/visualizations/` (5 panels: image, weak signal, raw SAM, GNN refined, GT). Disable with config `"visualization": {"enabled": false}`.

**Debug (small subset):**

```bash
python -m modules.wssis.prep.precompute_sam_embeddings --limit 32
python -m modules.wssis.prep.train_stage1_gnn --epochs 2 --max-instances 500
```

**Exp 2C (no symmetric loss GNN):**

```bash
python -m modules.wssis.prep.train_stage1_gnn \
  --symmetric-weight 0 \
  --output-name gnn_refiner_no_sym.pt
```

### Known limitation (PLAN §0.5)

Stage-1 currently trains GNN on SAM embeddings only (no SAM prompt decoder / 3-mask refinement). Fix before reporting Stage-1 metrics. Stage-2 experiment configs are still generated correctly.

---

## Step 5 — Run experiments

List experiments:

```bash
python -m modules.wssis.run_experiment --list
```

### Main result first (Exp 1C)

```bash
python scripts/experiments/run_exp_1c.py
# or
python -m modules.wssis.run_experiment --exp 1C --stage train
```

### Single experiment

| Script | Experiment |
|--------|------------|
| `scripts/experiments/run_exp_1a.py` | 1A — 5% supervised lower bound |
| `scripts/experiments/run_exp_1b.py` | 1B — raw SAM weak baseline |
| `scripts/experiments/run_exp_1c.py` | **1C — full SWSIS (main)** |
| `scripts/experiments/run_exp_1d.py` | 1D — 100% upper bound |
| `scripts/experiments/run_exp_2a.py` | 2A — no GNN |
| `scripts/experiments/run_exp_2b.py` | 2B — no distillation |
| `scripts/experiments/run_exp_2c.py` | 2C — no symmetric loss |
| `scripts/experiments/run_exp_3a.py` | 3A — boxes only |
| `scripts/experiments/run_exp_3b.py` | 3B — points only |
| `scripts/experiments/run_exp_3c.py` | 3C — mixed signals |
| `scripts/experiments/run_exp_4a.py` | 4A — YOLOv8-seg |

Dry-run (print commands only):

```bash
python -m modules.wssis.run_experiment --exp 1C --stage train --dry-run
```

### Run all experiments (PLAN order)

```bash
bash scripts/experiments/run_all_experiments.sh --with-p0   # includes P0 if needed
# without P0:
bash scripts/experiments/run_all_experiments.sh
```

Order: **1C → 1A → 1B → 1D → 2A → 2B → 2C → 3A → 3B → 3C → 4A**

Outputs: `outputs/experiments/<ID>/experiment_config.json` (+ Mask2Former/YOLO artifacts)

---

## Step 6 — Logging & evaluation

During training (see [EXPERIMENT.md](../report/EXPERIMENT.md)):
- `sup_loss`, `semi_loss`, `distill_loss`
- GNN `sym_loss`, `partial_ce`, agreement rate
- GPU memory, time/epoch

Use WandB:

```bash
export WANDB_PROJECT=wssis
wandb login
```

After each experiment, run COCO eval (AP, AP50, AP75, AP_S/M/L) and save qualitative grids.

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
├── checkpoints/
│   ├── sam_vit_b_01ec64.pth
│   └── gnn_refiner_stage1.pt
├── outputs/experiments/
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

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: modules` | `export PYTHONPATH=$PWD` from repo root |
| Kaggle 403 | Check `data/kaggle.json` permissions and API token |
| CUDA OOM on P0.2 | `--limit 100` for testing; reduce batch size |
| Detectron2 import error | Re-run `00_create_conda_env.sh` or install matching wheel |
| Mask2Former config missing | Add COCO configs under `modules/mask2former/configs/coco/` |

---

## Quick reference (copy-paste)

```bash
conda activate wssis
export WSSIS_REPO_ROOT=$(pwd) PYTHONPATH=$(pwd)

bash scripts/setup/00_create_conda_env.sh
cp ~/.kaggle/kaggle.json data/kaggle.json
bash scripts/setup/01_download_data.sh
bash scripts/prep/run_p0.sh
python scripts/experiments/run_exp_1c.py
```
