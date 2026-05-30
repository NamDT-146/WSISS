# SWSIS — Semi-Weakly Supervised Instance Segmentation

SAM + GNN Refiner + Feature Distillation → Mask2Former (or YOLOv8-seg) on COCO minitrain with a fixed 5% / 95% split.

**Run on a remote GPU machine** (clone from GitHub). Full step-by-step instructions: **[scripts/RUNBOOK.md](scripts/RUNBOOK.md)**.

## Documentation

| Doc | Purpose |
|-----|---------|
| [scripts/RUNBOOK.md](scripts/RUNBOOK.md) | **Start here** — conda, Kaggle data, P0, experiments |
| [report/PLAN.md](report/PLAN.md) | Architecture, prep artifacts, training loops |
| [report/EXPERIMENT.md](report/EXPERIMENT.md) | Experiment matrix, logging, figures |
| [report/PREPARATION.md](report/PREPARATION.md) | Dataset & weak-prompt policies |
| [report/ARCHITECTURE.md](report/ARCHITECTURE.md) | SAM vs Mask2Former specs |
| [report/RANDOM_NOTE.md](report/RANDOM_NOTE.md) | GPU allocation (1 GPU SAM, rest for student) |

## Quick start

```bash
git clone <repo-url> wssis && cd wssis
cp ~/.kaggle/kaggle.json data/kaggle.json

bash scripts/setup/00_create_conda_env.sh
conda activate wssis
export WSSIS_REPO_ROOT=$PWD PYTHONPATH=$PWD

bash scripts/setup/01_download_data.sh    # COCO + minitrain-10k → data/
bash scripts/prep/run_p0.sh               # splits, SAM cache, GNN ckpt
python scripts/experiments/run_exp_1c.py  # main experiment
```

## Project layout

```
wssis/
├── environment.yml          # conda env `wssis`
├── requirements.txt         # pip deps (SAM, ultralytics, kaggle, …)
├── data/
│   ├── kaggle.json          # your token (gitignored)
│   ├── coco2017/
│   ├── coco_minitrain_10k/
│   ├── splits/              # P0.1 fixed 5%/95% lists
│   └── cache/sam_embeddings/
├── checkpoints/
├── outputs/experiments/
├── modules/
│   ├── wssis/               # unified prep + experiment runner
│   ├── vig_refinenet/       # Stage-1 GNN + SAM helpers
│   ├── mask2former/         # Stage-2 student (Detectron2)
│   └── segment-anything/
└── scripts/
    ├── setup/               # conda, Kaggle download, SAM weights
    ├── prep/                # P0 pipeline
    └── experiments/         # run_exp_1a.py … run_exp_4a.py, run_all
```

## Experiments

| ID | Script | Description |
|----|--------|-------------|
| **1C** | `run_exp_1c.py` | **Main** — full SWSIS |
| 1A | `run_exp_1a.py` | Lower bound — 5% GT only |
| 1B | `run_exp_1b.py` | Weak baseline — raw SAM |
| 1D | `run_exp_1d.py` | Upper bound — 100% GT |
| 2A–2C | `run_exp_2a.py` … | Ablations |
| 3A–3C | `run_exp_3a.py` … | Signal sensitivity |
| 4A | `run_exp_4a.py` | YOLOv8-seg |

Run all: `bash scripts/experiments/run_all_experiments.sh`

CLI: `python -m modules.wssis.run_experiment --exp 1C --stage train`

## Preparation (P0)

| Step | Module / script |
|------|-----------------|
| P0.1 Fixed splits | `python -m modules.wssis.prep.generate_splits` |
| P0.2 SAM embeddings | `python -m modules.wssis.prep.precompute_sam_embeddings` |
| P0.4 Stage-1 GNN | `python -m modules.wssis.prep.train_stage1_gnn` |

## Stage-1 visualizations

Each training epoch saves **1×5 refinement grids** under `outputs/stage1/<run_name>/visualizations/`:

`Image | Weak signal | Raw SAM | GNN refined (pseudo) | GT`

Plus an `epoch_XXX_montage.png` combining all samples for that epoch.

## Known gap (Stage-1)

The GNN notebook/prototype trains on SAM embeddings only (no SAM decoder + 3-mask refinement). See [report/PLAN.md §0.5](report/PLAN.md). Fix before trusting Stage-1 metrics or Stage-2 teacher quality.

## Environment

- **Conda env:** `wssis` (Python 3.10, PyTorch 2.1, CUDA 11.8)
- **Detectron2:** installed by `scripts/setup/00_create_conda_env.sh`
- **Kaggle:** credentials in `data/kaggle.json`; downloads into `data/`
