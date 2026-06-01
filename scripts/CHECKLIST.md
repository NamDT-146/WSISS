# WSSIS project checklist

**Primary evaluation metric: instance-segmentation mask AP** (COCO-style; AP50/AP75/AP_S/AP_M/AP_L for full student eval).

Use this file to track progress before the written report and presentation. Check items when done; note output paths in the **Evidence** column.

---

## A. Environment & data

| Done | Item | Command / location | Evidence |
|------|------|-------------------|----------|
| ☐ | Conda env `wssis` | `bash scripts/setup/00_create_conda_env.sh` | `conda activate wssis` works |
| ☐ | Kaggle token | `data/kaggle.json` | Downloads succeed |
| ☐ | COCO 2017 | `bash scripts/setup/01_download_data.sh` | `data/coco2017/annotations/instances_train2017.json` |
| ☐ | coco-minitrain-10k | same | `data/coco_minitrain_10k/train2017.txt` |
| ☐ | SAM ViT-B weights | setup script | `checkpoints/sam_vit_b_01ec64.pth` |
| ☐ | Detectron2 + Mask2Former ops | setup script | `import detectron2` OK |
| ☐ | `PYTHONPATH` / `WSSIS_REPO_ROOT` | export from repo root | imports `modules.wssis` OK |

---

## B. P0 preparation (once per machine / split)

| Done | Step | Command | Output |
|------|------|---------|--------|
| ☐ | **P0.1** Fixed splits (seed=42) | `python -m modules.wssis.prep.generate_splits` | `labeled_5pct_train/val.txt`, `val_sample_20pct.txt`, `weak_95pct.txt` |
| ☐ | **P0.2** SAM embeddings cache | `python -m modules.wssis.prep.precompute_sam_embeddings` | `data/cache/sam_embeddings/` (~23 GB) |
| ☐ | **P0.4** Stage-1 GNN (5% labeled) | `bash scripts/prep/run_p0.sh --run-id $WSSIS_RUN_ID` | `checkpoints/gnn_refiner_stage1.pt`, `outputs/runs/<id>/checkpoints/best.pt` |
| ☐ | **P0.4b** GNN without sym loss (Exp 2C) | `train_stage1_gnn --symmetric-weight 0 --output-name gnn_refiner_no_sym.pt` | `checkpoints/gnn_refiner_no_sym.pt` |

**Stage-1 training design (current):**

- SAM embed → **node init only** (not sole input)
- GNN inputs: RGB image + 3 SAM masks + weak-signal maps `[point | box | scribble]`
- **Stage-1 train:** `labeled_5pct_train.txt` | **Stage-1 val (early stop):** `labeled_5pct_val.txt` | **Final teacher eval:** full `val_all`
- **Stage-2 routine eval:** `val_sample_20pct.txt` | **Final:** `--full-val`
- Unified training: all 3 map channels populated on 5% GT
- Early stopping on **`val_refined_ap`** (primary)

**Weak-signal 2D maps** (`modules/wssis/weak_prompts.py`):

| Type | Map rule |
|------|----------|
| Point | Gaussian blob (σ=4 px) |
| Box | Uniform 1.0 inside bbox |
| Scribble | Principal-axis line + Gaussian widen |

---

## C. Teacher evaluation (val set — AP report)

Run after P0.4. **No extra manual labeling step** — uses fixed val prompts + online SAM decoder.

| Done | Item | Command | Output |
|------|------|---------|--------|
| ☐ | Raw SAM AP (3 signal types) | `bash scripts/eval/run_teacher_eval.sh --run-id $ID --raw-only` | `eval/teacher_val_report.json` → `results.raw_sam.*` |
| ☐ | GNN-refined AP (training-matched) | `bash scripts/eval/run_teacher_eval.sh --run-id $ID --stage1-holdout --unified-weak-maps` | `eval/teacher_val_report_stage1_holdout_unified.json` → `unified_mixed` + **ΔAP** |
| ☐ | Per-signal ablation (raw SAM) | `bash scripts/eval/run_teacher_eval.sh --run-id $ID --full-val` | `results.raw_sam.*` (not for GNN claims) |
| ☐ | Per-experiment student eval | `bash scripts/eval/run_experiment_eval.sh 1C --run-id $ID` | student AP hook |
| ☐ | Batch student eval (all exps) | `bash scripts/eval/run_all_experiment_eval.sh --run-id $ID` | after training sweep |

**Signal types reported:** `boxes_only`, `points_only`, `scribbles_only`

**Metrics per type (teacher):**

| Mode | Primary | Also log |
|------|---------|----------|
| raw_sam | **AP**, AP50 | IoU |
| gnn_refined | **refined_AP**, **ΔAP** | raw_AP, AP50, ΔAP50, IoU |

**Report table template (fill from JSON):**

| Signal | Raw SAM AP | GNN refined AP | ΔAP |
|--------|------------|----------------|-----|
| Box | | | |
| Point | | | |
| Scribble | | | |

---

## D. Stage-2 experiments (student — mask AP on val)

| Done | ID | Command | Config highlight | Student AP |
|------|-----|---------|------------------|------------|
| ☐ | **1A** | `run_experiment --exp 1A --stage train` | 5% GT labeled-only | |
| ☐ | **1C** | `run_experiment --exp 1C --stage train` | True semi-weak SWSIS (main) | |
| ☐ | **4A** | `run_experiment --exp 4A --stage train` | YOLOv8-seg semi-weak | |
| ☐ | **1D** | *(reuse existing run)* | 100% GT upper bound | |

Smoke test: `bash scripts/experiments/run_smoke_test.sh`

Run all (train): `bash scripts/experiments/run_all_experiments.sh --run-id $WSSIS_RUN_ID`
Parallel: `bash scripts/experiments/run_experiments_parallel.sh --run-id $WSSIS_RUN_ID`
Student eval batch: `bash scripts/eval/run_all_experiment_eval.sh --run-id $WSSIS_RUN_ID --full-val`

**Student eval (Mask2Former):** COCO **AP, AP50, AP75, AP_S, AP_M, AP_L** on val — wired via `evaluate_experiment()` → `--eval-only` on `model_best.pth`.

| Done | Item | Status |
|------|------|--------|
| ☐ | Mask2Former `--eval-only` wired in WSSIS | ✅ `stage2.evaluate_experiment` |
| ☐ | Results logged per experiment | `outputs/runs/<id>/experiments/<EXP>/` |

---

## E. Logging during training (for report analysis)

| Metric | Stage | Where |
|--------|-------|-------|
| `train_bce_raw`, `train_bce_weighted`, `train_dice_raw`, `train_dice_weighted`, `train_seg_weighted`, `train_sym_raw`, `train_sym_weighted`, `train_total` | Stage-1 GNN (train) | `logs/metrics.jsonl` |
| `val_bce_raw`, `val_bce_weighted`, `val_dice_raw`, `val_dice_weighted`, `val_seg_weighted`, `val_total` | Stage-1 GNN (val) | same |
| Legacy aliases: `train_loss`, `train_bce_loss`, `train_dice_loss`, `train_sym_loss`, `val_*` | Stage-1 GNN | same |
| **`raw_sam_ap`, `val_refined_ap`, `delta_ap`** | Stage-1 GNN (primary AP) | same |
| `sup_loss`, `semi_loss`, `distill_loss` | Stage-2 | WandB / TB (when integrated) |
| GNN agreement rate | Stage-2 | WandB |
| GPU mem, time/epoch | both | `metrics.jsonl` |

Optional: `export WANDB_PROJECT=wssis`

---

## F. Figures & report deliverables ([report/EXPERIMENT.md](../report/EXPERIMENT.md))

| Done | Deliverable | Source |
|------|-------------|--------|
| ☐ | Refinement pipeline 1×5 grid | `outputs/runs/<id>/visualizations/` |
| ☐ | AP bar chart (annotation cost vs AP) | From experiment AP table |
| ☐ | Ablation table (2A–2C) | Student AP |
| ☐ | Teacher raw vs GNN AP table | `teacher_val_report.json` |
| ☐ | Signal sensitivity (3A–3C) | Student AP by signal |
| ☐ | Failure cases (3–5 images) | Manual export |
| ☐ | Loss curves | TensorBoard / WandB |
| ☐ | Method diagram (Stage 1 + 2) | [report/PLAN.md](../report/PLAN.md) |
| ☐ | Math: L_distill, symmetric loss, partial CE | [report/PLAN.md](../report/PLAN.md) §3–4 |

---

## G. Known gaps / before submission

| Item | Status | Action |
|------|--------|--------|
| Feature distillation loss in M2F trainer | ✅ | `loss_distill` on weak images via `res4` → projector → SAM cache |
| Old GNN checkpoints (pre-image-level) | ❌ Invalid | Re-run P0.4 (`wssis_ckpt_version=2`) |
| Old progress `exp_1C` from mislabeled full-sup run | ⚠️ | Rename/delete before new true 1C |

---

## H. One-run copy-paste (happy path)

```bash
conda activate wssis
export WSSIS_REPO_ROOT=$PWD PYTHONPATH=$PWD WSSIS_RUN_ID=wssis_main

bash scripts/prep/run_p0.sh --run-id $WSSIS_RUN_ID
bash scripts/experiments/run_all_experiments.sh --run-id $WSSIS_RUN_ID --parallel 5
bash scripts/eval/run_all_experiment_eval.sh --run-id $WSSIS_RUN_ID

# zip essentials (weights, metrics, viz — not entire outputs/)
bash scripts/package_results.sh --run-id $WSSIS_RUN_ID -o result.zip
```

---

## I. File map (where teammates look)

| What | Path |
|------|------|
| Run bundle | `outputs/runs/<run_id>/` |
| Teacher AP report | `outputs/runs/<run_id>/eval/teacher_val_report.json` |
| GNN best ckpt | `outputs/runs/<run_id>/checkpoints/best.pt` |
| Stage-1 metrics | `outputs/runs/<run_id>/logs/metrics.jsonl` |
| Experiment outputs | `outputs/runs/<run_id>/experiments/<ID>/` |
| Report upload folder | `outputs/runs/<run_id>/report/` |

Last updated: aligns with PLAN §0.5 fix, unified weak maps, and `evaluate_teacher` module.
