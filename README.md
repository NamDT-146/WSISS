# SWSIS — Semi-Weakly Supervised Instance Segmentation

**Problem:** Pixel-perfect instance masks are expensive; weak signals (points, boxes, scribbles) are cheap.  
**Goal:** Train a **student** segmenter (Mask2Former / YOLOv8-seg) on **5% fully labeled + 95% weak** COCO data, using SAM + a GNN refiner + feature distillation.  
**Primary metric:** **COCO instance-segmentation mask AP** (AP, AP50, AP75, AP_S, AP_M, AP_L).

This README is the **single entry point** for teammates running experiments and writing the project report. Deep dives live in `report/` and operations in `scripts/`.

---

## For teammates — start here

| Role | Read first | Then do |
|------|------------|---------|
| **Run experiments** | [scripts/RUNBOOK.md](scripts/RUNBOOK.md) | [scripts/CHECKLIST.md](scripts/CHECKLIST.md) |
| **Write report / slides** | Sections below + [report/EXPERIMENT.md](report/EXPERIMENT.md) | Fill checklist §F |
| **Architecture / math** | [report/PLAN.md](report/PLAN.md) | Methodology section |
| **Dataset & weak signals** | [report/PREPARATION.md](report/PREPARATION.md) | Data section |

```bash
git clone <repo-url> wssis && cd wssis
cp ~/.kaggle/kaggle.json data/kaggle.json

bash scripts/setup/00_create_conda_env.sh && conda activate wssis
export WSSIS_REPO_ROOT=$PWD PYTHONPATH=$PWD WSSIS_RUN_ID=wssis_main

bash scripts/setup/01_download_data.sh
bash scripts/prep/run_p0.sh --run-id $WSSIS_RUN_ID
bash scripts/eval/run_teacher_eval.sh --run-id $WSSIS_RUN_ID
python scripts/experiments/run_exp_1c.py --run-id $WSSIS_RUN_ID
```

Track progress: **[scripts/CHECKLIST.md](scripts/CHECKLIST.md)**

---

## 1. Problem formulation (for report §Introduction)

- **Fully supervised** instance segmentation needs costly polygon/mask annotations.
- **Weak supervision** uses cheap cues: one click, a bounding box, or a scribble.
- **Teacher:** SAM ViT-B produces mask proposals from weak prompts.
- **Gap:** Raw SAM boundaries are noisy; weak labels alone are insufficient for a strong student.
- **Our method (SWSIS):** GNN refines SAM masks → pseudo-GT for 95% data; feature distillation aligns Mask2Former (Swin-T) with SAM embeddings; 5% GT anchors quality.

**Bounds to report (Experiments 1A–1D):**

| Exp | Setting | Expected role in report |
|-----|---------|-------------------------|
| 1A | 5% GT only | Lower bound |
| 1B | 95% raw SAM pseudo | Weak baseline |
| **1C** | **Full SWSIS** | **Main result** |
| 1D | 100% GT | Upper bound |

Plot **annotation cost vs AP** for the presentation (see EXPERIMENT.md Phase 4).

---

## 2. Method overview (for report §Methodology)

### Stage 0 — Preparation (once)

Fixed splits (seed=42, image-level), SAM embedding cache, Stage-1 GNN on 5% labeled data.

```
data/splits/          labeled_5pct.txt, weak_95pct.txt, val_prompts_fixed.json
data/cache/sam_embeddings/   [256,64,64] fp16 per image
checkpoints/gnn_refiner_stage1.pt
```

### Stage 1 — GNN warm-up (5% labeled only)

**Training data:** `labeled_5pct_train.txt` (~80% of the 5% labeled image pool).  
**In-loop validation:** `labeled_5pct_val.txt` (holdout from the same 5% pool — not full val).  
**Final eval:** full `val_all.txt` after training (see `teacher_val_report_full.json`).  
**Not used in Stage 1:** `weak_95pct.txt` (Stage-2). Routine experiment eval uses `val_sample_20pct.txt` (~20% of val).

Pipeline ([report/PLAN.md](report/PLAN.md) §2):

1. Cached SAM embedding (frozen encoder)
2. Weak prompts → SAM decoder → **3 mask proposals**
3. **GNN refiner** inputs:
   - `sam_embed` → **first-layer node initialization only**
   - RGB **image**
   - **3 SAM masks**
   - **Weak-signal maps** (3 channels: point, box, scribble)
4. Output: **3 refined masks**; losses = Dice + BCE + symmetric loss

**Unified training:** all three weak-map channels are active.  
**Eval:** report **AP** separately for `boxes_only`, `points_only`, `scribbles_only`.

**Weak-signal rasterization** ([modules/wssis/weak_prompts.py](modules/wssis/weak_prompts.py)):

| Signal | 1×H×W map |
|--------|-----------|
| Point | Gaussian blob (σ=4 px) |
| Box | Uniform 1.0 inside bbox |
| Scribble | Line along principal axis + Gaussian widen |

Train: online jitter / random scribble length. Val: fixed prompts (`val_prompts_fixed.json`).

### Stage 2 — Semi-weak student training

- Batch: 50% `labeled_5pct` (GT) + 50% `weak_95pct` (GNN pseudo-GT, 2/3 mask agreement)
- **Student:** Mask2Former Swin-T @ 640² (main) or YOLOv8-seg (Exp 4A)
- **Losses:** supervised + semi-supervised + feature distillation (MSE on stride-16 vs SAM)
- **Eval:** **student forward only** → COCO mask **AP** on val (no SAM/GNN at test time)

Teacher stack (SAM + GNN) is also evaluated separately as a **pseudo-label quality baseline** — see teacher AP report.

### Architecture specs

| Model | Input | Encoder output | Embed dim |
|-------|-------|----------------|-----------|
| SAM ViT-B | 1024² | 64×64 | 256 |
| Mask2Former Swin-T | 640² | multi-scale (strides 8/16/32) | 256 (after projection) |

Details: [report/ARCHITECTURE.md](report/ARCHITECTURE.md)

---

## 3. Experiments matrix (for report §Experiments)

Full table: [report/PLAN.md](report/PLAN.md) §0.1

| ID | Category | What changes |
|----|----------|--------------|
| 1A–1D | Bounds | Supervision amount / raw SAM vs full method |
| 2A | Ablation | No GNN |
| 2B | Ablation | No feature distillation |
| 2C | Ablation | No symmetric loss (separate GNN ckpt) |
| 3A–3C | Signal | Box / point / mixed weak signals in Stage 2 |
| 4A | Generalize | YOLOv8-seg student |

Run order: **1C → 1A → 1B → 1D → 2A → 2B → 2C → 3A → 3B → 3C → 4A**

Scripts: `scripts/experiments/run_exp_<id>.py` or  
`python -m modules.wssis.run_experiment --exp 1C --stage all --run-id $WSSIS_RUN_ID`

---

## 4. Metrics (for report §Results) — AP first

### Primary: instance-segmentation AP

All final numbers should be **COCO mask AP** on the validation split (minitrain val list ∩ COCO JSON).

| Eval target | Metrics | Command / source |
|-------------|---------|------------------|
| **Teacher raw SAM** | AP, AP50 per signal type | `bash scripts/eval/run_teacher_eval.sh --raw-only` |
| **Teacher GNN refined** | raw AP, refined AP, **ΔAP** | `bash scripts/eval/run_teacher_eval.sh` |
| **Student (Mask2Former)** | AP, AP50, AP75, AP_S, AP_M, AP_L | Detectron2 eval on experiment ckpt |

Teacher report JSON: `outputs/runs/<run_id>/eval/teacher_val_report.json`

Stage-1 training logs (per epoch) in `logs/metrics.jsonl`:

- **Loss (all components):** `train_*` / `val_*` for `bce_raw`, `bce_weighted`, `dice_raw`, `dice_weighted`, `seg_weighted`, `sym_raw`, `sym_weighted`, `total` (legacy: `train_loss`, `train_bce_loss`, …)
- **AP (primary for early stop):** `raw_sam_ap`, `val_refined_ap`, `delta_ap`

Early stopping uses **`val_refined_ap`**.

### Secondary (analysis only)

- IoU / Dice during GNN training  
- Stage-2: `sup_loss`, `semi_loss`, `distill_loss` (when integrated)  
- GNN agreement rate (≥2/3 masks)  
- GPU memory, time/epoch  

### Report tables to fill

1. **Teacher:** Raw SAM vs GNN AP × (box, point, scribble)  
2. **Student bounds:** 1A, 1B, 1C, 1D AP  
3. **Ablations:** 2A, 2B, 2C vs 1C  
4. **Signals:** 3A, 3B, 3C vs 3C/1C  
5. **Architecture:** 4A vs 1C  

---

## 5. Figures (for report & slides)

From [report/EXPERIMENT.md](report/EXPERIMENT.md):

| Figure | Description | Where to get it |
|--------|-------------|-----------------|
| Refinement grid | Image → weak signal → raw SAM → GNN → GT | `outputs/runs/<id>/visualizations/` |
| AP vs annotation cost | Bar chart | Compile from AP tables |
| Ablation bars | 2A–2C | Student AP |
| Failure cases | 3–5 bad predictions | Manual selection |
| Loss curves | Training stability | TensorBoard / WandB |
| Pipeline diagram | Stage 1 + 2 | Draw from PLAN §2–3 |

Optional: t-SNE of stride-16 features with/without distillation.

---

## 6. Project layout

```
wssis/
├── README.md                 ← this file
├── scripts/
│   ├── RUNBOOK.md            ← operational steps
│   ├── CHECKLIST.md          ← progress tracker
│   ├── setup/                ← env + data + SAM weights
│   ├── prep/run_p0.sh        ← P0 pipeline
│   ├── eval/                 ← teacher AP eval
│   └── experiments/          ← run_exp_*.py, run_all
├── report/
│   ├── PLAN.md               ← architecture + training loops + math
│   ├── EXPERIMENT.md         ← experiments, logging, deliverables
│   ├── PREPARATION.md        ← dataset + weak-signal policy
│   ├── ARCHITECTURE.md       ← SAM vs Mask2Former specs
│   └── DATA.md               ← EDA notes
├── data/                     ← COCO, splits, SAM cache (not in git)
├── checkpoints/              ← GNN + SAM weights
├── outputs/runs/<run_id>/    ← logs, ckpt, eval, report bundle
└── modules/
    ├── wssis/                ← orchestration, weak prompts, eval
    ├── vig_refinenet/        ← GNN + SAM helpers
    ├── mask2former/          ← student
    └── segment-anything/     ← SAM
```

---

## 7. Run bundle (artifacts for the report)

```
outputs/runs/<run_id>/
├── progress.json
├── logs/metrics.jsonl        ← Stage-1 AP per epoch
├── checkpoints/best.pt
├── visualizations/           ← refinement grids
├── eval/teacher_val_report.json
├── experiments/<ID>/         ← Stage-2 outputs
└── report/                   ← zip this for submission
```

Set `export WSSIS_RUN_ID=your_team_run` for one consistent folder.

Optional: `export WANDB_PROJECT=wssis`

---

## 8. Discussion prompts (for report §Analysis)

Use AP gaps, not only absolute scores:

- Why does GNN refinement improve **ΔAP** over raw SAM? (boundaries, agreement filter)
- Why is distillation needed? (2B ablation — feature alignment)
- Why symmetric loss? (2C — mask consistency)
- Which weak signal is weakest on **AP_S** (small objects)? (3A–3C)
- How close is 1C to 1D (upper bound) at ~5% labeling cost?
- Failure modes: occlusion, shadows, ambiguous clicks

Include formulas from PLAN §3–4: \(L_{\text{distill}}\), symmetric loss, partial CE.

---

## 9. Known limitations (honest report footnotes)

| Item | Status |
|------|--------|
| Stage-2 Mask2Former full SWSIS loop (pseudo + distill in training) | Incrementally integrated; flags in experiment YAML |
| Automated student COCO AP export from WSSIS CLI | Use Detectron2 `--eval-only` on saved ckpt |
| GNN checkpoints before 3-channel weak-input refactor | Invalid — re-run P0.4 |

See [scripts/CHECKLIST.md](scripts/CHECKLIST.md) §G for full list.

---

## 10. Documentation index

| Document | Use when writing… |
|----------|-------------------|
| [scripts/RUNBOOK.md](scripts/RUNBOOK.md) | Setup & commands |
| [scripts/CHECKLIST.md](scripts/CHECKLIST.md) | Tracking what's done |
| [report/PLAN.md](report/PLAN.md) | Method, losses, pipeline |
| [report/EXPERIMENT.md](report/EXPERIMENT.md) | Experiment list, figures, slide outline |
| [report/PREPARATION.md](report/PREPARATION.md) | Dataset, weak-signal policy |
| [report/ARCHITECTURE.md](report/ARCHITECTURE.md) | SAM / Mask2Former dimensions |
| [report/DATA.md](report/DATA.md) | COCO EDA |

---

## Environment

- Conda env: `wssis` (Python 3.10, PyTorch 2.6, CUDA 12.4 — see `requirements.txt`)
- GPU: 1 GPU for SAM teacher inference; remaining GPUs for student (`WSSIS_NUM_GPUS`)
- Kaggle API: `data/kaggle.json` for dataset download

```bash
bash scripts/setup/00_create_conda_env.sh
```

---

## License & attribution

SAM (Meta, Apache 2.0), Mask2Former (Detectron2), COCO dataset — cite in report references section.
