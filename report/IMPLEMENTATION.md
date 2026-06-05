# WSSIS Implementation Map (GNN v2)

Code layout, data artifacts, and train/eval flows for the `wssis_v2` rerun.  
SAM P0.2 embeddings (`data/cache/sam_embeddings/`) are **reused** when ViT-B @ 1024² is unchanged.

---

## Repository layout

| Path | Role |
|------|------|
| [`modules/wssis/`](../modules/wssis/) | Orchestration: splits, Stage-1/2, Mask2Former integration, eval |
| [`modules/vig_refinenet/`](../modules/vig_refinenet/) | SAM + `SamStage1Refiner` GNN |
| [`modules/mask2former/`](../modules/mask2former/) | Student training (`train_net.py`) |
| [`scripts/setup/`](../scripts/setup/) | Data download, P0.1–P0.2 |
| [`scripts/eval/`](../scripts/eval/) | Teacher, student, YOLO eval shells |
| [`data/splits/`](../data/splits/) | Fixed image lists (seed 42) |
| [`data/cache/sam_embeddings/`](../data/cache/sam_embeddings/) | Precomputed `[256,64,64]` npy per image |
| [`checkpoints/`](../checkpoints/) | `sam_vit_b_01ec64.pth`, `gnn_refiner_stage1_v2.pt` |
| [`outputs/runs/<run_id>/`](../outputs/runs/) | Per-run logs, ckpt, eval, experiments |

---

## GNN v2 tensor contract

Per **batch row** (one weak-signal type for one instance):

| Tensor | Shape | Notes |
|--------|-------|--------|
| `sam_embed` | `[B, 256, 64, 64]` | Frozen; node init only |
| `images` | `[B, 3, 1024, 1024]` | RGB `[0,1]`; resized to 256 inside GNN |
| `sam_masks_3` | `[B, 3, 256, 256]` | SAM `multimask_output=True` for typed prompt |
| `weak_signal` | `[B, 1, 256, 256]` | Single channel: point **or** scribble **or** box |
| `mask_logits` | `[B, 1, 256, 256]` | One refined mask per row |

**Stage-1 training:** each GT instance → **3 rows** in the batch (`collate_instance_triplets` in [`coco_image_dataset.py`](../modules/wssis/datasets/coco_image_dataset.py)).

**Stage-2 weak images:** **one** weak type per image from [`weak_95pct_signal.json`](../data/splits/weak_95pct_signal.json) (equal thirds).

Checkpoint: `wssis_ckpt_version=3` in [`stage1.py`](../modules/wssis/training/stage1.py).

---

## Stage-1 losses ([`gnn_losses.py`](../modules/wssis/training/gnn_losses.py))

| Loss | Weight (default) | Description |
|------|------------------|-------------|
| BCE + Dice | 1.0 each | **Main** supervised loss vs GT |
| Hierarchical KL | 0.1 | `KL(point‖scribble)`, `KL(scribble‖box)` on refined probs |
| Triplet symmetric Dice | 0.1 | Consensus across three signal types per instance |
| Intra-SAM symmetric Dice | 0.1 | Consensus among SAM’s 3 multimask heads (same prompt) |
| Weak anchor BCE | 0.05 | Align prediction to weak map |

No KL in Stage-2.

---

## Key modules

### SAM + GNN

- [`sam_stage1_common.py`](../modules/vig_refinenet/sam_stage1_common.py) — encode, `decode_sam_masks_3_batch`, `forward_teacher_objects`, metrics
- [`sam_stage1_refiner.py`](../modules/vig_refinenet/sam_stage1_refiner.py) — `SamStage1Refiner`
- [`weak_prompts.py`](../modules/wssis/weak_prompts.py) — prompts + rasterize 1/3 channels
- [`sam_cache.py`](../modules/wssis/sam_cache.py) — P0.2 npy load

### Stage-1

- [`train_stage1_gnn.py`](../modules/wssis/prep/train_stage1_gnn.py) — CLI / config
- [`stage1.py`](../modules/wssis/training/stage1.py) — train loop, ckpt, viz hook
- [`evaluate_teacher.py`](../modules/wssis/training/evaluate_teacher.py) — per-signal teacher AP
- [`visualize.py`](../modules/wssis/training/visualize.py) — object-aware 1×5 grids (`prompt_space=1024`)

### Stage-2 (joint loss — [STAGE2_PROPOSAL.md](STAGE2_PROPOSAL.md))

| Loss | Weight (stable) | Module |
|------|-----------------|--------|
| Teacher PCE | 1.0 | `stage2_losses.partial_bce_loss` |
| Teacher sym (3 SAM heads) | 0.1 | `symmetric_sam_triplet_loss` |
| Teacher feedback | 0.05 (ramp) | `student_feedback_loss` |
| Student sup (CE+mask+dice) | 1.0 | M2F criterion / YOLO mask |
| Student unsup (voted pseudo) | 0→1 (ramp) | injected pseudo instances |
| Student semi (PCE) | 0.5 | `partial_bce` + `partial_dice` on strong aug |

- [`stage2.py`](../modules/wssis/training/stage2.py) — launch M2F / YOLO
- [`stage2_losses.py`](../modules/wssis/training/stage2_losses.py) — PCE, sym, vote, schedule
- [`stage2_augment.py`](../modules/wssis/training/stage2_augment.py) — dual weak/strong views
- [`stage2_trainer.py`](../modules/wssis/training/stage2_trainer.py) — M2F `_run_step_joint`
- [`stage2_yolo.py`](../modules/wssis/training/stage2_yolo.py) — YOLO joint loop
- [`mask2former_teacher.py`](../modules/wssis/mask2former_teacher.py) — `forward_trainable` (SAM frozen, GNN trainable)
- [`mask2former_mapper.py`](../modules/wssis/mask2former_mapper.py) — defers pseudo when `USE_STAGE2_JOINT_LOSS`
- [`mask2former_dataloader.py`](../modules/wssis/mask2former_dataloader.py) — 50/50 labeled/weak
- [`yolo_export.py`](../modules/wssis/yolo_export.py) — eval/offline export only
- [`evaluate_yolo.py`](../modules/wssis/training/evaluate_yolo.py) — YOLO val metrics JSON

Feature distillation (`wssis_maskformer_distill.py`) is **deprecated**.

### Splits

- [`generate_splits.py`](../modules/wssis/prep/generate_splits.py) — P0.1 lists + `weak_95pct_signal.json` (`--weak-signal-only`)

---

## Data splits (do not resample)

| File | Use |
|------|-----|
| `labeled_5pct_train.txt` / `labeled_5pct_val.txt` | Stage-1 train / early-stop val |
| `weak_95pct.txt` | Stage-2 weak pool (membership fixed) |
| `weak_95pct_signal.json` | `image_id → points_only \| scribbles_only \| boxes_only` |
| `val_sample_20pct.txt` / `val_all.txt` | Fast / full student & teacher eval |

Generate signal map only:  
`python -m modules.wssis.prep.generate_splits --weak-signal-only`

---

## Pseudo-labels → student (joint Stage-2)

1. Weak image: oracle geometry in `wssis_teacher_anns` (prompts only).
2. Teacher weak view: [`forward_teacher_objects_impl`](modules/vig_refinenet/sam_stage1_common.py) → SAM 3 heads + GNN.
3. Unsup pseudo: **vote** 3 SAM heads @ 0.9, keep pixels with ≥2 agreements.
4. Trainer injects pseudo as `instances` on student **strong** view; PCE/semi use jittered weak maps.

Labeled semi-weak half keeps real GT.

---

## Run `wssis_v2` (rerun checklist)

```bash
export WSSIS_RUN_ID=wssis_v2

# P0.1 signal map only (if splits already exist)
python -m modules.wssis.prep.generate_splits --weak-signal-only

# P0.2 — skip if cache intact
# python -m modules.wssis.prep.precompute_sam_embeddings

# P0.4 Stage-1 GNN v2
python -m modules.wssis.prep.train_stage1_gnn --run-id $WSSIS_RUN_ID

# Stage-2 experiments
python -m modules.wssis.run_experiment --exp 1A --stage train --run-id $WSSIS_RUN_ID
python -m modules.wssis.run_experiment --exp 1C --stage train --run-id $WSSIS_RUN_ID
python -m modules.wssis.run_experiment --exp 4A --stage train --run-id $WSSIS_RUN_ID

# Eval
bash scripts/eval/run_teacher_eval.sh --run-id $WSSIS_RUN_ID --stage1-holdout
bash scripts/eval/run_all_experiment_eval.sh --run-id $WSSIS_RUN_ID
bash scripts/eval/run_yolo_eval.sh 4A --run-id $WSSIS_RUN_ID
```

Default dataloader workers: **4** ([`stage2_constants.py`](../modules/wssis/stage2_constants.py), Stage-1 `num_workers=4`). Semi-weak **train** still forces `NUM_WORKERS=0` in mapper (GPU teacher).

---

## Visualization fixes (v2)

- SAM prompts at **1024** coords: `prompt_space=1024` when masks are built on 1024².
- Same `signal_type` for weak overlay, SAM decode, and GNN (`boxes_only` in epoch grids).
- Real `mask1024` for weak tensor rasterization (not empty mask).

Representative figures: [`run_representative.py`](../modules/wssis/inference/run_representative.py).

---

## Teacher metrics note

`RefinementMetricTracker` reports **per-instance** mask IoU and a COCO-threshold proxy AP (not full COCO instance AP). Student Mask2Former uses Detectron2 `COCOEvaluator` (true instance AP).
