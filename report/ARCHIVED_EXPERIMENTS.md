# Archived experiments (pre–True SWSIS)

These registry IDs are **not** part of the active 5-item report matrix. They remain in code for backward-compatible lookups only (`ARCHIVED_EXPERIMENT_IDS` in `modules/wssis/experiments/registry.py`).

| ID | Original role | Why archived |
|----|---------------|--------------|
| **1B** | 95% raw SAM pseudo baseline | Superseded by teacher eval (Report item 2) |
| **2A** | No GNN ablation | Out of scope for final report |
| **2B** | No distillation ablation | Out of scope for final report |
| **2C** | No symmetric loss GNN | Out of scope for final report |
| **3A** | Boxes-only weak signal | Out of scope for final report |
| **3B** | Points-only weak signal | Out of scope for final report |
| **3C** | Mixed-signal sweep | Default mixed signal kept in 1C/4A |

## Active report matrix (see `report/PLAN.md` §0.1)

| Report # | Registry ID | Description |
|----------|-------------|-------------|
| 1 | **1A** | 5% fully supervised Mask2Former |
| 2 | **P0.4 + teacher eval** | Raw SAM vs SAM+GNN AP on 5% holdout |
| 3 | **1C** | True semi-weak SWSIS (Mask2Former) |
| 4 | **4A** | True semi-weak SWSIS (YOLOv8-seg) |
| 5 | **1D** | 100% GT upper bound (reuse existing full-sup run) |

**Note:** An older run labeled `1C` used standard Mask2Former on `train_all` with full GT — that is **not** true semi-weak SWSIS. Treat it as Report item 5 / upper bound (`1D`).
