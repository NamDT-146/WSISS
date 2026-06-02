# WSSIS Results Summary

**Run ID:** `wssis_main` (legacy) · **`wssis_v2`** (GNN v2 — rerun with [IMPLEMENTATION.md](IMPLEMENTATION.md))  
**Recorded:** 2026-06-01  
**Active report matrix:** [EXPERIMENT.md](EXPERIMENT.md) — **1A**, **1C**, **1D**, **4A**, plus Stage-1 teacher (GNN).

> **Note:** Numbers below are from **pre–GNN-v2** (`wssis_main`). After `wssis_v2` training, refresh this file from `outputs/runs/wssis_v2/`. Teacher eval is per-signal (no unified mode). YOLO: `bash scripts/eval/run_yolo_eval.sh 4A`.

Primary metric for students: **instance segmentation mask AP** (COCO-style: AP, AP50, AP75, AP_S, AP_M, AP_L).

---

## 1. Headline comparison (student)

Post-train Mask2Former eval (`bash scripts/eval/run_experiment_eval.sh …`) on dataset **`wssis_val_<ID>`** (≈20% val subset, `val_sample_20pct.txt`).  
YOLO **4A** uses Ultralytics val metrics (format differs; scale breakdown may be `nan`).

| Exp | Role | Student | segm AP | AP50 | AP75 | AP_S | AP_M | AP_L |
|-----|------|---------|--------:|-----:|-----:|-----:|-----:|-----:|
| **1A** | Lower bound (5% GT only) | Mask2Former | **2.35** | 5.24 | 1.84 | 0.54 | 1.92 | 5.12 |
| **1C** | Full SWSIS (5% + 95% weak + GNN + distill) | Mask2Former | **6.37** | 12.45 | 5.64 | 1.54 | 6.59 | 10.78 |
| **1D** | Upper bound (100% GT) | Mask2Former | **32.00** | 50.67 | 33.20 | 12.82 | 34.70 | 53.18 |
| **4A** | CNN student (semi-weak) | YOLOv8-seg | **14.55** | 30.43 | — | — | — | — |

**Gap vs upper bound (segm AP):**

| Exp | AP | % of 1D AP |
|-----|---:|-----------:|
| 1A | 2.35 | 7.3% |
| 1C | 6.37 | 19.9% |
| 4A | 14.55 | 45.5% |

**Report narrative hooks:** 1C ≈ **2.7×** 1A on the same val subset → weak supervision + refinement helps, but remains far from 1D → label efficiency vs ceiling. 4A shows cross-architecture signal at mid AP (compare training setup and val protocol before claiming parity with 1C).

---

## 2. Stage-1 teacher (GNN on holdout)

Source: `outputs/runs/wssis_main/eval/teacher_val_report_stage1_holdout_unified.json`  
**Split:** `labeled_5pct_val.txt` (Stage-1 val / early-stop holdout, **not** the same as student `wssis_val_*`).  
**Protocol:** unified mixed weak maps (boxes + points + scribbles), 747 instances.

| Mode | IoU | AP | AP50 |
|------|----:|---:|-----:|
| Raw SAM | 0.182 | 2.07% | 5.09% |
| GNN refined | 0.536 | **29.13%** | **66.00%** |
| **Δ (refined − raw)** | **+0.354** | **+27.05 pp** | **+60.91 pp** |

GNN refinement strongly improves pseudo-mask quality on the holdout before Stage-2; student AP (§1) is on a different split and task (full instance segmentation).

---

## 3. Where these numbers live on disk

| Result | Path |
|--------|------|
| Student Mask2Former logs + copypaste AP | `outputs/runs/wssis_main/experiments/<ID>/mask2former/log.txt` |
| Student predictions | `…/experiments/<ID>/mask2former/inference/coco_instances_results.json` |
| Training-time metrics (incl. optional full-val hook) | `…/experiments/<ID>/mask2former/metrics.json` |
| Teacher JSON (canonical) | `outputs/runs/wssis_main/eval/teacher_val_report_*.json` |
| Upload bundle (subset) | `outputs/runs/wssis_main/report/` |

**Note:** `--full-val` on `run_experiment_eval.sh` currently affects **teacher** eval only (with `--with-teacher-eval`). Student post-train eval uses `wssis_val_<ID>` (20% subset) unless you override `DATASETS.TEST` or use end-of-training `wssis_val_full_<ID>` from `metrics.json`.

---

## 4. Per-experiment detail

### 4.1 Exp 1A — 5% supervised lower bound

```
Evaluation results for segm (wssis_val_1A)
```

| AP | AP50 | AP75 | AP_S | AP_M | AP_L |
|---:|-----:|-----:|-----:|-----:|-----:|
| 2.346 | 5.241 | 1.840 | 0.536 | 1.915 | 5.121 |

<details>
<summary>Per-category segm AP (1A)</summary>

| category | AP | category | AP | category | AP |
|:---------|---:|:---------|---:|:---------|---:|
| person | 7.154 | bicycle | 0.226 | car | 1.396 |
| motorcycle | 2.450 | airplane | 6.847 | bus | 9.693 |
| train | 17.008 | truck | 1.874 | boat | 1.215 |
| traffic light | 0.416 | fire hydrant | 0.000 | stop sign | 0.000 |
| parking meter | 0.000 | bench | 0.130 | bird | 1.083 |
| cat | 16.920 | dog | 2.574 | horse | 4.077 |
| sheep | 1.054 | cow | 2.808 | elephant | 4.531 |
| bear | 9.685 | zebra | 5.667 | giraffe | 5.725 |
| backpack | 0.000 | umbrella | 0.327 | handbag | 0.218 |
| tie | 0.000 | suitcase | 0.278 | frisbee | 6.253 |
| skis | 0.276 | snowboard | 0.240 | sports ball | 1.344 |
| kite | 0.571 | baseball bat | 0.881 | baseball glove | 1.198 |
| skateboard | 0.801 | surfboard | 0.520 | tennis racket | 4.544 |
| bottle | 0.400 | wine glass | 0.000 | cup | 0.435 |
| fork | 0.035 | knife | 0.008 | spoon | 0.000 |
| bowl | 0.120 | banana | 0.104 | apple | 0.049 |
| sandwich | 1.297 | orange | 0.010 | broccoli | 0.267 |
| carrot | 0.010 | hot dog | 0.151 | pizza | 11.615 |
| donut | 0.249 | cake | 0.223 | chair | 0.243 |
| couch | 2.190 | potted plant | 0.081 | bed | 4.363 |
| dining table | 5.655 | toilet | 14.991 | tv | 1.762 |
| laptop | 4.707 | mouse | 5.753 | remote | 0.162 |
| keyboard | 4.227 | cell phone | 0.456 | microwave | 0.396 |
| oven | 0.368 | toaster | — | sink | 1.496 |
| refrigerator | 0.000 | book | 0.016 | clock | 1.785 |
| vase | 0.008 | scissors | 0.000 | teddy bear | 1.716 |
| hair drier | 0.000 | toothbrush | 0.000 | | |

</details>

---

### 4.2 Exp 1C — full SWSIS (main method)

```
Evaluation results for segm (wssis_val_1C)
```

| AP | AP50 | AP75 | AP_S | AP_M | AP_L |
|---:|-----:|-----:|-----:|-----:|-----:|
| 6.374 | 12.450 | 5.635 | 1.543 | 6.594 | 10.775 |

<details>
<summary>Per-category segm AP (1C)</summary>

| category | AP | category | AP | category | AP |
|:---------|---:|:---------|---:|:---------|---:|
| person | 9.143 | bicycle | 1.133 | car | 2.822 |
| motorcycle | 3.975 | airplane | 22.003 | bus | 22.471 |
| train | 33.064 | truck | 5.015 | boat | 2.824 |
| traffic light | 2.819 | fire hydrant | 7.667 | stop sign | 3.539 |
| parking meter | 0.452 | bench | 0.106 | bird | 3.129 |
| cat | 26.131 | dog | 9.404 | horse | 8.767 |
| sheep | 1.003 | cow | 6.148 | elephant | 8.933 |
| bear | 23.182 | zebra | 18.109 | giraffe | 12.789 |
| backpack | 0.030 | umbrella | 3.467 | handbag | 0.069 |
| tie | 0.250 | suitcase | 0.331 | frisbee | 18.627 |
| skis | 1.154 | snowboard | 7.198 | sports ball | 10.274 |
| kite | 1.200 | baseball bat | 2.822 | baseball glove | 9.879 |
| skateboard | 6.860 | surfboard | 3.566 | tennis racket | 20.878 |
| bottle | 1.465 | wine glass | 0.303 | cup | 0.601 |
| fork | 0.404 | knife | 0.010 | spoon | 0.006 |
| bowl | 3.007 | banana | 0.856 | apple | 1.132 |
| sandwich | 1.824 | orange | 0.399 | broccoli | 1.733 |
| carrot | 0.235 | hot dog | 2.605 | pizza | 21.418 |
| donut | 3.977 | cake | 0.987 | chair | 0.921 |
| couch | 5.668 | potted plant | 1.475 | bed | 9.016 |
| dining table | 5.398 | toilet | 25.747 | tv | 3.586 |
| laptop | 16.085 | mouse | 29.363 | remote | 2.728 |
| keyboard | 7.660 | cell phone | 2.507 | microwave | 5.299 |
| oven | 1.404 | toaster | — | sink | 2.532 |
| refrigerator | 2.477 | book | 0.130 | clock | 5.296 |
| vase | 1.007 | scissors | 0.031 | teddy bear | 13.002 |
| hair drier | 0.000 | toothbrush | 0.002 | | |

</details>

---

### 4.3 Exp 1D — 100% supervised upper bound

```
Evaluation results for segm (wssis_val_1D)
```

| AP | AP50 | AP75 | AP_S | AP_M | AP_L |
|---:|-----:|-----:|-----:|-----:|-----:|
| 31.997 | 50.674 | 33.203 | 12.818 | 34.697 | 53.182 |

<details>
<summary>Per-category segm AP (1D)</summary>

| category | AP | category | AP | category | AP |
|:---------|---:|:---------|---:|:---------|---:|
| person | 40.159 | bicycle | 7.862 | car | 30.355 |
| motorcycle | 29.505 | airplane | 51.222 | bus | 62.068 |
| train | 57.407 | truck | 23.966 | boat | 19.721 |
| traffic light | 21.198 | fire hydrant | 56.527 | stop sign | 67.915 |
| parking meter | 38.941 | bench | 12.023 | bird | 15.572 |
| cat | 64.424 | dog | 55.213 | horse | 46.216 |
| sheep | 44.067 | cow | 44.646 | elephant | 53.461 |
| bear | 58.547 | zebra | 57.469 | giraffe | 56.025 |
| backpack | 8.794 | umbrella | 37.761 | handbag | 15.490 |
| tie | 17.495 | suitcase | 19.912 | frisbee | 71.002 |
| skis | 4.240 | snowboard | 18.599 | sports ball | 28.122 |
| kite | 25.659 | baseball bat | 24.614 | baseball glove | 37.501 |
| skateboard | 24.613 | surfboard | 38.973 | tennis racket | 51.449 |
| bottle | 26.307 | wine glass | 28.193 | cup | 32.716 |
| fork | 22.406 | knife | 8.358 | spoon | 4.612 |
| bowl | 27.222 | banana | 15.653 | apple | 14.719 |
| sandwich | 29.830 | orange | 23.846 | broccoli | 17.716 |
| carrot | 17.807 | hot dog | 18.416 | pizza | 39.460 |
| donut | 24.134 | cake | 29.944 | chair | 13.092 |
| couch | 36.659 | potted plant | 11.449 | bed | 28.627 |
| dining table | 19.693 | toilet | 68.290 | tv | 51.323 |
| laptop | 53.756 | mouse | 51.661 | remote | 20.678 |
| keyboard | 36.186 | cell phone | 35.678 | microwave | 57.877 |
| oven | 26.027 | toaster | — | sink | 30.022 |
| refrigerator | 25.505 | book | 5.993 | clock | 30.704 |
| vase | 20.529 | scissors | 34.158 | teddy bear | 43.169 |
| hair drier | 0.000 | toothbrush | 6.639 | | |

</details>

---

### 4.4 Exp 4A — YOLOv8-seg (cross-architecture)

Logged as COCO-style **segm** summary (Ultralytics / export path). AP75 and scale APs not reported in source log.

| AP | AP50 | AP75 | AP_S | AP_M | AP_L |
|---:|-----:|-----:|-----:|-----:|-----:|
| 14.552 | 30.434 | — | — | — | — |

---

## 5. Analysis checklist (for report §Discussion)

- [ ] State **val subset** vs full `val_all` for every table footnote.
- [ ] Do not compare teacher holdout AP (§2) directly to student val AP (§1) without explaining different splits/tasks.
- [ ] Comment on **AP_S** gap (1C AP_S = 1.54 vs 1D = 12.82) — weak prompts vs small objects.
- [ ] Highlight strong 1D classes (person, cat, bus) vs weak 1A/1C tails (fire hydrant, spoon, hair drier).
- [ ] Tie GNN **ΔAP** (§2) to why 1C > 1A, then explain remaining 1C ≪ 1D (student capacity, pseudo-noise, 10k subset).
- [ ] Add qualitative figures: refinement grid, failure cases ([EXPERIMENT.md](EXPERIMENT.md) Phase 3).

---

## 6. Reproduce

```bash
export WSSIS_RUN_ID=wssis_main

# Student (per experiment)
bash scripts/eval/run_experiment_eval.sh 1A --run-id $WSSIS_RUN_ID
bash scripts/eval/run_experiment_eval.sh 1C --run-id $WSSIS_RUN_ID
bash scripts/eval/run_experiment_eval.sh 1D --run-id $WSSIS_RUN_ID
bash scripts/eval/run_experiment_eval.sh 4A --run-id $WSSIS_RUN_ID

# Teacher (Stage-1 holdout, unified)
bash scripts/eval/run_teacher_eval.sh --run-id $WSSIS_RUN_ID --stage1-holdout --unified-weak-maps

# Package for submission
bash scripts/package_results.sh --run-id $WSSIS_RUN_ID -o result.zip
```
