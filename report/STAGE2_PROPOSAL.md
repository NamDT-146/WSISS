# Stage-2 SWSIS — Loss Design (WSSIS Implementation)

Joint teacher–student training for semi-weak instance segmentation (Exp **1C** Mask2Former, **4A** YOLOv8-seg).  
SAM encoder + decoder are **frozen**; **GNN refiner** is trainable (small LR). Student receives strong augmentation; teacher receives weak geom augmentation.

---

## 1. Weak-signal jitter (SOTA-aligned defaults)

| Signal | Policy | Default |
|--------|--------|---------|
| **Point** | Manhattan ±`POINT_JITTER_PX` (5 px) | Keeps click inside object |
| **Scribble** | Trim 10–20% from **head or tail** per epoch | `SCRIBBLE_TRIM_RATIO=0.15` |
| **Box** | **Expand only** by `BOX_EXPAND_RATIO` (5% per side) | Outside box = strict background for PCE |

Implementation: [`weak_prompts.py`](../modules/wssis/weak_prompts.py) (`train_online` policy).

---

## 2. Augmentation (dual view)

| View | Pipeline | Used by |
|------|----------|---------|
| **Weak** | Resize → RandomResizedCrop → HFlip | Teacher (SAM cache + decoder + GNN) |
| **Strong** | Weak + RandAugment (n=3, m=7, **no color**) + ImageNet norm | Student |

GT masks, pseudo masks, and weak maps are warped with the same `GeomTransformParams` ([`stage2_augment.py`](../modules/wssis/training/stage2_augment.py)).

---

## 3. Loss terms

Tensors: SAM `sam_masks_3` `[N,3,256,256]`, GNN `refined_logits` `[N,1,256,256]`, weak `weak_signal` `[N,1,256,256]`.

| Symbol | Formula | Valid region |
|--------|---------|--------------|
| **L_pce** | masked BCE(sigmoid(logits), target) | point/scribble: `weak>0`→fg; **box: outside box→bg** (inside ignored) |
| **L_sym** | mean pairwise `soft_dice_symmetric` on **3 SAM heads** | all pixels |
| **L_feedback** | PCE(teacher, student_mask.detach()) | student prob > `FEEDBACK_THRESHOLD` (0.95) |
| **L_sup** | Student CE + mask + dice (M2F criterion / YOLO mask) | labeled 50% — full GT |
| **L_unsup** | Student CE + dice vs **voted pseudo** | weak 50% — SAM 3 heads, thresh 0.9, vote ≥2 |
| **L_semi** | L_pce + partial dice on **student strong** view | jittered weak map |

**Voting:** threshold each SAM head at `PSEUDO_CONFIDENCE_THRESHOLD` (0.9); pixel kept if ≥ `PSEUDO_VOTE_MIN` (2) heads agree.  
**Partial (PCE) losses dominate direction** — pseudo/feedback weights are ramped.

### Combined objectives

```
L_teacher = λ_t1·L_pce + λ_t2·L_sym + λ_t3·L_feedback
L_student = λ_s1·L_sup + λ_s2·L_unsup + λ_s3·L_semi
L_total   = L_teacher + L_student
```

Feature distillation (`loss_distill`) is **removed**.

---

## 4. Weight schedule

| Phase | λ_t1 | λ_t2 | λ_t3 | λ_s1 | λ_s2 | λ_s3 |
|-------|------|------|------|------|------|------|
| Warmup (0–20% steps) | 1.0 | 0.1 | 0.0 | 1.0 | 0.0 | 0.5 |
| Stable (cosine ramp) | 1.0 | 0.1 | 0.05 | 1.0 | 0→1 | 0.5 |

`LossWeightSchedule` in [`stage2_losses.py`](../modules/wssis/training/stage2_losses.py).

---

## 5. Training loop (pseudocode)

```python
for step, batch in enumerate(loader):  # 50% labeled, 50% weak
    w = schedule.weights(step, total_steps)

    for rec in batch:
        dual = build_dual_views(read_image(rec))
        warp_gt_or_oracle_masks(dual.geom)

        if rec.labeled:
            student_targets = full_gt_instances
        else:
            sam3, refined, weak = teacher.forward_trainable(dual.image_weak, ...)
            L_t_pce += partial_bce(refined, weak)
            L_t_sym += symmetric_sam_triplet(sam3)
            pseudo = vote(sam3, thresh=0.9, min=2)
            student_targets = pseudo_instances

        head_out = student_forward(dual.image_strong)
        L_s_sup += m2f_criterion(head_out, student_targets)  # sup + unsup
        L_s_semi += partial_bce(best_query_mask(head_out), jittered_weak)

        L_t_fb += feedback(teacher_logits=refined, student_probs=best_mask)

    loss = w["λ_t1"]*L_t_pce + w["λ_t2"]*L_t_sym + w["λ_t3"]*L_t_fb \
         + w["λ_s1"]*L_s_sup + w["λ_s2"]*L_s_unsup + w["λ_s3"]*L_s_semi
    loss.backward()
    opt_student.step()
    opt_gnn.step()  # GNN only; SAM frozen
```

**Optimizers:** student `BASE_LR` (2e-4 M2F); GNN `GNN_LR=1e-5`.

---

## 6. Code map

| Module | Role |
|--------|------|
| [`stage2_losses.py`](../modules/wssis/training/stage2_losses.py) | PCE, sym, vote, feedback, schedule |
| [`stage2_augment.py`](../modules/wssis/training/stage2_augment.py) | Dual views + mask warp |
| [`stage2_trainer.py`](../modules/wssis/training/stage2_trainer.py) | M2F joint step |
| [`stage2_yolo.py`](../modules/wssis/training/stage2_yolo.py) | YOLO joint loop |
| [`mask2former_teacher.py`](../modules/wssis/mask2former_teacher.py) | `forward_trainable`, frozen SAM |
| [`train_net.py`](../modules/mask2former/train_net.py) | `_run_step_joint`, GNN optimizer |

Config: `WSSIS.USE_STAGE2_JOINT_LOSS=true` (1C/4A).

---

## 7. SAM multimask scale note

SAM’s 3 proposals may differ in spatial extent. **Sym loss** is on SAM heads (scale-tolerant Dice). **Voting** uses per-head threshold then 2/3 consensus — fine-grained heads are not forced to merge; only agreed pixels become pseudo-GT.
