## 0. Experiment Registry & Preparation

All experiments share frozen preparation artifacts (§0.1). Only the columns marked *varies* change between runs. Full logging and figure requirements: [EXPERIMENT.md](EXPERIMENT.md).

### 0.1 Experiment matrix

| ID | Category | Student | Labeled | Weak | GNN | Distill | Sym loss | Signal |
|----|----------|---------|---------|------|-----|---------|----------|--------|
| 1A | Baseline | M2F | 5% GT | — | — | — | — | — |
| 1B | Baseline | M2F | — | 95% raw SAM | — | — | — | mixed |
| **1C** | **Main** | **M2F** | **5% GT** | **95% pseudo** | **shared ckpt** | **yes** | **yes** | **mixed** |
| 1D | Upper bound | M2F | 100% GT | — | — | — | — | — |
| 2A | Ablation | M2F | 5% / 95% | pseudo | **off** | yes | — | mixed |
| 2B | Ablation | M2F | 5% / 95% | pseudo | shared | **off** | yes | mixed |
| 2C | Ablation | M2F | 5% / 95% | pseudo | shared* | yes | **off** | mixed |
| 3A | Signal | M2F | 5% / 95% | pseudo | shared | yes | yes | **boxes only** |
| 3B | Signal | M2F | 5% / 95% | pseudo | shared | yes | yes | **points only** |
| 3C | Signal | M2F | 5% / 95% | pseudo | shared | yes | yes | mixed |
| 4A | Generalize | YOLOv8-seg | 5% / 95% | pseudo | shared | yes | yes | mixed |

\*Exp 2C: if GNN is frozen after Stage 1 (trained *with* symmetric loss), retrain Stage 1 once **without** symmetric loss → `gnn_refiner_no_sym.pt` for 2C only.

---

### 0.2 Shared preparation artifacts (run once — P0)

#### A. Fixed dataset manifests (seed=42)

Generated once from coco-minitrain-10k + COCO JSON intersection. **Never resample at runtime.**

```
data/splits/
├── train_all.txt          # ~10k images (minitrain train2017.txt)
├── val_all.txt            # val subset (minitrain val2017.txt)
├── labeled_5pct.txt       # 500 images (~5% of train), image-level split
├── weak_95pct.txt         # remaining ~9500 images
├── labeled_5pct.json      # per-image instance ann_ids (Stage 1)
├── val_prompts_fixed.json # fixed val prompts per instance
└── split_report.json      # class balance, instance counts
```

Rules:
- Split at **image level** (not instance level) to avoid leakage.
- Use `np.random.RandomState(42)` on sorted image IDs.
- Dataloader intersects `instances_train2017.json` / `instances_val2017.json` with IDs from these lists (see [PREPARATION.md](PREPARATION.md)).

#### B. Precomputed SAM image embeddings

One pass over `train_all` + `val_all` with frozen SAM ViT-B @ 1024².

Helper: `modules/vig_refinenet/sam_stage1_common.py::encode_sam_embeddings`

```
data/cache/sam_embeddings/
├── train/{image_id:012d}.fp16.npy   # [256, 64, 64]
├── val/{image_id:012d}.fp16.npy
└── manifest.json                    # image_id → path, orig H×W, pad/scale metadata
```

~11k images × fp16 ≈ 23 GB. Pack as Kaggle Dataset or chunked archives.

#### C. Stage-1 GNN checkpoint (train once, reuse)

```
checkpoints/gnn_refiner_stage1.pt
```

- Train on `labeled_5pct` only (full GT).
- SAM encoder: load from cache (no encoder in training loop).
- **After §0.5 fix:** SAM decoder + weak prompts → GNN refines 3 SAM masks.
- Loss: Dice + BCE on 3 refined masks + 0.1 × symmetric loss.
- Default: **freeze** this checkpoint in all Stage-2 experiments.

---

### 0.3 Runtime compute policy

| Component | Train | Val/Test |
|-----------|-------|----------|
| SAM encoder | **Load from cache** | **Load from cache** |
| Weak prompts | **Online** (varies Exp 3A–C) | **Fixed** (`val_prompts_fixed.json`) |
| SAM decoder | Online | Online, fixed prompts |
| GNN refiner | Load shared ckpt; **frozen** | Frozen |
| Mask2Former / YOLO | Trained per experiment | Eval per experiment |

Weak prompt policies (training):
- **Exp 3A:** bbox from GT, ±5px jitter each epoch
- **Exp 3B:** 1 random point inside mask each epoch
- **Exp 3C / default:** mixed points + scribbles ([PREPARATION.md](PREPARATION.md) augmentation policy)
- **Exp 1B:** same weak prompts, **no GNN** (raw SAM pseudo-labels)

Stage-2 batch: 50% from `labeled_5pct`, 50% from `weak_95pct` — membership from fixed manifests only.

---

### 0.4 Execution order

```
P0  Preparation (once)
    P0.1  Generate fixed splits + val_prompts_fixed.json
    P0.2  Precompute SAM embeddings (train + val)
    P0.3  Fix Stage-1 notebook pipeline (§0.5)
    P0.4  Train GNN Stage-1 → gnn_refiner_stage1.pt
    P0.5  (Optional) Precompute raw SAM val masks for Exp 1B speed

P1  Exp 1C — full SWSIS (main result)

P2  Exp 1A, 1B, 1D — bounds

P3  Exp 2A, 2B, 2C — ablations

P4  Exp 3A, 3B, 3C — signal sensitivity

P5  Exp 4A — YOLOv8-seg
```

Log during every run (WandB/TensorBoard + `metrics.jsonl`):

- **Stage-1 GNN:** `bce_raw`, `bce_weighted`, `dice_raw`, `dice_weighted`, `seg_weighted`, `sym_raw`, `sym_weighted`, `total` (train + val); AP: `raw_sam_ap`, `val_refined_ap`, `delta_ap`. Train split = `labeled_5pct` only.
- **Stage-2 student:** `sup_loss`, `semi_loss`, `distill_loss`, GNN `sym_loss` / `partial_ce`, agreement rate, GPU mem, time/epoch.

After every run: COCO AP, AP50, AP75, AP_S/M/L + qualitative grids ([EXPERIMENT.md](EXPERIMENT.md) Phase 3).

---

### 0.5 Known issue — Stage-1 notebook (must fix before P0.4)

**File:** `modules/vig_refinenet/train_sam_gnn_stage1_kaggle.ipynb`

**Problem:** Train/val previously ran `refiner(sam_embed)` only — the GNN acted as a standalone segmentation head on image embeddings. The intended pipeline (§2 below) is:

```
SAM embed (node init only) + image + weak prompts + SAM decoder → 3 raw masks → GNN refiner → 3 refined masks
```

What the old notebook did instead:

```python
sam_embed = encode_sam_embeddings(sam_model, images, ...)
logits = refiner(sam_embed)   # no image, no prompts, no SAM decoder, 1 mask not 3
```

**Why this is wrong:**
- The GNN is a **refiner**, not a full segmenter. SAM embedding must **initialize first-layer graph nodes only**; trainable inputs are the **RGB image**, **3 SAM proposal masks**, and **weak-signal map**.
- Without prompt conditioning and SAM decoder masks, it cannot learn the intended refinement behavior and metrics overstate capability.
- `SamStage1Refiner.forward()` must accept `(sam_embed, images, sam_masks_3, weak_signal)` and output `[B, 3, 256, 256]` ([sam_stage1_refiner.py](../modules/vig_refinenet/sam_stage1_refiner.py)).
- Validation must compare **raw SAM vs refined** — not GNN-only output vs GT — so improvement from refinement is visible in IoU and instance-seg **AP**.

**Fix checklist (P0.3):**
1. Extend `SamStage1Refiner`: `sam_embed` → node init; fuse image + `sam_masks_3` + weak signal → GNN → `[B, 3, 256, 256]`.
2. Wire SAM prompt decoder in train/val loops (prompts from GT boxes/points on 5% labeled data).
3. Add symmetric loss; supervise all 3 refined masks against GT.
4. Val metrics: **raw SAM AP**, **refined AP**, **ΔAP** (+ IoU); on fixed val prompts.
5. Visualization: 1×5 grid — image → weak signal → raw SAM → GNN refined → GT.

Until P0.3 is done, do **not** reuse `sam_stage1_refiner.pt` for Stage 2 or report Stage-1 numbers in the paper.

---

### 0.6 Next step (start here)

1. **P0.1** — Script to write `data/splits/labeled_5pct.txt` and `weak_95pct.txt` from minitrain train list (seed=42, image-level).
2. **P0.3** — Fix Stage-1 notebook + `SamStage1Refiner` per §0.5 (can develop on a small subset first).
3. **P0.2** — Precompute SAM embeddings once splits exist.
4. **P0.4** — Train GNN; save `checkpoints/gnn_refiner_stage1.pt`.
5. **P1** — Implement Stage-2 loop (§3) and run **Exp 1C**.

---

### 1. Architectural Alignment & Projector Design

**The Resolution Discrepancy:**

* **SAM (Teacher):** Takes `1024x1024` → outputs `64x64x256` embedding.
* **Mask2Former / Swin-T (Student):** Takes `640x640` → outputs a feature pyramid. Stride-16 output is `40x40x256`, and Stride-32 output is `20x20x768`.

**The Projector (Feature Distillation):**
Cross-Resolution Feature Alignment (SAM-KD / MobileSAM style).

* **Method:** Mask2Former Swin-T stride-16 (`40x40x256`) → $1\times1$ Conv + LayerNorm → bilinear upsample to `64x64`.
* **Loss:** MSE between projected student features and frozen SAM features.

---

### 2. Stage 1: Warm-up (Training the GNN Refiner)

**Data:** `labeled_5pct` (fixed manifest).  
**Prerequisite:** P0.2 embeddings cached; P0.3 pipeline fix applied.

**Flow:**

1. Load cached SAM embedding `[256, 64, 64]` (or encode once if cache miss).
2. Weak signals (points/boxes from GT) prompt the SAM Decoder → 3 masks at `256x256`.
3. GNN Refiner takes `(sam_embed, images, sam_masks_3, weak_signal)` → 3 refined masks at `256x256`.
   - `sam_embed`: frozen SAM encoder output — **node initialization for layer 1 only**
   - `images`: RGB input at training resolution
   - `sam_masks_3`: 3 SAM decoder proposals from weak prompts
   - `weak_signal`: spatial map of point/bbox prompts
4. Interpolate to `640x640` for metric calculation.

**Stage 1 Losses:**

* **Supervised Loss:** Dice + BCE between the 3 refined masks and GT.
* **Symmetric Loss:** Pairwise Dice/MSE between the 3 refined masks.

**Output:** `checkpoints/gnn_refiner_stage1.pt` — frozen and reused in Stage 2 (except Exp 2A/1B and 2C variant noted above).

---

### 3. Stage 2: Semi-Weakly Supervised Training

**Data:** 100% (`labeled_5pct` + `weak_95pct`).  
**Batch:** 50% fully labeled / 50% weak (from fixed manifests).

**Flow:**

1. **Pseudo-Label (weak half):** Cached SAM embed → decoder → GNN → 3 refined masks → agreement (≥2/3 vote) → `640x640` pseudo-GT.
2. **Student (Mask2Former):** Combined batch forward at `640x640`.
3. **Update Mask2Former:** GT on labeled half; pseudo-GT on weak half.
4. **GNN (default frozen):** Optional slow LR fine-tune for Exp 1C only if not using frozen policy.

**Stage 2 Losses:**

* **Distillation:** MSE on weak half (projected M2F stride-16 vs SAM embed).
* **Mask2Former Semi-Loss:** Dice + BCE vs pseudo-GT (weak half).
* **Mask2Former Sup-Loss:** Dice + BCE vs GT (labeled half) + partial CE on weak pixels (weak half).
* **GNN Loss (if unfrozen):** Symmetric + partial CE on weak labels.

---

### 4. PyTorch Implementation Plan

#### A. The Feature Projector

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureProjector(nn.Module):
    def __init__(self, m2f_dim=256, sam_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(m2f_dim, sam_dim, kernel_size=1, bias=False),
            nn.LayerNorm([sam_dim, 64, 64])
        )

    def forward(self, m2f_feat_stride16):
        upsampled = F.interpolate(m2f_feat_stride16, size=(64, 64), mode='bilinear', align_corners=False)
        return self.proj(upsampled)

def feature_distillation_loss(aligned_m2f_feat, sam_feat):
    return F.mse_loss(aligned_m2f_feat, sam_feat)
```

#### B. The Symmetric & Agreement Logic

```python
def symmetric_loss(refined_masks):
    m1, m2, m3 = refined_masks[:, 0], refined_masks[:, 1], refined_masks[:, 2]
    loss_12 = F.mse_loss(m1, m2)
    loss_23 = F.mse_loss(m2, m3)
    loss_13 = F.mse_loss(m1, m3)
    return (loss_12 + loss_23 + loss_13) / 3.0

def generate_pseudo_label(refined_masks_logits):
    probs = torch.sigmoid(refined_masks_logits)
    binary_masks = (probs > 0.5).float()
    votes = torch.sum(binary_masks, dim=1, keepdim=True)
    agreed_mask = (votes >= 2).float()
    pseudo_gt = F.interpolate(agreed_mask, size=(640, 640), mode='nearest')
    return pseudo_gt.detach()
```

#### C. Stage 1: Warm-up Training Loop (target — after §0.5 fix)

```python
sam.eval()
gnn_refiner.train()
optimizer_gnn = torch.optim.AdamW(gnn_refiner.parameters(), lr=1e-4)

for images, weak_prompts, gt_masks in dataloader_5percent:
    with torch.no_grad():
        sam_embed = load_cached_embedding(image_ids)  # or encode_sam_embeddings(...)
        sam_masks_3 = sam.prompt_decoder(sam_embed, weak_prompts)  # [B, 3, 256, 256]

    refined_masks_3 = gnn_refiner(sam_embed, images, sam_masks_3, weak_signal)
    refined_640 = F.interpolate(refined_masks_3, size=(640, 640), mode='bilinear')

    loss_seg = calc_dice_bce(refined_640, gt_masks.unsqueeze(1).repeat(1, 3, 1, 1))
    loss_sym = symmetric_loss(refined_640)
    total_loss = loss_seg + 0.1 * loss_sym

    optimizer_gnn.zero_grad()
    total_loss.backward()
    optimizer_gnn.step()
```

#### D. Stage 2: SWSIS Training Loop

```python
mask2former.train()
gnn_refiner.eval()  # frozen by default; requires_grad=False
projector.train()

opt_m2f = torch.optim.AdamW(list(mask2former.parameters()) + list(projector.parameters()), lr=1e-4)

for batch_labeled, batch_weak in dataloader_swsis:
    img_lbl_640, gt_masks = batch_labeled
    img_weak_640, weak_prompts, weak_signals_for_loss = batch_weak

    with torch.no_grad():
        sam_embed_weak = load_cached_embedding(weak_image_ids)
        sam_masks_3_weak = sam.prompt_decoder(sam_embed_weak, weak_prompts)

    with torch.no_grad():  # frozen GNN
        refined_masks_3_weak = gnn_refiner(sam_embed_weak, weak_prompts, sam_masks_3_weak)
    pseudo_gt_weak = generate_pseudo_label(refined_masks_3_weak)

    combined_images_640 = torch.cat([img_lbl_640, img_weak_640], dim=0)
    m2f_outputs, m2f_features = mask2former(combined_images_640, return_features=True)

    batch_size = img_lbl_640.size(0)
    m2f_out_lbl = {k: v[:batch_size] for k, v in m2f_outputs.items()}
    m2f_out_weak = {k: v[batch_size:] for k, v in m2f_outputs.items()}

    loss_sup_full = mask2former_loss(m2f_out_lbl, gt_masks)
    m2f_stride16_weak = m2f_features['res3'][batch_size:]
    aligned_m2f_weak = projector(m2f_stride16_weak)
    loss_distill = feature_distillation_loss(aligned_m2f_weak, sam_embed_weak.detach())
    loss_semi = mask2former_loss(m2f_out_weak, pseudo_gt_weak)
    loss_sup_weak = partial_ce_loss(m2f_out_weak, weak_signals_for_loss)

    loss_m2f_total = loss_sup_full + loss_semi + loss_sup_weak + loss_distill
    opt_m2f.zero_grad()
    loss_m2f_total.backward()
    opt_m2f.step()
```

#### E. Multi-Instance Handling

```python
def process_multi_instance(sam_embed, multiple_prompts):
    with torch.no_grad():
        sam_masks_3 = sam.prompt_decoder(sam_embed, multiple_prompts)
    refined_masks_3 = gnn_refiner(
        sam_embed.repeat(N, 1, 1, 1), multiple_prompts, sam_masks_3
    )
    pseudo_gts = generate_pseudo_label(refined_masks_3)
    return pseudo_gts  # conflict resolution: argmax confidence per pixel
```

#### F. Dataloader contract (all experiments)

Every experiment reads:
- Split list: `labeled_5pct.txt` | `weak_95pct.txt` | `train_all.txt` (Exp 1D)
- Embeddings: `data/cache/sam_embeddings/manifest.json`
- Weak prompt generator keyed by experiment ID (3A/3B/3C)
- Val: always `val_prompts_fixed.json`

---

### 5. Dataloader & logging checklist

**During training:** sup/semi/distill losses, GNN sym/partial-ce, agreement rate, GPU mem, time/epoch.

**After training:** AP, AP50, AP75, AP_S/M/L; refinement pipeline grid; failure cases (3–5 images); optional t-SNE of stride-16 features.

See [EXPERIMENT.md](EXPERIMENT.md) for presentation/report mapping.
