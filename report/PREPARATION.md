Step 1: We will use all **80 COCO classes**. For computational feasibility, we use **coco-minitrain-10k** (~10k train images) for student training splits; full COCO 2017 JSON/val images are used for annotations and paths.

**Stage-1 GNN (P0.4):** trains on **`labeled_5pct` only** (~500 images from the 10k minitrain train list). The 95% weak split is not used until Stage-2.
Dataset filtering will be handled dynamically during dataloader initialization. We will pass a `train.txt` and `val.txt` file containing the paths of the specific images to load:
```text
./images/train2017/000000337246.jpg
./images/train2017/000000040361.jpg
...
```
The custom Detectron2 DatasetMapper/Dataloader will dynamically intersect the standard `instances_train2017.json` / `instances_val2017.json` files with the IDs present in these text files, allowing it to remain compatible with standard Kaggle COCO architectures.

Step 2: We will employ a **Dual-GPU Pipeline (Online Augmentation)** architecture during training to completely bypass disk storage limits and unlock infinite augmentation potential. Rather than precomputing SAM's outputs offline, we split the computational workload across the two Tesla T4 GPUs available on Kaggle. The prompt generation policy is as follows:

*   **Training Set (Online Infinite Augmentation Policy):** 
    *   **Points:** On every single epoch, we sample *new, randomly perturbed points* (simulating erratic user clicks) on the fly.
    *   **Scribbles:** On every single epoch, we dynamically generate synthetic scribbles by reducing lengths based on random ratios (e.g., 0.8, 0.5, 0.3) and adding slight spatial noise.
    *   **Advantage:** Because we compute this on the fly, the model never sees the exact same weak prompt twice across 50 epochs. This virtually eliminates prompt-overfitting and makes the GNN Refiner incredibly robust.
*   **Validation/Test Set (Standard Policy):** Used to ensure reproducibility and comparability.
    *   **Points:** Pick the point at the geometric centroid or the deepest point inside the object (maximal interior distance transform).
    *   **Scribbles:** Use a scribble with a fixed length (e.g., 70% of the longest edge of the bounding box) aligned along the object's principal axis.

We will use this dataset as our dataset for the whole project.

### Hardware Allocation (Remote machine — see RANDOM_NOTE.md)

**Updated for GitHub/remote training** (not Kaggle):

1. **Precompute SAM embeddings offline** (`data/cache/sam_embeddings/`) — see `scripts/prep/run_p0.sh`.
2. **At training time:** **1 GPU** runs SAM teacher inference (decoder + GNN); **all other GPUs** train Mask2Former / YOLO (student).
3. Set `CUDA_VISIBLE_DEVICES` and `WSSIS_NUM_GPUS` as documented in [scripts/RUNBOOK.md](../scripts/RUNBOOK.md).

Legacy dual-GPU online-SAM design (Kaggle) is superseded by precomputed embeddings + fixed splits in [PLAN.md](PLAN.md) §0.2.