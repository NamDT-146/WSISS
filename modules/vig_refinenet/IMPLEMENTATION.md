## **Primary Validation Experiment: Learned Mask Refinement vs DenseCRF**

---

## 1. Objective

The goal of the primary experiment is to answer:

> **Can a learned graph-based refinement network improve instance segmentation masks more effectively than DenseCRF, measured by Dice / IoU improvement over raw predictions?**

This experiment isolates the **refinement module** and does not modify the detector.

---

## 2. Experimental Design Principles

* Detector is **frozen**
* Same raw mask input for all refinement methods
* No additional training data
* Comparison focuses on **ΔDice / ΔIoU**

---

## 3. Dataset

* Fully supervised instance segmentation dataset
* Ground-truth masks available for **training and evaluation**
* Dataset size can be small (proof-of-concept)

---

## 4. Detector Setup (Fixed)

### Model

* **FCOS** instance segmentation
* Backbone: **ResNet-50 + FPN**
* Pretrained on standard benchmarks

### Implementation

* Framework: **Detectron2**
* Repository:

  * [https://github.com/facebookresearch/detectron2](https://github.com/facebookresearch/detectron2)

### Outputs Used

For each instance:

* Bounding box
* Raw mask logits (low resolution)

The detector is **not fine-tuned** in this experiment.

---

## 5. Baselines

### 5.1 Baseline 1 — No Refinement

* Raw mask logits
* Bilinear upsampling
* Thresholding
* Metric computation

### 5.2 Baseline 2 — DenseCRF

#### Method

* Dense Conditional Random Field
* Applied per instance inside bounding box

#### Implementation

* `pydensecrf`
* Repository:

  * [https://github.com/lucasb-eyer/pydensecrf](https://github.com/lucasb-eyer/pydensecrf)

#### Configuration

* Gaussian pairwise terms (position + RGB)
* Fixed hyperparameters
* No learning or tuning per dataset

---

## 6. Proposed Method: GNN-Based Refinement

### 6.1 Input to Refinement Network

For each instance:

* Cropped RGB image
* Raw mask logits (upsampled)
* Optional bounding box mask

---

### 6.2 Backbone

* CNN: **ResNet-50 (ImageNet pretrained)**
* Same backbone as detector
* Backbone weights **frozen**

This ensures a fair comparison with CRF.

---

### 6.3 Graph Construction

* Node: each pixel in the mask grid
* Edge:

  * 4- or 8-connected neighborhood
  * Optional dilation = 2
* Graph is fixed across layers

---

### 6.4 GNN Architecture (v0)

* Framework: PyTorch Geometric or DGL
* Layers:

  * 2 GNN layers
* Hidden dimension: 64
* Activation: ReLU
* Output: 1 logit per node

---

## 7. Training Protocol

### Loss Function

* **Dice loss** (primary)
* Optional BCE loss

No box loss, no pairwise loss in the primary experiment.

---

### Optimization

* Optimizer: Adam
* Learning rate: standard (e.g., 1e-3)
* Train only GNN parameters

---

## 8. Evaluation Metrics

### Metrics

* Dice coefficient
* Intersection-over-Union (IoU)

### Reported Results

* Absolute Dice / IoU
* **ΔDice = Dice(refined) − Dice(raw)**
* **ΔIoU = IoU(refined) − IoU(raw)**

Improvement metrics are the main comparison.

---

## 9. Expected Outcomes

| Method         | Dice     | ΔDice |
| -------------- | -------- | ----- |
| Raw mask       | baseline | –     |
| DenseCRF       | higher   | +     |
| GNN refinement | highest  | ++    |

Success criterion:

> GNN refinement consistently outperforms DenseCRF in ΔDice / ΔIoU.

---

## 10. Failure Analysis

If GNN underperforms:

* Check graph connectivity
* Increase message passing depth
* Inspect boundary behavior
* Visualize node embeddings

---

## 11. What This Experiment Validates

* Learned affinity > hand-crafted affinity
* Refinement network effectiveness
* Foundation for weakly supervised extension

---

## 12. What Is Explicitly Out of Scope

* Weak supervision
* Box-only training
* Pseudo labels
* Multi-scale refinement
* Superpixel graphs

These are explored only after primary validation succeeds.