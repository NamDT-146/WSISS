# PROPOSAL

# GNN Layer for Instance Mask Refinement

## Big Picture (1 image in your head)

```
Backbone feats
   ↓
Stage 1 (7×7 nodes, global GNN)
   ↓  inheritance
Stage 2 (mid-resolution nodes, local GNN)
   ↓  inheritance
Stage 3 (near-pixel nodes, smoothing GNN)
   ↓
Mask head → refined mask
```

Each **stage is the same template**, only:

* number of nodes changes
* connectivity changes
* feature dimension shrinks

---

## Core GNN Template (used in all stages)

Each stage does **one GNN layer**:

```
Nodes (features + positions)
   ↓
Edge MLP → edge weights/messages
   ↓
Message aggregation
   ↓
Node MLP → updated node features
```

This is **standard message-passing GNN**, not exotic.

---

## Stage 1 — Global Reasoning (49 nodes, fully connected)

### Inputs

* Nodes = 7×7 grid
* Feature per node:
  `f_i ∈ R^2048`
* Position per node:
  `(x_i, y_i)` normalized to [0,1]

---

## Step 1: Build edges (fully connected)

For **every pair (i, j)**:

```
Edge e_ij input:
[ f_i , f_j , Δx_ij , Δy_ij ]
```

This is **EdgeConv-style** (used in DGCNN).

---

## Step 2: Edge MLP (learns affinity)

```
w_ij = EdgeMLP([f_i, f_j, Δx, Δy])
```

Output:

* scalar weight OR
* small message vector (you chose scalar → good & stable)

Interpretation:

* “How much should node j influence node i?”

This is conceptually:

* learned CRF potential
* learned attention score
* learned affinity

✔ This **is exactly what many GNNs do**

---

## Step 3: Message aggregation (per node)

For node i:

```
m_i = Σ_j ( w_ij · f_j )
```

(you may normalize with softmax over j)

This is:

* attention-weighted sum
* dense global context

---

## Step 4: Node update MLP

```
f_i' = NodeMLP( f_i + m_i )
```

Important:

* residual connection ✔
* 2-layer MLP ✔

This is **standard in GNN / Transformer FFN**.

---

### Output of Stage 1

* Updated node features
  `f_i' ∈ R^2048`
* Still 49 nodes

---

# Stage 1 → Stage 2: Feature Inheritance (NOT message passing)

This is **your elegant idea** ⭐

### Superpixel / child mapping

Each parent node owns multiple child nodes:

```
parent_id → [child_id1, child_id2, ...]
```

---

## Inheritance forward

For each child c with parent p:

```
f_c = Conv1x1(f_c_raw) + f_p
```

Where:

* `f_c_raw` = backbone feature at child resolution
* `f_p` = refined parent feature

This is:

* **top-down conditioning**
* NOT graph reasoning
* NOT CRF-like
* very explainable

Think of it as:

> “Global semantics guide local refinement”

---

## Stage 2 — Mid-level Local GNN

Now nodes are:

* superpixels OR
* grid (e.g., 28×28 pooled)

### Differences from Stage 1

* fewer neighbors (k-NN or spatial radius)
* smaller feature dim (512)
* same forward logic

---

## Stage 2 forward (same template)

### Edge construction (local)

```
Edges only between nearby nodes
(distance ≤ 2 superpixels)
```

### Edge MLP

```
w_ij = EdgeMLP([f_i, f_j, Δx, Δy])
```

### Aggregate

```
m_i = Σ_{j ∈ N(i)} w_ij · f_j
```

### Node update

```
f_i' = NodeMLP( f_i + m_i )
```

---

# Stage 2 → Stage 3: Inheritance again

Same as before:

```
f_pixel = Conv1x1(f_pixel_raw) + f_superpixel
```

---

# Stage 3 — Fine Alignment / Smoothing

This stage is **intentionally weak** (reviewer-friendly).

### Nodes

* near-pixel or pixel-level
* feature dim ~64

### Graph

* 4 or 8-connected neighbors
* no global edges

---

## Forward

Exactly the same:

```
EdgeMLP → aggregate → NodeMLP
```

But:

* tiny MLPs
* acts like **learned smoothing**
* learned alternative to CRF

---

## Final Mask Prediction

For each pixel node:

```
logit = Conv1x1(f_pixel)
mask = sigmoid(logit)
```

---

# One-Stage Forward Summary (compact)

For **any stage s**:

```
Input nodes: {f_i, x_i}

for each edge (i, j):
    w_ij = EdgeMLP([f_i, f_j, Δx, Δy])

for each node i:
    m_i = Σ_j w_ij · f_j
    f_i' = NodeMLP(f_i + m_i)

Output nodes: {f_i'}
```

# Feature Map From ResNet

At code file `resnet_inter_taker.py`, the return layers is:
feat0: torch.Size([1, 64, 112, 112])
feat1: torch.Size([1, 256, 56, 56])
feat2: torch.Size([1, 512, 28, 28])
feat3: torch.Size([1, 1024, 14, 14])
feat4: torch.Size([1, 2048, 7, 7])

You can use:
- `feat4` (7x7, 2048-dim) for Stage 1 input nodes.
- `feat2` (28x28, 512-dim) for Stage 2 input nodes.
- `feat0` (112x112, 64-dim) for Stage 3

