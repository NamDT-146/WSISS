# Stage-1: Hungarian alignment of SAM multimasks

## Problem

For each COCO instance, Stage-1 runs three weak prompts (box, scribble, point). SAM’s mask decoder returns **three multimask proposals per prompt** — nine masks total. Those three heads are **not in a fixed semantic order** (part vs whole can swap across heads or prompt types).

Comparing or distilling masks **by raw head index** (head 0 vs head 0) is wrong: you may match “whole object” on the box prompt to “small part” on the point prompt.

## Objective

Treat alignment as a **linear assignment** problem on a **3×3 cost matrix**:

- Rows: the three anchor masks (by default **box** SAM heads).
- Columns: the three masks from the weaker prompt (e.g scribble).
- Cost: `1 − IoU(anchor_i, other_j)` (high cost = poor overlap).

The **Hungarian algorithm** (implemented via `scipy.optimize.linear_sum_assignment`, with a greedy 3×3 fallback) finds a permutation π that **minimizes total cost**:

\[
\min_{\pi} \sum_{i=0}^{2} \bigl(1 - \mathrm{IoU}(m^{\pi(i)}_{\text{other}},\, m^{i}_{\text{anchor}})\bigr)
\]

So each anchor head is paired with exactly one other head, and the pairs are globally optimal (not greedy per row).

## Chaining (box → scribble → point)

1. Align **scribble** SAM masks to **box** SAM masks; apply the same permutation to GNN refined scribble heads.
2. Align **point** SAM masks to the **aligned scribble** SAM masks; permute GNN refined point heads.

Consistency losses (hierarchical KL and symmetric Dice) are then computed **per matched head index** \(i \in \{0,1,2\}\) on the GNN refined probabilities, with the stronger signal detached as the teacher in KL terms.

## What alignment does *not* do

- It does **not** solve a single 9×9 assignment across all nine masks at once; it uses two successive 3×3 problems along the prompt hierarchy.
- Matching is computed from **detached SAM** masks; gradients do not flow through IoU or the assignment step.
- It does **not** train the SAM encoder/decoder (frozen in Stage-1).

## Code

`modules/wssis/training/gnn_losses.py`: `_pairwise_iou_cost`, `hungarian_match_perm`, `align_three_heads_pair`, `nine_aligned_proposal_loss`.
