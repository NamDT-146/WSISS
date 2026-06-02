Based on your professor's strict grading criteria—which heavily favors **problem formulation, methodology, and deep analysis** over just raw results—you need a structured, scientific approach to your experiments.

> **True SWSIS (active report matrix):** Report items 1–5 map to **1A**, **P0.4 teacher eval**, **1C**, **4A**, and **1D** (upper bound). Ablations 1B/2A–2C/3A–3C are archived — see [ARCHIVED_EXPERIMENTS.md](ARCHIVED_EXPERIMENTS.md). Run `bash scripts/experiments/run_smoke_test.sh` before full GPU training.
>
> **GNN v2 (`wssis_v2`):** Single weak channel in/out; Stage-1 triplet batches + KL/symmetric aux losses; Stage-2 one weak type per weak image. See [IMPLEMENTATION.md](IMPLEMENTATION.md). Reuse P0.2 SAM embeddings; retrain GNN and all experiments.

Here is the master list of every experiment you need to run, exactly what to log during training, and the figures you must generate for your presentation and formal report.

---

### Phase 1: The Experiments to Run

To prove your methodology is robust, you cannot just run your final model once. You must run four categories of experiments:

#### 1. The Boundary Baselines (Proving the Problem Formulation)

These experiments establish the lower and upper bounds of your problem.

* **Exp 1A (Lower Bound):** Train Mask2Former on only the 5% fully supervised data.
* **Exp 1B (Weak Baseline):** Train Mask2Former using the 95% weakly supervised data, but use raw SAM outputs as pseudo-labels (no GNN refinement).
* **Exp 1C (Your Full SWSIS Method):** Train the full pipeline (SAM + GNN Refiner + Feature Distillation + Mask2Former) on the 5% / 95% split.
* **Exp 1D (Upper Bound):** Train Mask2Former on 100% fully labeled ground truth (to show how close your SWSIS method gets to the ideal scenario).

#### 2. The Architectural Ablations (Proving Your Methodology)

This is where you earn the highest marks. You must turn off components to prove they are mathematically necessary.

* **Exp 2A (No GNN Refiner):** Remove the GNN. Pass SAM's 3 mask outputs directly to the student. (Analyzes the value of your symmetric/agreement logic).
* **Exp 2B (No Feature Distillation):** Remove the $1\times1$ projector and the L2/MSE feature alignment loss. (Analyzes if guiding Mask2Former's Swin-T with SAM's semantic space is actually helping).
* **Exp 2C (No Symmetric Loss):** Keep the GNN, but remove the pairwise Dice loss between the 3 refined masks. (Analyzes boundary sharpness).

#### 3. Signal Sensitivity (Proving Data Robustness)

Because your dataset class is configurable, run Stage 2 with the 95% weak data restricted to specific signal types:

* **Exp 3A:** 100% Bounding Boxes only.
* **Exp 3B:** 100% Points only (1 point per object).
* **Exp 3C:** Mixed Signals (Your default).

#### 4. Cross-Architecture Generality (The "Wow" Factor)

* **Exp 4A:** Swap out the heavy Mask2Former student for a lightweight CNN-based model (e.g., YOLOv8-seg). Apply the same feature distillation and GNN pseudo-labels to prove your framework works across different network paradigms (Transformer vs. CNN).

---

### Phase 2: What to Log DURING Training (For the Report)

Do not just look at the progress bar. You must use TensorBoard or Weights & Biases (WandB) to log these specific metrics at every epoch/iteration:

1. **Loss Decompositions (Crucial for Analysis):**
* **Student Losses:** Log `Mask2Former Sup-Loss` (on the 5% data) and `Mask2Former Semi-Loss` (on the 95% pseudo-labels) separately.
* **Distillation Loss:** Log the MSE between the projected Student Stride-16 features and SAM's frozen features. If this doesn't go down, your projector is failing.
* **GNN Losses:** Log the `Symmetric Loss` and `Partial CE Loss`.


2. **The GNN Agreement Rate:** Track pixels where ≥2/3 heads pass the effective pseudo threshold (`fixed` @ 0.9, `adamatch` batch-relative, or `freematch` EMA). Log **`over_threshold_ratio`** and **`train_effective_pseudo_threshold`** (mean cutoff per batch) in Stage-1 `metrics.jsonl`.
3. **Compute Metrics:** Log GPU Memory Usage and Time-per-Epoch. (You will use this in your report to justify downsizing to COCO-10k / 20 classes due to Kaggle constraints).

---

### Phase 3: What to Log AFTER Training (Evaluation & Visualizations)

#### 1. Quantitative Metrics (The Numbers)

For every experiment listed in Phase 1, evaluate on the COCO Validation set and log:

* Standard COCO Metrics: $AP$, $AP_{50}$, $AP_{75}$.
* Scale Metrics: $AP_S$ (Small), $AP_M$ (Medium), $AP_L$ (Large objects). *Analysis tip: Weak signals like a single point usually fail on small objects.*

#### 2. Qualitative Visualizations (The Figures)

You must generate these specific images for your Presentation Slides and Report:

* **The Refinement Pipeline Grid:** A 1x5 image grid showing: `Original Image` $\rightarrow$ `Weak Signal Overlay (e.g., a dot)` $\rightarrow$ `Raw SAM Output (noisy boundaries)` $\rightarrow$ `GNN Refined Output (sharp)` $\rightarrow$ `Ground Truth`. (Visually it with AP).
* **Feature Space t-SNE (Optional but highly academic):** Plot the Stride-16 feature embeddings of Mask2Former with and without feature distillation to visually prove that the projector aligns the student with SAM.
* **Failure Case Analysis (Mandatory for High Grade):** Save 3-5 images where your SWSIS model completely fails.
* *Example:* Heavy occlusion (two people overlapping with only one point prompt).
* *Example:* SAM segmenting shadows instead of the physical object.



---

### Phase 4: Mapping to Your Deliverables

#### For the 15-Minute Presentation Slides (Max 30 Slides)

* **Problem (Slides 2-4):** Define the massive annotation cost of pixel-perfect masks vs. the cheap cost of points/boxes. Show a visual comparison.
* **Methodology (Slides 5-10):** High-level diagrams of Stage 1 (GNN Warm-up) and Stage 2 (Student Distillation). Explain the Kaggle dataset constraints clearly.
* **Results (Slides 11-15):** A Bar Chart showing Annotation Cost vs. $AP$. Show that your method gets 90% of the fully-supervised performance at a fraction of the labeling cost.
* **Analysis (Slides 16-20):** Show the ablation table (Exp 2A-2C). Show the Qualitative Refinement Grid. Show the YOLO vs. Mask2Former comparison. Show the failure cases.
* **Task Assignment (Slide 21):** A clear table: `Member Name | Core Responsibilities (e.g., GNN design, Data pipeline) | % Contribution`.
* **Demo:** Have a Jupyter Notebook ready where you click a point on an image and your trained model instantly generates a high-quality mask.

#### For the Scientific Report (Hard Copy)

* Include all mathematical formulas for your losses ($L_{distill}$, Symmetric Loss, Partial CE).
* Include massive tables for your ablation studies.
* Spend at least 2 pages on the **Discussion/Analysis** section. Do not just list the AP scores. Write paragraphs explaining *why* the GNN was necessary, *why* feature distillation stabilized training, and *why* specific failure cases occur.