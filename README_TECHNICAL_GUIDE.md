# 🚀 HRI30 Action Recognition: Technical Reproduction Guide

**Status:** SOTA Achieved  
**Test Accuracy:** 90.23%  
**Validation Accuracy:** 89.85%  
**Architecture:** 3-Stream Topology-Aware Ensemble (Joint + Bone + Motion)

---

## 📋 1. Pipeline Overview

This project utilizes a highly optimized **3-Stream Pipeline** to achieve state-of-the-art results on the small-scale HRI30 dataset. Unlike traditional approaches that pre-generate heavy `.npy` files for every stream, we use **On-the-Fly Feature Computation** during training to save disk space and improve iteration speed.

### Workflow
1.  **Extract:** Raw Video $\to$ 2D Skeletons (YOLOv8-Pose-P6).
2.  **Train:** 3 Parallel Streams (Joint, Bone, Motion) with **stream-specific scaling**.
3.  **Inference:** Weighted Ensemble Fusion.

---

## 🛠️ 2. How to Run (Reproduction Steps)

### Step 1: Data Preprocessing
Extract 2D keypoints from raw `.avi` videos using YOLOv8-Pose (Large model).
*   **Input:** Raw Videos folder.
*   **Output:** `train_sequences.npy` (with labels), `test_sequences.npy`.

```bash
python preprocessing.py
```

### Step 2: 3-Stream Training
We train three separate models. The data loader calculates Bone and Motion features dynamically from the Joint data.

**A. Joint Stream** (Spatial Patterns)
```bash
python finetune_skeletonx.py --stream joint
```
*   *Config:* Normalization factor **3.0**.

**B. Bone Stream** (Structural Orientation)
```bash
python finetune_skeletonx.py --stream bone
```
*   *Config:* Normalization factor **4.0**.

**C. Motion Stream** (Temporal Dynamics)
```bash
python finetune_skeletonx.py --stream motion
```
*   *Config:* Amplification factor **10.0**.

### Step 3: Inference & Ensemble
Generate the final submission by fusing the predictions from all three saved checkpoints.

```bash
python make_submission.py
```
*   *Logic:* Loads `joint.pth`, `bone.pth`, `motion.pth`.
*   *Process:* Applies the specific scaling rules to test data, computes weighted average, and writes `submission.csv`.

---

## 🧠 3. Architecture & The "Scaling Secret"

The core innovation driving the **90.23% accuracy** is the precise input scaling strategy designed to align the HRI30 data with the pretrained NTU-RGB+D distribution.

### Stream 1: Joint (Spatial)
*   **Definition:** Raw $(x, y)$ coordinates.
*   **Scaling:** `Input / 3.0`
*   **Why?** The pretrained model expects coordinates in a specific range. Raw pixels are too large; dividing by 3.0 maps them to the optimal latent distribution.

### Stream 2: Bone (Vector)
*   **Definition:** Vector difference between connected joints: $J_{child} - J_{parent}$.
*   **Scaling:** `Input / 4.0`
*   **Why?** Bone vectors represent relative orientation. We divide by 4.0 to **dampen noise** and preventing gradients from exploding due to jittery skeleton detections.

### Stream 3: Motion (Temporal)
*   **Definition:** Displacement between frames: $Frame_{t+1} - Frame_{t}$.
*   **Scaling:** `Input * 10.0`
*   **Why?** At 60fps, frame-to-frame pixel displacement is microscopic. We **multiply by 10.0** to amplify these tiny signals, making the temporal patterns visible to the network.

---

## 📊 4. Ensemble Performance

We utilize a **Weighted Score Fusion** strategy to maximize robustness. The Motion and Bone streams are given higher importance as they proved more resilient to background clutter than raw joint positions.

| Metric | Value |
| :--- | :--- |
| **Validation Score** | **89.85%** |
| **Test Score** | **90.23%** |

**Fusion Weights:**
```python
Final_Score = (0.2 * Joint) + (0.4 * Bone) + (0.4 * Motion)
```

---
*Maintained by the AI Research Team.*
