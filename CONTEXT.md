# ROI-Based Object Detection Framework (ROI-SSD)

## 📌 Overview

This project implements and evaluates an ROI-based object detection framework for video processing.

The core idea is to **restrict detection to Regions of Interest (ROI)** instead of processing the full frame, in order to reduce computational cost while maintaining detection accuracy.

The system combines:

* object detectors (ROI-SSD, YOLO variants),
* ROI generation methods (static padding, memory, Kalman),
* ROI merging strategies,
* and runtime analysis.

---

## 🎯 Key Concept

Instead of:

> detection on full frame

we perform:

> **tracking-guided detection**

i.e.,

> motion predictions → spatial priors → restricted detection

---

## 🧠 Core Insight

> ROI efficiency is not universal — it depends on:

* detector architecture,
* context sensitivity,
* computational characteristics,
* input size constraints.

---

## 🏗 System Architecture

Pipeline:

1. Previous frame detections
2. ROI generation (tracking / padding)
3. ROI merging
4. ROI inference (ROI-SSD / YOLO)
5. Optional full-frame fallback
6. Metric aggregation

---

## 📦 Models

### ROI-SSD

* Based on SSD (VGG16 backbone)
* Supports arbitrary ROI sizes
* Uses dynamic backbone depth depending on ROI size
* Regenerates anchors per ROI

Characteristics:

* high per-pixel cost
* strong dependence on context

---

### YOLO (v26n / v26s / v26m)

* modern detector
* anchor-free head
* requires input sizes divisible by 32

Characteristics:

* low per-pixel cost
* weak dependence on context
* high fixed overhead

---

## 📐 Computational Model

Inference time is approximated as:

```
T(A) = K_t + c_t * A
```

Where:

* `A` — ROI area (pixels)
* `c_t` — per-pixel cost
* `K_t` — fixed overhead

Define threshold:

```
τ = K_t / c_t
```

Interpretation:

* if ROI_area < τ → ROI gives little speedup
* if ROI_area > τ → ROI is beneficial

---

## 📊 Measured Values (CPU)

| Model   | c_t     | K_t      | τ        |
| ------- | ------- | -------- | -------- |
| ROI-SSD | ~1.5e-6 | ~0.014 s | ~9.5k px |
| YOLO    | ~8e-8   | ~0.006 s | ~74k px  |

---

## 🔍 ROI Generation Methods

### 1. Static Padding

* ROI = bbox + fixed padding
* optional memory (hold for N frames)

Pros:

* simple
* stable

Cons:

* not motion-aware

---

### 2. Kalman Filter

* predicts object position
* ROI based on predicted bbox

Features:

* supports missing detections
* uses uncertainty (covariance)

Observed behavior:

* similar speed to static padding
* slightly better accuracy (better alignment)

---

### 3. (Planned) Anisotropic ROI

* ROI expands along motion direction
* reduces unnecessary area

---

## 🔗 ROI Merging

Goal:

* reduce number of ROIs
* reduce duplicate processing

Method:

* greedy cost-aware merging:

  * evaluate all ROI pairs
  * merge if computation decreases

Key insight:

* merging helps when detector produces duplicate detections

---

## ⚠️ Important Constraint: Input Quantization

YOLO requires input size divisible by 32.

Thus ROI is transformed as:

```
W' = ceil(W / 32) * 32
H' = ceil(H / 32) * 32
```

Effect:

* ROI sizes are **discretized**
* small differences in ROI → no runtime difference
* limits benefits of precise ROI generation

---

## 📊 Key Experimental Findings

### 1. Context Sensitivity

ROI-SSD:

* strong dependence on padding
* requires ~64 px padding for stable accuracy

YOLO:

* nearly invariant to padding
* stable even with tight crops

---

### 2. ROI Speed Behavior

ROI-SSD:

* latency strongly depends on ROI area
* ROI significantly improves FPS

YOLO:

* latency weakly depends on ROI area
* speedup is sublinear

---

### 3. Kalman vs Static Padding

For YOLO:

* similar speed
* slightly better accuracy

Reason:

* better ROI alignment
* quantization limits size advantage

---

### 4. ROI Role Depends on Model

| Model   | ROI Role             |
| ------- | -------------------- |
| ROI-SSD | compute optimization |
| YOLO    | context control      |

---

### 5. Quantization Effect

ROI precision is limited by /32 constraint:

* ROI size → step function
* many ROI strategies collapse to same effective input size

---

## ⚠️ Experimental Limitation

Most experiments:

* single object per frame

Implications:

* no evaluation of:

  * multi-object tracking
  * ROI conflicts
  * merging complexity

---

## 📈 Recommended Experiments (Next)

1. Multi-object scenarios:

   * crossing trajectories
   * dense scenes

2. ROI merging evaluation:

   * ROI count
   * merged area
   * merge errors

3. Kalman + anisotropic ROI

---

## 🧪 Evaluation Metrics

* mAP50 / mAP95
* FPS
* latency (mean, p50, p95)
* processed_area_ratio
* full_frame_fraction
* detector_recall

---

## 💡 Key Scientific Contributions

1. Computational model of ROI efficiency (τ-based)
2. Detector-dependent ROI behavior
3. Input quantization effect on ROI
4. Analysis of ROI generation strategies

---

## 🧠 Main Takeaway

> ROI-based detection is not universally beneficial —
> its effectiveness is determined by detector architecture, computational model, and input constraints.

---

## 🛠 Usage (high-level)

1. Run detector on video
2. Generate ROI from previous detections
3. Apply merging
4. Run inference on ROI
5. Collect metrics

---

## 📌 Notes for Claude Code

When working with this project:

* ROI size is NOT continuous (quantized by 32)
* YOLO speed ≠ ROI area linearly
* Kalman gains may be masked by quantization
* Always distinguish:

  * theoretical ROI
  * effective input size

---

## 🔚 End
