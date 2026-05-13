# Representative ImageNet VID Subset Construction for ROI Detection Research

## Objective

Construct a compact yet statistically representative subset of the ImageNet VID dataset for benchmarking ROI-based video object detection pipelines.

The original ImageNet VID validation dataset contains approximately:

* ~170,000 frames
* multiple object classes
* long temporal sequences
* highly redundant neighboring frames

Running full ROI-generation and tracking experiments on the entire dataset is computationally expensive.

The goal is to construct a subset that:

* significantly reduces evaluation cost,
* preserves detector behavior,
* preserves class-wise accuracy trends,
* preserves motion characteristics,
* remains suitable for ROI-generation and tracking experiments.

---

# Target Subset Size

Preferred target:

* ~200 video clips
* each clip = 100 consecutive frames

Result:

* ~20,000 frames total

This size is expected to be large enough for:

* ROI generation evaluation
* temporal tracking experiments
* detector benchmarking
* class-wise AP analysis

while remaining computationally manageable.

---

# IMPORTANT REQUIREMENT

The subset MUST preserve:

1. Detection difficulty
2. Per-class behavior
3. Motion diversity
4. Temporal consistency
5. Object scale distribution

The subset MUST NOT be constructed using purely random frame sampling.

Temporal continuity is essential because:

* ROI generation depends on previous frames
* Kalman prediction depends on motion continuity
* tracking-based methods require sequential data
* frame-to-frame correlations must be preserved

Therefore:

* sample contiguous clips
* not isolated frames

---

# Required Dataset Properties

The subset should approximately preserve the following properties of the full dataset.

---

## 1. Class Distribution

Preserve approximate class frequencies.

Example:

If the original dataset contains:

* car = 30%
* dog = 5%
* bicycle = 3%

then the subset should maintain approximately similar proportions.

Suggested tolerance:

* ±10% relative deviation per class

Goal:

```text
P(class_i)_subset ≈ P(class_i)_full
```

---

## 2. Object Size Distribution

Preserve distribution of:

* small objects
* medium objects
* large objects

Suggested metric:

Normalized bbox area:

```text
bbox_area / frame_area
```

Suggested bins:

* small: < 0.02
* medium: 0.02–0.15
* large: > 0.15

The subset should approximately preserve histogram proportions.

This is critical for ROI-based detection because ROI efficiency strongly depends on object scale.

---

## 3. Motion Distribution

Preserve motion diversity.

For each track estimate:

```text
motion = mean center displacement per frame
```

Suggested motion categories:

* static / very slow
* moderate motion
* fast motion

Additionally preserve:

* scale changes
* direction changes
* camera motion
* partial occlusions

This is especially important for:

* Kalman ROI generation
* memory-based tracking
* ROI stability evaluation

---

## 4. Object Density

Preserve approximate distribution of:

```text
objects_per_frame
```

Suggested categories:

* single-object scenes
* sparse multi-object scenes
* dense scenes

Even if the current experiments mainly use single-object scenes, the subset should include multi-object clips for future evaluation.

---

## 5. Temporal Diversity

Avoid selecting clips from only a few videos.

The subset should:

* cover many different source videos
* preserve scene diversity
* avoid overrepresentation of long easy sequences

Suggested constraint:

```text
max clips per source video <= 3
```

---

# Recommended Construction Strategy

---

## Step 1 — Compute Per-Video Statistics

For each source video compute:

* class histogram
* average object count per frame
* mean object size
* motion magnitude
* motion variance
* frame count
* occlusion ratio (if possible)

Store statistics in a metadata table.

---

## Step 2 — Split Videos into Candidate Clips

Divide videos into:

* contiguous 100-frame clips

Optional:

* allow overlap of 20–50 frames

Each clip becomes one candidate sample.

---

## Step 3 — Compute Clip-Level Features

For each clip compute:

* dominant class
* class histogram
* average object size
* average motion
* object density
* motion variance

---

## Step 4 — Diversity Sampling

Perform representative selection.

Possible methods:

* stratified sampling
* k-means clustering
* greedy diversity selection

Preferred approach:

* cluster clips by feature vector
* sample proportionally from clusters

This ensures:

* balanced difficulty
* balanced motion
* balanced classes

---

# Validation of Subset Quality

The subset MUST be validated experimentally.

---

## Validation Procedure

Run at least one baseline detector on:

1. Full ImageNet VID validation set
2. Constructed subset

Suggested baseline:

* YOLO full-frame inference

Optional:

* ROI-SSD full-frame inference

---

## Compare Metrics

Compare:

* mAP50
* mAP95
* per-class AP
* recall

Suggested acceptance criteria:

| Metric       | Target Difference |
| ------------ | ----------------- |
| mAP50        | <= 1–2%           |
| mAP95        | <= 2%             |
| per-class AP | <= 2–3%           |
| recall       | <= 2%             |

If metrics deviate too much:

* adjust clip selection
* improve class balance
* improve motion diversity

---

# IMPORTANT FOR ROI RESEARCH

The subset should contain scenes with:

* fast motion
* abrupt motion changes
* object scale changes
* partial object visibility
* camera motion
* multiple nearby objects

because these scenarios are critical for:

* Kalman prediction
* ROI stability
* ROI merging
* fallback triggering
* tracker robustness

---

# Recommended Final Metadata

For each selected clip store:

* source video id
* frame range
* dominant class
* object count statistics
* mean object size
* mean motion magnitude
* motion variance
* occlusion indicator (optional)

---

# Expected Benefits

The resulting subset should:

* reduce evaluation cost by ~8–10x
* preserve detector ranking
* preserve ROI-generation behavior
* preserve tracking difficulty
* remain statistically representative

---

# Expected Usage

The subset will be used for:

* ROI generation experiments
* Kalman filter evaluation
* static padding evaluation
* ROI merging evaluation
* detector comparison (ROI-SSD vs YOLO)
* runtime benchmarking
* accuracy-efficiency tradeoff analysis

---

# Notes for LLM / Claude Code

Key priorities:

1. Preserve temporal continuity
2. Preserve motion diversity
3. Preserve class balance
4. Preserve object size distribution
5. Preserve detector behavior relative to full dataset

Avoid:

* random frame sampling
* overrepresentation of easy scenes
* subsets dominated by single classes
* subsets dominated by static objects

The primary objective is NOT maximum compression.

The primary objective is:

```text
minimum subset size
subject to preserving detector behavior
```

---

# End
