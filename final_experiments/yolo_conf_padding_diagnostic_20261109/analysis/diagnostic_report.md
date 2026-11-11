# YOLO26n operational-confidence x padding diagnostic

## Protocol

- Evaluation confidence is fixed at 0.01 for the one full-frame baseline and all SEG runs.
- Only operational detections seed the static previous-localization tracker; hold is 0.
- Periodic full-frame interval is 10 frames; merge/CA/FULL-TECH are disabled.
- The same pinned 32-video, 320-frame pilot manifest is used without shuffling.
- Full-frame baseline mAP50: `0.818573`.
- Gates: mAP50 retention >= 0.95; GT ROI coverage mean >= 0.98.

## Gate matrix

| operational_conf | 64 | 96 | 128 | 160 | 192 | 224 |
|---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.01 | PASS | PASS | PASS | PASS | PASS | PASS |
| 0.05 | PASS | PASS | PASS | PASS | PASS | PASS |
| 0.10 | PASS | PASS | PASS | PASS | PASS | PASS |
| 0.20 | FAIL | FAIL | FAIL | FAIL | FAIL | PASS |
| 0.35 | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |

## Recommendation

Recommended `(operational_conf, padding)` = `(0.10, 64)`.
It has tensor area ratio `1.0304`, `2.016` candidate segments/frame, mAP50 retention `0.9763`, and coverage `1.0000`.
Selection minimizes tensor area among passing configurations; configurations within 1% of the minimum are ordered by fewer candidate segments, smaller padding, then higher confidence. This is diagnostic only and does not change the main Stage 3 configuration.

## Coverage by threshold and padding

Entries show `coverage / mAP50 retention / candidate segments per frame`.

| Conf | Padding | Coverage | mAP50 retention | Candidate seg/frame | Tensor area ratio | Gate |
|---:|---:|---:|---:|---:|---:|:---:|
| 0.01 | 64 | 1.0000 | 0.9745 | 6.400 | 2.2231 | PASS |
| 0.01 | 96 | 1.0000 | 0.9952 | 5.434 | 2.7428 | PASS |
| 0.01 | 128 | 1.0000 | 0.9988 | 4.406 | 2.8995 | PASS |
| 0.01 | 160 | 1.0000 | 1.0022 | 3.447 | 2.7226 | PASS |
| 0.01 | 192 | 1.0000 | 0.9985 | 3.147 | 2.7334 | PASS |
| 0.01 | 224 | 1.0000 | 0.9998 | 2.919 | 2.6954 | PASS |
| 0.05 | 64 | 1.0000 | 0.9682 | 2.556 | 1.1777 | PASS |
| 0.05 | 96 | 1.0000 | 0.9922 | 2.275 | 1.4511 | PASS |
| 0.05 | 128 | 1.0000 | 1.0001 | 2.091 | 1.6198 | PASS |
| 0.05 | 160 | 1.0000 | 1.0043 | 1.947 | 1.7005 | PASS |
| 0.05 | 192 | 1.0000 | 1.0021 | 1.916 | 1.8050 | PASS |
| 0.05 | 224 | 1.0000 | 0.9998 | 1.850 | 1.8061 | PASS |
| 0.10 | 64 | 1.0000 | 0.9763 | 2.016 | 1.0304 | PASS |
| 0.10 | 96 | 1.0000 | 0.9942 | 1.837 | 1.2421 | PASS |
| 0.10 | 128 | 1.0000 | 0.9873 | 1.806 | 1.4551 | PASS |
| 0.10 | 160 | 1.0000 | 1.0019 | 1.637 | 1.4992 | PASS |
| 0.10 | 192 | 1.0000 | 0.9998 | 1.628 | 1.5682 | PASS |
| 0.10 | 224 | 1.0000 | 1.0005 | 1.641 | 1.6242 | PASS |
| 0.20 | 64 | 0.9625 | 0.9762 | 1.659 | 0.9158 | FAIL |
| 0.20 | 96 | 0.9643 | 0.9978 | 1.544 | 1.1296 | FAIL |
| 0.20 | 128 | 0.9643 | 0.9916 | 1.522 | 1.3151 | FAIL |
| 0.20 | 160 | 0.9643 | 1.0005 | 1.441 | 1.3701 | FAIL |
| 0.20 | 192 | 0.9732 | 0.9991 | 1.434 | 1.4361 | FAIL |
| 0.20 | 224 | 0.9821 | 1.0007 | 1.459 | 1.4869 | PASS |
| 0.35 | 64 | 0.9625 | 0.9795 | 1.403 | 0.8714 | FAIL |
| 0.35 | 96 | 0.9625 | 0.9984 | 1.397 | 1.0888 | FAIL |
| 0.35 | 128 | 0.9625 | 0.9924 | 1.384 | 1.2500 | FAIL |
| 0.35 | 160 | 0.9625 | 0.9972 | 1.353 | 1.3387 | FAIL |
| 0.35 | 192 | 0.9696 | 0.9985 | 1.328 | 1.3822 | FAIL |
| 0.35 | 224 | 0.9750 | 0.9994 | 1.363 | 1.4385 | FAIL |

## Factor comparisons

Reducing operational confidence is sufficient at padding(s) [64, 96, 128, 160, 192, 224]; padding 224 alone does not rescue conf=0.35.
- At padding 64, reducing operational confidence from 0.35 to 0.01 changes coverage by +0.0375 (0.9625 -> 1.0000).
- At padding 96, reducing operational confidence from 0.35 to 0.01 changes coverage by +0.0375 (0.9625 -> 1.0000).
- At padding 128, reducing operational confidence from 0.35 to 0.01 changes coverage by +0.0375 (0.9625 -> 1.0000).
- At padding 160, reducing operational confidence from 0.35 to 0.01 changes coverage by +0.0375 (0.9625 -> 1.0000).
- At padding 192, reducing operational confidence from 0.35 to 0.01 changes coverage by +0.0304 (0.9696 -> 1.0000).
- At padding 224, reducing operational confidence from 0.35 to 0.01 changes coverage by +0.0250 (0.9750 -> 1.0000).
- At operational confidence 0.01, increasing padding from 64 to 224 changes coverage by +0.0000 (1.0000 -> 1.0000).
- At operational confidence 0.05, increasing padding from 64 to 224 changes coverage by +0.0000 (1.0000 -> 1.0000).
- At operational confidence 0.10, increasing padding from 64 to 224 changes coverage by +0.0000 (1.0000 -> 1.0000).
- At operational confidence 0.20, increasing padding from 64 to 224 changes coverage by +0.0196 (0.9625 -> 0.9821).
- At operational confidence 0.35, increasing padding from 64 to 224 changes coverage by +0.0125 (0.9625 -> 0.9750).

## Temporal misses

- conf=0.20, padding=64: 21 missed GT across 1 videos; video 23010 frames 83-89: 7 consecutive frame(s), 21 missed GT.
- conf=0.20, padding=96: 20 missed GT across 1 videos; video 23010 frames 83-89: 7 consecutive frame(s), 20 missed GT.
- conf=0.20, padding=128: 20 missed GT across 1 videos; video 23010 frames 83-89: 7 consecutive frame(s), 20 missed GT.
- conf=0.20, padding=160: 20 missed GT across 1 videos; video 23010 frames 83-89: 7 consecutive frame(s), 20 missed GT.
- conf=0.20, padding=192: 15 missed GT across 1 videos; video 23010 frames 84-89: 6 consecutive frame(s), 15 missed GT.
- conf=0.35, padding=64: 21 missed GT across 1 videos; video 23010 frames 83-89: 7 consecutive frame(s), 21 missed GT.
- conf=0.35, padding=96: 21 missed GT across 1 videos; video 23010 frames 83-89: 7 consecutive frame(s), 21 missed GT.
- conf=0.35, padding=128: 21 missed GT across 1 videos; video 23010 frames 83-89: 7 consecutive frame(s), 21 missed GT.
- conf=0.35, padding=160: 21 missed GT across 1 videos; video 23010 frames 83-89: 7 consecutive frame(s), 21 missed GT.
- conf=0.35, padding=192: 17 missed GT across 1 videos; video 23010 frames 83-89: 7 consecutive frame(s), 17 missed GT.
- conf=0.35, padding=224: 14 missed GT across 1 videos; video 23010 frames 83-89: 7 consecutive frame(s), 14 missed GT.

### Known video 23010, frames 83-89

- conf=0.01: p64:0, p96:0, p128:0, p160:0, p192:0, p224:0 missed GT.
- conf=0.05: p64:0, p96:0, p128:0, p160:0, p192:0, p224:0 missed GT.
- conf=0.10: p64:0, p96:0, p128:0, p160:0, p192:0, p224:0 missed GT.
- conf=0.20: p64:21, p96:20, p128:20, p160:20, p192:15, p224:10 missed GT.
- conf=0.35: p64:21, p96:21, p128:21, p160:21, p192:17, p224:14 missed GT.

## Output notes

- `sweep.csv` has one row per configuration and separate MAP_GATE_PASS, COVERAGE_GATE_PASS, and FINAL_GATE_PASS fields.
- `frame_level.csv` records evaluation/operational detections, current and next candidate segments, fallback, covered/uncovered counts, and each uncovered GT class/bbox.
- Timing is recorded by the shared Stage 3 pipeline but is not used to select a configuration.
- This diagnostic does not update the frozen operational confidence or main Stage 3 outputs.
