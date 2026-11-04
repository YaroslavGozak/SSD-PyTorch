# Fresh Stage-2 regression check

**Structural regression: PASS. Existing 100-shape sweep attribution to the selected best_map.pt: NOT VERIFIED.**

This check freshly loads the two locked checkpoints and tests runtime structure on boundary inputs. It is not a rerun of the dense Stage-2 accuracy diagnostic or the 100-shape sweep. No model code, weights, thresholds or depth policy were changed.

## Sweep provenance and checkpoint binding

### ROI-SSD

- Selected checkpoint: `trained_models/imagenet-vid-roissd-20261028/stage3/best_map.pt`.
- Selected SHA-256: `f8f49132e08a7c5bf3744ce32e46e1b0fed6595aabd9cf59cf68699569503f76`.
- Selected tensor-state SHA-256: `1db02be68173437ba6c8d19de394656a8bd03067b34d6dc2d9c536182aac8ffc`.
- Saved sweep: `final_experiments/roi-ssd-retrained/fixed_size_results_v2.csv`; SHA-256 `d98158d4c0952af518455ba1359e4804b2a8bc542974b94c1e892be46d63b493`.
- Grid completeness: 100 unique shapes, all H/W pairs in 32,64,...,320.
- CSV checkpoint field: last.pt. Current last.pt SHA-256: `33a5911273127c26539730ec8c3be4d4b5e9fdf03143251e0d5c6f0273e31a48`.
- Current last.pt tensor-state SHA-256: `c001d5efe39af648c27863a68988bbf9ea577da35e2c1d337f139daf342eb1ad`.
- Current last.pt and best_map.pt have identical tensors: False.
- Historical inference-time checkpoint hash: absent. File paths and current hashes cannot prove which bytes the earlier sweep loaded.
- Attribution decision: do not relabel these 100 accuracy values as best_map.pt measurements. Provenance links the saved CSV and the current selection in one audit record while preserving the mismatch.

### ROI-SSD-MobileNet

- Selected checkpoint: `trained_models/imagenet-vid-roissd-mobilenet-20261027/stage3/best_map.pt`.
- Selected SHA-256: `1a830c339f060905f1042e10e98906be0cab43fa94cb3867bfc683ad14adc804`.
- Selected tensor-state SHA-256: `b4f1d8748700e0c1e92f5b115ceb6eea8bc497815373624aca32a5431f960d7a`.
- Saved sweep: `final_experiments/roi-ssd-mobilenet-retrained/fixed_size_results_v2.csv`; SHA-256 `e640d6aff53765a7d9708ce56dcee76a2b2629685cbebd43c67ab78bcaa20cf8`.
- Grid completeness: 100 unique shapes, all H/W pairs in 32,64,...,320.
- CSV checkpoint field: last.pt. Current last.pt SHA-256: `57976c7b98cacea415b8fcee78a559faad0a9c07d406fb8adb435fa34d5cefbb`.
- Current last.pt tensor-state SHA-256: `620269c8ff7a1ce612d8ccd7b495decc20fb0f68b81dd75fae21ebf5f53ec546`.
- Current last.pt and best_map.pt have identical tensors: False.
- Historical inference-time checkpoint hash: absent. File paths and current hashes cannot prove which bytes the earlier sweep loaded.
- Attribution decision: do not relabel these 100 accuracy values as best_map.pt measurements. Provenance links the saved CSV and the current selection in one audit record while preserving the mismatch.

## Fresh runtime verification

CPU, batch=1, 8 PyTorch intra-op threads, 1 inter-op thread, 1 OpenCV thread. Three hashed real canonical source images per shape; shared project FixedSizeTestTransform with seed 20261023, confidence=0.5, NMS IoU=0.45, maximum 200 detections. Shape notation is H x W. No silent input padding or rounding.

19 shapes per model, 3 crops per shape: 114 fresh forwards total. All passed strict checkpoint loading, selected/loaded tensor identity, until_1x1 policy, observed-vs-reported active heads, no head after the first 1x1 map, and localization/classification/anchor slot equality. Anchor ordering and geometry independently agree within 1e-6. Decoded boxes/scores are finite, boxes are ordered/in bounds and class IDs valid. Tensor/GT hashes match across families; checkpoint and model tensor hashes remain unchanged.

| Model | H x W | Active depth | Heads (zero based) | Feature H x W | Loc = cls = anchors | Stop reason |
|---|---|---:|---|---|---:|---|
| ROI-SSD | 95 x 95 | 4 | [0, 1, 2, 3] | [[12, 12], [6, 6], [3, 3], [2, 2]] | 870 | INFEASIBLE |
| ROI-SSD | 96 x 96 | 4 | [0, 1, 2, 3] | [[12, 12], [6, 6], [3, 3], [2, 2]] | 870 | INFEASIBLE |
| ROI-SSD | 97 x 97 | 4 | [0, 1, 2, 3] | [[12, 12], [6, 6], [3, 3], [2, 2]] | 870 | INFEASIBLE |
| ROI-SSD | 127 x 127 | 4 | [0, 1, 2, 3] | [[16, 16], [8, 8], [4, 4], [2, 2]] | 1528 | INFEASIBLE |
| ROI-SSD | 128 x 128 | 4 | [0, 1, 2, 3] | [[16, 16], [8, 8], [4, 4], [2, 2]] | 1528 | INFEASIBLE |
| ROI-SSD | 129 x 129 | 4 | [0, 1, 2, 3] | [[16, 16], [8, 8], [4, 4], [2, 2]] | 1528 | INFEASIBLE |
| ROI-SSD | 139 x 139 | 4 | [0, 1, 2, 3] | [[17, 17], [8, 8], [4, 4], [2, 2]] | 1660 | INFEASIBLE |
| ROI-SSD | 140 x 140 | 5 | [0, 1, 2, 3, 4] | [[18, 18], [9, 9], [5, 5], [3, 3], [1, 1]] | 1990 | STOPPED_AT_1X1 |
| ROI-SSD | 141 x 141 | 5 | [0, 1, 2, 3, 4] | [[18, 18], [9, 9], [5, 5], [3, 3], [1, 1]] | 1990 | STOPPED_AT_1X1 |
| ROI-SSD | 255 x 255 | 5 | [0, 1, 2, 3, 4] | [[32, 32], [16, 16], [8, 8], [4, 4], [2, 2]] | 6128 | INFEASIBLE |
| ROI-SSD | 256 x 256 | 5 | [0, 1, 2, 3, 4] | [[32, 32], [16, 16], [8, 8], [4, 4], [2, 2]] | 6128 | INFEASIBLE |
| ROI-SSD | 257 x 257 | 5 | [0, 1, 2, 3, 4] | [[32, 32], [16, 16], [8, 8], [4, 4], [2, 2]] | 6128 | INFEASIBLE |
| ROI-SSD | 267 x 267 | 5 | [0, 1, 2, 3, 4] | [[33, 33], [16, 16], [8, 8], [4, 4], [2, 2]] | 6388 | INFEASIBLE |
| ROI-SSD | 268 x 268 | 6 | [0, 1, 2, 3, 4, 5] | [[34, 34], [17, 17], [9, 9], [5, 5], [3, 3], [1, 1]] | 7034 | STOPPED_AT_1X1 |
| ROI-SSD | 269 x 269 | 6 | [0, 1, 2, 3, 4, 5] | [[34, 34], [17, 17], [9, 9], [5, 5], [3, 3], [1, 1]] | 7034 | STOPPED_AT_1X1 |
| ROI-SSD | 128 x 160 | 4 | [0, 1, 2, 3] | [[16, 20], [8, 10], [4, 5], [2, 3]] | 1916 | INFEASIBLE |
| ROI-SSD | 160 x 128 | 4 | [0, 1, 2, 3] | [[20, 16], [10, 8], [5, 4], [3, 2]] | 1916 | INFEASIBLE |
| ROI-SSD | 256 x 320 | 5 | [0, 1, 2, 3, 4] | [[32, 40], [16, 20], [8, 10], [4, 5], [2, 3]] | 7664 | INFEASIBLE |
| ROI-SSD | 320 x 256 | 5 | [0, 1, 2, 3, 4] | [[40, 32], [20, 16], [10, 8], [5, 4], [3, 2]] | 7664 | INFEASIBLE |
| ROI-SSD-MobileNet | 95 x 95 | 4 | [0, 1, 2, 3] | [[6, 6], [3, 3], [2, 2], [1, 1]] | 228 | STOPPED_AT_1X1 |
| ROI-SSD-MobileNet | 96 x 96 | 4 | [0, 1, 2, 3] | [[6, 6], [3, 3], [2, 2], [1, 1]] | 228 | STOPPED_AT_1X1 |
| ROI-SSD-MobileNet | 97 x 97 | 4 | [0, 1, 2, 3] | [[7, 7], [4, 4], [2, 2], [1, 1]] | 322 | STOPPED_AT_1X1 |
| ROI-SSD-MobileNet | 127 x 127 | 4 | [0, 1, 2, 3] | [[8, 8], [4, 4], [2, 2], [1, 1]] | 382 | STOPPED_AT_1X1 |
| ROI-SSD-MobileNet | 128 x 128 | 4 | [0, 1, 2, 3] | [[8, 8], [4, 4], [2, 2], [1, 1]] | 382 | STOPPED_AT_1X1 |
| ROI-SSD-MobileNet | 129 x 129 | 5 | [0, 1, 2, 3, 4] | [[9, 9], [5, 5], [3, 3], [2, 2], [1, 1]] | 556 | STOPPED_AT_1X1 |
| ROI-SSD-MobileNet | 139 x 139 | 5 | [0, 1, 2, 3, 4] | [[9, 9], [5, 5], [3, 3], [2, 2], [1, 1]] | 556 | STOPPED_AT_1X1 |
| ROI-SSD-MobileNet | 140 x 140 | 5 | [0, 1, 2, 3, 4] | [[9, 9], [5, 5], [3, 3], [2, 2], [1, 1]] | 556 | STOPPED_AT_1X1 |
| ROI-SSD-MobileNet | 141 x 141 | 5 | [0, 1, 2, 3, 4] | [[9, 9], [5, 5], [3, 3], [2, 2], [1, 1]] | 556 | STOPPED_AT_1X1 |
| ROI-SSD-MobileNet | 255 x 255 | 5 | [0, 1, 2, 3, 4] | [[16, 16], [8, 8], [4, 4], [2, 2], [1, 1]] | 1532 | STOPPED_AT_1X1 |
| ROI-SSD-MobileNet | 256 x 256 | 5 | [0, 1, 2, 3, 4] | [[16, 16], [8, 8], [4, 4], [2, 2], [1, 1]] | 1532 | STOPPED_AT_1X1 |
| ROI-SSD-MobileNet | 257 x 257 | 6 | [0, 1, 2, 3, 4, 5] | [[17, 17], [9, 9], [5, 5], [3, 3], [2, 2], [1, 1]] | 1866 | STOPPED_AT_1X1 |
| ROI-SSD-MobileNet | 267 x 267 | 6 | [0, 1, 2, 3, 4, 5] | [[17, 17], [9, 9], [5, 5], [3, 3], [2, 2], [1, 1]] | 1866 | STOPPED_AT_1X1 |
| ROI-SSD-MobileNet | 268 x 268 | 6 | [0, 1, 2, 3, 4, 5] | [[17, 17], [9, 9], [5, 5], [3, 3], [2, 2], [1, 1]] | 1866 | STOPPED_AT_1X1 |
| ROI-SSD-MobileNet | 269 x 269 | 6 | [0, 1, 2, 3, 4, 5] | [[17, 17], [9, 9], [5, 5], [3, 3], [2, 2], [1, 1]] | 1866 | STOPPED_AT_1X1 |
| ROI-SSD-MobileNet | 128 x 160 | 5 | [0, 1, 2, 3, 4] | [[8, 10], [4, 5], [2, 3], [1, 2], [1, 1]] | 492 | STOPPED_AT_1X1 |
| ROI-SSD-MobileNet | 160 x 128 | 5 | [0, 1, 2, 3, 4] | [[10, 8], [5, 4], [3, 2], [2, 1], [1, 1]] | 492 | STOPPED_AT_1X1 |
| ROI-SSD-MobileNet | 256 x 320 | 6 | [0, 1, 2, 3, 4, 5] | [[16, 20], [8, 10], [4, 5], [2, 3], [1, 2], [1, 1]] | 1928 | STOPPED_AT_1X1 |
| ROI-SSD-MobileNet | 320 x 256 | 6 | [0, 1, 2, 3, 4, 5] | [[20, 16], [10, 8], [5, 4], [3, 2], [2, 1], [1, 1]] | 1928 | STOPPED_AT_1X1 |

## Interpretation and limits

No structural mismatch was detected on these fresh boundary probes. Observed feature block calls also match the selected depth exactly; no unused later block executes. The until_1x1 policy stops at the first 1x1 map when reachable, but also stops before a spatially infeasible convolution or when the available pyramid is exhausted. Therefore an INFEASIBLE stop with a larger final feature map is not itself a failure; the table records these cases explicitly. This supports current-checkpoint runtime correctness only at the tested shapes; it does not establish accuracy at every shape, explain the historical 128x128 point causally, or prove the current best weights reproduce the old CSV metrics.

To attribute a 100-shape accuracy sweep to best_map.pt, recover an inference-time hash or equivalent immutable provenance proving those exact weights were loaded, or rerun the sweep with the selected checkpoint hash recorded at execution. Matching directory names or an equal best_map scalar is insufficient. Both current last.pt tensor states differ from their selected best_map.pt states.

The Stage-3 pilot remains blocked by measured MobileNet accuracy/coverage criteria. A passing structural regression does not waive that gate. No CA/FULL-TECH, retraining, long accuracy sweep or Raspberry Pi run was performed.

Reproduce this check from the repository root: `.venv/Scripts/python.exe -m experiments.roissd_128_anomaly.fresh_regression`. Machine-readable results are sweep_provenance.json, boundary_records.jsonl and metadata.json alongside this report. No other report is required to interpret these findings.

## Verification suite

All 37 diagnostic/runtime tests passed. The reproducible synthetic-crop test now uses an inline synthetic fixture, independent of checkpoint-specific regenerated data directories. Fresh forward records were checked for 114 PASS rows; the CSV export contains observed heads, feature shapes and all three slot counts.
