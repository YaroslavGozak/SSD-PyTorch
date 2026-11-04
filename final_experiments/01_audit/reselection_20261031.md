# Checkpoint reselection: 2026-10-31

The user explicitly selected these checkpoints for all subsequent tests:

| Family | Selected checkpoint | Saved validation mAP50 |
|---|---|---:|
| ROI-SSD | `trained_models/imagenet-vid-roissd-20261028/stage3/best_map.pt` | 0.72504810 |
| ROI-SSD-MobileNet | `trained_models/imagenet-vid-roissd-mobilenet-20261027/stage3/best_map.pt` | 0.68411920 |

These values are checkpoint metadata corroborated by the corresponding training CSV. They use the saved training validation protocol, and are not fresh canonical audit measurements or proof of superiority over every previous candidate. The explicit selection supersedes the previous role-specific recommendations. YOLO26n is unchanged.

Both checkpoints passed strict state-dict loading and real-image CPU batch=1 inference at 320x320, 96x96, 128x128, 160x160 and 160x192. Predictions have finite scores and boxes, valid IDs, and normalized coordinate bounds. Both run configurations use the audited 30-class dataset order and 31 SSD outputs including background. Full hashes and evidence are in [reselection_20261031.json](reselection_20261031.json).

The canonical inference YAMLs preserve the selected runs' model parameters and confidence 0.5, explicitly point to `best_map.pt` rather than their original `last.pt`, and do not modify the training run files.

The stage-2 selection lock and stage-3 pilot preparation now use these selections. Previous artifacts are preserved in:

- `experiments/roissd_128_anomaly/data_before_20261031/`
- `final_experiments/diagnostics_128_before_20261031/`
- `final_experiments/pilot_before_20261031/`

The fresh pilot retains the exact original video split and manifest hashes. Its status is `PREPARED_NOT_BENCHMARKED`; old accuracy gates, padding choices and cost profiles do not describe these new checkpoints. Fresh canonical evaluation, stage-2 diagnostics, padding sweep and CPU calibration remain pending. Historical regression and crop-domain studies remain evidence about their explicitly named historical checkpoints.

Verification: 10 real-image forwards passed; pilot preparation and diagnostic lock preparation succeeded; both pilot runner helper tests passed. No training or long benchmark was launched.
