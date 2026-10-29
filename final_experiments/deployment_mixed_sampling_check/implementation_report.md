# Deployment-matched mixed sampling implementation check

**Result: PASS**

## Changed files and integration

New sampler/processor/collate and request-aware dataset wrapper are in `tools/train_samplers/deployment_mixed_sampling.py`. Rectangular statistics are in `deployment_sampling_stats.py`. `tools/train_mixed_sampling.py` calls `build_deployment_training_loader` for both Stage 2 and Stage 3; it has no old-sampler stage branch. The legacy sampler remains unchanged for old experiments. The active MobileNet YAML only gains `train_params.roi_sampling.stage2/stage3`; the exact addition is in [yaml_diff.patch](yaml_diff.patch). Existing stage choice, BN flags, optimizer, LR, model architecture, anchors, loss, checkpoints and inference are unchanged.

Changed/added source files:

- `tools/train_samplers/deployment_mixed_sampling.py`
- `tools/train_samplers/deployment_sampling_stats.py`
- `tools/train_mixed_sampling.py`
- `config/imagenet-vid-roissd-mobilenet.yaml`
- `experiments/cpu_pilot/deployment_sampling_check.py`
- `tests/test_deployment_mixed_sampling.py`
- `tests/test_model_first_loading.py`
- `tools/train_samplers/README.md`

## Algorithm

Each batch draws full/ROI, then a conditional area bin, a continuous uniform area, and a log-uniform W/H within the feasible intersection of YAML aspect/min-side/frame bounds. H/W round to the nearest integer (half up), with no stride alignment; invalid rounded shapes are resampled within that bin. This conditional aspect draw avoids silently skewing large-area bin frequencies through rejection. Every sample receives identical H/W and its own immutable deterministic seed. RNG is derived from seed/stage/epoch/batch/slot; `set_epoch(i)` is called before each epoch iteration, including resumed epochs.

The wrapper loads a raw dataset item, selects a feasible reference GT and jitters its crop center, enforcing reference visibility. Ineligible items/positions are deterministically resampled (maximum 20 item attempts, each with at most 20 position attempts). Exhaustion raises an explicit error: crop size is never enlarged, resized or replaced by full-frame. Such item resampling intentionally changes image sampling to the eligible GT subset; it is logged.

Source RGB is resized once to 320×320 with deployment torchvision bilinear antialiasing and uint8 rounding; raw float values decoded from uint8 are recovered before resize. The H×W tensor is sliced, targets are clipped/visibility-filtered, x divides by W and y by H, and ImageNet normalization is applied. Crop tensors are stacked directly, without crop resize or batch padding. Full-frame pixel values match the deployment transform exactly.

## Distribution sanity: 10,000 definitions per stage

| Stage | Full share | ROI bins conditional | Rectangular ROI | Unique ROI shapes | Mean ROI area | H std / W std |
|---|---:|---|---:|---:|---:|---|
| 2 | 49.47% | 4.83%, 15.65%, 34.55%, 44.96% | 98.04% | 4097 | 0.6764 | 55.74 / 54.75 |
| 3 | 29.53% | 25.05%, 35.49%, 24.55%, 14.91% | 98.99% | 6357 | 0.4619 | 67.26 / 66.83 |

The four bins are 10–25%, 25–50%, 50–75%, 75–100% frame area. Conditional ROI probabilities and full shares match YAML within ±2 percentage points. Bounds and rounding error were checked. Empirical definitions and absolute bin shares are saved in the CSVs. Distribution sanity is based on definitions; accepted-object geometry and resampling statistics are separately logged from real batches.

## Real forward/backward sanity

Configured initial checkpoint: `trained_models\best-imagenet-vid-roissd-mobilenet\20261001-phase3-preserved-forms\phase3.pt`; SHA256 `29a861219da99dbde13ba0c8004f945f3ece316d62c332f729712e370e79cd45`. Strict load has no missing/unexpected keys. B=2 real training-image batches passed for 128×224, 224×128, 160×256, 256×160, 240×288, 288×240 and 320×320 in both stages (device cuda). Finite images, valid labels/boxes, finite classification/regression losses, finite/nonzero gradients, and `N_loc == N_cls == N_anchors` all passed. These shapes are explicit sanity requests, not a replacement for continuous sampling. No optimizer was stepped; the complete model state and checkpoint files are unchanged.

## BatchNorm audit

YAML flags remain freeze_backbone_bn=True, freeze_extra_bn=True, train_bn_affine=True. All 65 BN modules remain in eval after `model.eval(); model.train()`. Backbone group is features_stage1; extra group includes features_stage2, conv_to_672, extra pyramid blocks and both head lists. All running means/variances and num_batches_tracked remain unchanged. All BN affine weights/biases are trainable, and every executed BN received finite affine gradients. Per-module details and execution counts are in [bn_audit.csv](bn_audit.csv).

## Statistics and reproducibility

Training writes per-epoch batch CSV, per-sample CSV and summary JSON using both H and W. They include stage/mode/shape/area bin, actual area/aspect, reference area/center offset/visibility, source index and resampling count, plus full/ROI/area-bin/rectangular shares and H/W/aspect/GT geometry distributions. Epoch files overwrite only that epoch on resume. The reference GT area ratio uses its visible target area; pre-clipping area ratio is also retained. Worker-count, global-RNG independence, same-epoch replay, target normalization and deployment pixel parity are covered by unit tests.

Other mixed-training YAMLs must provide their own explicit roi_sampling stage config; missing config fails before checkpoint side effects. Stage 1 stays on its separate script.

Reproduce: `.venv/Scripts/python.exe -m experiments.cpu_pilot.deployment_sampling_check`. This command performs sanity forward/backward only and never launches the training loop.

## Unit-test results

23 mixed-sampling tests and 5 fail-fast/model-loading tests passed. Windows worker-count parity was exercised with 0 and 2 workers. Details: [test_results.txt](test_results.txt).
