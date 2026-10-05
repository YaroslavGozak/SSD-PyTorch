# Smoke validation, 2026-09-28

Implemented the existing video benchmark extension and experiment launcher in
`experiments/roi_video_benchmark/run.py`. No replacement inference/evaluation
framework was introduced. PC and Raspberry configs are paired for MobileNet,
VGG16 and YOLO26n. Pre-existing cost-model README/config edits were preserved.

## Actual dataset

The current configured dataset is YOLO-converted sampled ImageNet-VID, YAML `val`
(loader `test`), rooted at
`D:/ImageNet-VID/ImageNet/data/ImageNet2015/imagenet_vid_yolo_sampled`.
Audit: **5,064 frames, 508 video IDs, 30 foreground classes, zero missing or empty
label files**. Sixteen image dimensions were observed, from 240x320 to 720x1280.
Conversion version, source FPS and sampling frequency are unspecified. Annotation
completeness beyond the supplied labels and historical tuning overlap are unknown.
The loader's empty-label filtering remains documented, not silently changed.

The 256-frame smoke is the fixed loader prefix spanning **27 video IDs**. Its
ordered frames and GT have manifest SHA256
`308ef068df05e80f220908731a062a085505059726bc34773cf5c0d664dcbd69`.
Each detector's four policies share identical frame/GT manifests and weight hashes.

## Checks and runs

- Before edits: 11 existing core unittest tests passed; pytest was not installed.
- After edits: all 42 existing cost-model tests, 2 existing label-compatibility
  tests, and 7 new video tests passed. New tests cover linear/tau equivalence,
  exact effective shapes, crop remapping, fallback/uncertainty counters, locked
  configs and tracker resets at video boundaries.
- Windows sandbox temporary-directory ACLs initially prevented two file-backed
  tests from running. The new tests now use mocked artifact I/O; the existing
  temporary-directory regression suites passed outside that sandbox.
- Initial redirected smoke exposed non-ASCII console printing and unconditional
  tensor dumps in the label adapter. Verbosity is now respected and tensor dumps
  removed; completed smoke runs are listed below.
- MobileNet: 32-frame and 256-frame four-policy runs completed. On the 256-frame
  fragment, ROI policies used 16 ROI frames, each with a single proposal, so there
  were no candidate pairs or merges. Zero fallback does **not** establish table
  coverage. AP50=.00204918, AP50:95=.00184426 for all four policies.
- YOLO26n: 256-frame four-policy run completed with real multiple-ROI calls,
  merges, out-of-envelope shapes and explicit fallbacks. A final 32-frame run also
  verified the updated metadata/sensor outputs; that prefix has no candidate pairs.
- VGG16 PC: dataset/weights/calibration/runtime preflight passed; inference was
  not run. Pi hardware was not available; no Pi execution is claimed.
- Re-evaluating saved MobileNet detections reproduced mAP50, mAP50:95 and recall
  exactly without inference.

## YOLO26n smoke counters and quality

| Policy | Frames | Inference calls | Merges | Fallback / candidate attempts | AP50 | AP50:95 | Recall50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full frame | 256 | 256 | 0 | 0 / 0 | .073715 | .058504 | .073770 |
| Existing ROI baseline | 256 | 321 | 36 | N/A (baseline not calibrated) | .073604 | .058153 | .073770 |
| Linear direct cost | 256 | 319 | 39 | 95 / 154 | .073660 | .058176 | .073770 |
| Conservative shape lookup | 256 | 319 | 39 | 95 / 154 | .073660 | .058176 | .073770 |

These numbers validate the implementation; they are **not FPS, accuracy, or policy
superiority conclusions**. VOC-trained weights cover only mapped VID classes;
absolute AP includes unsupported GT classes. The small fragment produced the same
decisions for both calibrated policies, which does not imply general equivalence.
Calibrated ROI coverage was .935780, with 28 uncovered GT objects; effective tensor
area ratio was .65875. Uncovered GT is a coverage diagnostic, not a false-negative
count. The main results preserve detector-driven tracker feedback; no GT proposals
or frozen/replayed proposals were substituted.

**95/154 (61.7%) candidate attempts fell outside the calibration envelope** for
each calibrated YOLO policy. All were recorded and kept separate. Extend the
independent calibration domain before a full scientific comparison. Do not tune
coefficients, margins or confidence thresholds on these validation videos.

Rough extrapolation from measured processing totals is **15–17 minutes for all four
policies over 5,064 frames**, one repetition, on this PC CPU setup (4 threads,
affinity 0–3). This excludes dataset initialization/evaluation/writing and is not a
reliable prediction of other videos' tracker trajectories. No Pi estimate is claimed.

## Outputs and examples

Local artifacts (ignored output directories):

- `outputs/roi_video_benchmark/smoke_256/`: MobileNet 256-frame run.
- `outputs/roi_video_benchmark/yolo_smoke_256/`: YOLO26n 256-frame run.
- `outputs/roi_video_benchmark/yolo_final_32/`: final 32-frame metadata verification.
- `outputs/roi_video_benchmark/vgg_audit/`: VGG16 preflight/audit.

Each contains the files described in [README.md](README.md). Full video lists,
dimensions, frame indices, labels, hashes and configs are stored there. An actual
YOLO frame record (reduced to relevant fields) illustrates that two crops count as
two inference calls:

```json
{
  "video_id": "00000005",
  "frame_index": 421,
  "inference_calls": 2,
  "tensor_shapes": [
    {"height": 224, "width": 128, "calls": 1},
    {"height": 224, "width": 96, "calls": 1}
  ],
  "merges": 0,
  "fallback_count": 1,
  "merge_decisions": [{
    "pair": [0, 1],
    "shapes": [[224, 128], [224, 96], [224, 128]],
    "reason": "outside_envelope",
    "predicted_merge": false,
    "predicted_gain_s": null,
    "fallback_used": "separate"
  }]
}
```

YOLO weights SHA256:
`41f2f96421ebd8d693e12ebca710e029a5f8d504cbcf1fc1c925fb705ab88c9b`.
YOLO calibration SHA256:
`246ecf3c599a545034c0fb69fdd08a807da3f0555f5c4e8c33ee599edb78dae5`.

The legacy CSV remains available, but its historical FPS excludes loading.
Authoritative `summary.json` exposes `FPS_total`, `total_time_s`, latency
mean/median/p95 and separate `inference_time_s` with explicit timing boundaries.
No frame-independent statistical intervals are claimed.

## Remaining limitations

Expand independent calibration to cover observed small/narrow crops; validate
independent held-out video selection before scientific claims. Full-frame warmup
does not cover all ROI shapes. Calibration and video resize antialiasing differ,
and external decoding/remapping/NMS adds runtime overhead. Runtime identity checks
cannot prove identical thermal history or CPU identity from old artifacts lacking
those fields. Pi VGG16 needs a genuine aarch64 artifact; Pi datasets, runtime versions
and cooling must be checked on the device. Use the full-run commands in
[README.md](README.md), including `--smoke-report` to print coverage warnings before
launch. Existing tracker sweeps/base configurations retain their prior behavior.
