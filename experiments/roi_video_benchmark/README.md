# ROI video benchmark

This launcher runs **full_frame**, **roi_baseline**, **linear_direct_cost**, and
**shape_lookup_conservative** through the existing `VideoSequenceBenchmark` and
`process_frame`. The launch script lives in this experiment directory. See
[AUDIT.md](AUDIT.md) for the pre-change audit and [SMOKE_REPORT.md](SMOKE_REPORT.md)
for measured smoke results and limitations.

Run all commands from the repository root. On Windows use `.venv/Scripts/python.exe`
instead of `python` if the environment is not activated.

```sh
# MobileNet: fixed 256-frame prefix, all four policies, one repetition
python -m experiments.roi_video_benchmark.run --config config/benchmark-vid-roi-policies.yaml --max-frames 256 --output outputs/roi_video_benchmark/mobilenet_smoke_new

# YOLO26n: same data and protocol, separately calibrated detector/runtime
python -m experiments.roi_video_benchmark.run --config config/benchmark-vid-roi-policies-yolo26n.yaml --max-frames 256 --output outputs/roi_video_benchmark/yolo_smoke_new

# Inspect dataset and validate calibration/runtime without inference
python -m experiments.roi_video_benchmark.run --config config/benchmark-vid-roi-policies-vgg16.yaml --audit-only --output outputs/roi_video_benchmark/vgg_audit_new

# Full data, after reviewing the smoke report and extending calibration coverage
python -m experiments.roi_video_benchmark.run --config config/benchmark-vid-roi-policies-yolo26n.yaml --max-frames 0 --smoke-report outputs/roi_video_benchmark/yolo_smoke_256/comparison.json --output outputs/roi_video_benchmark/yolo_full_new

# Re-evaluate saved outputs with the existing evaluator; no model/inference
python -m experiments.roi_video_benchmark.evaluate outputs/roi_video_benchmark/yolo_smoke_256/repeat_0/shape_lookup_conservative/detections.json --output outputs/roi_video_benchmark/yolo_smoke_256/reevaluated.json

# Focused tests (unittest, no pytest dependency)
python -m unittest experiments.roi_video_benchmark.test_video -q
```

Use the corresponding file under `config/raspberry/` on Pi. The three names are
`benchmark-vid-roi-policies.yaml` (MobileNet),
`benchmark-vid-roi-policies-yolo26n.yaml`, and
`benchmark-vid-roi-policies-vgg16.yaml`. Both platforms share the protocol by YAML
inheritance; Pi overrides data location, calibration, power policy, baseline tau,
cooling label and output directory. Pi VGG16 calibration is explicitly missing.
Do not replace it with an AMD64 artifact merely stored in `raspberry-outputs`.

Set `video_experiment.videos` to an ordered-data selection by video IDs (loader
order is preserved), `max_frames: 0` for all selected frames, and `repetitions`
for repeated four-policy runs. Seed, warmup, confidence bound/margin, cooling label
and output directory are in the same section. Existing keys supply dataset YAML
root/split, detector weights, device, confidence/NMS, ROI padding, tracker settings,
keyframe interval, and runtime threads/affinity/power. Batch size is explicitly
one. The existing full-frame fallback when no proposals exist remains enabled.
The main experiment rejects oracle tracking, input dropout, and GT crop transforms.
Tracker state resets at each video and policy. Policy order is fixed and recorded;
thermal/order effects require repeated independent sessions for a final study.

The current data is **sampled ImageNet-VID converted to YOLO, validation split**:
5,064 labeled frames in 508 video IDs, not the historical 8-video voc-vid set.
The loader's `test` means YAML `val`; it filters empty/missing-label frames. The
audited local copy has no missing or empty label files, but annotation completeness
beyond the supplied labels is unknown. FPS and conversion version are not supplied.
Only retained sampled frames are processed: keyframes use their original frame
indices, so this is not necessarily continuous real-time video.

All presets use available **VOC-trained weights**, remapped to VID labels.
The evaluator retains every loader class, including classes unsupported by VOC.
This limits absolute accuracy interpretation. Compare policies within each
detector/platform/runtime; do not compare these AP values across detectors or call
them an ImageNet-trained-model result. Calibration uses generated images; historical
parameter-selection overlap with the validation videos is unknown.

The baseline is unchanged: PC tau=17000 and Pi tau=225677 from their base configs
(the older tracker sweeps separately override tau=225000). It uses beta-weighted
area delta **greater than** tau, an area-ratio guard and canvas/full-frame shortcuts.
It is not the linear `Am-A1-A2<K/c` rule. Calibrated policies retain pair enumeration,
maximum-gain choice, first-tie behavior, pair removal and union append. They replace
baseline scores/shortcuts with calibrated decisions, without tuning the tracker.
End-to-end detector feedback may therefore change subsequent proposals.

`linear_direct_cost` invokes the existing `linear_tau` implementation of the
strict direct-time comparison. Lookup uses the existing one-sided confidence bound
`gain - z * sum(marginal_SE) > margin`, with worst-case correlation. Effective
shapes come from the actual coordinate roundtrip and crop function; lookup never
rounds an observed shape to a different table entry. Outside-envelope shapes,
missing table entries and missing uncertainty explicitly fall back to **separate**.
Candidate attempts are counted, including reconsidered pairs after a merge.
Fallback warnings above 10% appear in summaries; `--smoke-report` prints previous
warnings before a full run. Do not tune coefficients/margins on these videos.

Runtime checks reject mismatched weights, architecture, torch version, model config,
normalization, precision, stride, thread count or affinity. YOLO additionally checks
Ultralytics version and uses the calibrated raw-forward invocation; prediction-API
calibrations are not interchangeable. Detector decoding/label remapping and external
NMS add overhead beyond raw-forward calibration. Resize antialiasing differs from
the generated-image calibration; normalization and actual tensor shapes match.
Five full-frame warmup calls precede measurement, but not every possible ROI shape
is prewarmed. Temperature, frequency and Pi throttling are recorded when available.

Each `repeat_N/POLICY/` contains:

| File | Content |
|---|---|
| `frames.jsonl` | Every processed frame, actual shapes/calls/pixels, timings, ROI coverage, tracker resets, merge decisions and fallback reasons |
| `videos.json` | Quality and cost aggregates per video, computed from that video's evaluator inputs |
| `detections.json` | Existing evaluator's prediction/GT/difficult structures for repeat evaluation |
| `manifest.json` | Ordered paths, video/frame IDs, dimensions, GT and difficult flags; SHA256 |
| `summary.json` | Authoritative quality and end-to-end metrics, exact command, expanded configs, weights/calibration hashes, versions, git state, runtime, warnings and definitions |
| `legacy_metrics.csv` | Existing benchmark CSV; its historical `fps_total` is pipeline-only, excluding image loading. Use `summary.json` `FPS_total` for the experiment. |

At the run root, `dataset_audit.json`, `preflight.json`, and `comparison.json` record
the selection, runtime validation, shared frame/annotation/weight identity, warnings,
and a rough full-processing duration estimate. Outputs use fresh directories.
`FPS_total` includes decoding/preprocessing and synchronized pipeline processing,
but excludes initialization, warmup, evaluator work and artifact writing. Inference
time is reported separately. Area ratio is summed **actual tensor pixels** divided
by summed full-frame transformed tensor pixels across every processed frame.
ROI count excludes full-frame calls, which are counted in `inference_calls` and
`full_frame_fraction`. Coverage is the fraction of GT boxes whose intersection with
at least one actual ROI covers >=0.5 of GT area. `gt_outside_roi` counts uncovered
GT objects, not detector false negatives. `mAP50_95` averages the existing evaluator
at ten IoUs; it is not a replacement COCO evaluator. No frame-iid confidence
intervals are produced; future uncertainty analysis must resample whole videos.
