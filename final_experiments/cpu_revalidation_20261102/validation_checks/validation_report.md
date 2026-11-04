# Preparation validation — 2026-11-02

PASS: the final prepared session validates checkpoint/config/source/dataset YAML/
manifest hashes. All 5,064 validation image paths exist and all label SHA-256
values match the frozen validation manifest. Pilot sessions for all three models
load the same 320 pilot frames and 4,744 final-candidate frames without inference.

PASS: `--stage all --dry-run` produces nine accuracy commands (full-frame,
pad-loop, fixed-size-loop for each model) followed by three confidence-selection,
three padding, three cost and three pipeline commands, then aggregation. The model
order is YOLO26n, VGG16, MobileNet. No dry-run job markers/results were created.

PASS: 19 targeted unit tests across preparation and the full-frame extension:

- 7 launcher tests: full-frame/loop ordering/CPU, full-frame-only commands,
  full-frame aggregation, existing-output rejection, source-hash mismatch,
  child-failure stop, CA/FULL-TECH identical-workload explanation.
- 3 YOLO geometry tests: rectangular crop, independent stride padding, coordinate
  offsets, neutral fill, edge clipping, unchanged aligned square.
- 4 existing pilot helper tests: affine merger/effective shapes and default gate.
- 5 existing inference tests: crop reuse equivalence, atomic summary, mocked
  sweep control flow, batched metric equivalence, unsupported-resource fallback.

An additional existing worker-process test could not create a Windows
multiprocessing pipe in the sandbox (`PermissionError`, WinError 5). Prepared
commands use `num_workers=0`, and do not depend on that multiprocessing path.

PASS: one real-image padded example, one fixed 160×192 example, and one FF plus
one explicit SEG frame per checkpoint on CPU. Evidence: `technical_smoke.json`
and per-model `*_audit.json`. Observed YOLO padded tensor: 128×96 from a 101×72
geometric crop; fixed tensor: 160×192. All observed threads: torch=8, inter-op=1,
OpenCV=1. These are technical examples, not accuracy/performance conclusions.
Later preparation edits only strengthened nonfinite-output/device guards and
report/preflight handling; final source hashes are in the experiment manifest.

No full pad-loop, no real 100-shape inference sweep, no full Stage 03 pilot,
and no cost calibration grid was executed. Existing mocked sweep tests execute
control flow with fake detectors; they do not constitute experimental runs.

Syntax checks and `git diff --check` passed for the modified production files.

Full-frame extension: only unit tests, CLI/config checks and dry-run were run.
No full 5,064-frame accuracy evaluation was performed during preparation.

## Confidence separation update

Quality configs now use conf_floor=0.01 for all three models, including both SSD
runtime score cutoffs. Operational config templates are separate and require
frozen pilot selection; source selected configs/weights remain unchanged.

PASS: 21 current targeted tests (8 launcher/config tests, 9 confidence-selection
tests, 4 existing pilot-helper tests). These check F1 optimization, recall ties,
duplicate/wrong-class FP, undefined selection rejection, immutable evidence,
effective SSD/YOLO cutoff override and blocking before model load without a
frozen decision. A mocked FF pipeline verifies pilot-only inference evidence and
the 20-candidate curve. No detector is executed by these tests.

Final dry-run plans 21 jobs and aggregation. Config/session/source hashes and
all label hashes validate. Operational thresholds are UNSELECTED: neither the
320-frame selection stage nor long evaluation was run by the agent.

Earlier real-image smoke outputs here were produced before the confidence
separation. Their preparation identity is preserved in
`pre_confidence_split_session.json`; they are technical historical examples,
not measured quality results for the current configs or selected operational thresholds.

## ROI-SSD/VGG16 five-cutoff diagnostic update

Prepared separate full-validation configs for confidence 0.01, 0.05, 0.1, 0.25,
0.5. Each sets both the inference and SSD internal score cutoff. Standard quality
configs remain at 0.01; pilot selection and its frozen thresholds are unchanged.

PASS: 24 current tests (11 launcher/config/diagnostic tests, 9 operational
confidence tests, 4 pilot-helper tests). Added tests cover five sequential FF
commands, cutoff configuration without changing NMS or the quality config,
separate five-row summary/aggregation and rejection of incomplete frame coverage.

CLI dry-run: `all` plans 26 jobs, `accuracy --with-confidence-diagnostic` plans
14, standalone `confidence-diagnostic` plans five. Diagnostic outputs and summary
do not exist yet. Only preparation, tests and dry-run checks were performed.
