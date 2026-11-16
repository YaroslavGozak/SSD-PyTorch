# Execution blockers (implementation checkout, 2026-11-16)

Implemented code must not be confused with executed real-model experiments.

## Windows torchvision native extension (confirmed 2026-11-16)

The current `.venv` has `torch==2.12.0+cpu` and
`torchvision==0.27.0+cpu`. Importing torchvision raises
`RuntimeError: operator torchvision::nms does not exist`. Directly loading
`torchvision/_C.pyd` reveals the underlying error:
`[WinError 4551] An Application Control policy has blocked this file`.
Windows Code Integrity events 3033/3077 name this exact extension, alongside
a Smart App Control block event (3118). The failure also occurs outside the
agent sandbox.

Resolve the native-extension trust/signing block through the machine's
application-control administrator or use an approved environment where the
official torchvision extension is permitted to load. Do not register a dummy
NMS operator or suppress its registration error: real inference requires the
compiled operators. No Windows security settings were changed.

After resolving the block, run this from the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -c "import torch, torchvision; from torchvision.ops import nms; print(torch.__version__, torchvision.__version__); print(nms(torch.tensor([[0.,0.,10.,10.],[1.,1.,9.,9.]]), torch.tensor([0.9,0.8]), 0.5))"
```

Expected NMS output is `tensor([0])`. Then rerun Experiment B `inspect`.
The asset observations below describe the original implementation checkout;
rerun preflight to determine current asset availability.

| Required asset | Observed blocker | Recovery |
|---|---|---|
| `trained_models/best-imagenet-vid-yolo/best.pt` | Missing | Copy exact Stage 1 file or expose verified symlink at this path; expected SHA `5a35043701d4dadd30319e72246f2e6f349688642b75885bdb4a36b294f66027` |
| `trained_models/imagenet-vid-roissd-mobilenet-20261027/stage3/best_map.pt` | Missing | Copy exact Stage 1 file or verified symlink; expected SHA `1a830c339f060905f1042e10e98906be0cab43fa94cb3867bfc683ad14adc804` |
| `imagenet_vid_yolo_sampled` images, labels and YAML | No accessible dataset root found at the historical config paths | Set `SSD_DATASET_ROOT` and `SSD_DATASET_YAML` or pass `--dataset-root`; preserve relative image/label tree |
| Stage 3 `final_eval_manifest_candidate.txt` | Not present in this checkout | Optional: copy it and pass `--cohort-manifest` when preparing. Otherwise report the actual audit cohort and new test split, not the Stage 3 4,744-frame final cohort |
| Local implementation venv | Ultralytics 8.3.77 and no threadpoolctl; selected path requires 8.4.60 and threadpoolctl | Prepare the documented environment before real inference; no packages were replaced automatically |
| All three hosts | No remote execution access was provided | Run the documented preflight/pilot/calibration/evaluation on each host, transfer results, then aggregate the same cohort |

Run this after restoring assets:

```bash
python -m experiments.roi_merge_b inspect \
  --config experiments/roi_merge_b/config.example.yaml \
  --output outputs/merge_b/preflight
```

The command writes structured blockers rather than downloading alternate models
or inventing dimensions/timings. Until it passes, no real calibration, held-out
measurement, ImageNet workload digest, model latency comparison or scientific
numerical conclusion is claimed. Synthetic fixture results validate orchestration
and invariant checks only. A real portable manifest cannot be honestly generated
without reading and hashing the real images and labels.
