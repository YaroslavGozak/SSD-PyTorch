"""Run from repository root: python -m experiments.roi_video_benchmark.run --config ..."""
import argparse
import copy
import json
import platform
import random
from collections import Counter
from pathlib import Path
from importlib.metadata import version

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from experiments.cost_model_validation.artifacts import load_calibration
from experiments.cost_model_validation.common import apply_runtime_controls, file_sha256, read_cpu_temp, read_cpu_freq
from tools.benchmarks.benchmark_framework_vid import VideoSequenceBenchmark
from tools.benchmarks.video_artifacts import write_json, hardware_snapshot
from tools.helpers.config_reader import load_config, _deep_merge
from tools.helpers.pipeline import load_dataset, load_model, resolve_device

POLICIES = ("full_frame", "roi_baseline", "linear_direct_cost", "shape_lookup_conservative")


def weights_path(train):
    p = train["train_params"]
    if p["model"] == "yolo":
        candidate = Path(p.get("yolo_weights") or p["ckpt_name"])
        if candidate.is_file():
            return candidate
    else:
        candidate = Path(p["ckpt_name"])
    return Path("trained_models") / p["task_name"] / candidate


def validate_calibration(artifact, train, device, runtime, weights_hash):
    """Fail closed on identities that make a saved latency model inapplicable."""
    provenance = artifact["provenance"]
    is_yolo = train["train_params"]["model"] == "yolo"
    if is_yolo and train["train_params"].get("yolo_use_predict_api", True):
        raise ValueError("Raw-forward YOLO calibration requires yolo_use_predict_api: false")
    checks = {
        "weights": (provenance["weights_sha256"], weights_hash),
        "device": (provenance["device"], str(device)),
        "precision": (provenance["dtype"], "float32"),
        "architecture": (provenance["architecture"], platform.machine()),
        "torch": (provenance["versions"]["torch"], torch.__version__),
        "stride": (artifact["shape_policy"]["stride"], 32),
        "backend": (provenance["backend"], "ultralytics" if is_yolo else "roissd"),
        "normalization": (provenance["preprocessing"]["normalization"], "divide_by_255" if is_yolo else "divide_by_255_then_imagenet"),
        "batch_size": (provenance["preprocessing"]["batch_size"], 1),
    }
    previous = provenance["runtime_environment"]
    for key in ("torch_num_threads", "torch_num_interop_threads", "process_affinity"):
        checks[key] = (previous[key], runtime[key])
    if is_yolo:
        checks["ultralytics"] = (provenance["versions"]["ultralytics"], version("ultralytics"))
    else:
        calibration_cfg = load_config(provenance["model_config"])
        checks["model_config_hash"] = (provenance["model_config_sha256"], file_sha256(provenance["model_config"]))
        checks["model_params"] = (calibration_cfg["model_params"], train["model_params"])
        checks["model"] = (calibration_cfg["train_params"]["model"], train["train_params"]["model"])
    mismatches = {k:values for k,values in checks.items() if values[0] != values[1]}
    if mismatches:
        raise ValueError(f"Calibration/runtime mismatch: {mismatches}")
    if train["dataset_params"].get("transform_name") not in {"ssd", "resize_longer_edge"}:
        raise ValueError("Use a non-GT resize transform with this experiment")
    return ["Calibration uses synthetic inputs; historical video parameter-selection overlap is unknown.",
            "Calibration measures detector forward; video also includes label remapping, tracker and external NMS.",
            "Video resize uses torchvision antialiasing; calibration resizes synthetic inputs with torch interpolate. Tensor shape and normalization match; content/preprocessing latency does not.",
            "Full-frame warmup does not prewarm every possible ROI shape."]


def policy_config(base, policy, directory):
    cfg = copy.deepcopy(base)
    params = cfg["benchmark_vid_params"]
    params["roi_merge"]["adaptive_tau"] = False
    params["output"].update(results_dir=str(directory), results_filename="legacy_metrics.csv")
    if policy == "full_frame":
        params["inference"]["key_frame_interval"] = 1
        params["roi_merge"]["strategy"] = "none"
    elif policy == "roi_baseline":
        params["roi_merge"]["strategy"] = "greedy"
    else:
        params["roi_merge"].update(strategy="calibrated", policies=dict(
            primary="linear_tau" if policy == "linear_direct_cost" else policy,
            fallback="separate", confidence_level=cfg["video_experiment"]["confidence_level"],
            decision_margin_s=cfg["video_experiment"]["decision_margin_s"]))
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-frames", type=int, help="Fixed prefix after configured video selection; 0 means all")
    parser.add_argument("--output")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--smoke-report", help="Prior comparison.json; print its coverage warnings before a full run")
    args = parser.parse_args()
    base = load_config(args.config)
    options = base["video_experiment"]
    if args.max_frames is not None:
        options["max_frames"] = args.max_frames
    if args.output:
        options["output_dir"] = args.output
    if base["benchmark_vid_params"]["tracker"]["type"] == "oracle_gt":
        raise ValueError("Oracle GT is prohibited in this experiment")
    if base["benchmark_vid_params"].get("tracker_input_dropout", {}).get("enabled"):
        raise ValueError("Disable tracker input dropout for the locked comparison")
    if options["repetitions"] < 1 or options["warmup"] < 1 or options["max_frames"] < 0:
        raise ValueError("Invalid repetitions, warmup or frame count")
    directory = Path(options["output_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "comparison.json").exists() or any(directory.glob("repeat_*")):
        raise FileExistsError(f"Use a new output directory: {directory}")
    runtime = apply_runtime_controls(base)
    train = _deep_merge(load_config(base["train_config_path"]), base.get("train_overrides", {}))
    device = resolve_device(base["benchmark_vid_params"]["device"])
    if str(device) != base["benchmark_vid_params"]["device"]:
        raise ValueError("Requested device unavailable; automatic fallback would invalidate calibration")
    weights = weights_path(train)
    weights_hash = file_sha256(str(weights))
    calibration_path = base["benchmark_vid_params"]["roi_merge"]["calibration"]
    calibration = load_calibration(calibration_path)
    warnings = validate_calibration(calibration, train, device, runtime, weights_hash)
    if args.smoke_report:
        report = json.loads(Path(args.smoke_report).read_text(encoding="utf-8"))
        for run in report["runs"]:
            for warning in run["warnings"]:
                print(f"PRIOR SMOKE ({run['policy']}): {warning}")
    elif options["max_frames"] == 0:
        print("WARNING: Full run has no --smoke-report; verify lookup coverage on a fixed fragment first.")
    dataset = load_dataset(train, split="test")
    infos = getattr(dataset, "images_info", None)
    if infos is None:
        raise ValueError("Dataset must expose its ordered images_info manifest")
    videos = options.get("videos", [])
    indices = [i for i,info in enumerate(infos) if not videos or str(info["video_id"]) in videos]
    if videos and set(videos)-{str(infos[i]["video_id"]) for i in indices}:
        raise ValueError("Configured video IDs are missing")
    available_frames = len(indices)
    if options["max_frames"]:
        indices = indices[:options["max_frames"]]
    if not indices:
        raise ValueError("Empty dataset selection")
    audit = dict(dataset=train["train_params"]["dataset"], loader_split="test", source_split="val",
                 dataset_params=train["dataset_params"], classes=dataset.classes,
                 total_retained_frames=len(infos), available_selected_frames=available_frames,
                 selected_frames=len(indices), videos=dict(Counter(str(i["video_id"]) for i in infos)),
                 dimensions=dict(Counter(f'{i["height"]}x{i["width"]}' for i in infos)),
                 selected_paths=[infos[i]["filename"] for i in indices], fps=None, version=None,
                 annotation_protocol="Existing loader retains only frames with parsed nonempty labels.")
    data_yaml = train["dataset_params"].get("yolo_dataset_yaml")
    if data_yaml:
        from dataset.yolo_imagenet_vid import _load_yolo_data_yaml, _resolve_split_roots, _discover_images
        image_root, label_root = _resolve_split_roots(_load_yolo_data_yaml(data_yaml), "test")
        image_paths = _discover_images(image_root)
        label_paths = [Path(label_root) / Path(p).relative_to(image_root).with_suffix(".txt") for p in image_paths]
        audit.update(discovered_images=len(image_paths), missing_labels=sum(not p.exists() for p in label_paths),
                     empty_labels=sum(p.exists() and not p.read_text().strip() for p in label_paths),
                     dataset_yaml_sha256=file_sha256(data_yaml), dataset_root=image_root)
    if train["train_params"].get("model_label_space") == "voc":
        warnings.append("VOC weights cover only mapped VID classes; evaluation retains all loader classes, so cross-detector AP comparisons are not supported.")
    write_json(directory / "dataset_audit.json", audit)
    write_json(directory / "preflight.json", dict(runtime=runtime, weights=str(weights), weights_sha256=weights_hash,
               calibration_sha256=file_sha256(calibration_path), warnings=warnings, config=base, train_config=train))
    print(f"Dataset: {len(infos)} retained frames, {len(audit['videos'])} videos; selected {len(indices)}")
    if args.audit_only:
        return
    loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False, num_workers=0)
    model = load_model(train, dataset, model_device=device).to(device).eval()
    results = []
    for repetition in range(options["repetitions"]):
        for policy in POLICIES:
            seed = int(options["seed"])
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            cfg = policy_config(base, policy, directory / f"repeat_{repetition}" / policy)
            bench = VideoSequenceBenchmark.from_config_dict(cfg, extra_run_metadata=dict(
                policy=policy, repetition=repetition, seed=seed, runtime=runtime,
                weights_sha256=weights_hash, warnings=warnings,
                calibration_sha256=file_sha256(calibration_path), precision="float32", batch_size=1,
                versions={p:version(p) for p in ("torch","torchvision","numpy") + (("ultralytics",) if train["train_params"]["model"] == "yolo" else ())},
                backend="ultralytics_raw" if train["train_params"]["model"] == "yolo" else "pytorch_roissd",
                cpu_temp_c_before=read_cpu_temp(), cpu_freq_mhz_before=read_cpu_freq(),
                hardware_before=hardware_snapshot(),
                cooling_mode=options.get("cooling_mode")))
            summary = bench.run(prepared=(model, dataset, loader, train))["artifacts"]
            results.append(dict(policy=policy, repetition=repetition,
                                manifest=summary["dataset_manifest_sha256"], weights=weights_hash,
                                total_time_s=summary["total_time_s"], warnings=summary["warnings"]))
            for warning in summary["warnings"]:
                print("WARNING:", warning)
    if len({r["manifest"] for r in results}) != 1:
        raise RuntimeError("Policy frame/annotation manifests differ")
    estimate = sum(r["total_time_s"] for r in results) * available_frames / len(indices)
    write_json(directory / "comparison.json", dict(runs=results, same_frames_annotations_weights=True,
               estimated_full_processing_seconds=estimate,
               estimate_caveat="Rough scaling from selected frames; excludes model/data loading and evaluation; short prefixes do not predict whole-video ROI trajectories."))
    print(f"Saved {directory / 'comparison.json'}; rough full processing estimate {estimate/60:.1f} min")


if __name__ == "__main__":
    main()
