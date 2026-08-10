import argparse
import csv
import json
import os
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import cv2
import torch

from tools.helpers.config_reader import load_config
from tools.helpers.pipeline import (
    extract_gt_for_map,
    load_model_and_dataset,
    merge_detections_nms,
    run_model_inference,
    tensor_to_detection_list,
)
from tools.infer import compute_map


def log(message: str) -> None:
    print(f"[VID-APPEND] {message}")


@dataclass
class FrameLite:
    frame_idx: int
    rel_xml_path: str
    gt_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append class-specific videos to an existing ImageNet VID subset via live annotation traversal and on-the-fly AP50 checks."
    )
    parser.add_argument("--subset-dir", default="benchmark_results/vid/imagenet-vid-joint-subset")
    parser.add_argument("--roissd-config", required=True)
    parser.add_argument("--yolo-config", required=True)
    parser.add_argument("--source-data-root", required=True, help="Image root (split root or parent root).")
    parser.add_argument("--source-ann-root", required=True, help="Annotation root (split root or parent root).")
    parser.add_argument("--split", choices=("train", "test", "val"), default="test")
    parser.add_argument("--required-class", required=True, help="Class name that appended videos must contain.")
    parser.add_argument("--ap50-threshold", type=float, default=0.60)
    parser.add_argument("--max-additional-videos", type=int, default=10)
    parser.add_argument("--max-additional-frames", type=int, default=2000)
    parser.add_argument("--clip-length", type=int, default=300)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _normalize_rel_xml(rel_xml: str) -> str:
    rel = rel_xml.strip().replace("\\", "/").lstrip("/")
    if not rel.lower().endswith(".xml"):
        raise ValueError(f"Expected XML path, got: {rel_xml}")
    return rel


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _resolve_split_root(root: str, split: str) -> str:
    normalized = os.path.normpath(root)
    if os.path.basename(normalized).lower() == split.lower():
        return normalized
    return os.path.join(normalized, split)


def _safe_relpath(path: str, root: str) -> str:
    rel = os.path.relpath(path, root)
    if rel.startswith(".."):
        raise ValueError(f"Path {path} is outside root {root}")
    return rel


def _ann_path_from_image(image_path: str, data_root: str, ann_root: str) -> Tuple[str, str, str]:
    rel_image = _safe_relpath(image_path, data_root)
    rel_stem, _ = os.path.splitext(rel_image)
    rel_xml = f"{rel_stem}.xml"
    ann_path = os.path.join(ann_root, rel_xml)
    return ann_path, rel_image.replace("\\", "/"), rel_xml.replace("\\", "/")


def _to_str_path(fname: Any) -> str:
    return str(fname[0] if isinstance(fname, (list, tuple)) else fname)


def _to_target_dict(target: Any) -> Dict[str, Any]:
    return target[0] if isinstance(target, list) else target


def _extract_sequence_meta(target: Dict[str, Any], fname: Any) -> Tuple[str, Optional[int]]:
    default_path = _to_str_path(fname)
    default_video_id = os.path.basename(os.path.dirname(default_path))

    video_id = target.get("video_id", default_video_id)
    if isinstance(video_id, list):
        video_id = video_id[0] if video_id else default_video_id

    frame_idx = target.get("frame_idx", None)
    if isinstance(frame_idx, list):
        frame_idx = frame_idx[0] if frame_idx else None
    if isinstance(frame_idx, torch.Tensor):
        frame_idx = int(frame_idx.item())
    if frame_idx is not None:
        frame_idx = int(frame_idx)

    return str(video_id), frame_idx


def _detections_to_map_pred(detections: List[Dict[str, Any]]) -> Dict[str, List[List[float]]]:
    pred: Dict[str, List[List[float]]] = {}
    for det in detections:
        cls = str(det["class"])
        x1, y1, x2, y2 = det["bbox"]
        pred.setdefault(cls, []).append([float(x1), float(y1), float(x2), float(y2), float(det["confidence"])])
    return pred


def _frame_idx_from_name(frame_stem: str, fallback: int) -> int:
    return int(frame_stem) if frame_stem.isdigit() else fallback


def _scan_videos_for_required_class(
    ann_split_root: str,
    required_class: str,
) -> Tuple[Dict[str, List[FrameLite]], int, int]:
    frame_map: Dict[str, List[FrameLite]] = defaultdict(list)
    video_has_required: Dict[str, bool] = defaultdict(bool)
    malformed_xml = 0
    scanned_xml = 0

    for dirpath, _, filenames in os.walk(ann_split_root):
        xml_files = sorted(name for name in filenames if name.lower().endswith(".xml"))
        if not xml_files:
            continue

        rel_video_dir = _safe_relpath(dirpath, ann_split_root).replace("\\", "/")
        if rel_video_dir in (".", ""):
            continue

        for fallback_idx, xml_name in enumerate(xml_files):
            xml_path = os.path.join(dirpath, xml_name)
            scanned_xml += 1
            try:
                root = ET.parse(xml_path).getroot()
            except Exception:
                malformed_xml += 1
                continue

            objects = list(root.findall("object"))
            gt_count = len(objects)
            if gt_count == 0:
                continue

            has_required = False
            for obj in objects:
                label = obj.find("class")
                if label is None:
                    label = obj.find("name")
                if label is not None and (label.text or "").strip() == required_class:
                    has_required = True
                    break

            frame_stem = os.path.splitext(xml_name)[0]
            rel_xml = _normalize_rel_xml(f"{rel_video_dir}/{xml_name}")
            frame_map[rel_video_dir].append(
                FrameLite(
                    frame_idx=_frame_idx_from_name(frame_stem, fallback_idx),
                    rel_xml_path=rel_xml,
                    gt_count=gt_count,
                )
            )
            if has_required:
                video_has_required[rel_video_dir] = True

    filtered: Dict[str, List[FrameLite]] = {}
    for video_id, frames in frame_map.items():
        if not video_has_required.get(video_id, False):
            continue
        filtered[video_id] = sorted(frames, key=lambda f: f.frame_idx)

    return filtered, scanned_xml, malformed_xml


def _split_runs(frames: Sequence[FrameLite]) -> List[List[FrameLite]]:
    if not frames:
        return []
    ordered = sorted(frames, key=lambda f: f.frame_idx)
    runs: List[List[FrameLite]] = []
    current: List[FrameLite] = [ordered[0]]
    for frame in ordered[1:]:
        if frame.frame_idx == current[-1].frame_idx + 1:
            current.append(frame)
        else:
            runs.append(current)
            current = [frame]
    runs.append(current)
    return runs


def _choose_segment(frames: Sequence[FrameLite], clip_length: int) -> List[FrameLite]:
    runs = _split_runs(frames)
    if not runs:
        return []

    best_short_run = max(runs, key=lambda run: (len(run), sum(f.gt_count for f in run), -run[0].frame_idx))
    if len(best_short_run) <= clip_length:
        return list(best_short_run)

    best_window: Optional[List[FrameLite]] = None
    best_score: Optional[Tuple[int, int]] = None
    for run in runs:
        if len(run) < clip_length:
            continue
        window_sum = sum(frame.gt_count for frame in run[:clip_length])
        score = (window_sum, -run[0].frame_idx)
        if best_score is None or score > best_score:
            best_score = score
            best_window = list(run[:clip_length])
        for start in range(1, len(run) - clip_length + 1):
            window_sum += run[start + clip_length - 1].gt_count - run[start - 1].gt_count
            score = (window_sum, -run[start].frame_idx)
            if best_score is None or score > best_score:
                best_score = score
                best_window = list(run[start:start + clip_length])

    if best_window is not None:
        return best_window
    return list(best_short_run)


def _resolve_paths(args: argparse.Namespace) -> Dict[str, str]:
    subset_dir = os.path.abspath(args.subset_dir)
    return {
        "subset_dir": subset_dir,
        "selected_videos": os.path.join(subset_dir, "selected_videos.json"),
        "selected_frames": os.path.join(subset_dir, "selected_frames.txt"),
        "run_summary": os.path.join(subset_dir, "run_summary.json"),
        "selected_videos_csv": os.path.join(subset_dir, "selected_videos.csv"),
    }


def _load_existing_subset(selected_videos_path: str, selected_frames_path: str, run_summary_path: str) -> Dict[str, Any]:
    selected_videos = _read_json(selected_videos_path) if os.path.exists(selected_videos_path) else []
    run_summary = _read_json(run_summary_path) if os.path.exists(run_summary_path) else {}
    frame_lines: List[str] = []
    if os.path.exists(selected_frames_path):
        with open(selected_frames_path, "r", encoding="utf-8") as f:
            frame_lines = [_normalize_rel_xml(line) for line in f if line.strip()]

    selected_video_ids = {str(row.get("video_id", "")) for row in selected_videos if isinstance(row, dict)}
    selected_frame_set = set(frame_lines)
    return {
        "selected_videos": selected_videos,
        "selected_video_ids": selected_video_ids,
        "selected_frames": frame_lines,
        "selected_frame_set": selected_frame_set,
        "run_summary": run_summary,
    }


def _resolve_model_split_roots(config_path: str, split: str) -> Tuple[str, str]:
    cfg = load_config(config_path)
    dcfg = cfg["dataset_params"]
    if split == "train":
        return str(dcfg["train_data_root"]), str(dcfg["train_ann_root"])
    return str(dcfg["test_data_root"]), str(dcfg["test_ann_root"])


def _evaluate_required_class_ap50(
    *,
    config_path: str,
    split: str,
    required_class: str,
    nms_iou: float,
    rel_xml_to_video: Dict[str, str],
) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        log("CUDA not available, running on CPU (this can be very slow).")

    args = argparse.Namespace(config_path=config_path)
    model, dataset, data_loader, train_cfg = load_model_and_dataset(device, args)
    conf_threshold = float(train_cfg["train_params"]["infer_conf_threshold"])

    model_data_root, model_ann_root = _resolve_model_split_roots(config_path, split)
    model_data_root = os.path.abspath(model_data_root)
    model_ann_root = os.path.abspath(model_ann_root)

    per_video_preds: Dict[str, List[Dict[str, List[List[float]]]]] = defaultdict(list)
    per_video_gts: Dict[str, List[Dict[str, List[List[float]]]]] = defaultdict(list)
    per_video_diffs: Dict[str, List[Dict[str, List[int]]]] = defaultdict(list)

    total_frames = len(data_loader)
    processed = 0
    with torch.no_grad():
        for idx, (im_tensor, target, fname) in enumerate(data_loader, start=1):
            target_dict = _to_target_dict(target)
            image_path = os.path.abspath(_to_str_path(fname))
            _, _, rel_xml = _ann_path_from_image(image_path, model_data_root, model_ann_root)
            if rel_xml not in rel_xml_to_video:
                continue

            frame_bgr = cv2.imread(image_path)
            if frame_bgr is None:
                continue
            frame_h, frame_w = frame_bgr.shape[:2]

            video_id = rel_xml_to_video[rel_xml]

            _, det_batch = run_model_inference(model, im_tensor.float().to(device))
            detections = tensor_to_detection_list(det_batch[0], dataset.idx2label, frame_w, frame_h)
            detections = merge_detections_nms(detections, iou_threshold=nms_iou)
            detections = [det for det in detections if float(det["confidence"]) >= conf_threshold]

            gt_dict, diff_dict = extract_gt_for_map(target_dict, dataset.idx2label, frame_w, frame_h)
            per_video_preds[video_id].append(_detections_to_map_pred(detections))
            per_video_gts[video_id].append(gt_dict)
            per_video_diffs[video_id].append(diff_dict)

            processed += 1
            if processed == 1 or processed % 500 == 0:
                log(f"  evaluated {processed} matching frames ({idx}/{total_frames} scanned)")

    per_video_required_ap50: Dict[str, float] = {}
    for video_id in sorted(per_video_preds.keys()):
        _, aps50, _, _ = compute_map(
            per_video_preds[video_id],
            per_video_gts[video_id],
            iou_threshold=0.5,
            difficult=per_video_diffs[video_id],
        )
        per_video_required_ap50[video_id] = float(aps50.get(required_class, float("nan")))

    return per_video_required_ap50


def _copy_xml_for_selection(
    *,
    selections: Sequence[Dict[str, Any]],
    source_ann_root: str,
    subset_ann_root: str,
    overwrite: bool,
    dry_run: bool,
) -> Tuple[int, int, int, List[str]]:
    copied = 0
    skipped_existing = 0
    missing = 0
    missing_examples: List[str] = []

    for selection in selections:
        for frame in selection["segment"]:
            rel_xml = frame.rel_xml_path
            src = os.path.join(source_ann_root, rel_xml)
            dst = os.path.join(subset_ann_root, rel_xml)
            if not os.path.exists(src):
                missing += 1
                if len(missing_examples) < 20:
                    missing_examples.append(rel_xml)
                continue
            if os.path.exists(dst) and not overwrite:
                skipped_existing += 1
                continue

            if not dry_run:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
            copied += 1

    return copied, skipped_existing, missing, missing_examples


def _append_selected_videos(
    existing_rows: List[Dict[str, Any]],
    selections: Sequence[Dict[str, Any]],
    required_class: str,
) -> List[Dict[str, Any]]:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    new_rows: List[Dict[str, Any]] = []
    for selection in selections:
        segment: List[FrameLite] = selection["segment"]
        new_rows.append(
            {
                "video_id": selection["video_id"],
                "joint_score": selection["joint_score"],
                "qualifying_classes": [required_class],
                "selected_frame_count": len(segment),
                "segment_start_frame_idx": segment[0].frame_idx,
                "segment_end_frame_idx": segment[-1].frame_idx,
                "relative_xml_paths": [frame.rel_xml_path for frame in segment],
                "append_info": {
                    "timestamp_utc": now,
                    "required_class": required_class,
                    "roissd_ap50": selection["roissd_ap50"],
                    "yolo_ap50": selection["yolo_ap50"],
                },
            }
        )
    return existing_rows + new_rows


def _append_selected_videos_csv(csv_path: str, selections: Sequence[Dict[str, Any]], required_class: str, dry_run: bool) -> None:
    if not selections:
        return
    fieldnames = [
        "video_id",
        "joint_score",
        "qualifying_classes",
        "selected_frame_count",
        "segment_start_frame_idx",
        "segment_end_frame_idx",
    ]

    if dry_run:
        return

    need_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if need_header:
            writer.writeheader()
        for selection in selections:
            segment: List[FrameLite] = selection["segment"]
            writer.writerow(
                {
                    "video_id": selection["video_id"],
                    "joint_score": f"{selection['joint_score']:.6f}",
                    "qualifying_classes": required_class,
                    "selected_frame_count": len(segment),
                    "segment_start_frame_idx": segment[0].frame_idx,
                    "segment_end_frame_idx": segment[-1].frame_idx,
                }
            )


def _append_selected_frames(existing_lines: List[str], selections: Sequence[Dict[str, Any]]) -> List[str]:
    seen = set(existing_lines)
    out = list(existing_lines)
    for selection in selections:
        for frame in selection["segment"]:
            rel_xml = frame.rel_xml_path
            if rel_xml in seen:
                continue
            seen.add(rel_xml)
            out.append(rel_xml)
    return out


def _update_run_summary(
    run_summary: Dict[str, Any],
    *,
    required_class: str,
    ap50_threshold: float,
    selected_videos_added: int,
    selected_frames_total: int,
    copied: int,
    skipped_existing: int,
    missing: int,
    malformed_xml: int,
    scanned_xml: int,
) -> Dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    summary = dict(run_summary)
    summary["selected_video_count"] = int(summary.get("selected_video_count", 0)) + int(selected_videos_added)
    summary["selected_frame_count"] = int(selected_frames_total)

    history = summary.get("append_history", [])
    if not isinstance(history, list):
        history = []
    history.append(
        {
            "timestamp_utc": now,
            "required_class": required_class,
            "ap50_threshold": ap50_threshold,
            "added_videos": selected_videos_added,
            "copied_xml": copied,
            "skipped_existing_xml": skipped_existing,
            "missing_xml": missing,
            "scanned_xml": scanned_xml,
            "malformed_xml": malformed_xml,
        }
    )
    summary["append_history"] = history
    summary["last_append_utc"] = now
    return summary


def run_append(args: argparse.Namespace) -> Dict[str, Any]:
    paths = _resolve_paths(args)

    source_data_split_root = os.path.abspath(_resolve_split_root(args.source_data_root, args.split))
    source_ann_split_root = os.path.abspath(_resolve_split_root(args.source_ann_root, args.split))

    if not os.path.isdir(source_ann_split_root):
        raise FileNotFoundError(f"Annotation split root not found: {source_ann_split_root}")

    existing = _load_existing_subset(paths["selected_videos"], paths["selected_frames"], paths["run_summary"])

    log(f"Scanning annotations for class '{args.required_class}' under {source_ann_split_root}")
    class_videos, scanned_xml, malformed_xml = _scan_videos_for_required_class(
        source_ann_split_root,
        args.required_class,
    )

    candidate_segments: Dict[str, List[FrameLite]] = {}
    for video_id, frames in class_videos.items():
        if video_id in existing["selected_video_ids"]:
            continue
        segment = _choose_segment(frames, clip_length=int(args.clip_length))
        if segment:
            candidate_segments[video_id] = segment

    if not candidate_segments:
        return {
            "subset_dir": paths["subset_dir"],
            "split": args.split,
            "required_class": args.required_class,
            "candidate_videos_found": 0,
            "videos_added": 0,
            "selected_frames_total": len(existing["selected_frames"]),
            "dry_run": bool(args.dry_run),
            "message": "No new candidate videos contain the required class.",
        }

    rel_xml_to_video: Dict[str, str] = {}
    for video_id, segment in candidate_segments.items():
        for frame in segment:
            rel_xml_to_video[frame.rel_xml_path] = video_id

    log(f"Evaluating ROISSD AP50 on {len(candidate_segments)} candidate videos")
    roissd_ap50 = _evaluate_required_class_ap50(
        config_path=args.roissd_config,
        split=args.split,
        required_class=args.required_class,
        nms_iou=float(args.nms_iou),
        rel_xml_to_video=rel_xml_to_video,
    )

    log(f"Evaluating YOLO AP50 on {len(candidate_segments)} candidate videos")
    yolo_ap50 = _evaluate_required_class_ap50(
        config_path=args.yolo_config,
        split=args.split,
        required_class=args.required_class,
        nms_iou=float(args.nms_iou),
        rel_xml_to_video=rel_xml_to_video,
    )

    accepted: List[Dict[str, Any]] = []
    for video_id, segment in candidate_segments.items():
        ra = float(roissd_ap50.get(video_id, float("nan")))
        ya = float(yolo_ap50.get(video_id, float("nan")))
        if ra < float(args.ap50_threshold) or ya < float(args.ap50_threshold):
            continue
        accepted.append(
            {
                "video_id": video_id,
                "segment": segment,
                "roissd_ap50": ra,
                "yolo_ap50": ya,
                "joint_score": float(min(ra, ya)),
            }
        )

    accepted.sort(key=lambda row: (-row["joint_score"], row["video_id"]))

    final_selection: List[Dict[str, Any]] = []
    added_frames = 0
    max_videos = max(0, int(args.max_additional_videos))
    max_frames = max(0, int(args.max_additional_frames))
    for row in accepted:
        if len(final_selection) >= max_videos:
            break
        segment_len = len(row["segment"])
        if max_frames and (added_frames + segment_len > max_frames):
            continue
        final_selection.append(row)
        added_frames += segment_len

    split = str(existing["run_summary"].get("split", args.split))
    subset_ann_root = os.path.join(paths["subset_dir"], "annotations", split)
    copied, skipped_existing, missing, missing_examples = _copy_xml_for_selection(
        selections=final_selection,
        source_ann_root=source_ann_split_root,
        subset_ann_root=subset_ann_root,
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
    )

    new_selected_videos = _append_selected_videos(existing["selected_videos"], final_selection, args.required_class)
    new_selected_frames = _append_selected_frames(existing["selected_frames"], final_selection)
    new_run_summary = _update_run_summary(
        existing["run_summary"],
        required_class=args.required_class,
        ap50_threshold=float(args.ap50_threshold),
        selected_videos_added=len(final_selection),
        selected_frames_total=len(new_selected_frames),
        copied=copied,
        skipped_existing=skipped_existing,
        missing=missing,
        malformed_xml=malformed_xml,
        scanned_xml=scanned_xml,
    )

    if not args.dry_run:
        os.makedirs(paths["subset_dir"], exist_ok=True)
        _write_json(paths["selected_videos"], new_selected_videos)
        with open(paths["selected_frames"], "w", encoding="utf-8") as f:
            for rel_xml in new_selected_frames:
                f.write(f"{rel_xml}\n")
        _write_json(paths["run_summary"], new_run_summary)
        _append_selected_videos_csv(paths["selected_videos_csv"], final_selection, args.required_class, dry_run=False)

    return {
        "subset_dir": paths["subset_dir"],
        "split": split,
        "required_class": args.required_class,
        "ap50_threshold": float(args.ap50_threshold),
        "max_additional_videos": max_videos,
        "max_additional_frames": max_frames,
        "candidate_videos_found": len(candidate_segments),
        "videos_above_threshold": len(accepted),
        "videos_added": len(final_selection),
        "selected_frames_total": len(new_selected_frames),
        "xml_copied": copied,
        "xml_skipped_existing": skipped_existing,
        "xml_missing": missing,
        "xml_missing_examples": missing_examples,
        "scanned_xml": scanned_xml,
        "malformed_xml": malformed_xml,
        "dry_run": bool(args.dry_run),
        "source_data_split_root": source_data_split_root,
        "source_ann_split_root": source_ann_split_root,
    }


def main() -> None:
    args = parse_args()
    result = run_append(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
