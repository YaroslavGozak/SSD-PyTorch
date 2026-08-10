import argparse
import csv
import json
import os
import random
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
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
    print(f"[VID-JOINT] {message}")


@dataclass
class FrameMeta:
    video_id: str
    frame_idx: int
    image_path: str
    ann_path: str
    rel_image_path: str
    rel_xml_path: str
    gt_count: int


@dataclass
class VideoSelection:
    video_id: str
    joint_score: float
    qualifying_classes: List[str]
    selected_frames: List[FrameMeta]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ImageNet VID subset where ROISSD and YOLO are both strong per class."
    )
    parser.add_argument("--roissd-config", required=True, help="Training config used for ROISSD inference.")
    parser.add_argument("--yolo-config", required=True, help="Training config used for YOLO inference.")
    parser.add_argument("--split", choices=("train", "test", "val"), default="test")
    parser.add_argument("--target-videos", type=int, default=70)
    parser.add_argument("--clip-length", type=int, default=300)
    parser.add_argument("--ap50-threshold", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--output-dir", default="benchmark_results/vid/imagenet-vid-joint-subset")
    parser.add_argument(
        "--metrics-cache-file",
        default=None,
        help="Optional path to write per-video AP50 and frame metadata cache JSON.",
    )
    parser.add_argument("--copy-images", action="store_true", help="Also copy selected images (disabled by default).")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


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


def _resolve_split_roots(config_path: str, split: str) -> Tuple[str, str]:
    cfg = load_config(config_path)
    dcfg = cfg["dataset_params"]
    if split == "train":
        return str(dcfg["train_data_root"]), str(dcfg["train_ann_root"])
    return str(dcfg["test_data_root"]), str(dcfg["test_ann_root"])


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


def _run_per_video_ap50(
    *,
    config_path: str,
    split: str,
    nms_iou: float,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, List[FrameMeta]], Dict[str, Any]]:
    if not torch.cuda.is_available():
        log("CUDA not available, running on CPU (this may be slow).")
        raise RuntimeError("CUDA is required for inference. Please run on a machine with a compatible GPU.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    args = argparse.Namespace(config_path=config_path)
    model, dataset, data_loader, train_cfg = load_model_and_dataset(device, args)
    conf_threshold = float(train_cfg["train_params"]["infer_conf_threshold"])

    data_root, ann_root = _resolve_split_roots(config_path, split)
    data_root = os.path.abspath(data_root)
    ann_root = os.path.abspath(ann_root)

    per_video_preds: Dict[str, List[Dict[str, List[List[float]]]]] = defaultdict(list)
    per_video_gts: Dict[str, List[Dict[str, List[List[float]]]]] = defaultdict(list)
    per_video_diffs: Dict[str, List[Dict[str, List[int]]]] = defaultdict(list)
    frame_meta_by_video: Dict[str, List[FrameMeta]] = defaultdict(list)
    fallback_frame_idx: Dict[str, int] = defaultdict(int)

    total_frames = len(data_loader)
    log(f"Evaluating {total_frames} frames for config: {config_path}")

    with torch.no_grad():
        for i, (im_tensor, target, fname) in enumerate(data_loader, start=1):
            target_dict = _to_target_dict(target)
            image_path = os.path.abspath(_to_str_path(fname))

            frame_bgr = cv2.imread(image_path)
            if frame_bgr is None:
                continue
            frame_h, frame_w = frame_bgr.shape[:2]

            video_id, frame_idx = _extract_sequence_meta(target_dict, fname)
            if frame_idx is None:
                frame_idx = fallback_frame_idx[video_id]
            fallback_frame_idx[video_id] = frame_idx + 1

            _, det_batch = run_model_inference(model, im_tensor.float().to(device))
            detections = tensor_to_detection_list(det_batch[0], dataset.idx2label, frame_w, frame_h)
            detections = merge_detections_nms(detections, iou_threshold=nms_iou)
            detections = [det for det in detections if float(det["confidence"]) >= conf_threshold]

            gt_dict, diff_dict = extract_gt_for_map(target_dict, dataset.idx2label, frame_w, frame_h)
            per_video_preds[video_id].append(_detections_to_map_pred(detections))
            per_video_gts[video_id].append(gt_dict)
            per_video_diffs[video_id].append(diff_dict)

            ann_path, rel_image, rel_xml = _ann_path_from_image(image_path, data_root, ann_root)
            gt_count = sum(len(v) for v in gt_dict.values())
            frame_meta_by_video[video_id].append(
                FrameMeta(
                    video_id=video_id,
                    frame_idx=frame_idx,
                    image_path=image_path,
                    ann_path=ann_path,
                    rel_image_path=rel_image,
                    rel_xml_path=rel_xml,
                    gt_count=gt_count,
                )
            )

            if i == 1 or i % 500 == 0 or i == total_frames:
                log(f"  processed {i}/{total_frames} frames")

    per_video_ap50: Dict[str, Dict[str, float]] = {}
    for video_id in sorted(per_video_preds.keys()):
        _, aps50, _, _ = compute_map(
            per_video_preds[video_id],
            per_video_gts[video_id],
            iou_threshold=0.5,
            difficult=per_video_diffs[video_id],
        )
        per_video_ap50[video_id] = {cls: float(ap) for cls, ap in aps50.items()}

    for video_id, frames in frame_meta_by_video.items():
        frame_meta_by_video[video_id] = sorted(frames, key=lambda f: f.frame_idx)

    run_info = {
        "config_path": config_path,
        "num_videos": len(per_video_ap50),
        "num_frames": sum(len(v) for v in frame_meta_by_video.values()),
    }
    return per_video_ap50, frame_meta_by_video, run_info


def _split_runs(frames: Sequence[FrameMeta]) -> List[List[FrameMeta]]:
    if not frames:
        return []
    runs: List[List[FrameMeta]] = []
    current: List[FrameMeta] = [frames[0]]
    for frame in frames[1:]:
        if frame.frame_idx == current[-1].frame_idx + 1:
            current.append(frame)
        else:
            runs.append(current)
            current = [frame]
    runs.append(current)
    return runs


def _choose_segment(frames: Sequence[FrameMeta], clip_length: int) -> List[FrameMeta]:
    runs = _split_runs(frames)
    if not runs:
        return []

    best_short_run = max(runs, key=lambda run: (len(run), sum(f.gt_count for f in run), -run[0].frame_idx))

    if len(best_short_run) <= clip_length:
        return list(best_short_run)

    best_window: Optional[List[FrameMeta]] = None
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


def _build_joint_ranking(
    roissd_ap50: Dict[str, Dict[str, float]],
    yolo_ap50: Dict[str, Dict[str, float]],
    frame_meta: Dict[str, List[FrameMeta]],
    *,
    ap50_threshold: float,
    target_videos: int,
    clip_length: int,
) -> Tuple[List[VideoSelection], List[str]]:
    all_video_ids = sorted(set(roissd_ap50.keys()) & set(yolo_ap50.keys()))
    dropped: List[str] = []
    candidates: List[Tuple[str, float, List[str]]] = []

    for video_id in all_video_ids:
        aps_a = roissd_ap50[video_id]
        aps_b = yolo_ap50[video_id]
        classes = sorted(set(aps_a.keys()) | set(aps_b.keys()))
        qualifying: List[str] = []
        mins: List[float] = []
        for cls in classes:
            a = aps_a.get(cls, float("nan"))
            b = aps_b.get(cls, float("nan"))
            if np.isnan(a) or np.isnan(b):
                continue
            if a >= ap50_threshold and b >= ap50_threshold:
                qualifying.append(cls)
                mins.append(float(min(a, b)))

        if not qualifying:
            dropped.append(video_id)
            continue

        joint_score = float(np.mean(mins))
        candidates.append((video_id, joint_score, qualifying))

    candidates.sort(key=lambda row: (-row[1], row[0]))
    selected = candidates[:target_videos]

    selections: List[VideoSelection] = []
    for video_id, score, qualifying in selected:
        frames = _choose_segment(frame_meta[video_id], clip_length)
        selections.append(
            VideoSelection(
                video_id=video_id,
                joint_score=score,
                qualifying_classes=qualifying,
                selected_frames=frames,
            )
        )
    return selections, dropped


def _update_xml_image_references(tree: ET.ElementTree, image_path: str) -> None:
    root = tree.getroot()

    filename_node = root.find("filename")
    if filename_node is None:
        filename_node = ET.SubElement(root, "filename")
    filename_node.text = os.path.basename(image_path)

    path_node = root.find("path")
    if path_node is None:
        path_node = ET.SubElement(root, "path")
    path_node.text = os.path.abspath(image_path)


def _write_outputs(
    *,
    output_dir: str,
    split: str,
    selections: Sequence[VideoSelection],
    dropped_videos: Sequence[str],
    roissd_info: Dict[str, Any],
    yolo_info: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    annotations_root = os.path.join(output_dir, "annotations", split)

    selected_videos_json = os.path.join(output_dir, "selected_videos.json")
    selected_videos_csv = os.path.join(output_dir, "selected_videos.csv")
    selected_frames_txt = os.path.join(output_dir, "selected_frames.txt")
    run_summary_json = os.path.join(output_dir, "run_summary.json")

    selected_video_payload = []
    selected_frame_lines: List[str] = []

    for item in selections:
        frames = item.selected_frames
        start_idx = frames[0].frame_idx if frames else -1
        end_idx = frames[-1].frame_idx if frames else -1
        selected_video_payload.append(
            {
                "video_id": item.video_id,
                "joint_score": item.joint_score,
                "qualifying_classes": item.qualifying_classes,
                "selected_frame_count": len(frames),
                "segment_start_frame_idx": start_idx,
                "segment_end_frame_idx": end_idx,
                "relative_xml_paths": [frame.rel_xml_path for frame in frames],
            }
        )
        selected_frame_lines.extend(frame.rel_xml_path for frame in frames)

    with open(selected_videos_json, "w", encoding="utf-8") as f:
        json.dump(selected_video_payload, f, indent=2)

    with open(selected_videos_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_id",
                "joint_score",
                "qualifying_classes",
                "selected_frame_count",
                "segment_start_frame_idx",
                "segment_end_frame_idx",
            ],
        )
        writer.writeheader()
        for row in selected_video_payload:
            writer.writerow(
                {
                    "video_id": row["video_id"],
                    "joint_score": f"{row['joint_score']:.6f}",
                    "qualifying_classes": "|".join(row["qualifying_classes"]),
                    "selected_frame_count": row["selected_frame_count"],
                    "segment_start_frame_idx": row["segment_start_frame_idx"],
                    "segment_end_frame_idx": row["segment_end_frame_idx"],
                }
            )

    with open(selected_frames_txt, "w", encoding="utf-8") as f:
        for rel_xml in selected_frame_lines:
            f.write(f"{rel_xml}\n")

    xml_written = 0
    images_copied = 0
    if not args.dry_run:
        for item in selections:
            for frame in item.selected_frames:
                dst_xml = os.path.join(annotations_root, frame.rel_xml_path)
                os.makedirs(os.path.dirname(dst_xml), exist_ok=True)

                tree = ET.parse(frame.ann_path)
                _update_xml_image_references(tree, frame.image_path)
                tree.write(dst_xml, encoding="utf-8")
                xml_written += 1

                if args.copy_images:
                    dst_img = os.path.join(output_dir, "images", split, frame.rel_image_path)
                    os.makedirs(os.path.dirname(dst_img), exist_ok=True)
                    shutil.copy2(frame.image_path, dst_img)
                    images_copied += 1

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": split,
        "target_videos": args.target_videos,
        "clip_length": args.clip_length,
        "ap50_threshold": args.ap50_threshold,
        "selected_video_count": len(selections),
        "selected_frame_count": int(sum(len(item.selected_frames) for item in selections)),
        "dropped_video_count": len(dropped_videos),
        "dropped_videos": list(dropped_videos),
        "xml_annotations_written": xml_written,
        "images_copied": images_copied,
        "copy_images_enabled": bool(args.copy_images),
        "dry_run": bool(args.dry_run),
        "roissd_run": roissd_info,
        "yolo_run": yolo_info,
    }
    with open(run_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return {
        "selected_videos_json": selected_videos_json,
        "selected_videos_csv": selected_videos_csv,
        "selected_frames_txt": selected_frames_txt,
        "run_summary_json": run_summary_json,
        "annotations_root": annotations_root,
    }


def _resolve_metrics_cache_path(args: argparse.Namespace, output_dir: str) -> str:
    if args.metrics_cache_file:
        cache_path = args.metrics_cache_file
    else:
        cache_path = os.path.join(output_dir, "per_video_metrics_cache.json")
    return os.path.abspath(cache_path)


def _write_metrics_cache(
    *,
    args: argparse.Namespace,
    output_dir: str,
    roissd_ap50: Dict[str, Dict[str, float]],
    yolo_ap50: Dict[str, Dict[str, float]],
    frame_meta: Dict[str, List[FrameMeta]],
) -> str:
    cache_path = _resolve_metrics_cache_path(args, output_dir)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    data_root, ann_root = _resolve_split_roots(args.roissd_config, args.split)
    all_video_ids = sorted(set(roissd_ap50.keys()) & set(yolo_ap50.keys()))
    videos_payload = []
    for video_id in all_video_ids:
        frames = frame_meta.get(video_id, [])
        videos_payload.append(
            {
                "video_id": video_id,
                "roissd_ap50": roissd_ap50.get(video_id, {}),
                "yolo_ap50": yolo_ap50.get(video_id, {}),
                "frame_count": len(frames),
                "frames": [
                    {
                        "frame_idx": int(frame.frame_idx),
                        "rel_xml_path": frame.rel_xml_path,
                        "gt_count": int(frame.gt_count),
                    }
                    for frame in frames
                ],
            }
        )

    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": args.split,
        "roissd_config": args.roissd_config,
        "yolo_config": args.yolo_config,
        "nms_iou": float(args.nms_iou),
        "ap50_threshold": float(args.ap50_threshold),
        "data_root": os.path.abspath(data_root),
        "ann_root": os.path.abspath(ann_root),
        "num_videos": len(videos_payload),
        "videos": videos_payload,
    }

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return cache_path


def build_joint_subset(args: argparse.Namespace) -> Dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = os.path.abspath(args.output_dir)
    if os.path.exists(output_dir):
        if args.overwrite:
            shutil.rmtree(output_dir)
        elif not args.dry_run:
            raise FileExistsError(f"Output directory exists: {output_dir}. Use --overwrite.")

    log("Running ROISSD per-video AP50 pass...")
    roissd_ap50, roissd_frames, roissd_info = _run_per_video_ap50(
        config_path=args.roissd_config,
        split=args.split,
        nms_iou=args.nms_iou,
    )

    log("Running YOLO per-video AP50 pass...")
    yolo_ap50, _, yolo_info = _run_per_video_ap50(
        config_path=args.yolo_config,
        split=args.split,
        nms_iou=args.nms_iou,
    )

    selections, dropped_videos = _build_joint_ranking(
        roissd_ap50,
        yolo_ap50,
        roissd_frames,
        ap50_threshold=args.ap50_threshold,
        target_videos=args.target_videos,
        clip_length=args.clip_length,
    )

    log(
        f"Selected {len(selections)} videos (target={args.target_videos}), "
        f"dropped {len(dropped_videos)} videos with no jointly-strong classes."
    )

    output_files = _write_outputs(
        output_dir=output_dir,
        split=args.split,
        selections=selections,
        dropped_videos=dropped_videos,
        roissd_info=roissd_info,
        yolo_info=yolo_info,
        args=args,
    )

    cache_path = _write_metrics_cache(
        args=args,
        output_dir=output_dir,
        roissd_ap50=roissd_ap50,
        yolo_ap50=yolo_ap50,
        frame_meta=roissd_frames,
    )
    log(f"Wrote metrics cache: {cache_path}")

    result = {
        "output_dir": output_dir,
        "selected_video_count": len(selections),
        "selected_frame_count": int(sum(len(item.selected_frames) for item in selections)),
        "dropped_video_count": len(dropped_videos),
        "metrics_cache_file": cache_path,
        **output_files,
    }
    return result


def main() -> None:
    args = parse_args()
    result = build_joint_subset(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
