import argparse
import json
import os
import shutil
from typing import List, Sequence, Set


def log(message: str) -> None:
    print(f"[XML-COPY] {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy ImageNet VID XML annotations listed in generated subset files."
    )
    parser.add_argument(
        "--subset-dir",
        default="benchmark_results/vid/imagenet-vid-joint-subset",
        help="Directory containing generated files (selected_frames.txt and/or selected_videos.json).",
    )
    parser.add_argument(
        "--source-ann-root",
        required=True,
        help="Root folder containing original XML annotations for the split.",
    )
    parser.add_argument(
        "--dest-ann-root",
        required=True,
        help="Destination root where selected XML files will be copied.",
    )
    parser.add_argument(
        "--selected-frames-file",
        default=None,
        help="Optional explicit path to selected_frames.txt.",
    )
    parser.add_argument(
        "--selected-videos-file",
        default=None,
        help="Optional explicit path to selected_videos.json.",
    )
    parser.add_argument(
        "--from-format",
        choices=("auto", "frames", "videos"),
        default="auto",
        help="Input format to use. 'auto' prefers selected_frames.txt if present.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting destination files if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would be copied, without writing files.",
    )
    return parser.parse_args()


def _normalize_rel_path(rel_path: str) -> str:
    rel = rel_path.strip().replace("\\", "/")
    rel = rel.lstrip("/")
    if not rel.lower().endswith(".xml"):
        raise ValueError(f"Expected XML relative path, got: {rel_path}")
    return rel


def _unique_preserve_order(items: Sequence[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def read_relative_paths_from_frames(selected_frames_file: str) -> List[str]:
    if not os.path.exists(selected_frames_file):
        raise FileNotFoundError(f"selected_frames file not found: {selected_frames_file}")
    rel_paths: List[str] = []
    with open(selected_frames_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rel_paths.append(_normalize_rel_path(line))
    return _unique_preserve_order(rel_paths)


def read_relative_paths_from_videos(selected_videos_file: str) -> List[str]:
    if not os.path.exists(selected_videos_file):
        raise FileNotFoundError(f"selected_videos file not found: {selected_videos_file}")
    with open(selected_videos_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("selected_videos.json must contain a list")

    rel_paths: List[str] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        xml_paths = row.get("relative_xml_paths", [])
        if not isinstance(xml_paths, list):
            continue
        for rel in xml_paths:
            rel_paths.append(_normalize_rel_path(str(rel)))
    return _unique_preserve_order(rel_paths)


def resolve_input_file_paths(args: argparse.Namespace) -> tuple[str, str]:
    subset_dir = os.path.abspath(args.subset_dir)
    selected_frames_file = (
        os.path.abspath(args.selected_frames_file)
        if args.selected_frames_file
        else os.path.join(subset_dir, "selected_frames.txt")
    )
    selected_videos_file = (
        os.path.abspath(args.selected_videos_file)
        if args.selected_videos_file
        else os.path.join(subset_dir, "selected_videos.json")
    )
    return selected_frames_file, selected_videos_file


def choose_rel_paths(args: argparse.Namespace, selected_frames_file: str, selected_videos_file: str) -> List[str]:
    if args.from_format == "frames":
        return read_relative_paths_from_frames(selected_frames_file)
    if args.from_format == "videos":
        return read_relative_paths_from_videos(selected_videos_file)

    if os.path.exists(selected_frames_file):
        log(f"Using selected_frames list: {selected_frames_file}")
        return read_relative_paths_from_frames(selected_frames_file)
    if os.path.exists(selected_videos_file):
        log(f"Using selected_videos list: {selected_videos_file}")
        return read_relative_paths_from_videos(selected_videos_file)

    raise FileNotFoundError(
        "Neither selected_frames.txt nor selected_videos.json was found. "
        "Provide explicit --selected-frames-file or --selected-videos-file."
    )


def copy_xml_files(
    rel_paths: Sequence[str],
    source_ann_root: str,
    dest_ann_root: str,
    overwrite: bool,
    dry_run: bool,
) -> dict:
    source_root = os.path.abspath(source_ann_root)
    dest_root = os.path.abspath(dest_ann_root)

    missing: List[str] = []
    copied = 0
    skipped_existing = 0

    for idx, rel in enumerate(rel_paths, start=1):
        src = os.path.join(source_root, rel)
        dst = os.path.join(dest_root, rel)

        if not os.path.exists(src):
            missing.append(rel)
            continue

        if os.path.exists(dst) and not overwrite:
            skipped_existing += 1
            continue

        if not dry_run:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
        copied += 1

        if idx == 1 or idx % 1000 == 0 or idx == len(rel_paths):
            log(f"Progress {idx}/{len(rel_paths)}")

    return {
        "source_ann_root": source_root,
        "dest_ann_root": dest_root,
        "requested": len(rel_paths),
        "copied": copied,
        "skipped_existing": skipped_existing,
        "missing": len(missing),
        "missing_examples": missing[:20],
        "dry_run": dry_run,
    }


def main() -> None:
    args = parse_args()
    selected_frames_file, selected_videos_file = resolve_input_file_paths(args)
    rel_paths = choose_rel_paths(args, selected_frames_file, selected_videos_file)

    if not rel_paths:
        raise RuntimeError("No XML paths found in the selected input file.")

    log(f"Found {len(rel_paths)} unique XML paths to process")
    summary = copy_xml_files(
        rel_paths=rel_paths,
        source_ann_root=args.source_ann_root,
        dest_ann_root=args.dest_ann_root,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
