import argparse
import csv
import json
import math
import os
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from dataset.helpers.label_spaces import IMAGENET_VID_VOC_OVERLAP_CLASSES


SMALL_AREA_THRESHOLD = 0.02
MEDIUM_AREA_THRESHOLD = 0.15
DENSITY_LABELS = ('single', 'sparse', 'dense')
VIDEO_LOG_INTERVAL = 25
WRITE_LOG_INTERVAL = 500


def log(message: str) -> None:
    print(f'[VOC10K] {message}')


@dataclass
class DetectionRecord:
    class_name: str
    bbox: Tuple[int, int, int, int]

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


@dataclass
class FrameRecord:
    video_id: str
    rel_video_dir: str
    frame_name: str
    original_frame_idx: int
    width: int
    height: int
    source_xml_path: str
    rel_xml_path: str
    source_image_path: str
    rel_image_path: str
    detections: List[DetectionRecord]

    @property
    def object_count(self) -> int:
        return len(self.detections)

    @property
    def normalized_bbox_areas(self) -> List[float]:
        frame_area = max(float(self.width * self.height), 1.0)
        areas = []
        for det in self.detections:
            x1, y1, x2, y2 = det.bbox
            areas.append(max(0.0, float((x2 - x1) * (y2 - y1))) / frame_area)
        return areas

    @property
    def class_histogram(self) -> Counter:
        return Counter(det.class_name for det in self.detections)


@dataclass
class VideoRecord:
    video_id: str
    rel_video_dir: str
    frames: List[FrameRecord] = field(default_factory=list)


@dataclass
class ClipCandidate:
    video_id: str
    rel_video_dir: str
    start_frame_idx: int
    end_frame_idx: int
    frames: List[FrameRecord]
    class_histogram: Counter
    size_bin_counts: np.ndarray
    density_bin_counts: np.ndarray
    motion_values: List[float]
    mean_motion: float
    motion_variance: float
    mean_objects_per_frame: float
    dominant_class: Optional[str]
    feature_vector: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build a representative VOC-compatible ImageNet-VID annotation subset.')
    parser.add_argument('--data-root', required=True, help='Path to ImageNet-VID Data/VID root or split root.')
    parser.add_argument('--ann-root', required=True, help='Path to ImageNet-VID Annotations/VID root or split root.')
    parser.add_argument('--split', choices=('train', 'val'), required=True, help='Dataset split to process.')
    parser.add_argument('--clip-length', type=int, default=100, help='Number of consecutive frames per selected clip.')
    parser.add_argument('--clip-stride', type=int, default=100, help='Stride between candidate clip starts inside a valid run.')
    parser.add_argument('--target-clips', type=int, default=200, help='Target number of representative clips to select.')
    parser.add_argument('--max-clips-per-video', type=int, default=3, help='Maximum number of selected clips from any source video.')
    parser.add_argument('--output-folder-name', default='VOC10KAnnotations', help='Sibling annotation folder name to create.')
    parser.add_argument('--manifest-name', default='manifest.json', help='Manifest filename written under the output folder.')
    parser.add_argument('--csv-name', default='selected_clips.csv', help='CSV summary filename written under the output folder.')
    parser.add_argument('--seed', type=int, default=13, help='Random seed used for deterministic tie-breaking.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite an existing output folder.')
    parser.add_argument('--dry-run', action='store_true', help='Only compute and report the subset without writing XML files.')
    return parser.parse_args()


def resolve_split_root(root: str, split: str) -> str:
    normalized_root = os.path.normpath(root)
    if os.path.basename(normalized_root).lower() == split.lower():
        return normalized_root
    return os.path.join(normalized_root, split)


def resolve_output_split_root(annotation_root: str, split: str, output_folder_name: str) -> str:
    source_split_root = resolve_split_root(annotation_root, split)
    annotations_root = os.path.dirname(source_split_root)
    return os.path.join(annotations_root, output_folder_name, split)


def _read_object_label(obj: ET.Element) -> Optional[str]:
    label_node = obj.find('class')
    if label_node is None:
        label_node = obj.find('name')
    if label_node is None or label_node.text is None:
        return None
    return label_node.text.strip()


def prune_xml_to_voc_overlap(
    source_xml_path: str,
    destination_xml_path: str,
    allowed_class_names: Optional[Sequence[str]] = None,
) -> int:
    allowed = set(allowed_class_names or IMAGENET_VID_VOC_OVERLAP_CLASSES)
    tree = ET.parse(source_xml_path)
    root = tree.getroot()

    kept_count = 0
    for obj in list(root.findall('object')):
        label_name = _read_object_label(obj)
        if label_name not in allowed:
            root.remove(obj)
            continue
        kept_count += 1

    if kept_count == 0:
        raise ValueError(f'No VOC-overlap objects remain in {source_xml_path}')

    os.makedirs(os.path.dirname(destination_xml_path), exist_ok=True)
    tree.write(destination_xml_path, encoding='utf-8')
    return kept_count


def _has_group_folders(split_root: str) -> bool:
    if not os.path.isdir(split_root):
        return False
    root_entries = set(os.listdir(split_root))
    return any(entry in root_entries for entry in ['a', 'b', 'c', 'd', 'e'])


def iter_video_dirs(data_split_root: str, ann_split_root: str) -> Iterable[Tuple[str, str, str, str]]:
    grouped = _has_group_folders(data_split_root) or _has_group_folders(ann_split_root)
    if grouped:
        for group_name in ['a', 'b', 'c', 'd', 'e']:
            data_group_root = os.path.join(data_split_root, group_name)
            ann_group_root = os.path.join(ann_split_root, group_name)
            if not os.path.isdir(data_group_root) or not os.path.isdir(ann_group_root):
                continue

            video_names = sorted(
                video_name
                for video_name in os.listdir(data_group_root)
                if os.path.isdir(os.path.join(data_group_root, video_name))
                and os.path.isdir(os.path.join(ann_group_root, video_name))
            )
            for video_name in video_names:
                rel_video_dir = os.path.join(group_name, video_name)
                yield (
                    f'{group_name}/{video_name}',
                    rel_video_dir,
                    os.path.join(data_group_root, video_name),
                    os.path.join(ann_group_root, video_name),
                )
        return

    video_names = sorted(
        video_name
        for video_name in os.listdir(data_split_root)
        if os.path.isdir(os.path.join(data_split_root, video_name))
        and os.path.isdir(os.path.join(ann_split_root, video_name))
    )
    for video_name in video_names:
        yield (
            video_name,
            video_name,
            os.path.join(data_split_root, video_name),
            os.path.join(ann_split_root, video_name),
        )


def parse_voc_compatible_objects(xml_path: str, allowed_class_names: Sequence[str]) -> Tuple[int, int, List[DetectionRecord]]:
    allowed = set(allowed_class_names)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    if size is None:
        raise ValueError(f'Missing size node in {xml_path}')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    detections: List[DetectionRecord] = []
    for obj in root.findall('object'):
        label_name = _read_object_label(obj)
        if label_name not in allowed:
            continue

        bbox_info = obj.find('bndbox')
        if bbox_info is None:
            continue
        bbox = (
            int(float(bbox_info.find('xmin').text)) - 1,
            int(float(bbox_info.find('ymin').text)) - 1,
            int(float(bbox_info.find('xmax').text)) - 1,
            int(float(bbox_info.find('ymax').text)) - 1,
        )
        detections.append(DetectionRecord(class_name=label_name, bbox=bbox))

    return width, height, detections


def scan_split_videos(
    data_root: str,
    ann_root: str,
    split: str,
    allowed_class_names: Optional[Sequence[str]] = None,
) -> List[VideoRecord]:
    allowed = set(allowed_class_names or IMAGENET_VID_VOC_OVERLAP_CLASSES)
    data_split_root = resolve_split_root(data_root, split)
    ann_split_root = resolve_split_root(ann_root, split)

    if not os.path.isdir(data_split_root):
        raise FileNotFoundError(f'Data split root not found: {data_split_root}')
    if not os.path.isdir(ann_split_root):
        raise FileNotFoundError(f'Annotation split root not found: {ann_split_root}')

    video_dirs = list(iter_video_dirs(data_split_root, ann_split_root))
    log(
        f'Scanning split={split} under {data_split_root} using {ann_split_root}. '
        f'Found {len(video_dirs)} source videos to inspect.'
    )

    videos: List[VideoRecord] = []
    total_kept_frames = 0
    for video_index, (video_id, rel_video_dir, data_video_dir, ann_video_dir) in enumerate(video_dirs, start=1):
        image_files = sorted(
            file_name for file_name in os.listdir(data_video_dir)
            if file_name.lower().endswith(('.jpeg', '.jpg'))
        )
        if not image_files:
            continue

        frames: List[FrameRecord] = []
        for original_frame_idx, image_file_name in enumerate(image_files):
            frame_name = os.path.splitext(image_file_name)[0]
            source_xml_path = os.path.join(ann_video_dir, f'{frame_name}.xml')
            if not os.path.exists(source_xml_path):
                continue

            width, height, detections = parse_voc_compatible_objects(source_xml_path, allowed)
            if not detections:
                continue

            rel_xml_path = os.path.join(rel_video_dir, f'{frame_name}.xml')
            rel_image_path = os.path.join(rel_video_dir, image_file_name)
            frames.append(
                FrameRecord(
                    video_id=video_id,
                    rel_video_dir=rel_video_dir,
                    frame_name=frame_name,
                    original_frame_idx=original_frame_idx,
                    width=width,
                    height=height,
                    source_xml_path=source_xml_path,
                    rel_xml_path=rel_xml_path,
                    source_image_path=os.path.join(data_video_dir, image_file_name),
                    rel_image_path=rel_image_path,
                    detections=detections,
                )
            )

        if frames:
            videos.append(VideoRecord(video_id=video_id, rel_video_dir=rel_video_dir, frames=frames))
            total_kept_frames += len(frames)

        if video_index == 1 or video_index % VIDEO_LOG_INTERVAL == 0 or video_index == len(video_dirs):
            pct = 100.0 * video_index / max(len(video_dirs), 1)
            log(
                f'Scanned {video_index}/{len(video_dirs)} videos ({pct:.1f}%). '
                f'VOC-compatible videos kept: {len(videos)}. Kept frames so far: {total_kept_frames}.'
            )

    return videos


def split_into_consecutive_runs(frames: Sequence[FrameRecord]) -> List[List[FrameRecord]]:
    runs: List[List[FrameRecord]] = []
    current_run: List[FrameRecord] = []
    for frame in frames:
        if current_run and frame.original_frame_idx != current_run[-1].original_frame_idx + 1:
            runs.append(current_run)
            current_run = []
        current_run.append(frame)
    if current_run:
        runs.append(current_run)
    return runs


def size_bin_index(normalized_area: float) -> int:
    if normalized_area < SMALL_AREA_THRESHOLD:
        return 0
    if normalized_area <= MEDIUM_AREA_THRESHOLD:
        return 1
    return 2


def density_bin_index(object_count: int) -> int:
    if object_count <= 1:
        return 0
    if object_count <= 3:
        return 1
    return 2


def compute_frame_motion(previous_frame: FrameRecord, current_frame: FrameRecord) -> float:
    diagonal = max(math.hypot(previous_frame.width, previous_frame.height), 1.0)

    previous_by_class: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    current_by_class: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for det in previous_frame.detections:
        previous_by_class[det.class_name].append(det.center)
    for det in current_frame.detections:
        current_by_class[det.class_name].append(det.center)

    displacements: List[float] = []
    for class_name in sorted(set(previous_by_class) & set(current_by_class)):
        previous_centers = sorted(previous_by_class[class_name])
        current_centers = sorted(current_by_class[class_name])
        for (prev_x, prev_y), (curr_x, curr_y) in zip(previous_centers, current_centers):
            displacements.append(math.hypot(curr_x - prev_x, curr_y - prev_y) / diagonal)

    if displacements:
        return float(np.mean(displacements))

    previous_centers = [det.center for det in previous_frame.detections]
    current_centers = [det.center for det in current_frame.detections]
    prev_mean_x = float(np.mean([center[0] for center in previous_centers]))
    prev_mean_y = float(np.mean([center[1] for center in previous_centers]))
    curr_mean_x = float(np.mean([center[0] for center in current_centers]))
    curr_mean_y = float(np.mean([center[1] for center in current_centers]))
    return math.hypot(curr_mean_x - prev_mean_x, curr_mean_y - prev_mean_y) / diagonal


def compute_motion_values(frames: Sequence[FrameRecord]) -> List[float]:
    return [
        compute_frame_motion(frames[idx - 1], frames[idx])
        for idx in range(1, len(frames))
        if frames[idx].original_frame_idx == frames[idx - 1].original_frame_idx + 1
    ]


def build_candidate_clips(
    videos: Sequence[VideoRecord],
    clip_length: int,
    clip_stride: int,
) -> List[ClipCandidate]:
    if clip_length <= 0:
        raise ValueError('clip_length must be positive')
    if clip_stride <= 0:
        raise ValueError('clip_stride must be positive')

    log(
        f'Building candidate clips from {len(videos)} VOC-compatible videos '
        f'(clip_length={clip_length}, clip_stride={clip_stride}).'
    )

    candidates: List[ClipCandidate] = []
    for video_index, video in enumerate(videos, start=1):
        for run in split_into_consecutive_runs(video.frames):
            if len(run) < clip_length:
                continue

            window_starts = list(range(0, len(run) - clip_length + 1, clip_stride))
            last_start = len(run) - clip_length
            if window_starts[-1] != last_start:
                window_starts.append(last_start)

            for start_offset in window_starts:
                clip_frames = list(run[start_offset:start_offset + clip_length])
                class_histogram: Counter = Counter()
                size_bin_counts = np.zeros(3, dtype=np.float32)
                density_bin_counts = np.zeros(3, dtype=np.float32)

                for frame in clip_frames:
                    class_histogram.update(frame.class_histogram)
                    density_bin_counts[density_bin_index(frame.object_count)] += 1.0
                    for normalized_area in frame.normalized_bbox_areas:
                        size_bin_counts[size_bin_index(normalized_area)] += 1.0

                motion_values = compute_motion_values(clip_frames)
                mean_motion = float(np.mean(motion_values)) if motion_values else 0.0
                motion_variance = float(np.var(motion_values)) if motion_values else 0.0
                mean_objects_per_frame = float(np.mean([frame.object_count for frame in clip_frames]))
                dominant_class = None
                if class_histogram:
                    dominant_class = class_histogram.most_common(1)[0][0]

                candidates.append(
                    ClipCandidate(
                        video_id=video.video_id,
                        rel_video_dir=video.rel_video_dir,
                        start_frame_idx=clip_frames[0].original_frame_idx,
                        end_frame_idx=clip_frames[-1].original_frame_idx,
                        frames=clip_frames,
                        class_histogram=class_histogram,
                        size_bin_counts=size_bin_counts,
                        density_bin_counts=density_bin_counts,
                        motion_values=motion_values,
                        mean_motion=mean_motion,
                        motion_variance=motion_variance,
                        mean_objects_per_frame=mean_objects_per_frame,
                        dominant_class=dominant_class,
                    )
                )

        if video_index == 1 or video_index % VIDEO_LOG_INTERVAL == 0 or video_index == len(videos):
            pct = 100.0 * video_index / max(len(videos), 1)
            log(
                f'Processed {video_index}/{len(videos)} videos for clip generation ({pct:.1f}%). '
                f'Candidates so far: {len(candidates)}.'
            )

    if not candidates:
        raise RuntimeError('No candidate VOC-compatible clips were found with the requested clip length.')

    build_feature_vectors(candidates)
    log(f'Candidate clip generation finished with {len(candidates)} candidates.')
    return candidates


def build_feature_vectors(candidates: Sequence[ClipCandidate]) -> None:
    ordered_classes = sorted(IMAGENET_VID_VOC_OVERLAP_CLASSES)
    max_mean_motion = max((candidate.mean_motion for candidate in candidates), default=0.0)
    max_motion_variance = max((candidate.motion_variance for candidate in candidates), default=0.0)
    max_density = max((candidate.mean_objects_per_frame for candidate in candidates), default=0.0)

    motion_scale = max(max_mean_motion, 1e-6)
    variance_scale = max(max_motion_variance, 1e-6)
    density_scale = max(max_density, 1e-6)

    for candidate in candidates:
        total_objects = float(sum(candidate.class_histogram.values())) or 1.0
        total_sizes = float(np.sum(candidate.size_bin_counts)) or 1.0
        total_density = float(np.sum(candidate.density_bin_counts)) or 1.0

        class_vector = [candidate.class_histogram.get(class_name, 0.0) / total_objects for class_name in ordered_classes]
        size_vector = (candidate.size_bin_counts / total_sizes).tolist()
        density_vector = (candidate.density_bin_counts / total_density).tolist()
        scalar_vector = [
            candidate.mean_objects_per_frame / density_scale,
            candidate.mean_motion / motion_scale,
            candidate.motion_variance / variance_scale,
        ]
        candidate.feature_vector = np.asarray(class_vector + size_vector + density_vector + scalar_vector, dtype=np.float32)


def _clip_overlaps(first_clip: ClipCandidate, second_clip: ClipCandidate) -> bool:
    if first_clip.video_id != second_clip.video_id:
        return False
    return not (
        first_clip.end_frame_idx < second_clip.start_frame_idx
        or second_clip.end_frame_idx < first_clip.start_frame_idx
    )


def select_representative_clips(
    candidates: Sequence[ClipCandidate],
    target_clip_count: int,
    max_clips_per_video: int,
    seed: int,
) -> List[ClipCandidate]:
    if target_clip_count <= 0:
        return []

    log(
        f'Selecting up to {target_clip_count} representative clips from {len(candidates)} candidates '
        f'(max_clips_per_video={max_clips_per_video}, seed={seed}).'
    )

    rng = random.Random(seed)
    shuffled_candidates = list(candidates)
    rng.shuffle(shuffled_candidates)
    target_feature = np.mean(np.stack([candidate.feature_vector for candidate in shuffled_candidates], axis=0), axis=0)

    selected: List[ClipCandidate] = []
    selected_by_video: Dict[str, List[ClipCandidate]] = defaultdict(list)
    covered_classes = set()
    selected_feature_sum = np.zeros_like(target_feature)

    class_candidate_frequency = Counter()
    for candidate in shuffled_candidates:
        for class_name in candidate.class_histogram:
            class_candidate_frequency[class_name] += 1

    for class_name, _ in sorted(class_candidate_frequency.items(), key=lambda item: (item[1], item[0])):
        if len(selected) >= target_clip_count:
            break
        best_candidate = None
        best_score = None
        for candidate in shuffled_candidates:
            if candidate in selected:
                continue
            if class_name not in candidate.class_histogram:
                continue
            if len(selected_by_video[candidate.video_id]) >= max_clips_per_video:
                continue
            if any(_clip_overlaps(candidate, existing) for existing in selected_by_video[candidate.video_id]):
                continue

            class_fraction = candidate.class_histogram[class_name] / max(1.0, float(sum(candidate.class_histogram.values())))
            deviation = np.mean(np.abs(candidate.feature_vector - target_feature))
            score = deviation - 0.25 * class_fraction
            if best_score is None or score < best_score:
                best_score = score
                best_candidate = candidate

        if best_candidate is None:
            continue
        selected.append(best_candidate)
        selected_by_video[best_candidate.video_id].append(best_candidate)
        selected_feature_sum += best_candidate.feature_vector
        covered_classes.update(best_candidate.class_histogram.keys())

    if selected:
        log(
            f'Rare-class seeding selected {len(selected)} clips. '
            f'Covered classes: {len(covered_classes)}.'
        )

    while len(selected) < target_clip_count:
        best_candidate = None
        best_score = None
        for candidate in shuffled_candidates:
            if candidate in selected:
                continue
            if len(selected_by_video[candidate.video_id]) >= max_clips_per_video:
                continue
            if any(_clip_overlaps(candidate, existing) for existing in selected_by_video[candidate.video_id]):
                continue

            next_average = (selected_feature_sum + candidate.feature_vector) / float(len(selected) + 1)
            deviation = float(np.mean(np.abs(next_average - target_feature)))
            new_class_bonus = sum(1 for class_name in candidate.class_histogram if class_name not in covered_classes)
            reuse_penalty = 0.01 * len(selected_by_video[candidate.video_id])
            score = deviation + reuse_penalty - 0.005 * new_class_bonus
            if best_score is None or score < best_score:
                best_score = score
                best_candidate = candidate

        if best_candidate is None:
            break

        selected.append(best_candidate)
        selected_by_video[best_candidate.video_id].append(best_candidate)
        selected_feature_sum += best_candidate.feature_vector
        covered_classes.update(best_candidate.class_histogram.keys())

        if len(selected) == 1 or len(selected) % 25 == 0 or len(selected) == target_clip_count:
            log(
                f'Selected {len(selected)}/{target_clip_count} clips. '
                f'Covered classes: {len(covered_classes)}.'
            )

    log(f'Clip selection finished with {len(selected)} clips.')
    return selected


def compute_motion_thresholds(motion_values: Sequence[float]) -> Tuple[float, float]:
    if not motion_values:
        return 0.0, 0.0
    values = np.asarray(motion_values, dtype=np.float32)
    return float(np.quantile(values, 1.0 / 3.0)), float(np.quantile(values, 2.0 / 3.0))


def motion_bin_counts(motion_values: Sequence[float], thresholds: Tuple[float, float]) -> np.ndarray:
    low_threshold, high_threshold = thresholds
    bins = np.zeros(3, dtype=np.float32)
    if not motion_values:
        bins[0] = 1.0
        return bins
    for motion_value in motion_values:
        if motion_value <= low_threshold:
            bins[0] += 1.0
        elif motion_value <= high_threshold:
            bins[1] += 1.0
        else:
            bins[2] += 1.0
    return bins


def summarize_video_records(videos: Sequence[VideoRecord], motion_thresholds: Optional[Tuple[float, float]] = None) -> Dict[str, object]:
    class_histogram: Counter = Counter()
    size_bins = np.zeros(3, dtype=np.float32)
    density_bins = np.zeros(3, dtype=np.float32)
    motion_values: List[float] = []
    frame_count = 0
    object_count = 0

    for video in videos:
        frame_count += len(video.frames)
        for frame in video.frames:
            class_histogram.update(frame.class_histogram)
            object_count += frame.object_count
            density_bins[density_bin_index(frame.object_count)] += 1.0
            for normalized_area in frame.normalized_bbox_areas:
                size_bins[size_bin_index(normalized_area)] += 1.0
        motion_values.extend(compute_motion_values(video.frames))

    thresholds = motion_thresholds or compute_motion_thresholds(motion_values)
    motion_bins = motion_bin_counts(motion_values, thresholds)
    total_objects = float(sum(class_histogram.values())) or 1.0
    total_sizes = float(np.sum(size_bins)) or 1.0
    total_density = float(np.sum(density_bins)) or 1.0
    total_motions = float(np.sum(motion_bins)) or 1.0

    return {
        'frames': frame_count,
        'objects': object_count,
        'videos': len(videos),
        'class_histogram': dict(class_histogram),
        'class_distribution': {class_name: count / total_objects for class_name, count in sorted(class_histogram.items())},
        'size_bin_distribution': {
            'small': float(size_bins[0] / total_sizes),
            'medium': float(size_bins[1] / total_sizes),
            'large': float(size_bins[2] / total_sizes),
        },
        'density_distribution': {
            DENSITY_LABELS[idx]: float(density_bins[idx] / total_density)
            for idx in range(len(DENSITY_LABELS))
        },
        'motion_distribution': {
            'static_or_slow': float(motion_bins[0] / total_motions),
            'moderate': float(motion_bins[1] / total_motions),
            'fast': float(motion_bins[2] / total_motions),
        },
        'motion_thresholds': {'low': thresholds[0], 'high': thresholds[1]},
        'mean_objects_per_frame': float(object_count / max(frame_count, 1)),
        'mean_motion': float(np.mean(motion_values)) if motion_values else 0.0,
        'motion_variance': float(np.var(motion_values)) if motion_values else 0.0,
    }


def summarize_selected_clips(clips: Sequence[ClipCandidate], motion_thresholds: Tuple[float, float]) -> Dict[str, object]:
    class_histogram: Counter = Counter()
    size_bins = np.zeros(3, dtype=np.float32)
    density_bins = np.zeros(3, dtype=np.float32)
    motion_values: List[float] = []
    frame_count = 0
    selected_videos = set()

    for clip in clips:
        class_histogram.update(clip.class_histogram)
        size_bins += clip.size_bin_counts
        density_bins += clip.density_bin_counts
        motion_values.extend(clip.motion_values)
        frame_count += len(clip.frames)
        selected_videos.add(clip.video_id)

    motion_bins = motion_bin_counts(motion_values, motion_thresholds)
    total_objects = float(sum(class_histogram.values())) or 1.0
    total_sizes = float(np.sum(size_bins)) or 1.0
    total_density = float(np.sum(density_bins)) or 1.0
    total_motions = float(np.sum(motion_bins)) or 1.0

    return {
        'clips': len(clips),
        'frames': frame_count,
        'videos': len(selected_videos),
        'class_histogram': dict(class_histogram),
        'class_distribution': {class_name: count / total_objects for class_name, count in sorted(class_histogram.items())},
        'size_bin_distribution': {
            'small': float(size_bins[0] / total_sizes),
            'medium': float(size_bins[1] / total_sizes),
            'large': float(size_bins[2] / total_sizes),
        },
        'density_distribution': {
            DENSITY_LABELS[idx]: float(density_bins[idx] / total_density)
            for idx in range(len(DENSITY_LABELS))
        },
        'motion_distribution': {
            'static_or_slow': float(motion_bins[0] / total_motions),
            'moderate': float(motion_bins[1] / total_motions),
            'fast': float(motion_bins[2] / total_motions),
        },
        'mean_objects_per_frame': float(
            sum(clip.mean_objects_per_frame * len(clip.frames) for clip in clips) / max(frame_count, 1)
        ),
        'mean_motion': float(np.mean(motion_values)) if motion_values else 0.0,
        'motion_variance': float(np.var(motion_values)) if motion_values else 0.0,
    }


def write_selected_annotations(
    selected_clips: Sequence[ClipCandidate],
    annotation_root: str,
    split: str,
    output_folder_name: str,
    overwrite: bool = False,
) -> str:
    output_split_root = resolve_output_split_root(annotation_root, split, output_folder_name)
    output_root = os.path.dirname(output_split_root)

    if overwrite and os.path.isdir(output_root):
        shutil.rmtree(output_root)
    elif os.path.exists(output_root):
        raise FileExistsError(f'Output folder already exists: {output_root}. Use --overwrite to replace it.')

    total_unique_frames = len({frame.rel_xml_path for clip in selected_clips for frame in clip.frames})
    log(
        f'Writing {total_unique_frames} pruned XML files to {output_split_root} '
        f'from {len(selected_clips)} selected clips.'
    )

    written_rel_paths = set()
    for clip in selected_clips:
        for frame in clip.frames:
            if frame.rel_xml_path in written_rel_paths:
                continue
            destination_xml_path = os.path.join(output_split_root, frame.rel_xml_path)
            prune_xml_to_voc_overlap(frame.source_xml_path, destination_xml_path)
            written_rel_paths.add(frame.rel_xml_path)

    return output_split_root


def write_manifest(
    selected_clips: Sequence[ClipCandidate],
    full_summary: Dict[str, object],
    subset_summary: Dict[str, object],
    annotation_root: str,
    split: str,
    output_folder_name: str,
    manifest_name: str,
    csv_name: str,
    args: argparse.Namespace,
) -> Tuple[str, str]:
    output_split_root = resolve_output_split_root(annotation_root, split, output_folder_name)
    output_root = os.path.dirname(output_split_root)
    os.makedirs(output_root, exist_ok=True)

    manifest_path = os.path.join(output_root, manifest_name)
    csv_path = os.path.join(output_root, csv_name)

    manifest = {
        'split': split,
        'source_data_root': resolve_split_root(args.data_root, split),
        'source_annotation_root': resolve_split_root(annotation_root, split),
        'output_annotation_root': output_split_root,
        'clip_length': args.clip_length,
        'clip_stride': args.clip_stride,
        'target_clips': args.target_clips,
        'selected_clip_count': len(selected_clips),
        'selected_frame_count': int(sum(len(clip.frames) for clip in selected_clips)),
        'max_clips_per_video': args.max_clips_per_video,
        'seed': args.seed,
        'full_summary': full_summary,
        'subset_summary': subset_summary,
        'selected_clips': [
            {
                'video_id': clip.video_id,
                'relative_video_dir': clip.rel_video_dir,
                'start_frame_idx': clip.start_frame_idx,
                'end_frame_idx': clip.end_frame_idx,
                'frame_count': len(clip.frames),
                'dominant_class': clip.dominant_class,
                'class_histogram': dict(clip.class_histogram),
                'mean_objects_per_frame': clip.mean_objects_per_frame,
                'mean_motion': clip.mean_motion,
                'motion_variance': clip.motion_variance,
                'relative_xml_paths': [frame.rel_xml_path for frame in clip.frames],
            }
            for clip in selected_clips
        ],
    }

    with open(manifest_path, 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2)

    with open(csv_path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                'video_id', 'relative_video_dir', 'start_frame_idx', 'end_frame_idx',
                'frame_count', 'dominant_class', 'mean_objects_per_frame', 'mean_motion', 'motion_variance',
            ],
        )
        writer.writeheader()
        for clip in selected_clips:
            writer.writerow({
                'video_id': clip.video_id,
                'relative_video_dir': clip.rel_video_dir,
                'start_frame_idx': clip.start_frame_idx,
                'end_frame_idx': clip.end_frame_idx,
                'frame_count': len(clip.frames),
                'dominant_class': clip.dominant_class or '',
                'mean_objects_per_frame': f'{clip.mean_objects_per_frame:.6f}',
                'mean_motion': f'{clip.mean_motion:.6f}',
                'motion_variance': f'{clip.motion_variance:.6f}',
            })

    return manifest_path, csv_path


def build_subset(args: argparse.Namespace) -> Dict[str, object]:
    log(
        f'Starting subset build for split={args.split}, target_clips={args.target_clips}, '
        f'clip_length={args.clip_length}, dry_run={args.dry_run}.'
    )
    videos = scan_split_videos(args.data_root, args.ann_root, args.split)
    candidates = build_candidate_clips(videos, clip_length=args.clip_length, clip_stride=args.clip_stride)
    selected_clips = select_representative_clips(
        candidates,
        target_clip_count=args.target_clips,
        max_clips_per_video=args.max_clips_per_video,
        seed=args.seed,
    )

    if not selected_clips:
        raise RuntimeError('No representative clips were selected.')

    full_summary = summarize_video_records(videos)
    motion_thresholds = (
        full_summary['motion_thresholds']['low'],
        full_summary['motion_thresholds']['high'],
    )
    subset_summary = summarize_selected_clips(selected_clips, motion_thresholds)

    output_split_root = resolve_output_split_root(args.ann_root, args.split, args.output_folder_name)
    if not args.dry_run:
        output_split_root = write_selected_annotations(
            selected_clips,
            annotation_root=args.ann_root,
            split=args.split,
            output_folder_name=args.output_folder_name,
            overwrite=args.overwrite,
        )
    else:
        log('Dry run enabled. Skipping XML write step.')

    manifest_path, csv_path = write_manifest(
        selected_clips=selected_clips,
        full_summary=full_summary,
        subset_summary=subset_summary,
        annotation_root=args.ann_root,
        split=args.split,
        output_folder_name=args.output_folder_name,
        manifest_name=args.manifest_name,
        csv_name=args.csv_name,
        args=args,
    )

    log(f'Wrote manifest to {manifest_path}')
    log(f'Wrote clip summary CSV to {csv_path}')

    return {
        'videos': len(videos),
        'candidate_clips': len(candidates),
        'selected_clips': len(selected_clips),
        'selected_frames': int(sum(len(clip.frames) for clip in selected_clips)),
        'output_split_root': output_split_root,
        'manifest_path': manifest_path,
        'csv_path': csv_path,
    }


def main() -> None:
    args = parse_args()
    result = build_subset(args)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()