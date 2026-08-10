from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tools.trackers.sort import KalmanBoxTracker, Sort


@dataclass
class _TrackMeta:
	cls: Any
	confidence: float
	hits: int
	misses: int
	age: int
	bbox: List[float]
	roi: Optional[List[int]] = None


class SortTracker:
	"""Repository wrapper around upstream SORT tracker.

	Exposes the same `update()/reset()` contract used by benchmark and inference
	pipeline code while keeping SORT's internal tracking logic untouched.
	"""

	def __init__(
		self,
		max_age: int = 10,
		min_hits: int = 3,
		iou_threshold: float = 0.3,
		conf_det_min: float = 0.0,
		pad_x: int = 16,
		pad_y: int = 16,
		model_input_size: Optional[Tuple[int, int]] = None,
		hold_last_for_frames: int = 3,
		min_roi_size: int = 2,
	):
		self.max_age = int(max_age)
		self.min_hits = int(min_hits)
		self.iou_threshold = float(iou_threshold)
		self.conf_det_min = float(conf_det_min)
		self.pad_x = float(pad_x)
		self.pad_y = float(pad_y)
		self.hold_last_for_frames = max(0, int(hold_last_for_frames))
		self.min_roi_size = max(1, int(min_roi_size))

		self.model_input_size = model_input_size
		if model_input_size is None:
			self.model_input_h = None
			self.model_input_w = None
		else:
			if len(model_input_size) != 2:
				raise ValueError("model_input_size must have 2 elements: [height, width]")
			self.model_input_h = float(model_input_size[0])
			self.model_input_w = float(model_input_size[1])
			if self.model_input_h <= 1.0 or self.model_input_w <= 1.0:
				raise ValueError("model_input_size dimensions must be > 1")

		self._sort = Sort(
			max_age=self.max_age,
			min_hits=self.min_hits,
			iou_threshold=self.iou_threshold,
		)
		self._meta: Dict[int, _TrackMeta] = {}

	def reset(self) -> None:
		KalmanBoxTracker.count = 0
		self._sort = Sort(
			max_age=self.max_age,
			min_hits=self.min_hits,
			iou_threshold=self.iou_threshold,
		)
		self._meta = {}

	def update(self, detections: List[Dict[str, Any]], frame_shape: Tuple[int, int]) -> Dict[str, Any]:
		frame_h, frame_w = frame_shape

		prepared: List[Dict[str, Any]] = []
		det_rows: List[List[float]] = []
		for det in detections:
			conf = float(det.get("confidence", 1.0))
			if conf < self.conf_det_min:
				continue
			x1, y1, x2, y2 = [float(v) for v in det["bbox"]]
			if x2 <= x1 or y2 <= y1:
				continue
			prepared.append({
				"bbox": [x1, y1, x2, y2],
				"class": det.get("class", "unknown"),
				"confidence": conf,
			})
			det_rows.append([x1, y1, x2, y2, conf])

		if det_rows:
			dets_np = np.asarray(det_rows, dtype=float)
		else:
			dets_np = np.empty((0, 5), dtype=float)

		sort_out = self._sort.update(dets_np)

		active_tracker_stats: Dict[int, Dict[str, int]] = {}
		for trk in self._sort.trackers:
			active_tracker_stats[int(trk.id + 1)] = {
				"hits": int(getattr(trk, "hits", 0)),
				"age": int(getattr(trk, "age", 0)),
				"misses": int(getattr(trk, "time_since_update", 0)),
			}

		for tid, meta in list(self._meta.items()):
			if tid in active_tracker_stats:
				meta.misses = active_tracker_stats[tid]["misses"]
				meta.hits = max(meta.hits, active_tracker_stats[tid]["hits"])
				meta.age = max(meta.age, active_tracker_stats[tid]["age"])
			else:
				meta.misses += 1
				meta.age += 1

		matches = self._match_sort_to_detections(sort_out, prepared)

		tracks: List[Dict[str, Any]] = []
		rois: List[Dict[str, Any]] = []
		returned_ids = set()

		for sort_idx, sort_row in enumerate(sort_out):
			x1, y1, x2, y2, tid_raw = [float(v) for v in sort_row]
			tid = int(tid_raw)
			returned_ids.add(tid)

			stat = active_tracker_stats.get(tid, {"hits": 1, "age": 1, "misses": 0})
			det_idx = matches.get(sort_idx)

			if det_idx is not None:
				det = prepared[det_idx]
				cls = det["class"]
				conf = float(det["confidence"])
			elif tid in self._meta:
				cls = self._meta[tid].cls
				conf = self._meta[tid].confidence
			else:
				cls = "unknown"
				conf = 1.0

			bbox = [x1, y1, x2, y2]
			roi = self._build_roi(bbox=bbox, frame_w=frame_w, frame_h=frame_h)

			self._meta[tid] = _TrackMeta(
				cls=cls,
				confidence=conf,
				hits=stat["hits"],
				misses=stat["misses"],
				age=stat["age"],
				bbox=bbox,
				roi=roi,
			)

			tracks.append(
				{
					"track_id": tid,
					"class": cls,
					"bbox": bbox,
					"confidence": conf,
					"age": stat["age"],
					"hits": stat["hits"],
					"misses": stat["misses"],
					"roi": roi,
				}
			)
			rois.append(
				{
					"track_id": tid,
					"class": cls,
					"roi": roi,
					"bbox_pred": bbox,
					"confidence": conf,
					"misses": stat["misses"],
					"hits": stat["hits"],
				}
			)

		for tid in list(self._meta.keys()):
			meta = self._meta[tid]
			if tid not in returned_ids and meta.misses <= self.hold_last_for_frames and meta.roi is not None:
				tracks.append(
					{
						"track_id": tid,
						"class": meta.cls,
						"bbox": list(meta.bbox),
						"confidence": float(meta.confidence),
						"age": int(meta.age),
						"hits": int(meta.hits),
						"misses": int(meta.misses),
						"roi": list(meta.roi),
					}
				)
				rois.append(
					{
						"track_id": tid,
						"class": meta.cls,
						"roi": list(meta.roi),
						"bbox_pred": list(meta.bbox),
						"confidence": float(meta.confidence),
						"misses": int(meta.misses),
						"hits": int(meta.hits),
					}
				)

			if meta.misses > max(self.max_age, self.hold_last_for_frames):
				del self._meta[tid]

		return {"tracks": tracks, "rois": rois}

	def _build_roi(self, bbox: List[float], frame_w: int, frame_h: int) -> List[int]:
		x1, y1, x2, y2 = bbox

		if self.model_input_h is not None and self.model_input_w is not None:
			pad_x = self.pad_x * (float(frame_w) / self.model_input_w)
			pad_y = self.pad_y * (float(frame_h) / self.model_input_h)
		else:
			pad_x = self.pad_x
			pad_y = self.pad_y

		rx1 = int(max(0, np.floor(x1 - pad_x)))
		ry1 = int(max(0, np.floor(y1 - pad_y)))
		rx2 = int(min(frame_w - 1, np.ceil(x2 + pad_x)))
		ry2 = int(min(frame_h - 1, np.ceil(y2 + pad_y)))

		if rx2 <= rx1:
			rx2 = min(frame_w - 1, rx1 + 1)
		if ry2 <= ry1:
			ry2 = min(frame_h - 1, ry1 + 1)

		if frame_w >= self.min_roi_size and (rx2 - rx1) < self.min_roi_size:
			cx_i = int(round((rx1 + rx2) / 2.0))
			rx1 = max(0, min(frame_w - self.min_roi_size, cx_i - self.min_roi_size // 2))
			rx2 = rx1 + self.min_roi_size
		if frame_h >= self.min_roi_size and (ry2 - ry1) < self.min_roi_size:
			cy_i = int(round((ry1 + ry2) / 2.0))
			ry1 = max(0, min(frame_h - self.min_roi_size, cy_i - self.min_roi_size // 2))
			ry2 = ry1 + self.min_roi_size

		return [rx1, ry1, rx2, ry2]

	@staticmethod
	def _match_sort_to_detections(sort_out: np.ndarray, detections: List[Dict[str, Any]]) -> Dict[int, int]:
		"""Greedy IoU matching from SORT outputs to current-frame detections."""
		matches: Dict[int, int] = {}
		if len(sort_out) == 0 or len(detections) == 0:
			return matches

		used_dets = set()
		for si, row in enumerate(sort_out):
			sb = [float(row[0]), float(row[1]), float(row[2]), float(row[3])]
			best_iou = 0.0
			best_idx = None
			for di, det in enumerate(detections):
				if di in used_dets:
					continue
				iou = SortTracker._iou(sb, det["bbox"])
				if iou > best_iou:
					best_iou = iou
					best_idx = di
			if best_idx is not None and best_iou > 0.0:
				matches[si] = best_idx
				used_dets.add(best_idx)

		return matches

	@staticmethod
	def _iou(box_a: List[float], box_b: List[float]) -> float:
		ax1, ay1, ax2, ay2 = box_a
		bx1, by1, bx2, by2 = box_b

		inter_x1 = max(ax1, bx1)
		inter_y1 = max(ay1, by1)
		inter_x2 = min(ax2, bx2)
		inter_y2 = min(ay2, by2)

		inter_w = max(0.0, inter_x2 - inter_x1)
		inter_h = max(0.0, inter_y2 - inter_y1)
		inter_area = inter_w * inter_h

		area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
		area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

		union = area_a + area_b - inter_area
		if union <= 0.0:
			return 0.0

		return inter_area / union
