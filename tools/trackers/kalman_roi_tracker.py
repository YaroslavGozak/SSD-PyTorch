import math
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Track:
    track_id: int
    cls: Any
    x: np.ndarray
    P: np.ndarray
    age: int = 1
    hits: int = 1
    misses: int = 0
    last_confidence: float = 1.0
    roi: Optional[List[int]] = None


class KalmanRoiTracker:
    def __init__(
        self,
        dt: float = 1.0,
        conf_det_min: float = 0.2,
        conf_low: float = 0.4,
        conf_high: float = 0.7,
        iou_match_threshold: float = 0.1,
        max_misses: int = 10,
        pmin: float = 12.0,
        uncertainty_scale_pos: float = 2.5,
        uncertainty_scale_size: float = 1.0,
        confidence_roi_scale: float = 0.10,
        process_noise_pos: float = 2.0,
        process_noise_size: float = 1.0,
        process_noise_vel_pos: float = 1.0,
        process_noise_vel_size: float = 0.5,
        measurement_noise_pos: float = 4.0,
        measurement_noise_size: float = 2.0,
        init_pos_var: float = 25.0,
        init_size_var: float = 25.0,
        init_vel_var: float = 100.0,
        invalid_match_cost: float = 1e6,
        mahalanobis_threshold: float = 9.4877,
        cost_lambda_iou: float = 1.0,
        cost_lambda_mahalanobis: float = 0.0,
    ):
        self.dt = dt

        self.conf_det_min = conf_det_min
        self.conf_low = conf_low
        self.conf_high = conf_high
        self.iou_match_threshold = iou_match_threshold
        self.max_misses = max_misses

        self.pmin = pmin
        self.uncertainty_scale_pos = uncertainty_scale_pos
        self.uncertainty_scale_size = uncertainty_scale_size
        self.confidence_roi_scale = confidence_roi_scale
        self.invalid_match_cost = invalid_match_cost
        self.mahalanobis_threshold = mahalanobis_threshold

        self.cost_lambda_iou = cost_lambda_iou
        self.cost_lambda_mahalanobis = cost_lambda_mahalanobis

        self.F = np.array([
            [1, 0, 0, 0, dt, 0,  0,  0],
            [0, 1, 0, 0, 0,  dt, 0,  0],
            [0, 0, 1, 0, 0,  0,  dt, 0],
            [0, 0, 0, 1, 0,  0,  0,  dt],
            [0, 0, 0, 0, 1,  0,  0,  0],
            [0, 0, 0, 0, 0,  1,  0,  0],
            [0, 0, 0, 0, 0,  0,  1,  0],
            [0, 0, 0, 0, 0,  0,  0,  1],
        ], dtype=float)

        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ], dtype=float)

        self.Q = np.diag([
            process_noise_pos,
            process_noise_pos,
            process_noise_size,
            process_noise_size,
            process_noise_vel_pos,
            process_noise_vel_pos,
            process_noise_vel_size,
            process_noise_vel_size,
        ]).astype(float)

        self.R_base = np.diag([
            measurement_noise_pos,
            measurement_noise_pos,
            measurement_noise_size,
            measurement_noise_size,
        ]).astype(float)

        self.P0 = np.diag([
            init_pos_var,
            init_pos_var,
            init_size_var,
            init_size_var,
            init_vel_var,
            init_vel_var,
            init_vel_var,
            init_vel_var,
        ]).astype(float)

        self.tracks: List[Track] = []
        self.next_track_id = 1

    def update(self, detections: List[Dict[str, Any]], frame_shape: Tuple[int, int]) -> Dict[str, Any]:
        """
        detections: list of dicts with keys:
            - bbox: [x1, y1, x2, y2]
            - class: class label
            - confidence: float
        frame_shape: (height, width)

        Returns dict with:
            - tracks: active tracks
            - rois: list of ROI dicts
        """
        frame_h, frame_w = frame_shape

        filtered_detections = self._prepare_detections(detections)

        for track in self.tracks:
            self._predict(track)

        matches, unmatched_tracks, unmatched_detections = self._match_tracks_and_detections(
            self.tracks, filtered_detections
        )

        for track_idx, det_idx in matches:
            track = self.tracks[track_idx]
            det = filtered_detections[det_idx]
            self._update_matched_track(track, det)

        for track_idx in unmatched_tracks:
            track = self.tracks[track_idx]
            track.age += 1
            track.misses += 1

        for det_idx in unmatched_detections:
            det = filtered_detections[det_idx]
            self._start_new_track(det)

        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]

        rois = []
        for track in self.tracks:
            roi = self._build_roi(track, frame_w, frame_h)
            track.roi = roi
            rois.append({
                "track_id": track.track_id,
                "class": track.cls,
                "roi": roi,
                "bbox_pred": self._state_to_bbox(track.x),
                "confidence": track.last_confidence,
                "misses": track.misses,
                "hits": track.hits,
            })

        return {
            "tracks": self._export_tracks(),
            "rois": rois,
        }

    def get_tracks(self) -> List[Dict[str, Any]]:
        return self._export_tracks()

    def reset(self) -> None:
        self.tracks = []
        self.next_track_id = 1

    def _prepare_detections(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        prepared = []

        for det in detections:
            conf = float(det.get("confidence", 1.0))
            if conf < self.conf_det_min:
                continue

            bbox = det["bbox"]
            cls = det["class"]

            cx, cy, w, h = self._bbox_to_measurement(bbox)

            prepared.append({
                "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                "class": cls,
                "confidence": conf,
                "z": np.array([cx, cy, w, h], dtype=float),
            })

        return prepared

    def _predict(self, track: Track) -> None:
        track.x = self.F @ track.x
        track.P = self.F @ track.P @ self.F.T + self.Q

        # Enforce physically valid box size after prediction.
        min_size = 1.0
        if track.x[2] < min_size:
            track.x[2] = min_size
            # If width is at floor, don't keep pushing it down.
            if track.x[6] < 0.0:
                track.x[6] = 0.0

        if track.x[3] < min_size:
            track.x[3] = min_size
            # If height is at floor, don't keep pushing it down.
            if track.x[7] < 0.0:
                track.x[7] = 0.0

    def _update_matched_track(self, track: Track, det: Dict[str, Any]) -> None:
        conf = det["confidence"]

        if conf < self.conf_low:
            track.age += 1
            track.misses += 1
            return

        R = self._get_adaptive_R(conf)
        z = det["z"]

        y = z - self.H @ track.x
        S = self.H @ track.P @ self.H.T + R
        K = track.P @ self.H.T @ np.linalg.inv(S)

        track.x = track.x + K @ y
        I = np.eye(track.P.shape[0])
        track.P = (I - K @ self.H) @ track.P

        track.cls = det["class"]
        track.age += 1
        track.hits += 1
        track.misses = 0
        track.last_confidence = conf

        track.x[2] = max(track.x[2], 1.0)
        track.x[3] = max(track.x[3], 1.0)

    def _get_adaptive_R(self, conf: float) -> np.ndarray:
        if conf >= self.conf_high:
            return self.R_base.copy()

        alpha = (self.conf_high - conf) / max(self.conf_high - self.conf_low, 1e-6)
        scale = 1.0 + 2.0 * alpha
        return self.R_base * scale

    def _start_new_track(self, det: Dict[str, Any]) -> None:
        z = det["z"]
        x0 = np.array([z[0], z[1], z[2], z[3], 0.0, 0.0, 0.0, 0.0], dtype=float)

        track = Track(
            track_id=self.next_track_id,
            cls=det["class"],
            x=x0,
            P=self.P0.copy(),
            age=1,
            hits=1,
            misses=0,
            last_confidence=det["confidence"],
        )
        self.tracks.append(track)
        self.next_track_id += 1

    def _match_tracks_and_detections(
        self,
        tracks: List[Track],
        detections: List[Dict[str, Any]],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Classic greedy matching based on IoU cost, with class consistency and IoU thresholding.
         - More efficient than Hungarian for typical small numbers of tracks/detections.
        """

        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        candidate_pairs = []

        for ti, track in enumerate(tracks):
            track_bbox = self._state_to_bbox(track.x)

            for di, det in enumerate(detections):
                if track.cls != det["class"]:
                    continue

                iou = self._iou(track_bbox, det["bbox"])
                if iou < self.iou_match_threshold:
                    continue

                cost = 1.0 - iou
                candidate_pairs.append((cost, ti, di))

        candidate_pairs.sort(key=lambda x: x[0])

        matched_tracks = set()
        matched_detections = set()
        matches = []

        for cost, ti, di in candidate_pairs:
            if ti in matched_tracks or di in matched_detections:
                continue
            matches.append((ti, di))
            matched_tracks.add(ti)
            matched_detections.add(di)

        unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_tracks]
        unmatched_detections = [i for i in range(len(detections)) if i not in matched_detections]

        return matches, unmatched_tracks, unmatched_detections

    # def _match_tracks_and_detections(
    #     self,
    #     tracks: List[Track],
    #     detections: List[Dict[str, Any]],
    # ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    #     """
    #     Uses Hungarian algorithm for optimal assignment based on IoU cost.
    #      - Only considers pairs with matching class and IoU above threshold.
    #     """

    #     if not tracks or not detections:
    #         return [], list(range(len(tracks))), list(range(len(detections)))

    #     num_tracks = len(tracks)
    #     num_dets = len(detections)

    #     cost_matrix = np.full((num_tracks, num_dets), self.invalid_match_cost, dtype=float)

    #     for ti, track in enumerate(tracks):
    #         track_bbox = self._state_to_bbox(track.x)

    #         for di, det in enumerate(detections):
    #             if track.cls != det["class"]:
    #                 continue

    #             iou = self._iou(track_bbox, det["bbox"])
    #             if iou < self.iou_match_threshold:
    #                 continue

    #             cost_matrix[ti, di] = 1.0 - iou

    #     row_ind, col_ind = linear_sum_assignment(cost_matrix)

    #     matches = []
    #     matched_tracks = set()
    #     matched_detections = set()

    #     for ti, di in zip(row_ind, col_ind):
    #         if cost_matrix[ti, di] >= self.invalid_match_cost:
    #             continue

    #         matches.append((ti, di))
    #         matched_tracks.add(ti)
    #         matched_detections.add(di)

    #     unmatched_tracks = [i for i in range(num_tracks) if i not in matched_tracks]
    #     unmatched_detections = [i for i in range(num_dets) if i not in matched_detections]

    #     return matches, unmatched_tracks, unmatched_detections

    # def _match_tracks_and_detections(
    #     self,
    #     tracks: List[Track],
    #     detections: List[Dict[str, Any]],
    # ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    #     """
    #     Matches tracks and detections using a combination of IoU and Mahalanobis distance.
    #     Returns a list of matched track-detection pairs, unmatched tracks, and unmatched detections.
    #     """

    #     if not tracks or not detections:
    #         return [], list(range(len(tracks))), list(range(len(detections)))

    #     num_tracks = len(tracks)
    #     num_dets = len(detections)

    #     cost_matrix = np.full((num_tracks, num_dets), self.invalid_match_cost, dtype=float)

    #     for ti, track in enumerate(tracks):
    #         track_bbox = self._state_to_bbox(track.x)

    #         for di, det in enumerate(detections):
    #             if track.cls != det["class"]:
    #                 continue

    #             conf = det["confidence"]
    #             R = self._get_adaptive_R(conf)

    #             d2 = self._mahalanobis_squared(track, det["z"], R)
    #             if d2 > self.mahalanobis_threshold:
    #                 continue

    #             iou = self._iou(track_bbox, det["bbox"])
    #             if iou < self.iou_match_threshold:
    #                 continue

    #             cost = (
    #                 self.cost_lambda_iou * (1.0 - iou)
    #                 + self.cost_lambda_mahalanobis * d2
    #             )
    #             cost_matrix[ti, di] = cost

    #     row_ind, col_ind = linear_sum_assignment(cost_matrix)

    #     matches = []
    #     matched_tracks = set()
    #     matched_detections = set()

    #     for ti, di in zip(row_ind, col_ind):
    #         if cost_matrix[ti, di] >= self.invalid_match_cost:
    #             continue

    #         matches.append((ti, di))
    #         matched_tracks.add(ti)
    #         matched_detections.add(di)

    #     unmatched_tracks = [i for i in range(num_tracks) if i not in matched_tracks]
    #     unmatched_detections = [i for i in range(num_dets) if i not in matched_detections]

    #     return matches, unmatched_tracks, unmatched_detections
    
    def _mahalanobis_squared(self, track: Track, z: np.ndarray, R: np.ndarray) -> float:
        z_pred = self.H @ track.x
        y = z - z_pred
        S = self.H @ track.P @ self.H.T + R

        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)

        d2 = float(y.T @ S_inv @ y)
        return d2

    def _build_roi(self, track: Track, frame_w: int, frame_h: int) -> List[int]:
        cx, cy, w, h = track.x[0], track.x[1], track.x[2], track.x[3]

        sigma_cx = math.sqrt(max(track.P[0, 0], 0.0))
        sigma_cy = math.sqrt(max(track.P[1, 1], 0.0))
        sigma_w = math.sqrt(max(track.P[2, 2], 0.0))
        sigma_h = math.sqrt(max(track.P[3, 3], 0.0))

        conf_term_x = self.confidence_roi_scale * (1.0 - track.last_confidence) * w
        conf_term_y = self.confidence_roi_scale * (1.0 - track.last_confidence) * h

        pad_x = max(
            self.pmin,
            self.uncertainty_scale_pos * sigma_cx + self.uncertainty_scale_size * sigma_w + conf_term_x
        )
        pad_y = max(
            self.pmin,
            self.uncertainty_scale_pos * sigma_cy + self.uncertainty_scale_size * sigma_h + conf_term_y
        )

        x1 = cx - w / 2.0 - pad_x
        y1 = cy - h / 2.0 - pad_y
        x2 = cx + w / 2.0 + pad_x
        y2 = cy + h / 2.0 + pad_y

        x1 = int(max(0, math.floor(x1)))
        y1 = int(max(0, math.floor(y1)))
        x2 = int(min(frame_w - 1, math.ceil(x2)))
        y2 = int(min(frame_h - 1, math.ceil(y2)))

        if x2 <= x1:
            x2 = min(frame_w - 1, x1 + 1)
        if y2 <= y1:
            y2 = min(frame_h - 1, y1 + 1)

        return [x1, y1, x2, y2]

    def _export_tracks(self) -> List[Dict[str, Any]]:
        exported = []
        for track in self.tracks:
            exported.append({
                "track_id": track.track_id,
                "class": track.cls,
                "bbox": self._state_to_bbox(track.x),
                "state": track.x.copy(),
                "covariance": track.P.copy(),
                "confidence": track.last_confidence,
                "age": track.age,
                "hits": track.hits,
                "misses": track.misses,
                "roi": track.roi,
            })
        return exported

    @staticmethod
    def _bbox_to_measurement(bbox: List[float]) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = bbox
        w = max(float(x2 - x1), 1.0)
        h = max(float(y2 - y1), 1.0)
        cx = float(x1 + x2) / 2.0
        cy = float(y1 + y2) / 2.0
        return cx, cy, w, h

    @staticmethod
    def _state_to_bbox(x: np.ndarray) -> List[float]:
        cx, cy, w, h = x[0], x[1], x[2], x[3]
        w = max(float(w), 1.0)
        h = max(float(h), 1.0)

        x1 = float(cx - w / 2.0)
        y1 = float(cy - h / 2.0)
        x2 = float(cx + w / 2.0)
        y2 = float(cy + h / 2.0)

        return [x1, y1, x2, y2]

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