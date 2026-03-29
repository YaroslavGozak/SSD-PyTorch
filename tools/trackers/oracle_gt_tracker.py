"""
OracleGtTracker: GT-driven ROI generator for upper-bound benchmarking.

This tracker ignores detector outputs when oracle detections are provided and
builds padded ROIs from ground-truth boxes for the same frame.
"""
from typing import Any, Dict, List, Optional, Tuple


class OracleGtTracker:
    """
    Args:
        pad_x: Pixels added left/right in model tensor resolution.
        pad_y: Pixels added top/bottom in model tensor resolution.
        conf_det_min: Drop oracle detections below this confidence.
        model_input_size: (height, width) used to scale padding to frame space.
    """

    def __init__(
        self,
        pad_x: int = 50,
        pad_y: int = 50,
        conf_det_min: float = 0.0,
        model_input_size: Optional[Tuple[int, int]] = None,
    ):
        self.pad_x = pad_x
        self.pad_y = pad_y
        self.conf_det_min = conf_det_min
        self.model_input_size = model_input_size
        self._next_id = 1
        self._oracle_detections: Optional[List[Dict[str, Any]]] = None

    def set_oracle_detections(self, detections: List[Dict[str, Any]]) -> None:
        """Set per-frame GT-like detections for oracle ROI generation."""
        self._oracle_detections = detections

    def preview_rois(self, frame_shape: Tuple[int, int]) -> List[List[int]]:
        """Compute ROIs for current oracle detections without mutating tracker state."""
        _, rois, _ = self._build_outputs(
            self._oracle_detections or [], frame_shape=frame_shape, assign_ids=False
        )
        return [r["roi"] for r in rois]

    def update(self, detections: List[Dict[str, Any]], frame_shape: Tuple[int, int]) -> Dict[str, Any]:
        """
        detections: ignored when oracle detections are set; otherwise used as fallback.
        frame_shape: (height, width)
        Returns: {tracks: [...], rois: [...]}.
        """
        source_dets = self._oracle_detections if self._oracle_detections is not None else detections
        tracks, rois, next_id = self._build_outputs(source_dets, frame_shape=frame_shape, assign_ids=True)
        self._next_id = next_id
        return {"tracks": tracks, "rois": rois}

    def reset(self) -> None:
        self._next_id = 1
        self._oracle_detections = None

    def _build_outputs(
        self,
        detections: List[Dict[str, Any]],
        frame_shape: Tuple[int, int],
        assign_ids: bool,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
        frame_h, frame_w = frame_shape
        if self.model_input_size is not None:
            tensor_h, tensor_w = self.model_input_size
            pad_x = self.pad_x * (frame_w / float(tensor_w))
            pad_y = self.pad_y * (frame_h / float(tensor_h))
        else:
            pad_x = float(self.pad_x)
            pad_y = float(self.pad_y)

        tracks: List[Dict[str, Any]] = []
        rois: List[Dict[str, Any]] = []
        next_id = self._next_id

        for det in detections:
            conf = float(det.get("confidence", 1.0))
            if conf < self.conf_det_min:
                continue
            x1, y1, x2, y2 = [float(v) for v in det["bbox"]]
            cls = det["class"]

            rx1 = int(max(0, x1 - pad_x))
            ry1 = int(max(0, y1 - pad_y))
            rx2 = int(min(frame_w - 1, x2 + pad_x))
            ry2 = int(min(frame_h - 1, y2 + pad_y))
            if rx2 <= rx1 or ry2 <= ry1:
                continue

            roi = [rx1, ry1, rx2, ry2]
            tid = next_id if assign_ids else -1
            if assign_ids:
                next_id += 1

            tracks.append(
                {
                    "track_id": tid,
                    "class": cls,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "age": 1,
                    "hits": 1,
                    "misses": 0,
                    "roi": roi,
                }
            )
            rois.append(
                {
                    "track_id": tid,
                    "class": cls,
                    "roi": roi,
                    "bbox_pred": [x1, y1, x2, y2],
                    "confidence": conf,
                    "misses": 0,
                    "hits": 1,
                }
            )

        return tracks, rois, next_id
