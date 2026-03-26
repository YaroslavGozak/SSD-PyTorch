"""
RelativeObjectSizePaddingTracker: object-proportional ROI padding.

Padding = fraction of the detected object's own width/height.
Larger objects get larger absolute search regions.
No temporal state — suitable as a baseline variant.
Interface matches KalmanRoiTracker.
"""
import math
from typing import Any, Dict, List, Tuple


class RelativeObjectSizePaddingTracker:
    """
    Args:
        pad_ratio_x: Fraction of object width added on each horizontal side.
        pad_ratio_y: Fraction of object height added on each vertical side.
        min_pad_x:   Floor on horizontal padding (prevents zero padding on tiny objects).
        min_pad_y:   Floor on vertical padding.
        conf_det_min: Drop detections below this confidence.
    """
    def __init__(
        self,
        pad_ratio_x: float = 0.5,
        pad_ratio_y: float = 0.5,
        min_pad_x: int = 10,
        min_pad_y: int = 10,
        conf_det_min: float = 0.0,
    ):
        self.pad_ratio_x = pad_ratio_x
        self.pad_ratio_y = pad_ratio_y
        self.min_pad_x = min_pad_x
        self.min_pad_y = min_pad_y
        self.conf_det_min = conf_det_min
        self._next_id = 1

    def update(self, detections: List[Dict[str, Any]], frame_shape: Tuple[int, int]) -> Dict[str, Any]:
        """
        detections: [{bbox: [x1,y1,x2,y2], class: str, confidence: float}, ...]
        frame_shape: (height, width)
        Returns: {tracks: [...], rois: [...]}
        """
        frame_h, frame_w = frame_shape
        tracks, rois = [], []

        for det in detections:
            conf = float(det.get("confidence", 1.0))
            if conf < self.conf_det_min:
                continue
            x1, y1, x2, y2 = [float(v) for v in det["bbox"]]
            cls = det["class"]

            obj_w = max(x2 - x1, 1.0)
            obj_h = max(y2 - y1, 1.0)
            pad_x = max(self.min_pad_x, math.ceil(self.pad_ratio_x * obj_w))
            pad_y = max(self.min_pad_y, math.ceil(self.pad_ratio_y * obj_h))

            rx1 = int(max(0,           x1 - pad_x))
            ry1 = int(max(0,           y1 - pad_y))
            rx2 = int(min(frame_w - 1, x2 + pad_x))
            ry2 = int(min(frame_h - 1, y2 + pad_y))
            if rx2 <= rx1 or ry2 <= ry1:
                continue

            roi = [rx1, ry1, rx2, ry2]
            tid = self._next_id; self._next_id += 1

            tracks.append({"track_id": tid, "class": cls, "bbox": [x1,y1,x2,y2],
                           "confidence": conf, "age": 1, "hits": 1, "misses": 0, "roi": roi})
            rois.append({"track_id": tid, "class": cls, "roi": roi,
                         "bbox_pred": [x1,y1,x2,y2], "confidence": conf, "misses": 0, "hits": 1})

        return {"tracks": tracks, "rois": rois}

    def reset(self) -> None:
        self._next_id = 1