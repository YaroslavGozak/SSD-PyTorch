"""
StaticPaddingTracker: simplest possible ROI generator.

For each detection in the current frame, the ROI for the next frame is the
detection bounding box expanded by a fixed number of pixels in every direction.
No temporal state between calls — suitable as baseline B2.
Interface matches KalmanRoiTracker.
"""
from typing import Any, Dict, List, Tuple


class StaticPaddingTracker:
    """
    Args:
        pad_x: Pixels added left and right.
        pad_y: Pixels added above and below.
        conf_det_min: Drop detections below this confidence.
    """
    def __init__(self, pad_x: int = 50, pad_y: int = 50, conf_det_min: float = 0.5):
        self.pad_x = pad_x
        self.pad_y = pad_y
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

            rx1 = int(max(0,          x1 - self.pad_x))
            ry1 = int(max(0,          y1 - self.pad_y))
            rx2 = int(min(frame_w - 1, x2 + self.pad_x))
            ry2 = int(min(frame_h - 1, y2 + self.pad_y))
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