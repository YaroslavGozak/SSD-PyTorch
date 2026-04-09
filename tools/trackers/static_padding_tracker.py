"""
StaticPaddingTracker: fixed-padding ROI generator with optional short-term memory.

For each detection in the current frame, the ROI for the next frame is the
detection bounding box expanded by a fixed number of pixels in every direction.
Optionally, the tracker can keep the previous frame detections alive for a few
frames when the detector produces no usable detections.
Interface matches KalmanRoiTracker.
"""
from typing import Any, Dict, List, Optional, Tuple


class StaticPaddingTracker:
    """
    Args:
        pad_x: Pixels added left and right, expressed in model tensor resolution.
        pad_y: Pixels added above and below, expressed in model tensor resolution.
        conf_det_min: Drop detections below this confidence.
        model_input_size: (height, width) of the model input tensor used to scale
            pad_x / pad_y to original image space.  If None, padding is applied
            directly in original image space (legacy behaviour).
        hold_last_for_frames: Number of consecutive miss frames for which last
            valid detections are returned. 0 disables temporal hold.
    """
    def __init__(
        self,
        pad_x: int = 50,
        pad_y: int = 50,
        conf_det_min: float = 0.5,
        model_input_size: Optional[Tuple[int, int]] = None,
        hold_last_for_frames: int = 0,
    ):
        self.pad_x = pad_x
        self.pad_y = pad_y
        self.conf_det_min = conf_det_min
        self.model_input_size = model_input_size  # (tensor_h, tensor_w)
        self.hold_last_for_frames = max(0, int(hold_last_for_frames))
        self._next_id = 1
        self._active_tracks: List[Dict[str, Any]] = []

    def _track_to_roi(self, track: Dict[str, Any]) -> Dict[str, Any]:
        x1, y1, x2, y2 = [float(v) for v in track["bbox"]]
        return {
            "track_id": track["track_id"],
            "class": track["class"],
            "roi": list(track["roi"]),
            "bbox_pred": [x1, y1, x2, y2],
            "confidence": track["confidence"],
            "misses": track["misses"],
            "hits": track["hits"],
        }

    def update(self, detections: List[Dict[str, Any]], frame_shape: Tuple[int, int]) -> Dict[str, Any]:
        """
        detections: [{bbox: [x1,y1,x2,y2], class: str, confidence: float}, ...]
        frame_shape: (height, width)
        Returns: {tracks: [...], rois: [...]}
        """
        frame_h, frame_w = frame_shape

        if self.model_input_size is not None:
            tensor_h, tensor_w = self.model_input_size
            pad_x = self.pad_x * (frame_w / tensor_w)
            pad_y = self.pad_y * (frame_h / tensor_h)
        else:
            pad_x = float(self.pad_x)
            pad_y = float(self.pad_y)

        tracks: List[Dict[str, Any]] = []
        rois: List[Dict[str, Any]] = []

        for det in detections:
            conf = float(det.get("confidence", 1.0))
            if conf < self.conf_det_min:
                continue
            x1, y1, x2, y2 = [float(v) for v in det["bbox"]]
            cls = det["class"]

            rx1 = int(max(0,           x1 - pad_x))
            ry1 = int(max(0,           y1 - pad_y))
            rx2 = int(min(frame_w - 1, x2 + pad_x))
            ry2 = int(min(frame_h - 1, y2 + pad_y))
            if rx2 <= rx1 or ry2 <= ry1:
                continue

            roi = [rx1, ry1, rx2, ry2]
            tid = self._next_id; self._next_id += 1

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

        if tracks:
            self._active_tracks = tracks
            rois = [self._track_to_roi(track) for track in tracks]
            return {"tracks": tracks, "rois": rois}

        if self.hold_last_for_frames <= 0 or not self._active_tracks:
            self._active_tracks = []
            return {"tracks": [], "rois": []}

        carried_tracks: List[Dict[str, Any]] = []
        for track in self._active_tracks:
            misses = int(track.get("misses", 0)) + 1
            if misses > self.hold_last_for_frames:
                continue
            carried_tracks.append(
                {
                    "track_id": track["track_id"],
                    "class": track["class"],
                    "bbox": [float(v) for v in track["bbox"]],
                    "confidence": float(track.get("confidence", 1.0)),
                    "age": int(track.get("age", 1)) + 1,
                    "hits": int(track.get("hits", 1)),
                    "misses": misses,
                    "roi": list(track["roi"]),
                }
            )

        self._active_tracks = carried_tracks
        rois = [self._track_to_roi(track) for track in carried_tracks]

        return {"tracks": carried_tracks, "rois": rois}

    def reset(self) -> None:
        self._next_id = 1
        self._active_tracks = []