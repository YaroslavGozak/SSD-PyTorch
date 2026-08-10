from typing import Dict

from dataset.helpers.label_spaces import COCO_CLASSES, COCO_TO_VID
import torch


class YoloV8Adapter:
    """YOLO wrapper that returns normalized detections in project format."""

    def __init__(self, weights_path: str, device: torch.device, use_predict_api: bool = True):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "YOLO inference requires ultralytics. Install with: pip install ultralytics"
            ) from e
        self._yolo = YOLO(weights_path)
        self.device = device
        self.use_predict_api = use_predict_api
        print(f"[YOLO] Using {'predict API' if use_predict_api else 'raw model output'} mode for inference")

    def to(self, device: torch.device = None, **_kwargs):
        if device is not None:
            self.device = device
        return self

    def eval(self):
        return self

    def parameters(self):
        return self._yolo.model.parameters()

    def __call__(self, images: torch.Tensor, _targets=None):
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=images.device).view(1, 3, 1, 1)
        images_01 = (images.float() * std + mean).clamp(0.0, 1.0)

        device_arg = str(self.device) if self.device is not None else None

        if self.use_predict_api:
            results = self._yolo.predict(source=images_01, verbose=False, device=device_arg)

            _, _, h, w = images.shape
            out = []
            for res in results:
                if res.boxes is None or len(res.boxes) == 0:
                    out.append({
                        'boxes': torch.empty((0, 4), dtype=torch.float32, device=images.device),
                        'labels': torch.empty((0,), dtype=torch.int64, device=images.device),
                        'scores': torch.empty((0,), dtype=torch.float32, device=images.device),
                    })
                    continue

                xyxy = res.boxes.xyxy.to(images.device).float()
                boxes = xyxy.clone()
                boxes[:, [0, 2]] /= float(w)
                boxes[:, [1, 3]] /= float(h)

                out.append({
                    'boxes': boxes,
                    'labels': (res.boxes.cls.to(images.device).long() + 1),
                    'scores': res.boxes.conf.to(images.device).float(),
                })
            return None, out

        pred = self._yolo.model(images_01)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]

        if isinstance(pred, torch.Tensor) and pred.ndim == 3 and pred.shape[-1] == 6:
            batch = pred
        else:
            from ultralytics.utils.ops import non_max_suppression

            nms_list = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, max_det=300)
            batch = []
            for det in nms_list:
                if det is None or det.numel() == 0:
                    batch.append(torch.empty((0, 6), device=images.device))
                else:
                    batch.append(det[:, :6])
            batch = torch.stack([
                det if det.ndim == 2 else det.view(0, 6) for det in batch
            ], dim=0)

        _, _, h, w = images.shape
        out = []
        for det in batch:
            if det.numel() == 0:
                out.append({
                    'boxes': torch.empty((0, 4), dtype=torch.float32, device=images.device),
                    'labels': torch.empty((0,), dtype=torch.int64, device=images.device),
                    'scores': torch.empty((0,), dtype=torch.float32, device=images.device),
                })
                continue

            xyxy = det[:, :4].to(images.device).float()
            conf = det[:, 4].to(images.device).float()
            cls = det[:, 5].to(images.device).long()

            boxes = xyxy.clone()
            boxes[:, [0, 2]] /= float(w)
            boxes[:, [1, 3]] /= float(h)

            out.append({
                'boxes': boxes,
                'labels': (cls + 1),
                'scores': conf,
            })

        return None, out


class DetectionLabelRemapAdapter:
    """Adapter that remaps model output labels into the target dataset label space."""

    def __init__(
        self,
        base_model,
        source_idx2label: Dict[int, str],
        target_label2idx: Dict[str, int],
        class_name_mapping: Dict[str, str],
    ):
        self.base_model = base_model
        self._label_id_map = {
            source_idx: target_label2idx[target_name]
            for source_idx, source_name in source_idx2label.items()
            for target_name in [class_name_mapping.get(source_name)]
            if target_name is not None and target_name in target_label2idx
        }

    def to(self, *args, **kwargs):
        if hasattr(self.base_model, 'to'):
            self.base_model.to(*args, **kwargs)
        return self

    def eval(self):
        if hasattr(self.base_model, 'eval'):
            self.base_model.eval()
        return self

    def train(self, mode: bool = True):
        if hasattr(self.base_model, 'train'):
            self.base_model.train(mode)
        return self

    def parameters(self):
        return self.base_model.parameters()

    def __getattr__(self, name):
        return getattr(self.base_model, name)

    def __call__(self, *args, **kwargs):
        output = self.base_model(*args, **kwargs)
        if isinstance(output, tuple) and len(output) == 2:
            raw, detections = output
            print(f"[DetectionLabelRemapAdapter] Raw detections: {detections}")
            remapped = self._remap_output(detections)
            print(f"[DetectionLabelRemapAdapter] Remapped detections: {remapped}")
            return raw, remapped
        return self._remap_output(output)

    def _remap_output(self, detections):
        if isinstance(detections, dict):
            if self._is_detection_dict(detections):
                return self._remap_detection_dict(detections)
            return detections
        if isinstance(detections, list):
            return [self._remap_detection_dict(det) if self._is_detection_dict(det) else det for det in detections]
        return detections

    @staticmethod
    def _is_detection_dict(detections) -> bool:
        return isinstance(detections, dict) and {'boxes', 'labels', 'scores'}.issubset(detections.keys())

    def _remap_detection_dict(self, detections: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        labels = detections['labels']
        if labels.numel() == 0:
            return detections

        keep = torch.zeros(labels.shape, dtype=torch.bool, device=labels.device)
        mapped_labels = torch.zeros(labels.shape, dtype=torch.int64, device=labels.device)
        for source_label_id, target_label_id in self._label_id_map.items():
            matched = labels == int(source_label_id)
            keep |= matched
            mapped_labels[matched] = int(target_label_id)

        remapped = dict(detections)
        remapped['boxes'] = detections['boxes'][keep]
        remapped['scores'] = detections['scores'][keep]
        remapped['labels'] = mapped_labels[keep]
        return remapped


def unwrap_model(model):
    while isinstance(model, DetectionLabelRemapAdapter):
        model = model.base_model
    return model


