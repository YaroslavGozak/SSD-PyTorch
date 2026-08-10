import os
import tempfile
import unittest

import torch

from dataset.imagenet_vid import load_images_and_anns_imagenet_vid
from dataset.helpers.label_spaces import IMAGENET_VID_CLASSES, IMAGENET_VID_VOC_OVERLAP_CLASSES, build_label_maps
from tools.helpers.label_compat import VOC_LABEL2IDX
from model.model_adapters import DetectionLabelRemapAdapter


def _write_xml(xml_path, classes):
    objects = []
    for class_name in classes:
        objects.append(
            f"""
    <object>
        <class>{class_name}</class>
        <bndbox>
            <xmin>1</xmin>
            <ymin>2</ymin>
            <xmax>10</xmax>
            <ymax>12</ymax>
        </bndbox>
    </object>"""
        )
    xml_text = f"""<annotation>
    <size>
        <width>20</width>
        <height>30</height>
    </size>
    {''.join(objects)}
</annotation>"""
    with open(xml_path, 'w', encoding='utf-8') as handle:
        handle.write(xml_text)


class DummyDetectionModel:
    def __call__(self, _images, *_args, **_kwargs):
        return None, [{
            'boxes': torch.tensor([[0.1, 0.2, 0.4, 0.5], [0.2, 0.3, 0.6, 0.7]], dtype=torch.float32),
            'labels': torch.tensor([VOC_LABEL2IDX['person'], VOC_LABEL2IDX['tvmonitor']], dtype=torch.int64),
            'scores': torch.tensor([0.9, 0.8], dtype=torch.float32),
        }]

    def parameters(self):
        return iter([torch.zeros(1)])


class LabelCompatTests(unittest.TestCase):
    def test_imagenet_vid_loader_filters_to_voc_overlap_and_preserves_original_frame_idx(self):
        label2idx, _ = build_label_maps(IMAGENET_VID_CLASSES)

        with tempfile.TemporaryDirectory() as tmp_dir:
            data_root = os.path.join(tmp_dir, 'data')
            ann_root = os.path.join(tmp_dir, 'ann')
            video_dir = os.path.join(data_root, 'video_001')
            ann_video_dir = os.path.join(ann_root, 'video_001')
            os.makedirs(video_dir)
            os.makedirs(ann_video_dir)

            frame_specs = [
                ('000000.JPEG', ['zebra']),
                ('000001.JPEG', ['person', 'zebra']),
                ('000002.JPEG', ['dog']),
            ]
            for frame_name, classes in frame_specs:
                frame_stem = os.path.splitext(frame_name)[0]
                with open(os.path.join(video_dir, frame_name), 'wb') as handle:
                    handle.write(b'')
                _write_xml(os.path.join(ann_video_dir, f'{frame_stem}.xml'), classes)

            frames = load_images_and_anns_imagenet_vid(
                data_root=data_root,
                ann_root=ann_root,
                label2idx=label2idx,
                allowed_class_names=set(IMAGENET_VID_VOC_OVERLAP_CLASSES),
            )

        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[0]['frame_idx'], 1)
        self.assertTrue(frames[0]['is_first_frame'])
        self.assertEqual(len(frames[0]['detections']), 1)
        self.assertEqual(frames[0]['detections'][0]['label'], label2idx['person'])
        self.assertEqual(frames[1]['frame_idx'], 2)
        self.assertFalse(frames[1]['is_first_frame'])
        self.assertEqual(frames[1]['detections'][0]['label'], label2idx['dog'])

    def test_detection_label_remap_adapter_drops_unmapped_predictions(self):
        target_label2idx, _ = build_label_maps(IMAGENET_VID_CLASSES)
        source_idx2label = {idx: label for label, idx in VOC_LABEL2IDX.items()}
        adapter = DetectionLabelRemapAdapter(
            base_model=DummyDetectionModel(),
            source_idx2label=source_idx2label,
            target_label2idx=target_label2idx,
            class_name_mapping={
                'person': 'person',
            },
        )

        _, detections = adapter(torch.zeros((1, 3, 32, 32), dtype=torch.float32))

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]['labels'].tolist(), [target_label2idx['person']])
        self.assertAlmostEqual(detections[0]['scores'].tolist()[0], 0.9, places=6)
        self.assertEqual(detections[0]['boxes'].shape[0], 1)


if __name__ == '__main__':
    unittest.main()