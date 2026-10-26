from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from model.model_adapters import YoloV8Adapter, DetectionLabelRemapAdapter
from tools import infer


class YoloAdapterTests(unittest.TestCase):
    def adapter(self, normalization='none', raw=False):
        yolo = Mock()
        boxes = SimpleNamespace(
            xyxy=torch.tensor([[8., 4., 24., 12.]]),
            cls=torch.tensor([0.]), conf=torch.tensor([.8]),
        )
        boxes_type = type('Boxes', (), {'__len__': lambda self: 1})
        result_boxes = boxes_type()
        result_boxes.__dict__.update(boxes.__dict__)
        yolo.predict.return_value = [SimpleNamespace(boxes=result_boxes)]
        with patch('ultralytics.YOLO', return_value=yolo):
            adapter = YoloV8Adapter('unused.pt', torch.device('cpu'),
                                    use_predict_api=not raw,
                                    input_normalization=normalization,
                                    conf_threshold=.4, iou_threshold=.3, max_detections=17)
        return adapter, yolo

    def test_unnormalized_pixels_and_prediction_options(self):
        adapter, yolo = self.adapter()
        x = torch.linspace(0, 1, 3*16*32).reshape(1, 3, 16, 32)
        _, detections = adapter(x)
        kwargs = yolo.predict.call_args.kwargs
        torch.testing.assert_close(kwargs['source'], x)
        self.assertEqual((kwargs['conf'], kwargs['iou'], kwargs['max_det']), (.4, .3, 17))
        torch.testing.assert_close(detections[0]['boxes'], torch.tensor([[.25, .25, .75, .75]]))
        self.assertEqual(detections[0]['labels'].tolist(), [1])

    def test_imagenet_input_is_reversed_once(self):
        adapter, yolo = self.adapter('imagenet')
        x = torch.rand(1, 3, 16, 32)
        mean = x.new_tensor([.485, .456, .406]).view(1, 3, 1, 1)
        std = x.new_tensor([.229, .224, .225]).view(1, 3, 1, 1)
        adapter((x-mean)/std, conf_threshold=.7, iou_threshold=.6, max_detections=5)
        kwargs = yolo.predict.call_args.kwargs
        torch.testing.assert_close(kwargs['source'], x)
        self.assertEqual((kwargs['conf'], kwargs['iou'], kwargs['max_det']), (.7, .6, 5))

    def test_confidence_forwarded_through_label_remapping(self):
        adapter, yolo = self.adapter()
        wrapped = DetectionLabelRemapAdapter(adapter, {1: 'car'}, {'car': 1}, {'car': 'car'})
        infer.run_detector(wrapped, torch.zeros(1, 3, 16, 32), None,
                           model_name='yolo', conf_threshold=.61)
        self.assertEqual(yolo.predict.call_args.kwargs['conf'], .61)

    def test_roi_transform_selects_pixel_contract(self):
        for transform, expected in [('fixed_size_yolo_96x96', 'none'),
                                    ('fixed_padding_roi_crop_yolo_0', 'none'), ('ssd', 'imagenet')]:
            adapter, _ = self.adapter('imagenet')
            config = {'train_params': {}, 'dataset_params': {'transform_name': 'ssd'}}
            with patch.object(infer, 'pipeline_load_model_and_dataset',
                              return_value=(adapter, None, None, config)):
                infer.load_model_and_dataset(SimpleNamespace(), transform_name=transform)
            self.assertEqual(adapter.input_normalization, expected)

    def test_raw_end_to_end_filters_each_batch_element(self):
        adapter, yolo = self.adapter(raw=True)
        yolo.model.return_value = torch.tensor([
            [[0., 0., 4., 4., .8, 0.], [0., 0., 4., 4., .2, 0.]],
            [[0., 0., 4., 4., .1, 0.], [0., 0., 4., 4., .2, 0.]],
        ])
        _, detections = adapter(torch.zeros(2, 3, 16, 32), conf_threshold=.5)
        self.assertEqual([len(d['scores']) for d in detections], [1, 0])


if __name__ == '__main__':
    unittest.main()
