import unittest

import torch
from torchvision import tv_tensors

from dataset.transforms.fixed_padding_roi_crop_yolo_test_transform import FixedPaddingRoiCropYOLOTestTransform


class YOLORectangularPaddingTests(unittest.TestCase):
    def target(self, boxes):
        return {'boxes': tv_tensors.BoundingBoxes(boxes, format='XYXY', canvas_size=(320, 320)),
                'labels': torch.arange(1, len(boxes) + 1), 'difficult': torch.zeros(len(boxes), dtype=torch.long)}

    def test_wide_roi_is_not_expanded_to_square_and_padding_offsets_boxes(self):
        transform = FixedPaddingRoiCropYOLOTestTransform(320, [], [], 0, 0)
        image = torch.full((3, 320, 320), 200, dtype=torch.uint8)
        output, target = transform.transforms['test'](image, self.target([[20, 30, 90, 60], [1, 1, 10, 10]]))
        self.assertEqual(tuple(output.shape[-2:]), (32, 96))
        torch.testing.assert_close(torch.as_tensor(target['boxes']), torch.tensor([[13., 1., 83., 31.]]))
        self.assertEqual(target['labels'].tolist(), [1])
        torch.testing.assert_close(output[:, 1:31, 13:83], torch.full((3, 30, 70), 200 / 255))
        torch.testing.assert_close(output[:, 0, 0], torch.full((3,), 114 / 255))

    def test_tall_border_roi_clips_then_aligns_each_side(self):
        transform = FixedPaddingRoiCropYOLOTestTransform(320, [], [], 10, 10)
        image, target = transform.transforms['test'](torch.zeros(3, 320, 320, dtype=torch.uint8),
                                                     self.target([[0, 0, 61, 113]]))
        self.assertEqual(tuple(image.shape[-2:]), (128, 96))
        torch.testing.assert_close(torch.as_tensor(target['boxes']), torch.tensor([[12., 2., 73., 115.]]))

    def test_aligned_square_is_preserved(self):
        transform = FixedPaddingRoiCropYOLOTestTransform(320, [], [], 0, 0)
        image, _ = transform.transforms['test'](torch.zeros(3, 320, 320, dtype=torch.uint8),
                                               self.target([[32, 32, 96, 96]]))
        self.assertEqual(tuple(image.shape[-2:]), (64, 64))


if __name__ == '__main__':
    unittest.main()
