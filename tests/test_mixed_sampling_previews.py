import unittest
from unittest.mock import patch

import numpy as np
import torch

from tools.infer import prepare_mixed_sample
from tools.train_samplers.roi_mixed_sampling import build_stage_mode_configs


class MixedSamplingPreviewTests(unittest.TestCase):
    def test_crops_use_matching_pixels_and_coordinates(self):
        image = np.full((240, 480, 3), [10, 80, 200], dtype=np.uint8)
        detections = [{'bbox': [220, 110, 260, 130], 'label': 2}]
        for stage in (2, 3):
            for mode in build_stage_mode_configs(stage, 320):
                if mode.name == 'full':
                    continue
                with self.subTest(stage=stage, mode=mode.name), patch(
                    'tools.infer.random.choices', return_value=[mode],
                ) as choose:
                    tensor, target, preview = prepare_mixed_sample(image, detections, stage, 320)
                    self.assertTrue(all(m.name != 'full' for m in choose.call_args.args[0]))
                    side = tensor.shape[-1]
                    self.assertLess(side, 320)
                    self.assertGreaterEqual(side, mode.out_size)
                    self.assertEqual(preview.shape, (side, side, 3))
                    np.testing.assert_array_equal(preview[0, 0], image[0, 0])
                    mean = torch.tensor([.485, .456, .406]).view(3, 1, 1)
                    std = torch.tensor([.229, .224, .225]).view(3, 1, 1)
                    pixels = (tensor * std + mean).permute(1, 2, 0).mul(255).round().byte().numpy()
                    np.testing.assert_array_equal(pixels[:, :, ::-1], preview)
                    self.assertEqual(target['labels'].tolist(), [2])
                    box = target['bboxes'][0] * side
                    torch.testing.assert_close(box[2:] - box[:2], torch.tensor([40 / 480 * 320, 20 / 240 * 320]))
                    self.assertTrue(((target['bboxes'] >= 0) & (target['bboxes'] <= 1)).all())

    def test_empty_annotations(self):
        mode = build_stage_mode_configs(3, 320)[-1]
        with patch('tools.infer.random.choices', return_value=[mode]):
            tensor, target, preview = prepare_mixed_sample(np.zeros((240, 480, 3), dtype=np.uint8), [], 3, 320)
        self.assertEqual(tensor.shape, (3, 34, 34))
        self.assertEqual(preview.shape, (34, 34, 3))
        self.assertEqual(target['bboxes'].shape, (0, 4))


if __name__ == '__main__':
    unittest.main()
