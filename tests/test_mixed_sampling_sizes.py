import unittest
from unittest.mock import patch

import torch

from tools.train_samplers.roi_mixed_sampling import (
    BatchModeConfig, build_stage_mode_configs, MixedBatchSampler,
    MixedCollateFn, RoiBatchProcessor,
)


class MixedSamplingSizeTests(unittest.TestCase):
    def test_original_sizes_and_scaled_schedules(self):
        for stage, original in ((2, [300, 224, 160, 96, 64]),
                                (3, [300, 224, 160, 96, 64, 32])):
            with self.subTest(stage=stage):
                self.assertEqual([m.out_size for m in build_stage_mode_configs(stage)], original)
                self.assertEqual([m.out_size for m in build_stage_mode_configs(stage, 600)],
                                 [2*s for s in original])
                expected = [320, 239, 171, 102, 68] + ([34] if stage == 3 else [])
                self.assertEqual([m.out_size for m in build_stage_mode_configs(stage, 320)], expected)

    def test_invalid_sizes(self):
        for size in (0, -1, True, 320.5, "320", [320, 320], None):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "positive integer"):
                build_stage_mode_configs(2, size)

    def test_sampler_sizes_reach_processed_tensors(self):
        image = torch.zeros(3, 80, 120)
        target = {"boxes": torch.tensor([[.2, .2, .8, .8]]), "labels": torch.tensor([1])}
        for stage in (2, 3):
            sampler = MixedBatchSampler(range(2), batch_size=2, stage=stage, im_size=320)
            processor = RoiBatchProcessor(mode_configs=sampler.mode_configs)
            for mode in sampler.mode_configs:
                with self.subTest(stage=stage, mode=mode.name), \
                     patch("random.Random.choices", return_value=[mode]):
                    batch = next(iter(sampler))
                    self.assertTrue(all(item[2] == mode.out_size for item in batch))
                    _, name, size = batch[0]
                    result, annotations = processor.process_sample(image, target, name, size)
                    side = result.shape[-1]
                    self.assertEqual(tuple(result.shape), (3, side, side))
                    self.assertGreaterEqual(side, size)
                    self.assertLessEqual(side, 320)
                    self.assertTrue(torch.isfinite(annotations["boxes"]).all())
                    self.assertTrue(((annotations["boxes"] >= 0) & (annotations["boxes"] <= 1)).all())
                    if name != 'full':
                        box = annotations['boxes'][0] * side
                        self.assertAlmostEqual((box[2] - box[0]).item(), 192, places=3)
                        self.assertAlmostEqual((box[3] - box[1]).item(), 192, places=3)

    def test_batch_grows_to_fit_largest_reference_without_rescaling(self):
        mode = BatchModeConfig('roi', 96, 1.0, (0, 0))
        processor = RoiBatchProcessor(mode_configs=[BatchModeConfig('full', 320, 0, (0, 0)), mode])
        collate = MixedCollateFn(processor)
        batch = [
            {'image': torch.zeros(3, 160, 320), 'target': {
                'boxes': torch.tensor([[.4, .4, .6, .6]]), 'labels': torch.tensor([1])},
             'mode': 'roi', 'out_size': 96},
            {'image': torch.zeros(3, 320, 160), 'target': {
                'boxes': torch.tensor([[.25, .25, .75, .75]]), 'labels': torch.tensor([2])},
             'mode': 'roi', 'out_size': 96},
        ]
        with patch('tools.train_samplers.roi_mixed_sampling.choose_reference_box', return_value=0):
            images, targets = collate(batch)
        self.assertEqual(tuple(images.shape), (2, 3, 160, 160))
        for target, expected in zip(targets, (64, 160)):
            box = target['boxes'][0] * 160
            self.assertAlmostEqual((box[2] - box[0]).item(), expected, places=3)
            self.assertAlmostEqual((box[3] - box[1]).item(), expected, places=3)

    def test_touched_objects_use_visibility_threshold_without_growing_crop(self):
        mode = BatchModeConfig('roi', 96, 1.0, (0, 0), min_box_visibility=.5)
        processor = RoiBatchProcessor(mode_configs=[BatchModeConfig('full', 320, 0, (0, 0)), mode])
        boxes = torch.tensor([[100, 100, 140, 140], [160, 100, 200, 140],
                              [150, 100, 170, 140]], dtype=torch.float32) / 320
        target = {'boxes': boxes, 'labels': torch.tensor([1, 2, 3])}
        with patch('tools.train_samplers.roi_mixed_sampling.choose_reference_box', return_value=0):
            image, output = processor.process_sample(torch.zeros(3, 320, 320), target, 'roi', 96)
        self.assertEqual(tuple(image.shape), (3, 96, 96))
        self.assertEqual(output['labels'].tolist(), [1, 3])
        self.assertAlmostEqual((output['boxes'][1, 2] - output['boxes'][1, 0]).item() * 96,
                               18, places=3)

    def test_processor_uses_only_selected_stage(self):
        for stage, padding, jitter in ((2, (70, 120), .05), (3, (30, 80), .10)):
            modes = build_stage_mode_configs(stage, 320)
            processor = RoiBatchProcessor(mode_configs=modes)
            self.assertEqual(processor.mode_cfg_map['large_roi'].padding_px_range, padding)
            self.assertEqual(processor.mode_cfg_map['large_roi'].center_jitter_ratio, jitter)
            for mode in modes:
                self.assertIs(processor.mode_cfg_map[mode.name], mode)


if __name__ == "__main__":
    unittest.main()
