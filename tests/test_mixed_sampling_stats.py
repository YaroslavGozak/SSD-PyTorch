import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from tools.train_samplers.mixed_sampling_stats import MixedSamplingStats, save_epoch_stats
from tools.train_samplers.roi_mixed_sampling import BatchModeConfig, MixedCollateFn, RoiBatchProcessor


class MixedSamplingStatsTests(unittest.TestCase):
    def test_epoch_summary_records_size_use_and_enlargement(self):
        modes = [
            BatchModeConfig('full', 320, .3, (0, 0)),
            BatchModeConfig('small', 96, .5, (0, 0)),
            BatchModeConfig('medium', 160, .2, (0, 0)),
        ]
        stats = MixedSamplingStats(modes, image_size=320)
        for mode, actual, samples in [
            ('full', 320, 4), ('small', 96, 4), ('small', 128, 4),
            ('small', 320, 4), ('medium', 160, 2), ('medium', 200, 2),
        ]:
            stats.record(mode, actual, samples, object_mismatch=(mode == 'small' and actual == 320))
        rows = {row['mode']: row for row in stats.rows(epoch=3, stage=2)}

        self.assertEqual(rows['ALL']['batches'], 6)
        self.assertEqual(rows['ALL']['enlarged_batches'], 3)
        self.assertEqual(rows['ALL']['enlargement_rate_pct'], 50)
        self.assertEqual(rows['ALL']['enlarged_to_full_frame_batches'], 1)
        self.assertEqual(rows['ALL']['object_mismatch_batches'], 1)
        self.assertEqual(rows['ALL']['padding_or_jitter_only_batches'], 2)
        self.assertEqual(rows['ALL']['total_extra_px'], 296)
        self.assertEqual(rows['small']['batches'], 3)
        self.assertEqual(rows['small']['batch_share_pct'], 50)
        self.assertEqual(rows['small']['configured_probability_pct'], 50)
        self.assertEqual(rows['small']['enlarged_batches'], 2)
        self.assertEqual(rows['small']['exact_size_batches'], 1)
        self.assertEqual(rows['small']['enlargement_rank'], 1)
        self.assertEqual(rows['small']['object_mismatch_rank'], 1)
        self.assertEqual(rows['small']['total_extra_px'], 256)
        self.assertEqual(rows['medium']['enlargement_rank'], 2)
        self.assertEqual(json.loads(rows['small']['actual_size_counts']),
                         {'96': 1, '128': 1, '320': 1})

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'mixed_sampling_stats.csv'
            save_epoch_stats(path, list(rows.values()))
            save_epoch_stats(path, list(rows.values()))
            with path.open(newline='', encoding='utf-8') as handle:
                saved = list(csv.DictReader(handle))
            self.assertEqual(len(saved), 4)
            self.assertEqual({row['epoch'] for row in saved}, {'3'})
            self.assertEqual(saved[0]['mode'], 'ALL')

    def test_rejects_actual_size_below_requested_size(self):
        stats = MixedSamplingStats([BatchModeConfig('roi', 96, 1, (0, 0))], 320)
        with self.assertRaisesRegex(ValueError, 'outside'):
            stats.record('roi', 64, 4)

    def test_collated_batch_size_is_the_recorded_actual_size(self):
        modes = [BatchModeConfig('full', 320, .5, (0, 0)),
                 BatchModeConfig('roi', 96, .5, (0, 0))]
        batch = [{'image': torch.zeros(3, 240, 400), 'target': {
            'boxes': torch.tensor([[.25, .25, .75, .75]]), 'labels': torch.tensor([1])},
            'mode': 'roi', 'out_size': 96}]
        collate = MixedCollateFn(RoiBatchProcessor(mode_configs=modes),
                                 return_mode=True, return_sampling_info=True)
        with patch('tools.train_samplers.roi_mixed_sampling.choose_reference_box', return_value=0):
            images, targets, mode, sampling_info = collate(batch)
        stats = MixedSamplingStats(modes, 320)
        stats.record(mode, images.shape[-1], len(targets), sampling_info['object_mismatch'])
        roi = next(row for row in stats.rows(1, 2) if row['mode'] == 'roi')
        self.assertEqual(tuple(images.shape), (1, 3, 160, 160))
        self.assertEqual(roi['enlarged_batches'], 1)
        self.assertEqual(roi['object_mismatch_batches'], 1)
        self.assertEqual(json.loads(roi['actual_size_counts']), {'160': 1})

    def test_padding_only_enlargement_is_separate_from_object_mismatch(self):
        modes = [BatchModeConfig('full', 320, .5, (0, 0)),
                 BatchModeConfig('roi', 96, .5, (20, 20))]
        sample = {'image': torch.zeros(3, 320, 320), 'target': {
            'boxes': torch.tensor([[.375, .375, .625, .625]]), 'labels': torch.tensor([1])},
            'mode': 'roi', 'out_size': 96}
        collate = MixedCollateFn(RoiBatchProcessor(mode_configs=modes),
                                 return_mode=True, return_sampling_info=True)
        with patch('tools.train_samplers.roi_mixed_sampling.choose_reference_box', return_value=0):
            images, targets, mode, info = collate([sample])
        stats = MixedSamplingStats(modes, 320)
        stats.record(mode, images.shape[-1], len(targets), info['object_mismatch'])
        roi = next(row for row in stats.rows(1, 2) if row['mode'] == 'roi')
        self.assertEqual(tuple(images.shape), (1, 3, 120, 120))
        self.assertEqual(roi['object_mismatch_batches'], 0)
        self.assertEqual(roi['padding_or_jitter_only_batches'], 1)


if __name__ == '__main__':
    unittest.main()
