import copy
import os
import unittest
import torch
from unittest.mock import patch

from tools.train_mixed_sampling import reset_checkpoint_for_stage, save_validation_checkpoints


class MixedCheckpointTests(unittest.TestCase):
    def test_stage_transition_resets_state_and_uses_stage_schedule(self):
        for source_stage, target_stage in [(1, 2), (1, 3), (2, 3), (None, 2)]:
            with self.subTest(source_stage=source_stage, target_stage=target_stage):
                model = torch.nn.Linear(1, 1)
                optimizer = torch.optim.SGD(model.parameters(), lr=.00025, momentum=.9)
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [5, 15], gamma=.5)
                original = {'model': model.state_dict(), 'optimizer': {'stale': True},
                            'scheduler': {'stale': True}, 'epoch': 99, 'best_map': .9}
                if source_stage is not None:
                    original['stage'] = source_stage
                result = reset_checkpoint_for_stage(original, target_stage, optimizer, scheduler)
                self.assertIs(result['model'], original['model'])
                self.assertEqual(result['epoch'], -1)
                self.assertEqual(result['stage'], target_stage)
                self.assertNotIn('best_map', result)
                optimizer.load_state_dict(result['optimizer'])
                scheduler.load_state_dict(result['scheduler'])
                self.assertEqual(optimizer.param_groups[0]['lr'], .00025)
                for _ in range(5):
                    optimizer.step()
                    scheduler.step()
                self.assertEqual(optimizer.param_groups[0]['lr'], .000125)

    def test_same_or_later_stage_preserves_resume_state(self):
        for source_stage in [2, 3]:
            checkpoint = {'stage': source_stage, 'epoch': 14}
            self.assertIs(reset_checkpoint_for_stage(checkpoint, 2, None, None), checkpoint)

    def test_model_only_checkpoint_starts_at_first_epoch(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=.00025)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [5, 15])
        weights = model.state_dict()
        result = reset_checkpoint_for_stage(weights, 2, optimizer, scheduler)
        self.assertIs(result['model'], weights)
        self.assertEqual(result['epoch'], -1)
        self.assertEqual(result['stage'], 2)

    def test_best_survives_worse_tied_and_nonfinite_epochs_and_resume(self):
        files = {}
        def save(value, path):
            files[os.path.basename(path)] = copy.deepcopy(value)
        best = float('-inf')
        with patch('tools.train_mixed_sampling.torch.save', side_effect=save):
            for epoch, score in enumerate([.7, .6, float('nan'), .7, float('inf')]):
                best = save_validation_checkpoints({'epoch': epoch, 'model': epoch}, 'unused', score, best)
            self.assertEqual(files['best_map.pt']['epoch'], 0)
            self.assertEqual(files['last.pt']['epoch'], 4)
            resumed_best = files['last.pt']['best_map']
            save_validation_checkpoints({'epoch': 5, 'model': 5}, 'unused', .75, resumed_best)
            self.assertEqual(files['best_map.pt']['epoch'], 5)
            self.assertEqual(files['last.pt']['best_map'], .75)


if __name__ == '__main__':
    unittest.main()
