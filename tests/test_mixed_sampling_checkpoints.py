import copy
import os
import unittest
from unittest.mock import patch

from tools.train_mixed_sampling import save_validation_checkpoints


class MixedCheckpointTests(unittest.TestCase):
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
