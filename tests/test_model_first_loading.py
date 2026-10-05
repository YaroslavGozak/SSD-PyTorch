import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from tools.helpers import pipeline
from tools import train, train_mixed_sampling
from tools.benchmarks.benchmark_framework import BenchmarkFramework


class ModelFirstLoadingTests(unittest.TestCase):
    def test_shared_inference_loads_weights_before_dataset_then_wraps(self):
        events = []
        config = {'train_params': {'dataset': 'voc'}}
        raw_model = unittest.mock.MagicMock()
        wrapped_model = unittest.mock.MagicMock()
        dataset = object()

        def load_model(**kwargs):
            events.append('model')
            self.assertIsNone(kwargs['dataset'])
            return raw_model

        def load_dataset(*args, **kwargs):
            events.append('dataset')
            return dataset

        def wrap(model, loaded_dataset, *_args):
            events.append('wrap')
            self.assertIs(model, raw_model)
            self.assertIs(loaded_dataset, dataset)
            return wrapped_model

        with patch.object(pipeline, 'load_config', return_value=config), \
             patch.object(pipeline, 'load_model', side_effect=load_model), \
             patch.object(pipeline, 'load_dataset', side_effect=load_dataset), \
             patch.object(pipeline, 'maybe_wrap_model_for_dataset', side_effect=wrap), \
             patch.object(pipeline, 'DataLoader', return_value='loader'):
            result = pipeline.load_model_and_dataset('cpu', argparse.Namespace(config_path='unused'))

        self.assertEqual(events, ['model', 'dataset', 'wrap'])
        self.assertIs(result[0], wrapped_model)
        self.assertIs(result[1], dataset)

    def test_shared_inference_bad_weights_do_not_scan_dataset(self):
        with patch.object(pipeline, 'load_config', return_value={'train_params': {'dataset': 'voc'}}), \
             patch.object(pipeline, 'load_model', side_effect=FileNotFoundError('bad weights')), \
             patch.object(pipeline, 'load_dataset') as dataset_loader:
            with self.assertRaisesRegex(FileNotFoundError, 'bad weights'):
                pipeline.load_model_and_dataset('cpu', argparse.Namespace(config_path='unused'))
        dataset_loader.assert_not_called()

    def test_train_bad_model_fails_before_dataset(self):
        config = {'dataset_params': {'transform_name': 'ssd'}, 'train_params': {'seed': 1}}
        with patch.object(train, 'load_config', return_value=config), \
             patch.object(train, 'load_model', side_effect=FileNotFoundError('bad weights')), \
             patch.object(train, 'load_dataset') as dataset_loader:
            with self.assertRaisesRegex(FileNotFoundError, 'bad weights'):
                train.train(argparse.Namespace(config_path='unused'))
        dataset_loader.assert_not_called()

    def test_mixed_sampling_missing_checkpoint_fails_before_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            task_name = str(Path(directory).resolve())
            config = {
                'dataset_params': {'im_size': 320},
                'train_params': {'seed': 1, 'model': 'roissd', 'dataset': 'yolo-imagenet-vid',
                                 'task_name': task_name, 'ckpt_name': 'missing.pt'},
            }
            with patch.object(train_mixed_sampling, 'load_config', return_value=config), \
                 patch.object(train_mixed_sampling, 'load_model', return_value=torch.nn.Linear(1, 1)), \
                 patch.object(train_mixed_sampling, 'device', torch.device('cpu')), \
                 patch.object(train_mixed_sampling, 'YoloImageNetVidRawDataset') as dataset_loader:
                with self.assertRaisesRegex(FileNotFoundError, 'No mixed-sampling checkpoint found'):
                    train_mixed_sampling.train(argparse.Namespace(config_path='unused'))
            dataset_loader.assert_not_called()

    def test_legacy_benchmark_missing_checkpoint_fails_before_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {'benchmark_params': {
                'model': {'name': 'ssd', 'checkpoint_path': str(Path(directory) / 'missing.pt')},
                'dataset': {'name': 'voc', 'im_size': 300},
                'output': {'results_dir': directory},
            }}
            with patch.object(BenchmarkFramework, '_load_config', return_value=config), \
                 patch.object(BenchmarkFramework, '_setup_device', return_value=torch.device('cpu')), \
                 patch('tools.benchmarks.benchmark_framework.VOCDataset') as dataset_loader:
                benchmark = BenchmarkFramework('unused')
                with self.assertRaisesRegex(FileNotFoundError, 'Model checkpoint not found'):
                    benchmark.load_model_and_dataset()
            dataset_loader.assert_not_called()


if __name__ == '__main__':
    unittest.main()
