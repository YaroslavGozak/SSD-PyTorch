import argparse
import random
import unittest
from pathlib import Path
from test_batch_infer import test_directory
from unittest.mock import patch, MagicMock

import torch
from torchvision import tv_tensors

from tools import infer
from dataset.voc import VOCDataset
from dataset.transforms.fixed_size_test_transform import FixedSizeTestTransform
from dataset.transforms.fixed_padding_roi_crop_test_transform import FixedPaddingRoiCropTestTransform
from dataset.transforms.fixed_padding_roi_crop_yolo_test_transform import FixedPaddingRoiCropYOLOTestTransform


class EvaluationOptimizationTests(unittest.TestCase):
    def test_summary_commit_is_atomic_and_replaces_incomplete_rows(self):
        with test_directory() as tmp:
            path = tmp/'results.csv'
            kwargs = dict(model_task_path=str(tmp), config={'train_params': {}},
                          evaluated_dataset='pad_0', mean_ap=0.5, mean_recall=0.6,
                          mean_latency_ms=1.0, mean_fps=1000.0, replace_existing=True)
            infer.append_model_results_csv(**kwargs)
            original = path.read_bytes()
            with patch.object(infer.os, 'replace', side_effect=OSError('interrupted commit')):
                with self.assertRaises(OSError):
                    infer.append_model_results_csv(**dict(kwargs, evaluated_dataset='pad_10'))
            self.assertEqual(path.read_bytes(), original)
            with path.open('a', encoding='utf-8') as stream:
                stream.write(',,,,pad_10,0.5')
            self.assertEqual(len(infer.read_completed_results(path)), 1)
            infer.append_model_results_csv(**dict(kwargs, evaluated_dataset='pad_10'))
            infer.append_model_results_csv(**kwargs)
            rows = infer.read_completed_results(path)
            self.assertEqual(len(rows), 2)
            self.assertEqual({row['evaluated_dataset'] for row in rows}, {'pad_0', 'pad_10'})

    def test_sweeps_resume_at_interrupted_setting(self):
        for loop, csv_name, stop in (
            (infer.evaluate_map_pad_loop, 'results.csv', 16),
            (infer.evaluate_map_fixed_size_loop, 'fixed_size_results_v2.csv', 3),
        ):
            with self.subTest(loop=loop.__name__), test_directory() as tmp:
                config_path = Path(tmp)/'config.yaml'
                config_path.write_text('train_params:\n  model: roissd\n')
                args = self.args(resume_evaluation=True)
                args.config_path = str(config_path)
                args.summary_path = str(tmp)
                dataset = MagicMock()
                dataset.idx2label = {}
                loaded = (object(), dataset, [], {'train_params': {}})
                attempted = []

                def resources(args, transform, cached):
                    attempted.append(transform)
                    if len(attempted) == stop + 1:
                        raise KeyboardInterrupt
                    return loaded

                with patch.object(infer, 'evaluation_resources', side_effect=resources), patch.object(
                    infer, 'compute_map', return_value=(0.5, {}, 0.6, {})
                ):
                    with self.assertRaises(KeyboardInterrupt):
                        loop(args)
                    completed = infer.read_completed_results(Path(tmp)/csv_name)
                    self.assertEqual(len(completed), stop)
                    interrupted = attempted[-1]
                    attempted.clear()
                    # Disable further interruption and finish the sweep.
                    with patch.object(infer, 'evaluation_resources', side_effect=lambda a, t, r:
                                      (attempted.append(t) or loaded)):
                        result = loop(args)
                    self.assertEqual(attempted[0], interrupted)
                    self.assertTrue(all(row['evaluated_dataset'] not in attempted for row in completed))
                    rows = infer.read_completed_results(Path(tmp)/csv_name)
                    self.assertEqual(len(rows), len(result['runs']))
                    self.assertEqual(len(rows), len({row['evaluated_dataset'] for row in rows}))
                    with patch.object(infer, 'evaluation_resources') as load:
                        loop(args)
                        load.assert_not_called()

    def args(self, **kwargs):
        return argparse.Namespace(num_workers=0, pin_memory=False, eval_batch_size=1,
                                  summary_path='unused', results_path=None, config_path='unused', **kwargs)

    def test_reuse_keeps_metadata_and_original_transform_and_matches_fresh(self):
        dataset = VOCDataset.__new__(VOCDataset)
        dataset.images_info = []
        dataset.im_size = 64
        dataset.imagenet_mean = [0.485, 0.456, 0.406]
        dataset.imagenet_std = [0.229, 0.224, 0.225]
        dataset.transform_name = 'original'
        dataset.transforms = object()
        resources = [object(), dataset, None, {}]
        image = tv_tensors.Image(torch.arange(3*64*64, dtype=torch.uint8).reshape(3, 64, 64))
        def target():
            return {'boxes': tv_tensors.BoundingBoxes([[12, 12, 25, 25]], format='XYXY', canvas_size=(64, 64)),
                    'labels': torch.tensor([1]), 'difficult': torch.tensor([0])}
        cases = [
            ('fixed_size_32x64', FixedSizeTestTransform(32, 64, dataset.imagenet_mean, dataset.imagenet_std, frame_size=64)),
            ('fixed_size_yolo_32x64', FixedSizeTestTransform(32, 64, dataset.imagenet_mean, dataset.imagenet_std, normalize=False, frame_size=64)),
            ('fixed_padding_roi_crop_10', FixedPaddingRoiCropTestTransform(64, dataset.imagenet_mean, dataset.imagenet_std, 10, 10)),
            ('fixed_padding_roi_crop_yolo_10', FixedPaddingRoiCropYOLOTestTransform(64, dataset.imagenet_mean, dataset.imagenet_std, 10, 10)),
        ]
        with patch.object(infer, 'load_model_and_dataset') as load:
            for name, fresh in cases:
                model, reused, loader, config = infer.evaluation_resources(self.args(), name, resources)
                self.assertIs(model, resources[0])
                self.assertIs(reused.images_info, dataset.images_info)
                self.assertIsNot(reused, dataset)
                random.seed(7)
                expected = fresh.transforms['test'](image.clone(), target())
                random.seed(7)
                actual = reused.transforms['test'](image.clone(), target())
                torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
                for key in expected[1]:
                    torch.testing.assert_close(actual[1][key], expected[1][key], rtol=0, atol=0)
            load.assert_not_called()
        self.assertEqual(dataset.transform_name, 'original')

    def test_batched_metrics_match_single_images_including_empty_targets(self):
        samples = []
        for count in (1, 0, 1):
            samples.append((torch.zeros(3, 32, 32), {
                'bboxes': torch.tensor([[0.1, 0.1, 0.5, 0.5]]*count).reshape(-1, 4),
                'labels': torch.zeros(count, dtype=torch.long),
                'difficult': torch.zeros(count, dtype=torch.long)}, 'image'))
        dataset = MagicMock()
        dataset.label2idx = {'object': 0}
        dataset.idx2label = {0: 'object'}
        outputs = []
        for batch_size in (1, 2):
            loader = torch.utils.data.DataLoader(samples, batch_size=batch_size, collate_fn=infer.evaluation_collate)
            def detect(model, images, target, **kwargs):
                return [{'boxes': torch.tensor([[0.1, 0.1, 0.5, 0.5]]),
                         'labels': torch.tensor([0]), 'scores': torch.tensor([0.9])} for _ in images]
            with patch.object(infer, 'evaluation_resources', return_value=(object(), dataset, loader, {'train_params': {}})), \
                 patch.object(infer, 'run_detector', side_effect=detect), \
                 patch.object(infer, 'append_model_results_csv'), patch.object(infer, 'device', torch.device('cpu')):
                outputs.append(infer._evaluate_single_transform(self.args(), transform_name='fixed_size_32x32',
                               evaluated_dataset='test', model_name='roissd'))
        self.assertEqual(outputs[0]['mAP'], outputs[1]['mAP'])
        self.assertEqual(outputs[0]['mean_detector_recall'], outputs[1]['mean_detector_recall'])

    def test_worker_collation_preserves_variable_targets_and_last_batch(self):
        samples = [(torch.zeros(3, 32, 32), {'labels': torch.arange(n)}, str(n))
                   for n in (0, 2, 1)]
        loader = torch.utils.data.DataLoader(samples, batch_size=2, num_workers=1,
                                             collate_fn=infer.evaluation_collate)
        batches = list(loader)
        self.assertEqual([batch[0].shape[0] for batch in batches], [2, 1])
        self.assertEqual([len(t['labels']) for batch in batches for t in batch[1]], [0, 2, 1])

    def test_unknown_dataset_falls_back_and_padding_batches_rejected(self):
        with patch.object(infer, 'load_model_and_dataset', return_value=(object(), object(), None, {})) as load:
            infer.evaluation_resources(self.args(), 'fixed_size_32x32', [object(), object(), None, {}])
            load.assert_called_once()
            args = self.args()
            args.eval_batch_size = 2
            with self.assertRaisesRegex(ValueError, 'fixed-size'):
                infer.evaluation_resources(args, 'fixed_padding_roi_crop_10', None)


if __name__ == '__main__':
    unittest.main()
