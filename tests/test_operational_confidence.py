import json
import unittest
from unittest.mock import Mock, patch
import yaml

from experiments.cpu_pilot import confidence
from experiments import cpu_revalidation as runner
from test_cpu_revalidation import workspace_temporary_directory


class OperationalConfidenceTests(unittest.TestCase):
    def row(self, predictions, gt=None):
        return dict(prediction_records=json.dumps(predictions),
                    ground_truth_records=json.dumps(gt if gt is not None else {'object': [[0, 0, 10, 10]]}))

    def detection(self, score, bbox=None, label='object'):
        return {'confidence': score, 'bbox': bbox or [0, 0, 10, 10], 'class': label}

    def test_maximum_f1_removes_low_confidence_false_positive(self):
        selected, _ = confidence.select_threshold([self.row([
            self.detection(.8), self.detection(.2, [20, 20, 30, 30])])])
        self.assertEqual(selected['threshold'], .25)
        self.assertEqual((selected['f1'], selected['recall']), (1., 1.))

    def test_equal_f1_prefers_higher_recall(self):
        row = self.row([self.detection(.9), self.detection(.1, [20, 20, 30, 30]),
                        self.detection(.1, [50, 50, 60, 60]), self.detection(.1, [80, 80, 90, 90])],
                       {'object': [[0, 0, 10, 10], [20, 20, 30, 30]]})
        selected, curve = confidence.select_threshold([row])
        high = next(point for point in curve if point['threshold'] == .5)
        self.assertEqual(selected['f1'], high['f1'])
        self.assertEqual(selected['recall'], 1.)
        self.assertEqual(selected['threshold'], .01)

    def test_duplicates_and_wrong_classes_are_false_positives(self):
        selected, curve = confidence.select_threshold([self.row([
            self.detection(.8), self.detection(.7), self.detection(.6, label='absent')])])
        self.assertEqual(curve[0]['tp'], 1)
        self.assertEqual(curve[0]['fp'], 2)
        self.assertEqual(selected['threshold'], .75)

    def test_no_gt_or_no_true_positive_cannot_freeze_threshold(self):
        for rows in ([self.row([], {})], [self.row([self.detection(.8, [50, 50, 60, 60])])]):
            with self.assertRaises(ValueError):
                confidence.select_threshold(rows)

    def frozen_fixture(self, directory):
        (directory / 'configs').mkdir()
        (directory / 'configs/model.yaml').write_text('config')
        (directory / 'pilot_manifest.txt').write_text('pilot')
        (directory / 'confidence_frame_metrics.csv').write_text('evidence')
        runner.write_csv(directory / 'confidence_candidates.csv', [dict(threshold=.25, f1=1., recall=1.)])
        config = {'audit_protocol': {'checkpoint_sha256': 'abc'}}
        decision = dict(model='model', threshold=.25, selection_rule=confidence.SELECTION_RULE,
                        checkpoint_sha256='abc', pilot_manifest_sha256=confidence.digest(directory / 'pilot_manifest.txt'),
                        config_sha256=confidence.digest(directory / 'configs/model.yaml'),
                        evidence_sha256={filename: confidence.digest(directory / filename) for filename in
                                         ('confidence_candidates.csv', 'confidence_frame_metrics.csv')})
        runner.write_json(directory / 'operational_confidence.json', decision)
        (directory / 'operational_confidence.sha256').write_text(confidence.digest(directory / 'operational_confidence.json'))
        return config

    def test_frozen_threshold_validates_and_rejects_changed_decision(self):
        with workspace_temporary_directory() as directory:
            config = self.frozen_fixture(directory)
            self.assertEqual(confidence.load_frozen_threshold(directory, 'model', config), .25)
            decision = runner.read_json(directory / 'operational_confidence.json')
            decision['threshold'] = .5
            runner.write_json(directory / 'operational_confidence.json', decision)
            with self.assertRaisesRegex(ValueError, 'modified'):
                confidence.load_frozen_threshold(directory, 'model', config)

    def test_frozen_threshold_rejects_changed_evidence_or_manifest(self):
        for filename in ('pilot_manifest.txt', 'confidence_frame_metrics.csv'):
            with self.subTest(filename=filename), workspace_temporary_directory() as directory:
                config = self.frozen_fixture(directory)
                (directory / filename).write_text('changed')
                with self.assertRaises(ValueError):
                    confidence.load_frozen_threshold(directory, 'model', config)

    def test_pilot_loader_applies_frozen_threshold_instead_of_template(self):
        from experiments.cpu_pilot import run_pilot as pilot
        import torch

        with workspace_temporary_directory() as directory:
            (directory / 'configs').mkdir()
            config = {'confidence_protocol': {'role': 'operational'}, 'train_params': {'infer_conf_threshold': .5}}
            for name in ('roi_ssd_vgg16', 'yolo26n'):
                (directory / f'configs/{name}.yaml').write_text(yaml.safe_dump(config))
                model = Mock()
                model.parameters.return_value = []
                model._yolo.names = ['object']
                dataset = Mock(idx2label={0: 'background', 1: 'object'})
                with self.subTest(model=name), patch.object(pilot, 'PILOT_DIR', directory), patch.object(
                    pilot, 'load_model_and_dataset', return_value=(model, dataset, None, config)
                ), patch.object(confidence, 'load_frozen_threshold', return_value=.25):
                    result = pilot._load_model(name, torch.device('cpu'))
                self.assertEqual(result[-1], .25)
                self.assertEqual(model.low_score_threshold, .25)
                if name == 'yolo26n':
                    self.assertEqual(model.conf_threshold, .25)

    def test_pilot_loader_blocks_before_loading_model_without_selection(self):
        from experiments.cpu_pilot import run_pilot as pilot
        import torch

        with workspace_temporary_directory() as directory:
            (directory / 'configs').mkdir()
            (directory / 'configs/roi_ssd_vgg16.yaml').write_text('confidence_protocol:\n  role: operational\n')
            with patch.object(pilot, 'PILOT_DIR', directory), patch.object(pilot, 'load_model_and_dataset') as load:
                with self.assertRaisesRegex(FileNotFoundError, 'freeze operational confidence'):
                    pilot._load_model('roi_ssd_vgg16', torch.device('cpu'))
                load.assert_not_called()

    def test_selection_reuses_ff_pipeline_and_freezes_only_pilot_evidence(self):
        from experiments.cpu_pilot import run_pilot as pilot
        import torch

        with workspace_temporary_directory() as directory:
            (directory / 'configs').mkdir()
            (directory / 'configs/model.yaml').write_text('snapshot')
            (directory / 'pilot_manifest.txt').write_text('frozen pilot')
            config = {'audit_protocol': {'checkpoint_sha256': 'abc'}}
            row = self.row([self.detection(.8), self.detection(.2, [20, 20, 30, 30])])
            split = {'provenance': {'pilot_manifest_sha256': confidence.digest(directory / 'pilot_manifest.txt')}}
            manifests = [{'image': 'pilot-only.jpg'}]
            with patch.object(pilot, 'PILOT_DIR', directory), patch.object(pilot, 'MODEL_NAMES', ('model',)), patch.object(
                pilot, '_load_model', return_value=(Mock(), Mock(), config, .01)
            ) as load, patch.object(pilot, '_dataset_root', return_value=directory), patch.object(
                pilot, '_dataset_indices', return_value=[0]
            ), patch.object(pilot, '_run_pipeline', return_value=({'num_frames': 1}, [row])) as run:
                confidence.run_selection(split, manifests, torch.device('cpu'))
            self.assertEqual(load.call_args.kwargs['confidence_floor'], .01)
            self.assertEqual(run.call_args.args[5], manifests)
            self.assertEqual(run.call_args.args[8], 'FF')
            self.assertEqual(confidence.load_frozen_threshold(directory, 'model', config), .25)
            decision = runner.read_json(directory / 'operational_confidence.json')
            self.assertEqual(decision['pilot_frames'], 1)
            self.assertEqual(decision['status'], 'FROZEN_FROM_PILOT')
            self.assertEqual(len(runner.read_csv(directory / 'confidence_candidates.csv')), 20)


if __name__ == '__main__':
    unittest.main()
