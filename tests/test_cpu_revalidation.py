from pathlib import Path
from contextlib import contextmanager
import shutil
import uuid
from types import SimpleNamespace
import unittest
import yaml
from unittest.mock import patch

from experiments import cpu_revalidation as runner


@contextmanager
def workspace_temporary_directory():
    root = Path(__file__).resolve().parent
    directory = root / ('.tmp_cpu_revalidation_' + uuid.uuid4().hex)
    directory.mkdir()
    try:
        yield directory
    finally:
        if not directory.resolve().is_relative_to(root):
            raise ValueError('Test cleanup escaped tests directory')
        shutil.rmtree(directory)


class CPURevalidationTests(unittest.TestCase):
    def test_all_orders_full_frame_both_loops_then_padding_cost_pipeline_jobs(self):
        args = SimpleNamespace(output=Path('example'), stage='all', models=list(runner.MODELS),
                               accuracy_modes=list(runner.ACCURACY_MODES))
        jobs = runner.jobs(args)
        self.assertEqual(len(jobs), 26)
        self.assertTrue(all(directory.parent.parent.name == 'confidence_diagnostics' for directory, _, _ in jobs[9:14]))
        self.assertEqual([stage for _, stage, _ in jobs[14:]],
                         ['confidence'] * 3 + ['padding'] * 3 + ['cost'] * 3 + ['pipeline'] * 3)
        for _, _, command in jobs[:9]:
            self.assertEqual(command[command.index('--device') + 1], 'cpu')
            self.assertIn('--audit-details', command)
            self.assertNotIn('--resume-evaluation', command)
        self.assertEqual([c[c.index('--eval-mode') + 1] for _, _, c in jobs[:9]],
                         ['default'] * 3 + ['pad-loop'] * 3 + ['fixed-size-loop'] * 3)

    def test_full_frame_only_uses_frozen_manifest_and_isolated_outputs_for_every_model(self):
        args = SimpleNamespace(output=Path('example'), stage='accuracy', models=list(runner.MODELS),
                               accuracy_modes=['full-frame'])
        jobs = runner.jobs(args)
        self.assertEqual(len(jobs), 3)
        for (directory, stage, command), name in zip(jobs, runner.MODELS):
            self.assertEqual(directory, args.output / 'accuracy_sweeps' / name / 'full_frame')
            self.assertEqual(command[command.index('--frame-manifest') + 1], str(args.output / 'manifests/val_manifest.jsonl'))
            self.assertEqual(command[command.index('--eval-mode') + 1], 'default')
            self.assertEqual(stage, 'accuracy')

    def test_aggregate_full_frame_accuracy_without_inventing_a_padding_value(self):
        with workspace_temporary_directory() as root:
            runner.write_json(root / 'session.json', {'models': {'yolo26n': {'path': 'best.pt', 'sha256': 'abc'}}})
            directory = root / 'accuracy_sweeps/yolo26n/full_frame'
            directory.mkdir(parents=True)
            runner.write_json(directory / 'job_accuracy.json', {'status': 'COMPLETE'})
            runner.write_json(directory / 'ssd_audit.json', dict(evaluated_dataset='ssd', evaluated_frames=5064,
                total_frames=5064, mAP50=.8, mAP50_95=.5, detector_recall=.9,
                actual_input_shapes=[dict(h=320, w=320, frames=5064)]))
            runner.aggregate(root, ['yolo26n'])
            rows = runner.read_csv(root / 'combined_reports/full_frame_accuracy.csv')
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['checkpoint_sha256'], 'abc')
            self.assertEqual(rows[0]['evaluated_frames'], '5064')
            self.assertNotIn('padding', rows[0])
            self.assertTrue(runner.read_json(root / 'combined_reports/completeness.json')['yolo26n/full_frame'])

    def test_preflight_rejects_existing_accuracy_directory_and_failed_stage(self):
        with workspace_temporary_directory() as temporary:
            directory = Path(temporary)
            with self.assertRaises(FileExistsError):
                runner.validate_destinations([(directory, 'accuracy', [])])
            (directory / 'job_cost.json').write_text('{"status":"FAILED"}')
            with self.assertRaises(FileExistsError):
                runner.validate_destinations([(directory, 'cost', [])])

    def test_check_rejects_changed_source_before_any_experiment(self):
        with workspace_temporary_directory() as temporary:
            directory = Path(temporary)
            source = directory / 'source.py'
            source.write_text('original')
            runner.write_json(directory / 'session.json', {'locked_files': {str(source): runner.sha(source)}, 'models': {}})
            source.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'Locked file changed'):
                runner.check(directory)

    def test_execute_stops_after_failed_child_and_records_failure(self):
        with workspace_temporary_directory() as temporary:
            directory = Path(temporary)
            runner.write_json(directory / 'session.json', {})
            with patch.object(runner.subprocess, 'run', side_effect=RuntimeError('child failure')) as run:
                with self.assertRaisesRegex(RuntimeError, 'child failure'):
                    runner.execute([(directory / 'first', 'accuracy', ['python']),
                                    (directory / 'second', 'accuracy', ['python'])], directory, {'models': {}})
                self.assertEqual(run.call_count, 1)
            self.assertEqual(runner.read_json(directory / 'first/job_accuracy.json')['status'], 'FAILED')
            self.assertFalse((directory / 'second').exists())

    def test_identical_ca_full_tensors_and_predictions_explain_selector(self):
        with workspace_temporary_directory() as temporary:
            directory = Path(temporary)
            runner.write_json(directory / 'session.json', {'models': {'yolo26n': {'path': 'best.pt', 'sha256': 'abc'}}})
            pilot = directory / 'pilot/yolo26n'
            pilot.mkdir(parents=True)
            runner.write_csv(pilot / 'pipeline_summary.csv', [{'mode': mode} for mode in ('FF', 'SEG', 'CA', 'FULL-TECH')])
            base = dict(frame_id='video:1', actual_network_shapes_hw='[[320, 320]]', prediction_records='[]',
                        predicted_segment_cost='2', predicted_full_frame_cost='1', selector_decision='FF',
                        fallback_reason='cost_prefers_full_frame')
            runner.write_csv(pilot / 'frame_level_metrics.csv', [dict(base, mode='CA'), dict(base, mode='FULL-TECH')])
            runner.aggregate(directory, ['yolo26n'])
            rows = runner.read_csv(directory / 'combined_reports/ca_vs_full_tech.csv')
            self.assertIn('did not change actual NN workload', rows[0]['interpretation'])
            self.assertEqual(rows[0]['same_predictions'], 'True')
            self.assertFalse(all(runner.read_json(directory / 'combined_reports/completeness.json').values()))

    def test_quality_floor_does_not_modify_operational_template_or_source(self):
        source = dict(train_params={'infer_conf_threshold': .5}, model_params={'low_score_threshold': .5},
                      audit_protocol={'checkpoint_sha256': 'abc'})
        quality = runner.confidence_config(source, 'quality')
        operational = runner.confidence_config(source, 'operational')
        self.assertEqual(quality['train_params']['infer_conf_threshold'], .01)
        self.assertEqual(quality['model_params']['low_score_threshold'], .01)
        self.assertEqual(operational['train_params']['infer_conf_threshold'], .5)
        self.assertTrue(operational['confidence_protocol']['selection_required'])
        self.assertEqual(source['model_params']['low_score_threshold'], .5)
        self.assertNotIn('confidence_protocol', source)

    def test_confidence_diagnostic_runs_five_full_frame_jobs_and_does_not_run_pilot(self):
        args = SimpleNamespace(output=Path('example'), stage='confidence-diagnostic', models=list(runner.MODELS),
                               accuracy_modes=list(runner.ACCURACY_MODES))
        planned = runner.jobs(args)
        self.assertEqual(len(planned), 5)
        for (directory, stage, command), threshold in zip(planned, runner.ROI_SSD_CONFIDENCE_VALUES):
            self.assertEqual(directory, args.output / 'confidence_diagnostics/roi_ssd_vgg16' / runner.confidence_tag(threshold))
            self.assertEqual(command[command.index('--eval-mode') + 1], 'default')
            self.assertEqual(command[command.index('--device') + 1], 'cpu')
            self.assertEqual(stage, 'accuracy')
        args.stage = 'accuracy'
        args.with_confidence_diagnostic = True
        self.assertEqual(len(runner.jobs(args)), 14)

    def test_diagnostic_configs_set_both_cutoffs_and_preserve_quality_config(self):
        with workspace_temporary_directory() as root:
            (root / 'configs').mkdir()
            source = dict(train_params={'infer_conf_threshold': .01}, model_params={'low_score_threshold': .01, 'nms_threshold': .45},
                          audit_protocol={'checkpoint_sha256': 'abc'})
            source_path = root / 'configs/roi_ssd_vgg16.yaml'
            source_path.write_text(yaml.safe_dump(source))
            runner.prepare_confidence_diagnostic(root)
            self.assertEqual(yaml.safe_load(source_path.read_text()), source)
            for value in runner.ROI_SSD_CONFIDENCE_VALUES:
                cfg = yaml.safe_load((root / 'configs/confidence_diagnostic/roi_ssd_vgg16' / f'{runner.confidence_tag(value)}.yaml').read_text())
                self.assertEqual(cfg['train_params']['infer_conf_threshold'], value)
                self.assertEqual(cfg['model_params']['low_score_threshold'], value)
                self.assertEqual(cfg['model_params']['nms_threshold'], .45)
                self.assertEqual(cfg['confidence_protocol']['role'], 'confidence_diagnostic')

    def test_diagnostic_summary_is_separate_and_rejects_incomplete_coverage(self):
        with workspace_temporary_directory() as root:
            runner.write_json(root / 'session.json', {'models': {'roi_ssd_vgg16': {'path': 'best_map.pt', 'sha256': 'abc'}}})
            configs = root / 'configs/confidence_diagnostic/roi_ssd_vgg16'
            configs.mkdir(parents=True)
            for value in runner.ROI_SSD_CONFIDENCE_VALUES:
                tag = runner.confidence_tag(value)
                (configs / f'{tag}.yaml').write_text('config')
                run = root / 'confidence_diagnostics/roi_ssd_vgg16' / tag
                run.mkdir(parents=True)
                runner.write_json(run / 'job_accuracy.json', {'status': 'COMPLETE'})
                runner.write_json(run / 'ssd_audit.json', dict(confidence_floor=value, evaluated_frames=5064, total_frames=5064,
                    mAP50=.8, mAP50_95=.5, detector_recall=.9))
            runner.summarize_confidence_diagnostic(root)
            rows = runner.read_csv(root / 'confidence_diagnostics/roi_ssd_vgg16/summary.csv')
            self.assertEqual([float(row['confidence_threshold']) for row in rows], list(runner.ROI_SSD_CONFIDENCE_VALUES))
            runner.aggregate(root, ['roi_ssd_vgg16'])
            self.assertTrue((root / 'combined_reports/roi_ssd_confidence_diagnostic.csv').is_file())
            self.assertFalse((root / 'combined_reports/accuracy_sweeps.csv').exists())
            run = root / 'confidence_diagnostics/roi_ssd_vgg16/conf_0p5'
            row = runner.read_json(run / 'ssd_audit.json')
            row['evaluated_frames'] = 1
            runner.write_json(run / 'ssd_audit.json', row)
            with self.assertRaisesRegex(ValueError, 'coverage mismatch'):
                runner.confidence_diagnostic_rows(root)


if __name__ == '__main__':
    unittest.main()
