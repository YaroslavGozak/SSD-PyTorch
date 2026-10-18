import json
from pathlib import Path
from contextlib import contextmanager
import shutil
import uuid
import unittest
import yaml
from unittest.mock import MagicMock, patch

from tools import batch_infer


@contextmanager
def test_directory():
    parent = (Path.cwd()/'outputs').resolve()
    path = parent/('batch-infer-test-'+uuid.uuid4().hex)
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        assert path.resolve().parent == parent and path.name.startswith('batch-infer-test-')
        shutil.rmtree(path)


class BatchInferTests(unittest.TestCase):
    def test_resume_recovers_legacy_and_external_csvs_without_markers(self):
        with test_directory() as root:
            models = root/'trained_models'
            models.mkdir()
            checkpoint = models/'model.pt'
            checkpoint.write_bytes(b'checkpoint')
            output, external = root/'output', root/'external'
            destination, source = output/'model.pt', external/'model.pt'
            config = {'train_params': {'model': 'roissd', 'task_name': '.',
                                       'ckpt_name': 'model.pt', 'dataset': 'test'},
                      'dataset_params': {}}
            config_text = yaml.safe_dump(config, sort_keys=False)
            for directory in (destination, source):
                directory.mkdir(parents=True)
                (directory/'config.yaml').write_text(config_text)

            def rows(names):
                return [dict(task_name='.', model='roissd', dataset='test', ckpt_name='model.pt',
                             evaluated_dataset=name, mAP='0.5', mean_detector_recall='0.6',
                             mean_latency_ms='1', mean_fps='1000') for name in sorted(names)]

            pad_name, pads = batch_infer.sweep_summary('pad-loop', 'roissd')
            size_name, sizes = batch_infer.sweep_summary('fixed-size-loop', 'roissd')
            partial = rows({'fixed_padding_roi_crop_0', 'fixed_padding_roi_crop_10'})
            batch_infer.write_recovered_summary(source/'pad-loop'/pad_name, partial)
            batch_infer.write_recovered_summary(destination/'fixed-size-loop'/size_name, rows(sizes))
            original = (source/'pad-loop'/pad_name).read_bytes()
            process = MagicMock()
            process.__enter__.return_value = process
            process.stdout = []
            process.wait.return_value = 0
            argv = ['--models-root', str(models), '--output', str(output), '--resume',
                    '--resume-from', str(external), '--num-workers', '6']
            with patch.object(batch_infer, 'ROOT', root), patch.object(
                batch_infer, 'resolve_config', return_value=(config, 'test')
            ), patch.object(batch_infer.subprocess, 'Popen', return_value=process) as run:
                self.assertEqual(batch_infer.main(argv + ['--dry-run']), 0)
                run.assert_not_called()
                self.assertFalse((destination/'pad-loop'/pad_name).exists())
                self.assertEqual(batch_infer.main(argv), 0)
                self.assertEqual(run.call_count, 1)
                command = run.call_args.args[0]
                self.assertEqual(command[command.index('--eval-mode')+1], 'pad-loop')
                recovered = batch_infer.read_completed_results(destination/'pad-loop'/pad_name)
                self.assertEqual(recovered, partial)
                self.assertEqual((source/'pad-loop'/pad_name).read_bytes(), original)
                report = json.loads((output/'batch_report.json').read_text())
                self.assertEqual(report[0]['modes']['fixed-size-loop'], 'SKIPPED_COMPLETED')

            identity = {'path': str(checkpoint), 'size': checkpoint.stat().st_size,
                        'mtime_ns': checkpoint.stat().st_mtime_ns, 'config': config_text}
            self.assertEqual(batch_infer.recover_summary_rows(source, 'pad-loop', identity, config_text), partial)
            self.assertEqual(batch_infer.recover_summary_rows(source, 'pad-loop', identity, 'changed: true'), [])
            newer = dict(identity, mtime_ns=(source/'pad-loop'/pad_name).stat().st_mtime_ns + 1)
            self.assertEqual(batch_infer.recover_summary_rows(source, 'pad-loop', newer, config_text), [])
            (source/'pad-loop.progress.json').write_text(json.dumps(dict(identity, size=999)))
            self.assertEqual(batch_infer.recover_summary_rows(source, 'pad-loop', identity, config_text), [])

    def test_resume_ignores_only_execution_metadata(self):
        with test_directory() as root:
            marker = root/'complete.json'
            identity = {'path': 'model.pt', 'size': 10, 'mtime_ns': 123, 'config': 'model: ssd'}
            marker.write_text(json.dumps(dict(identity, execution={'num_workers': 1})))
            self.assertTrue(batch_infer.mode_completed(marker, identity))
            self.assertTrue(batch_infer.mode_completed(
                marker, dict(identity, execution={'num_workers': 6})))
            for key, changed in (('path', 'other.pt'), ('size', 20),
                                 ('mtime_ns', 456), ('config', 'model: roissd')):
                with self.subTest(key=key):
                    self.assertFalse(batch_infer.mode_completed(marker, dict(identity, **{key: changed})))
            marker.write_text('[]')
            self.assertFalse(batch_infer.mode_completed(marker, identity))

    def test_resume_after_interruption_skips_completed_methods_and_checkpoints(self):
        with test_directory() as root:
            models = root/'trained_models'
            models.mkdir()
            for name in ('a.pt', 'b.pt'):
                (models/name).write_bytes(b'checkpoint')
            output = root/'output'
            argv = ['--models-root', str(models), '--output', str(output), '--resume']
            config = {'train_params': {'model': 'roissd'}, 'dataset_params': {}}
            process = MagicMock()
            process.__enter__.return_value = process
            process.stdout = ['inference output\n']
            process.wait.return_value = 0
            # Complete a.pt and the first method of b.pt, then terminate.
            with patch.object(batch_infer, 'ROOT', root), patch.object(
                batch_infer, 'resolve_config', side_effect=lambda *args: (dict(
                    train_params=dict(config['train_params']), dataset_params={}), 'test')
            ), patch.object(batch_infer.subprocess, 'Popen',
                            side_effect=[process, process, process, KeyboardInterrupt]) as run:
                with self.assertRaises(KeyboardInterrupt):
                    batch_infer.main(argv)
                self.assertEqual(run.call_count, 4)
                interrupted = output/'b.pt'/batch_infer.MODES[1]
                interrupted.mkdir(parents=True)
                (interrupted/'fixed_size_results.csv').write_text('partial rows')
                # Shared configs and malformed markers cannot invalidate another
                # method's successful completion snapshot.
                (output/'a.pt'/'config.yaml').unlink()
                (output/'b.pt'/'config.yaml').write_text('stale config')
                (output/'b.pt'/f'{batch_infer.MODES[1]}.complete.json').write_text('{')
                # Legacy progress markers with execution metadata remain reusable.
                progress_path = output/'b.pt'/f'{batch_infer.MODES[1]}.progress.json'
                progress = json.loads(progress_path.read_text())
                progress['execution'] = {'num_workers': 1, 'pin_memory': False,
                                         'fixed_size_batch_size': 1, 'reuse': True}
                progress_path.write_text(json.dumps(progress))
                run.reset_mock(side_effect=True)
                run.return_value = process
                self.assertEqual(batch_infer.main(argv + ['--num-workers', '6']), 0)
                self.assertEqual(run.call_count, 1)
                command = run.call_args.args[0]
                self.assertEqual(command[command.index('--eval-mode')+1], batch_infer.MODES[1])
                self.assertIn('--resume-evaluation', command)
                self.assertEqual(command[command.index('--num-workers')+1], '6')
                self.assertEqual((interrupted/'fixed_size_results.csv').read_text(), 'partial rows')
                report = json.loads((output/'batch_report.json').read_text())
                self.assertEqual(list(report[0]['modes'].values()), ['SKIPPED_COMPLETED']*2)
                self.assertEqual(list(report[1]['modes'].values()), ['SKIPPED_COMPLETED', 'PASS'])
                # A changed checkpoint must still invalidate both methods.
                (models/'b.pt').write_bytes(b'updated checkpoint')
                run.reset_mock()
                self.assertEqual(batch_infer.main(argv), 0)
                self.assertEqual(run.call_count, 2)
                self.assertFalse((interrupted/'fixed_size_results.csv').exists())

    def test_recursive_modes_isolation_and_resume(self):
        with test_directory() as tmp:
            root = Path(tmp).resolve()
            models = root/'trained_models'
            for name in ['family/a.pt', 'family/nested/b.pth', 'family/epoch.pth']:
                p = models/name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b'checkpoint')
            mapping = root/'mapping.yaml'
            mapping.write_text('family: model.yaml\n')
            output = root/'output'
            argv = ['--models-root', str(models), '--output', str(output),
                    '--config-map', str(mapping), '--resume']
            config = {'train_params': {'model': 'roissd'}, 'dataset_params': {}}
            process = MagicMock()
            process.__enter__.return_value = process
            process.stdout = ['inference output\n']
            process.wait.return_value = 0
            with patch.object(batch_infer, 'ROOT', root), patch.object(
                batch_infer, 'load_config', return_value=config
            ), patch.object(batch_infer.subprocess, 'Popen', return_value=process) as run:
                self.assertEqual(batch_infer.main(argv), 0)
                self.assertEqual(run.call_count, 4)
                commands = [c.args[0] for c in run.call_args_list]
                self.assertEqual([c[c.index('--eval-mode')+1] for c in commands],
                                 ['pad-loop', 'fixed-size-loop']*2)
                self.assertEqual(len({c[c.index('--summary-path')+1] for c in commands}), 4)
                self.assertTrue(all(c[1] == '-u' for c in commands))
                for log in output.rglob('*.log'):
                    self.assertEqual(log.read_text(), 'inference output\n')
                self.assertEqual(batch_infer.main(argv), 0)
                self.assertEqual(run.call_count, 4)
                report = json.loads((output/'batch_report.json').read_text())
                self.assertEqual(len(report), 2)
                self.assertTrue(all(v == 'SKIPPED_COMPLETED' for r in report for v in r['modes'].values()))
                # Reversing scheduling must preserve completed evaluations.
                self.assertEqual(batch_infer.main(argv + ['--reverse']), 0)
                self.assertEqual(run.call_count, 4)
                reversed_report = json.loads((output/'batch_report.json').read_text())
                self.assertEqual([r['checkpoint'] for r in reversed_report],
                                 [r['checkpoint'] for r in reversed(report)])
                tuned = argv + ['--num-workers', '2', '--pin-memory', '--fixed-size-batch-size', '4', '--no-reuse']
                self.assertEqual(batch_infer.main(tuned), 0)
                self.assertEqual(run.call_count, 4)
                # Omitting --resume explicitly requests fresh evaluation.
                self.assertEqual(batch_infer.main([arg for arg in tuned if arg != '--resume']), 0)
                self.assertEqual(run.call_count, 8)
                commands = [c.args[0] for c in run.call_args_list[4:]]
                for command in commands:
                    self.assertEqual(command[command.index('--num-workers')+1], '2')
                    self.assertIn('--pin-memory', command)
                    self.assertIn('--no-reuse', command)
                    if command[command.index('--eval-mode')+1] == 'fixed-size-loop':
                        self.assertEqual(command[command.index('--eval-batch-size')+1], '4')
                    else:
                        self.assertNotIn('--eval-batch-size', command)
                self.assertEqual(batch_infer.main(tuned), 0)
                self.assertEqual(run.call_count, 8)
                # Switching back to defaults must accept tuned completion markers.
                self.assertEqual(batch_infer.main(argv), 0)
                self.assertEqual(run.call_count, 8)


if __name__ == '__main__':
    unittest.main()
