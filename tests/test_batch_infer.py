import json
from pathlib import Path
from contextlib import contextmanager
import shutil
import uuid
import unittest
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
                run.reset_mock(side_effect=True)
                run.return_value = process
                self.assertEqual(batch_infer.main(argv), 0)
                self.assertEqual(run.call_count, 1)
                command = run.call_args.args[0]
                self.assertEqual(command[command.index('--eval-mode')+1], batch_infer.MODES[1])
                self.assertFalse((interrupted/'fixed_size_results.csv').exists())
                report = json.loads((output/'batch_report.json').read_text())
                self.assertEqual(list(report[0]['modes'].values()), ['SKIPPED_COMPLETED']*2)
                self.assertEqual(list(report[1]['modes'].values()), ['SKIPPED_COMPLETED', 'PASS'])
                # A changed checkpoint must still invalidate both methods.
                (models/'b.pt').write_bytes(b'updated checkpoint')
                run.reset_mock()
                self.assertEqual(batch_infer.main(argv), 0)
                self.assertEqual(run.call_count, 2)

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


if __name__ == '__main__':
    unittest.main()
