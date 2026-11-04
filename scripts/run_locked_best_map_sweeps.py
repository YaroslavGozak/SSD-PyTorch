"""Run both inference sweeps for the two explicitly locked best_map checkpoints."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    'roi_ssd': ('imagenet-vid-roissd-20261028/stage3', 'f8f49132e08a7c5bf3744ce32e46e1b0fed6595aabd9cf59cf68699569503f76'),
    'roi_ssd_mobilenet': ('imagenet-vid-roissd-mobilenet-20261027/stage3', '1a830c339f060905f1042e10e98906be0cab43fa94cb3867bfc683ad14adc804'),
}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(f'Use a fresh directory or --resume: {output}')
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / 'run_status.json'
    status = {'status': 'RUNNING', 'pid': os.getpid(), 'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'python': sys.executable, 'jobs': [], 'scope': 'Full configured test split; batch=1; automatic inference device'}
    save(status_path, status)
    try:
        for name, (run, expected_hash) in MODELS.items():
            checkpoint = ROOT / 'trained_models' / run / 'best_map.pt'
            assert digest(checkpoint) == expected_hash, f'Locked checkpoint changed: {checkpoint}'
            source = ROOT / 'final_experiments/configs' / (name + '.yaml')
            source_bytes = source.read_bytes()
            cfg = yaml.safe_load(source_bytes)
            assert cfg['train_params']['task_name'] == run and cfg['train_params']['ckpt_name'] == 'best_map.pt'
            for mode in ('fixed-size-loop', 'pad-loop'):
                directory = output / name / mode
                directory.mkdir(parents=True, exist_ok=True)
                config = directory / 'config.yaml'
                identity_path = directory / 'run_identity.json'
                identity = {'checkpoint': checkpoint.relative_to(ROOT).as_posix(), 'checkpoint_sha256': expected_hash,
                            'config_sha256': hashlib.sha256(source_bytes).hexdigest(), 'mode': mode,
                            'sources_sha256': {p: digest(ROOT / p) for p in (
                                'tools/infer.py', 'tools/helpers/pipeline.py', 'model/roissd.py', 'model/roissd_mobilenet.py',
                                'dataset/transforms/fixed_size_test_transform.py', 'dataset/transforms/fixed_padding_roi_crop_test_transform.py')}}
                if identity_path.exists():
                    assert args.resume and json.loads(identity_path.read_text()) == identity, 'Resume identity mismatch'
                    assert digest(config) == identity['config_sha256']
                else:
                    assert not list(directory.glob('*.csv')), 'Existing unbound CSVs must not be reused'
                    config.write_bytes(source_bytes)
                    save(identity_path, identity)
                command = [sys.executable, '-u', '-m', 'tools.infer', '--config', str(config),
                           '--evaluate', 'true', '--infer-samples', 'false', '--eval-mode', mode,
                           '--summary-path', str(directory), '--results-path', str(directory)]
                if args.resume:
                    command.append('--resume-evaluation')
                job = {'model': name, 'mode': mode, 'status': 'RUNNING', 'command': command,
                       'directory': str(directory), 'started_at_utc': datetime.now(timezone.utc).isoformat()}
                status['jobs'].append(job)
                save(status_path, status)
                print(f'START {name} {mode}; log: {directory / "execution.log"}', flush=True)
                with (directory / 'execution.log').open('a' if args.resume else 'w', encoding='utf-8') as log:
                    result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                job['returncode'] = result.returncode
                assert result.returncode == 0, f'Inference failed: {name} {mode}; inspect execution.log'
                assert digest(checkpoint) == expected_hash, 'Checkpoint changed during inference'
                assert digest(config) == identity['config_sha256'], 'Configuration changed during inference'
                summary = directory / ('fixed_size_results_v2.csv' if mode == 'fixed-size-loop' else 'results.csv')
                rows = list(csv.DictReader(summary.open(newline='', encoding='utf-8')))
                expected = ({f'fixed_size_{h}x{w}' for h in range(32, 321, 32) for w in range(32, 321, 32)}
                            if mode == 'fixed-size-loop' else {f'fixed_padding_roi_crop_{p}' for p in range(0, 201, 10)})
                assert len(rows) == len(expected) and {r['evaluated_dataset'] for r in rows} == expected
                assert all(r['ckpt_name'] == 'best_map.pt' and r['task_name'] == run for r in rows)
                job.update(status='COMPLETE', rows=len(rows), csv_sha256=digest(summary),
                           finished_at_utc=datetime.now(timezone.utc).isoformat())
                save(status_path, status)
                print(f'COMPLETE {name} {mode}: {len(rows)} settings', flush=True)
        status['status'] = 'COMPLETE'
    except BaseException as error:
        status.update(status='FAILED', error=str(error))
        if status['jobs'] and status['jobs'][-1]['status'] == 'RUNNING':
            status['jobs'][-1]['status'] = 'FAILED'
        raise
    finally:
        save(status_path, status)


if __name__ == '__main__':
    main()
