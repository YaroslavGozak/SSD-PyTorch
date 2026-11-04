"""Apply the user's explicit checkpoint selection and verify local CPU inference."""
import csv
import hashlib
import json
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / 'final_experiments/01_audit'
SPECS = {
    'ROI-SSD': ('roi_ssd', 'imagenet-vid-roissd-20261028/stage3', 'roissd'),
    'ROI-SSD-MobileNet': ('roi_ssd_mobilenet', 'imagenet-vid-roissd-mobilenet-20261027/stage3', 'roissd-mobilenet'),
}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    torch.set_num_threads(8)
    roles_path = AUDIT / 'model_role_recommendations.json'
    roles = json.loads(roles_path.read_text(encoding='utf-8'))
    previous = AUDIT / 'model_role_recommendations_before_20261031.json'
    if not previous.exists():
        previous.write_text(json.dumps(roles, indent=2) + '\n', encoding='utf-8')
    evidence = {}
    for family, (slug, run, architecture) in SPECS.items():
        directory = ROOT / 'trained_models' / run
        checkpoint = directory / 'best_map.pt'
        source = directory / 'mixed_sampling_eval.yaml'
        cfg = yaml.safe_load(source.read_text(encoding='utf-8'))
        assert cfg['train_params']['model'] == architecture
        assert cfg['train_params']['dataset'] == 'yolo-imagenet-vid'
        assert cfg['dataset_params']['num_classes'] == 31
        cfg['train_params'].update(task_name=run, ckpt_name='best_map.pt', device='cpu', penalized_roissd=False)
        digest = sha(checkpoint)
        cfg['audit_protocol'] = {
            'purpose': 'User-selected checkpoint for all subsequent tests; inference only',
            'selection_date': '2026-10-31', 'selection_reason': 'Explicit user selection; no new comparative ranking claimed',
            'checkpoint_sha256': digest,
            'canonical_manifest': 'final_experiments/01_audit/canonical_validation.jsonl',
            'final_manifest': 'final_experiments/01_audit/val_manifest.jsonl',
            'source_config': source.relative_to(ROOT).as_posix(),
        }
        config_path = ROOT / 'final_experiments/configs' / (slug + '.yaml')
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
        state = saved.get('model', saved)
        from model.roissd import RoiSSD
        from model.roissd_mobilenet import RoiSSDMobileNet
        model = (RoiSSD if architecture == 'roissd' else RoiSSDMobileNet)(config=cfg['model_params'], num_classes=31)
        model.load_state_dict(state, strict=True)
        model.eval()
        import cv2
        import numpy as np
        dataset_yaml = Path(cfg['dataset_params']['yolo_dataset_yaml'])
        names = yaml.safe_load(dataset_yaml.read_text(encoding='utf-8'))['names']
        names = [names[i] for i in range(len(names))] if isinstance(names, dict) else names
        assert names == json.loads((AUDIT / 'dataset_audit.json').read_text())['names']
        row = json.loads((AUDIT / 'canonical_validation.jsonl').read_text().splitlines()[0])
        image_path = dataset_yaml.parent / row['image']
        image = cv2.imread(str(image_path))
        assert image is not None, image_path
        checks = []
        with torch.inference_mode():
            for h, w in [(320, 320), (96, 96), (128, 128), (160, 160), (160, 192)]:
                rgb = cv2.cvtColor(cv2.resize(image, (w, h)), cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255
                tensor = (tensor - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]
                _, detections = model(tensor.unsqueeze(0), None)
                det = detections[0]
                assert torch.isfinite(det['boxes']).all() and torch.isfinite(det['scores']).all()
                assert ((det['labels'] >= 1) & (det['labels'] <= 30)).all()
                assert ((det['boxes'] >= 0) & (det['boxes'] <= 1)).all()
                checks.append({'shape': [h, w], 'detections': len(det['scores']), 'status': 'PASS'})
        metrics = list(csv.DictReader((directory / 'training_metrics.csv').open()))
        peak = max(metrics, key=lambda x: float(x['mAP']))
        evidence[family] = {
            'checkpoint': checkpoint.relative_to(ROOT).as_posix(), 'checkpoint_sha256': digest,
            'config': config_path.relative_to(ROOT).as_posix(), 'source_config': source.relative_to(ROOT).as_posix(),
            'selection_reason': 'Explicit user selection for all tests; not a fresh comparative ranking',
            'checkpoint_metadata': {key: saved.get(key) for key in ('epoch', 'stage', 'best_map')},
            'training_metrics_peak': peak, 'metrics_scope': 'Saved run validation protocol; not the historical canonical audit',
            'class_compatibility': 'BACKGROUND_OFFSET_ONLY', 'smoke': checks,
            'smoke_image': str(image_path),
        }
        roles['families'][family] = {
            'full_frame_checkpoint': evidence[family]['checkpoint'], 'roi_checkpoint': evidence[family]['checkpoint'],
            'full_frame_config': evidence[family]['config'], 'roi_config': evidence[family]['config'],
            'checkpoint_sha256': digest, 'selection_reason': evidence[family]['selection_reason'],
            'full_frame_map50': None, 'full_frame_map50_95': None,
            'roi_evidence': {'status': 'REQUIRES_FRESH_EVALUATION', 'audit': 'final_experiments/01_audit/reselection_20261031.json'},
        }
        del model, saved, state
    roles.update(updated='2026-10-31', selection_scope='User-selected single checkpoint per family for all tests')
    roles_path.write_text(json.dumps(roles, indent=2) + '\n', encoding='utf-8')
    (AUDIT / 'reselection_20261031.json').write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
