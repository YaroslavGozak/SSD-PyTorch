"""Read-only model/data audit; writes only under final_experiments.

Run from repository root: .venv/Scripts/python.exe final_experiments/audit/run_audit.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import csv
import gc
import hashlib
import json
import re
import subprocess
import platform
from collections import Counter
import numpy as np
import torch
import torchvision
import cv2
import yaml
import ultralytics
from ultralytics import YOLO
from tools.helpers.config_reader import load_config
from dataset.yolo_imagenet_vid import _extract_video_tokens
from model.roissd import RoiSSD
from model.roissd_mobilenet import RoiSSDMobileNet

OUT = ROOT / 'final_experiments/audit'
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + '\n', encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_csv(name, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / name).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def dataset_audit():
    data_yaml = Path(load_config('config/imagenet-vid-yolo-base.yaml')['dataset_params']['yolo_dataset_yaml'])
    data = yaml.safe_load(data_yaml.read_text(encoding='utf-8'))
    names = [data['names'][i] for i in range(len(data['names']))]
    root = Path(data['path'])
    report = {'root': str(root), 'yaml': str(data_yaml), 'yaml_sha256': sha(data_yaml), 'names': names, 'splits': {}}
    all_rows = {}
    for split in ('train', 'val'):
        images = root / data[split]
        labels = root / data[split].replace('images', 'labels', 1)
        rows, missing, empty, invalid = [], 0, 0, []
        for image in sorted(p for p in images.rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}):
            rel = image.relative_to(images)
            label = labels / rel.with_suffix('.txt')
            if not label.exists():
                missing += 1
                continue
            objects = []
            for line in label.read_text(encoding='utf-8').splitlines():
                if not line.strip():
                    continue
                try:
                    fields = line.split()
                    cid = int(fields[0])
                    box = list(map(float, fields[1:]))
                    assert len(box) == 4 and 0 <= cid < len(names)
                    assert all(np.isfinite(box)) and all(0 <= x <= 1 for x in box) and box[2] > 0 and box[3] > 0
                    objects.append([cid, *box])
                except (ValueError, AssertionError, IndexError):
                    invalid.append({'label': str(label), 'line': line})
            if not objects:
                empty += 1
            video, frame = _extract_video_tokens(image.stem, image.parent.name)
            rows.append({'image': str(image.relative_to(root)).replace('\\', '/'), 'label': str(label.relative_to(root)).replace('\\', '/'), 'video_id': video, 'frame_index': frame, 'gt': objects, 'label_sha256': sha(label)})
        with (OUT / f'{split}_manifest.jsonl').open('w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n')
        videos = Counter(r['video_id'] for r in rows)
        hist = Counter(o[0] for r in rows for o in r['gt'])
        gaps = Counter()
        for video in videos:
            ids = sorted(r['frame_index'] for r in rows if r['video_id'] == video and r['frame_index'] is not None)
            gaps.update(b-a for a,b in zip(ids, ids[1:]))
        report['splits'][split] = {'frames': len(rows), 'videos': len(videos), 'missing_labels': missing, 'empty_labels': empty, 'invalid_labels': invalid, 'class_histogram': dict(sorted(hist.items())), 'frame_index_gaps': dict(sorted(gaps.items())), 'manifest_sha256': sha(OUT / f'{split}_manifest.jsonl')}
        all_rows[split] = rows
        print('Dataset', split, len(rows), len(videos), 'videos', flush=True)
    report['video_id_overlap'] = sorted({r['video_id'] for r in all_rows['train']} & {r['video_id'] for r in all_rows['val']})
    # Stratified, fixed subset: up to eight frames per class, then uniform fill.
    val = all_rows['val']
    chosen = set()
    for cid in range(len(names)):
        candidates = [i for i,r in enumerate(val) if any(o[0] == cid for o in r['gt'])]
        for i in np.linspace(0, len(candidates)-1, min(8, len(candidates)), dtype=int):
            chosen.add(candidates[i])
    for i in np.linspace(0, len(val)-1, 256, dtype=int):
        if len(chosen) >= 256:
            break
        chosen.add(int(i))
    subset = [val[i] for i in sorted(chosen)]
    with (OUT / 'canonical_validation.jsonl').open('w', encoding='utf-8') as f:
        for r in subset:
            f.write(json.dumps(r, sort_keys=True, separators=(',', ':')) + '\n')
    report['canonical'] = {'frames': len(subset), 'videos': len({r['video_id'] for r in subset}), 'classes': sorted({o[0] for r in subset for o in r['gt']}), 'manifest_sha256': sha(OUT / 'canonical_validation.jsonl'), 'selection': 'eight evenly spaced occurrences per class, union, uniform fill to 256; no metric-based selection'}
    dump('dataset_audit.json', report)
    return report, subset


def inventory(names):
    configs = {}
    configs_by_directory = {}
    for p in list((ROOT/'config').rglob('*.yaml')) + list((ROOT/'trained_models').rglob('*.yaml')):
        try:
            c = load_config(str(p))
            config_path = str(p.relative_to(ROOT)).replace('\\', '/')
            configs.setdefault(c.get('train_params', {}).get('task_name', ''), []).append(config_path)
            configs_by_directory.setdefault(p.parent, []).append(config_path)
        except Exception:
            pass
    rows = []
    checkpoints = sorted(p for p in ROOT.rglob('*') if p.suffix.lower() in {'.pt', '.pth'} and '.venv' not in p.parts and '.git' not in p.parts)
    for p in checkpoints:
        if p.name == 'epoch.pth':
            continue
        rel = str(p.relative_to(ROOT)).replace('\\', '/')
        run = str(p.parent.relative_to(ROOT)).replace('\\', '/')
        row = {'checkpoint': rel, 'run_directory': run, 'config_paths': configs_by_directory.get(p.parent, configs.get(run.replace('trained_models/', ''), [])), 'size_bytes': p.stat().st_size, 'training_git_commit': 'UNKNOWN', 'training_date': 'UNKNOWN', 'training_dataset': 'UNKNOWN', 'validation_dataset': 'UNKNOWN', 'exact_class_list': [], 'class_order_evidence': 'UNKNOWN'}
        try:
            # SSD checkpoints are plain tensors/dicts. YOLO contains a serialized module.
            c = torch.load(p, map_location='cpu', weights_only=True, mmap=True)
            s = c.get('model', c)
            if not isinstance(s, dict):
                raise ValueError('Not a state dictionary')
            heads = [(k,v) for k,v in s.items() if re.search(r'cls_heads\.0\.(weight|pointwise.weight)$', k)]
            if not heads:
                row.update(family='UNKNOWN', load_status='NO_CLASSIFICATION_HEAD')
            else:
                key,tensor = heads[0]
                classes = tensor.shape[0] // 4
                family = 'ROI-SSD-MobileNet' if 'pointwise' in key else ('SSD' if '/voc-ssd/' in rel or '/imagenet-vid-ssd-' in rel else 'ROI-SSD')
                row.update(family=family, model_class_count=classes, foreground_class_count=classes-1, background_index=0, load_status='STATE_DICT_READABLE', classification_head_shape=list(tensor.shape), checkpoint_epoch=c.get('epoch'), checkpoint_stage=c.get('stage'), checkpoint_best_map=c.get('best_map'), penalized_prefix=key.startswith('model.'))
                if classes == 31:
                    row.update(exact_class_list=names, class_order_evidence='31-channel head; sampled-YOLO config/run provenance required (head shape alone does not prove order)')
                if row['config_paths']:
                    cfg = load_config(row['config_paths'][0])
                    row['training_dataset'] = cfg.get('dataset_params', {}).get('yolo_dataset_yaml', cfg.get('train_params', {}).get('dataset', 'UNKNOWN'))
                    row['validation_dataset'] = row['training_dataset'] + ' [loader test -> val]'
            del c,s
        except Exception as err:
            if 'yolo' in rel.lower() and 'roissd' not in rel.lower() and 'ssd-yolo' not in rel.lower():
                try:
                    y = YOLO(str(p))
                    nc = [y.names[i] for i in range(len(y.names))]
                    args = y.ckpt.get('train_args', {})
                    row.update(family='YOLO', architecture=y.model.yaml.get('yaml_file'), scale=y.model.yaml.get('scale'), foreground_class_count=len(nc), model_class_count=len(nc), background_index='NONE', exact_class_list=nc, class_order_evidence='embedded checkpoint names', load_status='YOLO_LOADED', training_dataset=args.get('data', 'UNKNOWN'), validation_dataset=str(args.get('data', 'UNKNOWN')) + ' [val]', checkpoint_epoch=y.ckpt.get('epoch'), training_run=args.get('name'), training_epochs=args.get('epochs'), validation_metrics=y.ckpt.get('train_metrics'), training_ultralytics=y.ckpt.get('version'), training_date=y.ckpt.get('date'), class_compatible=nc==names)
                    del y
                except Exception as yerr:
                    row.update(load_status='ERROR', error=str(yerr))
            else:
                row.update(load_status='ERROR', error=str(err))
        # Training history is run-level evidence, never assign its peak to every checkpoint.
        metrics = p.parent/'training_metrics.csv'
        results = p.parent/'results.csv'
        for source in [metrics,results]:
            if source.exists():
                rr = list(csv.DictReader(source.open(encoding='utf-8-sig')))
                rr = [r for r in rr if r.get('mAP') and (source==metrics or r.get('evaluated_dataset')=='ssd')]
                if rr:
                    b = max(rr,key=lambda r:float(r['mAP']))
                    row[str(source.name)+'_run_peak'] = b
                    row[str(source.name)+'_source'] = str(source.relative_to(ROOT)).replace('\\', '/')
        losses = p.parent/'training_losses.csv'
        if losses.exists():
            history = list(csv.DictReader(losses.open(encoding='utf-8-sig')))
            row['training_losses_source'] = str(losses.relative_to(ROOT)).replace('\\', '/')
            row['training_losses_summary'] = {'rows':len(history),'first':history[0] if history else None,'last':history[-1] if history else None,'note':'Loss is not a checkpoint ranking criterion; epoch counters may reset between phases.'}
        rows.append(row)
        print('Inventory', rel, row.get('model_class_count'), row['load_status'], flush=True)
        gc.collect()
    dump('model_candidates.json', rows)
    write_csv('model_candidates.csv', rows)
    return rows


def ssd_model(path, family, config):
    cfg = load_config(config)
    params = dict(cfg['model_params'])
    params.update(low_score_threshold=0.01, nms_threshold=0.45, detections_per_img=200)
    model = (RoiSSDMobileNet if family=='ROI-SSD-MobileNet' else RoiSSD)(config=params, num_classes=31)
    c = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    state = c.get('model',c)
    if all(k.startswith('model.') for k in state):
        state = {k[len('model.'):]:v for k,v in state.items()}
    model.load_state_dict(state, strict=True)
    return model.eval()


def evaluate(data, subset):
    from tools.infer import compute_map
    specs = [
        ('YOLO26n','trained_models/best-imagenet-vid-yolo/best.pt','config/imagenet-vid-yolo.yaml'),
        ('YOLO26n','trained_models/imagenet-vid-yolo/best.pt','config/imagenet-vid-yolo.yaml'),
        ('YOLO26n','trained_models/imagenet-vid-yolo-val/best.pt','config/imagenet-vid-yolo.yaml'),
        ('ROI-SSD','trained_models/best-imagenet-vid-roissd/imagenet26022026stage3.pt','config/imagenet-vid-roissd.yaml'),
        ('ROI-SSD','trained_models/imagenet-vid-roissd-best-v3/imagenet-vid200712stage3.pth','config/imagenet-vid-roissd.yaml'),
        ('ROI-SSD','trained_models/imagenet-vid-roissd-20261002/best_map_stage2.pt','trained_models/imagenet-vid-roissd-20261002/mixed_sampling_eval.yaml'),
        ('ROI-SSD-MobileNet','trained_models/best-imagenet-vid-roissd-mobilenet/best_map.pt','config/imagenet-vid-roissd-mobilenet.yaml'),
        ('ROI-SSD-MobileNet','trained_models/best-imagenet-vid-roissd-mobilenet/20261001-phase2-preserved-forms/best_map.pt','trained_models/best-imagenet-vid-roissd-mobilenet/20261001-phase2-preserved-forms/mixed_sampling_eval.yaml'),
        ('ROI-SSD-MobileNet','trained_models/best-imagenet-vid-roissd-mobilenet/20261001-phase3-preserved-forms/best_map.pt','trained_models/best-imagenet-vid-roissd-mobilenet/20261001-phase3-preserved-forms/mixed_sampling_eval.yaml'),
    ]
    frames, gts, diffs = [], [], []
    names = data['names']
    for r in subset:
        image = cv2.cvtColor(cv2.imread(str(Path(data['root'])/r['image'])), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(cv2.resize(image,(320,320),interpolation=cv2.INTER_LINEAR)).permute(2,0,1).float().unsqueeze(0)/255
        frames.append(x)
        gt = {n:[] for n in names}
        for cid,cx,cy,w,h in r['gt']:
            gt[names[cid]].append([cx-w/2,cy-h/2,cx+w/2,cy+h/2])
        gts.append(gt)
        diffs.append({n:[0]*len(b) for n,b in gt.items()})
    mean=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
    std=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
    results=[]
    for family,path,config in specs:
        print('Evaluate',path,flush=True)
        result={'family':family,'checkpoint':path,'config_source':config,'manifest_sha256':data['canonical']['manifest_sha256']}
        try:
            if family=='YOLO26n':
                model=YOLO(path)
                assert [model.names[i] for i in range(len(model.names))]==names
                assert model.model.yaml.get('scale')=='n' and '26' in model.model.yaml.get('yaml_file','')
            else:
                model=ssd_model(path,family,config)
            preds=[]
            with torch.inference_mode():
                for i,x in enumerate(frames):
                    if family=='YOLO26n':
                        res=model.predict(x,device='cpu',imgsz=320,conf=0.01,iou=0.45,max_det=200,verbose=False)[0]
                        boxes=res.boxes.xyxy/320
                        labels=res.boxes.cls.long()
                        scores=res.boxes.conf
                    else:
                        _, det=model((x-mean)/std,None)
                        det=det[0]
                        boxes,labels,scores=det['boxes'],det['labels'].long()-1,det['scores']
                    assert torch.isfinite(boxes).all() and torch.isfinite(scores).all()
                    assert ((labels>=0)&(labels<len(names))).all()
                    assert (boxes[:,2:]>=boxes[:,:2]).all()
                    boxes=boxes.clamp(0,1)
                    keep=torchvision.ops.batched_nms(boxes,scores,labels,0.45)[:200]
                    keep=keep[scores[keep]>=0.01]
                    pred={n:[] for n in names}
                    for j in keep.tolist():
                        pred[names[labels[j].item()]].append([*boxes[j].tolist(),scores[j].item()])
                    preds.append(pred)
                    if (i+1)%32==0:
                        print('  frames',i+1,'/',len(frames),flush=True)
                maps=[compute_map(preds,gts,iou_threshold=float(t),method='area',difficult=diffs) for t in np.arange(.5,.951,.05)]
                result.update(status='PASS',map50=maps[0][0],map50_95=float(np.mean([m[0] for m in maps])),recall50=maps[0][2],per_class_ap50=maps[0][1],checkpoint_sha256=sha(Path(path)),smoke_full_frame='PASS: real frame, batch=1 CPU; normalized boxes converted to dataset IDs')
                if family!='YOLO26n':
                    checks=[]
                    for h,w in [(96,96),(128,128),(160,160),(160,192)]:
                        xx=torch.nn.functional.interpolate(frames[0],size=(h,w),mode='bilinear',align_corners=False)
                        _,dd=model((xx-mean)/std,None)
                        d=dd[0]
                        assert torch.isfinite(d['boxes']).all()
                        assert ((d['labels']>=1)&(d['labels']<=30)).all()
                        assert (d['boxes'][:,2:]>=d['boxes'][:,:2]).all()
                        checks.append({'shape':[h,w],'status':'PASS','detections':len(d['labels']),'box_min':float(d['boxes'].min()) if d['boxes'].numel() else None,'box_max':float(d['boxes'].max()) if d['boxes'].numel() else None})
                    result['variable_input_smoke']=checks
            dump(Path(path).parent.name+'_'+Path(path).stem+'_predictions.json',preds)
            del model
        except Exception as e:
            result.update(status='ERROR',error=repr(e))
        results.append(result)
        dump('canonical_results.json',results)
        write_csv('canonical_results.csv',results)
        print('RESULT',result.get('map50'),result.get('map50_95'),result.get('status'),result.get('error',''),flush=True)
        gc.collect()
    return results


if __name__=='__main__':
    data,subset=dataset_audit()
    candidates=inventory(data['names'])
    results=evaluate(data,subset)
    dump('software.json',{'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'python':platform.python_version(),'torch':torch.__version__,'torchvision':torchvision.__version__,'ultralytics':ultralytics.__version__,'opencv':cv2.__version__,'numpy':np.__version__,'yaml':yaml.__version__,'cpu_threads':torch.get_num_threads(),'device':'cpu'})
