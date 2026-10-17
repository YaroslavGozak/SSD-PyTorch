"""Build the stage-1 manifest from completed audit evidence."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import csv,json,hashlib
import yaml
from tools.helpers.config_reader import load_config
import torch
import cv2
from ultralytics import YOLO
from final_experiments.audit.run_audit import ssd_model
from final_experiments.audit.run_audit import write_csv

OUT=ROOT/'final_experiments/audit'
def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
data=read('dataset_audit.json')
results=read('canonical_results.json')
candidates=read('model_candidates.json')
software=read('software.json')
names=data['names']
# Recover snapshots stored beside renamed runs and summarize loss-only logs.
for c in candidates:
    directory=ROOT/Path(c['checkpoint']).parent
    snapshot=directory/'mixed_sampling_eval.yaml'
    if snapshot.exists():
        c['config_paths']=[snapshot.relative_to(ROOT).as_posix()]
        cfg=load_config(str(snapshot))
        c['training_dataset']=cfg['dataset_params'].get('yolo_dataset_yaml',cfg['train_params']['dataset'])
        c['validation_dataset']=str(c['training_dataset'])+' [loader test -> val]'
    c.setdefault('training_date','UNKNOWN')
    losses=directory/'training_losses.csv'
    if losses.exists():
        history=list(csv.DictReader(losses.open(encoding='utf-8-sig')))
        c['training_losses_source']=losses.relative_to(ROOT).as_posix()
        c['training_losses_summary']={'rows':len(history),'first':history[0] if history else None,'last':history[-1] if history else None,'note':'Run-level loss history; not comparable across stages or architectures.'}
(OUT/'model_candidates.json').write_text(json.dumps(candidates,indent=2)+'\n')
write_csv('model_candidates.csv',candidates)
families=['YOLO26n','ROI-SSD','ROI-SSD-MobileNet']
assert len(results)==9 and all(r['status']=='PASS' for r in results), 'Incomplete canonical comparison; do not publish final selections'
selected={}
for family in families:
    rr=[r for r in results if r['family']==family]
    top=max(r['map50'] for r in rr)
    tied=[r for r in rr if top-r['map50']<0.005]
    selected[family]=max(tied,key=lambda r:(r['map50_95'],r['recall50']))

provenance=[]
for p in (ROOT/'outputs').glob('ca_*/metadata.json'):
    d=json.loads(p.read_text(encoding='utf-8'))
    provenance.append({'source':p.relative_to(ROOT).as_posix(),'historical_checkpoint':d.get('model',{}).get('weights'),'checkpoint_sha256':d.get('checkpoint_sha256'),'git_commit':d.get('git_commit'),'timestamp_utc':d.get('timestamp_utc'),'matching_selected_families':[f for f,r in selected.items() if r['checkpoint_sha256']==d.get('checkpoint_sha256')],'dataset':d.get('dataset'),'resolved_config':d.get('resolved_config')})
(OUT/'benchmark_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')

# Also exercise a native-resolution source frame, rather than only the
# uniformly resized control inputs used for metric comparison.
native_row=json.loads((OUT/'canonical_validation.jsonl').read_text().splitlines()[0])
native=cv2.imread(str(Path(data['root'])/native_row['image']))
h,w=native.shape[:2]
native_checks=[]
for family,r in selected.items():
    with torch.inference_mode():
        if family=='YOLO26n':
            model=YOLO(r['checkpoint'])
            det=model.predict(native,device='cpu',imgsz=320,conf=0.01,iou=0.45,max_det=200,verbose=False)[0].boxes
            boxes=det.xyxy/torch.tensor([w,h,w,h])
            labels=det.cls.long()
        else:
            model=ssd_model(r['checkpoint'],family,r['config_source'])
            x=torch.from_numpy(cv2.cvtColor(native,cv2.COLOR_BGR2RGB)).permute(2,0,1).float().unsqueeze(0)/255
            mean=torch.tensor([.485,.456,.406]).view(1,3,1,1)
            std=torch.tensor([.229,.224,.225]).view(1,3,1,1)
            _,ds=model((x-mean)/std,None)
            boxes,labels=ds[0]['boxes'],ds[0]['labels']-1
        assert torch.isfinite(boxes).all() and ((labels>=0)&(labels<30)).all()
        assert (boxes>=-1e-6).all() and (boxes<=1.000001).all()
        assert (boxes[:,2:]>=boxes[:,:2]).all()
        native_checks.append({'family':family,'checkpoint':r['checkpoint'],'status':'PASS','source_shape':[h,w],'batch':1,'device':'cpu','detections':len(labels),'coordinate_system':'normalized source xyxy after YOLO source-pixel conversion'})
        del model
(OUT/'native_full_frame_smoke.json').write_text(json.dumps(native_checks,indent=2)+'\n')

# Loader video IDs omit train/val prefixes. Retain source identity separately.
source_sets={}
for split in ['train','val']:
    videos={}
    for line in (OUT/f'{split}_manifest.jsonl').read_text(encoding='utf-8').splitlines():
        r=json.loads(line)
        source=Path(r['image']).stem.rsplit('_',1)[0]
        videos.setdefault(source,[]).append(r['frame_index'])
    source_sets[split]=set(videos)
    data['splits'][split]['source_video_count']=len(videos)
data['source_video_overlap']=sorted(source_sets['train']&source_sets['val'])
data['video_id_overlap_note']='Numeric loader IDs are not global source identifiers; train shard/split prefixes must be retained. Numeric-ID overlap is not proof of leakage.'
data['absent_class_ids']=[i for i in range(len(names)) if str(i) not in data['splits']['val']['class_histogram']]
data['absent_class_names']=[names[i] for i in data['absent_class_ids']]

current=[json.loads(l) for l in (OUT/'val_manifest.jsonl').read_text().splitlines()]
historic=json.loads((ROOT/'outputs/roi_video_benchmark/yolo_smoke_256/dataset_audit.json').read_text())
data['historical_5064_check']={'same_yaml_sha256':historic['dataset_yaml_sha256']==data['yaml_sha256'],'same_frame_count':historic['total_retained_frames']==len(current),'same_video_count':len(historic['videos'])==data['splits']['val']['videos'],'same_first_256_image_paths':[str(Path(data['root'])/r['image']) for r in current[:256]]==historic['selected_paths'],'limitations':'Historical complete image/label hashes were not retained; full byte-for-byte identity with May benchmark cannot be proven.'}

# Inspect alternative dataset membership without modifying external datasets.
alternatives=[]
val_names={Path(r['image']).name for r in current}
val_sources={Path(r['image']).stem.rsplit('_',1)[0] for r in current}
for root in sorted(Path(data['root']).parent.glob('imagenet_vid*')):
    p=root/'imagenet_vid.yaml'
    if not p.exists():continue
    cfg=yaml.safe_load(p.read_text(encoding='utf-8'))
    lists={s:{q.name for q in (root/cfg[s]).rglob('*') if q.suffix.lower() in {'.jpeg','.jpg','.png'}} for s in ['train','val']}
    alt={'root':str(root),'yaml':str(p),'class_names':cfg['names'],'train_frames':len(lists['train']),'val_frames':len(lists['val']),'train_val_filename_overlap':len(lists['train']&lists['val']),'train_overlap_common_val_frames':len(lists['train']&val_names),'train_overlap_common_val_source_videos':len({Path(n).stem.rsplit('_',1)[0] for n in lists['train']}&val_sources)}
    alternatives.append(alt)
(OUT/'alternative_datasets.json').write_text(json.dumps(alternatives,indent=2)+'\n')
(OUT/'dataset_audit.json').write_text(json.dumps(data,indent=2)+'\n')

compat=[{'dataset_class_id':i,'dataset_class_name':n,'yolo_class_id':i,'roi_ssd_class_id':i+1,'roi_ssd_mobilenet_class_id':i+1,'status':'BACKGROUND_OFFSET_ONLY','validation_gt_count':data['splits']['val']['class_histogram'].get(str(i),0)} for i,n in enumerate(names)]
with (OUT/'class_compatibility.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(compat[0]));w.writeheader();w.writerows(compat)

configs=ROOT/'final_experiments/configs'
configs.mkdir(exist_ok=True)
for family,r in selected.items():
    cfg=load_config(r['config_source'])
    cfg['dataset_params'].update(num_classes=31,im_size=320,transform_name='ssd',yolo_dataset_yaml=data['yaml'])
    task=str(Path(r['checkpoint']).parent).replace('\\','/').removeprefix('trained_models/')
    cfg['train_params'].update(task_name=task,infer_conf_threshold=0.01,model_label_space='yolo-imagenet-vid',device='cpu')
    if family=='YOLO26n':
        cfg['train_params'].update(yolo_weights=r['checkpoint'],yolo_use_predict_api=True)
        cfg['dataset_params']['num_classes']=30
    else:
        cfg['train_params']['ckpt_name']=Path(r['checkpoint']).name
        cfg['model_params'].update(low_score_threshold=0.01,nms_threshold=0.45,detections_per_img=200)
    cfg['audit_protocol']={'purpose':'Inference only; never use for retraining','canonical_manifest':'final_experiments/audit/canonical_validation.jsonl','final_manifest':'final_experiments/audit/val_manifest.jsonl','resize':'squash 320x320 RGB OpenCV INTER_LINEAR; no letterbox','normalization':'ImageNet mean/std for SSD; [0,1] for YOLO','confidence':0.01,'class_aware_nms_iou':0.45,'max_detections':200,'checkpoint_sha256':r['checkpoint_sha256']}
    file=family.lower().replace('-','_')+'.yaml'
    (configs/file).write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    r['final_config']='final_experiments/configs/'+file

full_class='; '.join(f'{i}: {n}' for i,n in enumerate(names))
notes=[
    'mAP values below are newly measured on the locked 256-frame control subset, not on the entire 5,064-frame split. This is a short selection audit; no claim of statistical significance or global-best exhaustive search.',
    'There is no distinct test split in the audited YAML: loader test aliases validation. The final 5,064-frame manifest includes the 256 selection frames, so subsequent results are validation benchmarks, not an untouched held-out test.',
    'The class schema has 30 foreground entries, but domestic_cat (9), giant_panda (12), red_panda (20) have zero supplied GT in both train and validation. AP is averaged over the 27 present classes; the three absent-class APs are undefined, not zero.',
    'The common 5,064-frame validation set can support a locked sampled-frame end-to-end benchmark. It contains 508 short source clips (consecutive index gaps of 1 within each clip); original FPS, conversion version, negative-frame completeness and representativeness against full VID are unknown. Do not claim complete raw-video or full-VID generalization.',
    'Current dataset selection uses paths, class order, hashes, loader semantics and historical artifacts, not folder names alone. May comparison2026_5_29.csv records 5,064 frames and the same dataset; September audit matches YAML hash, count, video count and the exact 256-image prefix. Complete historical label-content hashes are unavailable.',
    'Other YOLO data copies contain train/validation overlap; their embedded historical metrics are not directly comparable. See alternative_datasets.json. Historical training/tuning overlap cannot be fully reconstructed from stripped weights.',
    'SSD checkpoints do not embed class-name lists or training git hashes. Foreground order is recovered from yolo-imagenet-vid training/evaluation provenance and the loader YAML; 31-way heads corroborate dimensions, and fresh class-wise evaluation checks decoding. Legacy 34-way VID, 21-way VOC, 13-class YOLO and 80-class COCO models are excluded.',
    'The old label_spaces.IMAGENET_VID_CLASSES constant is a different 33-foreground schema. Final configs must use yolo-imagenet-vid and the audited YAML; never replace it with that constant or silently claim the schemas are identical.',
    'Current training configs point to renamed/missing runs. Final configs snapshot the actually tested inference settings and explicit checkpoints. They are reconstructed inference configs, not assertions of original training settings.',
    'Historical ca_yolo26n_cpu weights hash exactly matches the selected YOLO checkpoint. Historical ca_roissd_mobilenet_cpu phase3.pt hash exactly matches the selected MobileNet phase2 best; the phase3 filename does not establish stage-3 training. See benchmark_provenance.json.',
    'YOLO26 is end-to-end: iou passed to predict alone does not ensure class-aware NMS. This audit applies torchvision batched_nms at 0.45 to all families after decoding; SSD also has internal NMS at the same threshold. YOLO internal top-k and SSD pre_nms_topK remain architecture-specific.',
    'Native Raspberry Pi inference was not tested. Windows CUDA wheels are not a Pi package lock. The tested package versions are recorded; Pi needs CPU/aarch64 builds and another load/forward smoke before long experiments.',
]
manifest=['# Final experiments manifest','','Audit date: 2026-10-14. Stage 1 only; models and datasets were not modified or retrained.','','## Common dataset',f'- Dataset name: YOLO-converted sampled ImageNet-VID (`yolo-imagenet-vid`).',f'- Dataset root: `{data["root"]}`.', '- Train manifest: [train_manifest.jsonl](audit/train_manifest.jsonl).', '- Validation manifest: [val_manifest.jsonl](audit/val_manifest.jsonl); loader `test` and `val` both resolve YAML `val`.', '- Final evaluation manifest: [val_manifest.jsonl](audit/val_manifest.jsonl), frozen ordered 5,064-frame split.','- Canonical selection manifest: [canonical_validation.jsonl](audit/canonical_validation.jsonl), 256 frames / '+str(data['canonical']['videos'])+' loader video IDs.',f'- Number of videos: validation {data["splits"]["val"]["videos"]}; train {data["splits"]["train"]["source_video_count"]} source IDs ({data["splits"]["train"]["videos"]} numeric loader IDs).',f'- Number of frames: train {data["splits"]["train"]["frames"]}; validation/final 5064.', '- Foreground class count: 30 schema entries; 27 GT-supported entries in supplied splits.',f'- Exact class list: {full_class}.','- Annotation format: YOLO TXT, zero-based foreground ID + normalized center-x/center-y/width/height; manifests preserve GT, relative paths and label SHA256.',f'- Notes: validation manifest SHA256 `{data["splits"]["val"]["manifest_sha256"]}`; canonical SHA256 `{data["canonical"]["manifest_sha256"]}`. No missing, empty or invalid label files. See [dataset audit](audit/dataset_audit.json).','']
for family,r in selected.items():
    cand=next(c for c in candidates if c['checkpoint']==r['checkpoint'])
    training=cand['training_dataset']
    if training=='UNKNOWN':training='Recovered sampled ImageNet-VID provenance; exact original training config not embedded.'
    manifest += [f'## {family}',f'- Selected run: `{Path(r["checkpoint"]).parent.as_posix()}`.',f'- Checkpoint: [{Path(r["checkpoint"]).name}](../{r["checkpoint"]}).',f'- Checkpoint SHA256: `{r["checkpoint_sha256"]}`.',f'- Config: [locked inference YAML](configs/{Path(r["final_config"]).name}); recovered from [{r["config_source"]}](../{r["config_source"]}).',f'- Training dataset: `{training}`.','- Validation dataset: common sampled ImageNet-VID validation, fixed 256-frame canonical control subset.','- Foreground class count: 30.']
    if family!='YOLO26n':manifest += ['- Background handling: 31 outputs including background index 0; dataset foreground ID i becomes internal SSD ID i+1.']
    manifest += [f'- Exact {"foreground " if family!="YOLO26n" else ""}class list: {full_class}.',f'- Validation mAP@0.5: **{r["map50"]:.8f}**.',f'- Validation mAP@0.5:0.95: **{r["map50_95"]:.8f}** (mean of ten area-integral AP metrics; not COCO 101-point interpolation).',f'- Validation Recall@0.5: **{r["recall50"]:.8f}**, mean class recall at confidence 0.01.', '- Selection reason: winner among the three compatible shortlisted checkpoints on identical control frames, pixel resize, confidence, output NMS, class set and metric implementation; mAP50 primary, mAP50:95 for differences <0.005, then recall.','- Smoke: strict checkpoint load; real full-frame image at native source resolution and at 320x320, batch=1 CPU, finite decoded predictions, valid IDs and source-coordinate bounds; SSD variable inputs 96x96, 128x128, 160x160, 160x192 passed. [Native evidence](audit/native_full_frame_smoke.json).','']
manifest += ['## Class compatibility','- Status: `BACKGROUND_OFFSET_ONLY` for all 30 foreground schema entries.','- Mapping required: no semantic class mapping; convert SSD internal labels back to zero-based dataset IDs by subtracting 1.','- Background offset: +1 inside SSD; YOLO has no background output. The existing loader adds 1; existing YOLO adapters also add 1 for project-format output.','- Full compatibility table: [class_compatibility.csv](audit/class_compatibility.csv).','','## Software']
manifest += [f'- {k}: `{v}`.' for k,v in software.items()]
manifest += ['','## Unresolved issues']+[f'- {n}' for n in notes]
manifest += ['','## Stage gate','- Stage-1 selection and local CPU smoke: PASS. One explicit checkpoint per family is locked.','- Stop if a checkpoint/hash, dataset YAML/class order or frame manifest differs; never rediscover best.pt by name.','- Stage 2 must consume these files and preserve the documented scope. Pi smoke is still required before Pi experiments. No stage-2 anomaly diagnosis or long benchmark was run.','']
(ROOT/'final_experiments/FINAL_EXPERIMENTS_MANIFEST.md').write_text('\n'.join(manifest),encoding='utf-8')
(OUT/'selected_models.json').write_text(json.dumps({'status':'PASS_LOCAL_CPU','models':selected,'canonical_manifest_sha256':data['canonical']['manifest_sha256'],'dataset_root':data['root'],'class_names':names},indent=2)+'\n')

table=['| Family | Checkpoint | mAP50 | mAP50:95 | Recall50 | Selected |','|---|---|---:|---:|---:|---|']
for r in results:
    table.append(f'| {r["family"]} | [{r["checkpoint"]}](../../{r["checkpoint"]}) | {r["map50"]:.6f} | {r["map50_95"]:.6f} | {r["recall50"]:.6f} | {"yes" if selected[r["family"]]["checkpoint"]==r["checkpoint"] else "no"} |')
report=['# Stage 1 model and dataset audit','','## Result','Three checkpoints selected using fresh canonical evaluation; see [manifest](../FINAL_EXPERIMENTS_MANIFEST.md). All nine candidate checkpoints loaded and passed real-image CPU and decoded-output checks. Models, training configs and external datasets were not changed.','','## Reproduction','Run from repository root:','','```powershell','.venv/Scripts/python.exe -u final_experiments/audit/run_audit.py','.venv/Scripts/python.exe final_experiments/audit/build_report.py','```','','The runner scans checkpoints across the repository (excluding epoch counters and environments), audits all labels in both common splits, writes immutable-order manifests, and compares the fixed shortlists. Stored predictions support metric rechecking without another inference pass.','','## Protocol','256 frames selected from label occurrences and uniform fill, before inspecting prediction metrics. 228 loader videos, all 27 available GT classes. CPU, four torch threads, batch=1. RGB, OpenCV bilinear squash to 320x320, divide by 255; ImageNet normalization only for SSD. Confidence 0.01, class-aware output NMS 0.45, maximum 200 predictions, SSD pre-NMS top-k 400. Existing tools.infer.compute_map area-integral AP at IoU 0.50:0.05:0.95 is shared by every model. Scores are not the old Ultralytics validation scores.','','## Ranking']+table+['','## Shortlist rationale','YOLO: three 30-class YOLO26n final/best checkpoints with distinct training-dataset provenance; best50 is a predecessor when best.pt exists, 13-class and VOC checkpoints are incompatible. SSD: completed stage-3 runs with strongest available saved metrics and stage-2 best with auditable epoch metadata; penalized wrappers do not establish a separate better trained state, and October stage3 has only 3/20 epochs and is excluded. MobileNet: preserved stage1 best and preserved-forms stage2/stage3 best; older runs have lower logged mAP, unfreezed phase2 has worse metrics, and phase3-bn-freeze peaks at 0.6692. Old logs do not bind every peak to every saved checkpoint; fresh common evaluation is the selection evidence.','','## Provenance and dataset checks',f'- Checkpoints inventoried: {len(candidates)}. [CSV](model_candidates.csv), [full JSON](model_candidates.json).','- Epoch.pth files are resume counters, not models. Embedded YOLO names and architecture are inspected; SSD class counts are checked against actual classification convolution output dimensions. State-dictionary readability is distinguished from strict instantiated-model loading.', '- Historical benchmark weight hashes and paths: outputs/ca_yolo26n_cpu/metadata.json, outputs/ca_roissd_cpu/metadata.json, outputs/ca_roissd_mobilenet_cpu/metadata.json. Names can be stale after folder renames; hashes remain authoritative.','- Common validation: 5,064 frames, 508 videos, no missing/empty/invalid labels. Common train: 64,265 frames. Source IDs include split/shard prefixes; do not infer leakage from reused numeric IDs.','- Dataset subset variants and train/validation overlaps: [alternative_datasets.json](alternative_datasets.json).','- Background-inclusive 31 outputs and explicit YAML ordering are required; old 34-output or VOC checkpoints cannot cover the same foreground schema.','','## Limitations']+[f'- {n}' for n in notes]+['','## Smoke evidence','[canonical_results.json](canonical_results.json) contains per-class AP50, full-frame load/forward status, checkpoint SHA256 and all four SSD variable-size checks for every candidate. Variable-size smoke checks feasibility only; no conclusion about the 128x128 anomaly is made.','','## Outputs','- [class compatibility CSV](class_compatibility.csv)','- [canonical result CSV](canonical_results.csv)','- [dataset audit](dataset_audit.json)','- [software versions](software.json)','- [machine-readable selections](selected_models.json)','- [execution log](execution.log)','']
(OUT/'audit_report.md').write_text('\n'.join(report),encoding='utf-8')
print('Selected:',{k:(v['checkpoint'],v['map50']) for k,v in selected.items()})
