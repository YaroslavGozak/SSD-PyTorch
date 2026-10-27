"""Independently derive global object outcomes from saved frame predictions."""
import json

from .crop_domain import OUT, write_csv, unique_matches
from .crop_domain_report import read
from .run_pilot import ROOT, _sha256
from tools.benchmarks.video_artifacts import write_json


def main():
    data=json.loads((OUT/'predictions.json').read_text())
    context=read('per_object_context_trajectories.csv')
    frame_keys=list(dict.fromkeys((r['video_id'],r['frame_index']) for r in context))
    assert len(frame_keys)==320
    matches={}
    for variant,frames in data['predictions'].items():
        for key,pred,gt in zip(frame_keys,frames,data['ground_truth']):
            dets=[{'class':cls,'bbox':box[:4],'confidence':box[4]} for cls,boxes in pred.items() for box in boxes]
            matches[(variant,*key)]=unique_matches(dets,gt)
    for filename in ('per_object_context_trajectories.csv','preprocessing_comparison.csv','clipped_vs_padded.csv'):
        rows=read(filename)
        for r in rows:
            variant=r['variant']
            if variant in ('CURRENT_PIPELINE','CLIPPED'):variant='P'+r['padding']
            if variant=='PADDED_OUTSIDE_FRAME':variant='PADDED_P96'
            selected=matches[(variant,r['video_id'],r['frame_index'])]
            r['TP50_after_global_nms']=(r['class'],int(r['class_gt_index'])) in selected
            # Pre-finalization files stored the first overlapping prediction as
            # an optional frame candidate. Own-crop matched-head attribution is
            # authoritative; do not mislabel candidates as unique global matches.
            if 'frame_matched_prediction_head' in r:
                r['frame_overlap_candidate_head']=r.pop('frame_matched_prediction_head')
            if 'frame_matched_origin_object_id' in r:
                r['frame_overlap_candidate_origin_object_id']=r.pop('frame_matched_origin_object_id')
        write_csv(filename,rows)
    assert sum(r['TP50_after_global_nms'] for r in read_bool_context())==sum(
        len(matches[('P96',*key)]) for key in frame_keys)
    meta=json.loads((OUT/'metadata.json').read_text())
    meta.setdefault('inference_source_sha256',dict(meta['source_sha256']))
    meta['source_sha256']['experiments/cpu_pilot/crop_domain.py']=_sha256(ROOT/'experiments/cpu_pilot/crop_domain.py')
    meta['source_sha256']['experiments/cpu_pilot/crop_domain_validate.py']=_sha256(ROOT/'experiments/cpu_pilot/crop_domain_validate.py')
    meta['global_object_outcomes_independently_verified']=True
    meta['global_head_candidate_fields_not_used_for_attribution']=True
    write_json(OUT/'metadata.json',meta)
    print('Global TP flags verified from saved detections; no additional inference or production mutation')


def read_bool_context():
    return [dict(r,TP50_after_global_nms=r['TP50_after_global_nms']=='True')
            for r in read('per_object_context_trajectories.csv') if r['padding']=='96']


if __name__=='__main__':main()
