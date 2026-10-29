"""Per-batch rectangular sampling records and per-epoch distributions."""
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np


class DeploymentSamplingStats:
    def __init__(self,stage,config):
        self.stage=stage;self.config=config;self.batch_rows=[];self.sample_rows=[]

    def record(self,mode,shape,sampling_info):
        samples=sampling_info['samples']
        h,w=map(int,shape)
        if not samples or any((r['stage'],r['mode'],r['batch_h'],r['batch_w'])!=(self.stage,mode,h,w) for r in samples):
            raise ValueError('Statistics do not match the actual rectangular batch')
        first=samples[0]
        mean=lambda key:float(np.mean([r[key] for r in samples if r.get(key) is not None])) if any(r.get(key) is not None for r in samples) else None
        row={'stage':self.stage,'epoch':first['epoch'],'batch_index':first['batch_index'],
             'mode':mode,'batch_h':h,'batch_w':w,'crop_area_fraction':first['crop_area_fraction'],
             'aspect_ratio':w/h,'area_bin':first['area_bin'],'batch_size':len(samples),
             'reference_gt_area_ratio':mean('reference_gt_area_ratio'),
             'reference_center_offset':mean('reference_center_offset'),
             'reference_visibility':mean('reference_visibility'),
             'resampling_attempts':sum(r['resampling_attempts'] for r in samples),
             'max_sample_resampling_attempts':max(r['resampling_attempts'] for r in samples)}
        self.batch_rows.append(row);self.sample_rows.extend(dict(r) for r in samples)

    @staticmethod
    def distribution(values):
        if not values:return {'N':0}
        array=np.asarray(values,dtype=float)
        return {'N':len(values),'min':float(array.min()),'p05':float(np.quantile(array,.05)),
                'median':float(np.median(array)),'mean':float(array.mean()),
                'p95':float(np.quantile(array,.95)),'max':float(array.max())}

    def summary(self):
        total=len(self.batch_rows)
        roi=[r for r in self.batch_rows if r['mode']=='roi']
        full=total-len(roi)
        count=Counter(r['area_bin'] for r in roi)
        return {'stage':self.stage,'batches':total,'samples':len(self.sample_rows),
                'full_batches':full,'roi_batches':len(roi),'full_share':full/max(1,total),
                'roi_share':len(roi)/max(1,total),
                'area_bin_shares':[{'bin':list(b[:2]),'batches':count[i],
                                   'conditional_roi_share':count[i]/max(1,len(roi)),
                                   'absolute_batch_share':count[i]/max(1,total)}
                                  for i,b in enumerate(self.config.roi_area_fraction_bins)],
                'rectangular_batch_share':sum(r['batch_h']!=r['batch_w'] for r in self.batch_rows)/max(1,total),
                'rectangular_roi_share':sum(r['batch_h']!=r['batch_w'] for r in roi)/max(1,len(roi)),
                'HW_counts':dict(Counter(f'{r["batch_h"]}x{r["batch_w"]}' for r in self.batch_rows)),
                'height_distribution':self.distribution([r['batch_h'] for r in roi]),
                'width_distribution':self.distribution([r['batch_w'] for r in roi]),
                'aspect_ratio_distribution':self.distribution([r['aspect_ratio'] for r in roi]),
                'actual_roi_area_fraction_distribution':self.distribution([r['crop_area_fraction'] for r in roi]),
                'reference_gt_area_ratio_distribution':self.distribution([r['reference_gt_area_ratio'] for r in self.sample_rows if r['reference_gt_area_ratio'] is not None]),
                'reference_center_offset_distribution':self.distribution([r['reference_center_offset'] for r in self.sample_rows if r['reference_center_offset'] is not None]),
                'reference_visibility_distribution':self.distribution([r['reference_visibility'] for r in self.sample_rows if r['reference_visibility'] is not None]),
                'resampling_attempts_distribution':self.distribution([r['resampling_attempts'] for r in self.sample_rows])}

    def save(self,directory,epoch):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        prefix=directory/f'deployment_sampling_stage{self.stage}_epoch{epoch:04d}'
        for suffix,rows in (('batches',self.batch_rows),('samples',self.sample_rows)):
            if not rows:continue
            fields=list(dict.fromkeys(k for r in rows for k in r))
            with Path(str(prefix)+f'_{suffix}.csv').open('w',newline='',encoding='utf-8') as stream:
                writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        Path(str(prefix)+'_summary.json').write_text(json.dumps(self.summary(),indent=2,allow_nan=False)+'\n',encoding='utf-8')
