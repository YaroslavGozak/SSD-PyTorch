"""Interleaved production-mode CPU timing to reduce ordering/thermal bias."""
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from model.roissd import RoiSSD
from .run import ROOT, HERE, DATA, crop, prepare, csv_rows, sha
from tools.helpers.config_reader import load_config
from .vgg16_fixed_regression import OUT, legacy_class, build


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    settings = yaml.safe_load((HERE / 'config.yaml').read_text())
    lock = prepare(settings)
    entry, = [e for e in lock['models'] if e['model_family'] == 'ROI-SSD']
    cfg = load_config(str(DATA / entry['config_snapshot']))
    params = dict(cfg['model_params'], low_score_threshold=settings['confidence'],
                  nms_threshold=settings['nms_iou'], detections_per_img=settings['max_detections'])
    state = torch.load(ROOT / entry['checkpoint'], map_location='cpu', weights_only=True)
    state = state.get('model', state)
    old_cls, _ = legacy_class()
    old, new = build(old_cls, params, state, 'cpu'), build(RoiSSD, params, state, 'cpu')
    row = json.loads((DATA / 'subset.jsonl').read_text().splitlines()[0])
    path = Path(lock['dataset_root']) / row['image']
    assert sha(path) == row['image_sha256']
    image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    results = []
    with torch.inference_mode():
        for n in (96, 128, 140, 256, 300):
            tensor = crop(image, row, (n, n), settings['seed'])[0]
            times = {'old': [], 'new': []}
            for i in range(35):
                order = [('old', old), ('new', new)]
                if i % 2:
                    order.reverse()
                for name, model in order:
                    start = time.perf_counter_ns()
                    model(tensor)
                    if i >= 5:
                        times[name].append((time.perf_counter_ns()-start)/1e6)
            result = {'input_h': n, 'input_w': n, 'threads': 4, 'repeats': 30,
                      'warmups': 5, 'record_execution_trace': False}
            for name, samples in times.items():
                result[name+'_median_latency_ms'] = statistics.median(samples)
                result[name+'_p95_latency_ms'] = float(np.percentile(samples, 95))
            result['median_ratio_new_old'] = result['new_median_latency_ms']/result['old_median_latency_ms']
            results.append(result)
            print(result, flush=True)
    csv_rows(OUT / 'latency_interleaved.csv', results)


if __name__ == '__main__':
    main()
