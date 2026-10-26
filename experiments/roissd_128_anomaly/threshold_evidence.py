"""Validate persisted stage-2d outputs and print compact paired evidence."""
import csv
import hashlib
import json
from pathlib import Path

from .all_thresholds import OUT, PRIOR, plan, verify_sources


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def paired_rows(rows):
    pairs = {}
    for row in rows:
        key = (row['transition'], int(row['input_h']), int(row['input_w']))
        pairs.setdefault(key, []).append(row)
    for key, pair in pairs.items():
        pair.sort(key=lambda row: int(row['forced_L_R']))
        assert len(pair) == 2, f'Missing adjacent-depth pair: {key}'
        assert int(pair[1]['forced_L_R']) == int(pair[0]['forced_L_R']) + 1
        assert pair[0]['input_gt_sha256'] == pair[1]['input_gt_sha256']
    return pairs


def validate(out=OUT):
    metadata = json.loads((out / 'all_thresholds_metadata.json').read_text())
    assert metadata['status'] == 'COMPLETE'
    jobs = plan()
    results = read_csv(out / 'all_thresholds_results.csv')
    latencies = read_csv(out / 'latency_summary.csv')
    assert len(results) == metadata['num_configurations'] == metadata['planned_configurations'] == 2 * len(jobs)
    expected = {(job['transition'], job['input_h'], job['input_w'], depth)
                for job in jobs for depth in (job['lower_depth'], job['upper_depth'])}
    assert {(r['transition'], int(r['input_h']), int(r['input_w']), int(r['forced_L_R']))
            for r in results} == expected
    for filename, key in (('all_thresholds_results.csv', 'results_sha256'),
                           ('latency_summary.csv', 'latency_sha256'),
                           ('input_hashes.jsonl', 'input_hashes_sha256')):
        assert hashlib.sha256((out / filename).read_bytes()).hexdigest() == metadata[key]
    verify_sources(metadata['source_sha256'])
    assert hashlib.sha256((PRIOR / 'depth_boundary_metadata.json').read_bytes()).hexdigest() == metadata['stage2c_metadata_sha256']
    assert hashlib.sha256((PRIOR / 'depth_boundary_validation.csv').read_bytes()).hexdigest() == metadata['stage2c_csv_sha256']
    hashes = {}
    for line in (out / 'input_hashes.jsonl').read_text().splitlines():
        row = json.loads(line)
        hashes.setdefault((row['input_h'], row['input_w']), []).append(row)
    assert len(hashes) == len(jobs)
    for records in hashes.values():
        assert len(records) == 256 and len({r['sample_id'] for r in records}) == 256
    for row in results:
        assert row['slot_counts_equal'] == 'True' or row['status'] == 'INFEASIBLE'
        if row['status'] == 'INFEASIBLE':
            assert row['error'] and int(row['num_samples']) == 0
            continue
        assert row['status'] == 'PASS' and int(row['num_samples']) == 256
        height, width, depth = (int(row[k]) for k in ('input_h', 'input_w', 'forced_L_R'))
        assert int(row['min_dim']) == min(height, width)
        assert int(row['actual_L_R']) == depth
        heads, shapes, anchors = (json.loads(row[k]) for k in (
            'active_heads', 'feature_map_shapes', 'anchors_per_head'))
        assert heads == list(range(depth)) and len(shapes) == len(anchors) == depth
        for shape, count, per_cell in zip(shapes, anchors, (4, 6, 6, 6, 4, 4)):
            assert len(shape) == 4 and shape[0] == 1
            assert shape[2] * shape[3] * per_cell == count
        assert int(row['N_loc_predictions']) == int(row['N_cls_predictions']) == int(row['N_anchors']) == sum(anchors)
        for key in ('map50', 'recall50', 'mean_raw_max_confidence', 'mean_best_anchor_iou',
                    'fraction_gt_best_anchor_iou_ge_05'):
            assert 0 <= float(row[key]) <= 1
        records = hashes[height, width]
        digest = hashlib.sha256(''.join(r['tensor_gt_sha256'] for r in records).encode('ascii')).hexdigest()
        assert digest == row['input_gt_sha256']
    pairs = paired_rows(results)
    for pair in pairs.values():
        if all(r['status'] == 'PASS' for r in pair):
            assert pair[0]['num_gt'] == pair[1]['num_gt']
            for key in ('map50', 'recall50'):
                delta = float(pair[1][key]) - float(pair[0][key])
                assert abs(delta - float(pair[0]['delta_' + key + '_upper_minus_lower'])) < 1e-12
    expected_timing = {(r['transition'], r['input_h'], r['input_w'], r['forced_L_R'], metadata['accuracy_device'])
                       for r in results if r['status'] == 'PASS'}
    expected_timing |= {(r['transition'], r['input_h'], r['input_w'], r['forced_L_R'], 'cpu')
                        for r in results if r['status'] == 'PASS' and r['cpu_representative'] == 'True'}
    assert {(r['transition'], r['input_h'], r['input_w'], r['forced_L_R'], r['latency_device'])
            for r in latencies} == expected_timing
    assert len(latencies) == len(expected_timing) == metadata['num_latency_configurations']
    for row in latencies:
        values = json.loads(row['latency_samples_ms'])
        assert len(values) == int(row['timed_forwards']) == metadata['latency_repeats']
        assert int(row['warmup_forwards']) == metadata['latency_warmup']
        assert int(row['slot_checked_forwards']) == len(values) + int(row['warmup_forwards'])
        assert min(values) > 0
        assert 0 < float(row['median_latency_ms']) <= float(row['p95_latency_ms'])
    return results, latencies, metadata


def print_evidence():
    results, _, metadata = validate()
    pairs = paired_rows(results)
    for transition in dict.fromkeys(row['transition'] for row in results):
        print('\n' + transition)
        for (name, height, width), (lower, upper) in pairs.items():
            if name != transition:
                continue
            if lower['status'] != 'PASS' or upper['status'] != 'PASS':
                print(height, width, lower['status'], upper['status'])
                continue
            delta_ap = 100 * (float(upper['map50']) - float(lower['map50']))
            delta_recall = 100 * (float(upper['recall50']) - float(lower['recall50']))
            cpu = ''
            if lower.get('cpu_median_latency_ms'):
                lo, hi = (float(row['cpu_median_latency_ms']) for row in (lower, upper))
                cpu = f' CPU {lo:.2f}->{hi:.2f} ms ({100*(hi/lo-1):+.1f}%)'
            print(f'{height}x{width}: AP {100*float(lower["map50"]):.2f}->{100*float(upper["map50"]):.2f}; '
                  f'R {100*float(lower["recall50"]):.2f}->{100*float(upper["recall50"]):.2f}; '
                  f'delta {delta_ap:+.2f}/{delta_recall:+.2f} pp{cpu}')
    print('\nVERIFIED', len(results), 'configurations;', metadata['stage2c_controls_checked'], 'prior controls')


if __name__ == '__main__':
    print_evidence()
