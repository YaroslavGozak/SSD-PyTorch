"""Reproducible policy aggregation, GT-ratio bins/correlations and integrity checks."""
from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .feature_map_policy import OUT, POLICIES, STAGE2D, legacy_depth, reference_depth, shapes
from .run import ROOT, csv_rows, sha, write_json
from .threshold_evidence import read_csv


RATIOS = ('gt_area_ratio', 'gt_width_ratio', 'gt_height_ratio', 'gt_max_side_ratio')
BINS = ((0., .25, '<0.25'), (.25, .5, '0.25-0.50'), (.5, .7, '0.50-0.70'),
        (.7, .85, '0.70-0.85'), (.85, .95, '0.85-0.95'), (.95, 1.000001, '>=0.95'))
OUTCOMES = ('delta_success50', 'delta_max_class_confidence', 'delta_best_predicted_iou',
            'delta_best_anchor_iou', 'delta_crop_raw_max_confidence')


def num(row, key):
    return float(row[key])


def in_subset(row, subset):
    if subset == 'all_pairs':
        return True
    if subset == 'single_added_1x1':
        return row['upper_adds_1x1'] == 'True' and int(row['num_added_heads']) == 1
    if subset == 'all_added_heads_1x1':
        return row['all_added_heads_1x1'] == 'True'
    raise ValueError(f'Unknown pair subset: {subset}')


def describe_group(records):
    if not records:
        return {'status': 'EMPTY', 'N_GT': 0, 'N_source_samples': 0, 'N_shapes': 0}
    result = {
        'status': 'PASS', 'N_GT': len(records),
        'N_source_samples': len({row['sample_id'] for row in records}),
        'N_shapes': len({(row['input_h'], row['input_w']) for row in records}),
    }
    for name in ('success50', 'max_class_confidence', 'best_predicted_iou',
                 'best_anchor_iou', 'crop_raw_max_confidence'):
        for suffix in ('lower', 'upper'):
            result[name + '_' + suffix + '_mean'] = float(np.mean(
                [num(row, name + '_' + suffix) for row in records]))
        result['mean_delta_' + name] = float(np.mean([num(row, 'delta_' + name) for row in records]))
    result.update(
        gained_GT=sum(num(row, 'delta_success50') > 0 for row in records),
        lost_GT=sum(num(row, 'delta_success50') < 0 for row in records),
    )
    return result


def ratio_bins(comparisons):
    result = []
    for kind in dict.fromkeys(row['pair_kind'] for row in comparisons):
        base = [row for row in comparisons if row['pair_kind'] == kind]
        for subset in ('all_pairs', 'single_added_1x1', 'all_added_heads_1x1'):
            selected = [row for row in base if in_subset(row, subset)]
            for ratio in RATIOS:
                ranges = list(BINS)
                if ratio == 'gt_area_ratio':
                    ranges.extend(((0., .8, '<0.8'), (.8, 1.000001, '>=0.8')))
                if ratio == 'gt_max_side_ratio':
                    ranges.extend(((0., .9, '<0.9'), (.9, 1.000001, '>=0.9')))
                for lower, upper, label in ranges:
                    group = [row for row in selected if lower <= num(row, ratio) < upper]
                    result.append({
                        'pair_kind': kind, 'pair_subset': subset, 'ratio': ratio, 'bin': label,
                        'bin_lower_inclusive': lower, 'bin_upper_exclusive': upper,
                        **describe_group(group),
                    })
    return result


def ranks(values):
    _, indices, counts = np.unique(values, return_inverse=True, return_counts=True)
    positions = np.cumsum(counts) - (counts - 1) / 2
    return positions[indices]


def correlation(x, y):
    if len(x) < 3 or np.ptp(x) < 1e-12 or np.ptp(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def correlations(comparisons):
    result = []
    for kind in dict.fromkeys(row['pair_kind'] for row in comparisons):
        for subset in ('all_pairs', 'single_added_1x1', 'all_added_heads_1x1'):
            selected = [row for row in comparisons if row['pair_kind'] == kind and in_subset(row, subset)]
            for population in ('all_GT', 'primary_GT_only'):
                records = [row for row in selected if population == 'all_GT' or int(row['gt_index']) == 0]
                groups = {}
                for index, row in enumerate(records):
                    groups.setdefault((row['input_h'], row['input_w']), []).append(index)
                for ratio in RATIOS:
                    x = np.array([num(row, ratio) for row in records])
                    for outcome in OUTCOMES:
                        y = np.array([num(row, outcome) for row in records])
                        cx, cy = x.copy(), y.copy()
                        for indices in groups.values():
                            cx[indices] -= np.mean(x[indices])
                            cy[indices] -= np.mean(y[indices])
                        pearson, spearman = correlation(x, y), correlation(ranks(x), ranks(y))
                        centered = correlation(cx, cy)
                        result.append({
                            'pair_kind': kind, 'pair_subset': subset, 'population': population,
                            'ratio': ratio, 'outcome': outcome, 'N_GT': len(records),
                            'pearson_r': pearson, 'spearman_r': spearman,
                            'within_shape_centered_pearson_r': centered,
                            'status': 'PASS' if pearson is not None else 'UNDEFINED_CONSTANT_OR_TOO_SMALL',
                        })
    return result


def summaries(per_shape):
    keyed = {(int(row['input_h']), int(row['input_w']), row['policy']): row for row in per_shape}
    result = []
    for policy in POLICIES:
        records = [row for row in per_shape if row['policy'] == policy]
        record = {'policy': policy, 'num_shapes': len(records), 'aggregation': 'unweighted_shape_mean_not_pooled_AP'}
        for name in ('map50', 'recall50', 'mean_raw_max_confidence', 'mean_best_anchor_iou',
                     'fraction_gt_best_anchor_iou_ge_05', 'micro_gt_recall50',
                     'cpu_median_latency_ms', 'cpu_p95_latency_ms'):
            record['shape_mean_' + name] = float(np.mean([num(row, name) for row in records]))
        record['selected_depth_distribution'] = dict(sorted(Counter(
            row['selected_L_R'] for row in records).items()))
        for reference in ('legacy', 'best_adjacent_reference'):
            deltas = [(num(row, 'map50') - num(keyed[int(row['input_h']), int(row['input_w']), reference], 'map50'),
                       num(row, 'recall50') - num(keyed[int(row['input_h']), int(row['input_w']), reference], 'recall50'))
                      for row in records]
            record['materially_worse_shapes_vs_' + reference] = sum(
                da < -.005 or dr < -.005 for da, dr in deltas)
            record['practically_equivalent_shapes_vs_' + reference] = sum(
                abs(da) <= .005 and abs(dr) <= .005 for da, dr in deltas)
            record['material_gain_shapes_vs_' + reference] = sum(
                (da > .005 or dr > .005) and da >= -.005 and dr >= -.005 for da, dr in deltas)
            record['worst_delta_map50_vs_' + reference] = min(da for da, _ in deltas)
            record['worst_delta_recall50_vs_' + reference] = min(dr for _, dr in deltas)
        result.append(record)
    return result


def validate(out=OUT):
    metadata = json.loads((out / 'metadata.json').read_text())
    assert metadata['status'] == 'COMPLETE'
    for name, digest in metadata['source_sha256'].items():
        assert sha(ROOT / name) == digest, f'Source drift: {name}'
    for name, digest in metadata['output_sha256'].items():
        assert sha(out / name) == digest, f'Output changed: {name}'
    assert sha(STAGE2D / 'all_thresholds_metadata.json') == metadata['stage2d_metadata_sha256']
    assert sha(STAGE2D / 'all_thresholds_results.csv') == metadata['stage2d_results_sha256']
    per_shape = read_csv(out / 'per_shape_results.csv')
    assert len(per_shape) == 144
    assert {(int(row['input_h']), int(row['input_w']), row['policy']) for row in per_shape} == {
        (h, w, policy) for h, w in shapes() for policy in POLICIES}
    prior = {(int(r['input_h']), int(r['input_w']), int(r['forced_L_R'])): r
             for r in read_csv(STAGE2D / 'all_thresholds_results.csv')}
    probes = {(r['input_h'], r['input_w']): r for r in json.loads((out / 'geometry_probes.json').read_text())}
    for row in per_shape:
        h, w, depth = (int(row[name]) for name in ('input_h', 'input_w', 'selected_L_R'))
        layouts = probes[h, w]['layouts']
        expected_max = max(map(int, layouts))
        expected_until = next((i for i in sorted(map(int, layouts))
                               if layouts[str(i)]['feature_map_shapes'][-1][-2:] == [1, 1]), expected_max)
        if row['policy'] == 'legacy':
            assert depth == legacy_depth(min(h, w))
        elif row['policy'] == 'max_feasible_depth':
            assert depth == expected_max
        elif row['policy'] == 'until_1x1':
            assert depth == expected_until
        else:
            pair = sorted([r for (ph, pw, _), r in prior.items() if (ph, pw) == (h, w)],
                          key=lambda r: int(r['forced_L_R']))
            assert depth == reference_depth(*pair)
        assert row['slot_counts_equal'] == 'True'
        assert int(row['N_loc_predictions']) == int(row['N_cls_predictions']) == int(row['N_anchors'])
        assert sum(json.loads(row['anchors_per_head'])) == int(row['N_anchors'])
        assert json.loads(row['active_heads']) == list(range(depth))
        assert float(row['cpu_median_latency_ms']) > 0
        assert float(row['cpu_p95_latency_ms']) >= float(row['cpu_median_latency_ms'])
    geometry_rows = read_csv(out / 'per_object_crop_geometry.csv')
    assert len(geometry_rows) == metadata['num_geometry_objects']
    geometry_by_key = {}
    for row in geometry_rows:
        key = tuple(int(row[k]) for k in ('input_h', 'input_w', 'sample_id', 'gt_index'))
        assert key not in geometry_by_key
        geometry_by_key[key] = row
        w, h, cw, ch = (num(row, k) for k in ('gt_w', 'gt_h', 'crop_w', 'crop_h'))
        assert abs(w * h - num(row, 'gt_area')) < 1e-6
        assert cw * ch == num(row, 'crop_area')
        assert abs(w / cw - num(row, 'gt_width_ratio')) < 1e-10
        assert abs(h / ch - num(row, 'gt_height_ratio')) < 1e-10
        assert abs(w * h / cw / ch - num(row, 'gt_area_ratio')) < 1e-10
        assert 0 < num(row, 'gt_area_ratio') <= num(row, 'gt_max_side_ratio') <= 1
    comparisons = read_csv(out / 'per_object_pair_comparisons.csv')
    assert len(comparisons) == metadata['num_object_pair_comparisons']
    for row in comparisons:
        key = tuple(int(row[k]) for k in ('input_h', 'input_w', 'sample_id', 'gt_index'))
        assert all(abs(num(row, k) - num(geometry_by_key[key], k)) < 1e-12 for k in RATIOS)
        for name in ('success50', 'max_class_confidence', 'best_predicted_iou', 'best_anchor_iou',
                     'crop_raw_max_confidence'):
            assert abs(num(row, name + '_upper') - num(row, name + '_lower') -
                       num(row, 'delta_' + name)) < 1e-12
    timings = read_csv(out / 'latency_summary.csv')
    for row in timings:
        values = json.loads(row['latency_samples_ms'])
        assert len(values) == metadata['repeats'] and min(values) > 0
        assert int(row['slot_checked_forwards']) == metadata['repeats'] + metadata['warmup']
        assert abs(float(np.median(values)) - num(row, 'median_latency_ms')) < 1e-12
    assert metadata['stage2d_controls_checked'] == 72
    return per_shape, comparisons, metadata


def plots(per_shape, comparisons, summary, out):
    for metric, filename, label in (
        ('shape_mean_map50', 'map_by_policy.png', 'Mean per-shape mAP@0.5'),
        ('shape_mean_cpu_median_latency_ms', 'latency_by_policy.png', 'Mean CPU median latency (ms)'),
    ):
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar([row['policy'] for row in summary], [row[metric] for row in summary])
        ax.set_ylabel(label)
        ax.tick_params(axis='x', rotation=15)
        fig.tight_layout()
        fig.savefig(out / filename, dpi=140)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    bottom = np.zeros(len(POLICIES))
    for depth in range(1, 7):
        counts = [sum(row['policy'] == policy and int(row['selected_L_R']) == depth
                      for row in per_shape) for policy in POLICIES]
        ax.bar(POLICIES, counts, bottom=bottom, label=f'L{depth}')
        bottom += counts
    ax.legend(ncol=6)
    ax.set_ylabel('Number of shapes')
    ax.tick_params(axis='x', rotation=15)
    fig.tight_layout()
    fig.savefig(out / 'selected_depth_distribution.png', dpi=140)
    plt.close(fig)
    coarse = [row for row in comparisons if row['pair_kind'] == 'stage2d_adjacent'
              and row['upper_adds_1x1'] == 'True']
    for ratio, filename in (('gt_area_ratio', 'gain_vs_gt_area_ratio.png'),
                             ('gt_max_side_ratio', 'gain_vs_gt_max_side_ratio.png')):
        fig, ax = plt.subplots(figsize=(7, 4))
        data = ax.hexbin([num(row, ratio) for row in coarse],
                         [num(row, 'delta_best_predicted_iou') for row in coarse],
                         gridsize=35, mincnt=1, bins='log')
        ax.set_xlabel(ratio)
        ax.set_ylabel('Delta best same-class predicted IoU (upper-lower)')
        ax.set_title('Adjacent pairs adding 1x1; repeated GT exposures, not causal')
        fig.colorbar(data, ax=ax, label='log10 count')
        fig.tight_layout()
        fig.savefig(out / filename, dpi=140)
        plt.close(fig)


def run():
    per_shape, comparisons, metadata = validate()
    summary = summaries(per_shape)
    bins = ratio_bins(comparisons)
    corr = correlations(comparisons)
    csv_rows(OUT / 'policy_summary.csv', summary)
    csv_rows(OUT / 'object_ratio_bins.csv', bins)
    csv_rows(OUT / 'object_ratio_correlations.csv', corr)
    plots(per_shape, comparisons, summary, OUT)
    analysis = {
        'status': 'COMPLETE', 'input_metadata_sha256': sha(OUT / 'metadata.json'),
        'analysis_source_sha256': sha(Path(__file__)),
        'N_shape_GT_exposures': metadata['num_geometry_objects'],
        'N_comparison_GT_exposures': metadata['num_object_pair_comparisons'],
        'warnings': ['Repeated source objects across shapes; not independent observations',
                     'Bin recall is pooled micro GT success, not class-macro evaluator recall',
                     'No per-sample AP decomposition; one-to-one localization success used',
                     'Undefined correlations remain blank with explicit status',
                     'No causal claims, confidence intervals or independence-based p-values',
                     'Primary GT-only and within-shape-centered analyses included',
                     'Policy summaries are equal-shape means, not pooled AP or deployment weights'],
        'output_sha256': {name: sha(OUT / name) for name in (
            'policy_summary.csv', 'object_ratio_bins.csv', 'object_ratio_correlations.csv',
            'map_by_policy.png', 'latency_by_policy.png', 'selected_depth_distribution.png',
            'gain_vs_gt_area_ratio.png', 'gain_vs_gt_max_side_ratio.png')},
    }
    write_json(OUT / 'analysis_metadata.json', analysis)
    print('VERIFIED and aggregated', len(per_shape), 'policy rows;', len(comparisons), 'GT-pair records')
    for row in summary:
        print(row)


if __name__ == '__main__':
    run()
