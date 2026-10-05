"""Epoch-level accounting for the actual tensor sizes used by mixed sampling."""

import csv
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path


FIELDS = (
    'epoch', 'stage', 'mode', 'required_size_px', 'configured_probability_pct',
    'batches', 'batch_share_pct', 'samples', 'sample_share_pct',
    'exact_size_batches', 'exact_size_rate_pct',
    'enlarged_batches', 'enlargement_rate_pct', 'share_of_all_enlargements_pct',
    'enlargement_rank',
    'object_mismatch_batches', 'object_mismatch_rate_pct',
    'share_of_all_object_mismatches_pct', 'object_mismatch_rank',
    'padding_or_jitter_only_batches',
    'enlarged_to_full_frame_batches', 'mean_actual_size_px', 'p95_actual_size_px',
    'max_actual_size_px', 'mean_extra_px_when_enlarged', 'max_extra_px',
    'total_extra_px', 'actual_size_counts',
)


class MixedSamplingStats:
    def __init__(self, mode_configs, image_size):
        self.mode_configs = {mode.name: mode for mode in mode_configs}
        self.image_size = int(image_size)
        self.actual_sizes = defaultdict(list)
        self.sample_counts = Counter()
        self.object_mismatch_counts = Counter()

    def record(self, mode, actual_size, samples, object_mismatch=False):
        if mode not in self.mode_configs:
            raise ValueError(f'Unknown mixed-sampling mode: {mode}')
        required = self.mode_configs[mode].out_size
        actual_size = int(actual_size)
        if not required <= actual_size <= self.image_size:
            raise ValueError(f'Actual size {actual_size} is outside [{required}, {self.image_size}] for {mode}')
        if samples <= 0:
            raise ValueError('A recorded batch must contain at least one sample')
        if object_mismatch and actual_size == required:
            raise ValueError('An object mismatch must enlarge the batch')
        self.actual_sizes[mode].append(actual_size)
        self.sample_counts[mode] += int(samples)
        self.object_mismatch_counts[mode] += int(bool(object_mismatch))

    def rows(self, epoch, stage):
        total_batches = sum(len(sizes) for sizes in self.actual_sizes.values())
        total_samples = sum(self.sample_counts.values())
        total_enlarged = sum(
            sum(size > self.mode_configs[mode].out_size for size in sizes)
            for mode, sizes in self.actual_sizes.items()
        )
        total_object_mismatch = sum(self.object_mismatch_counts.values())
        probability_sum = sum(mode.prob for mode in self.mode_configs.values())

        def make_row(mode_name, required, probability, sizes, samples):
            batches = len(sizes)
            extras = [size - required for size in sizes] if required is not None else []
            enlarged = [extra for extra in extras if extra > 0]
            mismatches = self.object_mismatch_counts[mode_name]
            return {
                'epoch': int(epoch), 'stage': int(stage), 'mode': mode_name,
                'required_size_px': required if required is not None else '',
                'configured_probability_pct': round(100 * probability / probability_sum, 4) if probability_sum else 0,
                'batches': batches,
                'batch_share_pct': round(100 * batches / total_batches, 4) if total_batches else 0,
                'samples': samples,
                'sample_share_pct': round(100 * samples / total_samples, 4) if total_samples else 0,
                'exact_size_batches': batches - len(enlarged),
                'exact_size_rate_pct': round(100 * (batches - len(enlarged)) / batches, 4) if batches else 0,
                'enlarged_batches': len(enlarged),
                'enlargement_rate_pct': round(100 * len(enlarged) / batches, 4) if batches else 0,
                'share_of_all_enlargements_pct': round(100 * len(enlarged) / total_enlarged, 4) if total_enlarged else 0,
                'enlargement_rank': 0,
                'object_mismatch_batches': mismatches,
                'object_mismatch_rate_pct': round(100 * mismatches / batches, 4) if batches else 0,
                'share_of_all_object_mismatches_pct': round(100 * mismatches / total_object_mismatch, 4) if total_object_mismatch else 0,
                'object_mismatch_rank': 0,
                'padding_or_jitter_only_batches': len(enlarged) - mismatches,
                'enlarged_to_full_frame_batches': sum(size == self.image_size and extra > 0 for size, extra in zip(sizes, extras)),
                'mean_actual_size_px': round(sum(sizes) / batches, 4) if batches else '',
                'p95_actual_size_px': sorted(sizes)[math.ceil(.95 * batches) - 1] if batches else '',
                'max_actual_size_px': max(sizes) if batches else '',
                'mean_extra_px_when_enlarged': round(sum(enlarged) / len(enlarged), 4) if enlarged else 0,
                'max_extra_px': max(extras, default=0),
                'total_extra_px': sum(extras),
                'actual_size_counts': json.dumps(dict(sorted(Counter(sizes).items())), separators=(',', ':')),
            }

        rows = []
        for mode in self.mode_configs.values():
            rows.append(make_row(mode.name, mode.out_size, mode.prob,
                                 self.actual_sizes[mode.name], self.sample_counts[mode.name]))
        positive_counts = sorted({row['enlarged_batches'] for row in rows if row['enlarged_batches']}, reverse=True)
        mismatch_counts = sorted({row['object_mismatch_batches'] for row in rows if row['object_mismatch_batches']}, reverse=True)
        for row in rows:
            if row['enlarged_batches']:
                row['enlargement_rank'] = positive_counts.index(row['enlarged_batches']) + 1
            if row['object_mismatch_batches']:
                row['object_mismatch_rank'] = mismatch_counts.index(row['object_mismatch_batches']) + 1

        all_sizes = [size for sizes in self.actual_sizes.values() for size in sizes]
        overall = make_row('ALL', None, probability_sum, all_sizes, total_samples)
        overall['enlarged_batches'] = total_enlarged
        overall['exact_size_batches'] = total_batches - total_enlarged
        overall['exact_size_rate_pct'] = round(100 * (total_batches - total_enlarged) / total_batches, 4) if total_batches else 0
        overall['enlargement_rate_pct'] = round(100 * total_enlarged / total_batches, 4) if total_batches else 0
        overall['share_of_all_enlargements_pct'] = 100 if total_enlarged else 0
        overall['object_mismatch_batches'] = total_object_mismatch
        overall['object_mismatch_rate_pct'] = round(100 * total_object_mismatch / total_batches, 4) if total_batches else 0
        overall['share_of_all_object_mismatches_pct'] = 100 if total_object_mismatch else 0
        overall['padding_or_jitter_only_batches'] = total_enlarged - total_object_mismatch
        overall['enlarged_to_full_frame_batches'] = sum(row['enlarged_to_full_frame_batches'] for row in rows)
        overall['mean_extra_px_when_enlarged'] = round(
            sum(row['total_extra_px'] for row in rows) / total_enlarged, 4,
        ) if total_enlarged else 0
        overall['max_extra_px'] = max((row['max_extra_px'] for row in rows), default=0)
        overall['total_extra_px'] = sum(row['total_extra_px'] for row in rows)
        return [overall, *rows]


def save_epoch_stats(path, rows):
    """Replace this epoch's rows on resume, preserving other completed epochs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError('No mixed-sampling statistics to save')
    existing = []
    if path.exists():
        with path.open(newline='', encoding='utf-8') as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != FIELDS:
                raise ValueError(f'Unexpected mixed-sampling stats columns: {path}')
            existing = [row for row in reader if (row['epoch'], row['stage']) !=
                        (str(rows[0]['epoch']), str(rows[0]['stage']))]
    fd, temporary = tempfile.mkstemp(prefix='mixed_sampling_stats_', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(existing)
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
