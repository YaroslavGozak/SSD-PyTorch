"""Shared epoch-metrics CSV writer for standard and mixed-stage training."""
import csv
import os
import tempfile


def append_epoch_metrics_csv(
    csv_file_path,
    *,
    epoch,
    stage,
    classification_loss,
    detection_loss,
    learning_rate,
    mean_ap,
    mean_detector_recall,
):
    fieldnames = [
        'epoch', 'stage', 'classification_loss', 'detection_loss',
        'learning_rate', 'mAP', 'mean_detector_recall',
    ]
    file_exists = os.path.exists(csv_file_path) and os.path.getsize(csv_file_path) > 0
    if file_exists:
        with open(csv_file_path, newline='', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            existing_fields = reader.fieldnames
            if 'stage' not in existing_fields:
                # Historical logs can contain several stages. Keep their stage
                # unknown instead of assigning the current training stage.
                rows = list(reader)
            else:
                rows = None
        if rows is not None:
            fieldnames = list(existing_fields)
            fieldnames.insert(fieldnames.index('epoch') + 1, 'stage')
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode='w', newline='', encoding='utf-8', delete=False,
                    dir=os.path.dirname(os.path.abspath(csv_file_path)),
                ) as csvfile:
                    temp_path = csvfile.name
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                os.replace(temp_path, csv_file_path)
            finally:
                if temp_path is not None and os.path.exists(temp_path):
                    os.remove(temp_path)
        else:
            fieldnames = existing_fields

    with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'epoch': int(epoch), 'stage': int(stage),
            'classification_loss': float(classification_loss),
            'detection_loss': float(detection_loss),
            'learning_rate': float(learning_rate), 'mAP': float(mean_ap),
            'mean_detector_recall': float(mean_detector_recall),
        })
