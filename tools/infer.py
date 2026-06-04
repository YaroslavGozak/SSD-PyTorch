import torch
import argparse
import os
import yaml
import random
import csv
from tqdm import tqdm
import numpy as np
import cv2
from model.model_adapters import DetectionLabelRemapAdapter, YoloV8Adapter
from tools.helpers.pipeline import (
    load_model_and_dataset as pipeline_load_model_and_dataset,
    resolve_device,
)

device = resolve_device(None)
print('Using device {}'.format(device))


def get_iou(det, gt):
    det_x1, det_y1, det_x2, det_y2 = det
    gt_x1, gt_y1, gt_x2, gt_y2 = gt

    x_left = max(det_x1, gt_x1)
    y_top = max(det_y1, gt_y1)
    x_right = min(det_x2, gt_x2)
    y_bottom = min(det_y2, gt_y2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    area_intersection = (x_right - x_left) * (y_bottom - y_top)
    det_area = (det_x2 - det_x1) * (det_y2 - det_y1)
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    area_union = float(det_area + gt_area - area_intersection + 1E-6)
    iou = area_intersection / area_union
    return iou


def compute_map(det_boxes, gt_boxes, iou_threshold=0.5, method='area', difficult=None):
    # det_boxes = [
    #   {
    #       'person' : [[x1, y1, x2, y2, score], ...],
    #       'car' : [[x1, y1, x2, y2, score], ...]
    #   }
    #   {det_boxes_img_2},
    #   ...
    #   {det_boxes_img_N},
    # ]
    #
    # gt_boxes = [
    #   {
    #       'person' : [[x1, y1, x2, y2], ...],
    #       'car' : [[x1, y1, x2, y2], ...]
    #   },
    #   {gt_boxes_img_2},
    #   ...
    #   {gt_boxes_img_N},
    # ]

    gt_labels = {cls_key for im_gt in gt_boxes for cls_key in im_gt.keys()}
    gt_labels = sorted(gt_labels)

    all_aps = {}
    all_recalls = {}
    # average precisions for ALL classes
    aps = []
    recall_values = []
    for idx, label in enumerate(gt_labels):
        # Get detection predictions of this class
        cls_dets = [
            [im_idx, im_dets_label] for im_idx, im_dets in enumerate(det_boxes)
            if label in im_dets for im_dets_label in im_dets[label]
        ]

        # cls_dets = [
        #   (0, [x1_0, y1_0, x2_0, y2_0, score_0]),
        #   ...
        #   (0, [x1_M, y1_M, x2_M, y2_M, score_M]),
        #   (1, [x1_0, y1_0, x2_0, y2_0, score_0]),
        #   ...
        #   (1, [x1_N, y1_N, x2_N, y2_N, score_N]),
        #   ...
        # ]

        # Sort them by confidence score
        cls_dets = sorted(cls_dets, key=lambda k: -k[1][-1])

        # For tracking which gt boxes of this class have already been matched
        gt_matched = [[False for _ in im_gts.get(label, [])] for im_gts in gt_boxes]
        # Number of gt boxes for this class for recall calculation
        num_gts = sum([len(im_gts.get(label, [])) for im_gts in gt_boxes])
        num_difficults = sum([sum(difficults_label.get(label, [])) for difficults_label in difficult])

        tp = [0] * len(cls_dets)
        fp = [0] * len(cls_dets)

        # For each prediction
        for det_idx, (im_idx, det_pred) in enumerate(cls_dets):
            # Get gt boxes for this image and this label
            im_gts = gt_boxes[im_idx].get(label, [])
            im_gt_difficults = difficult[im_idx].get(label, [])

            max_iou_found = -1
            max_iou_gt_idx = -1

            # Get best matching gt box
            for gt_box_idx, gt_box in enumerate(im_gts):
                gt_box_iou = get_iou(det_pred[:-1], gt_box)
                if gt_box_iou > max_iou_found:
                    max_iou_found = gt_box_iou
                    max_iou_gt_idx = gt_box_idx
            # TP only if iou >= threshold and this gt has not yet been matched
            if max_iou_found >= iou_threshold:
                if not im_gt_difficults[max_iou_gt_idx]:
                    if not gt_matched[im_idx][max_iou_gt_idx]:
                        # If tp then we set this gt box as matched
                        gt_matched[im_idx][max_iou_gt_idx] = True
                        tp[det_idx] = 1
                    else:
                        fp[det_idx] = 1
            else:
                fp[det_idx] = 1

        # Cumulative tp and fp
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)

        eps = np.finfo(np.float32).eps
        # recalls = tp / np.maximum(num_gts, eps)
        recalls = tp / np.maximum(num_gts - num_difficults, eps)
        precisions = tp / np.maximum((tp + fp), eps)

        # Operating-point recall: fraction of GT matched at this confidence threshold
        class_recall = float(recalls[-1]) if len(recalls) > 0 else 0.0

        if method == 'area':
            recalls = np.concatenate(([0.0], recalls, [1.0]))
            precisions = np.concatenate(([0.0], precisions, [0.0]))

            # Replace precision values with recall r with maximum precision value
            # of any recall value >= r
            # This computes the precision envelope
            for i in range(precisions.size - 1, 0, -1):
                precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
            # For computing area, get points where recall changes value
            i = np.where(recalls[1:] != recalls[:-1])[0]
            # Add the rectangular areas to get ap
            ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])
        elif method == 'interp':
            ap = 0.0
            for interp_pt in np.arange(0, 1 + 1E-3, 0.1):
                # Get precision values for recall values >= interp_pt
                prec_interp_pt = precisions[recalls >= interp_pt]

                # Get max of those precision values
                prec_interp_pt= prec_interp_pt.max() if prec_interp_pt.size>0.0 else 0.0
                ap += prec_interp_pt
            ap = ap / 11.0
        else:
            raise ValueError('Method can only be area or interp')
        if num_gts > 0:
            aps.append(ap)
            all_aps[label] = ap
            all_recalls[label] = class_recall
            recall_values.append(class_recall)
        else:
            all_aps[label] = np.nan
            all_recalls[label] = float('nan')
    # compute mAP at provided iou threshold
    mean_ap = sum(aps) / len(aps)
    mean_recall = float(np.mean(recall_values)) if recall_values else float('nan')
    return mean_ap, all_aps, mean_recall, all_recalls


def load_model_and_dataset(args, transform_name=None):
    model, dataset, test_dataset_loader, config = pipeline_load_model_and_dataset(
        device=device,
        args=args,
        transform_name=transform_name,
    )
    if hasattr(model, 'low_score_threshold'):
        model.low_score_threshold = float(config['train_params'].get('infer_conf_threshold', 0.05))
    return model, dataset, test_dataset_loader, config


def run_detector(model, im_tensor, target, model_name='ssd', conf_threshold=None):
    if model_name == 'yolo':
        if isinstance(model, (YoloV8Adapter, DetectionLabelRemapAdapter)):
            _, detections = model(im_tensor)
            return detections
        predict_kwargs = {'source': im_tensor, 'verbose': False}
        if conf_threshold is not None:
            predict_kwargs['conf'] = float(conf_threshold)
        results = model.predict(**predict_kwargs)
        detections = []
        _, _, h, w = im_tensor.shape
        for res in results:
            if res.boxes is None or len(res.boxes) == 0:
                detections.append({
                    'boxes': torch.empty((0, 4), dtype=torch.float32, device=im_tensor.device),
                    'labels': torch.empty((0,), dtype=torch.int64, device=im_tensor.device),
                    'scores': torch.empty((0,), dtype=torch.float32, device=im_tensor.device),
                })
                continue

            boxes_xyxy = res.boxes.xyxy.to(im_tensor.device).float()
            boxes_norm = boxes_xyxy.clone()
            boxes_norm[:, [0, 2]] /= float(w)
            boxes_norm[:, [1, 3]] /= float(h)
            detections.append({
                'boxes': boxes_norm,
                'labels': (res.boxes.cls.to(im_tensor.device).long() + 1),
                'scores': res.boxes.conf.to(im_tensor.device).float(),
            })
        return detections

    try:
        _, detections = model(im_tensor, [target])
    except Exception:
        try:
            _, detections = model(im_tensor, None)
        except Exception:
            _, detections = model(im_tensor)
    return detections


def append_model_results_csv(model_task_path, config, evaluated_dataset, mean_ap, mean_recall):
    """Append one validation summary row to model-level results.csv."""
    if not os.path.exists(model_task_path):
        os.makedirs(model_task_path, exist_ok=True)

    results_csv_path = os.path.join(model_task_path, 'results.csv')
    file_exists = os.path.exists(results_csv_path)

    train_cfg = config.get('train_params', {})
    row = {
        'task_name': str(train_cfg.get('task_name', '')),
        'model': str(train_cfg.get('model', '')),
        'dataset': str(train_cfg.get('dataset', '')),
        'ckpt_name': str(train_cfg.get('ckpt_name', '')),
        'evaluated_dataset': str(evaluated_dataset),
        'mAP': float(mean_ap),
        'mean_detector_recall': float(mean_recall),
    }
    fieldnames = [
        'task_name',
        'model',
        'dataset',
        'ckpt_name',
        'evaluated_dataset',
        'mAP',
        'mean_detector_recall',
    ]

    with open(results_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def infer(args):
    samples_path = args.results_path + '/samples' if args.results_path else 'samples'
    if not os.path.exists(samples_path):
        os.makedirs(samples_path, exist_ok=True)

    with open(args.config_path, 'r') as file:
        cfg_preview = yaml.safe_load(file)
    model_name = str(cfg_preview['train_params']['model'])
    transform_name = cfg_preview['dataset_params'].get('transform_name', 'ssd')
    if model_name == 'yolo':
        transform_name = 'fixed_padding_roi_crop_yolo_0'

    model, dataset_dataset, test_dataset_loader, config = load_model_and_dataset(args, transform_name)
    conf_threshold = config['train_params'].get('infer_conf_threshold', None)
    if hasattr(model, 'low_score_threshold'):
        model.low_score_threshold = conf_threshold

    num_samples = 5
    for i in tqdm(range(num_samples)):
        dataset_idx = random.randint(0, len(dataset_dataset))
        im_tensor, target, fname = dataset_dataset[dataset_idx]
        ssd_detections = run_detector(
            model,
            im_tensor.unsqueeze(0).float().to(device),
            target,
            model_name=model_name,
            conf_threshold=conf_threshold,
        )

        gt_im = cv2.imread(fname)
        h, w = gt_im.shape[:2]
        gt_im_copy = gt_im.copy()
        # Saving images with ground truth boxes
        for idx, box in enumerate(target['bboxes']):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            x1, y1, x2, y2 = int(w*x1), int(h*y1), int(w*x2), int(h*y2)
            cv2.rectangle(gt_im, (x1, y1), (x2, y2), thickness=2, color=[0, 255, 0])
            cv2.rectangle(gt_im_copy, (x1, y1), (x2, y2), thickness=2, color=[0, 255, 0])
            text = dataset_dataset.idx2label[target['labels'][idx].detach().cpu().item()]
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)
            text_w, text_h = text_size
            cv2.rectangle(gt_im_copy, (x1, y1), (x1 + 10 + text_w, y1 + 10 + text_h), [255, 255, 255], -1)
            cv2.putText(gt_im, text=dataset_dataset.idx2label[target['labels'][idx].detach().cpu().item()],
                        org=(x1 + 5, y1 + 15),
                        thickness=1,
                        fontScale=1,
                        color=[0, 0, 0],
                        fontFace=cv2.FONT_HERSHEY_PLAIN)
            cv2.putText(gt_im_copy, text=text,
                        org=(x1 + 5, y1 + 15),
                        thickness=1,
                        fontScale=1,
                        color=[0, 0, 0],
                        fontFace=cv2.FONT_HERSHEY_PLAIN)
        cv2.addWeighted(gt_im_copy, 0.7, gt_im, 0.3, 0, gt_im)
        cv2.imwrite(os.path.join(samples_path, 'output_ssd_gt_{}.png'.format(i)), gt_im)

        # Getting predictions from trained model
        boxes = ssd_detections[0]['boxes']
        labels = ssd_detections[0]['labels']
        scores = ssd_detections[0]['scores']
        im = cv2.imread(fname)
        im_copy = im.copy()

        # Saving images with predicted boxes
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            x1, y1, x2, y2 = int(w * x1), int(h * y1), int(w * x2), int(h * y2)
            cv2.rectangle(im, (x1, y1), (x2, y2), thickness=2, color=[0, 0, 255])
            cv2.rectangle(im_copy, (x1, y1), (x2, y2), thickness=2, color=[0, 0, 255])
            text = '{} : {:.2f}'.format(dataset_dataset.idx2label[labels[idx].detach().cpu().item()],
                                        scores[idx].detach().cpu().item())
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)
            text_w, text_h = text_size
            cv2.rectangle(im_copy, (x1, y1), (x1 + 10 + text_w, y1 + 10 + text_h), [255, 255, 255], -1)
            cv2.putText(im, text=text,
                        org=(x1 + 5, y1 + 15),
                        thickness=1,
                        fontScale=1,
                        color=[0, 0, 0],
                        fontFace=cv2.FONT_HERSHEY_PLAIN)
            cv2.putText(im_copy, text=text,
                        org=(x1 + 5, y1 + 15),
                        thickness=1,
                        fontScale=1,
                        color=[0, 0, 0],
                        fontFace=cv2.FONT_HERSHEY_PLAIN)
        cv2.addWeighted(im_copy, 0.7, im, 0.3, 0, im)
        cv2.imwrite(os.path.join(samples_path, 'output_ssd_{}.jpg'.format(i)), im)

    print('Done Detecting...')


def evaluate_map(args):
    with open(args.config_path, 'r') as file:
        cfg_preview = yaml.safe_load(file)
    model_name = str(cfg_preview['train_params']['model'])
    is_yolo = model_name == 'yolo'

    for pad in range(0, 201, 10):
        print('Evaluating mAP with padding {}...'.format(pad))
        if is_yolo:
            transform_name = 'fixed_padding_roi_crop_yolo_{}'.format(pad)
        else:
            transform_name = 'fixed_padding_roi_crop_{}'.format(pad)

        model, voc, test_dataset, config = load_model_and_dataset(args, transform_name=transform_name)
        model_task_path = os.path.join('trained_models', config['train_params']['task_name'])
        args.results_path = os.path.join(model_task_path, transform_name + '_results')

        if is_yolo:
            debug_root = os.path.join(args.results_path, 'samples', 'yolo_transform_debug')
            pad_debug_dir = os.path.join(debug_root, 'pad_{}'.format(pad))
            os.makedirs(pad_debug_dir, exist_ok=True)
            os.environ['YOLO_ROI_DEBUG_DIR'] = pad_debug_dir
            os.environ.setdefault('YOLO_ROI_DEBUG_MAX', '3')

        print('Results will be saved to {}'.format(args.results_path))

        gts = []
        preds = []
        difficults = []
        for im_tensor, target, fname in tqdm(test_dataset):
            im_tensor = im_tensor.float().to(device)
            target_bboxes = target['bboxes'].float()[0].to(device)
            target_labels = target['labels'].long()[0].to(device)
            difficult = target['difficult'].long()[0].to(device)
            conf_threshold = config['train_params'].get('infer_conf_threshold', None)
            ssd_detections = run_detector(
                model,
                im_tensor,
                target,
                model_name=model_name,
                conf_threshold=conf_threshold,
            )

            boxes = ssd_detections[0]['boxes']
            labels = ssd_detections[0]['labels']
            scores = ssd_detections[0]['scores']

            pred_boxes = {}
            gt_boxes = {}
            difficult_boxes = {}

            for label_name in voc.label2idx:
                pred_boxes[label_name] = []
                gt_boxes[label_name] = []
                difficult_boxes[label_name] = []

            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = box.detach().cpu().numpy()
                label = labels[idx].detach().cpu().item()
                score = scores[idx].detach().cpu().item()
                label_name = voc.idx2label[label]
                pred_boxes[label_name].append([x1, y1, x2, y2, score])
            for idx, box in enumerate(target_bboxes):
                x1, y1, x2, y2 = box.detach().cpu().numpy()
                label = target_labels[idx].detach().cpu().item()
                label_name = voc.idx2label[label]
                gt_boxes[label_name].append([x1, y1, x2, y2])
                difficult_boxes[label_name].append(difficult[idx].detach().cpu().item())

            gts.append(gt_boxes)
            preds.append(pred_boxes)
            difficults.append(difficult_boxes)
        mean_ap, all_aps, mean_recall, all_recalls = compute_map(preds, gts, method='area', difficult=difficults)
        print('Class Wise Average Precisions and Detector Recall')
        for idx in range(len(voc.idx2label)):
            lbl = voc.idx2label[idx]
            print('AP for class {} = {:.4f}  |  detector_recall = {:.4f}'.format(
                lbl, all_aps[lbl], all_recalls[lbl]))
        print('Mean Average Precision : {:.4f}'.format(mean_ap))
        print('Mean Detector Recall   : {:.4f}'.format(mean_recall))

        model_task_path = os.path.join('trained_models', config['train_params']['task_name'])
        # Write results to map.txt
        
        if args.results_path:
            map_file_path = os.path.join(args.results_path, 'mAp.txt')
            if not os.path.exists(args.results_path):
                os.makedirs(args.results_path, exist_ok=True)
            with open(map_file_path, 'w') as f:
                f.write('Class Wise Average Precisions and Detector Recall\n')
                f.write('=' * 50 + '\n')
                for idx in range(len(voc.idx2label)):
                    lbl = voc.idx2label[idx]
                    f.write('AP for class {} = {:.4f}  |  detector_recall = {:.4f}\n'.format(
                        lbl, all_aps[lbl], all_recalls[lbl]))
                f.write('=' * 50 + '\n')
                f.write('Mean Average Precision : {:.4f}\n'.format(mean_ap))
                f.write('Mean Detector Recall   : {:.4f}\n'.format(mean_recall))
            
            print(f'Results saved to {map_file_path}')
            
            # Save results to CSV file
            csv_file_path = os.path.join(args.results_path, 'mAp.csv')
            with open(csv_file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Header row: class names + mAP + detector_recall columns
                header = ([voc.idx2label[idx] for idx in range(len(voc.idx2label))] + ['mAP']
                        + ['detector_recall_' + voc.idx2label[idx] for idx in range(len(voc.idx2label))]
                        + ['mean_detector_recall'])
                writer.writerow(header)
                
                # Data row: AP values + mean AP + recall values + mean recall
                data = ([all_aps[voc.idx2label[idx]] for idx in range(len(voc.idx2label))] + [mean_ap]
                        + [all_recalls[voc.idx2label[idx]] for idx in range(len(voc.idx2label))]
                        + [mean_recall])
                writer.writerow(data)
            
            print(f'Results saved to {csv_file_path}')
        else:
            print('No results path provided, skipping saving mAP results to file.')

        append_model_results_csv(
            model_task_path=model_task_path,
            config=config,
            evaluated_dataset=transform_name,
            mean_ap=mean_ap,
            mean_recall=mean_recall,
        )
        print('Appended summary to {}'.format(os.path.join(model_task_path, 'results.csv')))

        if is_yolo:
            os.environ.pop('YOLO_ROI_DEBUG_DIR', None)

def infer_and_evaluate(args):
    with torch.no_grad():
        if args.infer_samples:
            infer(args)
        else:
            print('Not Inferring for samples as `infer_samples` argument is False')

        if args.evaluate:
            evaluate_map(args)
        else:
            print('Not Evaluating as `evaluate` argument is False')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ssd inference')
    parser.add_argument('--config', dest='config_path',
                        default='config/voc.yaml', type=str)
    parser.add_argument('--evaluate', dest='evaluate',
                        default=True, type=bool)
    parser.add_argument('--infer-samples', dest='infer_samples',
                        default=True, type=bool)
    parser.add_argument('--results-path', dest='results_path',
                        default=None, type=str)
    args = parser.parse_args()

    infer_and_evaluate(args)
