import argparse
import os
import time
from typing import Any, List, Optional, Tuple

import cv2
import torch

from tools.helpers.config_reader import load_config
from tools.helpers.pipeline import (
    MERGE_STRATEGIES,
    FrameResult,
    build_tracker,
    ensure_im_size_tuple,
    extract_gt_boxes,
    load_model_and_dataset,
    process_frame,
    extract_gt_for_tracker
)


def _extract_sequence_meta(target: dict, fname: Any) -> Tuple[str, bool, Optional[int]]:
    """Return (video_id, is_first_frame, frame_idx_in_video) for sequence resets."""
    default_path = fname[0] if isinstance(fname, (list, tuple)) else fname
    default_video_id = os.path.basename(os.path.dirname(str(default_path)))

    video_id = target.get("video_id", default_video_id)
    if isinstance(video_id, list):
        video_id = video_id[0] if video_id else default_video_id

    is_first = target.get("is_first_frame", False)
    if isinstance(is_first, list):
        is_first = is_first[0] if is_first else False
    if isinstance(is_first, torch.Tensor):
        is_first = bool(is_first.item())

    frame_idx = target.get("frame_idx", None)
    if isinstance(frame_idx, list):
        frame_idx = frame_idx[0] if frame_idx else None
    if isinstance(frame_idx, torch.Tensor):
        frame_idx = int(frame_idx.item())
    if frame_idx is not None:
        frame_idx = int(frame_idx)

    return str(video_id), bool(is_first), frame_idx


def infer_sequentially_with_roi(args):
    benchmark_cfg = load_config(args.benchmark_config)

    benchmark_params = benchmark_cfg['benchmark_vid_params']
    train_args = argparse.Namespace(config_path=benchmark_cfg['train_config_path'])
    model, dataset, test_dataset_loader, config = load_model_and_dataset(benchmark_params["device"], train_args)
    conf_threshold = config['train_params']['infer_conf_threshold']
    model.low_score_threshold = conf_threshold

    tracker = build_tracker(benchmark_params['tracker'])

    inference_cfg = benchmark_params['inference']
    tracker_input_dropout_cfg = benchmark_params.get('tracker_input_dropout', None)
    roi_merge_cfg = benchmark_params['roi_merge']
    merge_fn = MERGE_STRATEGIES[roi_merge_cfg['strategy']]
    merge_tau = float(roi_merge_cfg.get('tau', 150000.0))
    im_size_hw = ensure_im_size_tuple(config['dataset_params']['im_size'])

    window_name = 'Kalman ROI Streaming - {} dataset'.format(config['train_params']['dataset'])
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    paused = False
    step_once = False
    frame_delay = args.frame_delay if args.frame_delay is not None else int(args.default_frame_delay)
    key_frame_interval = max(1, int(inference_cfg['key_frame_interval']))
    current_frame_idx = 0
    total_frames = len(test_dataset_loader)
    data_iter = iter(test_dataset_loader)
    current_video_id: Optional[str] = None

    # ROIs predicted by tracker from previous frame.
    next_frame_rois: List[List[int]] = []

    print(f'Total frames: {total_frames}')
    print('Controls: SPACE=pause/resume, f=next(paused), ESC=quit, +/-=adjust delay')
    print('Starting Kalman ROI streaming inference...\n')

    with torch.no_grad():
        while True:
            loop_start = time.perf_counter()

            if (not paused) or step_once:
                try:
                    im_tensor, target, fname = next(data_iter)
                except StopIteration:
                    print(f'Reached end of sequence ({total_frames} frames)')
                    break

                current_frame_idx += 1
                fpath = fname[0] if isinstance(fname, (list, tuple)) else fname
                fpath = os.path.abspath(fpath)

                frame_bgr = cv2.imread(fpath)
                if frame_bgr is None:
                    print(f'Failed to read frame: {fpath}')
                    continue
                frame_h, frame_w = frame_bgr.shape[:2]

                gt_target = target[0] if isinstance(target, list) else target
                video_id, is_first_frame, frame_idx_in_video = _extract_sequence_meta(gt_target, fname)
                if current_video_id is None:
                    current_video_id = video_id

                if is_first_frame or video_id != current_video_id:
                    tracker.reset()
                    next_frame_rois = []
                    current_video_id = video_id

                # Oracle detector mode: use GT detections for ROI generation instead of tracker output.
                tracker_type = str(benchmark_cfg["benchmark_vid_params"]["tracker"]["type"])
                if tracker_type == "oracle_gt":
                    oracle_dets = extract_gt_for_tracker(gt_target, dataset.idx2label, frame_w, frame_h)
                    if hasattr(tracker, "set_oracle_detections"):
                        tracker.set_oracle_detections(oracle_dets)
                    if hasattr(tracker, "preview_rois"):
                        next_frame_rois = tracker.preview_rois((frame_h, frame_w))

                model_device = next(model.parameters()).device
                effective_frame_idx = frame_idx_in_video if frame_idx_in_video is not None else current_frame_idx
                result: FrameResult = process_frame(
                    model=model,
                    idx2label=dataset.idx2label,
                    frame_bgr=frame_bgr,
                    im_tensor=im_tensor,
                    tracker=tracker,
                    next_frame_rois=next_frame_rois,
                    frame_idx=effective_frame_idx,
                    key_frame_interval=key_frame_interval,
                    im_size_hw=im_size_hw,
                    conf_threshold=conf_threshold,
                    nms_iou=float(inference_cfg['nms_iou']),
                    merge_fn=merge_fn,
                    merge_tau=merge_tau,
                    model_device=model_device,
                    tracker_input_dropout_cfg=tracker_input_dropout_cfg,
                )
                next_frame_rois = result.next_frame_rois

                display_frame = frame_bgr.copy()

                # Draw detections in red.
                for det in result.final_detections:
                    x1, y1, x2, y2 = det['bbox']
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    label_text = '{}:{:.2f}'.format(det['class'], det['confidence'])
                    cv2.putText(
                        display_frame,
                        label_text,
                        (x1 + 4, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_PLAIN,
                        1.0,
                        (255, 255, 255),
                        1,
                    )

                # Draw detections omitted from tracker input in orange.
                for det in result.dropped_tracker_detections:
                    x1, y1, x2, y2 = det['bbox']
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    cv2.putText(
                        display_frame,
                        'SKIP',
                        (x1 + 4, y2 + 14),
                        cv2.FONT_HERSHEY_PLAIN,
                        1.0,
                        (0, 165, 255),
                        1,
                    )

                # Draw GT boxes in green.
                gt_boxes = extract_gt_boxes(gt_target, frame_w, frame_h)
                for box in gt_boxes:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw ROI boxes used for this frame in blue.
                for roi in result.rois_used:
                    x1, y1, x2, y2 = roi
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                mode_text = 'FULL' if result.use_full_frame else 'ROI'
                dropped_count = len(result.dropped_tracker_detections)
                overlay = (
                    f'Frame {current_frame_idx}/{total_frames} | Mode: {mode_text} '
                    f'| Delay: {frame_delay}ms | Dropped for tracker: {dropped_count}'
                )
                cv2.putText(display_frame, overlay, (10, 25), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 2)

                if paused:
                    cv2.putText(
                        display_frame,
                        '[PAUSED] (f=next, space=resume, ESC=quit)',
                        (10, 50),
                        cv2.FONT_HERSHEY_PLAIN,
                        1.0,
                        (0, 0, 255),
                        2,
                    )

                cv2.imshow(window_name, display_frame)
                step_once = False
            else:
                # Keep UI responsive while paused before first frame is processed.
                key = cv2.waitKey(10) & 0xFF
                if key == 27 or key == ord('q'):
                    print('Exiting...')
                    break
                if key == 32 or key == ord('p'):
                    paused = not paused
                    state = 'PAUSED' if paused else 'PLAYING'
                    print(f'[{state}]')
                elif key == ord('f') and paused:
                    step_once = True
                elif key == ord('+') or key == ord('='):
                    frame_delay = min(frame_delay + 50, 5000)
                    print(f'Frame delay: {frame_delay}ms')
                elif key == ord('-') or key == ord('_'):
                    frame_delay = max(frame_delay - 50, 10)
                    print(f'Frame delay: {frame_delay}ms')
                continue

            elapsed_ms = (time.perf_counter() - loop_start) * 1000.0
            wait_ms = max(1, int(frame_delay - elapsed_ms))
            key = cv2.waitKey(wait_ms) & 0xFF

            if key == 27 or key == ord('q'):
                print('Exiting...')
                break
            elif key == 32 or key == ord('p'):
                paused = not paused
                state = 'PAUSED' if paused else 'PLAYING'
                print(f'[{state}]')
            elif key == ord('f') and paused:
                step_once = True
                print(f'Frame {min(current_frame_idx + 1, total_frames)}/{total_frames}')
            elif key == ord('+') or key == ord('='):
                frame_delay = min(frame_delay + 50, 5000)
                print(f'Frame delay: {frame_delay}ms')
            elif key == ord('-') or key == ord('_'):
                frame_delay = max(frame_delay - 50, 10)
                print(f'Frame delay: {frame_delay}ms')

    cv2.destroyAllWindows()
    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sequential inference with Kalman ROI tracking')
    parser.add_argument(
        '--benchmark-config',
        dest='benchmark_config',
        default='config/benchmark-vid.yaml',
        type=str,
        help='Path to the benchmark configuration file',
    )
    parser.add_argument(
        '--frame-delay',
        dest='frame_delay',
        default=None,
        type=int,
        help='Override frame delay in milliseconds',
    )
    parser.add_argument(
        '--default-frame-delay',
        dest='default_frame_delay',
        default=33,
        type=int,
        help='Frame delay used when benchmark config does not define one and --frame-delay is not set',
    )

    args = parser.parse_args()
    with torch.no_grad():
        infer_sequentially_with_roi(args)
