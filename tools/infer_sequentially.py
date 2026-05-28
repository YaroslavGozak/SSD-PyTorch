from dataset.visdrone import VisDroneDataset
import torch
import argparse
import os
import yaml
from tqdm import tqdm
from dataset.ytbb import YTBBDataset
from dataset.yolo_imagenet_vid import YoloImageNetVidDataset
from model.roissd import RoiSSD
from model.ssd import SSD
import cv2
import time
from dataset.voc import VOCDataset
from dataset.voc_small_objects import VOCSmallObjectsDataset
from torch.utils.data.dataloader import DataLoader
from tools.helpers.label_compat import get_model_num_classes
from model.model_adapters import unwrap_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using mps')


def load_model_and_dataset(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    ########################

    dataset_config = config['dataset_params']
    train_config = config['train_params']
    model_num_classes = get_model_num_classes(train_config, dataset_config, train_config['dataset'])

    if str(train_config['dataset']) == 'vis-drone':
        dataset = VisDroneDataset('test',
                     im_sets=dataset_config['test_im_sets'],
                     im_size=dataset_config['im_size'])
    elif str(train_config['dataset']) == 'ytbb':
        dataset = YTBBDataset('test',
                     root_dir=dataset_config['root_dir'],
                     im_size=dataset_config['im_size'])
    elif str(train_config['dataset']) == 'voc':
        dataset = VOCDataset('test',
                     im_sets=dataset_config['test_im_sets'],
                     im_size=dataset_config['im_size'],
                     transform_name=dataset_config['transform_name'])
    elif str(train_config['dataset']) == 'voc-small-objects':
        dataset = VOCSmallObjectsDataset('test',
                     im_sets=dataset_config['test_im_sets'],
                     im_size=dataset_config['im_size'],
                     transform_name=dataset_config['transform_name'])
    elif str(train_config['dataset']) == 'yolo-imagenet-vid':
        dataset = YoloImageNetVidDataset(
                     'test',
                     yolo_dataset_yaml=dataset_config['yolo_dataset_yaml'],
                     im_size=dataset_config['im_size'],
                     transform_name=dataset_config['transform_name'])
    else:
        raise Exception('Unknown dataset name {}'.format(train_config['dataset']))
    test_dataset_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    if str(train_config['model']) == 'ssd':
        model = SSD(config=config['model_params'],
                num_classes=model_num_classes)
    elif str(train_config['model']) == 'roissd':
        model = RoiSSD(config=config['model_params'],
                num_classes=model_num_classes)
    model.to(device=torch.device(device))
    model.eval()

    model_task_path = os.path.join('trained_models', train_config['task_name'])
    model_checkpoint_path = os.path.join(model_task_path, train_config['ckpt_name'])
    assert os.path.exists(model_checkpoint_path), \
        "No checkpoint exists at {}".format(model_checkpoint_path)
    # Load checkpoint if it exists (after creating optimizer and scheduler)
    if os.path.exists(model_checkpoint_path):
        print('Loading checkpoint as one exists')
        checkpoint = torch.load(
            model_checkpoint_path,
            map_location=device)
        
        # Handle both old format (state_dict only) and new format (full checkpoint)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            print('Restored optimizer and scheduler state')
        else:
            # Old format - just model state_dict
            model.load_state_dict(checkpoint)
            print('Loaded model only (old checkpoint format)')

    return model, dataset, test_dataset_loader, config


def infer_sequentially(args):
    """
    Process and display test images in a single streaming-style loop.
    
    Controls:
    - SPACE or 'p': Pause/Resume
    - 'f': Forward to next frame (when paused)
    - 'ESC' or 'q': Quit
    - '+'/'-': Increase/Decrease frame delay
    """
    
    model, dataset, test_dataset_loader, config = load_model_and_dataset(args)
    conf_threshold = config['train_params']['infer_conf_threshold']
    unwrap_model(model).low_score_threshold = conf_threshold

    window_name = 'Sequential Inference Results - {} dataset'.format(
        config['train_params']['dataset'])
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    paused = False
    frame_delay = args.frame_delay  # milliseconds
    current_frame_idx = 0
    total_frames = len(test_dataset_loader)

    print(f'Total frames: {total_frames}')
    print('Controls: SPACE=pause/resume, f=next(paused), ESC=quit, +/-=adjust delay')
    print('Starting streaming inference...\n')

    data_iter = iter(test_dataset_loader)
    current_frame = None
    step_once = False

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
                im_tensor = im_tensor.float().to(device)
                _, ssd_detections = model(im_tensor, [target] if hasattr(model, 'training') and model.training else None)

                # Read original image
                gt_im = cv2.imread(fpath)
                h, w = gt_im.shape[:2]
                current_frame = gt_im.copy()

                # Draw predicted boxes
                boxes = ssd_detections[0]['boxes']
                labels = ssd_detections[0]['labels']
                scores = ssd_detections[0]['scores']

                for idx, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.detach().cpu().numpy()
                    x1, y1, x2, y2 = int(w * x1), int(h * y1), int(w * x2), int(h * y2)

                    # Draw bounding box
                    cv2.rectangle(current_frame, (x1, y1), (x2, y2), thickness=2, color=[0, 0, 255])

                    # Prepare label text
                    label_text = '{} : {:.2f}'.format(
                        dataset.idx2label[labels[idx].detach().cpu().item()],
                        scores[idx].detach().cpu().item())

                    text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_PLAIN, 1, 1)
                    text_w, text_h = text_size

                    # Draw background for text
                    cv2.rectangle(current_frame, (x1, y1 - text_h - 10),
                                 (x1 + text_w + 10, y1), [0, 0, 255], -1)

                    # Draw text
                    cv2.putText(current_frame, label_text,
                               org=(x1 + 5, y1 - 5),
                               fontFace=cv2.FONT_HERSHEY_PLAIN,
                               fontScale=1,
                               color=[255, 255, 255],
                               thickness=1)

                step_once = False

            if current_frame is None:
                continue

            display_frame = current_frame.copy()

            # Update frame info
            frame_info = f'Frame {current_frame_idx}/{total_frames}'
            cv2.putText(display_frame, frame_info,
                       org=(10, 25),
                       fontFace=cv2.FONT_HERSHEY_PLAIN,
                       fontScale=1.2,
                       color=[0, 255, 0],
                       thickness=2)

            if paused:
                status_text = '[PAUSED] (f=next, space=resume, ESC=quit)'
                cv2.putText(display_frame, status_text,
                           org=(10, 50),
                           fontFace=cv2.FONT_HERSHEY_PLAIN,
                           fontScale=1,
                           color=[0, 0, 255],
                           thickness=2)

            cv2.imshow(window_name, display_frame)

            if paused:
                key = cv2.waitKey(0) & 0xFF
            else:
                elapsed_ms = (time.perf_counter() - loop_start) * 1000.0
                # If model+render time exceeded frame_delay, move on immediately.
                wait_ms = max(1, int(frame_delay - elapsed_ms))
                key = cv2.waitKey(wait_ms) & 0xFF

            # Handle key presses
            if key == 27 or key == ord('q'):  # ESC or 'q'
                print('Exiting...')
                break
            elif key == 32 or key == ord('p'):  # SPACE or 'p'
                paused = not paused
                state = 'PAUSED' if paused else 'PLAYING'
                print(f'[{state}]')
            elif key == ord('f') and paused:  # 'f' - forward one frame while paused
                step_once = True
                print(f'Frame {min(current_frame_idx + 1, total_frames)}/{total_frames}')
            elif key == ord('+') or key == ord('='):  # Increase delay
                frame_delay = min(frame_delay + 50, 5000)
                print(f'Frame delay: {frame_delay}ms')
            elif key == ord('-') or key == ord('_'):  # Decrease delay
                frame_delay = max(frame_delay - 50, 10)
                print(f'Frame delay: {frame_delay}ms')

    cv2.destroyAllWindows()
    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sequential inference with visualization')
    parser.add_argument('--config', dest='config_path',
                        default='config/voc.yaml', type=str,
                        help='Path to the configuration file')
    parser.add_argument('--frame-delay', dest='frame_delay',
                        default=100, type=int,
                        help='Delay between frames in milliseconds (default: 100)')
    args = parser.parse_args()

    with torch.no_grad():
        infer_sequentially(args)
