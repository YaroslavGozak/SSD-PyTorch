"""Preparation-only smoke: one padded/fixed crop and one FF/SEG frame per model."""
import os
os.environ['SSD_INFERENCE_DEVICE'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import copy
from pathlib import Path
from types import SimpleNamespace
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import torch
from experiments import cpu_revalidation as launcher
from experiments.cpu_pilot import run_pilot as pilot
from tools import infer
from tools.helpers.pipeline import process_frame
from tools.trackers.static_padding_tracker import StaticPaddingTracker


def main():
    pilot._configure_runtime()
    infer.device = torch.device('cpu')
    output = launcher.DEFAULT_OUTPUT / 'validation_checks'
    output.mkdir(exist_ok=False)
    evidence = {}
    with torch.inference_mode():
        for name in launcher.MODELS:
            directory = output / name
            directory.mkdir()
            args = SimpleNamespace(config_path=str(launcher.DEFAULT_OUTPUT / 'configs' / f'{name}.yaml'),
                                   summary_path=str(directory), results_path=str(directory), audit_details=True,
                                   num_workers=0, eval_batch_size=1, pin_memory=False)
            yolo = name == 'yolo26n'
            pad = 'fixed_padding_roi_crop_yolo_10' if yolo else 'fixed_padding_roi_crop_10'
            loaded = list(infer.load_model_and_dataset(args, pad))
            model, dataset, _loader, config = loaded
            dataset.images_info = dataset.images_info[:1]
            resources = loaded
            runs = []
            for transform in (pad, 'fixed_size_yolo_160x192' if yolo else 'fixed_size_160x192'):
                runs.append(infer._evaluate_single_transform(args, transform_name=transform,
                    evaluated_dataset=transform, model_name=config['train_params']['model'], resources=resources))
            # Reuse the loaded model with its ordinary full-frame dataset transform.
            from dataset.transforms.ssd_transform import SsdTransform
            full_dataset = copy.copy(dataset)
            full_dataset.transforms = SsdTransform(320, dataset.im_mean, dataset.imagenet_mean, dataset.imagenet_std).transforms
            tensor, _target, filename = full_dataset[0]
            if yolo:
                model.input_normalization = 'imagenet'
            image = cv2.imread(filename)
            h, w = image.shape[:2]
            events, hook = pilot._attach_shape_observer(model)
            pipeline_results = []
            try:
                for mode, rois, frame_index in [('FF', [], 0), ('SEG', [[w//4, h//4, 3*w//4, 3*h//4]], 1)]:
                    before = len(events)
                    tracker = StaticPaddingTracker(pad_x=32, pad_y=32, conf_det_min=config['train_params']['infer_conf_threshold'],
                                                   model_input_size=(320, 320), hold_last_for_frames=0)
                    result = process_frame(model=model, idx2label=dataset.idx2label, frame_bgr=image,
                        im_tensor=tensor.unsqueeze(0), tracker=tracker, next_frame_rois=rois, frame_idx=frame_index,
                        key_frame_interval=10, im_size_hw=(320, 320), conf_threshold=config['train_params']['infer_conf_threshold'],
                        nms_iou=.45, merge_fn=lambda rois, **_: list(rois), model_device=torch.device('cpu'),
                        roi_grid=pilot.ROI_GRID[name])
                    assert result.use_full_frame == (mode == 'FF')
                    assert events[before:]
                    pipeline_results.append({'mode': mode, 'actual_shapes': events[before:], 'detections': len(result.final_detections)})
            finally:
                hook.remove()
            evidence[name] = {'status': 'PASS', 'accuracy_examples': runs, 'pipeline_examples': pipeline_results,
                              'device': 'cpu', 'torch_threads': torch.get_num_threads()}
            launcher.write_json(output / 'technical_smoke.json', evidence)
            del model, dataset, loaded, resources, full_dataset
    print('PASS: three checkpoints; one pad and one fixed-size example plus FF/SEG per model.')


if __name__ == '__main__':
    main()
