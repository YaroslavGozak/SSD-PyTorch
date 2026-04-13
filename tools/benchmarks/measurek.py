import argparse
import time

import numpy as np
import torchvision
from dataset.voc import VOCDataset
from model.roissd_mobilenet import RoiSSDMobileNet
from tools.voc.adapters.coco_to_voc_adapter import CocoToVocAdapter
import torch
import yaml

from model.roissd import RoiSSD


def run_forward(model, x, is_yolo=False):
    """Run forward for both project models and plain PyTorch modules."""
    if is_yolo:
        # YOLO model only takes input tensor, no targets argument
        out = model(x)
    else:
        # Project SSD models expect (x, None) for targets
        out = model(x, None)
    
    if isinstance(out, tuple):
        return out
    return None, out

def measure_time(model, device, iters=50, is_yolo=False):
    model.eval().to(device)
    is_yolo=True
    use_cuda_sync = str(device).startswith("cuda") and torch.cuda.is_available()

    # For YOLO, test standard detection sizes. For SSD, test custom sizes.
    if is_yolo:
        sizes = [(32,32), (64,64), (96,96), (320, 320), (416, 416), (640, 640)]  # Standard YOLO sizes
    else:
        sizes = [(32,32), (64,64), (96,96), (140,140), (200,200), (300,300)]
    results = []

    # Warmup - use consistent size for both models
    with torch.no_grad():
        for _ in range(10):
            if is_yolo:
                x = torch.randn(1, 3, 640, 640, device=device)
            else:
                x = torch.randn(1, 3, 300, 300, device=device)
            try:
                _ = run_forward(model, x, is_yolo=is_yolo)
            except RuntimeError as e:
                print(f"Warning during warmup: {str(e)[:100]}")
                pass

    for (H,W) in sizes:
        times = []
        with torch.no_grad():
            for _ in range(iters):
                x = torch.randn(1,3,H,W,device=device)
                if use_cuda_sync:
                    torch.cuda.synchronize()
                t0 = time.time()
                try:
                    _ = run_forward(model, x, is_yolo=is_yolo)
                except RuntimeError as e:
                    print(f"Error at {H}x{W}: {str(e)[:80]}")
                    continue
                if use_cuda_sync:
                    torch.cuda.synchronize()
                t1 = time.time()
                times.append(t1 - t0)
        
        if times:
            mean_t = sum(times) / len(times)
            A = H * W
            results.append((A, mean_t))
            print(f"Size {H}x{W}, A={A}, mean time={mean_t*1000:.3f} ms, iters={len(times)}")

    # Linear fit: T = K + c*A
    if len(results) >= 1:
        A = np.array([r[0] for r in results], dtype=np.float64)
        T = np.array([r[1] for r in results], dtype=np.float64)
        
        if len(results) > 1:
            c_t, K_t = np.polyfit(A, T, 1)
            τ = K_t / c_t if c_t != 0 else float('nan')
            print(f"Measured device {device}. Estimated c_t={c_t:.8f} sec/pixel, K_t={K_t:.8f} sec, τ={τ:.1f} pixels")
        else:
            print(f"Single measurement: {A[0]} pixels, {T[0]*1000:.3f} ms")
    
    return results

def build_fcos_model(num_classes):
    return torchvision.models.detection.fcos_resnet50_fpn(
        weights=None,
        weights_backbone=torchvision.models.ResNet50_Weights.DEFAULT,
        num_classes=num_classes,
    )


def build_fasterrcnn_model(num_classes):
    return torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=None,
        weights_backbone=torchvision.models.MobileNet_V3_Large_Weights.DEFAULT,
        num_classes=num_classes,
        box_score_thresh=0.9,
    )

# -------------------------------------------------------------
# Main execution
# -------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure inference cost model T = K + cA")
    parser.add_argument("--config", default="config/voc.yaml", help="Path to training config")
    parser.add_argument("--model", choices=["yolo", "roissd", "roissd-mobilenet", "fcos", "fasterrcnn"], default="yolo", help="Model to benchmark")
    parser.add_argument("--yolo-weights", default="yolov8n.pt", help="Ultralytics YOLO weights path")
    parser.add_argument("--iters", type=int, default=50, help="Timed iterations per image size")
    parser.add_argument("--cuda", action="store_true", help="Also benchmark on CUDA if available")
    args = parser.parse_args()

    # Read the config file #
    with open(args.config, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
            raise

    dataset_config = config['dataset_params']
    train_config = config['train_params']
    model = None
    is_yolo = False

    dataset = VOCDataset('test',
        im_sets=dataset_config['test_im_sets'],
        im_size=dataset_config['im_size'],
        transform_name='ssd')

    if args.model == "roissd":
        model = RoiSSD(
            config=config['model_params'],
            num_classes=dataset_config['num_classes']
        )
        print("Benchmarking model: RoiSSD")
        is_yolo = False
    elif args.model == "roissd-mobilenet":
        model = RoiSSDMobileNet(
            config=config['model_params'],
            num_classes=dataset_config['num_classes']
        )
        print("Benchmarking model: RoiSSDMobileNet")
        is_yolo = False
    elif args.model == 'fcos':
        model = build_fcos_model(num_classes=dataset_config['num_classes'])
        print("Benchmarking model: FCOS")
        is_yolo = False
    elif args.model == 'fasterrcnn':
        base_model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT,
            box_score_thresh=0.9,
        )
        model = CocoToVocAdapter(
            base_model=base_model,
            voc_label2idx=dataset.label2idx,
            conf_threshold=train_config.get('infer_conf_threshold', 0.05),
            normalize_boxes=True,
        )
        is_yolo = False
    elif args.model == 'yolo':
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "YOLO benchmark requires ultralytics. Install with: pip install ultralytics"
            ) from e
        model = YOLO(args.yolo_weights).model
        is_yolo = True
        print(f"Benchmarking model: YOLO ({args.yolo_weights})")
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    measure_time(model, device='cpu', iters=args.iters, is_yolo=is_yolo)
    if args.cuda and torch.cuda.is_available():
        measure_time(model, device='cuda', iters=args.iters, is_yolo=is_yolo)


    print("\nDone.")
