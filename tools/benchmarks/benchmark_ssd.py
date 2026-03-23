import numpy as np
import torch
import torch.nn as nn
from torchvision.models.detection import ssd300_vgg16, ssdlite320_mobilenet_v3_large
import yaml
from model.roissd import RoiSSD
from model.ssd import SSD
from model.ssd_mobilenet import SSDMobileNet
from thop import profile
from fvcore.nn import FlopCountAnalysis
import torch.nn.functional as F

import torch
import time

def measure_time(model, device="cuda", iters=50):
    model.eval().to(device)

    sizes = [(32,32), (64,64), (96,96), (140,140), (200,200), (300,300)]
    results = []

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            x = torch.randn(1, 3, 300, 300, device=device)
            _ = model(x)

    for (H,W) in sizes:
        times = []
        with torch.no_grad():
            for _ in range(iters):
                x = torch.randn(1,3,H,W,device=device)
                torch.cuda.synchronize()  # ensure previous ops finished
                t0 = time.time()
                _ = model(x)
                torch.cuda.synchronize()
                t1 = time.time()
                times.append(t1 - t0)

        mean_t = sum(times) / len(times)
        A = H * W
        results.append((A, mean_t))
        print(f"Size {H}x{W}, A={A}, mean time={mean_t*1000:.3f} ms")

    # Linear fit: T = K + c*A
    A = np.array([r[0] for r in results], dtype=np.float64)
    T = np.array([r[1] for r in results], dtype=np.float64)

    # fit T = c*A + K
    c_t, K_t = np.polyfit(A, T, 1)
    τ = K_t / c_t
    print(f"Estimated c_t={c_t:.8f} sec/pixel, K_t={K_t:.8f} sec, τ={τ:.1f} pixels")
    
    return c_t, K_t, results


# -------------------------------------------------------------
# Helper: run FLOP/MAC profiler using THOP + fvcore
# -------------------------------------------------------------
def benchmark_model(model, input_tensor, name="Model"):
    print(f"\n==================== {name} ====================")

    # 1. THOP MACs + Params
    macs, params = profile(model, inputs=(input_tensor,), verbose=False)
    print(f"[THOP] MACs: {macs/1e9:.3f} GMACs, Params: {params/1e6:.3f} M")

    # 2. fvcore FLOPS
    # flops = FlopCountAnalysis(model, input_tensor)
    # print(f"[fvcore] FLOPs: {flops.total()/1e9:.3f} GFLOPs")

    print("=============================================")


# -------------------------------------------------------------
# Main execution
# -------------------------------------------------------------
if __name__ == "__main__":
    device = "cpu"

    # Read the config file #
    with open('config/vis-drone.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    dataset_config = config['dataset_params']
    train_config = config['train_params']

    # Dummy images
    img_ssd300 = torch.randn(1, 3, 300, 300).to(device)
    img_ssd320 = torch.randn(1, 3, 320, 320).to(device)

    # ---------------------------------------------------------
    # Benchmark SSD-VGG (SSD300)
    # ---------------------------------------------------------
    ssd_vgg = ssd300_vgg16(weights=None).to(device)
    ssd_vgg.eval()
    benchmark_model(ssd_vgg, img_ssd300, "SSD-VGG (SSD300)")

    # ---------------------------------------------------------
    # Benchmark SSD-VGG (SSD300) ours
    # ---------------------------------------------------------
    ssd_vgg_ours = SSD(config=config['model_params'],
                num_classes=dataset_config['num_classes']).to(device)
    ssd_vgg_ours.eval()
    benchmark_model(ssd_vgg_ours, img_ssd300, "SSD-VGG (SSD300) ours")

    # ---------------------------------------------------------
    # Benchmark SSD-MobileNet-Lite (SSDlite320)
    # ---------------------------------------------------------
    ssd_mnet = ssdlite320_mobilenet_v3_large(weights=None).to(device)
    ssd_mnet.eval()
    benchmark_model(ssd_mnet, img_ssd320, "SSD-MobileNet (SSD-Lite320)")

    # ---------------------------------------------------------
    # Benchmark SSD-MobileNet-Lite (SSDlite320) ours
    # ---------------------------------------------------------
    ssd_mnet_ours = SSDMobileNet(config=config['model_params'],
                num_classes=dataset_config['num_classes']).to(device)
    ssd_mnet_ours.eval()
    benchmark_model(ssd_mnet_ours, img_ssd320, "SSD-MobileNet (SSD-Lite320) ours")

    # ---------------------------------------------------------
    # Benchmark ROI-SSD (selective)
    # ---------------------------------------------------------
    roi_ssd = RoiSSD(config=config['model_params'],
                num_classes=dataset_config['num_classes']).to(device)
    roi_ssd.eval()

    measure_time(roi_ssd, device=device, iters=50)

    # Full-frame case
    benchmark_model(roi_ssd, img_ssd300, "ROI-SSD (Full Frame)")

    # ROI = 25% of image (example)
    # for i in [4]:
    #     roi = torch.randn(1, 3, i, i).to(device)   # center crop
    #     print("\n--- ROI mode (only 25% of frame used) ---")
    #     # wrap forward into lambda so THOP can analyze:
    #     benchmark_model(roi_ssd, roi, "ROI-SSD {}x{}".format(i, i))

    print("\nDone.")
