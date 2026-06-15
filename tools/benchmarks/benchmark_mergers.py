
import glob
import os
import time
import psutil

from dataset.testtransform_dataset import TestTransformDataset, load_image_and_ann
from tools.mergers.greedy import greedy_roi_merge
from tools.mergers.simple import simple_roi_merge
from tools.mergers.simple2 import simple_roi_merge_v2

dataset = TestTransformDataset(512)

IMG_DIR = "H:\\Projects\\University\\NeuralNetworks_ModelsAndDatasets\\Datasets\\VisDrone2019-VID-train\\VisDrone2019-VID-train\\ResizedSequences\\uav0000124_00944_v"

if __name__ == '__main__':
    results = {}
    for merger_func, label in [(simple_roi_merge, "Simple"), (simple_roi_merge_v2, "Simple_v2"), (greedy_roi_merge, "Greedy")]:
        print(f"Benchmarking {label} merger...")
        image_paths = sorted(glob.glob(f"{IMG_DIR}/*.jpg"))
        ims_num = len(image_paths)
        start_time = time.time()
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / (1024 ** 2)
        for idx, img_path in enumerate(image_paths):
            im_info = load_image_and_ann(img_path, dataset.label2idx)
            bboxes = [detection['bbox'] for detection in im_info['detections']]
            bboxes = [dataset.add_padding_to_bbox(bbox, 512, 512, dataset.alpha_w, dataset.alpha_h, dataset.delta_x, dataset.delta_y) for bbox in bboxes]
        
            _ = merger_func(bboxes)
        end_time = time.time()
        end_mem = process.memory_info().rss / (1024 ** 2)
        results[label] = (end_time - start_time, end_mem - start_mem)

    print("Benchmark Results:")
    for label, (t, m) in results.items():
        print(f"{label}: Time {t/ims_num:.4f}s/im, Memory {m:.2f}MB")