import os
import random
import time
import cv2
import pandas as pd
from torchvision.io import read_image

from tools.utils import read_annotation_file

# --- CONFIG ---
ANNOTATIONS_DIR = "H:\\Projects\\University\\NeuralNetworks_ModelsAndDatasets\\Datasets\\VisDrone2019-VID-train\\VisDrone2019-VID-train\\SequenceAnnotations"
IMAGES_DIR = "H:\\Projects\\University\\NeuralNetworks_ModelsAndDatasets\\Datasets\\VisDrone2019-VID-train\\VisDrone2019-VID-train\\ResizedSequences"
DISPLAY_INTERVAL = 2.0
# ----------------

categories = [
    {"id": 1, "name": "pedestrian"},
    {"id": 2, "name": "people"},
    {"id": 3, "name": "bicycle"},
    {"id": 4, "name": "car"},
    {"id": 5, "name": "van"},
    {"id": 6, "name": "truck"},
    {"id": 7, "name": "tricycle"},
    {"id": 8, "name": "awning-tricycle"},
    {"id": 9, "name": "bus"},
    {"id": 10, "name": "motor"},
]
classes = [category['name'] for category in categories]
# classes = sorted(classes)
# We need to add background class as well with 0 index
classes = ['background'] + classes

label2idx = {classes[idx]: idx for idx in range(len(classes))}
idx2label = {idx: classes[idx] for idx in range(len(classes))}

# 1. Read list of annotation directories
dirs = [d for d in os.listdir(ANNOTATIONS_DIR) if os.path.isdir(os.path.join(ANNOTATIONS_DIR, d))]

if not dirs:
    raise RuntimeError("No directories found")


idx = 0
last_switch = time.time()

while True:
    now = time.time()
    # 3. Time to switch image?
    if now - last_switch < DISPLAY_INTERVAL:
        continue
    last_switch = now

    # 4. Load annotation directory
    ann_dir = random.choice(dirs)
    ann_dir_path = os.path.join(ANNOTATIONS_DIR, ann_dir)
    im_dir_path = os.path.join(IMAGES_DIR, ann_dir)
    files = [f for f in os.listdir(ann_dir_path) if os.path.isfile(os.path.join(ann_dir_path, f))]

    if not files:
        raise RuntimeError(f"No files found in directory: {ann_dir_path}")

    # 4.1 Pick a random annotation file
    ann_file = random.choice(files)

    im_info, success = read_annotation_file(ann_dir_path, im_dir_path, ann_file, label2idx)
    if not success:
        break
    if len(im_info.get('detections', [])) == 0:
        break
    
    im = read_image(im_info['filename'])
    # img = cv2.imread(image_paths[idx])
    if im is None:
        break
    
    # Convert PyTorch tensor to NumPy array for OpenCV
    # torchvision returns (C, H, W) format, OpenCV expects (H, W, C)
    im = im.permute(1, 2, 0).numpy()  # Change from (C, H, W) to (H, W, C)

    # 5.1. Define rectangle coordinates: (x, y, width, height)
    print(f"Detecions number: {len(im_info['detections'])} in image {im_info['filename']}")
    for det in im_info['detections']:
        label_idx = det['label']

        top_left     = (int(det['bbox'][0]), int(det['bbox'][1]))
        bottom_right = (int(det['bbox'][2]), int(det['bbox'][3]))
        cv2.rectangle(im, top_left, bottom_right, color=(0, 255, 0), thickness=2)
        # cv2.putText(im, idx2label[label_idx], (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        text = idx2label[label_idx]
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)
        text_w, text_h = text_size
        # cv2.rectangle(im, (top_left[0], top_left[1]), (top_left[0] + 10 + text_w, top_left[1] + 10 + text_h), [255, 255, 255], -1)
        # cv2.putText(im, text=idx2label[label_idx],
        #                 org=(top_left[0] + 5, top_left[1] + 15),
        #                 thickness=1,
        #                 fontScale=1,
        #                 color=[0, 0, 0],
        #                 fontFace=cv2.FONT_HERSHEY_PLAIN)
    cv2.imshow("Slideshow", im)

    # 6. WaitKey with small delay lets window refresh and catch keypress
    key = cv2.waitKey(100)
    if key == ord('q'):  # press 'q' to quit
        break

cv2.destroyAllWindows()
