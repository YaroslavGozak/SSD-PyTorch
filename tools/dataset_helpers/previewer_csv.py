import glob
import time
import cv2
import pandas as pd

from dataset.testtransform_dataset import TestTransformDataset

# --- CONFIG ---
VID_NAME = "uav0000145_00000_v"
ANNOTATIONS_PATH = f"H:/Projects/University/NeuralNetworks_ModelsAndDatasets/Datasets/VisDrone2019-VID-train/VisDrone2019-VID-train/annotations/{VID_NAME}.txt"
IMAGES_DIR = f"H:/Projects/University/NeuralNetworks_ModelsAndDatasets/Datasets/VisDrone2019-VID-train/VisDrone2019-VID-train/sequences/{VID_NAME}"
DISPLAY_INTERVAL = 0.01
OBJECT_ID = 5
# ----------------

dataset = TestTransformDataset(512)

# 1. Read the image
image_paths = sorted(glob.glob(f"{IMAGES_DIR}/*.jpg"))
print(f"Found {len(image_paths)} images in {IMAGES_DIR}")
if not image_paths:
    raise RuntimeError(f"No images found in {IMAGES_DIR}")

# 2. Read annotations
df = pd.read_csv(ANNOTATIONS_PATH, names=["frame_id", "object_id", "xpos", "ypos", "width", "height", "score", "object_category", "truncation", "occlusion"])

# rows = df.loc[True]

idx = 0
last_switch = time.time()

while True:
    now = time.time()
    # 3. Time to switch image?
    if now - last_switch >= DISPLAY_INTERVAL:
        idx += 1
        if idx >= len(image_paths):
            break
        last_switch = now

    # 4. Load and annotate
    rows = df[df['frame_id'] == idx]
    img = cv2.imread(image_paths[idx])
    if img is None:
        continue
    
    # 5.1. Define rectangle coordinates: (x, y, width, height)
    # row = rows.loc[df['frame_id'] == idx + 1]
    if len(rows) > 0:
        for _, row in rows.iterrows():
            xpos = int(row["xpos"])
            ypos = int(row["ypos"])
            width = int(row["width"])
            height = int(row["height"])
            label = dataset.idx2label.get(row["object_category"], str(row["object_category"]))
            occlusion = row["occlusion"]
        
            # 5.2. Draw a rectangle (color BGR=green, thickness=2)
            top_left     = (xpos, ypos)
            bottom_right = (xpos + width, ypos + height)
            cv2.rectangle(img, top_left, bottom_right, color=(0, 255, 0), thickness=2)
            cv2.putText(img, str(occlusion), (xpos, max(ypos - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)
    cv2.imshow("Slideshow", img)

    # 6. WaitKey with small delay lets window refresh and catch keypress
    key = cv2.waitKey(100)
    if key == ord('q'):  # press 'q' to quit
        break
    if key == ord('s'):  # press 's' to wait for a keypress
        cv2.waitKey(0)

cv2.destroyAllWindows()
