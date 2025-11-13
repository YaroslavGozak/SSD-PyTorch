import glob
import time
import cv2
import pandas as pd

# --- CONFIG ---
ANNOTATIONS_PATH = "H:/Projects/University/NeauralNetworks/Datasets/VisDrone2019-VID-train/VisDrone2019-VID-train/annotations/uav0000013_00000_v.txt"
IMAGES_DIR = "H:/Projects/University/NeauralNetworks/Datasets/VisDrone2019-VID-train/VisDrone2019-VID-train/sequences/uav0000013_00000_v"
DISPLAY_INTERVAL = 0.01
OBJECT_ID = 5
# ----------------

# 1. Read the image
image_paths = sorted(glob.glob(f"{IMAGES_DIR}/*.jpg"))
print(f"Found {len(image_paths)} images in {IMAGES_DIR}")
if not image_paths:
    raise RuntimeError(f"No images found in {IMAGES_DIR}")

# 2. Read annotations
df = pd.read_csv(ANNOTATIONS_PATH, names=["frame_id", "object_id", "xpos", "ypos", "width", "height", "class", "confidence", "na", "na2"])

rows = df.loc[df['object_id'] == OBJECT_ID]

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
    img = cv2.imread(image_paths[idx])
    if img is None:
        continue
    
    # 5.1. Define rectangle coordinates: (x, y, width, height)
    row = rows.loc[df['frame_id'] == idx + 1]
    if len(row) > 0:
        xpos = int(row["xpos"])
        ypos = int(row["ypos"])
        width = int(row["width"])
        height = int(row["height"])
    
        # 5.2. Draw a rectangle (color BGR=green, thickness=2)
        top_left     = (xpos, ypos)
        bottom_right = (xpos + width, ypos + height)
        cv2.rectangle(img, top_left, bottom_right, color=(0, 255, 0), thickness=2)
    cv2.imshow("Slideshow", img)

    # 6. WaitKey with small delay lets window refresh and catch keypress
    key = cv2.waitKey(100)
    if key == ord('q'):  # press 'q' to quit
        break

cv2.destroyAllWindows()
