import glob
import os
from os import walk
from pathlib import Path
import shutil
from PIL import Image
import pandas as pd
import xml.etree.ElementTree as ET

# --- CONFIG ---
COPY_IMAGES = False
COPY_ANNOTATIONS = True
ANNOTATIONS_DIR = "H:/Projects/University/NeauralNetworks/Datasets/VisDrone2019-VID-train/VisDrone2019-VID-train/Annotation"
ANNOTATIONS_OLD_DIR = "H:/Projects/University/NeauralNetworks/Datasets/VisDrone2019-VID-train/VisDrone2019-VID-train/annotations"
IMAGES_DIR = "H:/Projects/University/NeauralNetworks/Datasets/VisDrone2019-VID-train/VisDrone2019-VID-train/JPEGImages"
IMAGES_OLD_DIR = "H:/Projects/University/NeauralNetworks/Datasets/VisDrone2019-VID-train/VisDrone2019-VID-train/sequences"
DISPLAY_INTERVAL = 0.01
OBJECT_ID = 5
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

def copy_and_rename(src_path, dest_folder, new_name):
    # 2) Ensure destination exists
    os.makedirs(dest_folder, exist_ok=True)

    # 3) Copy the file into dest_folder
    shutil.copy(src_path, dest_folder)

    # 4) Rename it
    old_path = os.path.join(dest_folder, os.path.basename(src_path))
    new_path = os.path.join(dest_folder, new_name)
    os.rename(old_path, new_path)

def dict_to_xml(tag, data, parent = None):
    """
    Recursively convert a dict or list into an XML Element.
    
    Args:
        tag (str):  The tag name for this element.
        data:       A dict, list, or scalar to convert.
    
    Returns:
        xml.etree.ElementTree.Element
    """
    elem = ET.Element(tag)
    # If this is a dict, iterate its items
    if isinstance(data, dict):
        for key, val in data.items():
            child = dict_to_xml(key, val, elem)
            if child is not None:
                elem.append(child)
    # If this is a list, create repeated subelements
    elif isinstance(data, list):
        for item in data:
            # use the tag name for each item
            child = dict_to_xml(tag, item, elem)
            parent.append(child)
        return
    # Otherwise, treat as text node
    else:
        elem.text = str(data)
    return elem

if __name__ == "__main__":
    # 1. Read the images
    if COPY_IMAGES:
        for (dirpath, dirnames, filenames) in walk(IMAGES_OLD_DIR):
            for dirname in dirnames:
                image_paths = sorted(glob.glob(f"{dirpath}/{dirname}/*.jpg"))
                if not image_paths:
                    raise RuntimeError(f'No images found in {dirpath}/{dirname}')
                for idx, imgpath in enumerate(image_paths):
                    imgname = os.path.basename(imgpath)
                    copy_and_rename(imgpath, IMAGES_DIR, dirname + '_' + imgname)
                    if idx % 100 == 0:
                        print(f'Copied {idx} images of {len(image_paths)}')

    if COPY_ANNOTATIONS:
        for (_, _, filenames) in walk(ANNOTATIONS_OLD_DIR):
            for ann_idx, filename in enumerate(filenames):
                filename_no_ext = Path(filename).stem
                df = pd.read_csv(ANNOTATIONS_OLD_DIR + '/' + filename, 
                                 names=["frame_id", "object_id", "xpos", "ypos", "width", "height", "score", "class", "truncation", "occlusion"])
                df = df.reset_index()  # make sure indexes pair with number of rows
                frames = {}
                for index, row in df.iterrows():
                    frame_id = str(row["frame_id"])
                    frame_name = filename_no_ext + '_' + frame_id.zfill(7)
                    image_name = f"{frame_name}.jpg"
                    if frame_name not in frames:
                        im = Image.open(IMAGES_DIR + '/' + image_name)
                        width, height = im.size
                        frame = {
                            "annotation": {
                                "filename": image_name,
                                "folder": "VisDrone2019",
                                "size": {
                                    "width": width,
                                    "height": height,
                                    "depth": 3
                                },
                                "object": [
                                ],
                                "folder": filename_no_ext
                            }
                        }
                        frames[frame_name] = frame
                    frame = frames[frame_name]
                    frame["annotation"]["object"].append({
                        "name": row["object_id"],
                        "pose": "Unspecified",
                        "truncated": row["truncation"],
                        "bndbox":{
                            "xmin": row["xpos"],
                            "ymin": row["ypos"],
                            "xmax": row["xpos"] + row["width"],
                            "ymax": row["ypos"] + row["height"]
                        }
                    })

                for fr_idx, key in enumerate(frames):
                    # Build XML tree
                    print(f'Converting frame {fr_idx}/{len(frames)} of annotation {ann_idx}/{len(filenames)} to XML')
                    frame = frames[key]
                    root_key = next(iter(frame))
                    root = dict_to_xml(root_key, frame[root_key])
                    tree = ET.ElementTree(root)
                    ET.indent(tree, space="\t", level=0)

                    # Write to file
                    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
                    file_location = ANNOTATIONS_DIR + '/' + key + '.xml'
                    tree.write(file_location, encoding="utf-8", xml_declaration=False)                
