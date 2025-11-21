import argparse
import os
from os import walk
from pathlib import Path

import yaml
from PIL import Image
import pandas as pd
import xml.etree.ElementTree as ET

# --- CONFIG ---
IMG_SCALE_PX = 512

# Preview settings
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
classes = [category['name'] for category in categories]
# classes = sorted(classes)
# We need to add background class as well with 0 index
classes = ['background'] + classes

label2idx = {classes[idx]: idx for idx in range(len(classes))}
idx2label = {idx: classes[idx] for idx in range(len(classes))}

def copy_and_resize(src_path, dest_folder, new_name, target_size):
    """
    Copy an image from src_path to dest_folder with a new name, resizing it to target size.
    Returns the scale factor used for resizing.
    """
    # Ensure destination exists
    os.makedirs(dest_folder, exist_ok=True)

    # Read image using PIL
    img = Image.open(src_path)
    if img is None:
        raise ValueError("Image not found")

    # Get original dimensions
    original_width, original_height = img.size
    
    # Calculate scale to maintain aspect ratio
    scale = target_size / max(original_width, original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    # Resize image maintaining aspect ratio
    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Save resized image
    resized_img.save(os.path.join(dest_folder, new_name))
    
    return scale, (original_width, original_height), (new_width, new_height)

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

def convert_vis_to_voc_seq(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    #########################

    # Setup config
    dataset_config = config['dataset_params']
    split = args.split
    im_set_dir = dataset_config['train_im_sets'] if split == 'train' else dataset_config['test_im_sets']
    ANNOTATIONS_OLD_DIR = im_set_dir[0] + "/annotations"
    ANNOTATIONS_DIR = im_set_dir[0] + "/SequenceAnnotations"
    IMAGES_OLD_DIR = im_set_dir[0] + "/sequences"
    IMAGES_DIR = im_set_dir[0] + "/ResizedSequences"


    image_scales = {}
    for (_, _, filenames) in walk(ANNOTATIONS_OLD_DIR):
        for ann_idx, filename in enumerate(filenames):
            filename_no_ext = Path(filename).stem
            df = pd.read_csv(ANNOTATIONS_OLD_DIR + '/' + filename, 
                             names=["frame_id", "object_id", "xpos", "ypos", "width", "height", "score", "class", "truncation", "occlusion"])
            df = df.reset_index()  # make sure indexes pair with number of rows
            frames = {}
            video_dir = os.path.join(IMAGES_OLD_DIR, filename_no_ext)
            for _, row in df.iterrows():
                if int(row["class"]) > 10 or int(row["class"]) < 1:
                    # 'Invalid class: {}'.format(row["class"])
                    continue
                frame_id = str(row["frame_id"])
                frame_name = frame_id.zfill(7)
                image_name = f"{frame_name}.jpg"
                image_path = os.path.join(IMAGES_DIR, filename_no_ext, image_name)

                if image_path not in image_scales:
                    # If scale info is missing, we need to compute it
                    orig_image_path = os.path.join(video_dir, image_name)
                    scale, orig_size, new_size = copy_and_resize(orig_image_path, os.path.join(IMAGES_DIR, filename_no_ext), image_name, IMG_SCALE_PX)
                
                    # Store scale info for coordinate transformation
                    image_scales[image_path] = {
                        'scale': scale,
                        'orig_size': orig_size,
                        'new_size': new_size
                    }

                if frame_name not in frames:
                    scale_info = image_scales.get(image_path)
                    frame = {
                        "annotation": {
                            "filename": image_name,
                            "folder": filename_no_ext,
                            "size": {
                                "width": scale_info['new_size'][0],
                                "height": scale_info['new_size'][1],
                                "depth": 3
                            },
                            "object": [
                            ]
                        }
                    }
                    frames[frame_name] = frame
                
                frame = frames[frame_name]
                
                # Get scale information for coordinate transformation
                scale = scale_info['scale']
                
                # Scale the bounding box coordinates
                scaled_xmin = int(row["xpos"] * scale)
                scaled_ymin = int(row["ypos"] * scale)
                scaled_xmax = int((row["xpos"] + row["width"]) * scale)
                scaled_ymax = int((row["ypos"] + row["height"]) * scale)
                
                frame["annotation"]["object"].append({
                    "name": idx2label[row["class"]],
                    "pose": "Unspecified",
                    "truncated": row["truncation"],
                    "bndbox":{
                        "xmin": scaled_xmin,
                        "ymin": scaled_ymin,
                        "xmax": scaled_xmax,
                        "ymax": scaled_ymax
                    }
                })
            print(f'Processed annotation {ann_idx + 1}/{len(filenames)}: {filename}')

            
            # Iterate over video frames and save video annotations
            for _, key in enumerate(frames):
                # Build XML tree
                frame = frames[key]
                root_key = next(iter(frame))
                root = dict_to_xml(root_key, frame[root_key])
                tree = ET.ElementTree(root)
                ET.indent(tree, space="\t", level=0)

                # Write to file in video-specific folder
                video_annotations_dir = os.path.join(ANNOTATIONS_DIR, filename_no_ext)
                os.makedirs(video_annotations_dir, exist_ok=True)
                # Extract just the frame number from the key (e.g., "uav0000013_00000_v_0000001" -> "0000001")
                frame_number = key.split('_')[-1]
                file_location = os.path.join(video_annotations_dir, frame_number + '.xml')
                tree.write(file_location, encoding="utf-8", xml_declaration=False)      
            print(f'Converted annotation {ann_idx + 1}/{len(filenames)} to XML')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for VisDrone dataset converting to VOC-like format')
    parser.add_argument('--config', dest='config_path', default='config/vis-drone.yaml', type=str)
    parser.add_argument('--split', dest='split', default='train', type=str)
    args = parser.parse_args()
    convert_vis_to_voc_seq(args)        
