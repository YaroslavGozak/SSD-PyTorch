import os
import shutil
import random
from pathlib import Path
from typing import Optional, Tuple, Dict
from PIL import Image
import xml.etree.ElementTree as ET
from xml.dom import minidom


# =========================
# Configuration
# =========================

SOURCE_DIR = r"H:\Projects\UnrealEngine\Flying 5.3\Saved\MovieRenders\5"
OUTPUT_DIR = r"D:\VOC\VOCUE_test5"

CLASS_NAME = "car"

# File extensions to scan
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Keywords that identify mask files
MASK_KEYWORDS = ["mask", "stencil", "bw", "binary"]

# Optional explicit suffixes to strip when matching pairs
# Example:
# frame_000001_mask.png -> frame_000001
# frame_000001_rgb.png  -> frame_000001
MASK_SUFFIXES = ["_mask", "_stencil", "_bw", "_binary"]
RGB_SUFFIXES = ["_rgb", "_color", "_render", "_beauty"]

# Threshold for mask binarization
# Any grayscale value > MASK_THRESHOLD is treated as object
MASK_THRESHOLD = 10

# Dataset split
TRAIN_RATIO = 0.9
RANDOM_SEED = 42

# If True, output copied images as .jpg
# If False, keep original extension
CONVERT_TO_JPG = True

# Skip frames where mask is empty
SKIP_EMPTY_MASKS = True


# =========================
# Utility functions
# =========================

def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def normalize_stem(stem: str, suffixes: list[str]) -> str:
    result = stem
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if result.lower().endswith(suffix.lower()):
                result = result[:-len(suffix)]
                changed = True
    return result


def looks_like_mask(path: Path) -> bool:
    stem_lower = path.stem.lower()
    return any(keyword in stem_lower for keyword in MASK_KEYWORDS)


def pretty_xml(element: ET.Element) -> str:
    rough = ET.tostring(element, encoding="utf-8")
    reparsed = minidom.parseString(rough)
    return reparsed.toprettyxml(indent="  ")


def compute_bbox_from_mask(mask_path: Path, threshold: int) -> Optional[Tuple[int, int, int, int, int, int]]:
    """
    Returns:
        xmin, ymin, xmax, ymax, width, height
    Coordinates are Pascal VOC style, 1-based inclusive.
    """
    img = Image.open(mask_path).convert("L")
    width, height = img.size
    pixels = img.load()

    min_x = width
    min_y = height
    max_x = -1
    max_y = -1

    for y in range(height):
        for x in range(width):
            if pixels[x, y] > threshold:
                if x < min_x:
                    min_x = x
                if y < min_y:
                    min_y = y
                if x > max_x:
                    max_x = x
                if y > max_y:
                    max_y = y

    if max_x == -1 or max_y == -1:
        return None

    # Convert to 1-based coordinates for VOC
    xmin = min_x + 1
    ymin = min_y + 1
    xmax = max_x + 1
    ymax = max_y + 1

    return xmin, ymin, xmax, ymax, width, height


def save_voc_xml(
    xml_path: Path,
    folder_name: str,
    filename: str,
    full_image_path: Path,
    width: int,
    height: int,
    depth: int,
    class_name: str,
    bbox: Tuple[int, int, int, int],
) -> None:
    xmin, ymin, xmax, ymax = bbox

    annotation = ET.Element("annotation")

    folder = ET.SubElement(annotation, "folder")
    folder.text = folder_name

    fname = ET.SubElement(annotation, "filename")
    fname.text = filename

    path_el = ET.SubElement(annotation, "path")
    path_el.text = str(full_image_path)

    source = ET.SubElement(annotation, "source")
    database = ET.SubElement(source, "database")
    database.text = "UnrealEngineGenerated"

    size = ET.SubElement(annotation, "size")
    width_el = ET.SubElement(size, "width")
    width_el.text = str(width)
    height_el = ET.SubElement(size, "height")
    height_el.text = str(height)
    depth_el = ET.SubElement(size, "depth")
    depth_el.text = str(depth)

    segmented = ET.SubElement(annotation, "segmented")
    segmented.text = "0"

    obj = ET.SubElement(annotation, "object")

    name = ET.SubElement(obj, "name")
    name.text = class_name

    pose = ET.SubElement(obj, "pose")
    pose.text = "Unspecified"

    truncated = ET.SubElement(obj, "truncated")
    truncated.text = "0"

    difficult = ET.SubElement(obj, "difficult")
    difficult.text = "0"

    bndbox = ET.SubElement(obj, "bndbox")
    xmin_el = ET.SubElement(bndbox, "xmin")
    xmin_el.text = str(xmin)
    ymin_el = ET.SubElement(bndbox, "ymin")
    ymin_el.text = str(ymin)
    xmax_el = ET.SubElement(bndbox, "xmax")
    xmax_el.text = str(xmax)
    ymax_el = ET.SubElement(bndbox, "ymax")
    ymax_el.text = str(ymax)

    xml_content = pretty_xml(annotation)
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(xml_content)


def ensure_dirs(root: Path) -> Dict[str, Path]:
    jpeg_images = root / "JPEGImages"
    annotations = root / "Annotations"
    image_sets_main = root / "ImageSets" / "Main"

    jpeg_images.mkdir(parents=True, exist_ok=True)
    annotations.mkdir(parents=True, exist_ok=True)
    image_sets_main.mkdir(parents=True, exist_ok=True)

    return {
        "root": root,
        "jpeg_images": jpeg_images,
        "annotations": annotations,
        "image_sets_main": image_sets_main,
    }


def copy_or_convert_image(src: Path, dst_without_ext: Path) -> Tuple[Path, int]:
    """
    Returns:
        output_image_path, depth
    """
    img = Image.open(src)
    mode = img.mode

    if mode == "L":
        depth = 1
    elif mode in ("RGB", "P"):
        depth = 3
        if mode != "RGB":
            img = img.convert("RGB")
    elif mode == "RGBA":
        depth = 3
        img = img.convert("RGB")
    else:
        img = img.convert("RGB")
        depth = 3

    if CONVERT_TO_JPG:
        out_path = dst_without_ext.with_suffix(".jpg")
        img.save(out_path, quality=95)
    else:
        out_path = dst_without_ext.with_suffix(src.suffix.lower())
        if img is Image.open(src):
            shutil.copy2(src, out_path)
        else:
            img.save(out_path)

    return out_path, depth


# =========================
# Pair matching
# =========================

def build_file_index(files: list[Path]) -> Tuple[dict[str, Path], dict[str, Path]]:
    mask_index = {}
    rgb_index = {}

    for path in files:
        stem = path.stem

        if looks_like_mask(path):
            key = normalize_stem(stem, MASK_SUFFIXES)
            mask_index[key] = path
        else:
            key = normalize_stem(stem, RGB_SUFFIXES)
            rgb_index[key] = path

    return mask_index, rgb_index


# =========================
# Main
# =========================

def main() -> None:
    source_dir = Path(SOURCE_DIR)
    output_dir = Path(OUTPUT_DIR)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    dirs = ensure_dirs(output_dir)

    all_files = [p for p in source_dir.iterdir() if p.is_file() and is_image_file(p)]
    if not all_files:
        raise RuntimeError("No image files found in source directory.")

    mask_index, rgb_index = build_file_index(all_files)

    common_keys = sorted(set(mask_index.keys()) & set(rgb_index.keys()))
    only_masks = sorted(set(mask_index.keys()) - set(rgb_index.keys()))
    only_rgbs = sorted(set(rgb_index.keys()) - set(mask_index.keys()))

    print(f"Found RGB files:   {len(rgb_index)}")
    print(f"Found mask files:  {len(mask_index)}")
    print(f"Matched pairs:     {len(common_keys)}")
    print(f"Unmatched masks:   {len(only_masks)}")
    print(f"Unmatched RGBs:    {len(only_rgbs)}")

    if not common_keys:
        raise RuntimeError(
            "No RGB/mask pairs found. Check MASK_KEYWORDS, MASK_SUFFIXES, and RGB_SUFFIXES."
        )

    processed_ids = []
    skipped_empty = 0

    for key in common_keys:
        print(f"Processing pair: {key}")
        rgb_path = rgb_index[key]
        mask_path = mask_index[key]

        bbox_info = compute_bbox_from_mask(mask_path, MASK_THRESHOLD)
        if bbox_info is None:
            if SKIP_EMPTY_MASKS:
                skipped_empty += 1
                print(f"Skipping empty mask: {mask_path.name}")
                continue
            else:
                print(f"Warning: empty mask, but keeping frame without object is not supported in VOC object annotation: {mask_path.name}")
                continue

        xmin, ymin, xmax, ymax, width, height = bbox_info

        image_id = key
        dst_image_stem = dirs["jpeg_images"] / image_id
        copied_image_path, depth = copy_or_convert_image(rgb_path, dst_image_stem)

        xml_path = dirs["annotations"] / f"{image_id}.xml"
        save_voc_xml(
            xml_path=xml_path,
            folder_name="JPEGImages",
            filename=copied_image_path.name,
            full_image_path=copied_image_path.resolve(),
            width=width,
            height=height,
            depth=depth,
            class_name=CLASS_NAME,
            bbox=(xmin, ymin, xmax, ymax),
        )

        processed_ids.append(image_id)

    if not processed_ids:
        raise RuntimeError("No valid annotated samples were created.")

    random.seed(RANDOM_SEED)
    random.shuffle(processed_ids)

    split_idx = int(len(processed_ids) * TRAIN_RATIO)
    train_ids = sorted(processed_ids[:split_idx])
    val_ids = sorted(processed_ids[split_idx:])
    trainval_ids = sorted(processed_ids)

    with open(dirs["image_sets_main"] / "train.txt", "w", encoding="utf-8") as f:
        for item in train_ids:
            f.write(item + "\n")

    with open(dirs["image_sets_main"] / "val.txt", "w", encoding="utf-8") as f:
        for item in val_ids:
            f.write(item + "\n")

    with open(dirs["image_sets_main"] / "test.txt", "w", encoding="utf-8") as f:
        for item in trainval_ids:
            f.write(item + "\n")

    print()
    print("Done.")
    print(f"Created dataset at: {output_dir}")
    print(f"Annotated samples:   {len(processed_ids)}")
    print(f"Skipped empty masks: {skipped_empty}")


if __name__ == "__main__":
    main()