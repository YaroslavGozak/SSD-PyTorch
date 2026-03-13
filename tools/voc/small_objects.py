
import argparse
import os
import xml.etree.ElementTree as ET

import yaml


def find_small_object_images(im_sets, split):
    r"""
    For each image set, iterate over all images listed in the split's txt file
    and collect those where every annotated object is no more than half the
    image width and half the image height.

    :param im_sets: List of image-set root directories.
    :param split: Dataset split name (e.g. 'train', 'test'). Used both to
                  locate the annotation list file and to name the output file.
    :return: dict mapping each im_set path to a list of matching image names.
    """
    fname = 'trainval' if split == 'train' else split
    results = {}

    for im_set in im_sets:
        ann_list_path = os.path.join(im_set, 'ImageSets', 'Main', '{}.txt'.format(fname))
        ann_dir = os.path.join(im_set, 'Annotations')

        with open(ann_list_path, 'r') as f:
            im_names = [line.strip() for line in f if line.strip()]

        small_object_images = []
        for im_name in im_names:
            ann_file = os.path.join(ann_dir, '{}.xml'.format(im_name))
            root = ET.parse(ann_file).getroot()

            size = root.find('size')
            im_w = int(size.find('width').text)
            im_h = int(size.find('height').text)

            objects = root.findall('object')
            # Skip images with no annotations
            if not objects:
                continue

            all_small = all(
                (int(obj.find('bndbox').find('xmax').text) -
                 int(obj.find('bndbox').find('xmin').text)) <= im_w / 2
                and
                (int(obj.find('bndbox').find('ymax').text) -
                 int(obj.find('bndbox').find('ymin').text)) <= im_h / 2
                for obj in objects
            )

            if all_small:
                small_object_images.append(im_name)

        results[im_set] = small_object_images
        print('Image set "{}": {}/{} images have all objects within half the image dimensions'.format(
            im_set, len(small_object_images), len(im_names)))

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for searching images with small objects')
    parser.add_argument('--config', dest='config_path',
                        default='config/voc.yaml', type=str)
    args = parser.parse_args()
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    ########################

    dataset_config = config['dataset_params']

    splits = {
        'train': dataset_config['train_im_sets'],
        'test': dataset_config['test_im_sets'],
    }

    for split, im_sets in splits.items():
        print('--- Split: {} ---'.format(split))
        results = find_small_object_images(im_sets, split)

        for im_set, im_names in results.items():
            out_path = os.path.join(im_set, 'ImageSets', 'Main',
                                    'small_objects_{}.txt'.format(split))
            with open(out_path, 'w') as f:
                f.write('\n'.join(im_names))
            print('Written {} image indexes to "{}"'.format(len(im_names), out_path))