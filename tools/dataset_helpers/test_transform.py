import argparse
import glob

import yaml
from dataset.visdrone import VisDroneDataset
from dataset.voc import VOCDataset
from dataset.voc_small_objects import VOCSmallObjectsDataset
import torch
import matplotlib.pyplot as plt
import tqdm

from dataset.ytbb import YTBBDataset
from tools.roi_merger import area

IMG_DIR = "D:\\YouTube\\ytbb_dataset\\ResizedSequences\\AAB6lO-XiKE"

if not torch.cuda.is_available():
    raise Exception('CUDA not available')
else:
    print('Running on CUDA')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using mps')


def collate_function(data):
    return tuple(zip(*data))


def test_transform(args):
    def prep(img):
        img = img.cpu().detach()
        if img.ndim == 3:           # RGB or grayscale
            img = img.permute(1, 2, 0)
        return img.numpy()
    
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    #########################

    dataset_config = config['dataset_params']
    train_config = config['train_params']
    split = args.split
    im_sets = dataset_config['train_im_sets'] if split == 'train' else dataset_config['test_im_sets']
    transform_name = dataset_config['transform_name']
    # dataset = YTBBDataset('train', "D:\\Datasets\\YouTube\\ytbb_dataset", im_size=512)
    if str(train_config['dataset']) == 'vis-drone':
        dataset = VisDroneDataset('test',
                     im_sets=dataset_config['test_im_sets'],
                     im_size=dataset_config['im_size'])
    elif str(train_config['dataset']) == 'ytbb':
        dataset = YTBBDataset('test',
                     root_dir=dataset_config['root_dir'],
                     im_size=dataset_config['im_size'])
    elif str(train_config['dataset']) == 'voc':
        dataset = VOCDataset('test',
                     im_sets=dataset_config['test_im_sets'],
                     im_size=dataset_config['im_size'],
                     transform_name=dataset_config['transform_name'])
    elif str(train_config['dataset']) == 'voc-small-objects':
        dataset = VOCSmallObjectsDataset('test',
                     im_sets=dataset_config['test_im_sets'],
                     im_size=dataset_config['im_size'],
                     transform_name=dataset_config['transform_name'])
    else:
        raise Exception('Unknown dataset name {}'.format(train_config['dataset']))

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    image_paths = sorted(glob.glob(f"{IMG_DIR}/*.jpg"))

    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_function)

    for idx, (ims, targets, _) in enumerate(tqdm.tqdm(dataloader)):
    # for idx, img_path in enumerate(image_paths):

        # (_, _, _, _, _, _, transformed_im, transformed_targets) = dataset.get_image(img_path)
        
        # im = prep(im)


        transformed_targets = targets[0]
        transformed_im = prep(ims[0])

        h, w = ims[0].shape[-2:]
        wh_tensor = torch.as_tensor([[w, h, w, h]]).expand_as(transformed_targets['bboxes'])
        transformed_targets['bboxes'] = transformed_targets['bboxes'] * wh_tensor

        ax.clear()
        ax.imshow(transformed_im)
        ax.set_title('Transformed Image')
        ax.axis('off')
        bboxes = transformed_targets['bboxes'].cpu().numpy()
        print('original_boxes', bboxes)
        for box, label in zip(bboxes, transformed_targets['labels'].cpu().numpy()):
            xmin, ymin, xmax, ymax = box
            rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                fill=False, color='green', linewidth=2)
            ax.add_patch(rect)
            ax.text(xmin, ymin - 5, dataset.idx2label.get(label, str(label)),
                    color='green', fontsize=10, backgroundcolor='white')
        ax.text(transformed_im.shape[1] - 50, 15, str(len(bboxes)),
                    color='green', fontsize=10, backgroundcolor='white')
        ax.text(transformed_im.shape[1] - 50, 35, f'Area: {sum(area(box) for box in bboxes):.0f}',
                    color='green', fontsize=10, backgroundcolor='white')
        ax.text(transformed_im.shape[1] - 50, 55, f'Image area: {transformed_im.shape[1] * transformed_im.shape[0]:.0f}',
                    color='green', fontsize=10, backgroundcolor='white')
        plt.pause(4) 
      
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ssd training (testing transform)')
    parser.add_argument('--config', dest='config_path',
                        default='config/voc.yaml', type=str)
    parser.add_argument('--split', dest='split',
                        default='train', type=str)
    args = parser.parse_args()
    
    test_transform(args)
