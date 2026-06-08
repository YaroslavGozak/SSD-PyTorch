import argparse
import glob

import torch
import matplotlib.pyplot as plt
import tqdm

from tools.helpers.config_reader import load_config
from tools.helpers.pipeline import load_dataset

IMG_DIR = "D:\\YouTube\\ytbb_dataset\\ResizedSequences\\AAB6lO-XiKE"

if not torch.cuda.is_available():
    print('Running on CPU')
    # raise Exception('CUDA not available')
else:
    print('Running on CUDA')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def collate_function(data):
    return tuple(zip(*data))


def area(box):
    xmin, ymin, xmax, ymax = box
    return max(0.0, xmax - xmin) * max(0.0, ymax - ymin)


def test_transform(args):
    def prep(img):
        img = img.cpu().detach()
        if img.ndim == 3:           # RGB or grayscale
            img = img.permute(1, 2, 0)
        return img.numpy()
    
    # Read the config file #
    config = load_config(args.config_path)
    #########################

    dataset_config = config['dataset_params']
    split = args.split
    transform_name = args.transform or dataset_config['transform_name']
    dataset = load_dataset(config, split=split, transform_name=transform_name)

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
                        default='config/imagenet-vid-roissd.yaml', type=str)
    parser.add_argument('--split', dest='split',
                        default='train', type=str)
    parser.add_argument('--transform', dest='transform', default=None, type=str,
                        help='Override transform_name from config. '
                             'E.g. fixed_size_96x128  or  fixed_size_yolo_96x128')
    args = parser.parse_args()
    
    test_transform(args)
