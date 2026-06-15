import torch
from dataset.testtransform_dataset import TestTransformDataset
import matplotlib.pyplot as plt

from tools.mergers.merger_helper import area

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


def test_transform():
    def prep(img):
        img = img.cpu().detach()
        if img.ndim == 3:           # RGB or grayscale
            img = img.permute(1, 2, 0)
        return img.numpy()
    
    dataset = TestTransformDataset(512)

    (im, target, transformed_im, simple_targets, simple_v2_targets, greedy_targets) = dataset.get_image("D:\\VisDrone\\VisDrone2019-VID-train\\VisDrone2019-VID-train\\ResizedSequences\\uav0000266_04830_v\\0000030.jpg")
    
    im = prep(im)
    transformed_im = prep(transformed_im)
    fig, ax = plt.subplots(2, 2, figsize=(12, 6))
    ax[0,0].imshow(im)
    ax[0,0].set_title('Original Image')
    ax[0,0].axis('off')
    bboxes = target['bboxes'].cpu().numpy()
    print('original_boxes', bboxes)
    for box, label in zip(bboxes, target['labels'].cpu().numpy()):
        xmin, ymin, xmax, ymax = box
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                             fill=False, color='green', linewidth=2)
        ax[0,0].add_patch(rect)
        ax[0,0].text(xmin, ymin - 5, dataset.idx2label.get(label, str(label)),
                   color='green', fontsize=10, backgroundcolor='white')
    ax[0,0].text(im.shape[1] - 50, 15, str(len(bboxes)),
                   color='green', fontsize=10, backgroundcolor='white')
    ax[0,0].text(im.shape[1] - 50, 35, f'Area: {sum(area(box) for box in bboxes):.0f}',
                   color='green', fontsize=10, backgroundcolor='white')
    ax[0,0].text(im.shape[1] - 50, 55, f'Image area: {im.shape[1] * im.shape[0]:.0f}',
                   color='green', fontsize=10, backgroundcolor='white')

    ax[1,0].imshow(transformed_im)
    ax[1,0].set_title('Simple v2 merge Image')
    ax[1,0].axis('off')
    transformed_bboxes = simple_v2_targets['bboxes'].cpu().numpy()
    print('transformed_boxes', transformed_bboxes)
    for box in transformed_bboxes:
        xmin, ymin, xmax, ymax = box
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                             fill=False, color='red', linewidth=2)
        ax[1,0].add_patch(rect)
    ax[1,0].text(im.shape[1] - 50, 15, str(len(transformed_bboxes)),
                   color='green', fontsize=10, backgroundcolor='white')
    ax[1,0].text(im.shape[1] - 50, 35, f'Area: {sum(area(box) for box in transformed_bboxes):.0f}',
                   color='green', fontsize=10, backgroundcolor='white')
        
    ax[1,1].imshow(im)
    ax[1,1].set_title('Simple merge Image')
    ax[1,1].axis('off')
    simple_bboxes = simple_targets['bboxes'].cpu().numpy()
    print('simple_boxes', simple_bboxes)
    for box in simple_bboxes:
        xmin, ymin, xmax, ymax = box
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                             fill=False, color='red', linewidth=2)
        ax[1,1].add_patch(rect)
    ax[1,1].text(im.shape[1] - 50, 15, str(len(simple_bboxes)),
                   color='green', fontsize=10, backgroundcolor='white')
    ax[1,1].text(im.shape[1] - 50, 35, f'Area: {sum(area(box) for box in simple_bboxes):.0f}',
                   color='green', fontsize=10, backgroundcolor='white')
        
    ax[0,1].imshow(im)
    ax[0,1].set_title('Greedy merge Image')
    ax[0,1].axis('off')
    greedy_bboxes = greedy_targets['bboxes'].cpu().numpy()
    print('greedy_boxes', greedy_bboxes)
    for box in greedy_bboxes:
        xmin, ymin, xmax, ymax = box
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                             fill=False, color='red', linewidth=2)
        ax[0,1].add_patch(rect)
    ax[0,1].text(im.shape[1] - 50, 15, str(len(greedy_bboxes)),
                   color='green', fontsize=10, backgroundcolor='white')
    ax[0,1].text(im.shape[1] - 50, 35, f'Area: {sum(area(box) for box in greedy_bboxes):.0f}',
                   color='green', fontsize=10, backgroundcolor='white')
        
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    test_transform()
