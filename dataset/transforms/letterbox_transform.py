import torch
import torchvision.transforms.v2
from transformers.pad_square import PadToSquare
from transformers.resize_longer_edge import ResizeLongerEdge

class LetterboxTransform:
    def __init__(self, im_size, im_mean, imagenet_mean, imagenet_std):
        self.transforms = self._get_transforms(im_size, im_mean, imagenet_mean, imagenet_std)

    def _labels_getter(self, transform_input):
        """Helper function for SanitizeBoundingBoxes to extract labels and difficult flags."""
        return (transform_input[1]["labels"], transform_input[1]["difficult"])

    def _get_transforms(self, im_size, im_mean, imagenet_mean, imagenet_std):
        transforms = {
            'train': torchvision.transforms.v2.Compose([
                torchvision.transforms.v2.RandomPhotometricDistort(),
                torchvision.transforms.v2.RandomZoomOut(fill=im_mean),
                torchvision.transforms.v2.RandomIoUCrop(),
                torchvision.transforms.v2.RandomHorizontalFlip(p=0.5),
                ResizeLongerEdge(size=im_size),
                PadToSquare(size=im_size, fill=im_mean), # short side
                torchvision.transforms.v2.SanitizeBoundingBoxes(
                    labels_getter=self._labels_getter),
                torchvision.transforms.v2.ToPureTensor(),
                torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
                torchvision.transforms.v2.Normalize(mean=imagenet_mean,
                                                    std=imagenet_std)

            ]),
            'test': torchvision.transforms.v2.Compose([
                ResizeLongerEdge(size=im_size),
                PadToSquare(size=im_size, fill=im_mean), # short side
                torchvision.transforms.v2.ToPureTensor(),
                torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
                torchvision.transforms.v2.Normalize(mean=imagenet_mean,
                                                    std=imagenet_std)
            ]),
        }
        return transforms