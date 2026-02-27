import torch
import torchvision.transforms.v2
from transformers.fixed_roi_crop import FixedROICrop
from transformers.resize_longer_edge import ResizeLongerEdge

class FixedPaddingRoiCropTestTransform:
    def __init__(self, im_size, imagenet_mean, imagenet_std, pad_x, pad_y):
        self.transforms = self._get_transforms(im_size, imagenet_mean, imagenet_std, pad_x, pad_y)

    def _labels_getter(self, transform_input):
        """Helper function for SanitizeBoundingBoxes to extract labels and difficult flags."""
        return (transform_input[1]["labels"], transform_input[1]["difficult"])

    def _get_transforms(self, im_size, imagenet_mean, imagenet_std, pad_x, pad_y):
        transforms = {
            'test': torchvision.transforms.v2.Compose([
                ResizeLongerEdge(size=im_size),
                FixedROICrop(pad_x=pad_x, pad_y=pad_y, min_box_area=4.0),
                torchvision.transforms.v2.ToPureTensor(),
                torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
                torchvision.transforms.v2.Normalize(mean=imagenet_mean,
                                                    std=imagenet_std)
            ]),
        }
        return transforms