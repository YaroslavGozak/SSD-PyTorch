import torchvision

class ResizeLongerEdge(torchvision.transforms.v2.Transform):
    def __init__(self, size):
        super().__init__()
        self.size = size
    
    def forward(self, img, target):
        _, h, w = img.shape
        if h > w:
            new_h = self.size
            new_w = int(w * self.size / h)
        else:
            new_w = self.size
            new_h = int(h * self.size / w)
        
        resize_transform = torchvision.transforms.v2.Resize(size=(new_h, new_w), antialias=True)
        return resize_transform(img, target)
