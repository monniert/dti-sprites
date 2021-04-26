from torch.nn import functional as F


class TensorResize():
    def __init__(self, img_size):
        self.img_size = img_size

    def __call__(self, img):
        # XXX interpolate first dim is a batch dim
        return F.interpolate(img.unsqueeze(0), self.img_size, mode='bilinear')[0]

    def __repr__(self):
        return self.__class__.__name__ + '()'


class TensorCenterCrop():
    def __init__(self, img_size):
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size

    def __call__(self, img):
        image_width, image_height = img.shape[-2:]
        height, width = self.img_size

        top = int((image_height - height + 1) * 0.5)
        left = int((image_width - width + 1) * 0.5)
        return img[..., top:top + height, left:left + width]

    def __repr__(self):
        return self.__class__.__name__ + '()'
