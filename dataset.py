import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms

from interpolate import interpolate

def resize_transform(size, factor):
    new_size = (size[0] // factor, size[1] // factor)
    return transforms.Resize(new_size)

class ImageDataset(data.Dataset):
    def __init__(self, img_paths, img_size, scale_factor, transform=None):
        self.img_paths = img_paths
        self.img_size = img_size
        self.scale_factor = scale_factor
        self.transform = transform
    
    def __getitem__(self, idx):
        img = Image.open(str(self.img_paths[idx]))
        label = img
        img = interpolate(np.array(resize_transform(self.img_size, self.scale_factor)(img)), self.scale_factor)

        if self.transform:
            img = self.transform(img)
            label = self.transform(label)

        return img, label

    def __len__(self):
        return len(self.img_paths)
