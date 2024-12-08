import numpy as np
from pathlib import Path
from PIL import Image
import torch
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
        label = np.array(img, dtype=np.double).transpose((2, 0, 1))/256 - 0.5
        img = interpolate(np.array(resize_transform(self.img_size, self.scale_factor)(img)), self.scale_factor)

        if self.transform:
            img = self.transform(img)
            label = self.transform(label)
        
        img = np.double(img.transpose((2, 0, 1)))/256 - 0.5

        return img, label

    def get_order(self, idx):
        img_path_parts = self.img_paths[idx].parts
        return Path(img_path_parts[-2]) / img_path_parts[-1]

    def __len__(self):
        return len(self.img_paths)


class FeatureDataset(data.Dataset):
    def __init__(self, img_paths, img_size, net, device, transform=None):
        net = net.to(device)

        self.minimum_feature = 100
        self.maximum_feature = 0
        self.features = []
        for img_path in img_paths:
            img = Image.open(str(img_path))
            if transform:
                img = transform(img)
            img = img.view(1, 3, img_size, img_size)  # reshape image to match model dimensions (1 batch size)
            img = img.to(device)

            outputs = net(img)["avgpool"].cpu().detach().numpy().squeeze()
            self.minimum_feature = min(self.minimum_feature, outputs.min())
            self.maximum_feature = max(self.maximum_feature, outputs.min())
            self.features.append(torch.Tensor(outputs))

    def __getitem__(self, idx):
        return ((self.features[idx] - self.minimum_feature)/(self.maximum_feature - self.minimum_feature)*256).to(torch.uint8)

    def __len__(self):
        return len(self.features)
