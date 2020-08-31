from __future__ import print_function, division

import os

from skimage import io
from torch.utils.data import Dataset


class DogsCatsDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.pics_list = os.listdir(self.root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.pics_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.pics_list[idx])
        target = 0 if "cat" in self.pics_list[idx] else 1
        image = io.imread(img_name)
        if self.transform:
            image = self.transform(image)
        sample = {"image": image, "target": target}

        return sample
