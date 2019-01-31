from __future__ import print_function, division

import os

import numpy as np
import pandas as pd
from skimage import io
from torch.utils.data import Dataset


class AffectivNetDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with the 'IMAGE' folder.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_list = pd.read_csv(csv_file)
        self.image_list = np.array(self.image_list).reshape(len(self.image_list))
        self.target = []
        for i in range(len(self.image_list)):
            item = self.image_list[i].replace('E:/Databases/AffectivNet/ManAnnotated/', '').split()
            self.image_list[i] = item[0]
            self.target.append(int(item[1]))
        print(np.unique(self.target))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        item_path = self.image_list[idx]
        target = self.target[idx]
        img_name = os.path.join(self.root_dir, item_path)
        image = io.imread(img_name)
        if self.transform:
            image = self.transform(image)
        sample = {'image': image, 'target': target}

        return sample
