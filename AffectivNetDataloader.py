from __future__ import print_function, division
import os
import torch
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd


class AffectivNetDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with the 'IMAGE' folder.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_list = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.pics_list[idx])
        target = 0 if 'cat' in self.pics_list[idx] else 1
        image = io.imread(img_name)
        if self.transform:
            image = self.transform(image)
        sample = {'image': image, 'target': target}

        return sample