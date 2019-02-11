from __future__ import print_function, division

import os

import numpy as np
import pandas as pd
from skimage import io
from torch.utils.data import Dataset


class AffectivNetDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None, downsample=None, delete_classes = None, check = True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with the 'IMAGE' folder.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_list = pd.read_csv(csv_file)
        self.image_list = np.array(self.image_list).reshape(len(self.image_list))
        self.root_dir = root_dir
        self.transform = transform
        self.downsample = downsample
        self.delete_classes = delete_classes
        self.target = []

        for i in range(len(self.image_list)):
            item = self.image_list[i].replace('E:/Databases/AffectivNet/ManAnnotated/', '').split()
            self.image_list[i] = item[0]
            self.target.append(int(item[1]))
        if self.delete_classes:
            self.image_list, self.target = self.deleter(self.image_list, self.target, self.delete_classes)

        print('before check: {}'.format(len(self.image_list)))
        if check:
            self.image_list = self.checker(self.image_list)
            print('after check: {}'.format(len(self.image_list)))

        if self.downsample:
            self.image_list, self.target = self.downsampling(self.image_list, self.target, self.downsample)
        print(np.unique(self.target))

    def checker(self, image_list):
        new_image_list = []
        for i in image_list:
            img_name = os.path.join(self.root_dir, i)
            if os.path.isfile(img_name):
                new_image_list.append(i)
        return np.array(new_image_list)

    def deleter(self, image_list, target, delete_classes):
        new_image_list = np.array([])
        new_target = []
        j = 0
        for i in np.unique(target):
            if i in delete_classes:
                continue
            one_class_image_list = image_list[target == i]
            new_target += [j] * len(one_class_image_list)
            j += 1
            new_image_list = np.append(new_image_list, one_class_image_list)
        return new_image_list, new_target

    def downsampling(self, image_list, target, downsample):
        new_image_list = np.array([])
        new_target = []
        add_num = downsample
        for i in np.unique(target):
            one_class_image_list = image_list[target == i]
            new_target += [i]*add_num
            new_image_list = np.append(new_image_list, np.random.choice(one_class_image_list, add_num))
        return new_image_list, new_target

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
