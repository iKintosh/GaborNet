import os
from random import choice
from PIL import Image
import numpy as np


class DatasetLoader():
    def __init__(self, source_path='D:\!EMONEW_cropped', classes=7, train=0.66):
        self.source_path = os.path.normpath(source_path)
        self.data = []
        self.test_data = []
        self.num_classes = classes
        self.train = train

        for path in os.listdir(path=source_path):
            source_path = os.path.join(self.source_path, path)
            if os.path.isdir(source_path):
                classes = classes - 1
                target = [0]*self.num_classes
                target[classes] = 1
                self._list_to_jpeg(source_path, target)

    def _list_to_jpeg(self, path, target):
        for i in os.walk(path):
            for j in i[2]:
                if '.jpg' in j or '.png' in j:
                    if np.random.rand() <= self.train:
                        self.data.append([[os.path.join(i[0], j)], target])
                    else:
                        self.test_data.append([[os.path.join(i[0], j)], target])

    def get_batch(self, len_batch, train=True, keras_style=False):
        batch_data = []
        batch_target = []
        if keras_style:
            transpose = (0, 1, 2)
        else:
            transpose = (2, 0, 1)

        for i in range(len_batch):
            if train:
                d, t = choice(self.data)
            else:
                d, t = choice(self.test_data)
            batch_target.append(t)
            batch_data.append(np.asarray(Image.open(d[0])).transpose(transpose))
        return np.array(batch_data), np.array(batch_target)

    def lenght(self, train=True):
        if train:
            return len(self.data)
        else:
            return len(self.test_data)
