from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, models, transforms, utils
import torchvision.transforms.functional as TF

import numpy as np
import json
import pandas as pd
import pickle
import matplotlib.pyplot as plt
# import skimage
# import skimage.io
# import skimage.transform
from PIL import Image
import time
import os
from os.path import join, exists
import copy
import random
from collections import OrderedDict


class ImageFolderModified(Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.transform = transform
        self.idx2dir = []
        self.path_list = []
        for subdir in sorted(os.listdir(self.root_dir)):
            if not os.path.isfile(subdir):
                self.idx2dir.append(subdir)
        for class_idx, subdir in enumerate(self.idx2dir):
            class_dir = os.path.join(self.root_dir, subdir)
            for f in os.listdir(class_dir):
                if f[-4:] in ['.png', '.jpg', 'JPEG', 'jpeg']:
                    self.path_list.append(
                        [os.path.join(class_dir, f), class_idx])

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        img_path, class_idx = self.path_list[idx]
        image = Image.open(img_path)
        if not image.mode == 'RGB':
            image = image.convert('RGB')
        image = self.transform(image)
        sample = [image, class_idx, img_path]
        return sample


class ImagePredictDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.transform = transform
        self.idx2dir = []
        self.path_list = []
        for f in os.listdir(root_dir):
            if f[-4:] in ['.png', '.jpg', 'JPEG', 'jpeg']:
                self.path_list.append(os.path.join(root_dir, f))

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        img_path = self.path_list[idx]
        image = Image.open(img_path)
        if not image.mode == 'RGB':
            image = image.convert('RGB')
        image = self.transform(image)
        sample = [image, img_path]
        return sample
