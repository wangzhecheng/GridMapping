from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, models, transforms, utils
import torchvision.transforms.functional as TF

from tqdm import tqdm
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
from sklearn.metrics import r2_score


import torch.nn.functional as F
from torchvision.models import Inception3
from collections import namedtuple

_InceptionOuputs = namedtuple('InceptionOuputs', ['logits', 'aux_logits'])


class InceptionSegmentation(nn.Module):
    def __init__(self, num_outputs=2, level=1):
        super(InceptionSegmentation, self).__init__()
        assert level in [1, 2]
        self.level = level
        self.inception3 = Inception3_modified(num_classes=num_outputs, aux_logits=False, transform_input=False)
        self.convolution1 = nn.Conv2d(288, 512, bias=True, kernel_size=3, padding=1)
        if self.level == 1:
            self.linear1 = nn.Linear(512, num_outputs, bias=False)
        else:
            self.convolution2 = nn.Conv2d(512, 512, bias=True, kernel_size=3, padding=1)
            self.linear2 = nn.Linear(512, num_outputs, bias=False)

    def forward(self, x, testing=False, classify_only=False):
        logits, intermediate = self.inception3(x)
        if classify_only:
            return logits
        feature_map = self.convolution1(intermediate)  # N x 512 x 35 x 35
        feature_map = F.relu(feature_map)          # N x 512 x 35 x 35
        if self.level == 1:
            y = F.adaptive_avg_pool2d(feature_map, (1, 1))
            y = y.view(y.size(0), -1)    # N x 512
            y = self.linear1(y)          # N x 2
            if testing:
                CAM = self.linear1.weight.data[1, :] * feature_map.permute(0, 2, 3, 1)
                CAM = CAM.sum(dim=3)
        else:
            feature_map = self.convolution2(feature_map)     # N x 512 x 35 x 35
            feature_map = F.relu(feature_map)  # N x 512 x 35 x 35
            y = F.adaptive_avg_pool2d(feature_map, (1, 1))
            y = y.view(y.size(0), -1)    # N x 512
            y = self.linear2(y)          # N x 2
            if testing:
                CAM = self.linear2.weight.data[1, :] * feature_map.permute(0, 2, 3, 1)
                CAM = CAM.sum(dim=3)
        if testing:
            return y, logits, CAM # y: prediction of segmentation branch, logits: prediction of main branch, CAM: class activation map
        else:
            return y

    def load_basic_params(self, model_path, device=torch.device('cpu')):
        """Only load the parameters from main branch."""
        old_params = torch.load(model_path, map_location=device)
        if model_path[-4:] == '.tar':  # The file is not a model state dict, but a checkpoint dict
            old_params = old_params['model_state_dict']
        self.inception3.load_state_dict(old_params, strict=False)
        print('Loaded basic model parameters from: ' + model_path)

    def load_existing_params(self, model_path, device=torch.device('cpu')):
        """Load the parameters of main branch and parameters of level-1 layers (and perhaps level-2 layers.)"""
        old_params = torch.load(model_path, map_location=device)
        if model_path[-4:] == '.tar':  # The file is not a model state dict, but a checkpoint dict
            old_params = old_params['model_state_dict']
        self.load_state_dict(old_params, strict=False)
        print('Loaded existing model parameters from: ' + model_path)


class Inception3_modified(Inception3):
    def forward(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        intermediate = x.clone()
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        intermediate = x.clone()
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        if self.training and self.aux_logits:
            aux = self.AuxLogits(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 2048 x 1 x 1
        x = F.dropout(x, training=self.training)
        # N x 2048 x 1 x 1
        x = x.view(x.size(0), -1)
        # N x 2048
        x = self.fc(x)
        # N x 1000 (num_classes)
        if self.training and self.aux_logits:
            return _InceptionOuputs(x, aux)
        return x, intermediate
