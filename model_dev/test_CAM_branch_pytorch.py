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

from torch.nn import functional as F
from torchvision.models import Inception3

from inception_modified import InceptionSegmentation
from image_dataset import *

"""
This script is for running the line/pole model on the street view image 
test set to generate the Class Activation Maps (CAMs) for lines/poles.
It takes ~1 min to run on a Nvidia Tesla K80 GPU.
Note: if the script is run on CPU instead of GPU, please replace ".cpu().item()"
with ".item()". E.g., line 130.
"""

target = "line" # target object to identify: "line" or "pole"
pretrained_models = {
    "pole": 'deepGrid_DPNH2seg_pretrained.tar',
    "line": 'deepGrid_seg_pretrained.tar',
}
dataset_dirs = {
    "pole": "pole_image_dataset_demo",
    "line": "line_image_dataset_demo",
}

def determine_root_dir():
    """
    This function is used to locate the root dir back to the parent directory,
    i.e., "GridMapping" directory.
    """
    root_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    assert root_dir.strip('/')[-11:] == 'GridMapping'
    return root_dir

root_dir = determine_root_dir()

# Configuration
# directory for loading training/validation/test data
test_dir_list = [
    [
        join(root_dir, 'data', dataset_dirs[target], 'test', '0'), # negative samples
    ],
    [
        join(root_dir, 'data', dataset_dirs[target], 'test', '1'), # positive samples
    ]
]

# old model checkpoint for the entire model (main branch + CAM branch)
old_ckpt_path = join(root_dir, 'checkpoint', pretrained_models[target])
# save the derived CAMs into a pickle file
CAM_save_path = join(root_dir, 'results', 'CAM_test', 'CAMs_test_set_' + target + '.pickle')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_size = 640
batch_size = 1   # must be 1 for testing segmentation
threshold = 0.5  # threshold probability to identify am image as positive
level = 1

preview = False


def metrics(stats):
    """stats: {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN}
    return: must be a single number """
    precision = (stats['TP'] + 0.00001) * 1.0 / \
        (stats['TP'] + stats['FP'] + 0.00001)
    recall = (stats['TP'] + 0.00001) * 1.0 / \
        (stats['TP'] + stats['FN'] + 0.00001)
    # print('precision:%.4f recall:%.4f' % (precision, recall))
    return 2 * precision * recall / (precision + recall)


def test_model(model, dataloader, metrics, threshold):
    stats = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
    model.eval()
    CAM_list = []
    for inputs, labels, paths in tqdm(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(False):
            # CAM is a 1 x 35 x 35 activation map
            _, outputs, CAM = model(inputs, testing=True)
            prob = F.softmax(outputs, dim=1)
            preds = prob[:, 1] >= threshold

        # transform tensor into numpy array
        CAM = CAM.squeeze(0).cpu().numpy()
        for i in range(preds.size(0)):
            predicted_label = preds[i]
            label = predicted_label.cpu().item()
            if(preview):
                CAM_list.append((CAM, paths[i], label))
            else:
                if label:
                    # only use the generated CAM if it is predicted to be 1
                    CAM_list.append((CAM, paths[i], label))
                else:
                    # otherwise the CAM is a totally black one
                    CAM_list.append((np.zeros_like(CAM), paths[i], label))

        stats['TP'] += torch.sum((preds == 1) * (labels == 1)).cpu().item()
        stats['TN'] += torch.sum((preds == 0) * (labels == 0)).cpu().item()
        stats['FP'] += torch.sum((preds == 1) * (labels == 0)).cpu().item()
        stats['FN'] += torch.sum((preds == 0) * (labels == 1)).cpu().item()

    metric_value = metrics(stats)
    return stats, metric_value, CAM_list


transform_test = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

if __name__ == '__main__':
    # data
    dataset_test = FolderDirsDataset(test_dir_list, transform_test, return_path=True)
    dataloader_test = DataLoader(
        dataset_test, batch_size=batch_size, shuffle=False, num_workers=4)
    print('Test set size: ' + str(len(dataset_test)))
    # model
    model = InceptionSegmentation(num_outputs=2, level=level)
    model.load_existing_params(old_ckpt_path)

    model = model.to(device)

    stats, metric_value, CAM_list = test_model(
        model, dataloader_test, metrics, threshold=threshold)
    precision = (stats['TP'] + 0.00001) * 1.0 / \
        (stats['TP'] + stats['FP'] + 0.00001)
    recall = (stats['TP'] + 0.00001) * 1.0 / \
        (stats['TP'] + stats['FN'] + 0.00001)
    print('metric value: '+str(metric_value))
    print('precision: ' + str(round(precision, 4)))
    print('recall: ' + str(round(recall, 4)))

    with open(CAM_save_path, 'wb') as f:
        pickle.dump(CAM_list, f)
