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
from image_dataset import *

"""
This script is for running the line/pole identification model on the 
street view image test set and reporting the image-level metrics including 
precision and recall.
It takes ~1 min to run on a Nvidia Tesla K80 GPU.
Note: if the script is run on CPU instead of GPU, please replace ".cpu().item()"
with ".item()". E.g., line 107.
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

old_ckpt_path = join(root_dir, 'checkpoint', pretrained_models[target]) # path to model's old checkpoint

initialize_from_seg_model = True # if the model is initialized/loaded from a segmentation model (i.e. an instance of InceptionSegmentation), 
# then the layer names in model_state_dict should be renamed. 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_size = 640
batch_size = 16
given_threshold = 0.5  # threshold probability to identify am image as positive
threshold_list = np.linspace(0.0, 1.0, 101).tolist() + [given_threshold]

def metrics(stats):
    """stats: {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN}
    return: must be a single number """
    precision = (stats['TP'] + 0.00001) * 1.0 / \
        (stats['TP'] + stats['FP'] + 0.00001)
    recall = (stats['TP'] + 0.00001) * 1.0 / \
        (stats['TP'] + stats['FN'] + 0.00001)
    return 2 * precision * recall / (precision + recall)


def test_model(model, dataloader, metrics, threshold_list):
#     stats = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
    stats = {x: {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0} for x in threshold_list}
    model.eval()
    for inputs, labels in tqdm(dataloader, ascii=True):
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            prob = F.softmax(outputs, dim=1)
            for threshold in threshold_list:
                preds = prob[:, 1] >= threshold
                stats[threshold]['TP'] += torch.sum((preds == 1) * (labels == 1)).cpu().item()
                stats[threshold]['TN'] += torch.sum((preds == 0) * (labels == 0)).cpu().item()
                stats[threshold]['FP'] += torch.sum((preds == 1) * (labels == 0)).cpu().item()
                stats[threshold]['FN'] += torch.sum((preds == 0) * (labels == 1)).cpu().item()
    
    best_threshold = 0.0
    max_metrics = 0.0
    for threshold in threshold_list:
        metric_value = metrics(stats[threshold])
        if metric_value > max_metrics:
            best_threshold = threshold
            max_metrics = metric_value
#     metric_value = metrics(stats)
    return stats, max_metrics, best_threshold


transform_test = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

if __name__ == '__main__':
    # data
    dataset_test = FolderDirsDataset(test_dir_list, transform_test)
    dataloader_test = DataLoader(
        dataset_test, batch_size=batch_size, shuffle=True, num_workers=4)
    print('Test set size: ' + str(len(dataset_test)))
    # model
    model = Inception3(num_classes=2, aux_logits=True, transform_input=False)
    model = model.to(device)
    # load old parameters
    checkpoint = torch.load(old_ckpt_path, map_location=device)
    # it is a checkpoint dictionary rather than just model parameters
    if old_ckpt_path[-4:] == '.tar':
        if not initialize_from_seg_model:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model_state_dict_main_branch = OrderedDict()
            for k in checkpoint['model_state_dict'].keys():
                if k[:11] == 'inception3.':
                    model_state_dict_main_branch[k[11:]] = checkpoint['model_state_dict'][k]
            model.load_state_dict(model_state_dict_main_branch, strict=False)
    else:
        model.load_state_dict(checkpoint)
    print('Old checkpoint loaded: ' + old_ckpt_path)
    stats, best_metric_value, best_threshold = test_model(
        model, dataloader_test, metrics, threshold_list=threshold_list)
    precision = (stats[best_threshold]['TP'] + 0.00001) * 1.0 / \
        (stats[best_threshold]['TP'] + stats[best_threshold]['FP'] + 0.00001)
    recall = (stats[best_threshold]['TP'] + 0.00001) * 1.0 / \
        (stats[best_threshold]['TP'] + stats[best_threshold]['FN'] + 0.00001)
    for thres in threshold_list:
        prec = (stats[thres]['TP'] + 0.00001) * 1.0 / \
            (stats[thres]['TP'] + stats[thres]['FP'] + 0.00001)
        rec = (stats[thres]['TP'] + 0.00001) * 1.0 / \
            (stats[thres]['TP'] + stats[thres]['FN'] + 0.00001)
        print(thres, 'precision: ', prec, 'recall: ', rec, 'metrics: ', metrics(stats[thres]))
    print('best threshold: '+str(best_threshold))
    print('best metric value: '+str(best_metric_value))
    print('precision: ' + str(round(precision, 4)))
    print('recall: ' + str(round(recall, 4)))
    print('metric value under given threshold ' +str(given_threshold) + ': '+str(metrics(stats[given_threshold])))
    precision_given = (stats[given_threshold]['TP'] + 0.00001) * 1.0 / \
        (stats[given_threshold]['TP'] + stats[given_threshold]['FP'] + 0.00001)
    recall_given = (stats[given_threshold]['TP'] + 0.00001) * 1.0 / \
        (stats[given_threshold]['TP'] + stats[given_threshold]['FN'] + 0.00001)
    print('precision under given threshold: ' + str(round(precision_given, 4)))
    print('recall under given threshold: ' + str(round(recall_given, 4)))
