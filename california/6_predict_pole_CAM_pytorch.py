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
from image_dataset import ImagePredictDataset

"""
This script is for running the pole detetor on street view images.
Classification results and CAMs are generated and saved.
Note: this must be run after running step 2: "2_download_area_GSV.py".
"""

region = 'Salinas'
pole_model = 'ori0.5' # the model using 0.5 as the classification decision threshold

def determine_root_dir():
    """
    This function is used to locate the root dir back to the parent directory,
    i.e., "GridMapping" directory.
    """
    root_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    assert root_dir.strip('/')[-11:] == 'GridMapping'
    return root_dir

root_dir = determine_root_dir()
# change the root dir to "GridMapping"
os.chdir(root_dir)

pole_model_configs = {
    'ori0.5': {
            'ckpt_path': 'checkpoint/deepGrid_DPNH2seg_pretrained.tar',
            'threshold': 0.5,
        },
}

# Configuration
# directory for loading training/validation/test data
data_dir = join('data/GSV_images', region)
save_dir = join('data/CAM_images_for_poles', pole_model, region)
old_ckpt_path = pole_model_configs[pole_model]['ckpt_path']
CAM_save_path = join('results', region, pole_model, 'CAM_info_pole.pickle')

if not exists(join('data/CAM_images_for_poles', pole_model)):
    os.mkdir(join('data/CAM_images_for_poles', pole_model))
if not exists(save_dir):
    os.mkdir(save_dir)
if not exists(join('results', region, pole_model)):
    os.mkdir(join('results', region, pole_model))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_size = 640
batch_size = 1   # must be 1 for testing segmentation
threshold = pole_model_configs[pole_model]['threshold']  # threshold probability to identify am image as positive
level = 1

save_separated = True


def predict_model(model, dataloader, threshold):
    model.eval()
    CAM_list = []
    for inputs, paths in tqdm(dataloader):
        inputs = inputs.to(device)
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
            if label:
                # only use the generated CAM if it is predicted to be 1
                CAM_list.append((CAM, paths[i], label))
            else:
                # otherwise the CAM is a totally black one
                CAM_list.append((np.zeros_like(CAM), paths[i], label))

    return CAM_list


transform_test = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

if __name__ == '__main__':

    # data
    dataset_test = ImagePredictDataset(data_dir, transform_test)
    dataloader_test = DataLoader(
        dataset_test, batch_size=batch_size, shuffle=False, num_workers=4)
    # model
    model = InceptionSegmentation(num_outputs=2, level=level)
    model.load_existing_params(old_ckpt_path)
    model = model.to(device)
    CAM_list = predict_model(model, dataloader_test, threshold=threshold)

    print('Save\n')
    info_list = []
    if(save_separated):
        for cam, path, label in tqdm(CAM_list):
            if (label):
                filename = save_dir + os.sep + path[-16:-3] + 'cam'
                with open(filename, 'wb') as f:
                    np.save(f, cam)
                    f.close()
            info_list.append([path, label])
        with open(CAM_save_path, 'wb') as f:
            pickle.dump(info_list, f)
    else:
        with open(CAM_save_path, 'wb') as f:
            pickle.dump(CAM_list, f)
