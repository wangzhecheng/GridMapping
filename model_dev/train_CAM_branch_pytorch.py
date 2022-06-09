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
This script is for training the CAM branch of the line/pole identification model 
on the street view image training set and saving the model with the best performance
on the validation set.
CAM branch is used for estimating pole orientation/line directionsm and it is trained
when the main branch (classification branch) is frozen.
Due to the restriction by the imagery data source (Google street views), 
we are not able to share the full training/validation set. Instead, for line
or pole identification, we sample 20 images as a demo training set and 20 
images as a demo validation set.
Note: if the script is run on CPU instead of GPU, please replace ".cpu().item()"
with ".item()". E.g., line 216.
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
# directory for loading training/validation data
dirs_list_dict = {
    'train': [
        [
            join(root_dir, 'data', dataset_dirs[target], 'demo_train', '0'), # negative samples
        ],
        [
            join(root_dir, 'data', dataset_dirs[target], 'demo_train', '1'), # positive samples
        ]
    ],
    'val': [
        [
            join(root_dir, 'data', dataset_dirs[target], 'demo_val', '0'), # negative samples
        ],
        [
            join(root_dir, 'data', dataset_dirs[target], 'demo_val', '1'), # positive samples
        ]
    ],
}

# path to load basic main branch model, "None" if not loading.
# specify the better classification model that has been trained. Specify it when using level = 1
basic_params_path = join(root_dir, 'checkpoint', 'new_classification_model_' + target, 
                         'cls_ir_1_wd_0_lr_0.0001_8_last.tar')
# path to load old model parameters, "None" if not loading. Specify it when using level = 2
old_ckpt_path = None
# directory for saving model/checkpoint
ckpt_save_dir = join(root_dir, 'checkpoint', 'new_CAM_model_' + target)

# train the first level or second level of CAM branch, 
# the second level must be trained after the first has already been trained
level = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_name = 'cam'  # the prefix of the filename for saving model/checkpoint
return_best = True           # whether to return the best model according to the validation metrics
if_early_stop = True         # whether to stop early after validation metrics doesn't improve for definite number of epochs
input_size = 640              # image size fed into the mdoel
# imbalance_rate = 1            # weight given to the positive (rarer) samples in loss function
# learning_rate = 0.001          # learning rate
# weight_decay = 0.00           # l2 regularization coefficient
batch_size = 16
num_epochs = 100               # number of epochs to train
lr_decay_rate = 0.7           # learning rate decay rate for each decay step
lr_decay_epochs = 5          # number of epochs for one learning rate decay
early_stop_epochs = 10        # after validation metrics doesn't improve for "early_stop_epochs" epochs, stop the training.
save_epochs = 20              # save the model/checkpoint every "save_epochs" epochs
threshold = 0.5               # threshold probability to identify am image as positive

lr_list = [0.0001]
weight_decay_list = [0]
imbalance_rate_list = [1]

def RandomRotationNew(image):
    angle = random.choice([0, 90, 180, 270])
    image = TF.rotate(image, angle)
    return image

def only_train(model, trainable_params):
    """trainable_params: The list of parameters and modules that are set to be trainable.
    Set require_grad = False for all those parameters not in the trainable_params"""
    print('Only the following layers:')
    for name, p in model.named_parameters():
        p.requires_grad = False
        for target in trainable_params:
            if target == name or target in name:
                p.requires_grad = True
                print('    ' + name)
                break

def metrics(stats):
    """
    Self-defined metrics function to evaluate and compare models
    stats: {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN}
    return: must be a single number """
    precision = (stats['TP'] + 0.00001) * 1.0 / (stats['TP'] + stats['FP'] + 0.00001)
    recall = (stats['TP'] + 0.00001) * 1.0 / \
        (stats['TP'] + stats['FN'] + 0.00001)
    # print('precision:%.4f recall:%.4f' % (precision, recall))
    return 2 * precision * recall / (precision + recall)


def train_model(model, model_name, dataloaders, criterion, optimizer, metrics, num_epochs, threshold=0.5, training_log=None,
                verbose=True, return_best=True, if_early_stop=True, early_stop_epochs=10, scheduler=None,
                save_dir=None, save_epochs=5):
    since = time.time()
    if not training_log:
        training_log = dict()
        training_log['train_loss_history'] = []
        training_log['val_loss_history'] = []
        training_log['val_metric_value_history'] = []
        training_log['current_epoch'] = -1
    current_epoch = training_log['current_epoch'] + 1

    best_model_wts = copy.deepcopy(model.state_dict())
    best_optimizer_wts = copy.deepcopy(optimizer.state_dict())
    best_log = copy.deepcopy(training_log)

    best_metric_value = -np.inf
    nodecrease = 0  # to count the epochs that val loss doesn't decrease
    early_stop = False

    for epoch in range(current_epoch, current_epoch + num_epochs):
        if verbose:
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            stats = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase], ascii=True):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs, testing=False)
                    loss = criterion(outputs, labels)

                    prob = F.softmax(outputs, dim=1)
                    preds = prob[:, 1] >= threshold

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                stats['TP'] += torch.sum((preds == 1) * (labels == 1)).cpu().item()
                stats['TN'] += torch.sum((preds == 0) * (labels == 0)).cpu().item()
                stats['FP'] += torch.sum((preds == 1) * (labels == 0)).cpu().item()
                stats['FN'] += torch.sum((preds == 0) * (labels == 1)).cpu().item()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_metric_value = metrics(stats)

            if verbose:
                print('{} Loss: {:.4f} Metrics: {:.4f}'.format(phase, epoch_loss, epoch_metric_value))

            training_log['current_epoch'] = epoch
            if phase == 'val':
                training_log['val_metric_value_history'].append(epoch_metric_value)
                training_log['val_loss_history'].append(epoch_loss)
                # deep copy the model
                if epoch_metric_value > best_metric_value:
                    best_metric_value = epoch_metric_value
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_optimizer_wts = copy.deepcopy(optimizer.state_dict())
                    best_log = copy.deepcopy(training_log)
                    nodecrease = 0
                else:
                    nodecrease += 1
            else:  # train phase
                training_log['train_loss_history'].append(epoch_loss)
                if scheduler != None:
                    scheduler.step()

            if nodecrease >= early_stop_epochs:
                early_stop = True

        if save_dir and epoch % save_epochs == 0 and epoch > 0:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'training_log': training_log
            }
            torch.save(checkpoint,
                       os.path.join(save_dir, model_name + '_' + str(training_log['current_epoch']) + '.tar'))

        if if_early_stop and early_stop:
            print('Early stopped!')
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best validation metric value: {:4f}'.format(best_metric_value))

    # load best model weights
    if return_best:
        model.load_state_dict(best_model_wts)
        optimizer.load_state_dict(best_optimizer_wts)
        training_log = best_log

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'training_log': training_log
    }
    torch.save(checkpoint,
               os.path.join(save_dir, model_name + '_' + str(training_log['current_epoch']) + '_last.tar'))

    return model, training_log, best_metric_value


data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(input_size),
        transforms.Lambda(RandomRotationNew),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
}


if __name__ == '__main__':
    assert level in [1, 2]
    # data
    image_datasets = {x: FolderDirsDataset(dirs_list_dict[x], data_transforms[x]) for x in ['train', 'val']}
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                       shuffle=True, num_workers=4) for x in ['train', 'val']}
    print('Training set size: ' + str(len(image_datasets['train'])))
    print('Validation set size: ' + str(len(image_datasets['val'])))

    if not os.path.exists(ckpt_save_dir):
        os.makedirs(ckpt_save_dir)

    results_dict = {x: {y: {z: {} for z in lr_list} for y in weight_decay_list} for x in imbalance_rate_list}

    for imbalance_rate in imbalance_rate_list:
        for weight_decay in weight_decay_list:
            for learning_rate in lr_list:
                print('----------------------- ' +
                      str(imbalance_rate) + ', ' +
                      str(weight_decay) + ', ' +
                      str(learning_rate) +
                      ' -----------------------')

                # model
                model = InceptionSegmentation(num_outputs=2, level=level)
                if level == 1 and basic_params_path:
                    model.load_basic_params(basic_params_path)
                elif level == 2 and old_ckpt_path:
                    model.load_existing_params(old_ckpt_path)

                if level == 1:
                    trainable_params = ['convolution1', 'linear1']
                else:
                    trainable_params = ['convolution2', 'linear2']
                only_train(model, trainable_params)

                model = model.to(device)
                optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08,
                                       weight_decay=weight_decay, amsgrad=True)
                if (torch.torch.cuda.is_available()):
                    class_weight = torch.tensor([1, imbalance_rate], dtype=torch.float).cuda()
                else:
                    class_weight = torch.tensor([1, imbalance_rate], dtype=torch.float)
                loss_fn = nn.CrossEntropyLoss(weight=class_weight)
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_epochs, gamma=lr_decay_rate)

                training_log = None

                _, _, best_metric_value = train_model(model, model_name=model_name + '_ir_' + str(
                    imbalance_rate) + '_wd_' + str(weight_decay) + '_lr_' + str(learning_rate),
                                                      dataloaders=dataloaders_dict, criterion=loss_fn,
                                                      optimizer=optimizer, metrics=metrics,
                                                      num_epochs=num_epochs, threshold=threshold,
                                                      training_log=training_log, verbose=True,
                                                      return_best=return_best,
                                                      if_early_stop=if_early_stop, early_stop_epochs=early_stop_epochs,
                                                      scheduler=scheduler, save_dir=ckpt_save_dir,
                                                      save_epochs=save_epochs)

                results_dict[imbalance_rate][weight_decay][learning_rate] = {'best_metric_val': best_metric_value}

                with open(join(ckpt_save_dir, 'results_dict.pickle'), 'wb') as f:
                    pickle.dump(results_dict, f)

    print(results_dict)
