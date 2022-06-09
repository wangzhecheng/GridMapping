# Geospatial Mapping of Distribution Grid with Machine Learning and Multi-Modal Open Data

This repo is a temporary code repo compressed from a private GitHub repo for peer review. It will be made public before publication.

The operating system for developing this code repo is Ubuntu 16.04, but it should also be able to run in other environments. The Python version used for developing this code repo is Python 3.6.

## Install required packages

Run the following command line:

```
$ pip install -r requirements.txt
```

## Download data and model checkpoints

Run the following command lines to download the ZIP files right under the code repo directory:

```
$ curl -O https://opendatasharing.s3.us-west-2.amazonaws.com/GridMapping/checkpoint.zip
$ curl -O https://opendatasharing.s3.us-west-2.amazonaws.com/GridMapping/data.zip
$ curl -O https://opendatasharing.s3.us-west-2.amazonaws.com/GridMapping/results.zip
$ curl -O https://opendatasharing.s3.us-west-2.amazonaws.com/GridMapping/ground_truth.zip
```

Unzip them such that the directory structure looks like:

```
GridMapping/checkpoint/...
GridMapping/data/...
GridMapping/results/...
GridMapping/ground_truth/...
```

**Note 1**: for the street view imagery datasets (`line_image_dataset_demo` for line detection, `pole_image_dataset_demo` for pole detection) under `dataset`, due to the restriction of the imagery data source, we are not able to publicly share the full training/validation set. Instead, for each of the line detection and pole detection datasets, we sample 20 images as a demo training set and 20 images as a demo validation set. However, we share full imagery test sets for peer-review purpose only.

**Note 2**: to run Jupyter Notebook, the default kernel/environment is "conda_tensorflow_p36", which does not necessarily exist in your computer. Please change the kernel to the one where all required packages are installed.

## Functionality of each script/notebook

### Model development (CNN model training and testing)

This part is under `model_dev`. It is for training and testing the classification and Class Activation Map (CAM) generation for both line detector and pole detector. First, go to the directory:

```
$ cd model_dev
```

Below are the functionalities for each of the scripts. All scripts are run on a Nvidia Tesla K80 GPU by default. For each script, set `target = "line"` for line detection and `target = "pole"` for pole detection.

`test_classification_pytorch.py`: This script is for running the line/pole classification model on the street view image test set and reporting the image-level metrics including precision and recall. For line identification, the expected image-level precision and recall under the given threshold 0.5 are 0.982 and 0.937; for pole identificaion, the expected image-level precision and recall under the given threshold 0.5 are 0.982 and 0.850, respectively. It takes ~1 min to run on a Nvidia Tesla K80 GPU.

`train_classification_pytorch.py`: This script is for training the line/pole classification model on the street view image training set and saving the model with the best performance on the validation set. A pre-trained model can be specified. The default pre-trained model is the model pretrained on ImageNet dataset: `checkpoint/inception_v3_google-1a9a5a14.pth`.

`train_CAM_branch_pytorch.py`: This script is for training the CAM branch of the line/pole model on the street view image training set and saving the model with the best performance on the validation set. CAM branch is used for estimating pole orientation/line directions and it is trained when the main branch (classification branch) is frozen. It must take a classification model checkpoint outputted by `train_classification_pytorch.py` as input for model initialization.

`test_CAM_branch_pytorch.py`: This script is for running the line/pole model on the street view image test set to generate the Class Activation Maps (CAMs) for lines/poles. The CAMs will be saved into a pickle file at the given destination for further visualization.

### Model running and benchmarking (for California or Africa test areas)


