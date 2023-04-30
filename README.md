# Geospatial Mapping of Distribution Grid with Machine Learning and Publicly-Accessible Multi-Modal Data

This is the code repository for the following paper:

* Wang, Z., Majumdar, A., & Rajagopal, R. (2023). Geospatial Mapping of Distribution Grid with Machine Learning and Publicly-Accessible Multi-Modal Data. To appear in Nature Communications. 

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

For model running and benchmarking in the California test areas: San Carlos (`SanCarlos`), Newark (`Newark`), Santa Cruz (`SantaCruz`), Yuba City (`Yuba`), Pacific Grove (`Monterey`), and Salinas (`Salinas`):

```
$ cd california
```

For model running and benchmarking in the Sub-Saharan Africa (SSA) test areas: Ntinda, Kampala, Uganda (`Kampala_Ntinda`), Kololo, Kampala, Uganda (`Kampala_Kololo`), Highridge, Nairobi, Kenya (`Nairobi_Highridge`), Ngara, Nairobi, Kenya (`Nairobi_Ngara`), Ikeja, Lagos, Nigeria (`Lagos_Ikeja2`).

```
$ cd africa
```

They share the same pipeline for model running and benchmarking:

#### Step 1: Retrieve the meta data of street view images
A prerequisite of retrieving street view imagery meta data (presence or absence of street view images at a given location) is Google API keys. Please go to [Google Cloud Console](https://console.cloud.google.com/) to create private API keys (which is associated with Google account and cannot be shared). Add API key strings to the placeholder in `streetView.py`. Given a region (specified by its name, e.g., "Salinas"), we can run the following script for retrieving the street view meta data in this region.

```
$ python 1_search_area_GSV.py
```

#### Step 2: Download street view images
This is to download all available street view images in a given region (e.g., "Salinas").
```
$ python 2_download_area_GSV.py
```
Note that API keys are also required for this step.

#### Step 3: Line detection and CAM generation
This script is for running the line detetor on downloaded street view images. Classification results and CAMs are generated and saved:
```
$ python 3_predict_line_CAM_pytorch.py
```

#### Step 4: Extract line directions
This script is for using Hough transform to extract line directions from the CAMs generated in step 3:
```
$ python 4_CAM_to_line_directions.py
```

#### Step 5: Merge similar line directions
This script is used for merging similar line directions (i.e., parallel power lines with different phases) estimated in each CAM:
```
$ python 5_merge_line_directions.py
```

#### Step 6: Pole detection and CAM generation
This script is for running the pole detetor on downloaded street view images. Classification results and CAMs are generated and saved:
```
$ python 6_predict_pole_CAM_pytorch.py
```

#### Step 7: Extract pole orientations
This script is for extracting pole orientations from the CAMs generated in step 6 and obtaining the Line of Bearings (LOBs, i.e., rays that represent pole orientations):
```
$ python 7_CAM_to_pole_LOBs.py
```

#### Step 8: Pole localization
This script is for intersecting the Lines of Bearings (LOBs) obtained from step 7 to obtain the pole locations.
```
$ python 8_LOB_to_pole_locations.py
```

#### Step 9: Pre-processing road and building data
Run the Jupyter Notebook `9_preprocessing_road_and_building_data.ipynb` for filtering roads and buildings that are located in the given region. Two road segments are merged into a single segment if there is no road intersection between them. This is used for telling whether two detected poles are along the same roads. 

#### Step 10: Attach poles to nearby roads and generate the map of poles
This script is for attaching detected poles to the nearby road when there is a nearby road to that pole, and potentially inserting additional poles between two predicted poles that are too far apart in order to reduce the number of poles missed by the pole detection model. It finally generates the geospatial map of utility poles.
```
$ python 10_road_modeling_and_pole_prediction.py
```

#### Step 11: Dijkstra's algorithm to connect poles
This script is for running the modified Dijkstra's algorithm to predict line connections between poles. This algorithm greedily seeks the paths to connect all predicted poles with minimum total weight. Each cell on the raster map is assigned with a weight. The Dijkstra's algorithm is adapted from: https://github.com/facebookresearch/many-to-many-dijkstra. The prediction of this algorithm is used as a feature input to the link prediction model.
```
$ python 11_dijkstria_algorithm.py
```

#### Step 12: Link prediction
This script is for using link prediction model to predict whether there is a power line between two utility poles by leveraging the predicted pole/line information as well as road information.
```
$ python 12_line_prediction_test_overall.py
```
New link prediction models can be trained and compared with cross-validation by running the Jupyter Notebook `12_line_prediction_model_development.ipynb` under `california` directory (using "San Carlos" as a development set). However, training new models are not required for running the above script as the models used in the paper are already provided.

#### Step 13: Predict underground grids
This script is for predicting the underground grid on top of the predicted overhead grid by leveraging the modified Dijkstra's algorithm. This algorithm greedily seeks the paths with minimum total weight to connect all buildings that cannot be reached by predicted overhead grid within a certain distance.  Each cell in the raster is assigned with a weight. The Dijkstra's algorithm is adapted from: https://github.com/facebookresearch/many-to-many-dijkstra. The entire predicted grid map (overhead + underground) are benchmarked against the ground truth data.
```
$ python 13_connect_underground.py
```
**Note**: step 13 does not apply to the Sub-Saharan Africa test areas because the 100% building electricity access assumption does not necessarily hold in Sub-Saharan Africa.






