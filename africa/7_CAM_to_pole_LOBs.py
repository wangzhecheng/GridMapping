import numpy as np
import pickle
from tqdm import tqdm
import os
from math import atan2, pi, sin, cos
import matplotlib.pyplot as plt
from os.path import join, exists

"""
This script is for extracting pole orientations from CAMs and obtaining the
Line of Bearings (LOBs, rays that represent pole orientations)
Multiple regions can be run all at once.
Note: this must be run after running step 6: "6_predict_pole_CAM_pytorch.py".
"""

region_list = ['Kampala_Kololo'] # 'Kampala_Ntinda', 'Kampala_Kololo', 'Nairobi_Highridge', 'Nairobi_Ngara', 'Lagos_Ikeja2'

pole_model_list = ['ori0.2']

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

for region in region_list:
    for pole_model in pole_model_list:
        print('-'*20)
        print('Region: ' + region + '. Pole model: ' + pole_model)
        # datapath
        CAMdatapath = join('data/CAM_images_for_poles', pole_model, region)
        Line_CAM_info_path = join('results', region, 'CAM_info_line.pickle')
        Pole_CAM_info_path = join('results', region, pole_model, 'CAM_info_pole.pickle')
        Point_info_path = join('results', region, 'validGSVpoints.pickle')
        LOB_savefile = join('results', region, pole_model, 'pole_LOB_raw.pickle')
        threshold = 100

        with open(Pole_CAM_info_path, 'rb') as f:
            pole_infos = pickle.load(f)
            f.close()

        with open(Line_CAM_info_path, 'rb') as f:
            line_infos = pickle.load(f)
            f.close()

        with open(Point_info_path, 'rb') as f:
            point_infos = pickle.load(f)
            f.close()

        pole_directions = list()
        for i in tqdm(range(len(pole_infos))):
            path_pole, label_pole = pole_infos[i]
            path_line, label_line = line_infos[i]
            rank = int(path_pole[-14:-6])
            location = point_infos[rank]
            if ((label_line == 0) or (label_pole == 0)):
                continue
            with open(os.path.join(CAMdatapath, path_pole[-16:-3] + 'cam'), 'rb') as f:
                # cam = np.fromfile(f, dtype=np.float32)
                # cam = np.reshape(cam, (77, 77))
                cam = np.load(f)
                f.close()
            imgsize_y, imgsize_x = np.shape(cam)
            thetas = np.zeros(360)
            for i in range(imgsize_y):
                for j in range(imgsize_x):
                    dy = i - int(imgsize_y / 2)
                    dx = j - int(imgsize_x / 2)
                    alpha = int(round(atan2(dy, dx) * 180 / pi))
                    thetas[alpha] += cam[i, j]
            theta = np.argmax(thetas)
            pole_directions.append([location, theta])

        with open(LOB_savefile, 'wb') as f:
            pickle.dump(pole_directions, f)
