import numpy as np
import pickle
from tqdm import tqdm
import os
from math import atan2, sqrt, pi, sin, cos, tan
import matplotlib.pyplot as plt
from os.path import join, exists

"""
This script is for intersecting the Lines of Bearings (LOBs) obtained from step 7
to obtain the pole locations.
Multiple regions can be run all at once.
"""

region_list = ['Salinas'] # 'SanCarlos', 'Newark', 'SantaCruz', 'Yuba', 'Monterey', 'Salinas'
pole_model_list = ['ori0.5']


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

def getDistance(pt1, pt2):
    #latitude, longitude
    lat1, lon1 = pt1
    lat2, lon2 = pt2
    lat = (lat1 + lat2) / 2 * pi / 180
    dlat = (lat2 - lat1) * pi / 180
    dlon = (lon2 - lon1) * pi / 180
    R = 6378.137  # km
    dl = R * sqrt(dlat ** 2 + (cos(lat) * dlon) ** 2)
    return dl * 1000


def getIntersection(LOB1, LOB2):
    eps = 0.5
    location1, direction1 = LOB1
    location2, direction2 = LOB2
    lat1, lon1 = location1
    lat2, lon2 = location2
    y1, x1 = location1
    y2, x2 = location2
    lat = (lat1 + lat2) / 2 * pi / 180

    if (direction1 == direction2):
        # parallel
        return []

    beta1 = direction1 * pi / 180
    beta2 = direction2 * pi / 180
    tb1 = tan(beta1) * cos(lat)
    tb2 = tan(beta2) * cos(lat)

    if tb1 - tb2 == 0:
        # print(direction1, direction2)
        return []

    xt = (x1 * tb1 - x2 * tb2 + y2 - y1) / (tb1 - tb2)
    yt = ((x1 * tb1 - y1) * tb2 - (x2 * tb2 - y2) * tb1) / (tb1 - tb2)

    beta1_t = atan2(yt - y1, xt - x1)
    beta2_t = atan2(yt - y2, xt - x2)
    beta1_t = (beta1_t + 2 * pi) if (beta1_t < 0) else beta1_t
    beta2_t = (beta2_t + 2 * pi) if (beta2_t < 0) else beta2_t

    if ((abs(beta1_t - beta1) > eps) or (abs(beta2_t - beta2) > eps)):
        # no intersection, wrong direction
        return []
    pt = [yt, xt]

    if (getDistance(pt, location1) > area_threshold):
        return []
    if (getDistance(pt, location2) > area_threshold):
        return []
    return pt


for region in region_list:
    for pole_model in pole_model_list:
        print('-'*20)
        print('Region: ' + region + '. Pole model: ' + pole_model)
        # datapath
        pole_directions_path = join('results', region, pole_model, 'pole_LOB_raw.pickle')
        savefile = join('results', region, pole_model, 'pole_locations.pickle')
        attached_location_file = join('results', region, pole_model, 'pole_attached_GSVs.pickle')

        with open(pole_directions_path, 'rb') as f:
            pole_directions = pickle.load(f)
            f.close()

        distance_threshold = 40
        cluster_threshold = 10
        area_threshold = 20

        clusters = list()
        # [lat,lon,combo]
        LOB_len = len(pole_directions)
        attached_location = list()
        for i in tqdm(range(LOB_len)):
            location, direction = pole_directions[i]
            for j in range(i + 1, LOB_len):
                location2, direction2 = pole_directions[j]
                if (getDistance(location, location2) > distance_threshold):
                    continue
                pt = getIntersection([location, direction], [location2, direction2])
                if (pt == []):
                    continue
                # cluster insert
                for k, cluster in enumerate(clusters):
                    lat0, lon0, combo = cluster
                    lat1, lon1 = pt
                    if (getDistance([lat0, lon0], pt) < cluster_threshold):
                        lat_new = (lat0 * combo + lat1) / (combo + 1)
                        lon_new = (lon0 * combo + lon1) / (combo + 1)
                        combo_new = combo + 1
                        clusters[k] = [lat_new, lon_new, combo_new]
                        attached_location[k].extend([location, location2])
                        break
                else:
                    clusters.append([pt[0], pt[1], 1])
                    attached_location.append([location, location2])

        with open(savefile, 'wb') as f:
            pickle.dump(clusters, f)

        with open(attached_location_file, 'wb') as f:
            pickle.dump(attached_location, f)
