import math
from matplotlib.path import Path
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from os.path import join, exists
from tqdm import tqdm
from streetView import StreetView

from shapely.geometry import Polygon, Point
from shapely.ops import cascaded_union
import shapely

"""
This script is for retrieving the street view meta data for each test area.
"""

region = 'Kampala_Kololo'
start_i = 0

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

fips_list_dict = {
    'Kampala_Kololo': ['U000005', 'U000006', 'U000007', 'U000012'],
    'Nairobi_Highridge': ['K101060303'],
    'Kampala_Ntinda': ['U000069'],
    'Nairobi_Ngara': ['K101010501', 'K101010502'],
}

with open('data/boundary_list_NigeriaGhana_2.pickle', 'rb') as f:
    fips_list_dict_2 = pickle.load(f)
for x in fips_list_dict_2:
    if not x in fips_list_dict:
        fips_list_dict[x] = fips_list_dict_2[x]

boundaries_keys = fips_list_dict[region]


if not exists(join('results', region)):
    os.mkdir(join('results', region))

boundaries_file_filepath = 'data/africa_boundaries.pickle'
GSV_points_filepath = join('results', region, 'validGSVpoints.pickle')
fips_mapping_path = join('results', region, 'GSVpoint2fips.pickle')
save_per_points = 5000

# %% set boundary paths
with open(boundaries_file_filepath, 'rb') as f:
    boundaries_raw = pickle.load(f, encoding='latin1')
    f.close()

boundaries = []
for key in boundaries_keys:
    if type(key) == str:
        boundaries.append(boundaries_raw[key])
    else:
        boundaries.append(Polygon(np.flip(key)))

# %% get bounds for the scan area
lat_max_list = []
lon_max_list = []
lat_min_list = []
lon_min_list = []
for boundary in boundaries:
    lat_max_list.append(boundary.bounds[3])
    lat_min_list.append(boundary.bounds[1])
    lon_max_list.append(boundary.bounds[2])
    lon_min_list.append(boundary.bounds[0])
lat_min = np.min(lat_min_list)
lon_min = np.min(lon_min_list)
lat_max = np.max(lat_max_list)
lon_max = np.max(lon_max_list)

delta_lat = 0.000089
delta_lon = delta_lat / np.cos(np.deg2rad((lat_max + lat_min) / 2))
print('delta lat: ' + str(delta_lat) + ' delta lon: ' + str(delta_lon))
lat_range = np.arange(lat_min, lat_max, delta_lat)
lon_range = np.arange(lon_min, lon_max, delta_lon)
lat_points, lon_points = np.meshgrid(lat_range, lon_range)
lat_points = np.reshape(lat_points, np.size(lat_points))
lon_points = np.reshape(lon_points, np.size(lon_points))

# scan GSV meta
# %% scan on the mesh to get the points which contained by the path
print('Collect all points within the target region ...')
points = []
fips_mapping = {}
for i in tqdm(range(0, np.size(lat_points))):
    point = (lat_points[i], lon_points[i])
    pp = Point(point[1], point[0])
    for j, boundary in enumerate(boundaries):
        if (boundary.contains(pp)):
            points.append(point)
            if type(boundaries_keys[j]) == str:
                fips_mapping[point] = boundaries_keys[j]
            else:
                fips_mapping[point] = str(j)
            break

# %% search the points to get Google API meta locations
print('Check meta data ...')
sv = StreetView()
if exists(GSV_points_filepath) and exists(fips_mapping_path):
    with open(GSV_points_filepath, 'rb') as f:
        validPoints = set(pickle.load(f))
    with open(fips_mapping_path, 'rb') as f:
        fips_mapping_sub = pickle.load(f)
    print('Loaded existing valid points and FIPS mapping ...')
else:
    validPoints = set()
    fips_mapping_sub = {}

def save_found_points():
    with open(GSV_points_filepath, 'wb') as f:
        pickle.dump(list(validPoints), f)
        f.close()
    with open(fips_mapping_path, 'wb') as f:
        pickle.dump(fips_mapping_sub, f)
        f.close()


for i, point in tqdm(list(enumerate(points))):
    if i < start_i:
        continue
    status, meta = sv.getMetaStatus(point,  radius='10', returnMeta=True)
    if (status == 'OK'):
        validPoints.add((meta['location']['lat'], meta['location']['lng']))
        fips_mapping_sub[(meta['location']['lat'], meta['location']['lng'])] = fips_mapping[point]
    elif (status == 'ZERO_RESULTS'):
        pass
    else:
        print('ERROR: ' + status)
        # break
    if (i % save_per_points == 0):
        save_found_points()
else:
    print('Search Finish!')
save_found_points()
