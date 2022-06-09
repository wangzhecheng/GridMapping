import math
from matplotlib.path import Path
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from os.path import join, exists
from tqdm import tqdm
from streetView import StreetView

"""
This script is for retrieving the street view meta data for each test area.
"""

region = 'Salinas'
start_i = 0

# census tract list for each test area
fips_list_dict = {
    'SanCarlos': [6081609100, 6081609201, 6081609202, 6081609300, 6081609400, 6081609500, 6081609601, 6081609602, 6081609603],
    'Newark': [6001444100, 6001444200, 6001444301, 6001444400, 6001444500, 6001444601],
    'SantaCruz': [6087121300, 6087121401, 6087121402, 6087121403, 6087121700], 
    'Yuba': [6101050201, 6101050202, 6101050301, 6101050302],
    'Monterey': [6053012401, 6053012402, 6053012302, 6053012200, 6053012100, 6053012000],
    'Salinas': [6053000800, 6053000600, 6053000701, 6053000702],
}

boundaries_keys = fips_list_dict[region]

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

if not exists(join('results', region)):
    os.mkdir(join('results', region))

boundaries_file_filepath = '/home/ubuntu/projects/DeepGrid_run_data/tract_boundaries_us.pickle'
GSV_points_filepath = join('results', region, 'validGSVpoints.pickle')
fips_mapping_path = join('results', region, 'GSVpoint2fips.pickle')
save_per_points = 5000

# %% set boundary paths
with open(boundaries_file_filepath, 'rb') as f:
    boundaries_raw = pickle.load(f, encoding='latin1')
    f.close()

boundaries = []
for key in boundaries_keys:
    boundaries.append(boundaries_raw[str.format('%011d' % key)])

# %% get scan area mesh points
bd = []
paths = []
for boundary in boundaries:
    bd.extend(boundary)
    paths.append(Path(boundary))
lat_min, lon_min = np.min(bd, axis=0)
lat_max, lon_max = np.max(bd, axis=0)
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
    for j, path in enumerate(paths):
        if (path.contains_points([point])):
            points.append(point)
            fips_mapping[point] = boundaries_keys[j]
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
