import numpy as np
import pickle
from tqdm import tqdm
from streetView import StreetView
import os
from os.path import join, exists
import urllib.error
# import skimage.io

"""
This script is for downloading upward street view images for each test area.
"""

region = 'Salinas'

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

start_i = 0

GSV_points_filepath = join('results', region, 'validGSVpoints.pickle')
fips_mapping_path = join('results', region, 'GSVpoint2fips.pickle')

# directory to save street view images
GSV_image_path = join('data/GSV_images', region)

boundaries_keys_set = set(boundaries_keys)

if not exists(join('results', region)):
    os.mkdir(join('results', region))
if not exists(GSV_image_path):
    os.mkdir(GSV_image_path)

with open(GSV_points_filepath, 'rb') as f:
    validPoints = pickle.load(f)
    f.close()

with open(fips_mapping_path, 'rb') as f:
    fips_mapping = pickle.load(f)
    f.close()

sv = StreetView()
fov = '120'
error_list = []

for i, point in tqdm(list(enumerate(validPoints))):
    if i < start_i:
        continue
    if fips_mapping[point] in boundaries_keys_set:
        filename = GSV_image_path + os.sep + 'SC%08dNH.jpg' % i
        try:
            sv.getStreetView(point, filepath=filename, heading='0',
                         pitch='90', fov=fov, radius='10', uncheck=False)
        except urllib.error.HTTPError:
#             skimage.io.imsave(filename, np.zeros((640, 640, 3)).astype(int))
            error_list.append(i)
print(error_list)
