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

region = 'Kampala_Kololo'

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

start_i = 0

GSV_points_filepath = join('results', region, 'validGSVpoints.pickle')
fips_mapping_path = join('results', region, 'GSVpoint2fips.pickle')

# directory to save street view images
GSV_image_path = join('data/GSV_images', region)

boundaries_keys_set = set()
for i, key in enumerate(boundaries_keys):
    if type(key) == str:
        boundaries_keys_set.add(key)
    else:
        boundaries_keys_set.add(str(i))

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
