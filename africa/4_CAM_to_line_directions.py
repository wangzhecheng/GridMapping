import ctypes
import numpy as np
import pickle
from tqdm import tqdm
import os
from os.path import join, exists

"""
This script is for using Hough transform to extract line directions
from CAMs.
Note: this must be run after running step 3: "3_predict_line_CAM_pytorch.py".
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

# ctypes initial
HoughTransform_lib = ctypes.cdll.LoadLibrary(
    'HoughTransform_lib.so')
HoughTransform_lib.init()
cam2lines = HoughTransform_lib.cam2lines
cam2lines.argtypes = [np.ctypeslib.ndpointer(
    ctypes.c_double, flags='C_CONTIGUOUS')]
cam2lines.restype = ctypes.c_int
getlines = HoughTransform_lib.getlines
getlines.argtypes = []
getlines.restype = ctypes.POINTER(ctypes.c_double)

# rh & th
'''
getth = HoughTransform_lib.getth
getth.restype = ctypes.POINTER(ctypes.c_double)
getrh = HoughTransform_lib.getrh
getrh.restype = ctypes.POINTER(ctypes.c_double)
th_raw = getth()
rh_raw = getrh()
th_len = HoughTransform_lib.getthlen()
rh_len = HoughTransform_lib.getrhlen()
th = np.fromiter(th_raw, dtype=np.double, count=th_len)
rh = np.fromiter(rh_raw, dtype=np.double, count=rh_len)
'''

CAMinfopath = join('results', region, 'CAM_info_line.pickle')
rawdatapath = join('data/GSV_images', region)
CAMdatapath = join('data/CAM_images', region)
savefile = join('results', region, 'line_info_raw.pickle')
#savepath = projectPath+'data/CAM2Lines_new2'
#os.makedirs(savepath, exist_ok=True)

with open(CAMinfopath, 'rb') as info:
    caminfos = pickle.load(info)
    info.close()

linesDictionary = dict()


for path, label in tqdm(caminfos):
    if (label == 0):
        linesDictionary[path] = []
        continue
    with open(CAMdatapath+os.sep+path[-16:-3]+'cam', 'rb') as f:
        cam = np.load(f)
        f.close()
    #cam = np.reshape(cam, np.size(cam))
    cam_array = np.ascontiguousarray(cam, dtype=ctypes.c_double)
    #cam_p = cam_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    lines_len = cam2lines(cam_array)
    lines_raw = getlines()
    lines = np.fromiter(lines_raw, dtype=np.double, count=3 * lines_len)
    lines = np.resize(lines, (lines_len, 3))

    linesDictionary[path] = lines

with open(savefile, 'wb') as f:
    pickle.dump(linesDictionary, f)
    f.close()


HoughTransform_lib.destory()
