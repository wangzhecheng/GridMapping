import pickle
import numpy as np
import os
from tqdm import tqdm
from os.path import join, exists

"""
This script is used for merging similar line directions (i.e., parallel power lines with 
different phases) estimated in each CAM.
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

line_directions_filepath = join('results', region, 'line_info_raw.pickle')
merged_directions_filepath = join('results', region, 'line_info_merged.pickle')
Point_info_path = join('results', region, 'validGSVpoints.pickle')

with open(line_directions_filepath, 'rb') as f:
    line_info_raw = pickle.load(f)

with open(Point_info_path, 'rb') as f:
    point_infos = pickle.load(f)
    f.close()

directionInfo = []
for cam_path in line_info_raw:
    rank = int(cam_path[-14:-6])
    location = point_infos[rank]
    directionInfo.append((location, line_info_raw[cam_path], cam_path))


def lineSimilar(theta1, theta2):
    threhold = 20
    delta_theta = abs(theta1-theta2)
    if (delta_theta < threhold):
        return 1
    elif ((180 - delta_theta) < threhold):
        return 2
    else:
        return 0


def lineMean(line1, line2):
    if (lineSimilar(line1[1], line2[1]) == 1):
        lineM = (line1[1] * line1[2] + line2[1] *
                 line2[2]) / (line1[2] + line2[2])
    elif (lineSimilar(line1[1], line2[1]) == 2):
        if (line1[1] < 0):
            line1_s = line1[1] + 180
            line2_s = line2[1]
        else:
            line1_s = line1[1]
            line2_s = line2[1] + 180
        lineM = (line1_s * line1[2] + line2_s *
                 line2[2]) / (line1[2] + line2[2])
        if (lineM > 90):
            lineM = lineM - 90
    return lineM


def lineWeight(line):
    return line[2]


mergedInfo = list()

for location, lines, path in tqdm(directionInfo):
    # combine
    lineBuffer = list()
    for line in lines:
        # transform from sky view to ground view
        theta = -line[1] * 180 / np.pi
        line_regular = [1, theta, line[2]]
        if (lineBuffer == []):
            lineBuffer.append(line_regular)
        else:
            join = False
            for i, bufferedLine in enumerate(lineBuffer):
                if (lineSimilar(line_regular[1], bufferedLine[1]) > 0):
                    lineBuffer[i][0] += line_regular[0]
                    lineBuffer[i][1] = lineMean(line_regular, bufferedLine)
                    lineBuffer[i][2] += line_regular[2]
                    join = True
                    break
            if(join == False):
                lineBuffer.append(line_regular)
            lineBuffer.sort(key=lineWeight, reverse=True)

    mergedInfo.append([location, lineBuffer])

with open(merged_directions_filepath, 'wb') as f:
    pickle.dump(mergedInfo, f)
