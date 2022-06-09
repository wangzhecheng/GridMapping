from os import listdir
from os.path import join, exists
import pandas as pd
import numpy as np
import pickle
from bisect import bisect_left, bisect_right
import matplotlib.path as mplPath


class CensusTract(object):
    def __init__(self, tract_data_dir=None):
        self.tract_data = None
        if tract_data_dir is not None:
            if exists(tract_data_dir):
                with open(tract_data_dir, 'rb') as f:
                    self.tract_data = pickle.load(f)
        pass

    def save_tract_data(self, boundary_file_dir, tract_data_dir=None):

        raw_tract_names = []
        for bf in listdir(boundary_file_dir):
            # File name example: 1400000US01097001400_polylatlon1.csv
            if bf[-4:] == '.csv':
                tract_name = bf[9:20]
                raw_tract_names.append(tract_name)

        tract_names = sorted(raw_tract_names)

        tract_boundaries = dict()

        lms = ('lat_min', 'lat_max', 'lon_min', 'lon_max')
        lm_list = dict()
        for lm in lms:
            lm_list[lm] = []

        max_lat_dist = -1
        max_lon_dist = -1

        for tract_name in tract_names:
            bf = ''.join(['1400000US', tract_name, '_polylatlon1.csv'])
            lat_lon = pd.read_csv(join(boundary_file_dir, bf), header=-1).values

            tract_boundaries[tract_name] = lat_lon

            bf_lms = {
                'lat_min': np.min(lat_lon[:, 0]),
                'lat_max': np.max(lat_lon[:, 0]),
                'lon_min': np.min(lat_lon[:, 1]),
                'lon_max': np.max(lat_lon[:, 1])
            }

            max_lat_dist = max(max_lat_dist, bf_lms['lat_max'] - bf_lms['lat_min'])
            max_lon_dist = max(max_lon_dist, bf_lms['lon_max'] - bf_lms['lon_min'])

            for lm in lms:
                lm_list[lm].append(bf_lms[lm])

        lm_argsort = dict()
        lm_sort = dict()
        for lm in lms:
            lm_list[lm] = np.array(lm_list[lm])
            lm_argsort[lm] = np.argsort(lm_list[lm])
            lm_sort[lm] = lm_list[lm][lm_argsort[lm]]

        tract_data = dict()
        tract_data['tract_names'] = tract_names
        tract_data['tract_boundaries'] = tract_boundaries
        tract_data['lm_list'] = lm_list
        tract_data['lm_argsort'] = lm_argsort
        tract_data['lm_sort'] = lm_sort
        tract_data['max_lat_dist'] = max_lat_dist
        tract_data['max_lon_dist'] = max_lon_dist

        if self.tract_data is None:
            self.tract_data = tract_data

        if tract_data_dir is not None:
            if not exists(tract_data_dir):
                with open(tract_data_dir, 'wb') as f:
                    pickle.dump(tract_data, f)

    def get_tract(self, lat, lon):

        ll_arg = {'lat': lat, 'lon': lon}

        idx_dict = dict()

        tracts = None

        for ll in ['lat', 'lon']:
            idx_dict[ll] = dict()

            for mm in ['min', 'max']:

                idx_dict[ll][mm] = dict()
                for lr in ['left', 'right']:

                    if lr == 'left' and mm == 'min':
                        ref_coord = ll_arg[ll] - self.tract_data[''.join(('max_', ll, '_dist'))]
                    elif (lr == 'right' and mm == 'min') or (lr == 'left' and mm == 'max'):
                        ref_coord = ll_arg[ll]
                    else:
                        ref_coord = ll_arg[ll] + self.tract_data[''.join(('max_', ll, '_dist'))]

                    if lr == 'left':
                        idx_dict[ll][mm][lr] = bisect_left(
                            self.tract_data['lm_sort']['_'.join((ll, mm))],
                            ref_coord
                        )
                    else:
                        idx_dict[ll][mm][lr] = bisect_right(
                            self.tract_data['lm_sort']['_'.join((ll, mm))],
                            ref_coord
                        )

                temp_tracts = \
                    [self.tract_data['tract_names'][idx] for idx in self.tract_data['lm_argsort']['_'.join((ll, mm))][
                        idx_dict[ll][mm]['left']:idx_dict[ll][mm]['right']
                    ]]

                if tracts is None:
                    tracts = set(temp_tracts)
                else:
                    tracts = tracts.intersection(set(temp_tracts))

        for tract in list(tracts):
            bb_path = mplPath.Path(self.tract_data['tract_boundaries'][tract])
            if bb_path.contains_point((lat, lon)):
                return tract

        return -1

if __name__ == '__main__':
    ct = CensusTract()
    ct.save_tract_data(boundary_file_dir='./Tracts')
    ct.get_tract(lat=1, lon=1)
    ct.get_tract(lat=32, lon=-86)
    ct.get_tract(lat=32.48, lon=-86.5)



