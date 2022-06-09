from os.path import join, exists
import pandas as pd
import numpy as np
import pickle
from bisect import bisect_left, bisect_right
import os

from shapely.geometry import Polygon, Point
import shapely


class RegionLocator(object):
    def __init__(self, locator_data_dir=None):
        self.locator_data = None
        if locator_data_dir is not None:
            if exists(locator_data_dir):
                with open(locator_data_dir, 'rb') as f:
                    self.locator_data = pickle.load(f)
    
    def save_locator_data(self, boundary_file_dir, locator_data_dir=None):
        with open(boundary_file_dir, 'rb') as f:
            boundaries = pickle.load(f)
        region_names = list(boundaries.keys())
        lms = ('lat_min', 'lat_max', 'lon_min', 'lon_max')
        lm_list = dict()
        for lm in lms:
            lm_list[lm] = []
        max_lat_dist = -1
        max_lon_dist = -1
        for region in region_names:
            bd = boundaries[region]
            bf_lms = {
                'lat_min': bd.bounds[1],
                'lat_max': bd.bounds[3],
                'lon_min': bd.bounds[0],
                'lon_max': bd.bounds[2]
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

        locator_data = dict()
        locator_data['region_names'] = region_names
        locator_data['boundaries'] = boundaries
        locator_data['lm_list'] = lm_list
        locator_data['lm_argsort'] = lm_argsort
        locator_data['lm_sort'] = lm_sort
        locator_data['max_lat_dist'] = max_lat_dist
        locator_data['max_lon_dist'] = max_lon_dist
        
        if self.locator_data is None:
            self.locator_data = locator_data

        if locator_data_dir is not None:
            if not exists(locator_data_dir):
                with open(locator_data_dir, 'wb') as f:
                    pickle.dump(locator_data, f)
        
    def locate(self, lat, lon):
        ll_arg = {'lat': lat, 'lon': lon}
        idx_dict = dict()
        regions = None
        for ll in ['lat', 'lon']:
            idx_dict[ll] = dict()
            for mm in ['min', 'max']:
                idx_dict[ll][mm] = dict()
                for lr in ['left', 'right']:
                    if lr == 'left' and mm == 'min':
                        ref_coord = ll_arg[ll] - self.locator_data[''.join(('max_', ll, '_dist'))]
                    elif (lr == 'right' and mm == 'min') or (lr == 'left' and mm == 'max'):
                        ref_coord = ll_arg[ll]
                    else:
                        ref_coord = ll_arg[ll] + self.locator_data[''.join(('max_', ll, '_dist'))]

                    if lr == 'left':
                        idx_dict[ll][mm][lr] = bisect_left(
                            self.locator_data['lm_sort']['_'.join((ll, mm))],
                            ref_coord
                        )
                    else:
                        idx_dict[ll][mm][lr] = bisect_right(
                            self.locator_data['lm_sort']['_'.join((ll, mm))],
                            ref_coord
                        )

                temp_regions = \
                    [self.locator_data['region_names'][idx] for idx in self.locator_data['lm_argsort']['_'.join((ll, mm))][
                        idx_dict[ll][mm]['left']:idx_dict[ll][mm]['right']
                    ]]

                if regions is None:
                    regions = set(temp_regions)
                else:
                    regions = regions.intersection(set(temp_regions))

        for region in list(regions):
            poly = self.locator_data['boundaries'][region]
            if poly.contains(Point(lon, lat)):
                return region
        return -1

