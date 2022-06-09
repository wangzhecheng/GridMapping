from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement
import heapq
import os
import sys
import time
import pickle
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from numba import autojit
import numpy as np
import census_tract_locator
from tqdm import tqdm
from os.path import join, exists
from utils import *
from dijkstra import *

"""
This scripts is for running the modified Dijkstra's algorithm to predict 
the connections between poles. This algorithm greedily seeks the paths to
connect all predicted poles with minimum total weight. Each cell in the 
raster is assigned with a weight.
The Dijkstra's algorithm is adapted from:
https://github.com/facebookresearch/many-to-many-dijkstra
"""

pole_model_list = ['ori0.5']
region_list = ['SanCarlos', 'Newark', 'SantaCruz', 'Yuba', 'Monterey', 'Salinas']
save_raw_results = False

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

# model parameters
weight_scheme = {'motorway': 1/10, 'motorway_link': 1/10,
           'primary': 1/8, 'primary_link': 1/8,
           'secondary': 1/7, 'secondary_link': 1/7,
           'tertiary': 1/6, 'tertiary_link': 1/6,
           'unclassified': 1/5,
           'residential': 1/4, 'living_street': 1/4, 'cycleway': 1/4,
           'service': 1/3, 'footway': 1/3, 'track': 1/3, 'pedestrian': 1/3,
           'path': 1/2, 'highway': 2/3
          }

# the point to start
start_lat_lon_dict = {
    'SanCarlos': (37.4996447, -122.2440506),
    'Newark': (37.54784131, -122.04436975),
    'SantaCruz': (36.97630256, -121.9646515),
    'Yuba': (39.13770213, -121.6090814),
    'Monterey': (36.61717734, -121.91922238),
    'Salinas': (36.6738711, -121.62761453),
}


# functions
def find_nearest_road_point(ring_xy_diff_dict, pole_point, road_filled_dict, max_search_radius=5):
    assert 0 <= max_search_radius <= 200
    x, y = pole_point
    for radius in range(0, max_search_radius + 1):
        for dx, dy in ring_xy_diff_dict[radius]:
            if (x+dx, y+dy) in road_filled_dict:
                return (x+dx, y+dy)
    return -1


def scale_weight_with_street_view(ring_xy_diff_dict, road_weight_matrix, sv_pos_filled_grid, sv_neg_filled_grid,
                                  expand_radius=3, pos_scale=1.0, neg_scale=1.0):
    """scale the weights of road_weight_matrix with a factor that depends on whether there is a street view points,
    and whether the street view is positive (contains line) or negative (contains no line)."""
    assert pos_scale <= 1 and neg_scale >= 1
    x_max = road_weight_matrix.shape[0] - 1
    y_max = road_weight_matrix.shape[1] - 1
    if pos_scale != 1:
        for xy in sv_pos_filled_grid:
            x, y = xy
            for radius in range(0, expand_radius + 1):
                for dx, dy in ring_xy_diff_dict[radius]:
                    if 0 <= x + dx <= x_max and 0 <= y + dy <= y_max and road_weight_matrix[x+dx, y+dy] < 1:
                        road_weight_matrix[x+dx, y+dy] = road_weight_matrix[x+dx, y+dy] * pos_scale
    if neg_scale != 1:
        for xy in sv_neg_filled_grid:
            x, y = xy
            for radius in range(0, expand_radius + 1):
                for dx, dy in ring_xy_diff_dict[radius]:
                    if 0 <= x + dx <= x_max and 0 <= y + dy <= y_max and road_weight_matrix[x+dx, y+dy] < 1:
                        road_weight_matrix[x+dx, y+dy] = min(road_weight_matrix[x+dx, y+dy] * neg_scale, 1.0)
    return road_weight_matrix


def load_road_data(region):
    with open(join('data/road_info', region,
                   'way_coord_dict_processed.pickle'), 'rb') as f:
        way_coord_dict_sub = pickle.load(f)
    with open(join('data/road_info', region, 'way_tag_dict_local.pickle'),
              'rb') as f:
        way_tag_dict_sub = pickle.load(f)
    return way_coord_dict_sub, way_tag_dict_sub


def load_discretization_params(region):
    # discretization parameters
    with open('results/' + region + '/discretization_parameters.pickle', 'rb') as f:
        discretization_params = pickle.load(f)
    lat_s = discretization_params['lat_s']
    lat_n = discretization_params['lat_n']
    lon_w = discretization_params['lon_w']
    lon_e = discretization_params['lon_e']
    dlat0 = discretization_params['dlat0']
    dlon0 = discretization_params['dlon0']
    return lat_s, lat_n, lon_w, lon_e, dlat0, dlon0


def load_result_data(region, pole_model):
    with open(join('results', region, pole_model, 'all_predicted_poles.pickle'), 'rb') as f:
        all_predicted_poles = pickle.load(f)
    # line prediction
    with open(join('results', region, 'line_info_merged.pickle'), 'rb') as f:
        sv2line = pickle.load(f)
    idx2sv_pos = []
    idx2sv_neg = []
    for i in range(len(sv2line)):
        coord, line_list = sv2line[i]
        coord = tuple(coord)
        if len(line_list) > 0:
            idx2sv_pos.append(coord)
        else:
            idx2sv_neg.append(coord)
    return all_predicted_poles, sv2line, idx2sv_pos, idx2sv_neg


def discretize(sh2, way_coord_dict_sub, all_predicted_poles, idx2sv_pos, idx2sv_neg):
    # discretize roads
    road_grid = Grid(sh2)
    road_grid.construct_from_way_coord_dict(way_coord_dict_sub)

    # discretize predicted poles
    all_predicted_pole_list = []
    for idx in all_predicted_poles:
        lat, lon = all_predicted_poles[idx]
        all_predicted_pole_list.append((lat, lon, idx))

    all_pole_grid = Grid(sh2)
    all_pole_grid.construct_from_coord_list(all_predicted_pole_list)

    # discretize street view points
    sv_grid = Grid(sh2)
    sv_grid.construct_from_coord_list(idx2sv_pos)

    sv_grid_neg = Grid(sh2)
    sv_grid_neg.construct_from_coord_list(idx2sv_neg)
    return road_grid, all_pole_grid, sv_grid, sv_grid_neg, all_predicted_pole_list


def attach_poles_to_roads(ring_xy_diff_dict, road_grid, all_pole_grid):
    # attached poles
    max_search_radius = 5  # in number of grids, instead of meters!!
    unattached = 0
    attached_pole_dict = {}
    attached_pole_dict_inversed = {}
    for pole_point in all_pole_grid.filled_dict:
        nearest_road_point = find_nearest_road_point(ring_xy_diff_dict, pole_point, road_grid.filled_dict,
                                                     max_search_radius=max_search_radius)
        if nearest_road_point == -1:  # unattached
            unattached += 1
        else:
            attached_pole_dict[pole_point] = nearest_road_point
            attached_pole_dict_inversed[nearest_road_point] = pole_point
    print('# poles unattached to the road:', unattached)
    return attached_pole_dict, attached_pole_dict_inversed


def get_road_matrix(sh2, road_grid, way_tag_dict_sub):
    x_max = sh2.x_max
    y_max = sh2.y_max
    road_weight_matrix_original = np.ones((x_max + 1, y_max + 1))
    for xy in tqdm(road_grid.filled_dict):
        x, y = xy
        w = 1.0
        for way_id in road_grid.filled_dict[xy]:
            if not len(way_tag_dict_sub[way_id]) == 0:
                for tag in way_tag_dict_sub[way_id][0]:
                    if tag in weight_scheme:
                        w = min(w, weight_scheme[tag])
        road_weight_matrix_original[x, y] = w
    return road_weight_matrix_original


def get_target_and_origin_matrix(start_lat, start_lon, x_max, y_max, all_predicted_pole_list, all_pole_grid,
                                 attached_pole_dict, attached=False):
    dist_list = []
    for i, p in enumerate(all_predicted_pole_list):
        lat, lon, idx = p
        dist = calculate_dist(lat, lon, start_lat, start_lon)
        dist_list.append((i, dist, idx))
    dist_list = sorted(dist_list, key=lambda x: x[1])
    start_pole = dist_list[0][2]

    for xy in all_pole_grid.filled_dict:
        if start_pole in all_pole_grid.filled_dict[xy]:
            start_x, start_y = xy
            break
    print('start_x: ', start_x, 'start_y', start_y)

    if attached:
        ############# Attached case #############
        target_matrix = np.zeros((x_max + 1, y_max + 1))
        for xy in all_pole_grid.filled_dict:
            x, y = xy
            v = sorted(all_pole_grid.filled_dict[xy])[0] + 5
            if xy in attached_pole_dict:
                x, y = attached_pole_dict[xy]
            target_matrix[x, y] = v

        # configure start point and construct origin matrix
        start_x_road, start_y_road = attached_pole_dict[(start_x, start_y)]
        origin_matrix = np.zeros((x_max + 1, y_max + 1))
        origin_matrix[start_x, start_y] = target_matrix[start_x_road, start_y_road]
        target_matrix[start_x, start_y] = 0
    else:
        ############# Unattached case #############
        target_matrix = np.zeros((x_max + 1, y_max + 1))
        for xy in all_pole_grid.filled_dict:
            x, y = xy
            v = sorted(all_pole_grid.filled_dict[xy])[0] + 5
            target_matrix[x, y] = v

        # configure start point and construct origin matrix
        origin_matrix = np.zeros((x_max + 1, y_max + 1))
        origin_matrix[start_x, start_y] = target_matrix[start_x, start_y]
        target_matrix[start_x, start_y] = 0
    return target_matrix, origin_matrix


def get_edge_set(results, all_pole_grid, ring_xy_diff_dict, all_predicted_pole_list):
    edge_set = set()
    for xy1, xy2 in results['edges']:
        i1_set = all_pole_grid.filled_dict[xy1]
        if xy2 not in all_pole_grid.filled_dict:
            nearest = find_nearest_road_point(ring_xy_diff_dict, xy2, all_pole_grid.filled_dict, max_search_radius=1)
            i2_set = all_pole_grid.filled_dict[nearest]
        else:
            i2_set = all_pole_grid.filled_dict[xy2]
        for i1 in i1_set:
            for i2 in i2_set:
                pid1 = all_predicted_pole_list[i1][2]
                pid2 = all_predicted_pole_list[i2][2]
                if type(pid1) == type(pid2):
                    edge = sorted((pid1, pid2))
                elif type(pid1) == str and type(pid2) == int:
                    edge = (pid2, pid1)
                else:
                    edge = (pid1, pid2)
                edge_set.add(tuple(edge))
    return edge_set

if __name__ == '__main__':
    # ring (used for obtaining all cells within a certain radius of the center on a raster map)
    with open('data/ring_xy_diff_dict_200.pickle', 'rb') as f:
        ring_xy_diff_dict = pickle.load(f)

    for region in region_list:
        # load road data
        way_coord_dict_sub, way_tag_dict_sub = load_road_data(region)
        print('# roads:', len(way_coord_dict_sub))

        lat_s, lat_n, lon_w, lon_e, dlat0, dlon0 = load_discretization_params(region)

        for pole_model in pole_model_list:
            print('-' * 20 + ' Region: ' + region + ', Pole model: ' + pole_model + ' ' + '-' * 20)

            # load results from previous steps
            all_predicted_poles, sv2line, idx2sv_pos, idx2sv_neg = load_result_data(region, pole_model)
            print('# predicted poles in total (detected + inserted):', len(all_predicted_poles))
            print('# street view points', len(sv2line))
            print('# positive street view points', len(idx2sv_pos))
            print('# negative street view points', len(idx2sv_neg))

            sh2 = SpatialHashing(unit=2, lat_s=lat_s, lon_w=lon_w, lat_n=lat_n, lon_e=lon_e,
                                 dlat0=dlat0, dlon0=dlon0)

            # discretize roads, poles, and street view points
            road_grid, all_pole_grid, sv_grid, sv_grid_neg, all_predicted_pole_list = discretize(sh2,
                                                                                                 way_coord_dict_sub,
                                                                                                 all_predicted_poles,
                                                                                                 idx2sv_pos, idx2sv_neg)

            # attach poles to nearest roads
            attached_pole_dict, _ = attach_poles_to_roads(ring_xy_diff_dict, road_grid, all_pole_grid)

            # construct road_weight_matrix
            road_weight_matrix = get_road_matrix(sh2, road_grid, way_tag_dict_sub)

            road_weight_matrix = scale_weight_with_street_view(ring_xy_diff_dict, road_weight_matrix, sv_grid.filled_dict,
                                                               sv_grid_neg.filled_dict, expand_radius=5)

            start_lat, start_lon = start_lat_lon_dict[region]

            target_matrix, origin_matrix = get_target_and_origin_matrix(start_lat, start_lon, sh2.x_max, sh2.y_max,
                                                                        all_predicted_pole_list, all_pole_grid,
                                                                        attached_pole_dict, attached=False)

            origin_matrix_update = origin_matrix.copy()
            target_matrix_update = target_matrix.copy()

            results = seek(origins=origin_matrix_update,
                           targets=target_matrix_update,
                           weights=road_weight_matrix,
                           path_handling='assimilate',
                           debug=True,
                           film=False,
                           frame_dirname=None,
                           frame_rate=100000,
                           early_stop=True)

            edge_set = get_edge_set(results, all_pole_grid, ring_xy_diff_dict, all_predicted_pole_list)

            print('number of edges connected by dijkstra:', len(edge_set))

            with open(join('results', region, pole_model, 'dijkstra_edge_set_assimilate_unattached_noscaling.pickle'),
                      'wb') as f:
                pickle.dump(edge_set, f)

            if save_raw_results:
                with open(join('results', region, pole_model,
                               'dijkstra_raw_results_assimilate_unattached_noscaling.pickle'),
                          'wb') as f:
                    pickle.dump(results, f)

    print('Done.')
