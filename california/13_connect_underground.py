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
import copy
import geojson
from os.path import join, exists

from utils import *
from dijkstra import *
from evaluation_utils import *

"""
This script is for predicting the underground grid on top of the predicted overhead grid
by leveraging the modified Dijkstra's algorithm. This algorithm greedily seeks the paths 
with minimum total weight to connect all buildings that cannot be reached by predicted 
overhead grid within a certain distance.  Each cell in the raster is assigned with a weight.
The Dijkstra's algorithm is adapted from:
https://github.com/facebookresearch/many-to-many-dijkstra
"""

pole_model_list = ['ori0.5']
link_model_list = ['GB3ft2'] # 'GB3ft2': Gradient Boosting model
region_list = ['SanCarlos', 'Newark', 'SantaCruz', 'Yuba', 'Monterey', 'Salinas']

# benchmark_version: 'corrected_2' or 'augmented_2'
# 'corrected_2' is the utility-owned distribution grid without supplemented poles and edges
# 'augmented_2' is the utility-owned distribution grid with supplemented poles and edges added
benchmark_version = 'augmented_2'

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

augmented = 'augmented' in benchmark_version
data_dir = 'data'
performance_dict_save_path = 'results/aggregated/california_underground_perf_ori0.5_' + benchmark_version + '.pickle'
if_evaluate = True
save_raw_results = False

# model parameters
aboveground_line_dilate_radius = 35

weight_scheme = {'motorway': 1 / 10, 'motorway_link': 1 / 10,
                     'primary': 1 / 8, 'primary_link': 1 / 8,
                     'secondary': 1 / 7, 'secondary_link': 1 / 7,
                     'tertiary': 1 / 6, 'tertiary_link': 1 / 6,
                     'unclassified': 1 / 5,
                     'residential': 1 / 4, 'living_street': 1 / 4, 'cycleway': 1 / 4,
                     'service': 1 / 3, 'footway': 1 / 3, 'track': 1 / 3, 'pedestrian': 1 / 3,
                     'path': 1 / 2, 'highway': 2 / 3
                     }

# evaluation parameters
evaluate_with_discretization_gridsize = 40
evaluate_with_dilation_radius = 10  # in # grids, *2 in meters

fips_list_dict = {
    'SanCarlos': [6081609100, 6081609201, 6081609202, 6081609300, 6081609400, 6081609500, 6081609601, 6081609602, 6081609603],
    'Newark': [6001444100, 6001444200, 6001444301, 6001444400, 6001444500, 6001444601],
    'SantaCruz': [6087121300, 6087121401, 6087121402, 6087121403, 6087121700],
    'Yuba': [6101050201, 6101050202, 6101050301, 6101050302],
    'Monterey': [6053012401, 6053012402, 6053012302, 6053012200, 6053012100, 6053012000],
    'Salinas': [6053000800, 6053000600, 6053000701, 6053000702],
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


def dilate_line(ring_xy_diff_dict, matrix, radius):
    """Dilate the lines, which can be used to make them cover part of the points."""
    matrix_new = matrix.copy()
    m, n = matrix.shape
    for x in tqdm(range(m)):
        for y in range(n):
            if matrix[x, y] > 0:
                for radius in range(radius + 1):
                    for dx, dy in ring_xy_diff_dict[radius]:
                        if (0 <= x+dx <= m - 1) and (0 <= y+dy <= n - 1):
                            matrix_new[x+dx, y+dy] = matrix[x, y]
    return matrix_new


def load_ground_truth_data(region, benchmark_version):
    with open('ground_truth/' + region + '/pge_edge_coord_dict.pickle', 'rb') as f:
        pge_edge_coord_dict = pickle.load(f)
    if augmented:
        with open('ground_truth/' + region + '/ground_truth_poles_' + benchmark_version + '.pickle', 'rb') as f:
            true_poles_aug = pickle.load(f)
        with open('ground_truth/' + region + '/ground_truth_connections_' + benchmark_version + '.pickle', 'rb') as f:
            true_edge_set_aug = pickle.load(f)
        with open('ground_truth/' + region + '/ground_truth_connections_' + benchmark_version.replace('augmented', 'corrected') + '.pickle',
                  'rb') as f:
            true_edge_set = pickle.load(f)
        return pge_edge_coord_dict, true_poles_aug, true_edge_set_aug, true_edge_set
    else:
        return pge_edge_coord_dict


def load_result_data(region, pole_model, link_model_alias):
    # predicted poles
    with open(join('results', region, pole_model, 'all_predicted_poles.pickle'), 'rb') as f:
        all_predicted_poles = pickle.load(f)
    # link prediction data
    with open(join('results', region, pole_model, link_model_alias, 'predicted_binary_classes.pickle'), 'rb') as f:
        y_test_pred = pickle.load(f)
    with open(join('results', region, pole_model, link_model_alias, 'candidate_pair_list.pickle'), 'rb') as f:
        candidate_pair_list = pickle.load(f)
    return all_predicted_poles, y_test_pred, candidate_pair_list


def load_road_and_building_data(region):
    with open(
            join('data/road_info', region, 'way_coord_dict_processed.pickle'),
            'rb') as f:
        way_coord_dict_sub = pickle.load(f)
    with open(join('data/road_info', region, 'way_tag_dict_local.pickle'),
              'rb') as f:
        way_tag_dict_sub = pickle.load(f)
    with open(join(data_dir, 'building_info/building_list_' + region + '.pickle'), 'rb') as f:
        building_list = pickle.load(f)
    return way_coord_dict_sub, way_tag_dict_sub, building_list


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
    return lat_s, lat_n, lon_w, lon_e, dlat0, dlon0, discretization_params


def discretize(sh2, candidate_pair_list, y_test_pred, all_predicted_poles, way_coord_dict_sub, building_list):
    predicted_edge_dict = {}
    for i, p in enumerate(candidate_pair_list):
        if y_test_pred[i]:
            pid1, pid2 = p
            lat1, lon1 = all_predicted_poles[pid1]
            lat2, lon2 = all_predicted_poles[pid2]
            predicted_edge_dict[p] = [(lat1, lon1), (lat2, lon2)]
    # discretize road
    road_grid = Grid(sh2)
    road_grid.construct_from_way_coord_dict(way_coord_dict_sub)
    # discretize predicted edges
    predicted_edge_grid = Grid(sh2)
    predicted_edge_grid.construct_from_way_coord_dict(predicted_edge_dict)
    predicted_edge_mat = predicted_edge_grid.get_grid_matrix()
    # discretize building
    building_grid = Grid(sh2)
    building_grid.construct_from_coord_list(building_list)
    building_matrix = building_grid.get_grid_matrix()
    return road_grid, predicted_edge_grid, building_matrix, predicted_edge_mat


def get_road_matrix(x_max, y_max, road_grid, way_tag_dict_sub):
    road_weight_matrix_original = np.ones((x_max + 1, y_max + 1))
    for xy in road_grid.filled_dict:
        x, y = xy
        w = 1.0
        for way_id in road_grid.filled_dict[xy]:
            if not len(way_tag_dict_sub[way_id]) == 0:
                for tag in way_tag_dict_sub[way_id][0]:
                    if tag in weight_scheme:
                        w = min(w, weight_scheme[tag])
        road_weight_matrix_original[x, y] = w
    return road_weight_matrix_original


def attach_buildings_to_roads(x_max, y_max, ring_xy_diff_dict, road_grid, building_matrix_rest):
    # attach buildings to the roads
    max_search_radius = 50  # in number of grids, instead of meters!!
    unattached = 0
    attached_building_dict = {}
    attached_building_dict_inversed = {}
    building_matrix_rest_attached = np.zeros((x_max + 1, y_max + 1))
    building_locs = np.where(building_matrix_rest)
    for i, x in enumerate(building_locs[0]):
        y = building_locs[1][i]
        nearest_road_point = find_nearest_road_point(ring_xy_diff_dict, (x, y), road_grid.filled_dict,
                                                     max_search_radius=max_search_radius)
        if nearest_road_point == -1:  # unattached
            unattached += 1
        else:
            attached_building_dict[(x, y)] = nearest_road_point
            attached_building_dict_inversed[nearest_road_point] = (x, y)
            building_matrix_rest_attached[nearest_road_point] = 1
    print('# buildings unattached to the road:', unattached)
    return building_matrix_rest_attached


def evaluate_link_prediction(region, benchmark_version, ring_xy_diff_dict, discretization_params, sh2, predicted_grid):
    if augmented:
        pge_edge_coord_dict, true_poles_aug, true_edge_set_aug, true_edge_set = load_ground_truth_data(region, benchmark_version)
    else:
        pge_edge_coord_dict = load_ground_truth_data(region, benchmark_version)
    # discretize PG&E ground truth grid
    pge_line_grid = Grid(sh2)
    pge_line_grid.construct_from_way_coord_dict(pge_edge_coord_dict)
    if augmented:
        count = 0
        augmented_edge_coord_dict = {}
        for edge in true_edge_set_aug:
            if edge not in true_edge_set:
                lat1, lon1 = true_poles_aug[edge[0]]
                lat2, lon2 = true_poles_aug[edge[1]]
                augmented_edge_coord_dict['aug_' + str(count)] = [[lat1, lon1], [lat2, lon2]]
                count += 1
        pge_line_grid.construct_from_way_coord_dict(augmented_edge_coord_dict)
    pge_line_mat = pge_line_grid.get_grid_matrix()

    # evaluate with path dilation
    predicted_grid_filled_dict = {}
    locs = np.where(predicted_grid > 0)
    for i, x in enumerate(locs[0]):
        y = locs[1][i]
        predicted_grid_filled_dict[(x, y)] = {1}

    precision_dil, recall_dil, f1_dil, predicted_grid_dil, pge_line_grid_dil = evaluate_with_dilation_dijkstra(
        evaluate_with_dilation_radius, sh2.x_max, sh2.y_max, ring_xy_diff_dict, predicted_grid_filled_dict,
        pge_line_grid, return_matrix=True)
    print('similarity with path dilation @radius = ' + str(evaluate_with_dilation_radius) + ':')
    print('    precision:', precision_dil, 'recall:', recall_dil, 'F1', f1_dil)
    dilation_perf = {'precision': precision_dil, 'recall': recall_dil, 'F1': f1_dil,
                     # 'visualization': vis_dil
                     }

    # evaluation with discretization
    pge_pixel_list = []
    locs = np.where(pge_line_mat > 0)
    for i, x in enumerate(locs[0]):
        y = locs[1][i]
        lon = sh2.x2lon(x)
        lat = sh2.y2lat(y)
        pge_pixel_list.append((lat, lon))

    predicted_grid_pixel_list = []
    locs = np.where(predicted_grid > 0)
    for i, x in enumerate(locs[0]):
        y = locs[1][i]
        lon = sh2.x2lon(x)
        lat = sh2.y2lat(y)
        predicted_grid_pixel_list.append((lat, lon))

    precision_dis, recall_dis, f1_dis, IOU_dis, grid_mat_predicted, grid_mat_true = evaluate_with_discretization_dijkstra(
        evaluate_with_discretization_gridsize, discretization_params, pge_pixel_list,
        predicted_grid_pixel_list, return_matrix=True)

    print('similarity with discretization @gridsize = ' + str(evaluate_with_discretization_gridsize) + ':')
    print('    precision:', precision_dis, 'recall:', recall_dis, 'F1', f1_dis, 'IoU: ', IOU_dis)
    discretization_perf = {'precision': precision_dis, 'recall': recall_dis, 'F1': f1_dis, 'IoU': IOU_dis,
                           # 'visualization': vis_dis
                           }

    return discretization_perf, dilation_perf


if __name__ == '__main__':
    # ring
    with open('data/ring_xy_diff_dict_200.pickle', 'rb') as f:
        ring_xy_diff_dict = pickle.load(f)

    performance_dict = {x: {y: {} for y in pole_model_list} for x in region_list}

    for region in region_list:
        # load road data
        way_coord_dict_sub, way_tag_dict_sub, building_list = load_road_and_building_data(region)
        # load discretization parameters
        lat_s, lat_n, lon_w, lon_e, dlat0, dlon0, discretization_params = load_discretization_params(region)
        sh2 = SpatialHashing(unit=2, lat_s=lat_s, lon_w=lon_w, lat_n=lat_n, lon_e=lon_e,
                             dlat0=dlat0, dlon0=dlon0)
        x_max = sh2.x_max
        y_max = sh2.y_max

        for pole_model in pole_model_list:
            print('-' * 30 + ' Region: ' + region + ', Pole model: ' + pole_model + ' ' + '-' * 30)
            # load results of previous step
            for link_model_alias in link_model_list:
                print('-' * 20 + ' Link prediction model: ' + link_model_alias + ' ' + '-' * 20)
                all_predicted_poles, y_test_pred, candidate_pair_list = load_result_data(region, pole_model,
                                                                                         link_model_alias)

                # discretize roads, predicted grids, and buildings
                road_grid, predicted_edge_grid, building_matrix, predicted_edge_mat = discretize(sh2,
                                                                                                 candidate_pair_list,
                                                                                                 y_test_pred,
                                                                                                 all_predicted_poles,
                                                                                                 way_coord_dict_sub,
                                                                                                 building_list)
                # construct weight matrix for roads
                road_weight_matrix_original = get_road_matrix(x_max, y_max, road_grid, way_tag_dict_sub)

                # cover buildings with predicted distribution lines
                dilated_paths = dilate_line(ring_xy_diff_dict, predicted_edge_mat,
                                            radius=aboveground_line_dilate_radius)
                # buildings that are not covered by detected lines
                building_matrix_rest = building_matrix * (dilated_paths == 0)
                building_matrix_rest_attached = attach_buildings_to_roads(x_max, y_max, ring_xy_diff_dict, road_grid,
                                                                          building_matrix_rest)

                target_matrix_new = building_matrix_rest_attached.copy()
                origin_matrix_new = predicted_edge_mat.copy()
                weight_matrix_new = road_weight_matrix_original.copy()

                results_building = seek(origins=origin_matrix_new,
                                        targets=target_matrix_new,
                                        weights=weight_matrix_new,
                                        path_handling='link',
                                        debug=True,
                                        film=False,
                                        frame_dirname=None,
                                        frame_rate=100000,
                                        early_stop=True)

                predicted_grid = np.clip(results_building['paths'] + predicted_edge_mat, 0, 1)
                
                if save_raw_results:
                    with open(join('results', region, pole_model, link_model_alias, 
                                   'underground_dijkstra_raw_results_link.pickle'),
                              'wb') as f:
                        pickle.dump([results_building['paths'], predicted_edge_mat], f)

                if if_evaluate:
                    discretization_perf, dilation_perf = evaluate_link_prediction(region, benchmark_version,
                                                                                  ring_xy_diff_dict,
                                                                                  discretization_params,
                                                                                  sh2, predicted_grid)

                    performance_dict[region][pole_model][link_model_alias] = {
                        'discretization': discretization_perf,
                        'dilation': dilation_perf
                    }
                    with open(performance_dict_save_path, 'wb') as f:
                        pickle.dump(performance_dict, f)

    print('Done.')
