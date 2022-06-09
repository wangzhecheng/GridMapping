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
import geojson
import copy
from os.path import join, exists
from shapely.geometry import Polygon, Point
from shapely.ops import cascaded_union
from utils import *

"""
This scripts is for attaching detected poles to roads and potentially inserting additional poles,
and predict the geospatial map of utility poles.
"""
pole_model_list = ['ori0.5']
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

data_dir = 'data'
performance_dict_save_path = 'results/aggregated/california_pole_perf_ori0.5_' + benchmark_version + '.pickle'
if_evaluate = True

# polyline and matching model parameters
attach_pole_to_intersection_threshold = 25
pole_insertion_threshold = 70
pole_insertion_max_partitions = 5
pairwise_matching_threshold = 25

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


def remove_close_coord(ref_coord_dict, target_coord_dict, distance_threshold, spatial_hashing):
    """
    Given ref_coord_dict, remove those points in target_coord_dict which is close enough (< distance_threshold)
    to any coordinate in the ref_coord_dict.

    ** Inputs:
    ref_coord_dict: a dict of idx: (lat, lon) for reference.
    target_coord_dict: a dict of idx: (lat, lon) for point removal.
    distance_threshold: distance threshold (meter) for considering a coordinate pair as candidate
    spatial_hasing: an instance of SpatialHashing

    ** Return:
    new_target_coord_dict: a dict of idx: (lat, lon) after point removal.
    """
    idx2hv_ref, hv2idx_ref = spatial_hashing.hash_coordinates(ref_coord_dict)
    idx2hv_tar, hv2idx_tar = spatial_hashing.hash_coordinates(target_coord_dict)
    remove_index_set = set()
    for idx1 in target_coord_dict:
        x, y = idx2hv_tar[idx1]
        lat1, lon1 = target_coord_dict[idx1]
        for dx, dy in [(0, 0), (-1, 0), (1, 0), (0, 1), (0, -1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            if (x + dx, y + dy) in hv2idx_ref:
                for idx2 in hv2idx_ref[(x + dx, y + dy)]:
                    lat2, lon2 = ref_coord_dict[idx2]
                    d = calculate_dist(lat1, lon1, lat2, lon2)
                    if d < distance_threshold:
                        remove_index_set.add(idx1)
                        break
    new_target_coord_dict = copy.deepcopy(target_coord_dict)
    for idx in remove_index_set:
        del new_target_coord_dict[idx]
    return new_target_coord_dict


def pairwise_sort_matching(coord_dict_1, coord_dict_2, distance_threshold, spatial_hashing):
    """
    Given 2 lists of coordinates, generate the list of coordinate pair (one from each lists) within
    the given distance threshold, sorted in ascending distance order. And match the coordinates from
    these 2 lists.

    ** Inputs:
    coord_dict_1: a dict of idx: (lat, lon)
    coord_dict_2: a dict of idx: (lat, lon)
    distance_threshold: distance threshold (meter) for considering a coordinate pair as candidate
    spatial_hasing: an instance of SpatialHashing

    ** Return:
    matching: a list of (index1, index2, distance) that are matched, where index1 is from coord_list_1 and
              index2 is from coord_list_2
    unmatched_index_1: a dict of unmatched indices of coord_list_1
    unmatched_index_2: a dict of unmatched indices of coord_list_2
    """
    idx2hv_1, hv2idx_1 = spatial_hashing.hash_coordinates(coord_dict_1)
    idx2hv_2, hv2idx_2 = spatial_hashing.hash_coordinates(coord_dict_2)
    coord_pairs = []

    for idx1 in coord_dict_1:
        x, y = idx2hv_1[idx1]
        lat1, lon1 = coord_dict_1[idx1]
        for dx, dy in [(0, 0), (-1, 0), (1, 0), (0, 1), (0, -1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            if (x + dx, y + dy) in hv2idx_2:
                for idx2 in hv2idx_2[(x + dx, y + dy)]:
                    lat2, lon2 = coord_dict_2[idx2]
                    d = calculate_dist(lat1, lon1, lat2, lon2)
                    if d < distance_threshold:
                        coord_pairs.append((idx1, idx2, d))
    coord_pairs = sorted(coord_pairs, key=lambda x: x[2])
    unmatched_index_1 = copy.deepcopy(coord_dict_1)
    unmatched_index_2 = copy.deepcopy(coord_dict_2)
    matching = []
    for idx1, idx2, d in coord_pairs:
        if idx1 in unmatched_index_1 and idx2 in unmatched_index_2:
            matching.append((idx1, idx2, d))
            del unmatched_index_1[idx1]
            del unmatched_index_2[idx2]
    return matching, unmatched_index_1, unmatched_index_2

def load_road_data(region):
    # load road data
    with open(join('data/road_info', region,
                   'way_coord_dict_processed.pickle'), 'rb') as f:
        way_coord_dict_sub_sub_serialized = pickle.load(f)
    with open(join('data/road_info', region,
                   'intersection_dict_processed.pickle'), 'rb') as f:
        intersection_coords = pickle.load(f)
    with open(join('data/road_info', region, 'way_tag_dict_local.pickle'),
              'rb') as f:
        way_tag_dict_sub_sub = pickle.load(f)
    return way_coord_dict_sub_sub_serialized, intersection_coords, way_tag_dict_sub_sub


def load_result_data(region, pole_model):
    with open(join('results', region, 'line_info_merged.pickle'), 'rb') as f:
        sv2line = pickle.load(f)
    with open(join('results', region, pole_model, 'pole_locations.pickle'), 'rb') as f:
        poles = pickle.load(f)
    with open(join('results', region, pole_model, 'pole_attached_GSVs.pickle'), 'rb') as f:
        pole2sv = pickle.load(f)
    assert len(pole2sv) == len(poles)
    return sv2line, poles, pole2sv


def get_bound_of_boundary(fips_list_dict, boundaries_raw, region):
    boundaries_keys = fips_list_dict[region]
    boundaries = []
    for key in boundaries_keys:
        boundaries.append(boundaries_raw[str.format('%011d' % key)].tolist())

    bd = []
    for boundary in boundaries:
        bd.extend(boundary)
    lat_min, lon_min = np.min(bd, axis=0)
    lat_max, lon_max = np.max(bd, axis=0)
    return lat_min, lon_min, lat_max, lon_max


def process_detection_data(sv2line, poles, pole2sv):
    sv2idx = {}  # key: (lat, lon) of GSV, value: GSV index
    idx2sv = []  # list of (lat, lon) of GSV
    idx2line = []  # list of line info list
    for i in range(len(sv2line)):
        coord, line_list = sv2line[i]
        coord = tuple(coord)
        sv2idx[coord] = i
        idx2sv.append(coord)
        idx2line.append(line_list)

    pole2svidx = []  # GSV indices corresponding to each pole (pole index)
    idx2poles = [set() for i in range(len(idx2sv))]  # detected pole(s) (pole index) for each GSV (GSV index)
    for i in range(len(pole2sv)):
        idx_list = []
        for x in pole2sv[i]:
            x = tuple(x)
            idx = sv2idx[x]
            idx_list.append(idx)
            idx2poles[idx].add(i)
        pole2svidx.append(list(set(idx_list)))
    detected_pole_dict = {i: (poles[i][0], poles[i][1]) for i in
                          range(len(poles))}  # key: pole index, value: pole (lat, lon)
    assert len(idx2sv) == len(sv2line) and len(idx2line) == len(sv2line) and len(idx2poles) == len(sv2line)
    assert len(pole2svidx) == len(pole2sv)
    return sv2idx, idx2sv, idx2line, pole2svidx, idx2poles, detected_pole_dict


def get_disaggregated_way_coord_dict_and_bounds(way_coord_dict_sub_sub_serialized):
    # disaggregate road into line segments and find the boundaries
    lat_min_way = 1e3
    lat_max_way = -1e3
    lon_min_way = 1e3
    lon_max_way = -1e3
    way_coord_dict_disaggregated = {}
    for way_id in way_coord_dict_sub_sub_serialized:
        coord_list = way_coord_dict_sub_sub_serialized[way_id]
        lat_max_way = max(np.max(coord_list, axis=0)[0], lat_max_way)
        lon_max_way = max(np.max(coord_list, axis=0)[1], lon_max_way)
        lat_min_way = min(np.min(coord_list, axis=0)[0], lat_min_way)
        lon_min_way = min(np.min(coord_list, axis=0)[1], lon_min_way)
        for i in range(0, len(coord_list) - 1):
            coord_list_sub = coord_list[i: i + 2]
            way_coord_dict_disaggregated[str(way_id) + '_' + str(i)] = coord_list_sub  # i is the line id (lid)
    return lat_max_way, lon_max_way, lat_min_way, lon_min_way, way_coord_dict_disaggregated


def get_discretization_params(lat_min, lat_max, lon_min, lon_max, lat_min_way, lat_max_way, lon_min_way, lon_max_way, idx2sv):
    # dicretization
    dlat0 = 8.9831528e-6  # dy=1m
    dlon0 = dlat0 / np.cos(np.deg2rad((lat_max + lat_min) / 2))  # dx=1m
    # boundary for discretization
    lat_s = min(lat_min, lat_min_way) - 100 * dlat0
    lat_n = max(lat_max, lat_max_way) + 100 * dlat0
    lon_w = min(lon_min, lon_min_way) - 100 * dlon0
    lon_e = max(lon_max, lon_max_way) + 100 * dlon0
    # print(lat_s, lat_n, lon_w, lon_e)

    discretization_params = {
        'lat_s': lat_s, 'lat_n': lat_n, 'lon_w': lon_w, 'lon_e': lon_e, 'dlat0': dlat0, 'dlon0': dlon0
    }
    discretization_params_save_path = 'results/' + region + '/discretization_parameters.pickle'
    if not exists(discretization_params_save_path):
        with open(discretization_params_save_path, 'wb') as f:
            pickle.dump(discretization_params, f)
    return lat_s, lat_n, lon_w, lon_e, dlat0, dlon0


def discretize(sh2, way_coord_dict_disaggregated, poles, idx2sv):
    road_grid = Grid(sh2)
    road_grid.construct_from_way_coord_dict(way_coord_dict_disaggregated)
    pole_grid = Grid(sh2)
    pole_grid.construct_from_coord_list(poles)
    sv_grid = Grid(sh2)
    sv_grid.construct_from_coord_list(idx2sv)
    return road_grid, pole_grid, sv_grid


def attach_poles_and_street_views_to_roads(ring_xy_diff_dict, road_grid, pole_grid, sv_grid):
    # attached poles to roads
    max_search_radius = 10  # in number of grids, instead of meters!!
    unattached = 0
    attached_pole_dict = {}
    attached_pole_dict_inversed = {}
    for pole_point in pole_grid.filled_dict:
        nearest_road_point = find_nearest_road_point(ring_xy_diff_dict, pole_point, road_grid.filled_dict,
                                                     max_search_radius=max_search_radius)
        if nearest_road_point == -1:  # unattached
            unattached += 1
            print(poles[list(pole_grid.filled_dict[pole_point])[0]])
        else:
            attached_pole_dict[pole_point] = nearest_road_point
            attached_pole_dict_inversed[nearest_road_point] = pole_point
    print('# poles unattached to the road:', unattached)

    # attached street view points to roads
    max_search_radius = 10  # in number of grids, instead of meters!!
    unattached = 0
    attached_sv_dict = {}
    attached_sv_dict_inversed = {}
    for sv_point in sv_grid.filled_dict:
        nearest_road_point = find_nearest_road_point(ring_xy_diff_dict, sv_point, road_grid.filled_dict,
                                                     max_search_radius=max_search_radius)
        if nearest_road_point == -1:  # unattached
            unattached += 1
        else:
            attached_sv_dict[sv_point] = nearest_road_point
            attached_sv_dict_inversed[nearest_road_point] = sv_point
    print('# street view points unattached to the road:', unattached)
    return attached_pole_dict, attached_pole_dict_inversed, attached_sv_dict, attached_sv_dict_inversed


def polyline_modeling(sh2, way_coord_dict_sub_sub_serialized, way_tag_dict_sub_sub, intersection_coords,
                      attached_pole_dict, attached_sv_dict, road_grid, pole_grid, sv_grid):
    # create polyline models for poles, and register poles, street views, and intersections
    polyline_dict = {}
    for way_id in way_coord_dict_sub_sub_serialized:
        coord_list = way_coord_dict_sub_sub_serialized[way_id]
        if len(way_tag_dict_sub_sub[way_id]) == 0:
            way_type = 'none'
        else:
            way_type = way_tag_dict_sub_sub[way_id][0]
            way_type = [x for x in way_type if x != 'highway']
        polyline = PolyLine(way_id, coord_list, way_type)
        polyline_dict[way_id] = polyline

    # register intersections
    for coord in intersection_coords:
        points = set(intersection_coords[coord])
        for x1 in points:
            wid1, nid1 = x1.split('_')
            wid1 = int(wid1)
            nid1 = int(nid1)
            for x2 in points:
                if x1 == x2:
                    continue
                wid2, nid2 = x2.split('_')
                wid2 = int(wid2)
                nid2 = int(nid2)
                if not nid1 in polyline_dict[wid1].intersections:
                    polyline_dict[wid1].intersections[nid1] = []
                polyline_dict[wid1].intersections[nid1].append((wid2, nid2))

    # register poles
    for pole_point in attached_pole_dict:
        nearest_road_point = attached_pole_dict[pole_point]
        road_ids = road_grid.filled_dict[nearest_road_point]
        pole_ids = pole_grid.filled_dict[pole_point]
        way_id, lid = sorted(road_ids)[0].split('_')
        way_id = int(way_id)
        lid = int(lid)
        road_lat = sh2.y2lat(nearest_road_point[1])
        road_lon = sh2.x2lon(nearest_road_point[0])
        for pid in pole_ids:
            polyline_dict[way_id].register_pole_by_coord(pid, road_lat, road_lon, lid)

    # register street views
    for sv_point in attached_sv_dict:
        nearest_road_point = attached_sv_dict[sv_point]
        road_ids = road_grid.filled_dict[nearest_road_point]
        sv_ids = sv_grid.filled_dict[sv_point]
        way_id, lid = sorted(road_ids)[0].split('_')
        way_id = int(way_id)
        lid = int(lid)
        road_lat = sh2.y2lat(nearest_road_point[1])
        road_lon = sh2.x2lon(nearest_road_point[0])
        for sid in sv_ids:
            polyline_dict[way_id].register_street_view(sid, road_lat, road_lon, lid)

    # attach  poles to intersections and share the poles at intersections with all involved roads
    polyline_dict_updated = copy.deepcopy(polyline_dict)
    info_list = []
    for way_id in polyline_dict_updated:
        info = polyline_dict_updated[way_id].attach_pole_to_intersection(
            distance_threshold=attach_pole_to_intersection_threshold,
            mode='detected')
        info_list.append(info)

    for info in info_list:
        for neighbor_list, pid in info:
            for neighbor_way_id, nid in neighbor_list:
                polyline_dict_updated[neighbor_way_id].register_pole_by_nodeid(nid, pid, mode='detected')

    return polyline_dict_updated


def insert_virtual_poles(polyline_dict_updated, sh50, remove_close=True):
    # insert virtual poles
    for way_id in polyline_dict_updated:
        if len(polyline_dict_updated[way_id].detected_poles) > 0:
            polyline_dict_updated[way_id].insert_virtual_poles(poles, mode='detected',
                                                               distance_threshold=pole_insertion_threshold,
                                                               max_partitions=pole_insertion_max_partitions,
                                                               dist_mode="crow_flight")

    # get the set of virtual poles
    predicted_virtual_poles = dict()
    for way_id in tqdm(polyline_dict_updated):
        polyline = polyline_dict_updated[way_id]
        sorted_virtual_poles = sorted(list(polyline.virtual_poles.items()), key=lambda x: x[1])
        for i in range(len(sorted_virtual_poles)):
            pid1, length1 = sorted_virtual_poles[i]
            lat1, lon1 = polyline.length2coord(length1)
            predicted_virtual_poles[pid1] = (lat1, lon1)
    print('# inserted poles:', len(predicted_virtual_poles))

    if remove_close:
        # remove virtual poles that are too closed to detected poles and attach virtual poles to intersections
        predicted_virtual_poles_new = remove_close_coord(detected_pole_dict, predicted_virtual_poles, 20, sh50)
        print('# inserted poles after removing close ones:', len(predicted_virtual_poles_new))
        # update the polyline objects accordingly
        for way_id in polyline_dict_updated:
            delete_list = []
            for pid in polyline_dict_updated[way_id].virtual_poles:
                if pid not in predicted_virtual_poles_new:
                    delete_list.append(pid)
            for pid in delete_list:
                del polyline_dict_updated[way_id].virtual_poles[pid]
    else:
        predicted_virtual_poles_new = predicted_virtual_poles

    # Attach virtual pole to intersection
    info_list = []
    for way_id in polyline_dict_updated:
        info = polyline_dict_updated[way_id].attach_pole_to_intersection(
            distance_threshold=attach_pole_to_intersection_threshold,
            mode='virtual')
        info_list.append(info)
    for info in info_list:
        for neighbor_list, pid in info:
            for neighbor_way_id, nid in neighbor_list:
                polyline_dict_updated[neighbor_way_id].register_pole_by_nodeid(nid, pid, mode='virtual')

    return polyline_dict_updated, predicted_virtual_poles_new


def evaluate_pole_localization(region, benchmark_version, sh50, detected_pole_dict, predicted_virtual_poles_new):
    with open('ground_truth/' + region + '/ground_truth_poles_' + benchmark_version + '.pickle', 'rb') as f:
        true_poles = pickle.load(f)
    print('# ground-truth poles:', len(true_poles))

    matching_detected, unmatched_index_true, unmatched_index_detected = pairwise_sort_matching(true_poles,
                                                                                               detected_pole_dict,
                                                                                               pairwise_matching_threshold,
                                                                                               sh50)
    matching_virtual, unmatched_index_true_2, unmatched_index_virtual = pairwise_sort_matching(unmatched_index_true,
                                                                                               predicted_virtual_poles_new,
                                                                                               pairwise_matching_threshold,
                                                                                               sh50)
    if len(true_poles) > 0:
        recall = 1 - len(unmatched_index_true_2) / len(true_poles)  # detected rate
    else:
        recall = 1.0
    print('Pole localization recall:', recall)
    precision = 1 - (len(unmatched_index_detected) + len(unmatched_index_virtual)) / (
        len(predicted_virtual_poles_new) + len(detected_pole_dict))
    print('Pole localization precision:', precision)
    if precision + recall > 0:
        F1 = 2 * precision * recall / (precision + recall)
    else:
        F1 = 0.
    print('Pole localization F1 score:', F1)
    with open(join('results', region, pole_model, 'unmatched_detected_poles_' + benchmark_version + '.pickle'),
              'wb') as f:
        pickle.dump(unmatched_index_detected, f)
    with open(join('results', region, pole_model, 'unmatched_virtual_poles_' + benchmark_version + '.pickle'),
              'wb') as f:
        pickle.dump(unmatched_index_virtual, f)
    with open(join('results', region, pole_model, 'matched_detected_poles_' + benchmark_version + '.pickle'),
              'wb') as f:
        pickle.dump(matching_detected, f)
    with open(join('results', region, pole_model, 'matched_virtual_poles_' + benchmark_version + '.pickle'),
              'wb') as f:
        pickle.dump(matching_virtual, f)
    TP = len(matching_detected) + len(matching_virtual)
    FN = len(unmatched_index_true_2)
    FP = len(unmatched_index_detected) + len(unmatched_index_virtual)
    P = len(true_poles)
    return precision, recall, F1, TP, FN, FP, P

if __name__ == '__main__':
    # admin division boundaries
    locator = census_tract_locator.CensusTract(tract_data_dir=join(data_dir, 'tract_boundaries_us_for_locator.pickle'))
    boundaries_raw = locator.tract_data['tract_boundaries']

    # ring coordinates with certain radius when using discretization
    with open('data/ring_xy_diff_dict_200.pickle', 'rb') as f:
        ring_xy_diff_dict = pickle.load(f)

    performance_dict = {y: {x: {} for x in pole_model_list} for y in region_list}
    for region in region_list:
        # bounds of the target region
        lat_min, lon_min, lat_max, lon_max = get_bound_of_boundary(fips_list_dict, boundaries_raw, region)
        # load road network data
        way_coord_dict_sub_sub_serialized, intersection_coords, way_tag_dict_sub_sub = load_road_data(region)
        # get bounds of road network and disaggregated road coordinate dictionary
        lat_max_way, lon_max_way, lat_min_way, lon_min_way, way_coord_dict_disaggregated = \
            get_disaggregated_way_coord_dict_and_bounds(way_coord_dict_sub_sub_serialized)

        for pole_model in pole_model_list:
            print('-' * 20 + ' Region: ' + region + ', Pole model: ' + pole_model + ' ' + '-' * 20)
            # load results of previous step
            sv2line, poles, pole2sv = load_result_data(region, pole_model)
            print('# street view points:', len(sv2line))
            print('# detected poles', len(poles))

            sv2idx, idx2sv, idx2line, pole2svidx, idx2poles, detected_pole_dict = process_detection_data(sv2line, poles,
                                                                                                         pole2sv)

            # road modeling
            print('# roads:', len(way_coord_dict_sub_sub_serialized))
            print('# intersections:', len(intersection_coords))

            lat_s, lat_n, lon_w, lon_e, dlat0, dlon0 = get_discretization_params(lat_min, lat_max, lon_min, lon_max,
                                                                                 lat_min_way, lat_max_way, lon_min_way,
                                                                                 lon_max_way,
                                                                                 idx2sv)

            # spatial hashing with 2m as discretization step
            sh2 = SpatialHashing(unit=2, lat_s=lat_s, lon_w=lon_w, lat_n=lat_n, lon_e=lon_e, dlat0=dlat0, dlon0=dlon0)

            # discretize roads, poles, and street view points
            road_grid, pole_grid, sv_grid = discretize(sh2, way_coord_dict_disaggregated, poles, idx2sv)

            # attached poles and street view points to nearest roads
            attached_pole_dict, _, attached_sv_dict, _ = attach_poles_and_street_views_to_roads(ring_xy_diff_dict,
                                                                                                road_grid,
                                                                                                pole_grid, sv_grid)

            # construct the polyline model for each road, register poles and street views, and deal with intersections
            polyline_dict_updated = polyline_modeling(sh2, way_coord_dict_sub_sub_serialized, way_tag_dict_sub_sub,
                                                      intersection_coords, attached_pole_dict, attached_sv_dict,
                                                      road_grid, pole_grid, sv_grid)

            sh50 = SpatialHashing(unit=50, lat_s=lat_s, lon_w=lon_w, lat_n=lat_n, lon_e=lon_e, dlat0=dlat0, dlon0=dlon0)

            # insert virtual poles between two poles that are two far apart
            polyline_dict_updated, predicted_virtual_poles_new = insert_virtual_poles(polyline_dict_updated, sh50,
                                                                                      remove_close=True)

            with open(join('results', region, pole_model, 'road_polyline_dict.pickle'), 'wb') as f:
                pickle.dump(polyline_dict_updated, f)

            all_predicted_poles = dict()  # including detected and inserted poles
            for pid in detected_pole_dict:
                all_predicted_poles[pid] = detected_pole_dict[pid]
            for pid in predicted_virtual_poles_new:
                all_predicted_poles[pid] = predicted_virtual_poles_new[pid]
            print('# predicted poles in total (detected + inserted):', len(all_predicted_poles))

            with open(join('results', region, pole_model, 'all_predicted_poles.pickle'), 'wb') as f:
                pickle.dump(all_predicted_poles, f)

            if if_evaluate:
                precision, recall, F1, TP, FN, FP, P = evaluate_pole_localization(region, benchmark_version, sh50,
                                                                                  detected_pole_dict,
                                                                                  predicted_virtual_poles_new)
                performance_dict[region][pole_model] = {'precision': precision, 'recall': recall, 'F1': F1,
                                                        'num_detected_poles': len(poles),
                                                        'num_inserted_poles': len(predicted_virtual_poles_new),
                                                        'num_predicted_poles': len(all_predicted_poles),
                                                        'num_true_poles': P,
                                                        'TP': TP, 'FP': FP, 'FN': FN}
                with open(performance_dict_save_path, 'wb') as f:
                    pickle.dump(performance_dict, f)
    print('Done.')
