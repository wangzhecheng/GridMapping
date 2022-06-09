from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import geojson
import copy
import xgboost as xgb


class XGBoost:
    def __init__(self, max_depth=None, eta=None, gamma=None, Lambda=None, num_round=5, threshold=0.5):
        self.param = {'objective': 'binary:logistic'}
        if max_depth is not None:
            self.param['max_depth'] = max_depth
        if eta is not None:
            self.param['eta'] = eta
        if gamma is not None:
            self.param['gamma'] = gamma
        if Lambda is not None:
            self.param['lambda'] = Lambda
        self.num_round = num_round
        self.threshold = threshold
        self.bst = None

    def fit(self, X, y):
        dtrain = xgb.DMatrix(X, label=y)
        self.bst = xgb.train(self.param, dtrain, self.num_round)

    def predict(self, X):
        dtest = xgb.DMatrix(X)
        return self.bst.predict(dtest) >= self.threshold


def get_direction(lat1, lon1, lat2, lon2):
    lat1 = np.deg2rad(lat1)
    lat2 = np.deg2rad(lat2)
    lon1 = np.deg2rad(lon1)
    lon2 = np.deg2rad(lon2)
    delta_y = lat2 - lat1
    delta_x = (lon2 - lon1)
    d = np.rad2deg(np.arctan2(delta_y, delta_x))
    # make the output in [0, 180)
    if d < 0:
        d += 180
    elif d == 180:
        d = 0
    return d


def get_ray(center_lat, center_lon, angle, length=0.0004):
    """
    angle: in degree
    length: the length of ray
    """
    start_lat = center_lat - 0.5 * length * np.sin(np.deg2rad(angle))
    start_lon = center_lon - 0.5 * length * np.cos(np.deg2rad(angle))

    end_lat = center_lat + 0.5 * length * np.sin(np.deg2rad(angle))
    end_lon = center_lon + 0.5 * length * np.cos(np.deg2rad(angle))
    return [start_lon, end_lon], [start_lat, end_lat]


def weight_avg(val_list, weight_list):
    if len(val_list) == 0:
        return -180
    a = np.array(val_list)
    w = np.array(weight_list)
    return np.sum(a * w) / np.sum(w)


def raw_feature_extraction(candidate_pairs, pid2way, polyline_dict_updated, dijkstra_edge_set, pole2svidx):
    # raw feature extraction
    feature_dict = {}
    for pp in candidate_pairs:
        pid1, pid2, dist = pp
        at_intersection_1 = 0
        at_intersection_2 = 0
        if pid1 in pid2way and len(pid2way[pid1]) > 1:
            at_intersection_1 = 1
        if pid2 in pid2way and len(pid2way[pid2]) > 1:
            at_intersection_2 = 1
        if pid1 not in pid2way or pid2 not in pid2way:
            on_same_road = False
            # print(pp)
        else:
            ways1 = pid2way[pid1]
            ways2 = pid2way[pid2]
            on_same_road = False
            for wid1 in ways1:
                for wid2 in ways2:
                    if wid1 == wid2:
                        on_same_road = True
                        break
                if on_same_road:
                    break
        nhops = -1  # Number of hops between 2 poles. -1 means not on the same road. 1 means they are next to each other
        street_view_collections = []  # list of street view IDs between two poles
        if on_same_road:
            pole_index_1 = polyline_dict_updated[wid1].pole2idx[pid1]
            pole_index_2 = polyline_dict_updated[wid2].pole2idx[pid2]
            nhops = abs(pole_index_1 - pole_index_2)
            min_index = min(pole_index_1, pole_index_2)
            max_index = max(pole_index_1, pole_index_2)
            for i in range(min_index, max_index):
                street_view_collections += polyline_dict_updated[wid1].street_view_collections[i]

        # Dijkstra's prediction. 1 connected, 0 unconnected
        if (pid1, pid2) in dijkstra_edge_set:
            dijkstra = 1
        else:
            dijkstra = 0

        # if (pid1, pid2) in dijkstra_edge_set_2:
        #     dijkstra2 = 1
        # else:
        #     dijkstra2 = 0
        p1_street_views = pole2svidx[pid1] if type(pid1) == int else []
        p2_street_views = pole2svidx[pid2] if type(pid2) == int else []
        feature_dict[(pid1, pid2)] = [dist, on_same_road, nhops, street_view_collections, dijkstra, p1_street_views,
                                      p2_street_views, at_intersection_1, at_intersection_2]
    return feature_dict


def feature_transformation_1(pole_pair, raw_feature_list, idx2line, predicted_pole_coords):
    """
    For street views between poles, use angle with min angle difference with the pole-pole angle get the line angle.
    For street views at the pole, use angle with min angle difference with the pole-pole angle to get the line angle.
    Only provide info on whether they are adjacent, but not how many hops.
    idx2line: street view idx and its line information (empty list -> no line detection).
    """
    pid1, pid2 = pole_pair
    dist, on_same_road, nhops, street_view_collections, dijkstra, p1_street_views, p2_street_views, _, _ = raw_feature_list
    on_same_road = int(on_same_road)
    dijkstra = int(dijkstra)
    adjacent = int(nhops == 1)
    npos = 0  # number of street view points with positive line detection
    all_lines = []  # all lines between these two poles
    for svidx in street_view_collections:
        lines = idx2line[svidx]
        if lines:
            npos += 1
        all_lines += lines
    if len(street_view_collections) > 0:
        sv_pos_rate = npos * 1.0 / len(
            street_view_collections)  # ratio of street view points with positive line detection
    else:
        sv_pos_rate = 0

    lat1, lon1 = predicted_pole_coords[pid1]
    lat2, lon2 = predicted_pole_coords[pid2]
    direction_between_points = get_direction(lat1, lon1, lat2, lon2)  # pole-pole angle

    angles = []
    strengths = []
    angle_diff_list = []
    for _, angle, strength in all_lines:
        angle = 90 - angle
        if angle < 0:
            angle += 180
        elif angle == 180:
            angle = 0
        angles.append(angle)
        strengths.append(strength)
        angle_diff = np.abs(
            direction_between_points - angle)  # angle difference between angle between points and detected angles
        angle_diff_list.append((angle_diff, angle))
    if angle_diff_list:
        min_angle_diff, best_angle = min(angle_diff_list, key=lambda x: x[0])
    else:
        min_angle_diff = 180
    avg_angle = weight_avg(angles, strengths)
    isdetected1 = int(type(pid1) == int)  # Whether the pole is detected or inserted
    isdetected2 = int(type(pid2) == int)  # Whether the pole is detected or inserted
    angle_diff_at_poles = []
    for pole_street_views in [p1_street_views, p2_street_views]:
        if pole_street_views:
            all_lines_p = []  # all lines between these two poles
            angles_p = []
            strengths_p = []
            angle_diff_list_p = []
            for svidx in pole_street_views:
                lines = idx2line[svidx]
                all_lines_p += lines
            for _, angle, strength in all_lines_p:
                if angle < 0:
                    angle = 90 - angle
                    angle += 180
                elif angle == 180:
                    angle = 0
                angles_p.append(angle)
                strengths_p.append(strength)
                angle_diff_p = np.abs(direction_between_points - angle)
                angle_diff_list_p.append((angle_diff_p, angle))
            avg_angle = weight_avg(angles_p, strengths_p)
            if angle_diff_list_p:
                min_angle_diff_p, best_angle_p = min(angle_diff_list_p, key=lambda x: x[0])
            else:
                min_angle_diff_p = 180
            angle_diff_at_poles.append(min_angle_diff_p)
    if angle_diff_at_poles:
        angle_diff_pole = np.min(angle_diff_at_poles)
    else:
        angle_diff_pole = 180  # angle difference between angle between points and avg detected line angles at poles.
    # feature_list = [dist / 100, on_same_road, adjacent, sv_pos_rate, min_angle_diff / 360, dijkstra, isdetected1, isdetected2, angle_diff_pole / 360]
    feature_list = [dist / 100, on_same_road, adjacent, sv_pos_rate, dijkstra, isdetected1, isdetected2]
    return feature_list


def feature_transformation_flexible(pole_pair, raw_feature_list, idx2line, predicted_pole_coords, selected_feature_names):
    """
    For street views between poles, use angle with min angle difference with the pole-pole angle get the line angle.
    For street views at the pole, use angle with min angle difference with the pole-pole angle to get the line angle.
    Only provide info on whether they are adjacent, but not how many hops.
    idx2line: street view idx and its line information (empty list -> no line detection).
    """
    pid1, pid2 = pole_pair
    dist, on_same_road, nhops, street_view_collections, dijkstra, p1_street_views, p2_street_views, at_intersection_1, at_intersection_2 = raw_feature_list
    on_same_road = int(on_same_road)
    dijkstra = int(dijkstra)
    # dijkstra2 = int(dijkstra2)
    adjacent = int(nhops == 1)
    npos = 0  # number of street view points with positive line detection
    all_lines = []  # all lines between these two poles
    for svidx in street_view_collections:
        lines = idx2line[svidx]
        if lines:
            npos += 1
        all_lines += lines
    if len(street_view_collections) > 0:
        sv_pos_rate = npos * 1.0 / len(
            street_view_collections)  # ratio of street view points with positive line detection
    else:
        sv_pos_rate = 0

    lat1, lon1 = predicted_pole_coords[pid1]
    lat2, lon2 = predicted_pole_coords[pid2]
    direction_between_points = get_direction(lat1, lon1, lat2, lon2)  # pole-pole angle

    angles = []
    strengths = []
    angle_diff_list = []
    for _, angle, strength in all_lines:
        angle = 90 - angle
        if angle < 0:
            angle += 180
        elif angle == 180:
            angle = 0
        angles.append(angle)
        strengths.append(strength)
        angle_diff = np.abs(
            direction_between_points - angle)  # angle difference between angle between points and detected angles
        angle_diff_list.append((angle_diff, angle))
    if angle_diff_list:
        min_angle_diff, best_angle = min(angle_diff_list, key=lambda x: x[0])
    else:
        min_angle_diff = 180
    avg_angle = weight_avg(angles, strengths)
    if angles:
        avg_angle_diff = np.abs(avg_angle - direction_between_points)
    else:
        avg_angle_diff = 180
    isdetected1 = int(type(pid1) == int)  # Whether the pole is detected or inserted
    isdetected2 = int(type(pid2) == int)  # Whether the pole is detected or inserted
    angle_diff_at_poles = []
    for pole_street_views in [p1_street_views, p2_street_views]:
        if pole_street_views:
            all_lines_p = []  # all lines between these two poles
            angles_p = []
            strengths_p = []
            angle_diff_list_p = []
            for svidx in pole_street_views:
                lines = idx2line[svidx]
                all_lines_p += lines
            for _, angle, strength in all_lines_p:
                if angle < 0:
                    angle = 90 - angle
                    angle += 180
                elif angle == 180:
                    angle = 0
                angles_p.append(angle)
                strengths_p.append(strength)
                angle_diff_p = np.abs(direction_between_points - angle)
                angle_diff_list_p.append((angle_diff_p, angle))
            avg_angle = weight_avg(angles_p, strengths_p)
            if angle_diff_list_p:
                min_angle_diff_p, best_angle_p = min(angle_diff_list_p, key=lambda x: x[0])
            else:
                min_angle_diff_p = 180
            angle_diff_at_poles.append(min_angle_diff_p)
    if angle_diff_at_poles:
        angle_diff_pole = np.min(angle_diff_at_poles)
    else:
        angle_diff_pole = 180  # angle difference between angle between points and avg detected line angles at poles.
    # feature_list = [dist / 100, on_same_road, adjacent, sv_pos_rate, min_angle_diff / 360, dijkstra, isdetected1, isdetected2, angle_diff_pole / 360]
    both_detected = isdetected1 * isdetected2
    both_at_intersection = at_intersection_1 * at_intersection_2
    candidate_features = {
        'distance': dist / 100,
        'on_same_road': on_same_road,
        'adjacent': adjacent,
        'sv_pos_rate': sv_pos_rate,
        'min_sv_pole_angle_diff': min_angle_diff / 180,
        'avg_sv_pole_angle_diff': avg_angle_diff / 180,
        'angle_diff_between_poles': angle_diff_pole / 180,
        'dijkstra': dijkstra,
        # 'dijkstra2': dijkstra2,
        # 'dijkstra_and': dijkstra1 * dijkstra2,
        # 'dijkstra_sum': dijkstra1 + dijkstra2,
        'isdetected1': isdetected1,
        'isdetected2': isdetected2,
        'both_detected': both_detected,
        'at_intersection_1': at_intersection_1,
        'at_intersection_2': at_intersection_2,
        'both_at_intersection': both_at_intersection,
        'at_same_intersections': both_at_intersection * on_same_road,
        'sv_pos_rate_adj': adjacent * sv_pos_rate + (1 - adjacent) * ((len(street_view_collections) == 0) * 1 + (1 - len(street_view_collections) == 0) * sv_pos_rate),
        'sv_pos_rate_adj2': adjacent * sv_pos_rate,
        'min_sv_pole_angle_diff_adj': adjacent * 0 + (1 - adjacent) * min_angle_diff / 180,
        'avg_sv_pole_angle_diff_adj': adjacent * 0 + (1 - adjacent) * avg_angle_diff / 180,
        'angle_diff_between_poles_adj': adjacent * 0 + (1 - adjacent) * angle_diff_pole / 180,
        'either_at_intersection_nonadjacent': max(at_intersection_1 * (1 - at_intersection_2), at_intersection_2 * (1 - at_intersection_1)) * (1 - adjacent)
    }
    feature_list = []
    for feature_name in selected_feature_names:
        feature_list.append(candidate_features[feature_name])
    return feature_list


def feature_transformation_2(pole_pair, raw_feature_list, idx2line, predicted_pole_coords):
    selected_feature_names = [
        'distance', 'on_same_road', 'adjacent', 'sv_pos_rate', 'min_sv_pole_angle_diff', 'avg_sv_pole_angle_diff',
        'angle_diff_between_poles','dijkstra', 'isdetected1', 'isdetected2', 'both_detected', 'at_intersection_1',
        'at_intersection_2', 'both_at_intersection', 'at_same_intersections', 'sv_pos_rate_adj', 'sv_pos_rate_adj2',
        'min_sv_pole_angle_diff_adj', 'avg_sv_pole_angle_diff_adj', 'angle_diff_between_poles_adj',
        'either_at_intersection_nonadjacent'
    ]
    return feature_transformation_flexible(pole_pair, raw_feature_list, idx2line, predicted_pole_coords, selected_feature_names)


