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
from matplotlib.path import Path
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from numba import autojit
import numpy as np
import census_tract_locator
from tqdm import tqdm
import geojson
import copy
from os.path import exists, join

from utils import *
from feature_engineering import *
from evaluation_utils import *

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

"""
This scripts is for using link prediction model to predict whether there is 
a power line between two utility poles.
"""

pole_model_list = ['ori0.5']
link_model_list = ['DT4ft1', 'GB3ft2'] # 'GB3ft2': Gradient Boosting model, 'DT4ft1': Decision tree model
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
performance_dict_save_path = 'results/aggregated/california_link_perf_ori0.5_' + benchmark_version + '.pickle'
if_evaluate = True

# model parameters
candidate_pair_threshold = 100

# evaluation parameters
evaluate_with_discretization_gridsize = 40
evaluate_with_dilation_radius = 15  # in # grids, *2 in meters

link_model_path_dict = {
    'DT4ft1': 'checkpoint/link_prediction_models/SanCarlos_DT_depth4_fitall_feature_transform_1.pickle',
    'GB3ft2': 'checkpoint/link_prediction_models/SanCarlos_GB_depth3_fitall_feature_transform_2.pickle',
}

fips_list_dict = {
    'SanCarlos': [6081609100, 6081609201, 6081609202, 6081609300, 6081609400, 6081609500, 6081609601, 6081609602, 6081609603],
    'Newark': [6001444100, 6001444200, 6001444301, 6001444400, 6001444500, 6001444601], 
    'SantaCruz': [6087121300, 6087121401, 6087121402, 6087121403, 6087121700], 
    'Yuba': [6101050201, 6101050202, 6101050301, 6101050302],
    'Monterey': [6053012401, 6053012402, 6053012302, 6053012200, 6053012100, 6053012000],
    'Salinas': [6053000800, 6053000600, 6053000701, 6053000702],
}

# functions
def metrics(y_true, y_pred):
    TP = np.sum((y_true == 1) * (y_pred == 1))
    TN = np.sum((y_true == 0) * (y_pred == 0))
    FP = np.sum((y_true == 0) * (y_pred == 1))
    FN = np.sum((y_true == 1) * (y_pred == 0))
    precision = (TP + 1e-8) / (TP + FP + 1e-8)
    recall = (TP + 1e-8) / (TP + FN + 1e-8)
    F1 = 2 * precision * recall / (precision + recall)
    return round(precision, 4), round(recall, 4), round(F1, 4)


def load_result_data(region, pole_model):
    with open(join('results', region, 'line_info_merged.pickle'), 'rb') as f:
        sv2line = pickle.load(f)
    with open(join('results', region, pole_model, 'pole_locations.pickle'), 'rb') as f:
        poles = pickle.load(f)
    with open(join('results', region, pole_model, 'pole_attached_GSVs.pickle'), 'rb') as f:
        pole2sv = pickle.load(f)
    # all predicted poles (detected + inserted)
    with open(join('results', region, pole_model, 'all_predicted_poles.pickle'), 'rb') as f:
        all_predicted_poles = pickle.load(f)
    # road model (street views and poles registered)
    with open(join('results', region, pole_model, 'road_polyline_dict.pickle'), 'rb') as f:
        polyline_dict_updated = pickle.load(f)
    # matched/unmatched detected poles and virtual poles
    with open(join('results', region, pole_model, 'unmatched_detected_poles_' + benchmark_version + '.pickle'),
              'rb') as f:
        unmatched_index_detected = pickle.load(f)
    with open(join('results', region, pole_model, 'unmatched_virtual_poles_' + benchmark_version + '.pickle'),
              'rb') as f:
        unmatched_index_virtual = pickle.load(f)
    with open(join('results', region, pole_model, 'matched_detected_poles_' + benchmark_version + '.pickle'),
              'rb') as f:
        matching_detected = pickle.load(f)
    with open(join('results', region, pole_model, 'matched_virtual_poles_' + benchmark_version + '.pickle'), 'rb') as f:
        matching_virtual = pickle.load(f)

    # Load Dijkstra's predicted edge set
    with open(join('results', region, pole_model, 'dijkstra_edge_set_assimilate_unattached_noscaling.pickle'),
              'rb') as f:
        dijkstra_edge_set = pickle.load(f)

    assert len(pole2sv) == len(poles)
    return sv2line, poles, pole2sv, all_predicted_poles, polyline_dict_updated, unmatched_index_detected, \
           unmatched_index_virtual, matching_detected, matching_virtual, dijkstra_edge_set


def process_detection_data(sv2line, pole2sv, matching_detected, matching_virtual, polyline_dict_updated):
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

    assert len(idx2sv) == len(sv2line) and len(idx2line) == len(sv2line) and len(idx2poles) == len(sv2line)
    assert len(pole2svidx) == len(pole2sv)

    matching_predicted2true = {}
    matching_true2predicted = {}
    for true_pole_idx, detected_pole_idx, _ in matching_detected:
        matching_predicted2true[detected_pole_idx] = true_pole_idx
        matching_true2predicted[true_pole_idx] = detected_pole_idx
    for true_pole_idx, virtual_pole_idx, _ in matching_virtual:
        matching_predicted2true[virtual_pole_idx] = true_pole_idx
        matching_true2predicted[true_pole_idx] = virtual_pole_idx
    assert len(matching_predicted2true) == len(matching_true2predicted)

    pid2way = {}  # detected pole ID -> list of way IDs to which it is attached
    for way_id in polyline_dict_updated:
        for pid in polyline_dict_updated[way_id].detected_poles:
            if not pid in pid2way:
                pid2way[pid] = []
            pid2way[pid].append(way_id)
        for pid in polyline_dict_updated[way_id].virtual_poles:
            if not pid in pid2way:
                pid2way[pid] = []
            pid2way[pid].append(way_id)

    return sv2idx, idx2sv, idx2line, pole2svidx, idx2poles, matching_predicted2true, matching_true2predicted, pid2way


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


def pardon_skipping_and_redundancy(TP_edge_set, FN_edge_set, FP_edge_set, matching_predicted2true):
    FN_edge_set_updated = copy.deepcopy(FN_edge_set)
    FP_edge_set_updated = copy.deepcopy(FP_edge_set)
    TP_edge_set_updated = copy.deepcopy(TP_edge_set)
    matching_FN_TP = []

    nb_dict_FN = construct_neighbor_dict(FN_edge_set)
    nb_dict_FP = construct_neighbor_dict(FP_edge_set)
    endpoints_set_FN = find_single_way(nb_dict_FN)
    endpoints_set_FP = find_single_way(nb_dict_FP)
    used_FP_endpoints = set()
    used_FN_endpoints = set()
    for endpoints in endpoints_set_FP:
        pid1, pid2 = endpoints
        if pid1 in matching_predicted2true and pid2 in matching_predicted2true:
            pid1_true = matching_predicted2true[pid1]
            pid2_true = matching_predicted2true[pid2]
            if (pid1_true, pid2_true) in endpoints_set_FN:
                matched_FN = (pid1_true, pid2_true)
            elif (pid2_true, pid1_true) in endpoints_set_FN:
                matched_FN = (pid2_true, pid1_true)
            else:
                continue
            if matched_FN in used_FN_endpoints or endpoints in used_FP_endpoints:
                continue
            endpoints_FPs = generate_end_points([endpoints_set_FP[endpoints][0][0]] +
                                                [x[1] for x in endpoints_set_FP[endpoints]])
            endpoints_FNs = generate_end_points([endpoints_set_FN[matched_FN][0][0]] +
                                                [x[1] for x in endpoints_set_FN[matched_FN]])
            for pair in endpoints_FPs:
                if pair in endpoints_set_FP:
                    used_FP_endpoints.add(pair)
                elif (pair[1], pair[0]) in endpoints_set_FP:
                    used_FP_endpoints.add((pair[1], pair[0]))
            for pair in endpoints_FNs:
                if pair in endpoints_set_FN:
                    used_FN_endpoints.add(pair)
                elif (pair[1], pair[0]) in endpoints_set_FN:
                    used_FN_endpoints.add((pair[1], pair[0]))
                    #             print(endpoints, len(endpoints_set_FP[endpoints]), matched_FN, len(endpoints_set_FN[matched_FN]))
            for edge in endpoints_set_FP[endpoints]:
                if edge in FP_edge_set_updated:
                    FP_edge_set_updated.remove(edge)
                elif (edge[1], edge[0]) in FP_edge_set_updated:
                    FP_edge_set_updated.remove((edge[1], edge[0]))

            for edge in endpoints_set_FN[matched_FN]:
                if edge in FN_edge_set_updated:
                    FN_edge_set_updated.remove(edge)
                    TP_edge_set_updated.add(edge)
                elif (edge[1], edge[0]) in FN_edge_set_updated:
                    FN_edge_set_updated.remove((edge[1], edge[0]))
                    TP_edge_set_updated.add((edge[1], edge[0]))

            matching_FN_TP.append((endpoints_set_FN[matched_FN], endpoints_set_FP[endpoints]))

    TP_edge = len(TP_edge_set_updated)
    FP_edge = len(FP_edge_set_updated)
    FN_edge = len(FN_edge_set_updated)
    precision_edge = TP_edge / (TP_edge + FP_edge)
    if TP_edge > 0:
        recall_edge = TP_edge / (TP_edge + FN_edge)
    else:
        recall_edge = 1.0
    f1_edge = 2 * precision_edge * recall_edge / (precision_edge + recall_edge)
    print('Overall link prediction performance after pardoning skipping and redundancy cases: ')
    print('    TP:', TP_edge, 'FP:', FP_edge, 'FN:', FN_edge)
    print('    precision:', precision_edge, 'recall:', recall_edge, 'F1', f1_edge,
          'IoU: ', TP_edge / (TP_edge + FP_edge + FN_edge))
    results = {'TP': TP_edge, 'FP': FP_edge, 'FN': FN_edge, 'precision': precision_edge, 'recall': recall_edge,
               'F1': f1_edge, 'IoU': TP_edge / (TP_edge + FP_edge + FN_edge)}
    return results


def get_overall_link_prediction_perf(y_test_pred, feature_dict, matching_predicted2true, true_edge_set):
    # Link-level
    predicted_edges = set()
    FP_edge_set = set()
    FN_edge_set = set()
    TP_edge_set = set()
    TP_edge = 0
    FP_edge = 0
    FN_edge = 0
    for i, p in enumerate(feature_dict):
        pid1, pid2 = p
        if y_test_pred[i]:
            if pid1 in matching_predicted2true and pid2 in matching_predicted2true:
                true_pid1 = matching_predicted2true[pid1]
                true_pid2 = matching_predicted2true[pid2]
                edge = tuple(sorted((true_pid1, true_pid2)))
                predicted_edges.add(edge)
                if edge in true_edge_set:
                    TP_edge += 1
                    TP_edge_set.add(edge)
                else:
                    FP_edge += 1
                    FP_edge_set.add(p)
            else:
                FP_edge += 1
                FP_edge_set.add(p)

    for edge in true_edge_set:
        if not edge in predicted_edges:
            FN_edge += 1
            FN_edge_set.add(edge)

    precision_edge = TP_edge / (TP_edge + FP_edge)
    if TP_edge > 0:
        recall_edge = TP_edge / (TP_edge + FN_edge)
    else:
        recall_edge = 1.0
    f1_edge = 2 * precision_edge * recall_edge / (precision_edge + recall_edge)
    print('Overall link prediction performance: ')
    print('    TP:', TP_edge, 'FP:', FP_edge, 'FN:', FN_edge)
    print('    precision:', precision_edge, 'recall:', recall_edge, 'F1', f1_edge,
          'IoU: ', TP_edge / (TP_edge + FP_edge + FN_edge))
    results_raw = {'TP': TP_edge, 'FP': FP_edge, 'FN': FN_edge, 'precision': precision_edge, 'recall': recall_edge,
               'F1': f1_edge, 'IoU': TP_edge / (TP_edge + FP_edge + FN_edge)}
    results_mercy = pardon_skipping_and_redundancy(TP_edge_set, FN_edge_set, FP_edge_set, matching_predicted2true)
    return results_raw, results_mercy


def evaluate_link_prediction(region, benchmark_version, discretization_params, ring_xy_diff_dict,
                             feature_dict, all_predicted_poles, matching_predicted2true, y_test_pred):
    # load true poles and true edges
    with open('ground_truth/' + region + '/ground_truth_poles_' + benchmark_version + '.pickle', 'rb') as f:
        true_poles = pickle.load(f)
    with open('ground_truth/' + region + '/ground_truth_connections_' + benchmark_version + '.pickle', 'rb') as f:
        true_edge_set = pickle.load(f)
    print('# ground-truth poles: ', len(true_poles))
    print('# ground-truth links', len(true_edge_set))

    # construct label list
    label_list = []
    covered_count = 0
    incorrect_pole_count = 0
    for pole_pair in feature_dict:
        pid1, pid2 = pole_pair
        if pid1 in matching_predicted2true and pid2 in matching_predicted2true:
            true_pid1 = matching_predicted2true[pid1]
            true_pid2 = matching_predicted2true[pid2]
            edge = tuple(sorted((true_pid1, true_pid2)))
            #             candidate_edge_by_fips[fips].add(edge)
            if edge in true_edge_set:
                covered_count += 1
                label = 1
            else:
                label = 0
        else:
            incorrect_pole_count += 1
            label = 0
        label_list.append(label)
    true_edge_covered_rate = covered_count / len(true_edge_set)
    candidate_with_incorrect_pole_rate = incorrect_pole_count / len(feature_dict)
    print('Full dataset, # true edges covered: ', covered_count, '/', len(true_edge_set),
          ', # candidates with incorrect pole: ', incorrect_pole_count, '/', len(feature_dict))
    y_test = np.array(label_list)
    precision_test, recall_test, F1_test = metrics(y_test, y_test_pred)
    print("Link prediction precision, recall, and F1 (for matched poles only):", precision_test, recall_test, F1_test)
    link_pred_perf = {
        'true_edge_covered_rate': true_edge_covered_rate,
        'candidate_with_incorrect_pole_rate': candidate_with_incorrect_pole_rate,
        'precision': precision_test, 'recall': recall_test, 'F1': F1_test
    }
    # get overall link prediction performance (considering links between undetected poles)
    overall_perf_raw, overall_perf_mercy = get_overall_link_prediction_perf(y_test_pred, feature_dict,
                                                                            matching_predicted2true, true_edge_set)

    # evaluation with discretization
    _, _, _, precision_dis, recall_dis, f1_dis, IOU_dis, grid_mat_predicted, grid_mat_true = \
        evaluate_with_discretization(evaluate_with_discretization_gridsize, discretization_params,
                                     y_test_pred, feature_dict, true_edge_set, true_poles,
                                     all_predicted_poles, return_matrix=True)
    # vis_dis = np.stack([grid_mat_true, grid_mat_predicted, np.zeros_like(grid_mat_true)], axis=2) * 255
    # vis_dis = np.flip(vis_dis.transpose(1, 0, 2), axis=0)
    print('similarity with discretization @gridsize = ' + str(evaluate_with_discretization_gridsize) + ':')
    print('    precision:', precision_dis, 'recall:', recall_dis, 'F1', f1_dis, 'IoU: ', IOU_dis)
    discretization_perf = {'precision': precision_dis, 'recall': recall_dis, 'F1': f1_dis, 'IoU': IOU_dis,
                           # 'visualization': vis_dis
                           }
    # evaluation with path dilation
    sh2 = SpatialHashing(unit=2, lat_s=lat_s, lon_w=lon_w, lat_n=lat_n, lon_e=lon_e,
                         dlat0=dlat0, dlon0=dlon0)
    predicted_coord_dict = {}
    true_coord_dict = {}
    for i, p in enumerate(feature_dict):
        pid1, pid2 = p
        lat1, lon1 = all_predicted_poles[pid1]
        lat2, lon2 = all_predicted_poles[pid2]
        if y_test_pred[i]:
            predicted_coord_dict[p] = [[lat1, lon1], [lat2, lon2]]
    for p in true_edge_set:
        pid1, pid2 = p
        lat1, lon1 = true_poles[pid1]
        lat2, lon2 = true_poles[pid2]
        true_coord_dict[p] = [[lat1, lon1], [lat2, lon2]]
    predicted_edge_grid = Grid(sh2)
    predicted_edge_grid.construct_from_way_coord_dict(predicted_coord_dict)
    true_edge_grid = Grid(sh2)
    true_edge_grid.construct_from_way_coord_dict(true_coord_dict)
    _, _, _, _, precision_dil, recall_dil, f1_dil, grid_mat_predicted_dil, grid_mat_true_dil = evaluate_with_dilation(
        evaluate_with_dilation_radius, sh2.x_max, sh2.y_max, ring_xy_diff_dict, true_edge_grid,
        predicted_edge_grid, return_matrix=True)
    # vis_dil = np.stack([grid_mat_true_dil>0, grid_mat_predicted_dil>0, np.zeros_like(grid_mat_predicted_dil)], axis=2) * 255
    # vis_dil = np.flip(vis_dil.transpose(1, 0, 2), axis=0)
    print('similarity with path dilation @radius = ' + str(evaluate_with_dilation_radius) + ':')
    print('    precision:', precision_dil, 'recall:', recall_dil, 'F1', f1_dil)
    dilation_perf = {'precision': precision_dil, 'recall': recall_dil, 'F1': f1_dil,
                     # 'visualization': vis_dil
                     }
    return link_pred_perf, overall_perf_raw, overall_perf_mercy, discretization_perf, dilation_perf


if __name__ == '__main__':
    # ring (used for obtaining all cells within a certain radius of the center on a raster map)
    with open('data/ring_xy_diff_dict_200.pickle', 'rb') as f:
        ring_xy_diff_dict = pickle.load(f)

    performance_dict = {x: {y: {} for y in pole_model_list} for x in region_list}

    for region in region_list:
        lat_s, lat_n, lon_w, lon_e, dlat0, dlon0, discretization_params = load_discretization_params(region)

        for pole_model in pole_model_list:
            print('-' * 30 + ' Region: ' + region + ', Pole model: ' + pole_model + ' ' + '-' * 30)
            # load results of previous step
            sv2line, poles, pole2sv, all_predicted_poles, polyline_dict_updated, unmatched_index_detected, \
            unmatched_index_virtual, matching_detected, matching_virtual, dijkstra_edge_set = \
                load_result_data(region, pole_model)
            print('# street view points:', len(sv2line))
            print('# detected poles:', len(poles))
            print('# predicted poles in total (detected + virtual):', len(all_predicted_poles))
            print('# edges predicted by Dijkstra:', len(dijkstra_edge_set))

            sv2idx, idx2sv, idx2line, pole2svidx, idx2poles, matching_predicted2true, matching_true2predicted, pid2way = \
                process_detection_data(sv2line, pole2sv, matching_detected, matching_virtual, polyline_dict_updated)

            sh200 = SpatialHashing(unit=200, lat_s=lat_s, lon_w=lon_w, lat_n=lat_n, lon_e=lon_e,
                                   dlat0=dlat0, dlon0=dlon0)

            candidate_pairs = generate_candidate_pair(all_predicted_poles, candidate_pair_threshold, sh200)
            print('# candidate pairs:', len(candidate_pairs))

            for way_id in polyline_dict_updated:
                polyline_dict_updated[way_id].collect_info()

            # collect raw features
            feature_dict = raw_feature_extraction(candidate_pairs, pid2way, polyline_dict_updated, dijkstra_edge_set,
                                                  pole2svidx)


            # predict
            for link_model_alias in link_model_list:
                print('-' * 20 + ' Link prediction model: ' + link_model_alias + ' ' + '-' * 20)
                
                # feature engineering
                processed_feature_list = []
                for pole_pair in feature_dict:
                    raw_feature_list = feature_dict[pole_pair]
                    if 'ft1' in link_model_alias:
                        feature_list = feature_transformation_1(pole_pair, raw_feature_list, idx2line, all_predicted_poles)
                    else:
                        feature_list = feature_transformation_2(pole_pair, raw_feature_list, idx2line,
                                                                all_predicted_poles)
                    processed_feature_list.append(feature_list)
                
                with open(link_model_path_dict[link_model_alias], 'rb') as f:
                    model = pickle.load(f)
                X_test = np.array(processed_feature_list)
                y_test_pred = model.predict(X_test)

                if not exists(join('results', region, pole_model, link_model_alias)):
                    os.mkdir(join('results', region, pole_model, link_model_alias))
                with open(join('results', region, pole_model, link_model_alias, 'predicted_binary_classes.pickle'),
                          'wb') as f:
                    pickle.dump(y_test_pred, f)
                with open(join('results', region, pole_model, link_model_alias, 'candidate_pair_list.pickle'),
                          'wb') as f:
                    pickle.dump(list(feature_dict.keys()), f)

                if if_evaluate:
                    link_pred_perf, overall_perf_raw, overall_perf_mercy, discretization_perf, dilation_perf = \
                        evaluate_link_prediction(region, benchmark_version, discretization_params, ring_xy_diff_dict,
                                                 feature_dict, all_predicted_poles, matching_predicted2true,
                                                 y_test_pred)

                    performance_dict[region][pole_model][link_model_alias] = {
                        'raw_link_prediction': link_pred_perf,
                        'overall_link_prediction': overall_perf_raw,
                        'mercy_overall_link_prediction': overall_perf_mercy,
                        'discretization': discretization_perf,
                        'dilation': dilation_perf
                    }
                    with open(performance_dict_save_path, 'wb') as f:
                        pickle.dump(performance_dict, f)

    print('Done.')
