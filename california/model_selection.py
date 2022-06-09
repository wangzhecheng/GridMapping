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
import xgboost as xgb


all_feature_names = [
    'distance', 'on_same_road', 'adjacent', 'sv_pos_rate', 'min_sv_pole_angle_diff', 'avg_sv_pole_angle_diff',
    'angle_diff_between_poles','dijkstra1', 'dijkstra2', 'dijkstra_and', 'dijkstra_sum', 'isdetected1', 'isdetected2', 'both_detected', 'at_intersection_1',
    'at_intersection_2', 'both_at_intersection', 'at_same_intersections', 'sv_pos_rate_adj', 'sv_pos_rate_adj2',
    'min_sv_pole_angle_diff_adj', 'avg_sv_pole_angle_diff_adj', 'angle_diff_between_poles_adj',
    'either_at_intersection_nonadjacent'
]


feature_set_candidates = {
    # 'all_features': all_feature_names,
    # 'feature_set_1': ['distance', 'on_same_road', 'adjacent', 'sv_pos_rate', 'dijkstra1', 'isdetected1', 'isdetected2'],
    # 'feature_set_2': ['distance', 'on_same_road', 'adjacent', 'sv_pos_rate_adj', 'dijkstra1', 'both_detected',
    #                   'at_same_intersections', 'avg_sv_pole_angle_diff'],
    'feature_set_3': ['distance', 'on_same_road', 'adjacent', 'sv_pos_rate', 'min_sv_pole_angle_diff', 'avg_sv_pole_angle_diff',
    'angle_diff_between_poles','dijkstra1', 'isdetected1', 'isdetected2', 'both_detected', 'at_intersection_1',
    'at_intersection_2', 'both_at_intersection', 'at_same_intersections', 'sv_pos_rate_adj', 'sv_pos_rate_adj2',
    'min_sv_pole_angle_diff_adj', 'avg_sv_pole_angle_diff_adj', 'angle_diff_between_poles_adj',
    'either_at_intersection_nonadjacent']
}


def metrics(y_true, y_pred):
    TP = np.sum((y_true == 1) * (y_pred == 1))
    TN = np.sum((y_true == 0) * (y_pred == 0))
    FP = np.sum((y_true == 0) * (y_pred == 1))
    FN = np.sum((y_true == 1) * (y_pred == 0))
    precision = (TP + 1e-8) / (TP + FP + 1e-8)
    recall = (TP + 1e-8) / (TP + FN + 1e-8)
    F1 = 2 * precision * recall / (precision + recall)
    return round(precision, 4), round(recall, 4), round(F1, 4)


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


def get_average_test_perf(selected_feature_names, model, feature_dict_by_fips, label_dict_by_fips, idx2line, all_predicted_poles, ntrials=1):
    processed_feature_dict_by_fips = {}
    for fips in feature_dict_by_fips:
        processed_feature_dict_by_fips[fips] = []
        for pole_pair in feature_dict_by_fips[fips]:
            raw_feature_list = feature_dict_by_fips[fips][pole_pair]
            feature_list = feature_transformation_flexible(pole_pair, raw_feature_list, idx2line, all_predicted_poles, selected_feature_names)
            processed_feature_dict_by_fips[fips].append(feature_list)
    avg_F1_test = 0
    boundaries_keys = sorted(feature_dict_by_fips.keys())
    for n in range(ntrials):
        for test_fips in boundaries_keys:
            training_fips_list = copy.deepcopy(boundaries_keys)
            training_fips_list.remove(test_fips)
            training_feature_list = []
            training_label_list = []
            for fips in training_fips_list:
                training_feature_list += processed_feature_dict_by_fips[fips]
                training_label_list += label_dict_by_fips[fips]
            X_train = np.array(training_feature_list)
            y_train = np.array(training_label_list)
            X_test = np.array(processed_feature_dict_by_fips[test_fips])
            y_test = np.array(label_dict_by_fips[test_fips])
            model_use = copy.deepcopy(model)
            model_use.fit(X_train, y_train)
    #         y_train_pred = model_use.predict(X_train)
            y_test_pred = model_use.predict(X_test)
    #         precision_train, recall_train, F1_train = metrics(y_train, y_train_pred)
            precision_test, recall_test, F1_test = metrics(y_test, y_test_pred)
            avg_F1_test += F1_test
    avg_F1_test = avg_F1_test / (ntrials * len(boundaries_keys))
#     print(model_use.feature_importances_)
    return avg_F1_test


with open('results/SanCarlos_intermediate_data_for_model_selection.pickle', 'rb') as f:
    intermediate_result_dict = pickle.load(f)
feature_dict_by_fips = intermediate_result_dict['feature_dict_by_fips']
label_dict_by_fips = intermediate_result_dict['label_dict_by_fips']
idx2line = intermediate_result_dict['idx2line']
all_predicted_poles = intermediate_result_dict['all_predicted_poles']

perf_dict = {}

# Decision Tree
# print('----------------- Decision Tree')
# perf_dict['DT'] = {}
# for max_depth in [2, 3, 4, 5, 6, 7]:
#     print(max_depth)
#     perf_dict['DT'][max_depth] = {}
#     for feature_set in feature_set_candidates:
#         selected_feature_names = feature_set_candidates[feature_set]
#         model = DecisionTreeClassifier(max_depth=max_depth)
#         f1 = get_average_test_perf(selected_feature_names, model, feature_dict_by_fips, label_dict_by_fips,
#                                    idx2line, all_predicted_poles, ntrials=1)
#         perf_dict['DT'][max_depth][feature_set] = f1
#
# with open('results/model_selection_performance_dict.pickle', 'wb') as f:
#     pickle.dump(perf_dict, f)

# Logistic Regression
# print('----------------- Logistic Regression')
# perf_dict['LR'] = {}
# for l2_term in [1, 10, 100, 1000]:
#     print(l2_term)
#     perf_dict['LR'][l2_term] = {}
#     for feature_set in feature_set_candidates:
#         selected_feature_names = feature_set_candidates[feature_set]
#         model = LogisticRegression(C=l2_term)
#         f1 = get_average_test_perf(selected_feature_names, model, feature_dict_by_fips, label_dict_by_fips,
#                                    idx2line, all_predicted_poles, ntrials=1)
#         perf_dict['LR'][l2_term][feature_set] = f1
#
# with open('results/model_selection_performance_dict.pickle', 'wb') as f:
#     pickle.dump(perf_dict, f)


# SVC
# print('----------------- SVC')
# perf_dict['SVC'] = {}
# for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
#     print(kernel)
#     perf_dict['SVC'][kernel] = {}
#     for C in [0.1, 1, 3, 10]:
#         perf_dict['SVC'][kernel][C] = {}
#         for feature_set in feature_set_candidates:
#             selected_feature_names = feature_set_candidates[feature_set]
#             model = SVC(C=C, kernel=kernel)
#             f1 = get_average_test_perf(selected_feature_names, model, feature_dict_by_fips, label_dict_by_fips,
#                                        idx2line, all_predicted_poles, ntrials=1)
#             perf_dict['SVC'][kernel][C][feature_set] = f1
#
# with open('results/model_selection_performance_dict.pickle', 'wb') as f:
#     pickle.dump(perf_dict, f)

# Gradient Boosting
print('----------------- Gradient Boosting')
perf_dict['GB'] = {}
for max_depth in [2, 3, 4, 5, 6]:
    perf_dict['GB'][max_depth] = {}
    for eta in [0.1, 0.2, 0.4, 0.6, 0.8]:
        print(max_depth, eta)
        perf_dict['GB'][max_depth][eta] = {}
        for Lambda in [0, 1, 2, 5]:
            perf_dict['GB'][max_depth][eta][Lambda] = {}
            for num_round in [11, 13, 15]:
                perf_dict['GB'][max_depth][eta][Lambda][num_round] = {}
                for threshold in [0.25, 0.3, 0.35, 0.4]:
                    perf_dict['GB'][max_depth][eta][Lambda][num_round][threshold] = {}
                    for feature_set in feature_set_candidates:
                        selected_feature_names = feature_set_candidates[feature_set]
                        model = XGBoost(max_depth=max_depth, eta=eta, gamma=0, Lambda=Lambda,
                                        num_round=num_round, threshold=threshold)
                        f1 = get_average_test_perf(selected_feature_names, model, feature_dict_by_fips,
                                                   label_dict_by_fips,
                                                   idx2line, all_predicted_poles, ntrials=1)
                        perf_dict['GB'][max_depth][eta][Lambda][num_round][threshold][feature_set] = f1

# with open('results/model_selection_performance_dict.pickle', 'wb') as f:
#     pickle.dump(perf_dict, f)

# Random Forest
# print('----------------- Random Forest')
# perf_dict['RF'] = {}
# for n_estimators in [10, 30, 100, 150]:
#     perf_dict['RF'][n_estimators] = {}
#     for max_depth in [2, 3, 4, 5, 6]:
#         print(n_estimators, perf_dict)
#         perf_dict['RF'][n_estimators][max_depth] = {}
#         for feature_set in feature_set_candidates:
#             selected_feature_names = feature_set_candidates[feature_set]
#             model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
#             f1 = get_average_test_perf(selected_feature_names, model, feature_dict_by_fips, label_dict_by_fips,
#                                        idx2line, all_predicted_poles, ntrials=6)
#             perf_dict['RF'][n_estimators][max_depth][feature_set] = f1

with open('results/model_selection_performance_dict_GBonly_2.pickle', 'wb') as f:
    pickle.dump(perf_dict, f)

