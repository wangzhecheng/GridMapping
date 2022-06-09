import numpy as np
from utils import *

def rearrange_pair(p):
    """Given tuple (pid1, pid2), rearrange it based on: int < str."""
    if type(p[0]) == type(p[1]):
        return tuple(sorted(p))
    elif type(p[0]) == str and type(p[1]) == int:
        return (p[1], p[0])
    else:
        return tuple(p)


def construct_neighbor_dict(edge_set):
    """Given the edge set of a graph, return its corresponding neighbor dict."""
    nd = {}
    for edge in edge_set:
        p1, p2 = edge
        if p1 not in nd:
            nd[p1] = set()
        nd[p1].add(p2)
        if p2 not in nd:
            nd[p2] = set()
        nd[p2].add(p1)
    return nd


def generate_end_points(l):
    """Given a list, generate the endpoints for any sequence of elements (segments).
    E.g. given [a, b, c], return the dict of [a, b], [b, c], and [a, c]."""
    assert len(l) >= 2
    endpoints_dict = {}
    for i in range(2, len(l)+1):
        for j in range(0, len(l)-i+1):
            endpoints_dict[(l[j], l[j + i - 1])] = []
            for k in range(j, j + i - 1):
                endpoints_dict[(l[j], l[j + i - 1])].append((l[k], l[k + 1]))
    return endpoints_dict


def find_single_way(neighbor_dict):
    """Given a graph (neighbor dict), find all single way with no branch (i.e. a-b-c-d where b and c
    have no other neighbours), and return all endpoints of the segments of the way."""
    used_edges = set()
    path_list = []
    for start in neighbor_dict:
        if len(neighbor_dict[start]) == 2:
            continue
        for nb in neighbor_dict[start]:
            curr = nb
            last = start
            path = [start]
            while rearrange_pair((curr, last)) not in used_edges:
                path.append(curr)
                used_edges.add(rearrange_pair((curr, last)))
                last_val = last
                last = curr
                if len(neighbor_dict[curr]) != 2:
                    break
                else:
                    nb_list = list(neighbor_dict[curr])
                    if nb_list[0] == last_val:
                        curr = nb_list[1]
                    else:
                        curr = nb_list[0]
            if len(path) > 1:
                path_list.append(path)
    endpoints_dict = dict()
    for path in path_list:
        endpoints_dict_sub = generate_end_points(path)
        for endpoints in endpoints_dict_sub:
            endpoints_dict[endpoints] = endpoints_dict_sub[endpoints]
    return endpoints_dict


def evaluate_with_discretization(side_length, discretization_params, y_test_pred, feature_dict, true_edge_set, true_poles,
                                 all_predicted_poles, return_matrix=False):
    sh = SpatialHashing(unit=side_length,
                        lat_s=discretization_params['lat_s'],
                        lon_w=discretization_params['lon_w'],
                        lat_n=discretization_params['lat_n'],
                        lon_e=discretization_params['lon_e'],
                        dlat0=discretization_params['dlat0'],
                        dlon0=discretization_params['dlon0'])


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
    predicted_edge_grid = Grid(sh)
    predicted_edge_grid.construct_from_way_coord_dict(predicted_coord_dict)
    true_edge_grid = Grid(sh)
    true_edge_grid.construct_from_way_coord_dict(true_coord_dict)
    predicted_edge_pixels = set(predicted_edge_grid.filled_dict)
    true_edge_pixels = set(true_edge_grid.filled_dict)
    TP_pixels = predicted_edge_pixels & true_edge_pixels
    FP_pixels = predicted_edge_pixels - true_edge_pixels
    FN_pixels = true_edge_pixels - predicted_edge_pixels
    TP_edge = len(TP_pixels)
    FP_edge = len(FP_pixels)
    FN_edge = len(FN_pixels)
    assert TP_edge + FP_edge == len(predicted_edge_pixels)
    assert TP_edge + FN_edge == len(true_edge_pixels)
    precision_edge = TP_edge / (TP_edge + FP_edge)
    recall_edge = TP_edge / (TP_edge + FN_edge)
    f1_edge = 2 * precision_edge * recall_edge / (precision_edge + recall_edge)
    if return_matrix:
        grid_mat_predicted = predicted_edge_grid.get_grid_matrix()
        grid_mat_true = true_edge_grid.get_grid_matrix()
        return TP_edge, FP_edge, FN_edge, precision_edge, recall_edge, f1_edge, TP_edge / (TP_edge + FP_edge + FN_edge), grid_mat_predicted > 0, grid_mat_true > 0
    else:
        return TP_edge, FP_edge, FN_edge, precision_edge, recall_edge, f1_edge, TP_edge / (TP_edge + FP_edge + FN_edge)


def evaluate_with_discretization_for_all_subsets(side_length, discretization_params,
                                                 y_test_pred_by_fips,
                                                 feature_dict_by_fips,
                                                 true_edge_set_by_fips, true_poles,
                                                 all_predicted_poles,
                                                 verbose=False, return_matrix=False):

    TP_edge_all = 0
    FP_edge_all = 0
    FN_edge_all = 0
    grid_mat_predicted_all = 0.
    grid_mat_true_all = 0.
    for fips in sorted(feature_dict_by_fips.keys()):
        if return_matrix == True:
            TP_edge, FP_edge, FN_edge, precision_edge, recall_edge, f1_edge, _, grid_mat_predicted, grid_mat_true = evaluate_with_discretization(
                side_length,
                discretization_params,
                y_test_pred_by_fips[fips],
                feature_dict_by_fips[fips],
                true_edge_set_by_fips[fips],
                true_poles,
                all_predicted_poles,
                return_matrix
            )
            grid_mat_predicted_all += grid_mat_predicted
            grid_mat_true_all += grid_mat_true
        else:
            TP_edge, FP_edge, FN_edge, precision_edge, recall_edge, f1_edge, _= evaluate_with_discretization(
                side_length,
                discretization_params,
                y_test_pred_by_fips[fips],
                feature_dict_by_fips[fips],
                true_edge_set_by_fips[fips],
                true_poles,
                all_predicted_poles,
                return_matrix
            )
        if verbose:
            print(side_length, fips, TP_edge, FP_edge, FN_edge, precision_edge, recall_edge, f1_edge)
        TP_edge_all += TP_edge
        FP_edge_all += FP_edge
        FN_edge_all += FN_edge
    precision_edge_all = TP_edge_all / (TP_edge_all + FP_edge_all)
    recall_edge_all = TP_edge_all / (TP_edge_all + FN_edge_all)
    f1_edge_all = 2*precision_edge_all*recall_edge_all / (precision_edge_all + recall_edge_all)
    if return_matrix:
        return precision_edge_all, recall_edge_all, f1_edge_all, TP_edge_all / (TP_edge_all + FP_edge_all + FN_edge_all), grid_mat_predicted_all > 0, grid_mat_true_all > 0
    else:
        return precision_edge_all, recall_edge_all, f1_edge_all, TP_edge_all / (TP_edge_all + FP_edge_all + FN_edge_all)


def dilate_point_filled_dict_modified(point_filled_dict, ring_xy_diff_dict, dilate_radius, x_max, y_max):
    """
    set the neighbor grid for each point to be filled.
    """
#     new_point_filled_set = set(point_filled_dict.keys())
    new_point_filled_mat = np.zeros([x_max+1, y_max+1])
    for xy in point_filled_dict:
        x, y = xy
        for radius in range(dilate_radius + 1):
            for dx, dy in ring_xy_diff_dict[radius]:
                if (0 <= x + dx <= x_max) and (0 <= y + dy <= y_max):
                    #                         if (x+dx, y+dy) not in new_point_filled_set:
                    #                             new_point_filled_set.add((x+dx, y+dy))
                    new_point_filled_mat[x + dx, y + dy] = 1
    return new_point_filled_mat


def evaluate_with_dilation(dilate_radius, x_max, y_max, ring_xy_diff_dict,
                           true_edge_grid, predicted_edge_grid,
                           return_matrix=False):

    true_edge_grid_dilated = dilate_point_filled_dict_modified(true_edge_grid.filled_dict,
                                                               ring_xy_diff_dict,
                                                               dilate_radius=dilate_radius,
                                                               x_max=x_max,
                                                               y_max=y_max)
    predicted_edge_grid_dilated = dilate_point_filled_dict_modified(predicted_edge_grid.filled_dict,
                                                                    ring_xy_diff_dict,
                                                                    dilate_radius=dilate_radius,
                                                                    x_max=x_max,
                                                                    y_max=y_max)
    correct_count = 0
    for xy in predicted_edge_grid.filled_dict:  # number of predicted pixels that can be covered by PGE
        x, y = xy
        if true_edge_grid_dilated[x, y]:
            correct_count += 1
    precision = correct_count / len(predicted_edge_grid.filled_dict)

    covered_count = 0
    for xy in true_edge_grid.filled_dict:  # number of PGE pixcels that can be covered by predicted
        x, y = xy
        if predicted_edge_grid_dilated[x, y]:
            covered_count += 1
    recall = covered_count / len(true_edge_grid.filled_dict)
    f1 = 2 * precision * recall / (precision + recall)

    if return_matrix:
        return correct_count, len(predicted_edge_grid.filled_dict), covered_count, len(true_edge_grid.filled_dict), \
               precision, recall, f1, predicted_edge_grid_dilated, true_edge_grid_dilated
    else:
        return correct_count, len(predicted_edge_grid.filled_dict), covered_count, len(true_edge_grid.filled_dict), \
               precision, recall, f1


def evaluate_with_dilation_for_all_subsets(dilate_radius, x_max, y_max, ring_xy_diff_dict,
                                            true_edge_grid_by_fips, predicted_edge_grid_by_fips,
                                            return_matrix=False, verbose=False):
    count_correct_list = []
    count_predicted_list = []
    count_covered_list = []
    count_true_list = []
    grid_mat_predicted = 0.
    grid_mat_true = 0.
    for fips in sorted(true_edge_grid_by_fips.keys()):
        if return_matrix == True:
            correct_count, predicted_edge_pixel_count, covered_count, true_edge_pixel_count, \
            precision, recall, f1, predicted_edge_grid_dilated, true_edge_grid_dilated = \
                evaluate_with_dilation(dilate_radius, x_max, y_max, ring_xy_diff_dict,
                                       true_edge_grid_by_fips[fips], predicted_edge_grid_by_fips[fips], return_matrix)
            grid_mat_predicted += predicted_edge_grid_dilated
            grid_mat_true += true_edge_grid_dilated
        else:
            correct_count, predicted_edge_pixel_count, covered_count, true_edge_pixel_count, \
            precision, recall, f1 = evaluate_with_dilation(dilate_radius, x_max, y_max, ring_xy_diff_dict,
                                       true_edge_grid_by_fips[fips], predicted_edge_grid_by_fips[fips], return_matrix)

        count_correct_list.append(correct_count)
        count_predicted_list.append(predicted_edge_pixel_count)
        count_covered_list.append(covered_count)
        count_true_list.append(true_edge_pixel_count)

        if verbose:
            print(dilate_radius, fips, precision, recall, f1)

    precision_all = np.sum(count_correct_list) / np.sum(count_predicted_list)
    recall_all = np.sum(count_covered_list) / np.sum(count_true_list)
    f1_all = 2 * precision_all * recall_all / (precision_all + recall_all)

    if return_matrix:
        return precision_all, recall_all, f1_all, grid_mat_predicted, grid_mat_true
    else:
        return precision_all, recall_all, f1_all


def evaluate_with_dilation_dijkstra(dilate_radius, x_max, y_max, ring_xy_diff_dict, predicted_grid_filled_dict,
                                    pge_line_grid, return_matrix=False):
    pge_line_grid_dilated = dilate_point_filled_dict_modified(pge_line_grid.filled_dict, ring_xy_diff_dict,
                                                              dilate_radius=dilate_radius, x_max=x_max, y_max=y_max)
    predicted_grid_dilated = dilate_point_filled_dict_modified(predicted_grid_filled_dict, ring_xy_diff_dict,
                                                               dilate_radius=dilate_radius, x_max=x_max, y_max=y_max)
    count = 0
    for xy in predicted_grid_filled_dict:  # number of predicted pixels that can be covered by PGE
        x, y = xy
        if pge_line_grid_dilated[x, y]:
            count += 1
    precision = count / len(predicted_grid_filled_dict)

    count = 0
    for xy in pge_line_grid.filled_dict:  # number of PGE pixcels that can be covered by predicted
        x, y = xy
        if predicted_grid_dilated[x, y]:
            count += 1
    recall = count / len(pge_line_grid.filled_dict)
    f1 = 2 * precision * recall / (precision + recall)

    if return_matrix:
        return precision, recall, f1, predicted_grid_dilated, pge_line_grid_dilated
    return precision, recall, f1


def evaluate_with_discretization_dijkstra(side_length, discretization_params, pge_pixel_list, predicted_grid_pixel_list,
                                          return_matrix=False):
    sh_a = SpatialHashing(unit=side_length,
                        lat_s=discretization_params['lat_s'],
                        lon_w=discretization_params['lon_w'],
                        lat_n=discretization_params['lat_n'],
                        lon_e=discretization_params['lon_e'],
                        dlat0=discretization_params['dlat0'],
                        dlon0=discretization_params['dlon0'])
    pge_line_grid_coarse = Grid(sh_a)
    pge_line_grid_coarse.construct_from_coord_list(pge_pixel_list)
    predicted_grid_coarse = Grid(sh_a)
    predicted_grid_coarse.construct_from_coord_list(predicted_grid_pixel_list)

    predicted_edge_pixels = set(predicted_grid_coarse.filled_dict)
    true_edge_pixels = set(pge_line_grid_coarse.filled_dict)
    TP_pixels = predicted_edge_pixels & true_edge_pixels
    FP_pixels = predicted_edge_pixels - true_edge_pixels
    FN_pixels = true_edge_pixels - predicted_edge_pixels
    TP_edge = len(TP_pixels)
    FP_edge = len(FP_pixels)
    FN_edge = len(FN_pixels)
    assert TP_edge + FP_edge == len(predicted_edge_pixels)
    assert TP_edge + FN_edge == len(true_edge_pixels)
    precision_edge = TP_edge / (TP_edge + FP_edge)
    recall_edge = TP_edge / (TP_edge + FN_edge)
    f1_edge = 2 * precision_edge * recall_edge / (precision_edge + recall_edge)
    if return_matrix:
        grid_mat_predicted = predicted_grid_coarse.get_grid_matrix()
        grid_mat_true = pge_line_grid_coarse.get_grid_matrix()
        return precision_edge, recall_edge, f1_edge, TP_edge / (
        TP_edge + FP_edge + FN_edge), grid_mat_predicted > 0, grid_mat_true > 0
    else:
        return precision_edge, recall_edge, f1_edge, TP_edge / (TP_edge + FP_edge + FN_edge)

