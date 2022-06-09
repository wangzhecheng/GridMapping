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
from tqdm import tqdm


def seek(origins, targets=None, weights=None, path_handling='link', debug=False, film=False,
         frame_dirname='frames', frame_rate=1000000, early_stop=False):
    if weights is None:
        weights = np.ones(origins.shape)
    if targets is None:
        targets = np.zeros(origins.shape, dtype=np.int8)
    assert targets.shape == origins.shape
    assert targets.shape == weights.shape
    path_handling = path_handling.lower()
    assert path_handling in ['none', 'n', 'assimilate', 'a', 'link', 'l']
    n_rows, n_cols = origins.shape
    if path_handling[0] == 'n':
        path_handling = 0
    elif path_handling[0] == 'a':
        path_handling = 1
    elif path_handling[0] == 'l':
        path_handling = 2

    iteration = 0
    not_visited = 9999999999.

    if film:
        #         frame_rate = int(1e6)
        frame_counter = 100000
        #         frame_dirname = 'frames'
        try:
            os.mkdir(frame_dirname)
        except Exception:
            # NBD
            pass

        cwd = os.getcwd()
        try:
            os.chdir(frame_dirname)
            for filename in os.listdir('.'):
                os.remove(filename)
        except Exception:
            print('Frame deletion failed')
        finally:
            os.chdir(cwd)

    rendering = 1. / (2. * weights)
    rendering = np.minimum(rendering, 1.)
    target_locations = np.where(targets)
    n_targets = target_locations[0].size
    n_targets_remaining = n_targets
    n_targets_remaining_update = n_targets
    for i_target, row in enumerate(target_locations[0]):
        col = target_locations[1][i_target]
        wid = 8
        rendering[
        row - wid:
        row + wid + 1,
        col - wid:
        col + wid + 1] = .5

    # The distance array shows the shortest weighted distance from
    # each point in the grid to the nearest origin point.
    distance = np.ones((n_rows, n_cols)) * not_visited
    origin_locations = np.where(origins != 0)
    distance[origin_locations] = 0.

    # The paths array shows each of the paths that are discovered
    # from targets to their nearest origin point.
    paths = np.zeros((n_rows, n_cols), dtype=np.int8)

    # The halo is the set of points under evaluation. They surround
    # the origin points and expand outward, forming a growing halo
    # around the set of origins that eventually enevlops targets.
    # It is implemented using a heap queue, so that the halo point
    # nearest to an origin is always the next one that gets evaluated.
    halo = []
    for i, origin_row in enumerate(origin_locations[0]):
        origin_col = origin_locations[1][i]
        heapq.heappush(halo, (0., (origin_row, origin_col)))

    # The temporary array for tracking locations to add to the halo.
    # This gets overwritten with each iteration.
    new_locs = np.zeros((int(1e6), 3))
    n_new_locs = 0

    edges = []  #####

    while len(halo) > 0:
        iteration += 1
        if debug:
            if (n_targets_remaining > n_targets_remaining_update or
                            iteration % 1e4 == 0.):
                n_targets_remaining = n_targets_remaining_update
                print('\r {num} targets of {total} reached, {rem} remaining, {halo_len} to try '
                    .format(
                    num=n_targets - n_targets_remaining,
                    total=n_targets,
                    rem=n_targets_remaining,
                    halo_len=len(halo),
                ), end='')
                sys.stdout.flush()
        if film:
            if iteration % frame_rate == 0:
                frame_counter = render(
                    distance,
                    frame_counter,
                    frame_dirname,
                    not_visited,
                    rendering,
                )

        # Reinitialize locations to add.
        new_locs[:n_new_locs, :] = 0.
        n_new_locs = 0

        # Retrieve and check the location with shortest distance.
        (distance_here, (row_here, col_here)) = heapq.heappop(halo)
        n_new_locs, n_targets_remaining_update = nb_loop(
            col_here,
            distance,
            distance_here,
            n_cols,
            n_new_locs,
            n_rows,
            n_targets_remaining,
            new_locs,
            not_visited,
            origins,
            path_handling,
            paths,
            row_here,
            targets,
            weights,
            edges
        )
        for i_loc in range(n_new_locs):
            loc = (int(new_locs[i_loc, 1]), int(new_locs[i_loc, 2]))
            heapq.heappush(halo, (new_locs[i_loc, 0], loc))

        if early_stop and n_targets_remaining_update == 0:  #####
            break

    if debug:
        print('\r                                                 ', end='')
        sys.stdout.flush()
        print('')
    # Add the newfound paths to the visualization.
    rendering = 1. / (1. + distance / 10.)
    rendering[np.where(origins)] = 1.
    rendering[np.where(paths)] = .8
    results = {'paths': paths, 'distance': distance, 'rendering': rendering, 'edges': edges}
    return results


def render(
    distance,
    frame_counter,
    frame_dirname,
    not_visited,
    rendering,
):
    """
    Turn the progress of the algorithm into a pretty picture.
    """
    progress = rendering.copy()
    visited_locs = np.where(distance < not_visited)
    progress[visited_locs] = 1. / (1. + distance[visited_locs] / 10.)
    filename = 'pathfinder_frame_' + str(frame_counter) + '.png'
    cmap = 'inferno'
    dpi = 1200
    plt.figure(33374)
    plt.clf()
    plt.imshow(
        np.flip(progress.transpose(), axis=0),
        origin='higher',
        interpolation='nearest',
        cmap=plt.get_cmap(cmap),
        vmax=1.,
        vmin=0.,
    )
    filename_full = os.path.join(frame_dirname, filename)
    plt.tight_layout()
    plt.savefig(filename_full, dpi=dpi)
    frame_counter += 1
    return frame_counter


def nb_trace_back(
    distance,
    n_new_locs,
    new_locs,
    not_visited,
    origins,
    path_handling,
    paths,
    target,
    weights,
    edges #####
):
    """
    Connect each found electrified target to the grid through
    the shortest available path.
    """
    # Handle the case where you find more than one target.
    path = []
    distance_remaining = distance[target]
    current_location = target
    while distance_remaining > 0.:
        path.append(current_location)
        (row_here, col_here) = current_location
        # Check each of the neighbors for the lowest distance to grid.
        neighbors = [
            ((row_here - 1, col_here), 1.),
            ((row_here + 1, col_here), 1.),
            ((row_here, col_here + 1), 1.),
            ((row_here, col_here - 1), 1.),
            ((row_here - 1, col_here - 1), 2.**.5),
            ((row_here + 1, col_here - 1), 2.**.5),
            ((row_here - 1, col_here + 1), 2.**.5),
            ((row_here + 1, col_here + 1), 2.**.5),
        ]
        lowest_distance = not_visited
        # It's confusing, but keep in mind that
        # distance[neighbor] is the distance from the neighbor position
        # to the grid, while neighbor_distance is
        # the distance *through*
        # the neighbor position to the grid. It is distance[neighbor]
        # plus the distance to the neighbor from the current position.
        for (neighbor, scale) in neighbors:
            if neighbor not in path:
                distance_from_neighbor = scale * weights[current_location]
                neighbor_distance = (distance[neighbor] +
                                     distance_from_neighbor)
                if neighbor_distance < lowest_distance:
                    lowest_distance = neighbor_distance
                    best_neighbor = neighbor

        # This will fail if caught in a local minimum.
        if distance_remaining < distance[best_neighbor]:
            distance_remaining = 0.
            continue

        distance_remaining = distance[best_neighbor]
        current_location = best_neighbor

    # Add this new path.
    for i_loc, loc in enumerate(path):
        paths[loc] = 1
        # If paths are to be linked, include the entire paths as origins and
        # add them to new_locs. If targets are to be assimilated, just add
        # the target (the first point on the path) to origins and new_locs.
        if path_handling == 2 or (
                path_handling == 1 and i_loc == 0):
            origins[loc] = 1
            distance[loc] = 0.
            new_locs[n_new_locs, 0] = 0.
            new_locs[n_new_locs, 1] = loc[0]
            new_locs[n_new_locs, 2] = loc[1]
            n_new_locs += 1
    edges.append((path[0], path[-1])) ##### (target, origin)

    return n_new_locs


def nb_loop(
    col_here,
    distance,
    distance_here,
    n_cols,
    n_new_locs,
    n_rows,
    n_targets_remaining,
    new_locs,
    not_visited,
    origins,
    path_handling,
    paths,
    row_here,
    targets,
    weights,
    edges #####
):
    """
    This is the meat of the computation.
    Pull the computationally expensive operations from seek()
    out into their own function that can be pre-compiled using numba.
    """
    # Calculate the distance for each of the 8 neighbors.
    neighbors = [
        ((row_here - 1, col_here), 1.),
        ((row_here + 1, col_here), 1.),
        ((row_here, col_here + 1), 1.),
        ((row_here, col_here - 1), 1.),
        ((row_here - 1, col_here - 1), 2.**.5),
        ((row_here + 1, col_here - 1), 2.**.5),
        ((row_here - 1, col_here + 1), 2.**.5),
        ((row_here + 1, col_here + 1), 2.**.5),
    ]

    for (neighbor, scale) in neighbors:
        weight = scale * weights[neighbor]
        neighbor_distance = distance_here + weight

        if distance[neighbor] == not_visited:
            if targets[neighbor]:
                n_new_locs = nb_trace_back(
                    distance,
                    n_new_locs,
                    new_locs,
                    not_visited,
                    origins,
                    path_handling,
                    paths,
                    neighbor,
                    weights,
                    edges
                )
                targets[neighbor] = 0
                n_targets_remaining -= 1
        if neighbor_distance < distance[neighbor]:
            distance[neighbor] = neighbor_distance
            if (neighbor[0] > 0 and
                    neighbor[0] < n_rows - 1 and
                    neighbor[1] > 0 and
                    neighbor[1] < n_cols - 1):
                new_locs[n_new_locs, 0] = distance[neighbor]
                new_locs[n_new_locs, 1] = neighbor[0]
                new_locs[n_new_locs, 2] = neighbor[1]
                n_new_locs += 1
    return n_new_locs, n_targets_remaining

