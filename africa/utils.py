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
import simplekml
from simplekml import Kml, Types


def calculate_dist(lat1, lon1, lat2, lon2):
    lat1 = np.deg2rad(lat1)
    lat2 = np.deg2rad(lat2)
    lon1 = np.deg2rad(lon1)
    lon2 = np.deg2rad(lon2)
    dy = (lat1 - lat2) * 6378000
    dx = (lon1 - lon2) * 6378000 * np.cos((lat1 + lat2) / 2)
    dist = np.sqrt(dy**2 + dx**2)
    return dist


def generate_candidate_pair(coord_dict, distance_threshold, spatial_hashing):
    """
    Generate all pairs of nodes which has distance less than distance_threshold
    coord_dict: a dict of id: (lat, lon)
    """
    idx2hv, hv2idx = spatial_hashing.hash_coordinates(coord_dict)
    candidate_pairs = set()
    for idx1 in coord_dict:
        x, y = idx2hv[idx1]
        lat1, lon1 = coord_dict[idx1]
        for dx, dy in [(0, 0), (-1, 0), (1, 0), (0, 1), (0, -1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            if (x + dx, y + dy) in hv2idx:
                for idx2 in hv2idx[(x + dx, y + dy)]:
                    if type(idx1) == str and type(idx2) == int:
                        continue
                    if type(idx1) == type(idx2) and idx1 >= idx2:
                        continue
                    lat2, lon2 = coord_dict[idx2]
                    d = calculate_dist(lat1, lon1, lat2, lon2)
                    if d < distance_threshold:
                        candidate_pairs.add((idx1, idx2, d))
    return candidate_pairs


class SpatialHashing(object):
    def __init__(self, unit, lat_s, lon_w, lat_n, lon_e, dlat0, dlon0):
        """
        unit: side length of each tile in meters
        lat_s, lon_w, lat_n, lon_e: South, West, North, East bound
        dlat0: difference of latitude for one tile
        dlon0: difference of longitude for one tile
        """
        self.unit = unit
        self.lat_s = lat_s
        self.lon_w = lon_w
        self.lat_n = lat_n
        self.lon_e = lon_e
        self.dlat = dlat0 * unit
        self.dlon = dlon0 * unit
        self.x_max = int((lon_e - lon_w) / self.dlon)
        self.y_max = int((lat_n - lat_s) / self.dlat)
        # The grid is (x_max + 1) by (y_max + 1) in 0-index
        print('x_max: ', self.x_max)
        print('y_max: ', self.y_max)

    def x2lon(self, x):
        assert x >= 0 and x <= self.x_max
        return self.lon_w + x * self.dlon

    def y2lat(self, y):
        assert y >= 0 and y <= self.y_max
        return self.lat_s + y * self.dlat

    def lon2x(self, lon):
        x = int((lon - self.lon_w) / self.dlon)
        assert x >= 0 and x <= self.x_max
        return x

    def lat2y(self, lat):
        y = int((lat - self.lat_s) / self.dlat)
        assert y >= 0 and y <= self.y_max
        return y

    def hash_coordinates(self, coord_dict):
        """coordinates: a dict of idx: (lat, lon)"""
        idx2hv = {}  # index to hash value (x, y)
        hv2idx = {}  # hash value (x, y) to index
        for i in coord_dict:
            p = coord_dict[i]
            lat, lon = p
            x = self.lon2x(lon)
            y = self.lat2y(lat)
            idx2hv[i] = (x, y)
            if (x, y) not in hv2idx:
                hv2idx[(x, y)] = []
            hv2idx[(x, y)].append(i)
        return idx2hv, hv2idx


class Grid:
    def __init__(self, spatial_hashing):
        self.x_max = spatial_hashing.x_max
        self.y_max = spatial_hashing.y_max
        self.sh = spatial_hashing
        self.filled_dict = dict()
        self.grid_mat = np.zeros((self.x_max + 1, self.y_max + 1))

    def get_lat_along_line(self, lon, lat1, lat2, lon1, lon2):
        return lat1 + (lat2 - lat1) / (lon2 - lon1) * (lon - lon1)

    def get_lon_along_line(self, lat, lat1, lat2, lon1, lon2):
        return lon1 + (lon2 - lon1) / (lat2 - lat1) * (lat - lat1)

    def fill_single_point(self, point, idx):
        lat, lon = point
        x = self.sh.lon2x(lon)
        y = self.sh.lat2y(lat)
        if (x, y) not in self.filled_dict:
            self.filled_dict[(x, y)] = set()
        self.filled_dict[(x, y)].add(idx)

    def fill_line(self, point1, point2, idx):
        lat1, lon1 = point1
        lat2, lon2 = point2
        xy_set = set()
        assert not (lat1 == lat2 and lon1 == lon2)
        x1 = self.sh.lon2x(lon1)
        x2 = self.sh.lon2x(lon2)
        y1 = self.sh.lat2y(lat1)
        y2 = self.sh.lat2y(lat2)
        xy_set.add((x1, y1))
        xy_set.add((x2, y2))
        for x_cross in range(min(x1, x2) + 1, max(x1, x2) + 1):
            lon_cross = self.sh.x2lon(x_cross)
            lat_cross = self.get_lat_along_line(lon_cross, lat1, lat2, lon1, lon2)
            y_cross = self.sh.lat2y(lat_cross)
            xy_set.add((x_cross, y_cross))
            xy_set.add((x_cross - 1, y_cross))
        for y_cross in range(min(y1, y2) + 1, max(y1, y2) + 1):
            lat_cross = self.sh.y2lat(y_cross)
            lon_cross = self.get_lon_along_line(lat_cross, lat1, lat2, lon1, lon2)
            x_cross = self.sh.lon2x(lon_cross)
            xy_set.add((x_cross, y_cross))
            xy_set.add((x_cross, y_cross - 1))
        for xy in xy_set:
            x, y = xy
            if (x, y) not in self.filled_dict:
                self.filled_dict[(x, y)] = set()
            self.filled_dict[(x, y)].add(idx)

    def fill_single_way_coord(self, coord_list, way_idx):
        if len(coord_list) == 1:
            self.fill_single_point(coord_list[0], way_idx)
        else:
            for i in range(0, len(coord_list) - 1):
                self.fill_line(coord_list[i], coord_list[i + 1], way_idx)

    def construct_from_way_coord_dict(self, way_coord_dict):
        for way_idx in way_coord_dict:
            coord_list = way_coord_dict[way_idx]
            self.fill_single_way_coord(coord_list, way_idx)

    def construct_from_coord_list(self, coord_list):
        for i in range(len(coord_list)):
            self.fill_single_point((coord_list[i][0], coord_list[i][1]), i)

    def get_grid_matrix(self):
        #         self.grid_mat = np.zeros((self.x_max + 1, self.y_max + 1))
        for xy in self.filled_dict:
            x, y = xy
            self.grid_mat[x, y] = 1
        return self.grid_mat


class PolyLine:
    def __init__(self, plid, coord_list, way_type):
        """coord_list must be a list of (lat, lon)"""
        assert len(coord_list) > 1
        self.plid = plid  # PolyLine ID
        self.way_type = way_type
        self.points = coord_list  # nid start from 0 to len(self.npoints) - 1
        self.npoints = len(self.points)
        self.nlines = self.npoints - 1
        self.intersections = {}  # nid -> list of [plid, nid]
        self.intersections_with_poles = {}  # nid -> list of pid
        self.street_views = {}  # sid -> (lid, length)
        self.detected_poles = {}  # dpid -> (lid, length)
        self.virtual_poles = {}  # vpid -> (lid, length)
        self._update_length()

    def _update_length(self):
        self.segment_length = []
        self.cumulative_length = []
        self.total_length = 0
        for i in range(self.nlines):
            lat1, lon1 = self.points[i]
            lat2, lon2 = self.points[i + 1]
            l = calculate_dist(lat1, lon1, lat2, lon2)
            self.total_length += l
            self.segment_length.append(l)
            self.cumulative_length.append(self.total_length)

    def add_new_point(self, coord, lid):
        """Could be used for create new intersection point which is not among any of the existing road points.
            Added to the line indexed by lid.
            coord sould be (lat, lon).
            !!! Warning: this function cannot be run after any following functions, otherwise everything get messed
            up !!!"""
        self.points.insert(lid + 1, coord)
        self.npoints = len(self.points)
        self.nlines = self.npoints - 1
        new_intersections = {}
        for nid in self.intersections:
            if nid <= lid:
                new_intersections[nid] = self.intersections[nid]
            else:
                new_intersections[nid + 1] = self.intersections[nid]
        self.intersections = new_intersections
        self._update_length()

    def _get_equisection_points(self, a, b, n):
        """n=2 means mid-point"""
        pp = []
        for i in range(0, n + 1):
            pp.append(a + i * (b - a) / n)
        return pp

    def coord2length(self, lat, lon, lid):
        lat1, lon1 = self.points[lid]
        lat2, lon2 = self.points[lid + 1]
        if lon1 == lon2:
            lat_approx = lat
            lon_approx = lon1
            proportion = (lat_approx - lat1) / (lat2 - lat1)
        elif lat1 == lat2:
            lat_approx = lat1
            lon_approx = lon
            proportion = (lon_approx - lon1) / (lon2 - lon1)
        else:
            A = np.array([[(lat2 - lat1), -(lon2 - lon1)], [(lon2 - lon1), (lat2 - lat1)]])
            b = np.array([(lat2 - lat1) * lon1 - (lon2 - lon1) * lat1, (lat2 - lat1) * lat + (lon2 - lon1) * lon])
            lon_approx, lat_approx = np.linalg.solve(A, b)
            proportion = (lat_approx - lat1) / (lat2 - lat1)
        # print(proportion)
        proportion = min(max(0, proportion), 1)
        length = proportion * self.segment_length[lid]
        if lid >= 1:
            length += self.cumulative_length[lid - 1]
        return length

    def length2coord(self, length):
        assert -0.01 <= length <= self.total_length + 0.01
        length = min(max(0, length), self.total_length)
        for i, cl in enumerate(self.cumulative_length):
            if cl > length:
                break
        if i == 0:
            proportion = length / self.segment_length[i]
        else:
            proportion = (length - self.cumulative_length[i - 1]) / self.segment_length[i]
        # print(proportion)
        proportion = min(max(0, proportion), 1)
        lat1, lon1 = self.points[i]
        lat2, lon2 = self.points[i + 1]
        lat = proportion * (lat2 - lat1) + lat1
        lon = proportion * (lon2 - lon1) + lon1
        return lat, lon

    def register_pole_by_coord(self, dpid, lat, lon, lid):
        length = self.coord2length(lat, lon, lid)
        self.detected_poles[dpid] = length

    def register_pole_by_nodeid(self, nid, pid, mode='detected'):
        """Could be used for registering virtual pole at the intersection. """
        assert 0 <= nid <= self.npoints - 1
        assert mode == 'detected' or mode == 'virtual'
        if nid == 0:
            length = 0
        else:
            length = self.cumulative_length[nid - 1]
        if mode == 'detected':
            self.detected_poles[pid] = length
        else:
            self.virtual_poles[pid] = length
        if not nid in self.intersections_with_poles:
            self.intersections_with_poles[nid] = set()
        self.intersections_with_poles[nid].add(pid)

    def register_street_view(self, sid, lat, lon, lid):
        length = self.coord2length(lat, lon, lid)
        self.street_views[sid] = length

    def _calculate_distances_for_sequence_of_points(self, lengths, pid1, pid2, poles, mode="crow_flight"):
        """Given a list of lengths in order, calculate the distances (crow flight or along road) between
        every pair of poles in order."""
        npoints = len(lengths)
        dist_list = []
        assert npoints >= 2
        for i in range(0, npoints - 1):
            l1 = lengths[i]
            l2 = lengths[i + 1]
            if mode == "crow_flight":
                lat1, lon1 = self.length2coord(l1)
                if i == 0:
                    lat1, lon1, _ = poles[pid1]
                lat2, lon2 = self.length2coord(l2)
                if i == npoints - 2:
                    lat2, lon2, _ = poles[pid2]
                dist = calculate_dist(lat1, lon1, lat2, lon2)
            else:
                dist = l2 - l1
            dist_list.append(dist)
        return dist_list

    def _insert_virtual_poles_helper(self, pole_dict, poles, distance_threshold=60, max_partitions=5,
                                     dist_mode="crow_flight"):
        """Could be used for insert virtual poles between detected poles that are too far apart, or
            along the road where there is no pole at all.
            Insert poles until distance between any two poles in order is less than distance_threshold
            max_partitions means that there can insert at most (max_partitions - 1) poles.
            pole_dict: a dict of pole_id -> length."""
        assert max_partitions >= 2
        # sort the existing poles by length in ascending order
        sorted_poles = sorted(pole_dict.items(), key=lambda x: x[1])
        n = len(sorted_poles)
        for i in range(n - 1):
            pid1, l1 = sorted_poles[i]
            pid2, l2 = sorted_poles[i + 1]
            if max(self._calculate_distances_for_sequence_of_points([l1, l2], pid1, pid2, poles,
                                                                    dist_mode)) > distance_threshold:
                for npartions in range(2, max_partitions + 1):
                    equisection_points = self._get_equisection_points(l1, l2, npartions)
                    if max(self._calculate_distances_for_sequence_of_points(equisection_points, pid1, pid2, poles,
                                                                            dist_mode)) <= distance_threshold:
                        new_idx = len(self.virtual_poles)
                        for j in range(1, len(equisection_points) - 1):
                            l = equisection_points[j]
                            self.virtual_poles[str(self.plid) + '_' + str(new_idx)] = l
                            new_idx += 1
                        break

    def insert_virtual_poles(self, poles, mode='detected', distance_threshold=60, max_partitions=5,
                             dist_mode="crow_flight"):
        assert mode == 'detected' or mode == 'virtual'
        if mode == 'detected':
            all_poles = self.detected_poles.copy()
            self._insert_virtual_poles_helper(all_poles, poles, distance_threshold, max_partitions, dist_mode)
        else:
            # Firstly, add the start point, and end point to self.virtual_poles
            new_idx = len(self.virtual_poles)
            self.virtual_poles[str(self.plid) + '_' + str(new_idx)] = 0
            new_idx += 1
            self.virtual_poles[str(self.plid) + '_' + str(new_idx)] = self.total_length
            # Secondly, insert more virtual poles.
            self._insert_virtual_poles_helper(self.virtual_poles, poles, distance_threshold, max_partitions, dist_mode)

    def attach_pole_to_intersection(self, distance_threshold=10, mode='detected'):
        assert mode == 'detected' or mode == 'virtual'
        info = []
        if mode == 'detected':
            pole_dict = self.detected_poles
        else:
            pole_dict = self.virtual_poles

        attached_poles = set()
        for nid in self.intersections:
            if nid == 0:
                length = 0
            else:
                length = self.cumulative_length[nid - 1]
            nearest_poles = []
            for pid in pole_dict:
                pole_length = pole_dict[pid]
                if abs(pole_length - length) <= distance_threshold and pid not in attached_poles:
                    nearest_poles.append((pid, abs(pole_length - length)))
            if nearest_poles:
                nearest_poles = sorted(nearest_poles, key=lambda x: x[1])
                pid, _ = nearest_poles[0]
                attached_poles.add(pid)
                if not nid in self.intersections_with_poles:
                    self.intersections_with_poles[nid] = set()
                self.intersections_with_poles[nid].add(pid)
                # change the pole's length to the nearby intersection's length
                if mode == 'detected':
                    self.detected_poles[pid] = length
                else:
                    self.virtual_poles[pid] = length
                info.append((self.intersections[nid],
                             pid))  # This is to inform the intersection points at other ways to be attached by the pole too
        return info

    def collect_info(self):
        """
        Collect the info for generating features, including the sequential order of poles and street views
        between them."""
        all_poles = copy.deepcopy(self.detected_poles)
        self.idx2pole = []
        self.pole2idx = {}
        for pid in self.virtual_poles:
            all_poles[pid] = self.virtual_poles[pid]
        sorted_poles = sorted(all_poles.items(), key=lambda x: x[1])
        for i, p in enumerate(sorted_poles):
            pid, l = p
            self.idx2pole.append(pid)
            self.pole2idx[pid] = i
        poles_and_svs = copy.deepcopy(all_poles)
        for sid in self.street_views:
            poles_and_svs['sv_' + str(sid)] = self.street_views[sid]
        sorted_poles_and_svs = sorted(poles_and_svs.items(), key=lambda x: x[1])
        self.street_view_collections = {}  # pole_index: a list of sid between pole_index and pole_index + 1
        for i in range(-1, len(sorted_poles)):
            self.street_view_collections[i] = []  # initialization
        pole_index = -1
        for i, p in enumerate(sorted_poles_and_svs):
            idx, l = p
            if type(idx) == str and idx[:2] == 'sv':
                self.street_view_collections[pole_index].append(int(idx[3:]))
            else:
                pole_index += 1


def geojson2kml(geojson_path,
                doc_name,
                kml_save_path=None,
                color=simplekml.Color.red,
                altitudemode='relativeToGround',
                altitude=30,
                include_properties=True):
    with open(geojson_path, 'r') as f:
        gj = geojson.load(f)
    features = gj['features']
    kml = Kml()
    kml.document.name = doc_name
    for i, feature in tqdm(enumerate(features)):
        ed = simplekml.ExtendedData()
        if include_properties:
            for k in feature['properties']:
                if feature['properties'][k] is not None:
                    ed.newdata(k, feature['properties'][k])
        if feature['geometry']['type'] == 'Point':
            lon, lat = feature['geometry']['coordinates']
            obj = kml.newpoint(name=str(i),
                               altitudemode=altitudemode,
                               coords=[(lon, lat, altitude)],
                               extendeddata=ed)
            obj.style.iconstyle.color = color

        elif feature['geometry']['type'] == 'LineString':
            coord_list = []
            for lon, lat in feature['geometry']['coordinates']:
                coord_list.append((lon, lat, altitude))
            obj = kml.newlinestring(name=str(i),
                                    altitudemode=altitudemode,
                                    coords=coord_list,
                                    extendeddata=ed)
            obj.style.linestyle.color = color
        else:
            raise
    if kml_save_path:
        kml.save(kml_save_path)
    return kml
