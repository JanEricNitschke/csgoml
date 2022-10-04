"""Collection of functions to extend awpy's navigation capabilities


    Typical usage example:

    area_matrix = get_area_distance_matrix()
    logging.info(area_matrix["de_dust2"]["ExtendedA"]["CTSpawn"])
    logging.info(area_matrix["de_dust2"]["CTSpawn"]["ExtendedA"])

    centroids, reps = generate_centroids("de_dust2")
    graph = area_distance(
        "de_dust2", centroids["ExtendedA"], centroids["CTSpawn"], dist_type="graph"
    )
    geodesic = area_distance(
        "de_dust2",  centroids["ExtendedA"], centroids["CTSpawn"], dist_type="geodesic"
    )
    plot_path("de_dust2", graph, geodesic)
    graph = area_distance(
        "de_dust2", centroids["CTSpawn"], centroids["ExtendedA"], dist_type="graph"
    )
    geodesic = area_distance(
        "de_dust2", centroids["CTSpawn"],  centroids["ExtendedA"], dist_type="geodesic"
    )
    plot_path("de_dust2", graph, geodesic)
    graph = area_distance(
        "de_dust2", reps["ExtendedA"], reps["CTSpawn"], dist_type="graph"
    )
    geodesic = area_distance(
        "de_dust2",  reps["ExtendedA"], reps["CTSpawn"], dist_type="geodesic"
    )
    plot_path("de_dust2", graph, geodesic)
    graph = area_distance(
        "de_dust2", reps["CTSpawn"], reps["ExtendedA"], dist_type="graph"
    )
    geodesic = area_distance(
        "de_dust2", reps["CTSpawn"],  reps["ExtendedA"], dist_type="geodesic"
    )
    plot_path("de_dust2", graph, geodesic)


"""
#!/usr/bin/env python

from collections import defaultdict
import itertools
import os
import sys
import math
import time
import timeit
import logging
import json
import argparse
from statistics import mean, median
import numpy as np
from shapely.geometry import Polygon
from matplotlib import patches
import matplotlib.pyplot as plt
from awpy.visualization.plot import plot_map, position_transform
from awpy.data import NAV, AREA_DIST_MATRIX, PLACE_DIST_MATRIX
from awpy.analytics.nav import (
    area_distance,
    find_closest_area,
    generate_area_distance_matrix,
    point_distance,
    generate_place_distance_matrix,
    tree,
)
from plotting_utils import get_areas_hulls_centers, stepped_hull
from pathlib import Path


def mark_areas(map_name, areas):
    """Plots the given map and marks the tiles of the two given areas.

    Args:
        map_name (string): Map to plot
        area (list[int]): List of ids of the areas to highlight on the map

    Returns:
        None (Directly saves the plot to file)
    """
    fig, axis = plot_map(map_name=map_name, map_type="simplerader", dark=True)
    fig.set_size_inches(19.2, 10.8)
    for a in NAV[map_name]:
        area = NAV[map_name][a]
        if a not in areas:
            continue
        south_east_x = position_transform(map_name, area["southEastX"], "x")
        north_west_x = position_transform(map_name, area["northWestX"], "x")
        south_east_y = position_transform(map_name, area["southEastY"], "y")
        north_west_y = position_transform(map_name, area["northWestY"], "y")
        # Get its lower left points, height and width
        width = south_east_x - north_west_x
        height = north_west_y - south_east_y
        southwest_x = north_west_y
        southwest_y = south_east_y
        rect = patches.Rectangle(
            (southwest_x, southwest_y),
            width,
            height,
            linewidth=1,
            edgecolor="red",
            facecolor="None",
        )
        axis.add_patch(rect)
    plt.savefig(
        f"D:\\CSGO\\ML\\CSGOML\\Plots\\distance_matrix_fails\\{map_name}_{'_'.join(str(area) for area in areas)}.png",
        bbox_inches="tight",
        dpi=1000,
    )
    fig.clear()
    plt.close(fig)


def plot_path(map_name, graph, geodesic):
    """Plots the given map and two paths between two areas.

    Args:
        map_name (string): Map to plot
        graph (dict): Distance dictionary. The graph["distance"] is the distance between two areas.
                      graph["areas"] is a list of int area ids in the shortest path between the first and last entry.
        geodesic (dict): Distance dictionary. The geodesic["distance"] is the distance between two areas.
                      geodesic["areas"] is a list of int area ids in the shortest path between the first and last entry.

    Returns:
        None (Directly saves the plot to file)
    """
    fig, axis = plot_map(map_name=map_name, map_type="simplerader", dark=True)
    fig.set_size_inches(19.2, 10.8)
    for a in NAV[map_name]:
        area = NAV[map_name][a]
        if a not in graph["areas"] and a not in geodesic["areas"]:
            continue
        try:
            south_east_x = position_transform(map_name, area["southEastX"], "x")
            north_west_x = position_transform(map_name, area["northWestX"], "x")
            south_east_y = position_transform(map_name, area["southEastY"], "y")
            north_west_y = position_transform(map_name, area["northWestY"], "y")
        except KeyError:
            pass
        # Get its lower left points, height and width
        width = south_east_x - north_west_x
        height = north_west_y - south_east_y
        southwest_x = north_west_x
        southwest_y = south_east_y
        color = "yellow"
        if a in graph["areas"] and a in geodesic["areas"]:
            color = "green"
        elif a in graph["areas"]:
            color = "red"
        elif a in geodesic["areas"]:
            color = "blue"
        if graph["areas"] and a == graph["areas"][-1]:
            color = "orange"
        if graph["areas"] and a == graph["areas"][0]:
            color = "purple"
        rect = patches.Rectangle(
            (southwest_x, southwest_y),
            width,
            height,
            linewidth=1,
            edgecolor=color,
            facecolor="None",
        )
        axis.add_patch(rect)
    plt.savefig(
        f"D:\\CSGO\\ML\\CSGOML\\Plots\\distance_matrix_examples\\{map_name}_{graph['areas'][0]}_{graph['areas'][-1]}_{graph['distance']}_{geodesic['distance']}.png",
        bbox_inches="tight",
        dpi=1000,
    )
    fig.clear()
    plt.close(fig)


def plot_mid(map_name):
    """Plots the given map with the hulls, centroid and a representative_point for each named area.

    Args:
        map_name (string): Map to plot

    Returns:
        None (Directly saves the plot to file)
    """
    cent_ids, rep_ids = generate_centroids(map_name)
    output_path = r"D:\\CSGO\\ML\\CSGOML\\Plots\\"
    fig, axis = plot_map(map_name=map_name, map_type="simpleradar", dark=True)
    fig.set_size_inches(19.2, 10.8)
    # Grab points, hull and centroid for each named area
    area_points, hulls, _ = get_areas_hulls_centers(map_name)
    for area in area_points:
        # Dont plot the "" area as it stretches across the whole map and messes with clarity
        # but add it to the legend
        if not area:
            axis.plot(np.NaN, np.NaN, label="None")
            continue
        axis.plot(hulls[area][:, 0], hulls[area][:, 1], "-", lw=3, label=area)
    handles, labels = axis.get_legend_handles_labels()
    lgd = axis.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1.01))
    for a in NAV[map_name]:
        area = NAV[map_name][a]
        if a not in cent_ids.values() and a not in rep_ids.values():
            continue
        # logging.info(a)
        # logging.info(NAV["de_dust2"][a])
        try:
            south_east_x = position_transform(map_name, area["southEastX"], "x")
            north_west_x = position_transform(map_name, area["northWestX"], "x")
            south_east_y = position_transform(map_name, area["southEastY"], "y")
            north_west_y = position_transform(map_name, area["northWestY"], "y")
        except KeyError:
            pass
        # Get its lower left points, height and width
        width = south_east_x - north_west_x
        height = north_west_y - south_east_y
        southwest_x = north_west_x
        southwest_y = south_east_y
        color = "yellow"
        if a in cent_ids.values() and a in rep_ids.values():
            color = "green"
        elif a in cent_ids.values():
            color = "red"
        elif a in rep_ids.values():
            color = "blue"
        rect = patches.Rectangle(
            (southwest_x, southwest_y),
            width,
            height,
            linewidth=1,
            edgecolor=color,
            facecolor="None",
        )
        axis.add_patch(rect)
    plt.savefig(
        os.path.join(output_path, f"test_{map_name}.png"),
        bbox_inches="tight",
        bbox_extra_artists=(lgd,),
        dpi=1000,
    )
    fig.clear()
    plt.close(fig)


def get_tile_distance_matrix(
    save=False, json_path="D:\\CSGO\\ML\\CSGOML\\data\\tile_dist_matrix.json"
):
    """Generates or grabs a tree like nested dictionary containing distance matrices (as dicts) for each map for all tiles
    Structures is [dist_type(euclidean,graph,geodesic)][map_name][area1id][area2id]

    Args:
        save (boolean): Whether to also save the dictionary as a json file
        json_path (string): Path to either grab an existing matrix from a save a new one to

    Returns:
        Tree structure containing distances for all tile pairs on all maps
    """
    if os.path.exists(json_path):
        with open(json_path, encoding="utf8") as f:
            tile_distance_matrix = json.load(f)
        return tile_distance_matrix
    tile_distance_matrix = {
        "euclidean": tree(),
        "graph": tree(),
        "geodesic": tree(),
    }
    for map_name, tiles in NAV.items():
        logging.debug(map_name)
        for area1 in tiles:
            area1 = int(area1)
            area1_x = (
                NAV[map_name][area1]["southEastX"] + NAV[map_name][area1]["northWestX"]
            ) / 2
            area1_y = (
                NAV[map_name][area1]["southEastY"] + NAV[map_name][area1]["northWestY"]
            ) / 2
            area1_z = (
                NAV[map_name][area1]["southEastZ"] + NAV[map_name][area1]["northWestZ"]
            ) / 2
            for area2 in tiles:
                area2 = int(area2)
                area2_x = (
                    NAV[map_name][area2]["southEastX"]
                    + NAV[map_name][area2]["northWestX"]
                ) / 2
                area2_y = (
                    NAV[map_name][area2]["southEastY"]
                    + NAV[map_name][area2]["northWestY"]
                ) / 2
                area2_z = (
                    NAV[map_name][area2]["southEastZ"]
                    + NAV[map_name][area2]["northWestZ"]
                ) / 2
                tile_distance_matrix["euclidean"][map_name][area1][area2] = math.sqrt(
                    (area1_x - area2_x) ** 2
                    + (area1_y - area2_y) ** 2
                    + (area1_z - area2_z) ** 2
                )
                graph = area_distance(map_name, area1, area2, dist_type="graph")
                tile_distance_matrix["graph"][map_name][area1][area2] = graph[
                    "distance"
                ]
                geodesic = area_distance(map_name, area1, area2, dist_type="geodesic")
                tile_distance_matrix["geodesic"][map_name][area1][area2] = geodesic[
                    "distance"
                ]
    if save:
        with open(json_path, "w", encoding="utf8") as json_file:
            json.dump(tile_distance_matrix, json_file)
    return tile_distance_matrix


def get_area_distance_matrix(
    tile_distance_matrix=None,
    save=False,
    json_path="D:\\CSGO\\ML\\CSGOML\\data\\area_dist_matrix.json",
):
    """Generates or grabs a tree like nested dictionary containing distance matrices (as dicts) for each map for all regions
    Structures is [map_name][area1id][area2id][dist_type(euclidean,graph,geodesic)][reference_point(centroid,representative_point,median)]

    Args:
        tile_distance_matrix (tree): A tree like structure containing the distances between all tile combinations
        save (boolean): Whether to also save the dictionary as a json file
        json_path (string): Path to either grab an existing matrix from a save a new one to

    Returns:
        Tree structure containing distances for all tile pairs on all maps
    """
    if os.path.exists(json_path):
        with open(json_path, encoding="utf8") as f:
            area_distance_matrix = json.load(f)
        return area_distance_matrix
    area_distance_matrix = tree()
    for dist_type in tile_distance_matrix:
        logging.debug("Dist_type: %s", dist_type)
        for map_name, tiles in NAV.items():
            logging.debug("Map_name: %s", map_name)
            area_mapping = defaultdict(list)
            for area in tiles:
                area_mapping[tiles[area]["areaName"]].append(area)
            centroids, reps = generate_centroids(map_name)
            for area1, centroid1 in centroids.items():
                logging.debug("area1: %s", area1)
                for area2, centroid2 in centroids.items():
                    logging.debug("area2: %s", area2)
                    if tile_distance_matrix is None:
                        area_distance_matrix[map_name][area1][area2][dist_type][
                            "centroid"
                        ] = area_distance(
                            map_name, centroid1, centroid2, dist_type=dist_type
                        )[
                            "distance"
                        ]
                        area_distance_matrix[map_name][area1][area2][dist_type][
                            "representative_point"
                        ] = area_distance(
                            map_name, reps[area1], reps[area2], dist_type=dist_type
                        )[
                            "distance"
                        ]
                        connections = []
                        for sub_area1 in area_mapping[area1]:
                            for sub_area2 in area_mapping[area2]:
                                connections.append(
                                    area_distance(
                                        map_name,
                                        sub_area1,
                                        sub_area2,
                                        dist_type=dist_type,
                                    )["distance"]
                                )
                        area_distance_matrix[map_name][area1][area2][dist_type][
                            "median_dist"
                        ] = median(connections)
                    else:
                        area_distance_matrix[map_name][area1][area2][dist_type][
                            "centroid"
                        ] = tile_distance_matrix[dist_type][map_name][str(centroid1)][
                            str(centroid2)
                        ]
                        area_distance_matrix[map_name][area1][area2][dist_type][
                            "representative_point"
                        ] = tile_distance_matrix[dist_type][map_name][str(reps[area1])][
                            str(reps[area2])
                        ]
                        connections = []
                        for sub_area1 in area_mapping[area1]:
                            for sub_area2 in area_mapping[area2]:
                                connections.append(
                                    tile_distance_matrix[dist_type][map_name][
                                        str(sub_area1)
                                    ][str(sub_area2)]
                                )
                        area_distance_matrix[map_name][area1][area2][dist_type][
                            "median_dist"
                        ] = median(connections)
    if save:
        with open(json_path, "w", encoding="utf8") as json_file:
            json.dump(area_distance_matrix, json_file)
    return area_distance_matrix


def generate_centroids(map_name):
    """For each region in the given map calculates the centroid and a representative point and finds the closest tile for each

    Args:
        map_name (string): Name of the map for which to calculate the centroids

    Returns:
        Tuple of dictionaries containing the centroid and representative tiles for each region of the map
    """
    area_points = defaultdict(list)
    z_s = defaultdict(list)
    area_ids_cent = {}
    area_ids_rep = {}
    for a in NAV[map_name]:
        area = NAV[map_name][a]
        cur_x = []
        cur_y = []
        cur_x.append(area["southEastX"])
        cur_x.append(area["northWestX"])
        cur_y.append(area["southEastY"])
        cur_y.append(area["northWestY"])
        z_s[area["areaName"]].append(area["northWestZ"])
        z_s[area["areaName"]].append(area["southEastZ"])
        for x, y in itertools.product(cur_x, cur_y):
            area_points[area["areaName"]].append((x, y))
    for area in area_points:
        hull = np.array(stepped_hull(area_points[area]))
        my_centroid = list(np.array(Polygon(hull).centroid.coords)[0]) + [
            mean(z_s[area])
        ]
        rep_point = list(np.array(Polygon(hull).representative_point().coords)[0]) + [
            mean(z_s[area])
        ]
        area_ids_cent[area] = find_closest_area(map_name, my_centroid)["areaId"]
        area_ids_rep[area] = find_closest_area(map_name, rep_point)["areaId"]
    return area_ids_cent, area_ids_rep


def main(args):
    """Uses awpy and the extension functions defined in this module to plots player positions in 1 or 10 rounds, position tokens in 1 round
    and each maps nav mesh and named areas."""
    parser = argparse.ArgumentParser("Analyze the early mid fight on inferno")
    parser.add_argument(
        "-d", "--debug", action="store_true", default=False, help="Enable debug output."
    )
    parser.add_argument(
        "-l",
        "--log",
        default=r"D:\CSGO\ML\CSGOML\logs\nav_tests.log",
        help="Path to output log.",
    )
    options = parser.parse_args(args)

    if options.log == "None":
        options.log = None
    if options.debug:
        logging.basicConfig(
            filename=options.log,
            encoding="utf-8",
            level=logging.DEBUG,
            filemode="w",
            format="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        logging.basicConfig(
            filename=options.log,
            encoding="utf-8",
            level=logging.INFO,
            filemode="w",
            format="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    for map_name in NAV:
        generate_area_distance_matrix(map_name, save=True)
    # for map_name in NAV:
    #     generate_place_distance_matrix(map_name, save=True)


if __name__ == "__main__":
    main(sys.argv[1:])
