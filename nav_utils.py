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

import sys
import logging
import argparse
import numpy as np
from matplotlib import patches
import matplotlib.pyplot as plt
from awpy.visualization.plot import plot_map, position_transform
from awpy.data import NAV
from awpy.analytics.nav import (
    position_state_distance,
    token_state_distance,
)


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


def trajectory_distance(
    map_name,
    trajectory_array_1,
    trajectory_array_2,
    distance_type="geodesic",
):
    """Calculates a distance distance between two trajectories

    Args:
        map_name (string): Map under consideration
        trajectory_array_1: Numpy array with shape (n_Time,2|1, 5, 3) with the first index indicating the team, the second the player and the third the coordinate
        trajectory_array_2: Numpy array with shape (n_Time,2|1, 5, 3) with the first index indicating the team, the second the player and the third the coordinate
        distance_type: String indicating how the distance between two player positions should be calculated. Options are "geodesic", "graph", "euclidean" and "edit_distance"
        precomputed_areas (boolean): Indicates whether the position arrays already contain the precomputed areas in the x coordinate of the position

    Returns:
        A float representing the distance between these two trajectories
    """
    distance = 0
    length = max(len(trajectory_array_1), len(trajectory_array_2))
    if len(trajectory_array_1.shape) > 2.5:
        for time in range(length):
            distance += (
                position_state_distance(
                    map_name=map_name,
                    position_array_1=trajectory_array_1[time]
                    if time in range(len(trajectory_array_1))
                    else trajectory_array_1[-1],
                    position_array_2=trajectory_array_2[time]
                    if time in range(len(trajectory_array_2))
                    else trajectory_array_2[-1],
                    distance_type=distance_type,
                )
                / length
            )
    else:
        for time in range(length):
            distance += (
                token_state_distance(
                    map_name=map_name,
                    token_array_1=trajectory_array_1[time]
                    if time in range(len(trajectory_array_1))
                    else trajectory_array_1[-1],
                    token_array_2=trajectory_array_2[time]
                    if time in range(len(trajectory_array_2))
                    else trajectory_array_2[-1],
                    distance_type=distance_type,
                )
                / length
            )
    return distance


def trajectory_distance_wrapper(args):
    """Calculates a distance distance between two trajectories

    Args:
        args (tuple): Contains the arguments to pass to trajectory_distance. Order in the tuple is (map_name,array1,array2,distance_type)

    Returns:
        A float representing the distance between these two trajectories
    """
    map_name, trajectory_array_1, trajectory_array_2, distance_type = args
    return trajectory_distance(
        map_name=map_name,
        trajectory_array_1=trajectory_array_1,
        trajectory_array_2=trajectory_array_2,
        distance_type=distance_type,
    )


def transform_to_traj_dimensions(pos_array):
    """Transforms a numpy array of shape (Time,5,3) to (5,Time,1,1,3) to allow for individual trajectory distances.
    Only needed when trying to get the distance between single player trajectories within "get_shortest_distance_mapping"

    Args:
        pos_array (numpy array): Numpy array with shape (Time, 2|1, 5, 3)

        Returns:
            numpy array
    """
    shape = pos_array.shape
    dimensions = [shape[1], len(pos_array), 1, 1, shape[2]]
    return_array = np.zeros(tuple(dimensions))
    for i in range(5):
        return_array[i, :, :, :, :] = pos_array[:, np.newaxis, np.newaxis, i, :]
    return return_array


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

    # logging.info(PLACE_DIST_MATRIX["de_nuke"]["TSpawn"]["Silo"])
    # logging.info(PLACE_DIST_MATRIX["de_nuke"]["Silo"]["TSpawn"])


if __name__ == "__main__":
    main(sys.argv[1:])
