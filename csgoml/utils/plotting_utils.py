#!/usr/bin/env python
"""Collection of functions to extend awpy plotting capabilites.

Example::

    plot_round_tokens(
        filename=plot_token_file,
        frames=frames,
        map_name=demo_map_name,
        map_type="simpleradar",
        dark=False,
        fps=2,
    )
    for map_name in NAV:
        plot_map_areas(
            output_path=options.output,
            map_name=map_name,
            map_type="simpleradar",
            dark=False,
        )
        plot_map_tiles(
            output_path=options.output,
            map_name=map_name,
            map_type="simpleradar",
            dark=False,
        )
"""

# pylint: disable=consider-using-enumerate

import argparse
import collections
import itertools
import logging
import os
import shutil
import sys
from typing import Literal, TypedDict

import imageio.v3 as imageio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from awpy.analytics.nav import (
    area_distance,
    generate_centroids,
    generate_position_token,
    point_distance,
    stepped_hull,
)
from awpy.data import AREA_DIST_MATRIX, MAP_DATA, NAV, NAV_GRAPHS
from awpy.types import DistanceType, GameFrame, PlotPosition
from awpy.visualization.plot import (
    plot_map,
    plot_positions,
    position_transform,
)
from matplotlib import patches
from tqdm import tqdm

from csgoml.utils.nav_utils import trajectory_distance, transform_to_traj_dimensions


def get_areas_hulls_centers(
    map_name: str,
) -> tuple[
    dict[str, list[tuple[float, float]]],
    dict[str, np.ndarray],
    dict[str, tuple[float, float]],
]:
    """Gets the sets of points making up the named areas of a map.

    Then builds their hull and centroid.

    Args:
        map_name (str): Specifying the map for which the features should be build.

    Returns:
        Three dictionary containing the
        points, hull and centroid of each named area in the map
    """
    hulls: dict[str, np.ndarray] = {}
    centers: dict[str, tuple[float, float]] = {}
    area_points: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
    for a in NAV[map_name]:
        area = NAV[map_name][a]
        try:
            cur_x = [
                position_transform(map_name, area["southEastX"], "x"),
                position_transform(map_name, area["northWestX"], "x"),
            ]
            cur_y = [
                position_transform(map_name, area["southEastY"], "y"),
                position_transform(map_name, area["northWestY"], "y"),
            ]
        except KeyError:
            cur_x = [area["southEastX"], area["northWestX"]]
            cur_y = [area["southEastY"], area["northWestY"]]
        for x, y in itertools.product(cur_x, cur_y):
            area_points[area["areaName"]].append((x, y))
    for area in area_points:
        points_array = np.array(area_points[area])
        hull = np.array(stepped_hull(area_points[area]))
        hulls[area] = hull
        centers[area] = points_array.mean(axis=0)
    return area_points, hulls, centers


def plot_round_tokens(
    filename: str,
    frames: list[GameFrame],
    map_name: str = "de_ancient",
    map_type: str = "original",
    *,
    dark: bool = False,
    fps: int = 2,
    dpi: int = 300,
) -> Literal[True]:
    """Plots the position tokens of a round and saves as a .gif.

    CTs are blue, Ts are orange. Only use untransformed coordinates.

    Args:
        filename (str): Filename to save the gif
        frames (list): List of frames from a parsed demo
        map_name (string): Map to search
        map_type (string): "original" or "simpleradar"
        dark (boolean): Only for use with map_type="simpleradar".
            Indicates if you want to use the SimpleRadar dark map type
        fps (integer): Number of frames per second in the gif
        dpi (int): DPI of the resulting gif

    Returns:
        True (result is directly saved to disk)
    """
    if os.path.isdir("csgo_tmp"):
        shutil.rmtree("csgo_tmp/")
    os.mkdir("csgo_tmp")
    # Colors of borders of each area depending on number of t's/ct's that reside in it
    colors = {
        "t": ["lightsteelblue", "#300000", "#7b0000", "#b10000", "#c80000", "#ff0000"],
        "ct": ["lightsteelblue", "#360CCD", "#302DD9", "#2A4EE5", "#246FF0", "#1E90FC"],
    }
    image_files = []
    # Grab points, hull and centroid for each named area of the map
    area_points, hulls, centers = get_areas_hulls_centers(map_name)
    # Loop over each frame of the round
    for frame_id, frame in tqdm(enumerate(frames)):
        fig, axis = plot_map(map_name=map_name, map_type=map_type, dark=dark)
        axis.get_xaxis().set_visible(b=False)
        axis.get_yaxis().set_visible(b=False)
        # Grab the position token
        tokens = generate_position_token(map_name, frame)
        # Loop over each named area
        for area_id, area in enumerate(sorted(area_points)):
            text = ""
            # Plot each area twice. Once for each side.
            # If both sides have players in one tile the colors get added as
            # the alphas arent 1.
            for side in ["t", "ct"]:
                # Dont plot the "" area as it stretches across
                # the whole map and messes with clarity
                if not area:
                    continue
                # Plot the hull of the area
                axis.plot(
                    hulls[area][:, 0],
                    hulls[area][:, 1],
                    "-",
                    alpha=0.8 if int(tokens[f"{side}Token"][area_id]) > 0 else 0.2,
                    c=colors[side][int(tokens[f"{side}Token"][area_id])],
                    lw=3,
                )
                if tokens[f"{side}Token"][area_id] != "0":
                    text += f'{side}:{tokens[f"{side}Token"][area_id]} '
            # If there are any players inside the area plot a text
            if int(tokens["ctToken"][area_id]) + int(tokens["tToken"][area_id]) > 0:
                axis.text(
                    centers[area][0] - (centers[area][0] - hulls[area][0][0]) / 1.5,
                    centers[area][1],
                    text,
                    c="#EB0841",
                    size="x-small",
                )
        image_files.append(f"csgo_tmp/{frame_id}.png")
        fig.savefig(image_files[-1], dpi=dpi, bbox_inches="tight")
        plt.close()
    images = [imageio.imread(file) for file in image_files]
    imageio.imwrite(filename, images, duration=1000 / fps)
    shutil.rmtree("csgo_tmp/")
    return True


def plot_map_areas(
    output_path: str,
    map_name: str = "de_ancient",
    map_type: str = "original",
    *,
    dark: bool = False,
    dpi: int = 1000,
) -> None:
    """Plot all named areas in the given map.

    Args:
        output_path (string): Path to the output folder
        map_name (string): Map to search
        map_type (string): "original" or "simpleradar"
        dark (boolean): Only for use with map_type="simpleradar".
            Indicates if you want to use the SimpleRadar dark map type
        dpi (int): DPI of the resulting image

    Returns:
        None, saves .png
    """
    logging.info("Plotting areas for %s", map_name)
    fig, axis = plot_map(map_name=map_name, map_type=map_type, dark=dark)
    fig.set_size_inches(19.2, 10.8)
    cent_ids, _ = generate_centroids(map_name)
    # Grab points, hull and centroid for each named area
    area_points, hulls, centers = get_areas_hulls_centers(map_name)
    for area in sorted(area_points):
        # Dont plot the "" area as it stretches across
        # the whole map and messes with clarity
        # but add it to the legend
        if not area:
            axis.plot(np.NaN, np.NaN, label="None")
            continue
        text_y = centers[area][1]
        if (
            map_name in MAP_DATA
            and "z_cutoff" in MAP_DATA[map_name]
            and (
                NAV[map_name][cent_ids[area]]["southEastZ"]
                + NAV[map_name][cent_ids[area]]["northWestZ"]
            )
            / 2
            < MAP_DATA[map_name]["z_cutoff"]
        ):
            hulls[area][:, 1] = np.array([y + 1024 for y in hulls[area][:, 1]])
            text_y += 1024
        axis.plot(hulls[area][:, 0], hulls[area][:, 1], "-", lw=3, label=area)
        # Add name of area into its middle
        axis.text(
            centers[area][0] - (centers[area][0] - hulls[area][0][0]) / 1.5,
            text_y,
            area,
            c="#EB0841",
        )
    # Place legend outside of the plot
    handles, labels = axis.get_legend_handles_labels()
    lgd = axis.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1.01))
    plt.savefig(
        os.path.join(output_path, f"hulls_{map_name}.png"),
        bbox_inches="tight",
        bbox_extra_artists=(lgd,),
        dpi=dpi,
    )
    plt.close()


def plot_mid(
    output_path: str = r"D:\\CSGO\\ML\\CSGOML\\Plots\\",
    map_name: str = "de_ancient",
    dpi: int = 1000,
) -> None:
    """Plots map with the hulls, centroid and a representative_point for each place.

    Args:
        output_path (string): Path to the output folder
        map_name (string): Map to plot
        dpi (int): DPI of the resulting image

    Returns:
        None (Directly saves the plot to file)
    """
    logging.info("Plotting mid points for %s", map_name)
    cent_ids, rep_ids = generate_centroids(map_name)
    fig, axis = plot_map(map_name=map_name, map_type="simpleradar", dark=True)
    fig.set_size_inches(19.2, 10.8)
    # Grab points, hull and centroid for each named area
    area_points, hulls, _ = get_areas_hulls_centers(map_name)
    for area_name in area_points:
        # Dont plot the "" area as it stretches across
        # the whole map and messes with clarity
        # but add it to the legend
        if not area_name:
            axis.plot(np.NaN, np.NaN, label="None")
            continue
        if (
            map_name in MAP_DATA
            and "z_cutoff" in MAP_DATA[map_name]
            and (
                NAV[map_name][cent_ids[area_name]]["southEastZ"]
                + NAV[map_name][cent_ids[area_name]]["northWestZ"]
            )
            / 2
            < MAP_DATA[map_name]["z_cutoff"]
        ):
            hulls[area_name][:, 1] = np.array(
                [y + 1024 for y in hulls[area_name][:, 1]]
            )
        axis.plot(
            hulls[area_name][:, 0], hulls[area_name][:, 1], "-", lw=3, label=area_name
        )
    handles, labels = axis.get_legend_handles_labels()
    lgd = axis.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1.01))
    for area_id in NAV[map_name]:
        area = NAV[map_name][area_id]
        if area_id not in cent_ids.values() and area_id not in rep_ids.values():
            continue
        try:
            south_east_x = position_transform(map_name, area["southEastX"], "x")
            north_west_x = position_transform(map_name, area["northWestX"], "x")
            south_east_y = position_transform(map_name, area["southEastY"], "y")
            north_west_y = position_transform(map_name, area["northWestY"], "y")
        except KeyError:
            south_east_x = area["southEastX"]
            north_west_x = area["northWestX"]
            south_east_y = area["southEastY"]
            north_west_y = area["northWestY"]
        # Get its lower left points, height and width
        width = south_east_x - north_west_x
        height = north_west_y - south_east_y
        southwest_x = north_west_x
        southwest_y = south_east_y
        south_east_z = area["southEastZ"]
        north_west_z = area["northWestZ"]
        avg_z = (south_east_z + north_west_z) / 2
        if (
            map_name in MAP_DATA
            and "z_cutoff" in MAP_DATA[map_name]
            and avg_z < MAP_DATA[map_name]["z_cutoff"]
        ):
            southwest_y += 1024
        color = "yellow"
        if area_id in cent_ids.values() and area_id in rep_ids.values():
            color = "green"
        elif area_id in cent_ids.values():
            color = "red"
        elif area_id in rep_ids.values():
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
        os.path.join(output_path, f"midpoints_{map_name}.png"),
        bbox_inches="tight",
        bbox_extra_artists=(lgd,),
        dpi=dpi,
    )
    fig.clear()
    plt.close(fig)


def plot_map_tiles(
    output_path: str,
    map_name: str = "de_ancient",
    map_type: str = "original",
    *,
    dark: bool = False,
    dpi: int = 1000,
) -> None:
    """Plot all navigation mesh tiles in a given map.

    Args:
        output_path (string): Path to the output folder
        map_name (string): Map to search
        map_type (string): "original" or "simpleradar"
        dark (boolean): Only for use with map_type="simpleradar".
            Indicates if you want to use the SimpleRadar dark map type
        dpi (int): DPI of the resulting image

    Returns:
        None, saves .png
    """
    logging.info("Plotting tiles for %s", map_name)
    fig, axis = plot_map(map_name=map_name, map_type=map_type, dark=dark)
    fig.set_size_inches(19.2, 21.6)
    # Loop over each nav mesh tile
    for a in NAV[map_name]:
        area = NAV[map_name][a]
        south_east_x = position_transform(map_name, area["southEastX"], "x")
        north_west_x = position_transform(map_name, area["northWestX"], "x")
        south_east_y = position_transform(map_name, area["southEastY"], "y")
        north_west_y = position_transform(map_name, area["northWestY"], "y")
        south_east_z = area["southEastZ"]
        north_west_z = area["northWestZ"]
        avg_z = (south_east_z + north_west_z) / 2
        # Get its lower left points, height and width
        width = south_east_x - north_west_x
        height = north_west_y - south_east_y
        southwest_x = north_west_x
        southwest_y = south_east_y
        if (
            map_name in MAP_DATA
            and "z_cutoff" in MAP_DATA[map_name]
            and avg_z < MAP_DATA[map_name]["z_cutoff"]
        ):
            southwest_y += 1024
        rect = patches.Rectangle(
            (southwest_x, southwest_y),
            width,
            height,
            linewidth=1,
            edgecolor="yellow",
            facecolor="None",
        )
        axis.add_patch(rect)
    plt.savefig(
        os.path.join(output_path, f"tiles_{map_name}.png"),
        bbox_inches="tight",
        dpi=dpi,
    )
    fig.clear()
    plt.close()


def plot_map_connections(
    output_path: str,
    map_name: str = "de_ancient",
    map_type: str = "original",
    *,
    dark: bool = False,
    dpi: int = 1000,
) -> None:
    """Plot all navigation mesh tiles in a given map.

    Args:
        output_path (string): Path to the output folder
        map_name (string): Map to search
        map_type (string): "original" or "simpleradar"
        dark (boolean): Only for use with map_type="simpleradar".
            Indicates if you want to use the SimpleRadar dark map type
        dpi (int): DPI of the resulting image

    Returns:
        None, saves .png
    """
    logging.info("Plotting connections for %s", map_name)
    fig, axis = plot_map(map_name=map_name, map_type=map_type, dark=dark)
    fig.set_size_inches(19.2, 21.6)
    # Loop over each nav mesh tile
    for a in NAV[map_name]:
        area = NAV[map_name][a]
        south_east_x = position_transform(map_name, area["southEastX"], "x")
        north_west_x = position_transform(map_name, area["northWestX"], "x")
        south_east_y = position_transform(map_name, area["southEastY"], "y")
        north_west_y = position_transform(map_name, area["northWestY"], "y")
        south_east_z = area["southEastZ"]
        north_west_z = area["northWestZ"]
        avg_z = (south_east_z + north_west_z) / 2
        # Get its lower left points, height and width
        width = south_east_x - north_west_x
        height = north_west_y - south_east_y
        southwest_x = north_west_x
        southwest_y = south_east_y
        if (
            map_name in MAP_DATA
            and "z_cutoff" in MAP_DATA[map_name]
            and avg_z < MAP_DATA[map_name]["z_cutoff"]
        ):
            southwest_y += 1024
        rect = patches.Rectangle(
            (southwest_x, southwest_y),
            width,
            height,
            linewidth=1,
            edgecolor="yellow",
            facecolor="None",
        )
        axis.add_patch(rect)
    for source, dest in NAV_GRAPHS[map_name].edges():
        source_node = NAV_GRAPHS[map_name].nodes()[source]
        dest_node = NAV_GRAPHS[map_name].nodes()[dest]

        x1, x2 = source_node["center"][0], dest_node["center"][0]
        y1, y2 = source_node["center"][1], dest_node["center"][1]
        try:
            x1, x2 = position_transform(map_name, x1, "x"), position_transform(
                map_name, x2, "x"
            )
            y1, y2 = position_transform(map_name, y1, "y"), position_transform(
                map_name, y2, "y"
            )
        finally:
            if (
                map_name in MAP_DATA
                and "z_cutoff" in MAP_DATA[map_name]
                and source_node["center"][2] < MAP_DATA[map_name]["z_cutoff"]
            ):
                y1, y2 = y1 + 1024, y2 + 1024
            axis.plot(
                [x1, x2],
                [y1, y2],
                color="red",
            )
    plt.savefig(
        os.path.join(output_path, f"connections_{map_name}.png"),
        bbox_inches="tight",
        dpi=dpi,
    )
    fig.clear()
    plt.close()


class LeadersLastLevel(TypedDict):
    """Typed dict for the last level of the nested leader dict."""

    pos: list[float]
    index: int


def get_shortest_distances_mapping(
    map_name: str,
    leaders: dict[str, LeadersLastLevel],
    current_positions: list[list[float]],
    dist_type: DistanceType = "geodesic",
) -> list[str]:
    """Gets the shortest mapping between players in the current round and lead players.

    Args:
        map_name (str): Name of the current map
        leaders (dict): Dict of leaders position,
            and color index in the current frame
        current_positions (list): List of lists of players x, y, z, area_id coordinates
            in the current round and frame
        dist_type (str): Indicating the type of distance to use.
            Can be graph, geodesic, euclidean.

    Returns:
        A list mapping the player at index i in the current round
        to the leader at position list[i] in the leaders dictionary.
    """
    smallest_distance = float("inf")
    best_mapping: tuple[int, ...] = tuple(range(len(current_positions)))
    # Get all distance pairs
    distance_pairs: dict[int, dict[int, float]] = collections.defaultdict(
        lambda: collections.defaultdict(float)
    )
    for leader_i in range(len(leaders)):
        for current_i in range(len(current_positions)):
            if dist_type in ["geodesic", "graph"]:
                if map_name not in AREA_DIST_MATRIX:
                    this_dist = min(
                        area_distance(
                            map_name,
                            leaders[list(leaders)[leader_i]]["pos"][3],
                            current_positions[current_i][3],
                            dist_type=dist_type,
                        )["distance"],
                        area_distance(
                            map_name,
                            current_positions[current_i][3],
                            leaders[list(leaders)[leader_i]]["pos"][3],
                            dist_type=dist_type,
                        )["distance"],
                    )
                else:
                    this_dist = min(
                        AREA_DIST_MATRIX[map_name][
                            str(int(leaders[list(leaders)[leader_i]]["pos"][3]))
                        ][str(int(current_positions[current_i][3]))][dist_type],
                        AREA_DIST_MATRIX[map_name][
                            str(int(current_positions[current_i][3]))
                        ][str(int(leaders[list(leaders)[leader_i]]["pos"][3]))][
                            dist_type
                        ],
                    )
            else:
                this_dist = point_distance(
                    map_name,
                    current_positions[current_i][:3],
                    leaders[list(leaders)[leader_i]]["pos"][:3],
                    dist_type,
                )["distance"]
            distance_pairs[leader_i][current_i] = this_dist
    for mapping in itertools.permutations(range(len(leaders)), len(current_positions)):
        dist = 0
        for current_pos, leader_pos in enumerate(mapping):
            this_dist = distance_pairs[leader_pos][current_pos]
            dist += this_dist
        if dist < smallest_distance:
            smallest_distance = dist
            best_mapping = mapping
    return_mapping: list[str] = [""] * len(best_mapping)
    for i, leader_pos in enumerate(best_mapping):
        return_mapping[i] = list(leaders)[leader_pos]
    return return_mapping


def get_shortest_distances_mapping_trajectory(
    map_name: str,
    leaders: np.ndarray,
    current_positions: np.ndarray,
    dist_type: DistanceType = "geodesic",
    *,
    dtw: bool = False,
) -> tuple[int, ...]:
    """Gets the shortest mapping between players in the current round and lead players.

    Args:
        map_name (str): Name of the current map
        leaders (np.ndarray): Array of leaders position,
            and color index in the current frame
        current_positions (np.ndarray): Numpy array of shape (5,Time,1,1,X) of players
            x, y, z or area_id coordinates at each
            time step in the current round and frame
        dist_type (str): Indicating the type of distance to use.
            Can be graph, geodesic, euclidean.
        dtw: Boolean indicating whether matching should be performed via
            dynamic time warping (true) or euclidean (false)

    Returns:
        A list mapping the player at index i in the current round to the
        leader at position list[i] in the leaders dictionary.
    """
    smallest_distance = float("inf")
    best_mapping: tuple[int, ...] = tuple(range(len(current_positions)))
    # Get all distance pairs
    distance_pairs: dict[int, dict[int, float]] = collections.defaultdict(
        lambda: collections.defaultdict(float)
    )
    for leader_i in range(len(leaders)):
        for current_i in range(len(current_positions)):
            if current_positions[current_i] is None:
                continue
            this_dist = trajectory_distance(
                map_name,
                current_positions[current_i],
                leaders[leader_i],
                distance_type=dist_type,
                dtw=dtw,
            )
            distance_pairs[leader_i][current_i] = this_dist
    for mapping in itertools.permutations(range(len(leaders)), len(current_positions)):
        dist: float = 0
        for current_pos, leader_pos in enumerate(mapping):
            # Remove dead players from consideration
            if current_positions[current_pos] is None:
                continue
            this_dist = distance_pairs[leader_pos][current_pos]
            dist += this_dist
        if dist < smallest_distance:
            smallest_distance = dist
            best_mapping = mapping
    return best_mapping


colors_list = {
    1: ["cyan", "yellow", "fuchsia", "lime", "orange"],
    0: ["red", "green", "black", "white", "gold"],
}


def plot_rounds_different_players_trajectory_image(
    filename: str,
    frames_list: np.ndarray | list[np.ndarray],
    map_name: str = "de_ancient",
    map_type: str = "original",
    dist_type: DistanceType = "geodesic",
    *,
    dark: bool = False,
    dtw: bool = False,
    dpi: int = 1000,
) -> Literal[True]:
    """Plots a collection of rounds and saves as a .gif.

    Each player in the first round is assigned a separate color.
    Players in the other rounds are matched by proximity.
    Only use untransformed coordinates.

    Args:
        filename (string): Filename to save the gif
        frames_list (Union[np.ndarray, list[np.ndarray]]):
            (List or higher dimensional array) of np arrays.
            One array for each round.
            Each round entry should have shape (Time_steps,2|1(sides),5,3)
        map_name (str): Map to search
        map_type (str): "original" or "simpleradar"
        dist_type (str): String indicating the type of distance to use.
            Can be graph, geodesic, euclidean.
        dark (boolean): Only for use with map_type="simpleradar".
            Indicates if you want to use the SimpleRadar dark map type
        dtw: Boolean indicating whether matching should be performed via
            dynamic time warping (true) or euclidean (false)
        dpi (int): DPI of the resulting image

    Returns:
        True, saves .gif
    """
    f, a = plot_map(map_name=map_name, map_type=map_type, dark=dark)
    reference_traj: dict[int, np.ndarray] = {
        side: transform_to_traj_dimensions(frames_list[0][:, side, :, :])[
            :, :, :, :, (3,)
        ]
        for side in range(frames_list[0].shape[1])
    }
    for frames in tqdm(frames_list):
        # Initialize lists used to store values for this round for this frame
        for side in range(frames.shape[1]):
            mapping = get_shortest_distances_mapping_trajectory(
                map_name,
                reference_traj[side],
                # Already pass the precomputed area_id
                # all the way through to the distance calculations
                transform_to_traj_dimensions(frames[:, side, :, :])[:, :, :, :, (3,)],
                dist_type=dist_type,
                dtw=dtw,
            )
            for player in range(frames.shape[2]):
                a.plot(
                    [
                        position_transform(map_name, x, "x")
                        for x in frames[:, side, player, 0]
                    ],
                    [
                        position_transform(map_name, y, "y")
                        for y in frames[:, side, player, 1]
                    ],
                    c=colors_list[side][mapping[player]],
                    linestyle="-",
                    linewidth=0.25,
                    alpha=0.6,
                )
    a.get_xaxis().set_visible(b=False)
    a.get_yaxis().set_visible(b=False)
    f.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close()
    return True


def plot_rounds_different_players_trajectory_gif(
    filename: str,
    frames_list: np.ndarray | list[np.ndarray],
    map_name: str = "de_ancient",
    map_type: str = "original",
    dist_type: DistanceType = "geodesic",
    *,
    dark: bool = False,
    fps: int = 2,
    n_frames: int = 9000,
    dtw: bool = False,
    dpi: int = 300,
) -> Literal[True]:
    """Plots a collection of rounds and saves as a .gif.

    Each player in the first round is assigned a separate color.
    Players in the other rounds are matched by proximity.
    Only use untransformed coordinates.

    Args:
        filename (string): Filename to save the gif
        frames_list (Union[np.ndarray, list[np.ndarray]]):
            (List or higher dimensional array) of np arrays.
            One array for each round.
            Each round entry should have shape (Time_steps,2|1(sides),5,3)
        map_name (str): Map to search
        map_type (str): "original" or "simpleradar"
        dist_type (str): String indicating the type of distance to use.
            Can be graph, geodesic, euclidean.
        dark (bool): Only for use with map_type="simpleradar".
            Indicates if you want to use the SimpleRadar dark map type
        fps (int): Number of frames per second in the gif
        n_frames (int): The first how many frames should be plotted
        dtw: Boolean indicating whether matching should be performed via
            dynamic time warping (true) or euclidean (false)
        dpi (int): DPI of the resulting gif

    Returns:
        True, saves .gif
    """
    if os.path.isdir("csgo_tmp"):
        shutil.rmtree("csgo_tmp/")
    os.mkdir("csgo_tmp")
    image_files = []
    mappings = collections.defaultdict(list)
    reference_traj: dict[int, np.ndarray] = {
        side: transform_to_traj_dimensions(frames_list[0][:, side, :, :])[
            :, :, :, :, (3,)
        ]
        for side in range(frames_list[0].shape[1])
    }
    for frames in tqdm(frames_list):
        # Initialize lists used to store values for this round for this frame
        for side in range(frames.shape[1]):
            mapping = get_shortest_distances_mapping_trajectory(
                map_name,
                reference_traj[side],
                # Already pass the precomputed area_id
                # all the way through to the distance calculations
                transform_to_traj_dimensions(frames[:, side, :, :])[:, :, :, :, (3,)],
                dist_type=dist_type,
                dtw=dtw,
            )
            mappings[side].append(mapping)
    # Determine how many frames are there in total
    max_frames = max(frames.shape[0] for frames in frames_list)
    # For each side the keys are a players steamd id + "_" + frame_number
    # in case the same steamid occurs in multiple rounds
    for i in tqdm(range(min(max_frames, n_frames))):
        # Initialize lists used to store things from all rounds to plot for each frame
        positions: list[PlotPosition] = []
        # Now do another loop to add all players in
        # all frames with their appropriate colors.
        for frame_index, frames in enumerate(frames_list):
            # Initialize lists used to store values for this round for this frame
            if i in range(len(frames)):
                for side in range(frames.shape[1]):
                    # Now do the actual plotting
                    for player_index, pos in enumerate(frames[i][side]):
                        marker = rf"$ {frame_index} $"
                        color = colors_list[side][
                            mappings[side][frame_index][player_index]
                        ]
                        # If we are an alive leader we get opaque and big markers
                        if frame_index == 0:
                            alpha = 1
                            size = mpl.rcParams["lines.markersize"] ** 2
                        # If not we get partially transparent and small ones
                        else:
                            alpha = 0.5
                            size = 0.3 * mpl.rcParams["lines.markersize"] ** 2
                        positions.append(PlotPosition(pos, color, marker, alpha, size))
        f, _ = plot_positions(
            positions=positions,
            map_name=map_name,
            map_type=map_type,
            dark=dark,
            apply_transformation=True,
        )
        image_files.append(f"csgo_tmp/{i}.png")
        f.savefig(image_files[-1], dpi=dpi, bbox_inches="tight")
        plt.close()
    images = [imageio.imread(file) for file in image_files]
    imageio.imwrite(filename, images, duration=1000 / fps)
    shutil.rmtree("csgo_tmp/")
    return True


def plot_rounds_different_players_position_image(
    filename: str,
    frames_list: np.ndarray | list[np.ndarray],
    map_name: str = "de_ancient",
    map_type: str = "original",
    dist_type: DistanceType = "geodesic",
    *,
    dark: bool = False,
    n_frames: int = 9000,
    dpi: int = 1000,
) -> Literal[True]:
    """Plots a collection of rounds and saves as a .gif.

    Each player in the first round is assigned a separate color.
    Players in the other rounds are matched by proximity.
    Only use untransformed coordinates.

    Args:
        filename (string): Filename to save the gif
        frames_list (Union[np.ndarray, list[np.ndarray]]):
            (List or higher dimensional array) of np arrays.
            One array for each round.
            Each round entry should have shape (Time_steps,2|1(sides),5,3)
        map_name (str): Map to search
        map_type (str): "original" or "simpleradar"
        dist_type (str): String indicating the type of distance to use.
            Can be graph, geodesic, euclidean.
        dark (bool): Only for use with map_type="simpleradar".
            Indicates if you want to use the SimpleRadar dark map type
        n_frames (int): The first how many frames should be plotted
        dpi (int): DPI of the resulting image

    Returns:
        True, saves .gif
    """
    # frame_positions->frame_index->side->player_index->list[pos]
    # frame_colors->frame_index->side->player_index->list[colors]
    frame_positions: dict[
        int, dict[int, dict[int, list[list[float]]]]
    ] = collections.defaultdict(
        lambda: collections.defaultdict(lambda: collections.defaultdict(list))
    )
    frame_colors: dict[int, dict[int, dict[int, list[str]]]] = collections.defaultdict(
        lambda: collections.defaultdict(lambda: collections.defaultdict(list))
    )
    # Needed to check if leaders have been fully initialized
    dict_initialized = {0: False, 1: False}
    # Used to store values for each round separately. Needed when a round ends.
    # Determine how many frames are there in total
    max_frames = max(frames.shape[0] for frames in frames_list)
    # Build tree data structure for leaders
    # For each side the keys are a players steamd id + "_" + frame_number
    # in case the same steamid occurs in multiple rounds
    leaders: dict[int, dict[str, LeadersLastLevel]] = collections.defaultdict(
        lambda: collections.defaultdict(lambda: LeadersLastLevel(pos=[], index=0))
    )
    for i in tqdm(range(min(max_frames, n_frames))):
        # Is used to determine if a specific leader has already been seen.
        # Needed when a leader drops out because their round has already ended
        checked_in = set()
        # Loop over all the rounds and update the position and status of all leaders
        for frame_index, frames in enumerate(frames_list):
            # Check if the current frame has already ended
            if i in range(frames.shape[0]):
                for side in range(frames.shape[1]):
                    # Do not do this if leaders has not been fully initialized
                    if dict_initialized[side] is True:
                        for player_index, p in enumerate(frames[i][side]):
                            player_id = f"{player_index}_{frame_index}_{side}"
                            if (
                                player_id not in leaders[side]
                                or player_id in checked_in
                            ):
                                continue
                            leaders[side][player_id]["pos"] = p
        # Now do another loop to add all players in
        # all frames with their appropriate colors.
        for frame_index, frames in enumerate(frames_list):
            # Initialize lists used to store values for this round for this frame
            if i in range(len(frames)):
                for side in range(frames.shape[1]):
                    # If we have already initialized leaders
                    if dict_initialized[side] is True:
                        current_positions = list(frames[i][side])
                        # Find the best mapping between current players and leaders
                        mapping = get_shortest_distances_mapping(
                            map_name,
                            leaders[side],
                            current_positions,
                            dist_type=dist_type,
                        )
                    # Now do the actual plotting
                    for index, pos in enumerate(frames[i][side]):
                        player_index = index
                        player_id = f"{player_index}_{frame_index}_{side}"

                        # If the leaders have not been initialized yet, do so
                        if dict_initialized[side] is False:
                            leaders[side][player_id]["index"] = player_index
                            leaders[side][player_id]["pos"] = pos

                        # This is relevant for all subsequent frames
                        # If we are a leader we update our values
                        # Should be able to be dropped as we
                        # already updated leaders in the earlier loop
                        if player_id in leaders[side]:
                            # Grab our current player_index from what it was the
                            # previous round to achieve color consistency
                            player_index = leaders[side][player_id]["index"]
                            # Update our position
                            leaders[side][player_id]["pos"] = pos
                        # If not a leader
                        else:
                            # Grab the id of the leader assigned to this player
                            assigned_leader = mapping[player_index]
                            # If the assigned leader  has not been assigned
                            # (happens when his round is already over)
                            # Then we take over that position
                            if assigned_leader not in checked_in:
                                # Remove the previous leaders entry from the dict
                                old_index = leaders[side][assigned_leader]["index"]
                                del leaders[side][assigned_leader]
                                # Fill with our own values but use their prior index to
                                # keep color consistency when switching leaders
                                leaders[side][player_id]["index"] = old_index
                                leaders[side][player_id]["pos"] = pos
                                player_index = leaders[side][player_id]["index"]
                            # If the leader is alive and present or if we are also dead
                            else:
                                # We just grab our color
                                player_index = leaders[side][assigned_leader]["index"]
                        frame_colors[frame_index][side][player_index].append(
                            colors_list[side][player_index]
                        )
                        frame_positions[frame_index][side][player_index].append(pos)
                        # If we are a leader we are now checked in
                        # so everyone knows our round has not ended yet
                        if player_id in leaders[side]:
                            checked_in.add(player_id)
                    # Once we have done our first loop over a side we are initialized
                    dict_initialized[side] = True
    f, a = plot_map(map_name=map_name, map_type=map_type, dark=dark)
    for frame, value in frame_positions.items():
        for side in value:
            for player in frame_positions[frame][side]:
                a.plot(
                    [
                        position_transform(map_name, x[0], "x")
                        for x in frame_positions[frame][side][player]
                    ],
                    [
                        position_transform(map_name, x[1], "y")
                        for x in frame_positions[frame][side][player]
                    ],
                    c=frame_colors[frame][side][player][0],
                    linestyle="-",
                    linewidth=0.25,
                    alpha=0.6,
                )
    a.get_xaxis().set_visible(b=False)
    a.get_yaxis().set_visible(b=False)
    f.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close()
    return True


def plot_rounds_different_players_position_gif(
    filename: str,
    frames_list: np.ndarray | list[np.ndarray],
    map_name: str = "de_ancient",
    map_type: str = "original",
    dist_type: DistanceType = "geodesic",
    *,
    dark: bool = False,
    fps: int = 2,
    n_frames: int = 9000,
    dpi: int = 300,
) -> Literal[True]:
    """Plots a collection of rounds and saves as a .gif.

    Each player in the first round is assigned a separate color.
    Players in the other rounds are matched by proximity.
    Only use untransformed coordinates.

    Args:
        filename (string): Filename to save the gif
        frames_list (Union[np.ndarray, list[np.ndarray]]):
            (List or higher dimensional array) of np arrays.
            One array for each round.
            Each round entry should have shape (Time_steps,2|1(sides),5,3)
        map_name (str): Map to search
        map_type (str): "original" or "simpleradar"
        dist_type (str): String indicating the type of distance to use.
            Can be graph, geodesic, euclidean.
        dark (bool): Only for use with map_type="simpleradar".
            Indicates if you want to use the SimpleRadar dark map type
        fps (int): Number of frames per second in the gif
        n_frames (int): The first how many frames should be plotted
        dtw: Boolean indicating whether matching should be performed via
            dynamic time warping (true) or euclidean (false)
        dpi (int): DPI of the resulting gif

    Returns:
        True, saves .gif
    """
    if os.path.isdir("csgo_tmp"):
        shutil.rmtree("csgo_tmp/")
    os.mkdir("csgo_tmp")
    image_files = []
    # Needed to check if leaders have been fully initialized
    dict_initialized = {0: False, 1: False}
    # Used to store values for each round separately. Needed when a round ends.
    # Determine how many frames are there in total
    max_frames = max(frames.shape[0] for frames in frames_list)
    # Build tree data structure for leaders
    # For each side the keys are a players steamd id + "_" + frame_number
    # in case the same steamid occurs in multiple rounds
    leaders: dict[int, dict[str, LeadersLastLevel]] = collections.defaultdict(
        lambda: collections.defaultdict(lambda: LeadersLastLevel(pos=[], index=0))
    )
    for i in tqdm(range(min(max_frames, n_frames))):
        # Initialize lists used to store things from all rounds to plot for each frame
        positions: list[PlotPosition] = []
        # Is used to determine if a specific leader has already been seen.
        # Needed when a leader drops out because their round has already ended
        checked_in = set()
        # Loop over all the rounds and update the position and status of all leaders
        for frame_index, frames in enumerate(frames_list):
            # Check if the current frame has already ended
            if i in range(frames.shape[0]):
                for side in range(frames.shape[1]):
                    # Do not do this if leaders has not been fully initialized
                    if dict_initialized[side] is True:
                        for player_index, p in enumerate(frames[i][side]):
                            player_id = f"{player_index}_{frame_index}_{side}"
                            if (
                                player_id not in leaders[side]
                                or player_id in checked_in
                            ):
                                continue
                            leaders[side][player_id]["pos"] = p
        # Now do another loop to add all players in
        # all frames with their appropriate colors.
        for frame_index, frames in enumerate(frames_list):
            # Initialize lists used to store values for this round for this frame
            if i in range(len(frames)):
                for side in range(frames.shape[1]):
                    # If we have already initialized leaders
                    if dict_initialized[side] is True:
                        current_positions = list(frames[i][side])
                        # Find the best mapping between current players and leaders
                        mapping = get_shortest_distances_mapping(
                            map_name,
                            leaders[side],
                            current_positions,
                            dist_type=dist_type,
                        )
                    # Now do the actual plotting
                    for index, pos in enumerate(frames[i][side]):
                        player_index = index
                        player_id = f"{player_index}_{frame_index}_{side}"

                        # If the leaders have not been initialized yet, do so
                        if dict_initialized[side] is False:
                            leaders[side][player_id]["index"] = player_index
                            leaders[side][player_id]["pos"] = pos

                        # This is relevant for all subsequent frames
                        # If we are a leader we update our values
                        # Should be able to be dropped as we
                        # already updated leaders in the earlier loop
                        if player_id in leaders[side]:
                            # Grab our current player_index from what it was
                            # the previous round to achieve color consistency
                            player_index = leaders[side][player_id]["index"]
                            # Update our position
                            leaders[side][player_id]["pos"] = pos
                        # If not a leader
                        else:
                            # Grab the id of the leader assigned to this player
                            assigned_leader = mapping[player_index]
                            # If the assigned leader is now dead
                            # or has not been assigned
                            # (happens when his round is already over)
                            # Then we take over that position if we are not also dead
                            if assigned_leader not in checked_in:
                                # Remove the previous leaders entry from the dict
                                old_index = leaders[side][assigned_leader]["index"]
                                del leaders[side][assigned_leader]
                                # Fill with our own values but use their prior index to
                                # keep color consistency when switching leaders
                                leaders[side][player_id]["index"] = old_index
                                leaders[side][player_id]["pos"] = pos
                                player_index = leaders[side][player_id]["index"]
                            # If the leader is alive and present or if we are also dead
                            else:
                                # We just grab our color
                                player_index = leaders[side][assigned_leader]["index"]
                        marker = rf"$ {frame_index} $"
                        color = colors_list[side][player_index]
                        # If we are an alive leader we get opaque and big markers
                        if player_id in leaders[side] and player_id not in checked_in:
                            alpha = 1
                            size = mpl.rcParams["lines.markersize"] ** 2
                        # If not we get partially transparent and small ones
                        else:
                            alpha = 0.5
                            size = 0.3 * mpl.rcParams["lines.markersize"] ** 2
                        positions.append(PlotPosition(pos, color, marker, alpha, size))
                        # If we are a leader we are now checked in
                        # so everyone knows our round has not ended yet
                        if player_id in leaders[side]:
                            checked_in.add(player_id)
                    # Once we have done our first loop over a side we are initialized
                    dict_initialized[side] = True
        f, _ = plot_positions(
            positions=positions,
            map_name=map_name,
            map_type=map_type,
            dark=dark,
            apply_transformation=True,
        )
        image_files.append(f"csgo_tmp/{i}.png")
        f.savefig(image_files[-1], dpi=dpi, bbox_inches="tight")
        plt.close()
    images = [imageio.imread(file) for file in image_files]
    imageio.imwrite(filename, images, duration=1000 / fps)
    shutil.rmtree("csgo_tmp/")
    return True


def plot_rounds_different_players(
    filename: str,
    frames_list: np.ndarray | list[np.ndarray],
    map_name: str = "de_ancient",
    map_type: str = "original",
    dist_type: DistanceType = "geodesic",
    *,
    dark: bool = False,
    fps: int = 10,
    n_frames: int = 9000,
    image: bool = False,
    trajectory: bool = False,
    dtw: bool = False,
    dpi: int | None = None,
) -> Literal[True]:
    """Plots a list of rounds and saves as a .gif.

    Each player in the first round is assigned a separate color.
    Players in the other rounds are matched by proximity.
    Only use untransformed coordinates.

    Args:
        filename (string): Filename to save the gif
        frames_list (Union[np.ndarray, list[np.ndarray]]):
            (List or higher dimensional array) of np arrays.
            One array for each round.
            Each round entry should have shape (Time_steps,2|1(sides),5,3)
        map_name (str): Map to search
        map_type (str): "original" or "simpleradar"
        dist_type (str): String indicating the type of distance to use.
            Can be graph, geodesic, euclidean.
        dark (bool): Only for use with map_type="simpleradar".
            Indicates if you want to use the SimpleRadar dark map type
        fps (int): Number of frames per second in the gif
        n_frames (int): The first how many frames should be plotted
        image (bool): Boolean indicating whether a gif of positions or
            a singular image of trajectories should be produced
        trajectory (bool): Indicates whether the clustering of players should
            be done for the whole trajectories instead of each individual time step
        dtw: Boolean indicating whether matching should be performed via
            dynamic time warping (true) or euclidean (false)
        dpi (int): DPI of the result (defaults to 1000 for images and 300 for gifs)

    Returns:
        True, saves .gif
    """
    if trajectory:
        return (
            plot_rounds_different_players_trajectory_image(
                filename=filename,
                frames_list=frames_list,
                map_name=map_name,
                map_type=map_type,
                dark=dark,
                dist_type=dist_type,
                dtw=dtw,
                dpi=dpi if dpi else 1000,
            )
            if image
            else plot_rounds_different_players_trajectory_gif(
                filename=filename,
                frames_list=frames_list,
                map_name=map_name,
                map_type=map_type,
                dark=dark,
                fps=fps,
                n_frames=n_frames,
                dist_type=dist_type,
                dtw=dtw,
                dpi=dpi if dpi else 300,
            )
        )
    if image:
        return plot_rounds_different_players_position_image(
            filename=filename,
            frames_list=frames_list,
            map_name=map_name,
            map_type=map_type,
            dark=dark,
            dist_type=dist_type,
            dpi=dpi if dpi else 1000,
        )
    # gif
    return plot_rounds_different_players_position_gif(
        filename=filename,
        frames_list=frames_list,
        map_name=map_name,
        map_type=map_type,
        dark=dark,
        fps=fps,
        n_frames=n_frames,
        dist_type=dist_type,
        dpi=dpi if dpi else 300,
    )


def main(args: list[str]) -> None:
    """Plots player positions, position tokens and maps nav mesh and named areas."""
    parser = argparse.ArgumentParser("Analyze the early mid fight on inferno")
    parser.add_argument(
        "-d", "--debug", action="store_true", default=False, help="Enable debug output."
    )
    parser.add_argument(
        "-l",
        "--log",
        default=r"D:\CSGO\ML\CSGOML\logs\plotting_tests.log",
        help="Path to output log.",
    )
    parser.add_argument(
        "-f",
        "--file",
        default=r"D:\CSGO\Demos\Maps\dust2\713.json",  # r"D:\CSGO\Demos\713.dem",  #
        help="Path to output log.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=r"D:\CSGO\ML\CSGOML\Plots",
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

    logging.info("Generating tile and area files for all maps.")
    before_nuke = True
    for map_name in NAV:
        if map_name == "de_nuke":
            before_nuke = False
        if before_nuke:
            continue
        plot_map_areas(
            output_path=options.output,
            map_name=map_name,
            map_type="simpleradar",
            dark=False,
        )
        plot_map_tiles(
            output_path=options.output,
            map_name=map_name,
            map_type="simpleradar",
            dark=False,
        )
        plot_map_connections(
            output_path=options.output,
            map_name=map_name,
            map_type="simpleradar",
            dark=False,
        )
        plot_mid(
            output_path=options.output,
            map_name=map_name,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
