"""Collection of functions to extend awpy plotting capabilites

    Typical usage example:

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
#!/usr/bin/env python
# pylint: disable=consider-using-enumerate

import collections
import itertools
import os
import sys
from typing import Optional, Literal, TypedDict, Union
import logging
import shutil
import argparse
import numpy as np
from matplotlib import patches
import imageio
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from awpy.visualization.plot import (
    plot_map,
    position_transform,
    plot_positions,
)
from awpy.analytics.nav import (
    generate_position_token,
    area_distance,
    point_distance,
    stepped_hull,
    generate_centroids,
)
from awpy.data import NAV, MAP_DATA, AREA_DIST_MATRIX
from csgoml.utils.nav_utils import transform_to_traj_dimensions, trajectory_distance


def get_areas_hulls_centers(map_name: str) -> tuple[dict, dict, dict]:
    """Gets the sets of points making up the named areas of a map. Then builds their hull and centroid.

    Args:
        map_name (str): A string specifying the map for which the features should be build.

    Returns:
        Three dictionary containing the points, hull and centroid of each named area in the map
    """
    hulls = {}
    centers = {}
    area_points = collections.defaultdict(list)
    for a in NAV[map_name]:
        area = NAV[map_name][a]
        try:
            cur_x = []
            cur_y = []
            cur_x.append(position_transform(map_name, area["southEastX"], "x"))
            cur_x.append(position_transform(map_name, area["northWestX"], "x"))
            cur_y.append(position_transform(map_name, area["southEastY"], "y"))
            cur_y.append(position_transform(map_name, area["northWestY"], "y"))
        except KeyError:
            cur_x = []
            cur_y = []
            cur_x.append(area["southEastX"])
            cur_x.append(area["northWestX"])
            cur_y.append(area["southEastY"])
            cur_y.append(area["northWestY"])
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
    frames: list,
    map_name: str = "de_ancient",
    map_type: str = "original",
    dark: bool = False,
    fps: int = 2,
    dpi: int = 300,
) -> Literal[True]:
    """Plots the position tokens of a round and saves as a .gif. CTs are blue, Ts are orange. Only use untransformed coordinates.

    Args:
        filename (string): Filename to save the gif
        frames (list): List of frames from a parsed demo
        map_name (string): Map to search
        map_type (string): "original" or "simpleradar"
        dark (boolean): Only for use with map_type="simpleradar". Indicates if you want to use the SimpleRadar dark map type
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
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)
        # Grab the position token
        tokens = generate_position_token(map_name, frame)
        # Loop over each named area
        for area_id, area in enumerate(sorted(area_points)):
            text = ""
            # Plot each area twice. Once for each side. If both sides have players in one tile the colors get added as
            # the alphas arent 1.
            for side in ["t", "ct"]:
                # Dont plot the "" area as it stretches across the whole map and messes with clarity
                if not area:
                    # axis.plot(np.NaN, np.NaN, label="None_" + tokens[side + "Token"][i])
                    continue
                # Plot the hull of the area
                axis.plot(
                    hulls[area][:, 0],
                    hulls[area][:, 1],
                    "-",
                    # Set alpha and color depending on players inside.
                    # 0 players from this team means low alpha gray
                    # Otherwise blue/red with darkness depending on number of players
                    alpha=0.8 if int(tokens[side + "Token"][area_id]) > 0 else 0.2,
                    c=colors[side][int(tokens[side + "Token"][area_id])],
                    lw=3,
                )
                if tokens[side + "Token"][area_id] != "0":
                    text += side + ":" + tokens[side + "Token"][area_id] + " "
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
    images = []
    for file in image_files:
        images.append(imageio.imread(file))
    imageio.mimsave(filename, images, fps=fps)
    shutil.rmtree("csgo_tmp/")
    return True


def plot_map_areas(
    output_path: str,
    map_name: str = "de_ancient",
    map_type: str = "original",
    dark: bool = False,
    dpi: int = 1000,
) -> None:
    """Plot all named areas in the given map.

    Args:
        output_path (string): Path to the output folder
        map_name (string): Map to search
        map_type (string): "original" or "simpleradar"
        dark (boolean): Only for use with map_type="simpleradar". Indicates if you want to use the SimpleRadar dark map type
        dpi (int): DPI of the resulting image

    Returns:
        None, saves .png
    """
    logging.info("Plotting areas for %s", map_name)
    fig, axis = plot_map(map_name=map_name, map_type=map_type, dark=dark)
    fig.set_size_inches(19.2, 10.8)
    # Grab points, hull and centroid for each named area
    area_points, hulls, centers = get_areas_hulls_centers(map_name)
    for area in sorted(area_points):
        # Dont plot the "" area as it stretches across the whole map and messes with clarity
        # but add it to the legend
        if not area:
            axis.plot(np.NaN, np.NaN, label="None")
            continue
        axis.plot(hulls[area][:, 0], hulls[area][:, 1], "-", lw=3, label=area)
        # Add name of area into its middle
        axis.text(
            centers[area][0] - (centers[area][0] - hulls[area][0][0]) / 1.5,
            centers[area][1],
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
    """Plots the given map with the hulls, centroid and a representative_point for each named area.

    Args:
        output_path (string): Path to the output folder
        map_name (string): Map to plot
        dpi (int): DPI of the resulting image

    Returns:
        None (Directly saves the plot to file)
    """
    cent_ids, rep_ids = generate_centroids(map_name)
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
    dark: bool = False,
    dpi: int = 1000,
) -> None:
    """Plot all navigation mesh tiles in a given map.

    Args:
        output_path (string): Path to the output folder
        map_name (string): Map to search
        map_type (string): "original" or "simpleradar"
        dark (boolean): Only for use with map_type="simpleradar". Indicates if you want to use the SimpleRadar dark map type
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


class LeadersLastLevel(TypedDict):
    """Typed dict for the last level of the nested leader dict"""

    pos: list[float]
    index: int


def get_shortest_distances_mapping(
    map_name: str,
    leaders: dict[str, LeadersLastLevel],
    current_positions: list[list[float]],
    dist_type: str = "geodesic",
) -> list[str]:
    """Gets the mapping between players in the current round and lead players that has the shortest total distance between mapped players.

    Args:
        map_name (str): Name of the current map
        leaders (dictionary): Dictionary of leaders position, and color index in the current frame
        current_positions (list): List of lists of players x, y, z, area_id coordinates in the current round and frame
        dist_type (string): String indicating the type of distance to use. Can be graph, geodesic, euclidean, manhattan, canberra or cosine.

    Returns:
        A list mapping the player at index i in the current round to the leader at position list[i] in the leaders dictionary.
        (Requires python 3.6 because it relies on the order of elements in the dict)"""
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
            if dist_type in ["geodesic", "graph"]:
                if map_name not in AREA_DIST_MATRIX:
                    this_dist = min(
                        area_distance(
                            map_name,
                            str(int(leaders[list(leaders)[leader_i]]["pos"][3])),
                            str(int(current_positions[current_i][3])),
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
                    current_positions[current_i][0:3],
                    leaders[list(leaders)[leader_i]]["pos"][0:3],
                    dist_type,
                )["distance"]
            distance_pairs[leader_i][current_i] = this_dist
    for mapping in itertools.permutations(range(len(leaders)), len(current_positions)):
        dist = 0
        for current_pos, leader_pos in enumerate(mapping):
            # Remove dead players from consideration
            if current_positions[current_pos] is None:
                continue
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
    dist_type: str = "geodesic",
    dtw: bool = False,
) -> tuple[int, ...]:
    """Gets the mapping between players in the current round and lead players that has the shortest total distance between mapped players.

    Args:
        map_name (str): Name of the current map
        leaders (np.ndarray): np.ndarray of leaders position, and color index in the current frame
        current_positions (np.ndarray): Numpy array of shape (5,Time,1,1,X) of players x, y, z or area_id coordinates at each time step in the current round and frame
        dist_type (string): String indicating the type of distance to use. Can be graph, geodesic, euclidean, manhattan, canberra or cosine.
        dtw: Boolean indicating whether matching should be performed via dynamic time warping (true) or euclidean (false)

    Returns:
        A list mapping the player at index i in the current round to the leader at position list[i] in the leaders dictionary.
        (Requires python 3.6 because it relies on the order of elements in the dict)"""
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
    frames_list: Union[np.ndarray, list[np.ndarray]],
    map_name: str = "de_ancient",
    map_type: str = "original",
    dark: bool = False,
    dist_type: str = "geodesic",
    dtw: bool = False,
    dpi: int = 1000,
) -> Literal[True]:
    """Plots a collection of rounds and saves as a .gif. Each player in the first round is assigned a separate color. Players in the other rounds are matched by proximity.
    Only use untransformed coordinates.

    Args:
        filename (string): Filename to save the gif
        frames_list (Union[np.ndarray, list[np.ndarray]]): (List or higher dimensional array) of np arrays. One array for each round. Each round entry should have shape (Time_steps,2|1(sides),5,3)
        map_name (string): Map to search
        map_type (string): "original" or "simpleradar"
        dark (boolean): Only for use with map_type="simpleradar". Indicates if you want to use the SimpleRadar dark map type
        dist_type (string): String indicating the type of distance to use. Can be graph, geodesic, euclidean, manhattan, canberra or cosine
        dtw: Boolean indicating whether matching should be performed via dynamic time warping (true) or euclidean (false)
        dpi (int): DPI of the resulting image

    Returns:
        True, saves .gif
    """
    f, a = plot_map(map_name=map_name, map_type=map_type, dark=dark)
    reference_traj: dict[int, np.ndarray] = {}
    for side in range(frames_list[0].shape[1]):
        reference_traj[side] = transform_to_traj_dimensions(
            frames_list[0][:, side, :, :]
        )[:, :, :, :, (3,)]
    for frames in tqdm(frames_list):
        # Initialize lists used to store values for this round for this frame
        for side in range(frames.shape[1]):
            mapping = get_shortest_distances_mapping_trajectory(
                map_name,
                reference_traj[side],
                # Already pass the precomputed area_id all the way through to the distance calculations
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
    a.get_xaxis().set_visible(False)
    a.get_yaxis().set_visible(False)
    f.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close()
    return True


def plot_rounds_different_players_trajectory_gif(
    filename: str,
    frames_list: Union[np.ndarray, list[np.ndarray]],
    map_name: str = "de_ancient",
    map_type: str = "original",
    dark: bool = False,
    fps: int = 2,
    n_frames: int = 9000,
    dist_type: str = "geodesic",
    dtw: bool = False,
    dpi: int = 300,
) -> Literal[True]:
    """Plots a collection of rounds and saves as a .gif. Each player in the first round is assigned a separate color. Players in the other rounds are matched by proximity.
    Only use untransformed coordinates.

    Args:
        filename (string): Filename to save the gif
        frames_list (Union[np.ndarray, list[np.ndarray]]): (List or higher dimensional array) of np arrays. One array for each round. Each round entry should have shape (Time_steps,2|1(sides),5,3)
        map_name (string): Map to search
        map_type (string): "original" or "simpleradar"
        dark (boolean): Only for use with map_type="simpleradar". Indicates if you want to use the SimpleRadar dark map type
        fps (integer): Number of frames per second in the gif
        n_frames (integer): The first how many frames should be plotted
        dist_type (string): String indicating the type of distance to use. Can be graph, geodesic, euclidean, manhattan, canberra or cosine
        dtw: Boolean indicating whether matching should be performed via dynamic time warping (true) or euclidean (false)
        dpi (int): DPI of the resulting gif

    Returns:
        True, saves .gif
    """
    if os.path.isdir("csgo_tmp"):
        shutil.rmtree("csgo_tmp/")
    os.mkdir("csgo_tmp")
    image_files = []
    mappings = collections.defaultdict(list)
    reference_traj: dict[int, np.ndarray] = {}
    for side in range(frames_list[0].shape[1]):
        reference_traj[side] = transform_to_traj_dimensions(
            frames_list[0][:, side, :, :]
        )[:, :, :, :, (3,)]
    for frame_index, frames in enumerate(tqdm(frames_list)):
        # Initialize lists used to store values for this round for this frame
        for side in range(frames.shape[1]):
            mapping = get_shortest_distances_mapping_trajectory(
                map_name,
                reference_traj[side],
                # Already pass the precomputed area_id all the way through to the distance calculations
                transform_to_traj_dimensions(frames[:, side, :, :])[:, :, :, :, (3,)],
                dist_type=dist_type,
                dtw=dtw,
            )
            mappings[side].append(mapping)
    # Determine how many frames are there in total
    max_frames = max(frames.shape[0] for frames in frames_list)
    # For each side the keys are a players steamd id + "_" + frame_number in case the same steamid occurs in multiple rounds
    for i in tqdm(range(min(max_frames, int(n_frames)))):
        # Initialize lists used to store things from all rounds to plot for each frame
        positions: list[list[float]] = []
        colors: list[str] = []
        markers: list[str] = []
        alphas: list[float] = []
        sizes: list[float] = []
        # Now do another loop to add all players in all frames with their appropriate colors.
        for frame_index, frames in enumerate(frames_list):
            # Initialize lists used to store values for this round for this frame
            if i in range(len(frames)):
                for side in range(frames.shape[1]):
                    # Now do the actual plotting
                    for player_index, p in enumerate(frames[i][side]):
                        pos = p
                        markers.append(rf"$ {frame_index} $")
                        colors.append(
                            colors_list[side][mappings[side][frame_index][player_index]]
                        )
                        positions.append(pos)
                        # If we are an alive leader we get opaque and big markers
                        if frame_index == 0:
                            alphas.append(1)
                            sizes.append(mpl.rcParams["lines.markersize"] ** 2)
                        # If not we get partially transparent and small ones
                        else:
                            alphas.append(0.5)
                            sizes.append(0.3 * mpl.rcParams["lines.markersize"] ** 2)
        f, _ = plot_positions(
            positions=positions,
            colors=colors,
            markers=markers,
            alphas=alphas,
            sizes=sizes,
            map_name=map_name,
            map_type=map_type,
            dark=dark,
            apply_transformation=True,
        )
        image_files.append(f"csgo_tmp/{i}.png")
        f.savefig(image_files[-1], dpi=dpi, bbox_inches="tight")
        plt.close()
    images = []
    for file in image_files:
        images.append(imageio.imread(file))
    imageio.mimsave(filename, images, fps=fps)
    shutil.rmtree("csgo_tmp/")
    return True


def plot_rounds_different_players_position_image(
    filename: str,
    frames_list: Union[np.ndarray, list[np.ndarray]],
    map_name: str = "de_ancient",
    map_type: str = "original",
    dark: bool = False,
    n_frames: int = 9000,
    dist_type: str = "geodesic",
    dpi: int = 1000,
) -> Literal[True]:
    """Plots a collection of rounds and saves as a .gif. Each player in the first round is assigned a separate color. Players in the other rounds are matched by proximity.
    Only use untransformed coordinates.

    Args:
        filename (string): Filename to save the gif
        frames_list (Union[np.ndarray, list[np.ndarray]]): (List or higher dimensional array) of np arrays. One array for each round. Each round entry should have shape (Time_steps,2|1(sides),5,3)
        map_name (string): Map to search
        map_type (string): "original" or "simpleradar"
        dark (boolean): Only for use with map_type="simpleradar". Indicates if you want to use the SimpleRadar dark map type
        dist_type (string): String indicating the type of distance to use. Can be graph, geodesic, euclidean, manhattan, canberra or cosine
        dpi (int): DPI of the resulting image

    Returns:
        True, saves .gif
    """
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
    # For each side the keys are a players steamd id + "_" + frame_number in case the same steamid occurs in multiple rounds
    leaders: dict[int, dict[str, LeadersLastLevel]] = collections.defaultdict(
        lambda: collections.defaultdict(lambda: LeadersLastLevel(pos=[], index=0))
    )
    for i in tqdm(range(min(max_frames, int(n_frames)))):
        # Is used to determine if a specific leader has already been seen. Needed when a leader drops out because their round has already ended
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
        # Now do another loop to add all players in all frames with their appropriate colors.
        for frame_index, frames in enumerate(frames_list):
            # Initialize lists used to store values for this round for this frame
            if i in range(len(frames)):
                for side in range(frames.shape[1]):
                    # If we have already initialized leaders
                    if dict_initialized[side] is True:
                        # Get the positions of all players in the current frame and round
                        current_positions = []
                        for player_index, p in enumerate(frames[i][side]):
                            current_positions.append(p)
                        # Find the best mapping between current players and leaders
                        mapping = get_shortest_distances_mapping(
                            map_name,
                            leaders[side],
                            current_positions,
                            dist_type=dist_type,
                        )
                    # Now do the actual plotting
                    for player_index, p in enumerate(frames[i][side]):
                        pos = p
                        player_id = f"{player_index}_{frame_index}_{side}"

                        # If the leaders have not been initialized yet, do so
                        if dict_initialized[side] is False:
                            leaders[side][player_id]["index"] = player_index
                            leaders[side][player_id]["pos"] = pos

                        # This is relevant for all subsequent frames
                        # If we are a leader we update our values
                        # Should be able to be dropped as we already updated leaders in the earlier loop
                        if player_id in leaders[side]:
                            # Grab our current player_index from what it was the previous round to achieve color consistency
                            player_index = leaders[side][player_id]["index"]
                            # Update our position
                            leaders[side][player_id]["pos"] = pos
                        # If not a leader
                        else:
                            # Grab the id of the leader assigned to this player
                            assigned_leader = mapping[player_index]
                            # If the assigned leader  has not been assigned (happens when his round is already over)
                            # Then we take over that position
                            if assigned_leader not in checked_in:
                                # Remove the previous leaders entry from the dict
                                old_index = leaders[side][assigned_leader]["index"]
                                del leaders[side][assigned_leader]
                                # Fill with our own values but use their prior index to keep color consistency when switching leaders
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
                        # If we are a leader we are now checked in so everyone knows our round has not ended yet
                        if player_id in leaders[side]:
                            checked_in.add(player_id)
                    # Once we have done our first loop over a side we are initialized
                    dict_initialized[side] = True
    f, a = plot_map(map_name=map_name, map_type=map_type, dark=dark)
    for frame in frame_positions:
        for side in frame_positions[frame]:
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
    a.get_xaxis().set_visible(False)
    a.get_yaxis().set_visible(False)
    f.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close()
    return True


def plot_rounds_different_players_position_gif(
    filename: str,
    frames_list: Union[np.ndarray, list[np.ndarray]],
    map_name: str = "de_ancient",
    map_type: str = "original",
    dark: bool = False,
    fps: int = 2,
    n_frames: int = 9000,
    dist_type: str = "geodesic",
    dpi: int = 300,
) -> Literal[True]:
    """Plots a collection of rounds and saves as a .gif. Each player in the first round is assigned a separate color. Players in the other rounds are matched by proximity.
    Only use untransformed coordinates.

    Args:
        filename (string): Filename to save the gif
        frames_list (Union[np.ndarray, list[np.ndarray]]): (List or higher dimensional array) of np arrays. One array for each round. Each round entry should have shape (Time_steps,2|1(sides),5,3)
        map_name (string): Map to search
        map_type (string): "original" or "simpleradar"
        dark (boolean): Only for use with map_type="simpleradar". Indicates if you want to use the SimpleRadar dark map type
        fps (integer): Number of frames per second in the gif
        n_frames (integer): The first how many frames should be plotted
        dist_type (string): String indicating the type of distance to use. Can be graph, geodesic, euclidean, manhattan, canberra or cosine
        dtw: Boolean indicating whether matching should be performed via dynamic time warping (true) or euclidean (false)
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
    # For each side the keys are a players steamd id + "_" + frame_number in case the same steamid occurs in multiple rounds
    leaders: dict[int, dict[str, LeadersLastLevel]] = collections.defaultdict(
        lambda: collections.defaultdict(lambda: LeadersLastLevel(pos=[], index=0))
    )
    for i in tqdm(range(min(max_frames, int(n_frames)))):
        # Initialize lists used to store things from all rounds to plot for each frame
        positions: list[list[float]] = []
        colors: list[str] = []
        markers: list[str] = []
        alphas: list[float] = []
        sizes: list[float] = []
        # Is used to determine if a specific leader has already been seen. Needed when a leader drops out because their round has already ended
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
        # Now do another loop to add all players in all frames with their appropriate colors.
        for frame_index, frames in enumerate(frames_list):
            # Initialize lists used to store values for this round for this frame
            if i in range(len(frames)):
                for side in range(frames.shape[1]):
                    # If we have already initialized leaders
                    if dict_initialized[side] is True:
                        # Get the positions of all players in the current frame and round
                        current_positions = []
                        for player_index, p in enumerate(frames[i][side]):
                            current_positions.append(p)
                        # Find the best mapping between current players and leaders
                        mapping = get_shortest_distances_mapping(
                            map_name,
                            leaders[side],
                            current_positions,
                            dist_type=dist_type,
                        )
                    # Now do the actual plotting
                    for player_index, p in enumerate(frames[i][side]):
                        pos = p
                        player_id = f"{player_index}_{frame_index}_{side}"

                        # If the leaders have not been initialized yet, do so
                        if dict_initialized[side] is False:
                            leaders[side][player_id]["index"] = player_index
                            leaders[side][player_id]["pos"] = pos

                        # This is relevant for all subsequent frames
                        # If we are a leader we update our values
                        # Should be able to be dropped as we already updated leaders in the earlier loop
                        if player_id in leaders[side]:
                            # Grab our current player_index from what it was the previous round to achieve color consistency
                            player_index = leaders[side][player_id]["index"]
                            # Update our position
                            leaders[side][player_id]["pos"] = pos
                        # If not a leader
                        else:
                            # Grab the id of the leader assigned to this player
                            assigned_leader = mapping[player_index]
                            # If the assigned leader is now dead or has not been assigned (happens when his round is already over)
                            # Then we take over that position if we are not also dead
                            if assigned_leader not in checked_in:
                                # Remove the previous leaders entry from the dict
                                old_index = leaders[side][assigned_leader]["index"]
                                del leaders[side][assigned_leader]
                                # Fill with our own values but use their prior index to keep color consistency when switching leaders
                                leaders[side][player_id]["index"] = old_index
                                leaders[side][player_id]["pos"] = pos
                                player_index = leaders[side][player_id]["index"]
                            # If the leader is alive and present or if we are also dead
                            else:
                                # We just grab our color
                                player_index = leaders[side][assigned_leader]["index"]
                        markers.append(rf"$ {frame_index} $")
                        colors.append(colors_list[side][player_index])
                        positions.append(pos)
                        # If we are an alive leader we get opaque and big markers
                        if player_id in leaders[side] and player_id not in checked_in:
                            alphas.append(1)
                            sizes.append(mpl.rcParams["lines.markersize"] ** 2)
                        # If not we get partially transparent and small ones
                        else:
                            alphas.append(0.5)
                            sizes.append(0.3 * mpl.rcParams["lines.markersize"] ** 2)
                        # If we are a leader we are now checked in so everyone knows our round has not ended yet
                        if player_id in leaders[side]:
                            checked_in.add(player_id)
                    # Once we have done our first loop over a side we are initialized
                    dict_initialized[side] = True
        f, _ = plot_positions(
            positions=positions,
            colors=colors,
            markers=markers,
            alphas=alphas,
            sizes=sizes,
            map_name=map_name,
            map_type=map_type,
            dark=dark,
            apply_transformation=True,
        )
        image_files.append(f"csgo_tmp/{i}.png")
        f.savefig(image_files[-1], dpi=dpi, bbox_inches="tight")
        plt.close()
    images = []
    for file in image_files:
        images.append(imageio.imread(file))
    imageio.mimsave(filename, images, fps=fps)
    shutil.rmtree("csgo_tmp/")
    return True


def plot_rounds_different_players(
    filename: str,
    frames_list: Union[np.ndarray, list[np.ndarray]],
    map_name: str = "de_ancient",
    map_type: str = "original",
    dark: bool = False,
    fps: int = 10,
    n_frames: int = 9000,
    dist_type: str = "geodesic",
    image: bool = False,
    trajectory: bool = False,
    dtw: bool = False,
    dpi: Optional[int] = None,
) -> Literal[True]:
    """Plots a list of rounds and saves as a .gif. Each player in the first round is assigned a separate color. Players in the other rounds are matched by proximity.
    Only use untransformed coordinates.

    Args:
        filename (string): Filename to save the gif
        frames_list (Union[np.ndarray, list[np.ndarray]]): (List or higher dimensional array) of np arrays. One array for each round. Each round entry should have shape (Time_steps,2|1(sides),5,3)
        map_name (string): Map to search
        map_type (string): "original" or "simpleradar"
        dark (boolean): Only for use with map_type="simpleradar". Indicates if you want to use the SimpleRadar dark map type
        fps (integer): Number of frames per second in the gif
        n_frames (integer): The first how many frames should be plotted
        dist_type (string): String indicating the type of distance to use. Can be graph, geodesic, euclidean, manhattan, canberra or cosine
        image (boolean): Boolean indicating whether a gif of positions or a singular image of trajectories should be produced
        trajectory (boolean): Indicates whether the clustering of players should be done for the whole trajectories instead of each individual time step
        dtw: Boolean indicating whether matching should be performed via dynamic time warping (true) or euclidean (false)
        dpi (int): DPI of the result (defaults to 1000 for images and 300 for gifs)

    Returns:
        True, saves .gif
    """
    if trajectory:
        if image:
            return plot_rounds_different_players_trajectory_image(
                filename=filename,
                frames_list=frames_list,
                map_name=map_name,
                map_type=map_type,
                dark=dark,
                dist_type=dist_type,
                dtw=dtw,
                dpi=dpi if dpi else 1000,
            )
        else:  # gif
            return plot_rounds_different_players_trajectory_gif(
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
    else:  # position
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
        else:  # gif
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


if __name__ == "__main__":
    main(sys.argv[1:])