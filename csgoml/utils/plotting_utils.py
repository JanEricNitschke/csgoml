#!/usr/bin/env python
"""Collection of functions to extend awpy plotting capabilities.

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
from collections.abc import Generator
from contextlib import contextmanager
from typing import Literal, TypeAlias, TypedDict

import imageio.v3 as imageio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from awpy.analytics.nav import (
    area_distance,
    generate_centroids,
    generate_position_token,
    point_distance,
    stepped_hull,
)
from awpy.data import AREA_DIST_MATRIX, MAP_DATA, NAV, NAV_GRAPHS
from awpy.types import Area, DistanceType, GameFrame, PlotPosition, Token
from awpy.visualization.plot import (
    plot_map,
    plot_positions,
    position_transform,
)
from matplotlib import patches
from matplotlib.axes import Axes
from tqdm import tqdm

from csgoml.helpers import setup_logging
from csgoml.trajectories import trajectory_handler
from csgoml.utils.nav_utils import (
    get_area_dimensions,
    trajectory_distance,
    transform_to_traj_dimensions,
)

_TMP_DIR = "csgo_tmp"

AreaPoints: TypeAlias = dict[str, list[tuple[float, float]]]
Hulls: TypeAlias = dict[str, npt.NDArray]
Centers: TypeAlias = dict[str, tuple[float, float]]
AreasHullsCenters: TypeAlias = tuple[
    AreaPoints,
    Hulls,
    Centers,
]


def get_areas_hulls_centers(
    map_name: str,
) -> AreasHullsCenters:
    """Gets the sets of points making up the named areas of a map.

    Then builds their hull and centroid.

    Args:
        map_name (str): Specifying the map for which the features should be build.

    Returns:
        Three dictionary containing the
        points, hull and centroid of each named area in the map
    """
    hulls: dict[str, npt.NDArray] = {}
    centers: dict[str, tuple[float, float]] = {}
    area_points: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
    for area in NAV[map_name].values():
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


@contextmanager
def with_tmp_dir() -> Generator[None, None, None]:
    """Try to create and finally delete tmp dir."""
    if os.path.isdir(_TMP_DIR):
        shutil.rmtree(_TMP_DIR)
    os.mkdir(_TMP_DIR)
    try:
        yield
    finally:
        shutil.rmtree(_TMP_DIR)


def _plot_place_from_token(
    area_id: int,
    hull: npt.NDArray,
    center: tuple[float, float],
    tokens: Token,
    axis: Axes,
) -> None:
    """Plot hull and players for one specific token entry."""
    text = ""
    # Colors of borders of each area depending on number of t's/ct's that reside in it
    colors = {
        "t": ["lightsteelblue", "#300000", "#7b0000", "#b10000", "#c80000", "#ff0000"],
        "ct": ["lightsteelblue", "#360CCD", "#302DD9", "#2A4EE5", "#246FF0", "#1E90FC"],
    }
    # Plot each area twice. Once for each side.
    # If both sides have players in one tile the colors get added as
    # the alphas arent 1.
    for side in ["t", "ct"]:
        # Dont plot the "" area as it stretches across
        # the whole map and messes with clarity
        # Plot the hull of the area
        axis.plot(
            hull[:, 0],
            hull[:, 1],
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
            center[0] - (center[0] - hull[0, 0]) / 1.5,
            center[1],
            text,
            c="#EB0841",
            size="x-small",
        )


def _plot_token_frame(
    frame: GameFrame, map_name: str, areas_hulls_centers: AreasHullsCenters, axis: Axes
) -> None:
    area_points, hulls, centers = areas_hulls_centers
    # Grab the position token
    tokens = generate_position_token(map_name, frame)
    # Loop over each named area
    for area_id, area in enumerate(sorted(area_points)):
        if area:
            _plot_place_from_token(area_id, hulls[area], centers[area], tokens, axis)


@with_tmp_dir()
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
    image_files = []
    # Grab points, hull and centroid for each named area of the map
    areas_hulls_centers = get_areas_hulls_centers(map_name)
    # Loop over each frame of the round
    for frame_id, frame in tqdm(enumerate(frames)):
        fig, axis = plot_map(map_name, map_type, dark=dark)
        axis.get_xaxis().set_visible(b=False)
        axis.get_yaxis().set_visible(b=False)
        # Grab the position token
        _plot_token_frame(frame, map_name, areas_hulls_centers, axis)
        image_files.append(os.path.join(_TMP_DIR, f"{frame_id}.png"))
        fig.savefig(image_files[-1], dpi=dpi, bbox_inches="tight")
        plt.close()
    images = [imageio.imread(file) for file in image_files]
    imageio.imwrite(filename, images, duration=1000 / fps)
    return True


def _plot_area_with_name(
    area: str,
    hull: npt.NDArray,
    center: tuple[float, float],
    cent_id: int,
    axis: Axes,
    map_name: str,
) -> None:
    text_y = center[1]
    if (
        map_name in MAP_DATA
        and "z_cutoff" in (current_map_data := MAP_DATA[map_name])
        and (
            NAV[map_name][cent_id]["southEastZ"] + NAV[map_name][cent_id]["northWestZ"]
        )
        / 2
        < current_map_data["z_cutoff"]
    ):
        hull[:, 1] = np.array([y + 1024 for y in hull[:, 1]])
        text_y += 1024
    axis.plot(hull[:, 0], hull[:, 1], "-", lw=3, label=area)
    # Add name of area into its middle
    axis.text(
        center[0] - (center[0] - hull[0, 0]) / 1.5,
        text_y,
        area,
        c="#EB0841",
    )


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
    for area in area_points:
        if area:
            _plot_area_with_name(
                area, hulls[area], centers[area], cent_ids[area], axis, map_name
            )
        # Dont plot the "" area as it stretches across
        # the whole map and messes with clarity
        # but add it to the legend
        else:
            axis.plot(np.NaN, np.NaN, label="None")
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


def _plot_area(
    area: str,
    hull: npt.NDArray,
    cent_id: int,
    axis: Axes,
    map_name: str,
) -> None:
    if (
        map_name in MAP_DATA
        and "z_cutoff" in (current_map_data := MAP_DATA[map_name])
        and (
            NAV[map_name][cent_id]["southEastZ"] + NAV[map_name][cent_id]["northWestZ"]
        )
        / 2
        < current_map_data["z_cutoff"]
    ):
        hull[:, 1] = np.array([y + 1024 for y in hull[:, 1]])
    axis.plot(hull[:, 0], hull[:, 1], "-", lw=3, label=area)


def get_area_avg_z(area: Area) -> float:
    """Get the average z coordinate for an Area.

    Args:
        area (Area): Area to get the avg z coordinatefor.

    Returns:
        float: Average z coordinate of the area.
    """
    south_east_z = area["southEastZ"]
    north_west_z = area["northWestZ"]
    return (south_east_z + north_west_z) / 2


def get_z_cutoff_shift(map_name: str, avg_z: float) -> float:
    """Get the y shift needed to adjsut for potential z cutoff.

    Args:
        map_name (str): Map to consider
        avg_z (float): z value to consider

    Returns:
        float: Modifier for y values depending on z cutoff
    """
    if (
        map_name in MAP_DATA
        and "z_cutoff" in (current_map_data := MAP_DATA[map_name])
        and avg_z < current_map_data["z_cutoff"]
    ):
        return 1024
    return 0


def get_adjusted_area_dimension(
    map_name: str, area: Area
) -> tuple[float, float, float, float]:
    """Get the dimensions and corner of an area adjusted for multi level maps.

    Args:
        map_name (str): Map to consider
        area (Area): Area to get values for

    Returns:
        tuple[float, float, float, float]: width, height, southwest_x, southwest_y
    """
    width, height, southwest_x, southwest_y = get_area_dimensions(map_name, area)
    avg_z = get_area_avg_z(area)
    southwest_y += get_z_cutoff_shift(map_name, avg_z)
    return width, height, southwest_x, southwest_y


def _get_area_mid_color(area_id: int, cent_ids: set[int], rep_ids: set[int]) -> str:
    color = "yellow"
    if area_id in cent_ids and area_id in rep_ids:
        color = "green"
    elif area_id in cent_ids:
        color = "red"
    elif area_id in rep_ids:
        color = "blue"
    return color


def _plot_areas(cent_ids: dict[str, int], axis: Axes, map_name: str) -> None:
    area_points, hulls, _ = get_areas_hulls_centers(map_name)
    for area_name in area_points:
        if area_name:
            _plot_area(area_name, hulls[area_name], cent_ids[area_name], axis, map_name)
        # Dont plot the "" area as it stretches across
        # the whole map and messes with clarity
        # but add it to the legend
        else:
            axis.plot(np.NaN, np.NaN, label="None")


def _plot_mids(
    map_name: str, cent_ids: dict[str, int], rep_ids: dict[str, int], axis: Axes
) -> None:
    for area_id, area in NAV[map_name].items():
        if area_id not in cent_ids.values() and area_id not in rep_ids.values():
            continue
        width, height, southwest_x, southwest_y = get_adjusted_area_dimension(
            map_name, area
        )
        color = _get_area_mid_color(
            area_id, set(cent_ids.values()), set(rep_ids.values())
        )
        rect = patches.Rectangle(
            (southwest_x, southwest_y),
            width,
            height,
            linewidth=1,
            edgecolor=color,
            facecolor="None",
        )
        axis.add_patch(rect)


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
    _plot_areas(cent_ids, axis, map_name)
    lgd = axis.legend(
        *axis.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(1.01, 1.01)
    )
    _plot_mids(map_name, cent_ids, rep_ids, axis)
    plt.savefig(
        os.path.join(output_path, f"midpoints_{map_name}.png"),
        bbox_inches="tight",
        bbox_extra_artists=(lgd,),
        dpi=dpi,
    )
    fig.clear()
    plt.close(fig)


def _plot_tiles(map_name: str, axis: Axes) -> None:
    for area in NAV[map_name].values():
        width, height, southwest_x, southwest_y = get_adjusted_area_dimension(
            map_name, area
        )
        rect = patches.Rectangle(
            (southwest_x, southwest_y),
            width,
            height,
            linewidth=1,
            edgecolor="yellow",
            facecolor="None",
        )
        axis.add_patch(rect)


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
    _plot_tiles(map_name, axis)
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
    fig, axis = plot_map(map_name, map_type, dark=dark)
    fig.set_size_inches(19.2, 21.6)
    _plot_tiles(map_name, axis)
    # networkX type hints suck. So pyright does not know that we only
    # get a two sized tuple for our case.
    for source, dest in NAV_GRAPHS[  # pyright: ignore [reportGeneralTypeIssues]
        map_name
    ].edges():
        x1, y1, z1 = NAV_GRAPHS[map_name].nodes[source]["center"]
        x2, y2, _ = NAV_GRAPHS[map_name].nodes[dest]["center"]
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
                and "z_cutoff" in (current_map_data := MAP_DATA[map_name])
                and z1 < current_map_data[map_name]["z_cutoff"]
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
    leaders: npt.NDArray,
    current_positions: npt.NDArray,
    dist_type: DistanceType = "geodesic",
) -> tuple[int, ...]:
    """Gets the shortest mapping between players in the current round and lead players.

    Args:
        map_name (str): Name of the current map
        leaders (npt.NDArray): Numpy array of leader positions:
            Shape is: (player(5), features(4(x,y,z,area)))
        current_positions (npt.NDArray):  Numpy array of player positions:
            Shape is: (player(5), features(4(x,y,z,area)))
        dist_type (str): Indicating the type of distance to use.
            Can be graph, geodesic, euclidean.

    Returns:
        A list mapping the player at index i in the current round
        to the leader at position list[i] in the leaders dictionary.
    """
    smallest_distance = float("inf")
    best_mapping: tuple[int, ...] = tuple(range(len(current_positions)))
    # Get all distance pairs
    distance_pairs: npt.NDArray = np.zeros((len(leaders), len(current_positions)))
    for leader_i in range(len(leaders)):
        for current_i in range(len(current_positions)):
            if dist_type in ["geodesic", "graph"]:
                if map_name not in AREA_DIST_MATRIX:
                    this_dist = min(
                        area_distance(
                            map_name,
                            leaders[leader_i, 3],
                            current_positions[current_i, 3],
                            dist_type=dist_type,
                        )["distance"],
                        area_distance(
                            map_name,
                            current_positions[current_i, 3],
                            leaders[leader_i, 3],
                            dist_type=dist_type,
                        )["distance"],
                    )
                else:
                    this_dist = min(
                        AREA_DIST_MATRIX[map_name][str(int(leaders[leader_i, 3]))][
                            str(int(current_positions[current_i, 3]))
                        ][dist_type],
                        AREA_DIST_MATRIX[map_name][
                            str(int(current_positions[current_i, 3]))
                        ][str(int(leaders[leader_i, 3]))][dist_type],
                    )
            else:
                this_dist = point_distance(
                    map_name,
                    tuple(
                        current_positions[current_i, :3]
                    ),  # pyright: ignore [reportGeneralTypeIssues]
                    tuple(
                        leaders[leader_i, :3]
                    ),  # pyright: ignore [reportGeneralTypeIssues]
                    dist_type,
                )["distance"]
            distance_pairs[leader_i, current_i] = this_dist
    for mapping in itertools.permutations(range(len(leaders)), len(current_positions)):
        dist = sum(
            distance_pairs[leader_pos, current_pos]
            for current_pos, leader_pos in enumerate(mapping)
        )
        if dist < smallest_distance:
            smallest_distance = dist
            best_mapping = mapping
    return best_mapping


def get_shortest_distances_mapping_trajectory(
    map_name: str,
    leaders: npt.NDArray,
    current_positions: npt.NDArray,
    dist_type: DistanceType = "geodesic",
    *,
    dtw: bool = False,
) -> tuple[int, ...]:
    """Gets the shortest mapping between players in the current round and lead players.

    Args:
        map_name (str): Name of the current map
        leaders (npt.NDArray): Array of leaders position,
            and color index in the current frame
        current_positions (npt.NDArray): Numpy array of shape (5,Time,1,1,X) of players
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
    distance_pairs: npt.NDArray = np.zeros((len(leaders), len(current_positions)))
    for leader_i in range(len(leaders)):
        for current_i in range(len(current_positions)):
            distance_pairs[leader_i][current_i] = trajectory_distance(
                map_name,
                current_positions[current_i],
                leaders[leader_i],
                distance_type=dist_type,
                dtw=dtw,
            )
    for mapping in itertools.permutations(range(len(leaders)), len(current_positions)):
        dist: float = sum(
            distance_pairs[leader_pos, current_pos]
            for current_pos, leader_pos in enumerate(mapping)
            if current_positions[current_pos] is not None
        )
        if dist < smallest_distance:
            smallest_distance = dist
            best_mapping = mapping
    return best_mapping


colors_list = (
    ["red", "green", "black", "white", "gold"],
    ["cyan", "yellow", "fuchsia", "lime", "orange"],
)


def _get_reference_trajectory(
    frames_list: npt.NDArray,
) -> list[npt.NDArray]:
    """Get reference trajectories for multi round plotting.

    Takes the trajectories from the first round and transforms it to trajectory
    dimensions.
    """
    return [
        transform_to_traj_dimensions(frames_list[0, :, side, :, :])[:, :, :, :, (3,)]
        for side in range(frames_list.shape[2])
    ]


def plot_rounds_different_players_trajectory_image(
    filename: str,
    frames_list: npt.NDArray,
    map_name: str = "de_ancient",
    map_type: str = "original",
    dist_type: DistanceType = "geodesic",
    *,
    dark: bool = False,
    dtw: bool = False,
    dpi: int = 1000,
) -> Literal[True]:
    """Plots a collection of rounds and saves as a .png.

    Each player in the first round is assigned a separate color.
    Players in the other rounds are matched by proximity.
    Only use untransformed coordinates.

    Args:
        filename (string): Filename to save the png
        frames_list (Union[npt.NDArray, list[npt.NDArray]]):
            (List or higher dimensional array) of np arrays.
            One array for each round.
            Each round entry should have shape:
            (Time_steps,2|1(sides),5(players),4(features: x,y,z,area))
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
        True, saves .png
    """
    figure, axes = plot_map(map_name=map_name, map_type=map_type, dark=dark)
    reference_traj = _get_reference_trajectory(frames_list)
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
                axes.plot(
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
    axes.get_xaxis().set_visible(b=False)
    axes.get_yaxis().set_visible(b=False)
    figure.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close()
    return True


Mapping: TypeAlias = tuple[int, ...]


def _get_trajectory_mappings(
    frames_list: npt.NDArray,
    map_name: str,
    dist_type: DistanceType,
    *,
    dtw: bool,
) -> tuple[list[Mapping], list[Mapping]]:
    """Get a mapping of each trajectory to the reference trajectories.

    Maps each (of the 5 player) trajectory of each of the rounds
    represented in frames_list to the trajectories in the first round (reference_traj)
    by minimizing the total distance.

    The result is a tuple containing per side:
    A list that for each frame (list indices) contains the mapping of players
    to the reference trajectories.
    """
    mappings: tuple[list[Mapping], list[Mapping]] = ([], [])
    reference_traj = _get_reference_trajectory(frames_list)
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
    return mappings


def _fill_plot_positions_traj_gif(
    frames_list: npt.NDArray,
    mappings: tuple[list[Mapping], list[Mapping]],
    time_step: int,
) -> list[PlotPosition]:
    # Initialize list used to store things from all rounds to plot for each frame
    # at the given timestep
    positions: list[PlotPosition] = []
    # Do a loop to add all players in
    # all frames with their appropriate colors.
    for frame_index, frames in enumerate(frames_list):
        # Skip if this frame round has already ended
        for side in range(frames.shape[1]):
            # Now do the actual plotting
            for player_index, pos in enumerate(frames[time_step, side]):
                marker = rf"$ {frame_index} $"
                color = colors_list[side][mappings[side][frame_index][player_index]]
                # If we are an alive leader we get opaque and big markers
                if frame_index == 0:
                    alpha = 1
                    size = (mpl.rcParams["lines.markersize"] or 1) ** 2
                # If not we get partially transparent and small ones
                else:
                    alpha = 0.5
                    size = 0.3 * (mpl.rcParams["lines.markersize"] or 1) ** 2
                positions.append(PlotPosition(pos, color, marker, alpha, size))
    return positions


@with_tmp_dir()
def plot_rounds_different_players_trajectory_gif(
    filename: str,
    frames_list: npt.NDArray,
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
        frames_list (Union[npt.NDArray, list[npt.NDArray]]):
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
    image_files = []
    mappings = _get_trajectory_mappings(frames_list, map_name, dist_type, dtw=dtw)
    # Loop over all timesteps
    for i in tqdm(range(min(frames_list.shape[1], n_frames))):
        # Initialize lists used to store things from all rounds to plot for each frame
        positions = _fill_plot_positions_traj_gif(frames_list, mappings, time_step=i)
        figure, _ = plot_positions(
            positions=positions,
            map_name=map_name,
            map_type=map_type,
            dark=dark,
            apply_transformation=True,
        )
        image_files.append(os.path.join(_TMP_DIR, f"{i}.png"))
        figure.savefig(image_files[-1], dpi=dpi, bbox_inches="tight")
        plt.close()
    imageio.imwrite(
        filename, [imageio.imread(file) for file in image_files], duration=1000 / fps
    )
    return True


def _map_leaders_over_frames_list(
    frames_list: npt.NDArray, map_name: str, dist_type: DistanceType, n_frames: int
) -> npt.NDArray:
    plotting_frames = np.copy(frames_list)
    reference_players = plotting_frames[0]
    for i in tqdm(range(min(plotting_frames.shape[1], n_frames))):
        # Now do another loop to add all players in
        # all frames with their appropriate colors.
        for frame_index, frames in enumerate(plotting_frames):
            # Initialize lists used to store values for this round for this frame
            for side in range(frames.shape[1]):
                # If we have already initialized leaders
                mapping = get_shortest_distances_mapping(
                    map_name,
                    reference_players[i, side],
                    plotting_frames[frame_index, i, side],
                    dist_type=dist_type,
                )
                plotting_frames[frame_index, i, side, [mapping], :] = plotting_frames[
                    frame_index, i, side, range(len(mapping)), :
                ]
    return plotting_frames


# Currently there is no way to judge if someone has died or their round
# has ended from the input this function is actually getting.
# so just make the players from the first round the leaders and thats it.
# Can add back later functionality for passing player alive status
# and dropping and acquiring leader status.
# Leader status should be passed based on proximity to avoid random long range
# color swapping.
# https://github.com/JanEricNitschke/csgoml/issues/6
def plot_rounds_different_players_position_image(
    filename: str,
    frames_list: npt.NDArray,
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
        frames_list (npt.NDArray): Numpy array of player positions of shape:
            N_round, Time_steps, sides(1|2), player(5), features(4(x,y,z,area))
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
    plotting_frames = _map_leaders_over_frames_list(
        frames_list, map_name, dist_type, n_frames
    )
    figure, axes = plot_map(map_name=map_name, map_type=map_type, dark=dark)
    for round_num, side, player in np.ndindex(
        tuple(np.array(plotting_frames.shape)[[0, 2, 3]])
    ):
        axes.plot(
            [
                position_transform(map_name, x, "x")
                for x in plotting_frames[round_num, :, side, player, 0]
            ],
            [
                position_transform(map_name, y, "y")
                for y in plotting_frames[round_num, :, side, player, 1]
            ],
            c=colors_list[side][player],
            linestyle="-",
            linewidth=0.25,
            alpha=0.6,
        )
    axes.get_xaxis().set_visible(b=False)
    axes.get_yaxis().set_visible(b=False)
    figure.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close()
    return True


def _fill_plot_positions_pos_gif(
    frames_list: npt.NDArray,
    map_name: str,
    dist_type: DistanceType,
    time_step: int,
) -> list[PlotPosition]:
    # Initialize list used to store things from all rounds to plot for each frame
    # at the given timestep
    positions: list[PlotPosition] = []
    # Do a loop to add all players in
    # all frames with their appropriate colors.
    for frame_index, frames in enumerate(frames_list):
        # Skip if this frame round has already ended
        for side in range(frames.shape[1]):
            # Now do the actual plotting
            mapping = get_shortest_distances_mapping(
                map_name,
                frames_list[0, time_step, side],
                frames_list[frame_index, time_step, side],
                dist_type=dist_type,
            )
            for player_index, pos in enumerate(frames[time_step, side]):
                marker = rf"$ {frame_index} $"
                color = colors_list[side][mapping[player_index]]
                # If we are an alive leader we get opaque and big markers
                if frame_index == 0:
                    alpha = 1
                    size = (mpl.rcParams["lines.markersize"] or 1) ** 2
                # If not we get partially transparent and small ones
                else:
                    alpha = 0.5
                    size = 0.3 * (mpl.rcParams["lines.markersize"] or 1) ** 2
                positions.append(PlotPosition(pos, color, marker, alpha, size))
    return positions


@with_tmp_dir()
def plot_rounds_different_players_position_gif(
    filename: str,
    frames_list: npt.NDArray,
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
        frames_list (Union[npt.NDArray, list[npt.NDArray]]):
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
    image_files = []
    for i in tqdm(range(min(frames_list.shape[1], n_frames))):
        positions = _fill_plot_positions_pos_gif(
            frames_list, map_name, dist_type, time_step=i
        )
        figure, _ = plot_positions(
            positions=positions,
            map_name=map_name,
            map_type=map_type,
            dark=dark,
            apply_transformation=True,
        )
        image_files.append(os.path.join(_TMP_DIR, f"{i}.png"))
        figure.savefig(image_files[-1], dpi=dpi, bbox_inches="tight")
        plt.close()
    images = [imageio.imread(file) for file in image_files]
    imageio.imwrite(filename, images, duration=1000 / fps)
    return True


def plot_rounds_different_players(
    filename: str,
    frames_list: npt.NDArray,
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
        frames_list (Union[npt.NDArray, list[npt.NDArray]]):
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
                dpi=dpi or 1000,
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
                dpi=dpi or 300,
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
            dpi=dpi or 1000,
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
        dpi=dpi or 300,
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
    setup_logging(options)

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

    handler = trajectory_handler.TrajectoryHandler(
        json_path=r"D:\CSGO\Demos\Maps\mirage"
        r"\Analysis\Prepared_Input_Tensorflow_mirage.json",
        random_state=16,
        map_name="de_mirage",
    )
    plotting_array, _ = handler.get_clustering_input(10, "geodesic", "T", 20)
    plot_rounds_different_players_position_gif(
        "pos_gif.gif", plotting_array, "de_mirage"
    )
    plot_rounds_different_players_position_image(
        "pos_img.png", plotting_array, "de_mirage"
    )
    plot_rounds_different_players_trajectory_gif(
        "traj_gif.gif", plotting_array, "de_mirage"
    )
    plot_rounds_different_players_trajectory_image(
        "traj_img.png", plotting_array, "de_mirage"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
