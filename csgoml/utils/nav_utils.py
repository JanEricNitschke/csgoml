#!/usr/bin/env python
"""Collection of functions to extend awpy's navigation capabilities.

Example::

    centroids, reps = generate_centroids("de_dust2")
    graph = area_distance(
        "de_dust2", centroids["ExtendedA"], centroids["CTSpawn"], dist_type="graph"
    )
    geodesic = area_distance(
        "de_dust2",  centroids["ExtendedA"], centroids["CTSpawn"], dist_type="geodesic"
    )
    plot_path("test_path","de_dust2", graph, geodesic)
    graph = area_distance(
        "de_dust2", centroids["CTSpawn"], centroids["ExtendedA"], dist_type="graph"
    )
    geodesic = area_distance(
        "de_dust2", centroids["CTSpawn"],  centroids["ExtendedA"], dist_type="geodesic"
    )
    plot_path("test_path","de_dust2", graph, geodesic)
    graph = area_distance(
        "de_dust2", reps["ExtendedA"], reps["CTSpawn"], dist_type="graph"
    )
    geodesic = area_distance(
        "de_dust2",  reps["ExtendedA"], reps["CTSpawn"], dist_type="geodesic"
    )
    plot_path("test_path","de_dust2", graph, geodesic)
    graph = area_distance(
        "de_dust2", reps["CTSpawn"], reps["ExtendedA"], dist_type="graph"
    )
    geodesic = area_distance(
        "de_dust2", reps["CTSpawn"],  reps["ExtendedA"], dist_type="geodesic"
    )
    plot_path("test_path","de_dust2", graph, geodesic)

    from csgoml.utils.nav_utils import (
        get_traj_matrix_area,
        get_traj_matrix_token,
        get_traj_matrix_position,
    )
    precomputed = get_traj_matrix_token(
        precompute_array=clustering_array,
        dist_matrix=dist_matrix,
        map_area_names=map_area_names,
        dtw=dtw,
    )



"""

# pylint: disable=invalid-name, consider-using-enumerate

import argparse
import itertools
import logging
import os
import sys
from cmath import inf

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from awpy.analytics.nav import (
    area_distance,
    generate_centroids,
    position_state_distance,
    token_state_distance,
)
from awpy.data import AREA_DIST_MATRIX, NAV, PLACE_DIST_MATRIX
from awpy.types import Area, DistanceObject, DistanceType
from awpy.visualization.plot import plot_map, position_transform
from matplotlib import patches
from numba import njit, typed, types
from sympy.utilities.iterables import multiset_permutations

from csgoml.helpers import setup_logging


def get_area_dimensions(map_name: str, area: Area) -> tuple[float, float, float, float]:
    """Get dimensions and corners for an area on a map.

    Args:
        map_name (str): Map to check
        area (Area): Area to get dimensions for

    Returns:
        tuple[float, float, float,float]: width, height, southwest_x, southwest_y
    """
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
    return width, height, southwest_x, southwest_y


def mark_areas(
    output_path: str = "D:\\CSGO\\ML\\CSGOML\\Plots\\distance_matrix_fails\\",
    map_name: str = "de_ancient",
    areas: set[int] | None = None,
    dpi: int = 1000,
) -> None:
    """Plots the given map and marks the tiles of the two given areas.

    Directly saves the plot to file

    Args:
        output_path (string): Path to the output folder
        map_name (string): Map to plot
        areas (set[int]): List of ids of the areas to highlight on the map
        dpi (int): DPI of the resulting image
    """
    if areas is None:
        areas = set()
    fig, axis = plot_map(map_name=map_name, map_type="simpleradar", dark=False)
    fig.set_size_inches(19.2, 10.8)
    for a, area in NAV[map_name].items():
        if a not in areas:
            continue
        width, height, southwest_x, southwest_y = get_area_dimensions(map_name, area)
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
        os.path.join(
            output_path, f"{map_name}_{'_'.join(str(area) for area in areas)}.png"
        ),
        bbox_inches="tight",
        dpi=dpi,
    )
    fig.clear()
    plt.close(fig)


def _get_color_for_area(
    area_id: int, graph_areas: list[int], geodesic_areas: list[int]
) -> str:
    if graph_areas:
        if area_id == graph_areas[0]:
            return "purple"
        if area_id == graph_areas[-1]:
            return "orange"
    if area_id in graph_areas:
        return "green" if area_id in geodesic_areas else "red"
    return "blue" if area_id in geodesic_areas else "yellow"


def plot_path(
    output_path: str = "D:\\CSGO\\ML\\CSGOML\\Plots\\distance_matrix_examples\\",
    map_name: str = "de_ancient",
    graph: DistanceObject | None = None,
    geodesic: DistanceObject | None = None,
    dpi: int = 1000,
) -> None:
    """Plots the given map and two paths between two areas.

    Args:
        output_path (string): Path to the output folder
        map_name (string): Map to plot
        graph (dict): The graph["distance"] is the distance between two areas.
            graph["areas"] is a list of int area ids in the
            shortest path between the first and last entry.
        geodesic (dict): The geodesic["distance"] is the distance between two areas.
            geodesic["areas"] is a list of int area ids in the
            shortest path between the first and last entry.
        dpi (int): DPI of the resulting image

    Returns:
        None (Directly saves the plot to file)
    """
    if graph is None:
        graph = {
            "distanceType": "graph",
            "distance": float("inf"),
            "areas": [],
        }
    if geodesic is None:
        geodesic = {
            "distanceType": "geodesic",
            "distance": float("inf"),
            "areas": [],
        }
    fig, axis = plot_map(map_name=map_name, map_type="simplerader", dark=True)
    fig.set_size_inches(19.2, 10.8)
    for area_id, area in NAV[map_name].items():
        if area_id not in graph["areas"] and area_id not in geodesic["areas"]:
            continue
        width, height, southwest_x, southwest_y = get_area_dimensions(map_name, area)
        color = _get_color_for_area(area_id, graph["areas"], geodesic["areas"])
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
        os.path.join(
            output_path,
            f"{map_name}_{graph['areas'][0]}_{graph['areas'][-1]}_"
            f"{graph['distance']}_{geodesic['distance']}.png",
        ),
        bbox_inches="tight",
        dpi=dpi,
    )
    fig.clear()
    plt.close(fig)


value_float = types.float64
key_float = types.UniTuple(types.int64, 2)


def trajectory_distance(
    map_name: str,
    trajectory_array_1: np.ndarray,
    trajectory_array_2: np.ndarray,
    distance_type: DistanceType = "geodesic",
    *,
    dtw: bool = False,
) -> float:
    """Calculates a distance distance between two trajectories.

    Args:
        map_name (string): Map under consideration
        trajectory_array_1: Numpy array with shape (n_Time,2|1, 5, 3) with the
            first index indicating the team,
            the second the player and the third the coordinate
            Alternatively the last dimension can have size 1 containing the area_id.
            Used only with geodesic and graph distance
        trajectory_array_2: Numpy array with shape (n_Time,2|1, 5, 3) with the
            first index indicating the team,
            the second the player and the third the coordinate
            Alternatively the last dimension can have size 1 containing the area_id.
            Used only with geodesic and graph distance
        distance_type: String indicating how the distance between two player
            positions should be calculated.
            Options are "geodesic", "graph", "euclidean" and "edit_distance"
        dtw: Boolean indicating whether matching should be performed
            via dynamic time warping (true) or euclidean (false).

    Returns:
        A float representing the distance between these two trajectories.
    """
    length = max(len(trajectory_array_1), len(trajectory_array_2))
    # Token arrays have shape (n_time, token_length) so 2 dimensions
    if len(trajectory_array_1.shape) > 2.5:  # noqa: PLR2004
        dist_func = position_state_distance
    else:
        dist_func = token_state_distance

    if dtw:
        dtw_matrix = typed.Dict.empty(key_float, value_float)

        for i in range(len(trajectory_array_1)):
            dtw_matrix[(i, -1)] = inf
        for i in range(len(trajectory_array_2)):
            dtw_matrix[(-1, i)] = inf
        dtw_matrix[(-1, -1)] = 0

        for i in range(len(trajectory_array_1)):
            for j in range(len(trajectory_array_2)):
                dist = dist_func(
                    map_name,
                    trajectory_array_1[i],
                    trajectory_array_2[j],
                    distance_type,
                )
                dtw_matrix[(i, j)] = dist + min(
                    dtw_matrix[(i - 1, j)],
                    dtw_matrix[(i, j - 1)],
                    dtw_matrix[(i - 1, j - 1)],
                )

        return (
            dtw_matrix[len(trajectory_array_1) - 1, len(trajectory_array_2) - 1]
        ) / length

    return (
        sum(
            (
                dist_func(
                    map_name,
                    trajectory_array_1[time]
                    if time in range(len(trajectory_array_1))
                    else trajectory_array_1[-1],
                    trajectory_array_2[time]
                    if time in range(len(trajectory_array_2))
                    else trajectory_array_2[-1],
                    distance_type,
                )
            )
            for time in range(length)
        )
        / length
    )


# @np.vectorize
@njit
def compute_dtw(
    trajectory_array_1: np.ndarray,
    trajectory_array_2: np.ndarray,
    dist_matrix: np.ndarray,
) -> float:
    """Calculates a distance distance between two trajectories.

    Args:
        trajectory_array_1: Numpy array with shape (n_Time,2|1, 5, 3)
            with the first index indicating the team,
            the second the player and the third the coordinate
        trajectory_array_2: Numpy array with shape (n_Time,2|1, 5, 3)
            with the first index indicating the team,
            the second the player and the third the coordinate
        dist_matrix: Numpy array that contains the
            precomputed distance between any pair of areas
        dtw: Boolean indicating whether matching should be performed via
            dynamic time warping (true) or euclidean (false)

    Returns:
        A float representing the distance between these two trajectories
    """
    length = max(len(trajectory_array_1), len(trajectory_array_2))
    dtw_matrix = typed.Dict.empty(key_float, value_float)
    dtw_matrix[(-1, -1)] = 0

    for i in range(len(trajectory_array_1)):
        dtw_matrix[(i, -1)] = inf
        for j in range(len(trajectory_array_2)):
            dtw_matrix[(-1, j)] = inf
            dtw_matrix[(i, j)] = dist_matrix[i, j] + min(
                dtw_matrix[(i - 1, j)],
                dtw_matrix[(i, j - 1)],
                dtw_matrix[(i - 1, j - 1)],
            )

    return (
        dtw_matrix[len(trajectory_array_1) - 1, len(trajectory_array_2) - 1]
    ) / length


# @njit
def fast_token_trajectory_distance(
    trajectory_array_1: np.ndarray,
    trajectory_array_2: np.ndarray,
    dist_matrix: np.ndarray,
    *,
    dtw: bool = False,
) -> float:
    """Calculates a distance distance between two trajectories.

    Args:
        trajectory_array_1: Numpy array with shape (n_Time,len(token))
        trajectory_array_2: Numpy array with shape (n_Time,len(token))
        dist_matrix: Numpy array that contains the
            precomputed distance between any pair of areas
        map_area_names: List of strings of area names
        dtw: Boolean indicating whether matching should be performed via
            dynamic time warping (true) or euclidean (false)

    Returns:
        A float representing the distance between these two trajectories
    """
    length = max(len(trajectory_array_1), len(trajectory_array_2))
    if dtw:
        dtw_matrix = typed.Dict.empty(key_float, value_float)

        for i in range(len(trajectory_array_1)):
            dtw_matrix[(i, -1)] = inf
        for i in range(len(trajectory_array_2)):
            dtw_matrix[(-1, i)] = inf
        dtw_matrix[(-1, -1)] = 0

        for i in range(len(trajectory_array_1)):
            for j in range(len(trajectory_array_2)):
                dist = fast_token_state_distance(
                    token_array_1=trajectory_array_1[i],
                    token_array_2=trajectory_array_2[j],
                    dist_matrix=dist_matrix,
                )
                dtw_matrix[(i, j)] = dist + min(
                    dtw_matrix[(i - 1, j)],
                    dtw_matrix[(i, j - 1)],
                    dtw_matrix[(i - 1, j - 1)],
                )

        return (
            dtw_matrix[len(trajectory_array_1) - 1, len(trajectory_array_2) - 1]
        ) / length
    dist = sum(
        fast_token_state_distance(
            token_array_1=trajectory_array_1[time]
            if time in range(len(trajectory_array_1))
            else trajectory_array_1[-1],
            token_array_2=trajectory_array_2[time]
            if time in range(len(trajectory_array_2))
            else trajectory_array_2[-1],
            dist_matrix=dist_matrix,
        )
        for time in range(length)
    )
    return dist / length


def _get_compressed_area_dist_matrix(
    map_name: str,
) -> tuple[npt.NDArray, dict[int, int]]:
    """Generates a compressed area distance matrix.

    Args:
        map_name (str): Map under consideration

    Returns:
        Array of compressed matrix and matching between indices and areas.
    """
    old_dist_matrix = AREA_DIST_MATRIX[map_name]
    dist_matrix = np.zeros((len(old_dist_matrix), len(old_dist_matrix)))
    matching: dict[int, int] = {}
    for idx1, area1 in enumerate(sorted(old_dist_matrix)):
        matching[int(area1)] = idx1
        for idx2, area2 in enumerate(sorted(old_dist_matrix[area1])):
            dist_matrix[idx1, idx2] = min(
                old_dist_matrix[area1][area2]["geodesic"],
                old_dist_matrix[area2][area1]["geodesic"],
                sys.maxsize / 6,
            )
    return dist_matrix, matching


def _apply_matching(
    precompute_array: npt.NDArray, matching: dict[int, int]
) -> npt.NDArray:
    def get_matching(x: int | str | float) -> int:
        return matching[int(x)] if int(x) in matching else next(iter(matching.values()))

    return np.vectorize(get_matching)(precompute_array)


def _prepare_trajectories(
    precompute_array: npt.NDArray, map_name: str
) -> tuple[npt.NDArray, npt.NDArray]:
    dist_matrix, matching = _get_compressed_area_dist_matrix(map_name)
    logging.info(precompute_array.shape)
    precompute_array = _apply_matching(precompute_array, matching)

    logging.info(precompute_array.shape)
    # Reduce Shape of precompute array is:
    # N_rounds, N_times, N_teams, N_players, N_features
    # With N_features always being 1 here. So just squeeze that away at the start.
    return np.squeeze(precompute_array, axis=-1), dist_matrix


def _get_traj_matrix_area_dtw(
    precompute_array: npt.NDArray, permutations: npt.NDArray, dist_matrix: npt.NDArray
) -> npt.NDArray:
    """Calculate the distance matrix using dynamic time warping.

    Args:
        precompute_array (npt.NDArray): Array of trajectories to compute matrix for.
        permutations (npt.NDArray): Array of permutation indices to permute players.
        dist_matrix (npt.NDArray): Compressed distance matrix between any two positions.

    Returns:
        npt.NDArray: Matrix of distances between trajectories.
    """
    precomputed = np.zeros((len(precompute_array), len(precompute_array)))
    for i in range(len(precompute_array)):
        if i % 50 == 0:
            logging.info(i)
        for j in range(i + 1, len(precompute_array)):
            traj_array_1 = precompute_array[i]
            traj_array_2 = precompute_array[j]
            dtw_matrix = (
                dist_matrix[
                    traj_array_1[:, np.newaxis, ..., permutations],
                    traj_array_2[..., np.newaxis, :],
                ]
                .mean(-1)
                .min(-1)
                .mean(-1)
            )
            precomputed[i, j] = compute_dtw(
                precompute_array[i], precompute_array[j], dtw_matrix
            )
    precomputed += precomputed.T
    return precomputed


def _get_traj_matrix_area_linear(
    precompute_array: npt.NDArray, permutations: npt.NDArray, dist_matrix: npt.NDArray
) -> npt.NDArray:
    """Calculate the distance matrix using dynamic time warping.

    Args:
        precompute_array (npt.NDArray): Array of trajectories to compute matrix for.
        permutations (npt.NDArray): Array of permutation indices to permute players.
        dist_matrix (npt.NDArray): Compressed distance matrix between any two positions.

    Returns:
        npt.NDArray: Matrix of distances between trajectories.
    """
    try:
        precomputed = (
            dist_matrix[
                precompute_array[:, np.newaxis, ..., permutations],
                precompute_array[..., np.newaxis, :],
            ]
            .mean(-1)
            .min(-1)
            .mean(-1)
            .mean(-1)
        )
    except MemoryError as e:
        logging.info("Caught memory error %s", e)
        precomputed = np.zeros((len(precompute_array), len(precompute_array)))
        for i in range(len(precompute_array)):
            if i % 50 == 0:
                logging.info(i)
            for j in range(i + 1, len(precompute_array)):
                traj_array_1 = precompute_array[i]
                traj_array_2 = precompute_array[j]
                precomputed[i, j] = (
                    dist_matrix[
                        traj_array_1[..., permutations],
                        traj_array_2[..., np.newaxis, :],
                    ]
                    .mean(-1)
                    .min(-1)
                    .mean(-1)
                    .mean(-1)
                )
        precomputed += precomputed.T
    return precomputed


def get_traj_matrix_area(
    precompute_array: np.ndarray,
    map_name: str,
    *,
    dtw: bool,
) -> np.ndarray:
    """Precompute the distance matrix for all trajectories of areas.

    Args:
        precompute_array: Numpy array of trajectories for which to
            compute the distance matrix
        map_name: Map under consideration
        dtw: Boolean indicating whether matching should be performed via
            dynamic time warping (true) or euclidean (false)

    Returns:
        Numpy array of distances between trajectories
    """
    precompute_array, dist_matrix = _prepare_trajectories(precompute_array, map_name)
    permutations = np.array(
        list(itertools.permutations(range(precompute_array.shape[-1])))
    )
    if dtw:
        return _get_traj_matrix_area_dtw(precompute_array, permutations, dist_matrix)
    return _get_traj_matrix_area_linear(precompute_array, permutations, dist_matrix)


def _get_compressed_places_dist_matrix(
    map_name: str,
) -> tuple[npt.NDArray, dict[str, int]]:
    """Generates a compressed place distance matrix for individual places.

    Args:
        map_name (str): Map under consideration

    Returns:
        Array of compressed matrix and matching between indices and areas.
    """
    old_dist_matrix = PLACE_DIST_MATRIX[map_name]
    dist_matrix = np.zeros((len(old_dist_matrix), len(old_dist_matrix)))
    matching: dict[str, int] = {}
    for idx1, place1 in enumerate(sorted(old_dist_matrix)):
        matching[place1] = idx1
        for idx2, place2 in enumerate(sorted(old_dist_matrix[place1])):
            dist_matrix[idx1, idx2] = min(
                old_dist_matrix[place1][place2]["geodesic"]["centroid"],
                old_dist_matrix[place2][place1]["geodesic"]["centroid"],
                sys.maxsize / 6,
            )
    return dist_matrix, matching


def _apply_matching_place(
    precompute_array: npt.NDArray, matching: dict[str, int], map_name: str
) -> npt.NDArray:
    map_areas = NAV[map_name]

    def get_matching(x: int | float | str) -> int:
        return (
            matching[map_areas[int(x)]["areaName"]]
            if int(x) in map_areas and map_areas[int(x)]["areaName"] in matching
            else next(iter(matching.values()))
        )

    return np.vectorize(get_matching)(precompute_array)


def _prepare_trajectories_place(
    precompute_array: npt.NDArray, map_name: str
) -> tuple[npt.NDArray, npt.NDArray]:
    dist_matrix, matching = _get_compressed_places_dist_matrix(map_name)
    logging.info(precompute_array.shape)
    precompute_array = _apply_matching_place(precompute_array, matching, map_name)

    logging.info(precompute_array.shape)
    # Reduce Shape of precompute array is:
    # N_rounds, N_times, N_teams, N_players, N_features
    # With N_features always being 1 here. So just squeeze that away at the start.
    return np.squeeze(precompute_array, axis=-1), dist_matrix


def get_traj_matrix_place(
    precompute_array: np.ndarray,
    map_name: str,
    *,
    dtw: bool,
) -> np.ndarray:
    """Precompute the distance matrix for all trajectories of areas.

    Args:
        precompute_array: Numpy array of trajectories for which to
            compute the distance matrix
        map_name: Map under consideration
        dtw: Boolean indicating whether matching should be performed via
            dynamic time warping (true) or euclidean (false)

    Returns:
        Numpy array of distances between trajectories
    """
    precompute_array, dist_matrix = _prepare_trajectories_place(
        precompute_array, map_name
    )
    permutations = np.array(
        list(itertools.permutations(range(precompute_array.shape[-1])))
    )
    if dtw:
        return _get_traj_matrix_area_dtw(precompute_array, permutations, dist_matrix)
    return _get_traj_matrix_area_linear(precompute_array, permutations, dist_matrix)


def _get_map_area_names(map_name: str) -> list[str]:
    """Generates list of all named place on a map in sorted order.

    Args:
        map_name (str): Map under consideration.

    Returns:
        sorted list of named places
    """
    map_area_names = {NAV[map_name][area_id]["areaName"] for area_id in NAV[map_name]}
    return sorted(map_area_names)


def _get_compressed_place_dist_matrix(
    map_name: str,
) -> dict[str, dict[str, float]]:
    """Generates a compressed place distance matrix for tokens.

    Args:
        map_name (str): Map under consideration
    Returns:
        typed dict of compressed matrix
    """
    old_dist_matrix = PLACE_DIST_MATRIX[map_name]
    dist_matrix: dict[str, dict[str, float]] = {}
    for place1 in old_dist_matrix:
        for place2 in old_dist_matrix[place1]:
            if place1 not in dist_matrix:
                dist_matrix[place1] = {}
            dist_matrix[place1][place2] = min(
                old_dist_matrix[place1][place2]["geodesic"]["centroid"],
                old_dist_matrix[place2][place1]["geodesic"]["centroid"],
                sys.maxsize / 6,
            )
    return dist_matrix


def get_traj_matrix_token(
    precompute_array: np.ndarray,
    map_name: str,
    *,
    dtw: bool,
) -> np.ndarray:
    """Precompute the distance matrix for all trajectories of tokens.

    Args:
        precompute_array: Array of trajectories for which to compute the distance matrix
        map_name (str): Map under considertation
        dtw: Boolean indicating whether matching should be performed via
            dynamic time warping (true) or euclidean (false)

    Returns:
        Numpy array of distances between trajectories
    """
    dist_matrix, _ = _get_compressed_places_dist_matrix(map_name)
    precomputed = np.zeros((len(precompute_array), len(precompute_array)))
    for i in range(len(precompute_array)):
        if i % 50 == 0:
            logging.info(i)
        for j in range(i + 1, len(precompute_array)):
            precomputed[i, j] = fast_token_trajectory_distance(
                precompute_array[i],
                precompute_array[j],
                dist_matrix,
                dtw=dtw,
            )
    precomputed += precomputed.T
    return precomputed


def _get_traj_matrix_position_dtw(precompute_array: npt.NDArray) -> npt.NDArray:
    permutations = np.array(
        list(itertools.permutations(range(precompute_array.shape[-2])))
    )
    precomputed = np.zeros((len(precompute_array), len(precompute_array)))
    for i in range(len(precompute_array)):
        if i % 50 == 0:
            logging.info(i)
        for j in range(i + 1, len(precompute_array)):
            traj_array_1 = precompute_array[i]
            traj_array_2 = precompute_array[j]
            dtw_matrix = (
                np.linalg.norm(
                    traj_array_1[:, np.newaxis, ..., permutations, :]
                    - traj_array_2[..., np.newaxis, :, :],
                    ord=2,
                    axis=-1,
                )
                .mean(-1)
                .min(-1)
                .mean(-1)
            )
            precomputed[i, j] = compute_dtw(
                precompute_array[i], precompute_array[j], dtw_matrix
            )
    precomputed += precomputed.T
    return precomputed


def _get_traj_matrix_position_linear(precompute_array: npt.NDArray) -> npt.NDArray:
    permutations = np.array(
        list(itertools.permutations(range(precompute_array.shape[-2])))
    )
    try:
        precomputed = (
            np.linalg.norm(
                precompute_array[:, np.newaxis, ..., permutations, :]
                - precompute_array[..., np.newaxis, :, :],
                ord=2,
                axis=-1,
            )
            .mean(-1)
            .min(-1)
            .mean(-1)
            .mean(-1)
        )
    except MemoryError as e:
        logging.info("Caught memory error %s", e)
        precomputed = np.zeros((len(precompute_array), len(precompute_array)))
        for i in range(len(precompute_array)):
            if i % 50 == 0:
                logging.info(i)
            for j in range(i + 1, len(precompute_array)):
                traj_array_1 = precompute_array[i]
                traj_array_2 = precompute_array[j]
                precomputed[i, j] = (
                    np.linalg.norm(
                        traj_array_1[..., permutations, :]
                        - traj_array_2[..., np.newaxis, :, :],
                        ord=2,
                        axis=-1,
                    )
                    .mean(-1)
                    .min(-1)
                    .mean(-1)
                    .mean(-1)
                )
        precomputed += precomputed.T
    return precomputed


def get_traj_matrix_position(
    precompute_array: npt.NDArray, *, dtw: bool
) -> npt.NDArray:
    """Precompute the distance matrix for all trajectories of positions.

    Args:
        precompute_array: Array of trajectories for which to compute the distance matrix
        dtw: Boolean indicating whether matching should be performed via
            dynamic time warping (true) or euclidean (false)

    Returns:
        Numpy array of distances between trajectories
    """
    if dtw:
        return _get_traj_matrix_position_dtw(precompute_array)
    return _get_traj_matrix_position_linear(precompute_array)


# @njit
def fast_token_state_distance(
    token_array_1: np.ndarray,
    token_array_2: np.ndarray,
    dist_matrix: np.ndarray,
) -> float:
    """Calculates a distance between two game states based on tokens.

    Args:
        token_array_1: Numpy array with shape (len(token))
        token_array_2: Numpy array with shape (len(token))
        dist_matrix: Numpy array that contains the precomputed distance
            between any pair of areas

    Returns:
        A float representing the distance between these two game states
    """
    # pylint: disable=too-many-locals
    # Disable this check because we want maximum performance here and
    # so want to inline everything.
    # More complicated distances based on actual area locations
    side_distance = inf
    # Get the sub arrays for this team from the total array
    # Make sure array1 is the larger one
    if sum(token_array_1) < sum(token_array_2):
        token_array_1, token_array_2 = token_array_2, token_array_1
    size = sum(token_array_2)
    if not size:
        return sys.maxsize
    # Get the indices where array1 and array2 have larger values than the other.
    # Use each index as often as it is larger
    diff_array = np.subtract(token_array_1, token_array_2)
    pos_indices: list[int] = []
    neg_indices: list[int] = []
    for i, difference in enumerate(diff_array):
        if difference > 0:
            pos_indices.extend([i] * int(difference))
        elif difference < 0:
            neg_indices.extend([i] * int(abs(difference)))
    if len(pos_indices) < len(neg_indices):
        neg_indices, pos_indices = pos_indices, neg_indices
    # Get all possible mappings between the differences
    # Eg: diff array is [1,1,-1,-1] then pos_indices is [0,1] and neg_indices is [2,3]
    # The possible mappings are then [(0,2),(1,3)] and [(0,3),(1,2)]
    for mapping in {
        tuple(sorted(zip(perm, neg_indices, strict=True)))
        for perm in multiset_permutations(pos_indices, len(neg_indices))
    }:
        cur_dist: float = sum(dist_matrix[area1, area2] for area1, area2 in mapping)
        side_distance = min(side_distance, cur_dist)
    return side_distance / size


def transform_to_traj_dimensions(pos_array: np.ndarray) -> np.ndarray:
    """Transforms a numpy array of shape (Time,5,X) to (5,Time,1,1,X).

    To allow for individual trajectory distances.
    Only needed when trying to get the distance between
    single player trajectories within "get_shortest_distance_mapping".

    Args:
        pos_array (numpy array): Numpy array with shape (Time,5,3)

    Returns:
            numpy array of shape  (5,Time,1,1,3)
    """
    shape = pos_array.shape
    dimensions = [shape[1], len(pos_array), 1, 1, shape[2]]
    return_array = np.zeros(tuple(dimensions))
    for i in range(5):
        return_array[i, :, :, :, :] = pos_array[:, np.newaxis, np.newaxis, i, :]
    return return_array


def main(args: list[str]) -> None:
    """Plots paths between a variety of tiles.

    Uses awpy and the extension functions defined in this module.
    """
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
    setup_logging(options)

    centroids, reps = generate_centroids("de_dust2")
    graph = area_distance(
        "de_dust2", centroids["ExtendedA"], centroids["CTSpawn"], dist_type="graph"
    )
    geodesic = area_distance(
        "de_dust2", centroids["ExtendedA"], centroids["CTSpawn"], dist_type="geodesic"
    )
    plot_path("test_path", "de_dust2", graph, geodesic)
    graph = area_distance(
        "de_dust2", centroids["CTSpawn"], centroids["ExtendedA"], dist_type="graph"
    )
    geodesic = area_distance(
        "de_dust2", centroids["CTSpawn"], centroids["ExtendedA"], dist_type="geodesic"
    )
    plot_path("test_path", "de_dust2", graph, geodesic)
    graph = area_distance(
        "de_dust2", reps["ExtendedA"], reps["CTSpawn"], dist_type="graph"
    )
    geodesic = area_distance(
        "de_dust2", reps["ExtendedA"], reps["CTSpawn"], dist_type="geodesic"
    )
    plot_path("test_path", "de_dust2", graph, geodesic)
    graph = area_distance(
        "de_dust2", reps["CTSpawn"], reps["ExtendedA"], dist_type="graph"
    )
    geodesic = area_distance(
        "de_dust2", reps["CTSpawn"], reps["ExtendedA"], dist_type="geodesic"
    )
    plot_path("test_path", "de_dust2", graph, geodesic)


if __name__ == "__main__":
    main(sys.argv[1:])
