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
# pylint: disable=invalid-name, consider-using-enumerate

import sys
import logging
import argparse
from cmath import inf
import numpy as np
from matplotlib import patches
import matplotlib.pyplot as plt
from numba import njit, typed, types
from sympy.utilities.iterables import multiset_permutations
from awpy.visualization.plot import plot_map, position_transform
from awpy.data import NAV
from awpy.analytics.nav import (
    position_state_distance,
    token_state_distance,
)


def mark_areas(map_name: str, areas: list[float]) -> None:
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


def plot_path(map_name: str, graph: dict, geodesic: dict) -> None:
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


value_float = types.float64
key_float = types.UniTuple(types.int64, 2)


def trajectory_distance(
    map_name: str,
    trajectory_array_1: np.ndarray,
    trajectory_array_2: np.ndarray,
    distance_type: str = "geodesic",
    dtw: bool = False,
) -> float:
    """Calculates a distance distance between two trajectories
    Args:
        map_name (string): Map under consideration
        trajectory_array_1: Numpy array with shape (n_Time,2|1, 5, 3) with the first index indicating the team, the second the player and the third the coordinate
        trajectory_array_2: Numpy array with shape (n_Time,2|1, 5, 3) with the first index indicating the team, the second the player and the third the coordinate
        distance_type: String indicating how the distance between two player positions should be calculated. Options are "geodesic", "graph", "euclidean" and "edit_distance"
        dtw: Boolean indicating whether matching should be performed via dynamic time warping (true) or euclidean (false)
    Returns:
        A float representing the distance between these two trajectories
    """

    length = max(len(trajectory_array_1), len(trajectory_array_2))
    if len(trajectory_array_1.shape) > 2.5:
        dist_func = position_state_distance
    else:
        dist_func = token_state_distance

    if dtw:
        DTW = typed.Dict.empty(key_float, value_float)

        for i in range(len(trajectory_array_1)):
            DTW[(i, -1)] = inf
        for i in range(len(trajectory_array_2)):
            DTW[(-1, i)] = inf
        DTW[(-1, -1)] = 0

        for i in range(len(trajectory_array_1)):
            for j in range(len(trajectory_array_2)):
                dist = position_state_distance(
                    map_name=map_name,
                    position_array_1=trajectory_array_1[i],
                    position_array_2=trajectory_array_2[j],
                    distance_type=distance_type,
                )
                DTW[(i, j)] = dist + min(
                    DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)]
                )

        return (DTW[len(trajectory_array_1) - 1, len(trajectory_array_2) - 1]) / length

    distance = 0
    for time in range(length):
        distance += (
            dist_func(
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
    return distance


@njit
def fast_area_trajectory_distance(
    trajectory_array_1: np.ndarray,
    trajectory_array_2: np.ndarray,
    dist_matrix: dict,
    dtw: bool = False,
) -> float:
    """Calculates a distance distance between two trajectories

    Args:
        trajectory_array_1: Numpy array with shape (n_Time,2|1, 5, 3) with the first index indicating the team, the second the player and the third the coordinate
        trajectory_array_2: Numpy array with shape (n_Time,2|1, 5, 3) with the first index indicating the team, the second the player and the third the coordinate
        dist_matrix: Nested dict that contains the precomputed distance between any pair of areas
        dtw: Boolean indicating whether matching should be performed via dynamic time warping (true) or euclidean (false)

    Returns:
        A float representing the distance between these two trajectories
    """
    length = max(len(trajectory_array_1), len(trajectory_array_2))
    if dtw:
        DTW = typed.Dict.empty(key_float, value_float)

        for i in range(len(trajectory_array_1)):
            DTW[(i, -1)] = inf
        for i in range(len(trajectory_array_2)):
            DTW[(-1, i)] = inf
        DTW[(-1, -1)] = 0

        for i in range(len(trajectory_array_1)):
            for j in range(len(trajectory_array_2)):
                dist = fast_area_state_distance(
                    position_array_1=trajectory_array_1[i],
                    position_array_2=trajectory_array_2[j],
                    dist_matrix=dist_matrix,
                )
                DTW[(i, j)] = dist + min(
                    DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)]
                )

        return (DTW[len(trajectory_array_1) - 1, len(trajectory_array_2) - 1]) / length
    dist = 0
    for time in range(length):
        dist += fast_area_state_distance(
            position_array_1=trajectory_array_1[time]
            if time in range(len(trajectory_array_1))
            else trajectory_array_1[-1],
            position_array_2=trajectory_array_2[time]
            if time in range(len(trajectory_array_2))
            else trajectory_array_2[-1],
            dist_matrix=dist_matrix,
        )
    return dist / length


# @njit
def fast_token_trajectory_distance(
    trajectory_array_1: np.ndarray,
    trajectory_array_2: np.ndarray,
    dist_matrix: dict,
    map_area_names: list[str],
    dtw: bool = False,
) -> float:
    """Calculates a distance distance between two trajectories

    Args:
        trajectory_array_1: Numpy array with shape (n_Time,len(token))
        trajectory_array_2: Numpy array with shape (n_Time,len(token))
        dist_matrix: Nested dict that contains the precomputed distance between any pair of areas
        map_area_names: List of strings of area names
        dtw: Boolean indicating whether matching should be performed via dynamic time warping (true) or euclidean (false)

    Returns:
        A float representing the distance between these two trajectories
    """
    length = max(len(trajectory_array_1), len(trajectory_array_2))
    if dtw:
        DTW = typed.Dict.empty(key_float, value_float)

        for i in range(len(trajectory_array_1)):
            DTW[(i, -1)] = inf
        for i in range(len(trajectory_array_2)):
            DTW[(-1, i)] = inf
        DTW[(-1, -1)] = 0

        for i in range(len(trajectory_array_1)):
            for j in range(len(trajectory_array_2)):
                dist = fast_token_state_distance(
                    token_array_1=trajectory_array_1[i],
                    token_array_2=trajectory_array_2[j],
                    dist_matrix=dist_matrix,
                    map_area_names=map_area_names,
                )
                DTW[(i, j)] = dist + min(
                    DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)]
                )

        return (DTW[len(trajectory_array_1) - 1, len(trajectory_array_2) - 1]) / length
    dist = 0
    for time in range(length):
        dist += fast_token_state_distance(
            token_array_1=trajectory_array_1[time]
            if time in range(len(trajectory_array_1))
            else trajectory_array_1[-1],
            token_array_2=trajectory_array_2[time]
            if time in range(len(trajectory_array_2))
            else trajectory_array_2[-1],
            dist_matrix=dist_matrix,
            map_area_names=map_area_names,
        )
    return dist / length


@njit
def fast_position_trajectory_distance(
    trajectory_array_1: np.ndarray, trajectory_array_2: np.ndarray, dtw: bool = False
) -> float:
    """Calculates a distance distance between two trajectories

    Args:
        trajectory_array_1: Numpy array with shape (n_Time,len(token))
        trajectory_array_2: Numpy array with shape (n_Time,len(token))
        dist_matrix: Nested dict that contains the precomputed distance between any pair of areas
        dtw: Boolean indicating whether matching should be performed via dynamic time warping (true) or euclidean (false)

    Returns:
        A float representing the distance between these two trajectories
    """
    length = max(len(trajectory_array_1), len(trajectory_array_2))
    if dtw:
        DTW = typed.Dict.empty(key_float, value_float)

        for i in range(len(trajectory_array_1)):
            DTW[(i, -1)] = inf
        for i in range(len(trajectory_array_2)):
            DTW[(-1, i)] = inf
        DTW[(-1, -1)] = 0

        for i in range(len(trajectory_array_1)):
            for j in range(len(trajectory_array_2)):
                dist = fast_position_state_distance(
                    position_array_1=trajectory_array_1[i],
                    position_array_2=trajectory_array_2[j],
                )
                DTW[(i, j)] = dist + min(
                    DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)]
                )

        return (DTW[len(trajectory_array_1) - 1, len(trajectory_array_2) - 1]) / length
    dist = 0
    for time in range(length):
        dist += fast_position_state_distance(
            position_array_1=trajectory_array_1[time]
            if time in range(len(trajectory_array_1))
            else trajectory_array_1[-1],
            position_array_2=trajectory_array_2[time]
            if time in range(len(trajectory_array_2))
            else trajectory_array_2[-1],
        )
    return dist / length


def get_traj_matrix_area(
    precompute_array: np.ndarray, dist_matrix: dict, dtw: bool
) -> np.ndarray:
    """Precompute the distance matrix for all trajectories of areas

    Args:
        precompute_array: Numpy array of trajectories for which to compute the distance matrix
        dist_matrix: Dict as distance matrix between areas
        dtw: Boolean indicating whether matching should be performed via dynamic time warping (true) or euclidean (false)

    Returns:
        Numpy array of distances between trajectories
    """
    precomputed = np.zeros((len(precompute_array), len(precompute_array)))
    for i in range(len(precompute_array)):
        if i % 50 == 0:
            logging.info(i)
        for j in range(i + 1, len(precompute_array)):
            precomputed[i][j] = fast_area_trajectory_distance(
                precompute_array[i], precompute_array[j], dist_matrix, dtw=dtw
            )
    precomputed += precomputed.T
    return precomputed


def get_traj_matrix_token(
    precompute_array: np.ndarray,
    dist_matrix: dict,
    map_area_names: list[str],
    dtw: bool,
) -> np.ndarray:
    """Precompute the distance matrix for all trajectories of tokens

    Args:
        precompute_array: Numpy array of trajectories for which to compute the distance matrix
        dist_matrix: Dict as distance matrix between places
        map_area_names: List of strings of area names
        dtw: Boolean indicating whether matching should be performed via dynamic time warping (true) or euclidean (false)

    Returns:
        Numpy array of distances between trajectories"""
    precomputed = np.zeros((len(precompute_array), len(precompute_array)))
    for i in range(len(precompute_array)):
        if i % 50 == 0:
            logging.info(i)
        for j in range(i + 1, len(precompute_array)):
            precomputed[i][j] = fast_token_trajectory_distance(
                precompute_array[i],
                precompute_array[j],
                dist_matrix,
                map_area_names=map_area_names,
                dtw=dtw,
            )
    precomputed += precomputed.T
    return precomputed


def get_traj_matrix_position(precompute_array: np.ndarray, dtw: bool) -> np.ndarray:
    """Precompute the distance matrix for all trajectories of positions

    Args:
        precompute_array: Numpy array of trajectories for which to compute the distance matrix
        dtw: Boolean indicating whether matching should be performed via dynamic time warping (true) or euclidean (false)

    Returns:
        Numpy array of distances between trajectories"""
    precomputed = np.zeros((len(precompute_array), len(precompute_array)))
    for i in range(len(precompute_array)):
        if i % 50 == 0:
            logging.info(i)
        for j in range(i + 1, len(precompute_array)):
            precomputed[i][j] = fast_position_trajectory_distance(
                precompute_array[i],
                precompute_array[j],
                dtw=dtw,
            )
    precomputed += precomputed.T
    return precomputed


# @njit
# def permutations(A, k):
#     """Calculates permutation of k elements in A. Needed because numba doesnt like itertools"""
#     r = [[i for i in range(0)]]
#     for i in range(k):
#         r = [[a] + b for a in A for b in r if (a in b) == False]
#     return r


@njit
def permutations(A: list, k: int) -> list[list]:
    """Calculates permutation of k elements in A. Needed because numba doesnt like itertools

     Args:
        A: List of elements for which to compute the permutations
        k: Size of the permutations to be computed

    Returns:
        List of lists of all permutations of k elements in A"""
    # pylint: disable=singleton-comparison
    r = [[i for i in range(0)]]
    for i in range(k):
        r = [[a] + b for a in A for b in r if (a in b) == False]
    return r


@njit
def fast_area_state_distance(
    position_array_1: np.ndarray,
    position_array_2: np.ndarray,
    dist_matrix: dict,
) -> float:
    """Calculates a distance between two game states based on player positions

    Args:
        position_array_1 (numpy array): Numpy array with shape (2|1, 5, 1) with the first index indicating the team, the second the player and the third the area
        position_array_2 (numpy array): Numpy array with shape (2|1, 5, 1) with the first index indicating the team, the second the player and the third the area
        dist_matrix: Dict as distance matrix between areas

    Returns:
        A float representing the distance between these two game states
    """
    if (not np.any(position_array_1)) or (not np.any(position_array_2)):
        return sys.maxsize
    pos_distance = 0
    for team in range(position_array_1.shape[0]):
        side_distance = inf
        # Generate all possible mappings between players from array1 and array2. (Map player1 from array1 to player1 from array2 and player2's to each other or match player1's with player2's and so on)
        for mapping in permutations(
            range(position_array_1.shape[1]), position_array_2.shape[1]
        ):
            # Distance team distance for the current mapping
            cur_dist = 0
            n_skipped = 0
            # Calculate the distance between each pair of players in the current mapping
            for player2, player1 in enumerate(mapping):
                # If x, y and z coordinates or the area are all 0 then this is filler and there was no actual player playing that round
                if (
                    not position_array_1[team][player1].any()
                    or not position_array_2[team][player2].any()
                ):
                    n_skipped += 1
                    continue
                area1 = int(position_array_1[team][player1][0])
                area2 = int(position_array_2[team][player2][0])
                this_dist = min(
                    dist_matrix[area1][area2],
                    dist_matrix[area2][area1],
                )
                if this_dist == inf:
                    this_dist = sys.maxsize / 6
                    # Build up the overall distance for the current mapping of the current side
                cur_dist += this_dist
            # Only keep the smallest distance from all the mappings
            if n_skipped >= position_array_1.shape[1]:
                cur_dist = sys.maxsize
            else:
                cur_dist /= position_array_1.shape[1] - n_skipped
            side_distance = min(side_distance, cur_dist)
        # Build the total distance as the sum of the individual side's distances
        pos_distance += side_distance
    return pos_distance / position_array_1.shape[0]


@njit
def euclidean(a, b):
    """Calculates the euclidean distance between two points"""
    return np.sqrt(((a - b) ** 2).sum())


@njit
def fast_position_state_distance(
    position_array_1: np.ndarray,
    position_array_2: np.ndarray,
) -> float:
    """Calculates a distance between two game states based on player positions

    Args:
        position_array_1 (numpy array): Numpy array with shape (2|1, 5, 3) with the first index indicating the team, the second the player and the third the coordinate
        position_array_2 (numpy array): Numpy array with shape (2|1, 5, 3) with the first index indicating the team, the second the player and the third the coordinate

    Returns:
        A float representing the distance between these two game states
    """
    pos_distance = 0
    for team in range(position_array_1.shape[0]):
        side_distance = inf
        # Generate all possible mappings between players from array1 and array2. (Map player1 from array1 to player1 from array2 and player2's to each other or match player1's with player2's and so on)
        for mapping in permutations(
            range(position_array_1.shape[1]), position_array_2.shape[1]
        ):
            # Distance team distance for the current mapping
            cur_dist = 0
            # Calculate the distance between each pair of players in the current mapping
            for player2, player1 in enumerate(mapping):
                # If x, y and z coordinates or the area are all 0 then this is filler and there was no actual player playing that round
                this_dist = euclidean(
                    position_array_1[team][player1], position_array_2[team][player2]
                )
                # Build up the overall distance for the current mapping of the current side
                cur_dist += this_dist
            # Only keep the smallest distance from all the mappings
            side_distance = min(side_distance, cur_dist)
        # Build the total distance as the sum of the individual side's distances
        pos_distance += side_distance
    return pos_distance / (position_array_1.shape[0] * position_array_1.shape[1])


# @njit
def fast_token_state_distance(
    token_array_1: np.ndarray,
    token_array_2: np.ndarray,
    dist_matrix: dict,
    map_area_names: list[str],
) -> float:
    """Calculates a distance between two game states based on tokens

    Args:
        trajectory_array_1: Numpy array with shape (n_Time,len(token))
        trajectory_array_2: Numpy array with shape (n_Time,len(token))
        dist_matrix: Nested dict that contains the precomputed distance between any pair of areas
        map_area_names: List of strings of area names
        dtw: Boolean indicating whether matching should be performed via dynamic time warping (true) or euclidean (false)

    Returns:
        A float representing the distance between these two game states
    """
    # More complicated distances based on actual area locations
    side_distance = inf
    # Get the sub arrays for this team from the total array
    # Make sure array1 is the larger one
    if sum(token_array_1) < sum(token_array_2):
        token_array_1, token_array_2 = token_array_2, token_array_1
    size = sum(token_array_2)
    if size < 0.5:
        return sys.maxsize
    # Get the indices where array1 and array2 have larger values than the other.
    # Use each index as often as it is larger
    diff_array = np.subtract(token_array_1, token_array_2)
    pos_indices = []
    neg_indices = []
    for i, difference in enumerate(diff_array):
        if difference > 0:
            pos_indices.extend([i] * int(difference))
        elif difference < 0:
            neg_indices.extend([i] * int((abs(difference))))
    if len(pos_indices) < len(neg_indices):
        neg_indices, pos_indices = pos_indices, neg_indices
    # Get all possible mappings between the differences
    # Eg: diff array is [1,1,-1,-1] then pos_indices is [0,1] and neg_indices is [2,3]
    # The possible mappings are then [(0,2),(1,3)] and [(0,3),(1,2)]
    for mapping in set(
        tuple(sorted(zip(perm, neg_indices)))
        for perm in multiset_permutations(pos_indices, len(neg_indices))
    ):
        # for perm in itertools.permutations(pos_indices, len(neg_indices)):
        cur_dist = 0
        # Iterate of the mapping. Eg: [(0,2),(1,3)] and get their total distance
        # For the example this would be dist(0,2)+dist(1,3)
        for area1, area2 in mapping:
            this_dist = min(
                dist_matrix[map_area_names[area1]][map_area_names[area2]],
                dist_matrix[map_area_names[area2]][map_area_names[area1]],
            )
            if this_dist == inf:
                this_dist = sys.maxsize / 6
            cur_dist += this_dist
        side_distance = min(side_distance, cur_dist)
    return side_distance / size


def trajectory_distance_wrapper(args: tuple) -> float:
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


def transform_to_traj_dimensions(pos_array: np.ndarray) -> np.ndarray:
    """Transforms a numpy array of shape (Time,5,X) to (5,Time,1,1,X) to allow for individual trajectory distances.
    Only needed when trying to get the distance between single player trajectories within "get_shortest_distance_mapping"

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
