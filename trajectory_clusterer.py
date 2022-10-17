"""Will build up the clusterer here"""

import os
import numpy as np
import random
import logging
from awpy.data import NAV, AREA_DIST_MATRIX, PLACE_DIST_MATRIX
from numba import njit, typeof, typed, types
from nav_utils import (
    get_traj_matrix_area,
    get_traj_matrix_token,
    get_traj_matrix_position,
)


class TrajectoryClusterer:
    """Will be handling the actual clustering of trajectories"""

    def __init__(
        self,
        analysis_path,
        trajectory_handler,
        random_state=None,
        map_name="de_ancient",
    ):
        self.analysis_path = analysis_path
        self.map_name = map_name
        if random_state is None:
            self.random_state = random.randint(1, 10**8)
        else:
            self.random_state = random_state
        self.trajectory_handler = trajectory_handler

    def do_clustering(self, coordinate_type, n_rounds, time, side, dtw):
        """Does everything needed to cluster a configuration and plot the results"""
        plotting_array, clustering_array = self.trajectory_handler.get_clustering_input(
            n_rounds, coordinate_type, side, time
        )
        config_snippet = f"{self.map_name}_{side}_{time}_{dtw}_{coordinate_type}_{n_rounds}_{self.random_state}"
        precomputed_matrix_path = os.path.join(
            self.analysis_path,
            f"pre_computed_round_distances_{config_snippet}.npy",
        )
        precomputed_matrix = self.get_trajectory_distance_matrix(
            precomputed_matrix_path,
            clustering_array,
            coordinate_type,
            dtw,
        )

        # Next is plot distance histogram
        # THen plot different k-nearest-neightbors
        # Then run different clustering settings
        return True

    def get_trajectory_distance_matrix(
        self, precomputed_matrix_path, clustering_array, coordinate_type, dtw
    ):
        """Gets the precomputed distance matrix
        Args:
            precomputed_matrix_path (path): Path from which to load from or save to the precomputed distance matrix
            clustering_array (numpy array): Array containing all the trajectories for which a distance matrix should be computed
            coordinate_tpye (string): Determines which distance function to use. Needs to match shape of clustering_array. Options are 'position','area','token'.

        Returns:
            A float representing the distance between these two game states"""
        if os.path.exists(precomputed_matrix_path):
            logging.info("Loading precomputed distances from file")
            precomputed = np.load(precomputed_matrix_path)
        else:
            logging.info("Precomputing areas")
            logging.info(
                "Precomputing all round distances for %s combinations.",
                len(clustering_array) ** 2,
            )
            if coordinate_type in ["area", "token"]:
                if coordinate_type == "area":
                    old_dist_matrix = AREA_DIST_MATRIX[self.map_name]
                    d1_type = types.DictType(types.int64, types.float64)
                    dist_matrix = typed.Dict.empty(types.int64, d1_type)
                    for area1 in old_dist_matrix:
                        for area2 in old_dist_matrix[area1]:
                            if (int(area1)) not in dist_matrix:
                                dist_matrix[int(area1)] = typed.Dict.empty(
                                    key_type=types.int64,
                                    value_type=types.float64,
                                )
                            dist_matrix[int(area1)][int(area2)] = old_dist_matrix[
                                area1
                            ][area2]["geodesic"]
                    precomputed = get_traj_matrix_area(
                        precompute_array=clustering_array,
                        dist_matrix=dist_matrix,
                        dtw=dtw,
                    )
                else:  # coordinate_type == "token"
                    map_area_names = set()
                    for area_id in NAV[self.map_name]:
                        map_area_names.add(NAV[self.map_name][area_id]["areaName"])
                    map_area_names = sorted(list(map_area_names))
                    map_area_names = typed.List(map_area_names)
                    old_dist_matrix = PLACE_DIST_MATRIX[self.map_name]
                    d1_type = types.DictType(types.string, types.float64)
                    dist_matrix = typed.Dict.empty(types.string, d1_type)
                    for place1 in old_dist_matrix:
                        for place2 in old_dist_matrix[place1]:
                            if place1 not in dist_matrix:
                                dist_matrix[place1] = typed.Dict.empty(
                                    key_type=types.string,
                                    value_type=types.float64,
                                )
                            dist_matrix[place1][place2] = old_dist_matrix[place1][
                                place2
                            ]["geodesic"]["centroid"]
                    precomputed = get_traj_matrix_token(
                        precompute_array=clustering_array,
                        dist_matrix=dist_matrix,
                        map_area_names=map_area_names,
                        dtw=dtw,
                    )
            else:  # coordinate_type == "position"
                precomputed = get_traj_matrix_position(
                    precompute_array=clustering_array, dtw=dtw
                )
            np.save(
                precomputed_matrix_path,
                precomputed,
            )
            logging.info("Saved distances to file.")
        return precomputed
