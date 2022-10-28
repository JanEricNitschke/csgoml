"""
This module contains the TrajectoryClusterer.
It gets its inputs properly formatted from a TrajectoryHandler and is then able to cluster rounds based on trajectories of different types

Typical usage example:

    handler = trajectory_handler.TrajectoryHandler(
        json_path=file, random_state=random_state, map_name="de_" + options.map
    )
    clusterer = trajectory_clusterer.TrajectoryClusterer(
        analysis_path="D:\\CSGO\\Demos\\Maps\\" + options.map + "\\Analysis\\",
        trajectory_handler=handler,
        random_state=random_state,
        map_name="de_" + options.map,
    )
    traj_config = ("area", 50, 20, "T", True)
    clust_config = {
        "do_histogram": True,
        "n_bins": 20,
        "do_knn": True,
        "knn_ks": [2, 3, 4, 5, 10, 20, 50, 100],
        "plot_all_trajectories": True,
        "do_dbscan": True,
        "dbscan_eps": 700,
        "dbscan_minpt": 4,
        "do_kmed": True,
        "kmed_n_clusters": 4,
    }
    logging.info(
        clusterer.do_clustering(
            trajectory_config=traj_config, clustering_config=clust_config
        )
    )
"""

import os
from typing import Optional, Dict
import random
import logging
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn_extra.cluster import KMedoids
from numba import typed, types
from awpy.data import NAV, AREA_DIST_MATRIX, PLACE_DIST_MATRIX
from trajectory_handler import TrajectoryHandler
from nav_utils import (
    get_traj_matrix_area,
    get_traj_matrix_token,
    get_traj_matrix_position,
)
from plotting_utils import plot_rounds_different_players


class TrajectoryClusterer:
    """Clusters rounds by trajectories of different configurations by grabbing them from its TrajetoryHandler.

    Calculates a distance matrix between all trajectories and is able to call one of two clustering algorithms and visualize their results.

    Attributes:
        analysis_input (string): Path to where the results (distance matrix and plots) should be stored
        trajectory_handler (trajectory_handler.TrajectoryHandler): trajectory_handler.TrajectoryHandler from which to grab requested datasets
        random_state (int): Integer for random_states
        map_name (string): Name of the map under consideration
    """

    def __init__(
        self,
        analysis_path: str,
        trajectory_handler: TrajectoryHandler,
        random_state: Optional[int] = None,
        map_name: str = "de_ancient",
    ):
        self.analysis_path = os.path.join(analysis_path, "clustering")
        if not os.path.exists(self.analysis_path):
            os.makedirs(self.analysis_path)
        self.map_name = map_name
        if random_state is None:
            self.random_state = random.randint(1, 10**8)
        else:
            self.random_state = random_state
        self.trajectory_handler = trajectory_handler

    def do_clustering(
        self,
        trajectory_config: tuple[str, int, int, str, bool],
        clustering_config: dict,
    ) -> True:
        """Does everything needed to cluster a configuration and plot the results

        Args:
            trajectory_config (tuple): Tuple of (coordinate_type, n_rounds, time, side, dtw) where:
                coordinate_type_for_distance (string): A string indicating whether player coordinates should be used directly ("position"), the areas ("area") or the summarizing tokens ("token") instead.
                n_rounds (int): How many rounds should be in the final output. Can be necessary to not use all of them due to time constraints.
                side (string): A string indicating whether to include positions for players on the CT side ('CT'), T  side ('T') or both sides ('BOTH')
                time (integer): An integer indicating the first how many seconds should be considered
                dtw (boolean): Indicates whether trajectory distance should use dynamic time warping (True) or euclidean matching (False)
            clustering_config (dict): Dictionary containing settings for clustering. Contents:
                'do_histogram' (bool): Whether to plot a histogram of all distances
                'n_bins' (int): How many bins the histogram should have
                'do_knn' (bool): Whether to plot k distanc distribution for the given configuration
                'knn_ks' (list[int]): All the k's for which to calculate the k-distance distribution. Example [2, 3, 4, 5, 10, 20, 50, 100]
                'plot_all_trajectories' (bool): Whether To produce plot containing ALL trajectories in the current dataset
                'do_dbscan' (bool): Whether to run dbscan clustering
                'dbscan_eps' (int): The maximum distance between two samples for one to be considered as in the neighborhood of the other. This is not a maximum bound on the distances of points within a cluster.
                                    This is the most important DBSCAN parameter to choose appropriately for your data set and distance function.
                'dbscan_minpt' (int): The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself.
                Example Values:  500, 4
                'do_kmed' (bool): Whether to run k-medoids clustering
                'kmed_n_clusters' (int): The number of clusters to form as well as the number of medoids to generate. Example: 4
        Returns:
            w.i.p.
        """
        # Get config and set up paths
        coordinate_type, n_rounds, time, side, dtw = trajectory_config
        config_snippet = f"{self.map_name}_{side}_{time}_{dtw}_{coordinate_type}_{n_rounds}_{self.random_state}"
        config_path = os.path.join(self.analysis_path, config_snippet)
        if not os.path.exists(config_path):
            os.makedirs(config_path)

        # Grab dataset corresponding to config
        plotting_array, clustering_array = self.trajectory_handler.get_clustering_input(
            n_rounds, coordinate_type, side, time
        )

        # Get/calculate distance matrix
        precomputed_matrix_path = os.path.join(
            config_path,
            f"pre_computed_round_distances_{config_snippet}.npy",
        )
        precomputed_matrix = self.get_trajectory_distance_matrix(
            precomputed_matrix_path,
            clustering_array,
            coordinate_type,
            dtw,
        )
        logging.info(precomputed_matrix)

        # Create path for plots
        plot_path = os.path.join(config_path, "plots")
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        # Plot histogram of distances
        if clustering_config["do_histogram"]:
            self.plot_histogram(
                precomputed_matrix,
                plot_path,
                config_snippet,
                n_bins=clustering_config["n_bins"],
            )

        # Plot k nearest neighbors for different k's
        if clustering_config["do_knn"]:
            self.do_knn(
                clustering_config, precomputed_matrix, plot_path, config_snippet
            )

        # Plot ALL trajectories in the current dataset
        distance_variant = "euclidean" if coordinate_type == "position" else "geodesic"
        if clustering_config["plot_all_trajectories"]:
            logging.info("Plotting all trajectories")
            plot_rounds_different_players(
                os.path.join(
                    plot_path,
                    f"ALL_trajectories_{config_snippet}.png",
                ),
                plotting_array,
                map_name=self.map_name,
                map_type="simpleradar",
                fps=1,
                dist_type=distance_variant,
                image=True,
                trajectory=True,
                dtw=dtw,
            )

        # Do actual clustering
        if clustering_config["do_dbscan"]:
            # eps = 500
            # minpt = 4
            eps = clustering_config["dbscan_eps"]
            minpt = clustering_config["dbscan_minpt"]
            logging.info("eps: %s, minpt: %s", eps, minpt)
            dbscan_dict = self.run_dbscan(eps, minpt, precomputed_matrix)
            for cluster_id, rounds in dbscan_dict.items():
                dbscan_path = os.path.join(
                    plot_path, f"dbscan_{minpt}_{eps}_{cluster_id}_{config_snippet}.png"
                )
                plot_rounds_different_players(
                    dbscan_path,
                    plotting_array[rounds],
                    map_name=self.map_name,
                    map_type="simpleradar",
                    fps=1,
                    dist_type=distance_variant,
                    image=True,
                    trajectory=True,
                    dtw=dtw,
                )

        if clustering_config["do_kmed"]:
            # n_clusters = 4
            n_clusters = clustering_config["kmed_n_clusters"]
            kmed_dict = self.run_kmed(n_clusters, precomputed_matrix)
            for cluster_id, rounds in kmed_dict.items():
                kmed_path = os.path.join(
                    plot_path, f"kmed_{n_clusters}_{cluster_id}_{config_snippet}.png"
                )
                plot_rounds_different_players(
                    kmed_path,
                    plotting_array[rounds],
                    map_name=self.map_name,
                    map_type="simpleradar",
                    fps=1,
                    dist_type=distance_variant,
                    image=True,
                    trajectory=True,
                    dtw=dtw,
                )

        return True

    def run_kmed(self, n_cluster: int, precomputed: np.ndarray) -> Dict[int, list[int]]:
        """Run k-medoids on the precomputed matrix with the given parameters

        Args:
            n_clusters (int): The number of clusters to form as well as the number of medoids to generate.
            precomputed (numpy array): Distance matrix for which to perform k-medoids clutering

        Returns:
            Dictionary of cluster id's and all indixes that belong to the cluster as values"""
        logging.info("Running kmedoids clustering")
        kmed = KMedoids(n_clusters=n_cluster, metric="precomputed").fit(
            precomputed
        )  # fitting the model
        labels = kmed.labels_  # getting the labels
        logging.info(labels)
        kmed_dict = defaultdict(list)
        for round_num, cluster in enumerate(labels):
            kmed_dict[cluster].append(round_num)
        logging.info("%s", [(key, len(cluster)) for key, cluster in kmed_dict.items()])
        return kmed_dict

    def run_dbscan(
        self, eps: int, minpt: int, precomputed: np.ndarray
    ) -> Dict[int, list[int]]:
        """Run dbscan on the precomputed matrix with the given parameters

        Args:
            eps (int): The maximum distance between two samples for one to be considered as in the neighborhood of the other. This is not a maximum bound on the distances of points within a cluster.
                                    This is the most important DBSCAN parameter to choose appropriately for your data set and distance function.
            minpt (int): The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself.
            precomputed (numpy array): Distance matrix for which to perform dbscan clutering

        Returns:
            Dictionary of cluster id's and all indixes that belong to the cluster as values"""
        logging.info("Running dbscan clustering")
        dbscan = DBSCAN(eps=eps, min_samples=minpt, metric="precomputed").fit(
            precomputed
        )  # fitting the model
        logging.info(
            "core indices: %s, %s",
            len(dbscan.core_sample_indices_),
            dbscan.core_sample_indices_,
        )
        # for core_index in dbscan.core_sample_indices_:
        #     logging.info(core_index)
        #     logging.info(precomputed[core_index])
        logging.info("featrues: %s", dbscan.n_features_in_)
        labels = dbscan.labels_  # getting the labels
        logging.info(labels)
        dbscan_dict = defaultdict(list)
        for round_num, cluster in enumerate(labels):
            dbscan_dict[cluster].append(round_num)
        logging.info(
            "%s", [(key, len(cluster)) for key, cluster in dbscan_dict.items()]
        )
        return dbscan_dict

    def do_knn(
        self,
        clustering_config: dict,
        precomputed_matrix: np.ndarray,
        plot_path: str,
        config_snippet: str,
    ) -> None:
        """Runs and plots k-nearest-neighbors distance for all trajectories.

        Args:
            clustering_config (dict): Dictionary containing all settings needed for clustering.
            precomputed_matrix (numpy array): Numpy array of the distance matrix of all trajectories under consideration
            plot_path (str): Path of the directory where all plots should be saved to
            config_snippet (str): String containing all dataset configurations to include in the file name

        Returns:
            None (files are directly saved to disk)"""
        logging.info("Plotting knns")
        for k in clustering_config[
            "knn_ks"
        ]:  # [2, 3, 4, 5, 10, 20, 50, 100]:  # , 250, 500, 1000]
            if k < len(precomputed_matrix):
                self.plot_knn(
                    k,
                    precomputed_matrix,
                    os.path.join(
                        plot_path,
                        f"knn_{k}_{config_snippet}.png",
                    ),
                )

    def plot_knn(self, k: int, precomputed: np.ndarray, path: str) -> None:
        """Plot k-distance distribution with for precomputed distance matrix

        Args:
            k (int): k in k-nearest neighbors
            precomputed (numpy array): Distance matrix for which to plot the histogram
            path (path/string): Path to save the histogram to

        Returns:
            None. Histogram is directly saved to disk"""
        neighbors = NearestNeighbors(n_neighbors=k, metric="precomputed")
        neighbors_fit = neighbors.fit(precomputed)
        distances, _ = neighbors_fit.kneighbors(precomputed)
        distances = np.sort(distances, axis=0)
        distance = distances[:, k - 1]
        distance = [dist for dist in distance if dist < 1e10]
        plt.plot(distance)
        plt.savefig(path)
        plt.close()

    def plot_histogram(
        self,
        distance_matrix: np.ndarray,
        plot_path: str,
        config_snippet: str,
        n_bins: int = 20,
    ) -> None:
        """Plots a histogram of the distances in the precomputed distance matrix

        Args:
            distance_matrix (numpy array): Distance matrix for which to plot the histogram
            plot_path (str): Path of the directory where all plots should be saved to
            config_snippet (str): String containing all dataset configurations to include in the file name
            n_bins (int): Integer of the number of bins the histogram should have

        Returns:
            None. Histogram is directly saved to disk"""
        logging.info("Plotting histogram of distances")
        plt.hist(
            [dist for dist in distance_matrix.flatten() if dist < 1e10],
            density=False,
            bins=n_bins,
        )  # density=False would make counts
        plt.ylabel("Probability")
        plt.xlabel("Data")
        plt.savefig(
            os.path.join(
                plot_path,
                f"hist_distances_{config_snippet}.png",
            ),
        )
        plt.close()

    def get_trajectory_distance_matrix(
        self,
        precomputed_matrix_path: str,
        clustering_array: np.ndarray,
        coordinate_type: str,
        dtw: bool,
    ) -> np.ndarray:
        """Gets the precomputed distance matrix

        Args:
            precomputed_matrix_path (path): Path from which to load from or save to the precomputed distance matrix
            clustering_array (numpy array): Array containing all the trajectories for which a distance matrix should be computed
            coordinate_tpye (string): Determines which distance function to use. Needs to match shape of clustering_array. Options are 'position','area','token'.
            dtw (boolean): Indicates whether trajectory distance should use dynamic time warping (True) or euclidean matching (False)

        Returns:
            A numpy array of the distance matrix of all trajectories in clustering_array"""
        if os.path.exists(precomputed_matrix_path):
            logging.info("Loading precomputed distances from file")
            precomputed = np.load(precomputed_matrix_path)
        else:
            logging.info("Precomputing areas")
            logging.info(
                "Precomputing all round distances for %s combinations.",
                (len(clustering_array) ** 2) // 2,
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
