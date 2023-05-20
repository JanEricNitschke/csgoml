r"""This module contains the TrajectoryClusterer.

It gets its inputs properly formatted from a TrajectoryHandler and
is then able to cluster rounds based on trajectories of different types.

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
    traj_config = {
        "coordinate_type_for_distance": "area",
        "n_rounds": 1000,
        "time": 10,
        "side": "T",
        "dtw": False,
    }
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

import logging
import os
import random
import sys
from collections import defaultdict
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from awpy.data import AREA_DIST_MATRIX, NAV, PLACE_DIST_MATRIX
from awpy.types import DistanceType
from numba import typed, types
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn_extra.cluster import KMedoids

from csgoml.trajectories.trajectory_handler import TrajectoryHandler
from csgoml.types import (
    ClusteringConfig,
    TrajectoryConfig,
    UserClusteringConfig,
    UserTrajectoryConfig,
)
from csgoml.utils.nav_utils import (
    get_traj_matrix_area,
    get_traj_matrix_position,
    get_traj_matrix_token,
)
from csgoml.utils.plotting_utils import plot_rounds_different_players


class TrajectoryClusterer:
    """Clusters rounds by trajectories of different configurations.

    Grabs configurations from its TrajetoryHandler.
    Calculates a distance matrix between all trajectories and
    is able to call one of two clustering algorithms and visualize their results.

    Attributes:
        analysis_path (str): Path to where the results
            (distance matrix and plots) should be stored
        trajectory_handler (trajectory_handler.TrajectoryHandler):
            trajectory_handler.TrajectoryHandler from which to grab requested datasets
        random_state (int): Integer for random_states
        map_name (str): Name of the map under consideration
    """

    def __init__(
        self,
        analysis_path: str,
        trajectory_handler: TrajectoryHandler,
        random_state: int | None = None,
        map_name: str = "de_ancient",
    ) -> None:
        """Initialize an instance.

        Args:
            analysis_path (str): Path to where the results
                (distance matrix and plots) should be stored
            trajectory_handler (trajectory_handler.TrajectoryHandler):
                TrajectoryHandler from which to grab requested datasets
            random_state (int): Integer for random_states
            map_name (str): Name of the map under consideration. Defaults to "ancient"
        """
        self.analysis_path = os.path.join(analysis_path, "clustering")
        if not os.path.exists(self.analysis_path):
            os.makedirs(self.analysis_path)
        self.map_name = map_name
        if random_state is None:
            # Not doing cryptography
            self.random_state = random.randint(1, 10**8)  # noqa: S311
        else:
            self.random_state = random_state
        self.trajectory_handler = trajectory_handler

    def _get_default_configs(
        self,
        trajectory_config: UserTrajectoryConfig | None,
        clustering_config: UserClusteringConfig | None,
    ) -> tuple[TrajectoryConfig, ClusteringConfig]:
        """Get default values for trajectory and clustering configuration.

        Returns:
            tuple[TrajectoryConfig, ClusteringConfig]: _description_
        """
        if trajectory_config is None:
            trajectory_config = {}
        if clustering_config is None:
            clustering_config = {}

        default_trajectory_config: TrajectoryConfig = {
            "coordinate_type_for_distance": "area",
            "n_rounds": 1000,
            "time": 10,
            "side": "T",
            "dtw": False,
        }
        for key in trajectory_config:
            default_trajectory_config[key] = trajectory_config[key]

        default_clustering_config: ClusteringConfig = {
            "do_histogram": False,
            "n_bins": 50,
            "do_knn": False,
            "knn_ks": [2, 3, 4, 5, 10, 20, 50, 100, 200, 400, 500, 600],
            "plot_all_trajectories": False,
            "do_dbscan": False,
            "dbscan_eps": 500,
            "dbscan_minpt": 2,
            "do_kmed": False,
            "kmed_n_clusters": 3,
        }
        for key in clustering_config:
            default_clustering_config[key] = clustering_config[key]

        return default_trajectory_config, default_clustering_config

    def do_clustering(
        self,
        traj_config: UserTrajectoryConfig | None,
        clust_config: UserClusteringConfig | None,
    ) -> Literal[True]:
        """Does everything needed to cluster a configuration and plot the results.

        Args:
            traj_config (dict): Dict containing settings for trajectories:
                coordinate_type_for_distance (string): String indicating whether
                    player coordinates should be used directly ("position"),
                    the areas ("area") or the summarizing tokens ("token") instead.
                n_rounds (int): How many rounds should be in the final output.
                    Can be necessary to not use all of them due to time constraints.
                time (integer): Integer indicating the first how many
                    seconds should be considered
                side (string): String indicating whether to include positions
                    for players on the CT side ('CT'),
                    T  side ('T') or both sides ('BOTH')
                dtw (boolean): Indicates whether trajectory distance should use
                    dynamic time warping (True) or euclidean matching (False)
            clust_config (dict): Dict containing settings for clustering:
                'do_histogram' (bool): Whether to plot a histogram of all distances
                'n_bins' (int): How many bins the histogram should have
                'do_knn' (bool): Whether to plot k distance distribution
                    for the given configuration
                'knn_ks' (list[int]): All the k's for which to calculate the
                    k-distance distribution. Example [2, 3, 4, 5, 10, 20, 50, 100]
                'plot_all_trajectories' (bool): Whether To produce plot containing
                    ALL trajectories in the current dataset
                'do_dbscan' (bool): Whether to run dbscan clustering
                'dbscan_eps' (int): The maximum distance between two samples for one to
                    be considered as in the neighborhood of the other.
                    This is not a maximum bound on the distances of points
                    within a cluster.
                    This is the most important DBSCAN parameter to
                    choose appropriately for your data set and distance function.
                'dbscan_minpt' (int): The number of samples (or total weight) in a
                    neighborhood for a point to be considered as a core point.
                    This includes the point itself.
                Example Values:  500, 4
                'do_kmed' (bool): Whether to run k-medoids clustering
                'kmed_n_clusters' (int): The number of clusters to form as
                    well as the number of medoids to generate. Example: 4
        Returns:
            True
        """
        # Get config and set up paths
        trajectory_config, clustering_config = self._get_default_configs(
            traj_config, clust_config
        )

        config_snippet = (
            f"{self.map_name}_{trajectory_config['side']}_"
            f"{trajectory_config['time']}_{trajectory_config['dtw']}_"
            f"{trajectory_config['coordinate_type_for_distance']}_"
            f"{trajectory_config['n_rounds']}_{self.random_state}"
        )
        config_path = os.path.join(self.analysis_path, config_snippet)
        if not os.path.exists(config_path):
            os.makedirs(config_path)

        # Grab dataset corresponding to config
        plotting_array, clustering_array = self.trajectory_handler.get_clustering_input(
            trajectory_config["n_rounds"],
            trajectory_config["coordinate_type_for_distance"],
            trajectory_config["side"],
            trajectory_config["time"],
        )

        # Get/calculate distance matrix
        precomputed_matrix_path = os.path.join(
            config_path,
            f"pre_computed_round_distances_{config_snippet}.npy",
        )
        precomputed_matrix = self.get_trajectory_distance_matrix(
            precomputed_matrix_path,
            clustering_array,
            trajectory_config["coordinate_type_for_distance"],
            dtw=trajectory_config["dtw"],
        )
        logging.info(precomputed_matrix.shape)
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
        distance_variant = (
            "euclidean"
            if trajectory_config["coordinate_type_for_distance"] == "position"
            else "geodesic"
        )
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
                dtw=trajectory_config["dtw"],
            )

        self._run_clustering(
            clustering_config,
            precomputed_matrix,
            config_snippet,
            plotting_array,
            distance_variant,
            plot_path,
            dtw=trajectory_config["dtw"],
        )
        # Do actual clustering

        return True

    def _run_clustering(
        self,
        clustering_config: ClusteringConfig,
        precomputed_matrix: np.ndarray,
        config_snippet: str,
        plotting_array: np.ndarray,
        distance_variant: DistanceType,
        plot_path: str,
        *,
        dtw: bool,
    ) -> None:
        """Run the actual clustering and plotting depending on the config."""
        if clustering_config["do_dbscan"]:
            self._do_dbscan(
                clustering_config,
                precomputed_matrix,
                config_snippet,
                plotting_array,
                distance_variant,
                plot_path,
                dtw=dtw,
            )

        if clustering_config["do_kmed"]:
            self._do_kmed(
                clustering_config,
                precomputed_matrix,
                config_snippet,
                plotting_array,
                distance_variant,
                plot_path,
                dtw=dtw,
            )

    def _do_dbscan(
        self,
        clustering_config: ClusteringConfig,
        precomputed_matrix: np.ndarray,
        config_snippet: str,
        plotting_array: np.ndarray,
        distance_variant: DistanceType,
        plot_path: str,
        *,
        dtw: bool,
    ) -> None:
        """Run and plot dbscan."""
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

    def _do_kmed(
        self,
        clustering_config: ClusteringConfig,
        precomputed_matrix: np.ndarray,
        config_snippet: str,
        plotting_array: np.ndarray,
        distance_variant: DistanceType,
        plot_path: str,
        *,
        dtw: bool,
    ) -> None:
        """Run and plot kmed."""
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

    def run_kmed(self, n_cluster: int, precomputed: np.ndarray) -> dict[int, list[int]]:
        """Run k-medoids on the precomputed matrix with the given parameters.

        Args:
            n_cluster (int): The number of clusters to form as well
                as the number of medoids to generate.
            precomputed (np.ndarray): Distance matrix for which to
                perform k-medoids clustering

        Returns:
            Dictionary of cluster id's and all indices
            that belong to the cluster as values
        """
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
    ) -> dict[int, list[int]]:
        """Run dbscan on the precomputed matrix with the given parameters.

        Args:
            eps (int): Mximum distance between two samples for one to be considered
                as in the neighborhood of the other.
                This is not a maximum bound on the distances of points within a cluster.
                This is the most important DBSCAN parameter to choose
                appropriately for your data set and distance function.
            minpt (int): Number of samples (or total weight) in a neighborhood for a
                point to be considered as a core point. This includes the point itself.
            precomputed (np.ndarray): Distance matrix for which to
                perform dbscan clustering

        Returns:
            Dictionary of cluster id's and all indices
            that belong to the cluster as values
        """
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
        clustering_config: ClusteringConfig,
        precomputed_matrix: np.ndarray,
        plot_path: str,
        config_snippet: str,
    ) -> None:
        """Runs and plots k-nearest-neighbors distance for all trajectories.

        Args:
            clustering_config (dict): Dict containing all settings
                needed for clustering.
            precomputed_matrix (np.ndarray): Numpy array of the distance matrix
                of all trajectories under consideration
            plot_path (str): Path of the directory where all plots should be saved to
            config_snippet (str): String containing all dataset
                configurations to include in the file name

        Returns:
            None (files are directly saved to disk)
        """
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
        """Plot k-distance distribution with for precomputed distance matrix.

        Args:
            k (int): k in k-nearest neighbors
            precomputed (numpy array): Distance matrix for which to plot the histogram
            path (path/string): Path to save the histogram to

        Returns:
            None. Histogram is directly saved to disk
        """
        max_value = 1e10
        neighbors = NearestNeighbors(n_neighbors=k, metric="precomputed")
        neighbors_fit = neighbors.fit(precomputed)
        distances, _ = neighbors_fit.kneighbors(precomputed)
        distances = np.sort(distances, axis=0)
        distance = distances[:, k - 1]
        distance = [dist for dist in distance if dist < max_value]
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
        """Plots a histogram of the distances in the precomputed distance matrix.

        Args:
            distance_matrix (np.ndarray): Distance matrix for
                which to plot the histogram
            plot_path (str): Path of the directory where all plots should be saved to
            config_snippet (str): String containing all dataset configurations
                to include in the file name
            n_bins (int): Integer of the number of bins the histogram should have

        Returns:
            None. Histogram is directly saved to disk
        """
        max_value = 1e10
        logging.info("Plotting histogram of distances")
        plt.hist(
            [dist for dist in distance_matrix.flatten() if dist < max_value],
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

    def get_compressed_area_dist_matrix(
        self,
    ) -> tuple[npt.NDArray, dict[types.int64, types.float64]]:
        """Generates a compressed area distance matrix.

        Args:
            None
        Returns:
            typed dict of compressed matrix
        """
        old_dist_matrix = AREA_DIST_MATRIX[self.map_name]
        dist_matrix = np.zeros((len(old_dist_matrix), len(old_dist_matrix)))
        matching: dict[types.int64, types.float64] = {}
        for idx1, area1 in enumerate(sorted(old_dist_matrix)):
            matching[int(area1)] = idx1
            for idx2, area2 in enumerate(sorted(old_dist_matrix[area1])):
                dist_matrix[idx1, idx2] = min(
                    old_dist_matrix[area1][area2]["geodesic"],
                    old_dist_matrix[area2][area1]["geodesic"],
                    sys.maxsize / 6,
                )
        return dist_matrix, matching

    def get_compressed_place_dist_matrix(
        self,
    ) -> dict[types.string, dict[types.string, types.float64]]:
        """Generates a compressed place distance matrix.

        Args:
            None
        Returns:
            typed dict of compressed matrix
        """
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
                dist_matrix[place1][place2] = old_dist_matrix[place1][place2][
                    "geodesic"
                ]["centroid"]
        return dist_matrix

    def get_map_area_names(self) -> typed.List:
        """Generates list of all named place on a map in sorted order.

        Args:
            None

        Returns:
            sorted list of named places
        """
        map_area_names = {
            NAV[self.map_name][area_id]["areaName"] for area_id in NAV[self.map_name]
        }
        map_area_names = sorted(map_area_names)
        return typed.List(map_area_names)

    def get_trajectory_distance_matrix(
        self,
        precomputed_matrix_path: str,
        clustering_array: np.ndarray,
        coordinate_type: str,
        *,
        dtw: bool,
    ) -> np.ndarray:
        """Gets the precomputed distance matrix.

        Args:
            precomputed_matrix_path (path): Path from which to load from or save
                to the precomputed distance matrix
            clustering_array (np.ndarray): Array containing all the trajectories
                for which a distance matrix should be computed
            coordinate_type (string): Determines which distance function to use.
                Needs to match shape of clustering_array.
                Options are 'position','area','token'.
            dtw (boolean): Indicates whether trajectory distance should
                use dynamic time warping (True) or euclidean matching (False)

        Returns:
            A numpy array of the distance matrix of all trajectories in clustering_array
        """
        if os.path.exists(precomputed_matrix_path):
            logging.info(
                "Loading precomputed distances from file %s", precomputed_matrix_path
            )
            precomputed = np.load(precomputed_matrix_path)
        else:
            logging.info("Precomputing areas")
            logging.info(
                "Precomputing all round distances for %s combinations.",
                (len(clustering_array) ** 2) // 2,
            )
            if coordinate_type == "area":
                dist_matrix, matching = self.get_compressed_area_dist_matrix()
                logging.info(clustering_array.shape)

                def get_matching(x: str) -> float:
                    return matching[141] if int(x) not in matching else matching[int(x)]

                clustering_array = np.vectorize(get_matching)(clustering_array)
                logging.info(clustering_array.shape)
                precomputed = get_traj_matrix_area(
                    precompute_array=clustering_array,
                    dist_matrix=dist_matrix,
                    dtw=dtw,
                )
            elif coordinate_type == "token":
                map_area_names = self.get_map_area_names()
                dist_matrix = self.get_compressed_place_dist_matrix()
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
