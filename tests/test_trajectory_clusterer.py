"""Tests for trajectory_clusterer.py."""

# pylint: disable=attribute-defined-outside-init

import json
import os
import shutil

import polars as pl
import requests

from csgoml.trajectories.trajectory_clusterer import TrajectoryClusterer
from csgoml.trajectories.trajectory_handler import TrajectoryHandler


class TestTrajectoryClusterer:
    """Class to test TrajectoryClusterer."""

    def setup_class(self):
        """Setup class by defining loading dictionary of test json files."""
        with open("tests/test_trajectory_json.json", encoding="utf-8") as f:
            self.json_data = json.load(f)
        for file in self.json_data:
            self._get_jsonfile(json_link=self.json_data[file]["url"], json_name=file)
        self.complete_dataframe = pl.read_ndjson("ancient_trajectory_json.json")
        self.random_state = 123
        self.map_name = "de_ancient"
        self.json_path = "ancient_trajectory_json.json"
        self.handler = TrajectoryHandler(
            json_path=self.json_path,
            random_state=self.random_state,
            map_name=self.map_name,
        )
        self.outputpath = "testing_clusterer"
        self.clusterer = TrajectoryClusterer(
            analysis_path=self.outputpath,
            trajectory_handler=self.handler,
            random_state=self.random_state,
            map_name=self.map_name,
        )

    def teardown_class(self):
        """Set sorter to none, deletes all demofiles, JSON and directories."""
        files_in_directory = os.listdir()
        if filtered_files := [
            file for file in files_in_directory if file.endswith(".json")
        ]:
            for f in filtered_files:
                os.remove(f)
        shutil.rmtree(self.outputpath)
        self.clusterer = None
        self.handler = None

    @staticmethod
    def _get_jsonfile(json_link: str, json_name: str) -> None:
        print(f"Requesting {json_link}")
        r = requests.get(json_link, timeout=20)
        with open(f"{json_name}.json", "wb") as json_file:
            json_file.write(r.content)

    def test_get_trajectory_distance_matrix(self):
        """Tests get_trajectory_distance_matrix."""
        coordinate_type, n_rounds, time, side, dtw = "token", 10, 10, "T", False
        config_snippet = (
            f"{self.clusterer.map_name}_{side}_{time}_{dtw}_"
            f"{coordinate_type}_{n_rounds}_{self.clusterer.random_state}"
        )
        config_path = os.path.join(self.clusterer.analysis_path, config_snippet)
        if not os.path.exists(config_path):
            os.makedirs(config_path)

        # Grab dataset corresponding to config
        _, clustering_array = self.handler.get_clustering_input(
            n_rounds, coordinate_type, side, time
        )

        # Get/calculate distance matrix
        precomputed_matrix_path = os.path.join(
            config_path,
            f"pre_computed_round_distances_{config_snippet}.npy",
        )
        assert not os.path.exists(precomputed_matrix_path)
        traj_matrix1 = self.clusterer.get_trajectory_distance_matrix(
            precomputed_matrix_path=precomputed_matrix_path,
            clustering_array=clustering_array,
            coordinate_type=coordinate_type,
            dtw=dtw,
        )
        assert os.path.exists(precomputed_matrix_path)
        traj_matrix2 = self.clusterer.get_trajectory_distance_matrix(
            precomputed_matrix_path=precomputed_matrix_path,
            clustering_array=clustering_array,
            coordinate_type=coordinate_type,
            dtw=dtw,
        )
        assert (traj_matrix1 == traj_matrix2).all()
        coordinate_type, n_rounds, time, side, dtw = "position", 10, 10, "T", False
        config_snippet = (
            f"{self.clusterer.map_name}_{side}_{time}_{dtw}_"
            f"{coordinate_type}_{n_rounds}_{self.clusterer.random_state}"
        )
        config_path = os.path.join(self.clusterer.analysis_path, config_snippet)
        if not os.path.exists(config_path):
            os.makedirs(config_path)

        # Grab dataset corresponding to config
        _, clustering_array = self.handler.get_clustering_input(
            n_rounds, coordinate_type, side, time
        )

        # Get/calculate distance matrix
        precomputed_matrix_path = os.path.join(
            config_path,
            f"pre_computed_round_distances_{config_snippet}.npy",
        )
        assert not os.path.exists(precomputed_matrix_path)
        traj_matrix1 = self.clusterer.get_trajectory_distance_matrix(
            precomputed_matrix_path=precomputed_matrix_path,
            clustering_array=clustering_array,
            coordinate_type=coordinate_type,
            dtw=dtw,
        )
        assert os.path.exists(precomputed_matrix_path)

        coordinate_type, n_rounds, time, side, dtw = "area", 10, 10, "T", False
        config_snippet = (
            f"{self.clusterer.map_name}_{side}_{time}_{dtw}_"
            f"{coordinate_type}_{n_rounds}_{self.clusterer.random_state}"
        )
        config_path = os.path.join(self.clusterer.analysis_path, config_snippet)
        if not os.path.exists(config_path):
            os.makedirs(config_path)

        # Grab dataset corresponding to config
        _, clustering_array = self.handler.get_clustering_input(
            n_rounds, coordinate_type, side, time
        )

        # Get/calculate distance matrix
        precomputed_matrix_path = os.path.join(
            config_path,
            f"pre_computed_round_distances_{config_snippet}.npy",
        )
        assert not os.path.exists(precomputed_matrix_path)
        traj_matrix1 = self.clusterer.get_trajectory_distance_matrix(
            precomputed_matrix_path=precomputed_matrix_path,
            clustering_array=clustering_array,
            coordinate_type=coordinate_type,
            dtw=dtw,
        )
        assert os.path.exists(precomputed_matrix_path)

    def test_run_kmed(self):
        """Tests run_kmed."""
        coordinate_type, n_rounds, time, side, dtw = "token", 10, 10, "T", False
        config_snippet = (
            f"{self.clusterer.map_name}_{side}_{time}_{dtw}_"
            f"{coordinate_type}_{n_rounds}_{self.clusterer.random_state}"
        )
        config_path = os.path.join(self.clusterer.analysis_path, config_snippet)
        if not os.path.exists(config_path):
            os.makedirs(config_path)

        # Get/calculate distance matrix
        precomputed_matrix_path = os.path.join(
            config_path,
            f"pre_computed_round_distances_{config_snippet}.npy",
        )
        assert os.path.exists(precomputed_matrix_path)
        precomputed_matrix = self.clusterer.get_trajectory_distance_matrix(
            precomputed_matrix_path,
            [],
            coordinate_type,
            dtw=dtw,
        )
        kmed_dict = self.clusterer.run_kmed(3, precomputed=precomputed_matrix)
        assert len(kmed_dict) == 3
        assert dict(kmed_dict) == {0: [0, 1], 1: [2, 5, 6, 7], 2: [3, 4, 8, 9]}

    def test_run_dbscan(self):
        """Tests run_dbscan."""
        coordinate_type, n_rounds, time, side, dtw = "token", 10, 10, "T", False
        config_snippet = (
            f"{self.clusterer.map_name}_{side}_{time}_{dtw}_"
            f"{coordinate_type}_{n_rounds}_{self.clusterer.random_state}"
        )
        config_path = os.path.join(self.clusterer.analysis_path, config_snippet)
        if not os.path.exists(config_path):
            os.makedirs(config_path)

        # Get/calculate distance matrix
        precomputed_matrix_path = os.path.join(
            config_path,
            f"pre_computed_round_distances_{config_snippet}.npy",
        )
        assert os.path.exists(precomputed_matrix_path)
        precomputed_matrix = self.clusterer.get_trajectory_distance_matrix(
            precomputed_matrix_path,
            [],
            coordinate_type,
            dtw=dtw,
        )
        dbscan_dict = self.clusterer.run_dbscan(500, 2, precomputed=precomputed_matrix)
        assert len(dbscan_dict) == 2
        assert dict(dbscan_dict) == {0: [0, 1, 3, 4, 5, 6, 8, 9], 1: [2, 7]}
