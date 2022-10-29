"""Tests for trajectory_clusterer.py"""
# pylint: disable=attribute-defined-outside-init

import os
import json
import shutil
import pandas as pd
import requests
from numba import typed
from csgoml.trajectories.trajectory_handler import TrajectoryHandler
from csgoml.trajectories.trajectory_clusterer import TrajectoryClusterer


class TestTrajectoryClusterer:
    """Class to test TrajectoryClusterer"""

    def setup_class(self):
        """Setup class by defining loading dictionary of test json files"""
        with open("tests/test_trajectory_json.json", encoding="utf-8") as f:
            self.json_data = json.load(f)
        for file in self.json_data:
            self._get_jsonfile(json_link=self.json_data[file]["url"], json_name=file)
        with open("ancient_trajectory_json.json", encoding="utf-8") as pre_analyzed:
            self.complete_dataframe = pd.read_json(pre_analyzed)
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
        """Set sorter to none, deletes all demofiles, JSON and directories"""
        files_in_directory = os.listdir()
        filtered_files = [file for file in files_in_directory if file.endswith(".json")]
        if len(filtered_files) > 0:
            for f in filtered_files:
                os.remove(f)
        shutil.rmtree(self.outputpath)
        self.clusterer = None
        self.handler = None

    @staticmethod
    def _get_jsonfile(json_link, json_name):
        print("Requesting " + json_link)
        r = requests.get(json_link, timeout=20)
        open(json_name + ".json", "wb").write(r.content)

    def test_get_compressed_area_dist_matrix(self):
        """Tests get_compressed_area_dist_matrix"""
        compressed_matrix = self.clusterer.get_compressed_area_dist_matrix()
        keys = list(compressed_matrix.keys())
        assert isinstance(compressed_matrix, typed.typeddict.Dict)
        assert isinstance(compressed_matrix[keys[0]], typed.typeddict.Dict)
        assert isinstance(compressed_matrix[keys[0]][keys[1]], float)
        assert compressed_matrix[keys[0]][keys[0]] == 0

    def test_get_compressed_place_dist_matrix(self):
        """Tests get_compressed_place_dist_matrix"""
        compressed_matrix = self.clusterer.get_compressed_place_dist_matrix()
        keys = list(compressed_matrix.keys())
        assert isinstance(compressed_matrix, typed.typeddict.Dict)
        assert isinstance(compressed_matrix[keys[0]], typed.typeddict.Dict)
        assert isinstance(compressed_matrix[keys[0]][keys[1]], float)
        assert compressed_matrix[keys[0]][keys[0]] == 0

    def test_get_map_area_names(self):
        """Tests get_map_area_names"""
        map_area_names = self.clusterer.get_map_area_names()
        assert len(map_area_names) == 20
        assert map_area_names == typed.List(
            [
                "",
                "Alley",
                "BackHall",
                "BombsiteA",
                "BombsiteB",
                "CTSpawn",
                "House",
                "MainHall",
                "Middle",
                "Outside",
                "Ramp",
                "Ruins",
                "SideEntrance",
                "SideHall",
                "TSideLower",
                "TSideUpper",
                "TSpawn",
                "TopofMid",
                "Tunnel",
                "Water",
            ]
        )

    def test_get_trajectory_distance_matrix(self):
        """Tests get_trajectory_distance_matrix"""
        traj_config = ("token", 10, 10, "T", False)
        coordinate_type, n_rounds, time, side, dtw = traj_config
        config_snippet = f"{self.clusterer.map_name}_{side}_{time}_{dtw}_{coordinate_type}_{n_rounds}_{self.clusterer.random_state}"
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

        traj_config = ("position", 10, 10, "T", False)
        coordinate_type, n_rounds, time, side, dtw = traj_config
        config_snippet = f"{self.clusterer.map_name}_{side}_{time}_{dtw}_{coordinate_type}_{n_rounds}_{self.clusterer.random_state}"
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

        traj_config = ("area", 10, 10, "T", False)
        coordinate_type, n_rounds, time, side, dtw = traj_config
        config_snippet = f"{self.clusterer.map_name}_{side}_{time}_{dtw}_{coordinate_type}_{n_rounds}_{self.clusterer.random_state}"
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
        """Tests run_kmed"""
        traj_config = ("token", 10, 10, "T", False)
        coordinate_type, n_rounds, time, side, dtw = traj_config
        config_snippet = f"{self.clusterer.map_name}_{side}_{time}_{dtw}_{coordinate_type}_{n_rounds}_{self.clusterer.random_state}"
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
            dtw,
        )
        kmed_dict = self.clusterer.run_kmed(3, precomputed=precomputed_matrix)
        assert len(kmed_dict) == 3
        assert dict(kmed_dict) == {0: [0, 1], 1: [2, 5, 6, 7], 2: [3, 4, 8, 9]}

    def test_run_dbscan(self):
        """Tests run_dbscan"""
        traj_config = ("token", 10, 10, "T", False)
        coordinate_type, n_rounds, time, side, dtw = traj_config
        config_snippet = f"{self.clusterer.map_name}_{side}_{time}_{dtw}_{coordinate_type}_{n_rounds}_{self.clusterer.random_state}"
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
            dtw,
        )
        dbscan_dict = self.clusterer.run_dbscan(500, 2, precomputed=precomputed_matrix)
        assert len(dbscan_dict) == 3
        assert dict(dbscan_dict) == {0: [0, 1, 3, 8, 9], 1: [2, 7], 2: [4, 5, 6]}
