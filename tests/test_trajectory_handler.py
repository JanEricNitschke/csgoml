"""Tests for trajectory_handler.py"""
# pylint: disable=attribute-defined-outside-init

import os
import json
import pytest
import numpy as np
import pandas as pd
import requests
from csgoml.trajectories.trajectory_handler import TrajectoryHandler


class TestTrajectoryHandler:
    """Class to test TrajectoryHandler"""

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

    def teardown_class(self):
        """Set sorter to none, deletes all demofiles, JSON and directories"""
        files_in_directory = os.listdir()
        filtered_files = [file for file in files_in_directory if file.endswith(".json")]
        if len(filtered_files) > 0:
            for f in filtered_files:
                os.remove(f)
        self.handler = None

    @staticmethod
    def _get_jsonfile(json_link, json_name):
        print("Requesting " + json_link)
        r = requests.get(json_link, timeout=20)
        open(json_name + ".json", "wb").write(r.content)

    def test_TrajectoryHandler_init(self):
        """Tests TrajectoryHandler.__init__"""
        with pytest.raises(FileNotFoundError):
            TrajectoryHandler(
                json_path="file_does_not_exists.json",
                random_state=self.random_state,
                map_name=self.map_name,
            )
        length_set = set()
        for value in self.handler.aux.values():
            assert isinstance(value, np.ndarray)
            length_set.add(len(value))
        assert len(length_set) == 1
        assert "token" in self.handler.datasets
        assert isinstance(self.handler.datasets["token"], np.ndarray)
        shape = self.handler.datasets["token"].shape
        assert len(shape) == 3
        assert shape[1] == self.handler.time
        assert shape[0] == len(self.complete_dataframe.index)
        assert self.handler.datasets["token"].dtype == np.int64
        assert "position" in self.handler.datasets
        assert isinstance(self.handler.datasets["position"], np.ndarray)
        shape = self.handler.datasets["position"].shape
        assert len(shape) == 5
        assert shape[4] == 5
        assert shape[2] == 2
        assert shape[1] == self.handler.time
        assert shape[0] == len(self.complete_dataframe.index)
        assert self.handler.datasets["position"].dtype == np.float64

    def test_transform_to_data_frame(self):
        """Tests __transform_to_data_frame"""
        test_df = self.complete_dataframe.copy()
        assert isinstance(test_df["position_df"].iloc[0], dict)
        test_df["position_df"] = test_df["position_df"].apply(
            self.handler._TrajectoryHandler__transform_to_data_frame
        )
        assert isinstance(test_df["position_df"].iloc[0], pd.DataFrame)

    def test_transform_ticks_to_seconds(self):
        """Tests __transform_ticks_to_seconds"""
        tick = 128
        start_tick = 0
        second = self.handler._TrajectoryHandler__transform_ticks_to_seconds(
            tick, start_tick
        )
        assert isinstance(second, int)
        assert second == 1
        tick = 1280
        second = self.handler._TrajectoryHandler__transform_ticks_to_seconds(
            tick, start_tick
        )
        assert isinstance(second, int)
        assert second == 10
        start_tick = 640
        second = self.handler._TrajectoryHandler__transform_ticks_to_seconds(
            tick, start_tick
        )
        assert isinstance(second, int)
        assert second == 5

    def test_get_predictor_input(self):
        """Tests get_predictor_input"""
        coordinate_type = "position"  # "token"
        side = "CT"  # "BOTH" "T"
        side_dim = {"CT": 1, "T": 1, "BOTH": 2}
        time = 20  # 175
        consider_alive = False  # True
        (
            train_labels,
            val_labels,
            test_labels,
            train_features,
            val_features,
            test_features,
        ) = self.handler.get_predictor_input(
            coordinate_type, side, time, consider_alive
        )
        labels = [train_labels, val_labels, test_labels]
        for label in labels:
            assert isinstance(label, np.ndarray)
            assert label.dtype == np.int64
            assert sorted(set(label)) == [0, 1]
        assert (
            abs(len(train_labels) / (len(test_labels) + len(val_labels)) - (6 / 4))
            < 0.1
        )
        features = [train_features, val_features, test_features]
        for feature in features:
            assert isinstance(feature, np.ndarray)
            assert feature.dtype == np.float64
            shape = feature.shape
            assert len(shape) == 5
            assert shape[1] == time
            assert shape[2] == side_dim[side]
            assert shape[3] == 5
            assert shape[4] == 3
        assert (
            abs(
                len(train_features) / (len(test_features) + len(val_features)) - (6 / 4)
            )
            < 0.1
        )
        assert len(train_features) == len(train_labels)
        assert len(test_features) == len(test_labels)
        assert len(val_features) == len(val_labels)
        consider_alive = True
        side = "BOTH"
        (
            train_labels,
            val_labels,
            test_labels,
            train_features,
            val_features,
            test_features,
        ) = self.handler.get_predictor_input(
            coordinate_type, side, time, consider_alive
        )
        shape = train_features.shape
        assert shape[4] == 4
        assert shape[2] == side_dim[side]

        coordinate_type = "token"
        (
            train_labels,
            val_labels,
            test_labels,
            train_features,
            val_features,
            test_features,
        ) = self.handler.get_predictor_input(
            coordinate_type, side, time, consider_alive
        )
        shape = train_features.shape
        assert len(shape) == 3
        token_length = shape[2]
        side = "T"
        (
            train_labels,
            val_labels,
            test_labels,
            train_features,
            val_features,
            test_features,
        ) = self.handler.get_predictor_input(
            coordinate_type, side, time, consider_alive
        )
        shape = train_features.shape
        assert shape[2] == token_length // 2

    def test_get_clustering_input(self):
        """Tests get_clustering_input"""
        max_length = len(self.handler.datasets["token"])
        n_rounds = 100
        coordinate_type_for_distance = "position"  # "area", "token"
        side = "CT"  # "BOTH" "T"
        side_dim = {"CT": 1, "T": 1, "BOTH": 2}
        time = 20
        array_for_plotting, array_for_clustering = self.handler.get_clustering_input(
            n_rounds, coordinate_type_for_distance, side, time
        )
        assert isinstance(array_for_plotting, np.ndarray)
        assert isinstance(array_for_clustering, np.ndarray)
        plot_shape = array_for_plotting.shape
        clust_shape = array_for_clustering.shape
        assert len(plot_shape) == 5
        assert plot_shape[0] == max_length
        assert plot_shape[1] == time
        assert plot_shape[2] == side_dim[side]
        assert plot_shape[3] == 5
        assert plot_shape[4] == 3
        assert len(clust_shape) == 5
        assert clust_shape[0] == max_length
        assert clust_shape[1] == time
        assert clust_shape[2] == side_dim[side]
        assert clust_shape[3] == 5
        assert clust_shape[4] == 3
        side = "BOTH"
        coordinate_type_for_distance = "area"
        n_rounds = 50
        array_for_plotting, array_for_clustering = self.handler.get_clustering_input(
            n_rounds, coordinate_type_for_distance, side, time
        )
        plot_shape = array_for_plotting.shape
        clust_shape = array_for_clustering.shape
        assert len(plot_shape) == 5
        assert plot_shape[0] == n_rounds
        assert plot_shape[1] == time
        assert plot_shape[2] == side_dim[side]
        assert plot_shape[3] == 5
        assert plot_shape[4] == 4
        assert len(clust_shape) == 5
        assert clust_shape[0] == n_rounds
        assert clust_shape[1] == time
        assert clust_shape[2] == side_dim[side]
        assert clust_shape[3] == 5
        assert clust_shape[4] == 1
        coordinate_type_for_distance = "token"
        array_for_plotting, array_for_clustering = self.handler.get_clustering_input(
            n_rounds, coordinate_type_for_distance, side, time
        )
        plot_shape = array_for_plotting.shape
        clust_shape = array_for_clustering.shape
        assert len(plot_shape) == 5
        assert plot_shape[0] == n_rounds
        assert plot_shape[1] == time
        assert plot_shape[2] == side_dim[side]
        assert plot_shape[3] == 5
        assert plot_shape[4] == 4
        assert len(clust_shape) == 3
        assert clust_shape[0] == n_rounds
        assert clust_shape[1] == time
        token_length = clust_shape[2]
        side = "T"
        array_for_plotting, array_for_clustering = self.handler.get_clustering_input(
            n_rounds, coordinate_type_for_distance, side, time
        )
        assert array_for_clustering.shape[2] == token_length // 2
