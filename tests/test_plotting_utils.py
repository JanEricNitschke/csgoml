"""Tests for plotting_utils.py."""
# pylint: disable=attribute-defined-outside-init

import json
import os
import re
import shutil

import numpy as np
import pytest
import requests

from csgoml.trajectories.trajectory_handler import TrajectoryHandler
from csgoml.utils.plotting_utils import (
    get_shortest_distances_mapping,
    get_shortest_distances_mapping_trajectory,
    plot_map_areas,
    plot_map_tiles,
    plot_mid,
    plot_round_tokens,
    plot_rounds_different_players,
)


class TestPlottingUtils:
    """Class to test plotting_utils.py."""

    def setup_class(self):
        """Setup class by defining loading dictionary of test json files."""
        with open("tests/test_trajectory_json.json", encoding="utf-8") as f:
            self.json_data = json.load(f)
        for file in self.json_data:
            self._get_jsonfile(json_link=self.json_data[file]["url"], json_name=file)
        with open("tests/test_jsons.json", encoding="utf-8") as f:
            self.json_data = json.load(f)
            for file in self.json_data:
                self._get_jsonfile(
                    json_link=self.json_data[file]["url"], json_name=file
                )
        self.handler = TrajectoryHandler(
            json_path="ancient_trajectory_json.json",
            random_state=123,
            map_name="de_ancient",
        )
        with open("infero_round_json1.json", encoding="utf-8") as f:
            self.inferno_round = json.load(f)
        with open("mirage_round_json1.json", encoding="utf-8") as f:
            self.mirage_round = json.load(f)
        with open("rush_round_json1.json", encoding="utf-8") as f:
            self.rush_round = json.load(f)
        self.outputpath = "plotting_tests"
        if not os.path.exists(self.outputpath):
            os.makedirs(self.outputpath)

    def teardown_class(self):
        """Set sorter to none, deletes all demofiles, JSON and directories."""
        files_in_directory = os.listdir()
        if filtered_files := [
            file
            for file in files_in_directory
            if (file.endswith((".json", ".png", ".gif")))
        ]:
            for f in filtered_files:
                os.remove(f)
        shutil.rmtree(self.outputpath)
        self.handler = None

    @staticmethod
    def _get_jsonfile(json_link: str, json_name: str) -> None:
        print(f"Requesting {json_link}")
        r = requests.get(json_link, timeout=20)
        with open(f"{json_name}.json", "wb") as json_file:
            json_file.write(r.content)

    def test_get_shortest_distances_mapping(self):
        """Tests get_shortest_distances_mapping."""
        map_name = "de_inferno"
        dist_type = "euclidean"
        current_positions = np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0]])
        leaders = np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0]])
        mapping = get_shortest_distances_mapping(
            map_name, leaders, current_positions, dist_type
        )
        assert mapping == (0, 1, 2)
        current_positions = np.array([[3, 0, 0], [1, 0, 0], [2, 0, 0]])
        mapping = get_shortest_distances_mapping(
            map_name, leaders, current_positions, dist_type
        )
        assert mapping == (2, 0, 1)
        leaders = np.array([[2, 0, 0], [1, 0, 0], [3, 0, 0]])
        mapping = get_shortest_distances_mapping(
            map_name, leaders, current_positions, dist_type
        )
        assert mapping == (2, 1, 0)

        dist_type = "geodesic"
        current_positions = np.array([[1, 1, 1, 74], [2, 2, 2, 1403]])
        leaders = np.array([[1, 1, 1, 74], [2, 2, 2, 1403]])
        mapping = get_shortest_distances_mapping(
            map_name, leaders, current_positions, dist_type
        )
        assert mapping == (0, 1)
        current_positions = np.array([[2, 2, 2, 1403], [1, 1, 1, 74]])
        mapping = get_shortest_distances_mapping(
            map_name, leaders, current_positions, dist_type
        )
        assert mapping == (1, 0)

    def test_get_shortest_distances_mapping_trajectory(self):
        """Tests get_shortest_distances_mapping_trajectory."""
        map_name = "de_inferno"
        leaders = [
            np.array([[[[101, 0, 0]]], [[[201, 0, 0]]], [[[301, 0, 0]]]]),
            np.array([[[[1, 0, 0]]], [[[101, 0, 0]]], [[[201, 0, 0]]]]),
        ]
        current_positions = [
            np.array([[[[1, 0, 0]]], [[[101, 0, 0]]], [[[201, 0, 0]]]]),
            np.array([[[[101, 0, 0]]], [[[201, 0, 0]]], [[[301, 0, 0]]]]),
        ]
        dist_type = "euclidean"
        dtw = False
        mapping = get_shortest_distances_mapping_trajectory(
            map_name, leaders, current_positions, dist_type, dtw=dtw
        )
        assert mapping == (1, 0)
        leaders = [
            np.array([[[[1, 0, 0]]], [[[101, 0, 0]]], [[[201, 0, 0]]]]),
            np.array([[[[101, 0, 0]]], [[[201, 0, 0]]], [[[301, 0, 0]]]]),
        ]
        mapping = get_shortest_distances_mapping_trajectory(
            map_name, leaders, current_positions, dist_type, dtw=dtw
        )
        assert mapping == (0, 1)

    def test_plot_round_tokens(self):
        """Tests plot_round_tokens."""
        assert plot_round_tokens(
            "test_plot_round_token_inferno.gif",
            frames=self.inferno_round["gameRounds"][2]["frames"],
            map_name="de_inferno",
            dpi=100,
        )
        assert plot_round_tokens(
            "test_plot_round_token_mirage.gif",
            frames=self.mirage_round["gameRounds"][2]["frames"],
            map_name="de_mirage",
            dpi=100,
        )
        with pytest.raises(KeyError):
            plot_round_tokens(
                "test_plot_round_token_rush.gif",
                frames=self.rush_round["gameRounds"][5]["frames"],
                map_name="cs_rush",
                dpi=100,
            )

    def test_plot_map_areas(self):
        """Tests plot_map_areas."""
        assert plot_map_areas(self.outputpath, map_name="de_mirage", dpi=100) is None
        assert plot_map_areas(self.outputpath, map_name="de_inferno", dpi=100) is None
        with pytest.raises(FileNotFoundError):
            plot_map_areas(self.outputpath, map_name="de_does_not_exist", dpi=100)

    def test_plot_mid(self):
        """Tests plot_mid."""
        assert plot_mid(self.outputpath, "de_mirage", dpi=100) is None
        assert plot_mid(self.outputpath, "de_inferno", dpi=100) is None
        with pytest.raises(ValueError, match=re.escape("Map not found.")):
            plot_mid(self.outputpath, "de_does_not_exist", dpi=100)

    def test_plot_map_tiles(self):
        """Tests plot_map_tiles."""
        assert plot_map_tiles(self.outputpath, map_name="de_mirage", dpi=100) is None
        assert plot_map_tiles(self.outputpath, map_name="de_inferno", dpi=100) is None
        with pytest.raises(FileNotFoundError):
            plot_map_tiles(self.outputpath, map_name="de_does_not_exist", dpi=100)

    def test_plot_rounds_different_players(self):
        """Tests plot_rounds_different_players."""
        plotting_array, _ = self.handler.get_clustering_input(
            n_rounds=10,
            coordinate_type_for_distance="area",
            side="CT",
            time=20,
        )

        assert plot_rounds_different_players(
            filename="traj.png",
            frames_list=plotting_array,
            map_name="de_ancient",
            image=True,
            trajectory=True,
            dpi=100,
        )
        assert plot_rounds_different_players(
            filename="traj.gif",
            frames_list=plotting_array,
            map_name="de_ancient",
            image=False,
            trajectory=True,
            dpi=100,
        )

        assert plot_rounds_different_players(
            filename="pos.gif",
            frames_list=plotting_array,
            map_name="de_ancient",
            image=False,
            trajectory=False,
            dpi=100,
        )
        assert plot_rounds_different_players(
            filename="pos.png",
            frames_list=plotting_array,
            map_name="de_ancient",
            image=True,
            trajectory=False,
            dpi=100,
        )
