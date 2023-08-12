"""Tests for demo_analyzer_sorter.py."""
# pylint: disable=attribute-defined-outside-init

import json
import os
import shutil
from pathlib import Path

import requests

from csgoml.preparation.demo_analyzer_sorter import DemoAnalyzerSorter


class TestDemoAnalyzerSorter:
    """Class to test DemoAnalyzerSorter."""

    def setup_class(self):
        """Setup class by defining loading dictionary of test demo files."""
        with open("tests/test_data.json", encoding="utf-8") as f:
            self.demo_data = json.load(f)
        for file in self.demo_data:
            self._get_demofile(demo_link=self.demo_data[file]["url"])
        maps_dir = os.path.join(os.getcwd(), "Maps")
        if os.path.exists(maps_dir):
            msg = (
                "This test needs to be executed in a directory where "
                "it can safely create and delete a 'Maps' subdir!"
            )
            raise AssertionError(msg)
        self.sorter = DemoAnalyzerSorter(
            dirs=[os.getcwd()],
            ids=[100, 700, 500],
            maps_dir=maps_dir,
        )

    def teardown_class(self):
        """Set sorter to none, deletes all demofiles, JSON and directories."""
        files_in_directory = os.listdir()
        if filtered_files := [
            file for file in files_in_directory if file.endswith((".dem", ".json"))
        ]:
            for f in filtered_files:
                os.remove(f)
        shutil.rmtree(self.sorter.maps_dir)
        self.sorter = None

    @staticmethod
    def _get_demofile(demo_link: str) -> None:
        print(f"Requesting {demo_link}")
        r = requests.get(demo_link, timeout=20)
        with open(demo_link.split(r"/")[-1], "wb") as demo_file:
            demo_file.write(r.content)

    def test_get_ids(self):
        """Tests get_ids."""
        assert self.sorter.get_ids(Path("510.dem")) == ("510", 510)
        assert self.sorter.get_ids(Path("015.dem")) == ("015", 15)
        assert self.sorter.get_ids(
            Path("match730_003088286727278690550_1176193512_134.dem")
        ) == ("match730_003088286727278690550_1176193512_134", 500)

    def test_get_map_name(self):
        """Tests get_map_name."""
        test_dict = {"mapName": "de_inferno"}
        assert self.sorter.get_map_name(test_dict) == "inferno"
        test_dict = {"mapName": "cs_rush"}
        assert self.sorter.get_map_name(test_dict) == "cs_rush"

    def test_move_json(self):
        """Tests move_json."""
        target_folder = os.path.join(os.getcwd(), "target_folder")
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        with open("test.json", "w", encoding="utf-8") as f:
            f.write("Create a new text file!")
        assert not os.path.exists(os.path.join(target_folder, "test.json"))
        self.sorter.move_json("test.json", os.path.join(target_folder, "test.json"))
        assert os.path.exists(os.path.join(target_folder, "test.json"))
        assert not os.path.exists("test.json")
        with open("test.json", "w", encoding="utf-8") as f:
            f.write("Create a new text file!")
        self.sorter.move_json("test.json", os.path.join(target_folder, "test.json"))
        assert os.path.exists(os.path.join(target_folder, "test.json"))
        shutil.rmtree(target_folder)

    def test_parse_demos(self):
        """Tests parse_demos."""
        assert self.sorter.n_analyzed == 0
        assert self.sorter.maps_dir == os.path.join(os.getcwd(), "Maps")
        assert not os.path.exists(self.sorter.maps_dir)
        self.sorter.parse_demos()
        assert self.sorter.n_analyzed == 7
        assert os.path.exists(self.sorter.maps_dir)
        my_dirs = os.listdir(self.sorter.maps_dir)
        assert len(my_dirs) == 7
        assert "cache" not in my_dirs
        assert "overpass" not in my_dirs
        assert "inferno" in my_dirs
        assert "nuke" in my_dirs
