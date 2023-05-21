"""Tests for nav_utils.py."""

import itertools

# pylint: disable=attribute-defined-outside-init
import math
import os
import shutil
import sys
from cmath import inf
from unittest.mock import patch

import numpy as np
from awpy.analytics.nav import (
    area_distance,
)
from numba import typed, types
from scipy.spatial import distance

from csgoml.utils.nav_utils import (
    euclidean,
    fast_position_state_distance,
    fast_position_trajectory_distance,
    fast_token_state_distance,
    fast_token_trajectory_distance,
    get_traj_matrix_area,
    get_traj_matrix_position,
    get_traj_matrix_token,
    mark_areas,
    plot_path,
    trajectory_distance,
    trajectory_distance_wrapper,
    transform_to_traj_dimensions,
)


class TestNavUtils:
    """Class to test nav_utils.py."""

    def setup_class(self):
        """Setup class by creating plot directory."""
        self.outputpath = "plotting_tests"
        self.map_name = "de_ancient"
        if not os.path.exists(self.outputpath):
            os.makedirs(self.outputpath)

    def teardown_class(self):
        """Set sorter to none, deletes all demofiles, JSON and directories."""
        shutil.rmtree(self.outputpath)

    def test_trajectory_distance(self):
        # sourcery skip: extract-duplicate-method
        """Tests trajectory_distance."""
        traj_pos_state1 = np.array(
            [[[[1, 0, 0]]], [[[101, 0, 0]]], [[[201, 0, 0]]], [[[201, 0, 0]]]]
        )
        traj_pos_state2 = np.array([[[[101, 0, 0]]], [[[201, 0, 0]]], [[[301, 0, 0]]]])
        dist = trajectory_distance(
            "de_ancient",
            traj_pos_state1,
            traj_pos_state2,
            distance_type="euclidean",
            dtw=False,
        )
        assert isinstance(dist, float)
        assert round(dist, 2) == round(400 / 4, 2)
        dist = trajectory_distance(
            "de_ancient",
            traj_pos_state1,
            traj_pos_state2,
            distance_type="euclidean",
            dtw=True,
        )
        assert isinstance(dist, float)
        assert round(dist, 2) == round(200 / 4, 2)

        token_array1 = np.array(
            [
                [
                    0.0,
                    2.0,
                    0.0,
                    2.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    2.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    1.0,
                ]
            ]
        )
        token_array2 = np.array(
            [
                [
                    0.0,
                    2.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    2.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    5.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ]
        )
        dist = trajectory_distance(
            "de_ancient", token_array1, token_array2, distance_type="graph", dtw=False
        )
        assert isinstance(dist, float)
        assert round(dist, 2) == 7.10
        dist = trajectory_distance(
            "de_ancient", token_array1, token_array2, distance_type="graph", dtw=True
        )
        assert isinstance(dist, float)
        assert round(dist, 2) == 7.10

    def test_fast_token_trajectory_distance(self):
        """Tests fast_token_trajectory_distance."""
        token_array1 = np.array(
            [
                [
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                ],
            ]
        )
        token_array2 = np.array(
            [
                [
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ],
            ]
        )
        d1_type = types.DictType(types.string, types.float64)
        dist_matrix = typed.Dict.empty(types.string, d1_type)
        map_area_names = ["One", "Two", "Three", "Four"]
        for i in range(4):
            if map_area_names[i] not in dist_matrix:
                dist_matrix[map_area_names[i]] = typed.Dict.empty(
                    key_type=types.string,
                    value_type=types.float64,
                )
            for j in range(4):
                dist_matrix[map_area_names[i]][map_area_names[j]] = float(
                    abs(i - j) ** 2
                )
        dist = fast_token_trajectory_distance(
            token_array1, token_array2, dist_matrix, map_area_names, dtw=False
        )
        assert isinstance(dist, float)
        assert round(dist, 2) == round(4 / 4, 2)
        dist = fast_token_trajectory_distance(
            token_array1, token_array2, dist_matrix, map_area_names, dtw=True
        )
        assert isinstance(dist, float)
        assert round(dist, 2) == round(2 / 4, 2)

    def test_fast_position_trajectory_distance(self):
        """Tests fast_position_trajectory_distance."""
        traj_pos_state1 = np.array(
            [[[[1, 0, 0]]], [[[101, 0, 0]]], [[[201, 0, 0]]], [[[201, 0, 0]]]]
        )
        traj_pos_state2 = np.array([[[[101, 0, 0]]], [[[201, 0, 0]]], [[[301, 0, 0]]]])
        dist = fast_position_trajectory_distance(
            traj_pos_state1,
            traj_pos_state2,
            dtw=False,
        )
        assert isinstance(dist, float)
        assert round(dist, 2) == round(400 / 4, 2)
        dist = fast_position_trajectory_distance(
            traj_pos_state1,
            traj_pos_state2,
            dtw=True,
        )
        assert isinstance(dist, float)
        assert round(dist, 2) == round(200 / 4, 2)

    def test_get_traj_matrix_area(self):
        """Tests get_traj_matrix_area."""
        to_precompute = np.array(
            [
                [[[[1]]], [[[2]]], [[[3]]]],
                [[[[2]]], [[[3]]], [[[4]]]],
                [[[[3]]], [[[4]]], [[[5]]]],
            ]
        )
        dist_matrix = np.zeros((6, 6))
        for i, j in itertools.product(range(1, 6), range(1, 6)):
            dist_matrix[i, j] = float(abs(i - j))
        target_precomputed = np.zeros((len(to_precompute), len(to_precompute)))
        target_precomputed[0][1] = target_precomputed[1][0] = 1.0
        target_precomputed[0][2] = target_precomputed[2][0] = 2.0
        target_precomputed[1][2] = target_precomputed[2][1] = 1.0
        with patch("csgoml.utils.nav_utils._prepare_trajectories") as prepare_mock:
            prepare_mock.return_value = (
                np.squeeze(to_precompute, axis=-1),
                dist_matrix,
            )
            calc_precomputed = get_traj_matrix_area(
                to_precompute, self.map_name, dtw=False
            )
        print(calc_precomputed.shape)
        print(calc_precomputed)
        assert target_precomputed.shape == calc_precomputed.shape
        assert (target_precomputed == calc_precomputed).all()
        with patch("csgoml.utils.nav_utils._prepare_trajectories") as prepare_mock:
            prepare_mock.return_value = (
                np.squeeze(to_precompute, axis=-1),
                dist_matrix,
            )
            calc_precomputed = get_traj_matrix_area(
                to_precompute, self.map_name, dtw=True
            )
        target_precomputed[0][1] = target_precomputed[1][0] = 2 / 3
        target_precomputed[0][2] = target_precomputed[2][0] = 2.0
        target_precomputed[1][2] = target_precomputed[2][1] = 2 / 3
        assert target_precomputed.shape == calc_precomputed.shape
        assert (target_precomputed == calc_precomputed).all()

    def test_get_traj_matrix_token(self):
        """Tests get_traj_matrix_token."""
        token_array1 = np.array(
            [
                [
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                ],
            ]
        )
        token_array2 = np.array(
            [
                [
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                ],
            ]
        )
        token_array3 = np.array(
            [
                [
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ],
            ]
        )
        to_precompute = np.stack([token_array1, token_array2, token_array3])
        d1_type = types.DictType(types.string, types.float64)
        dist_matrix = typed.Dict.empty(types.string, d1_type)
        map_area_names = ["One", "Two", "Three", "Four", "Five"]
        for i in range(5):
            if map_area_names[i] not in dist_matrix:
                dist_matrix[map_area_names[i]] = typed.Dict.empty(
                    key_type=types.string,
                    value_type=types.float64,
                )
            for j in range(5):
                dist_matrix[map_area_names[i]][map_area_names[j]] = float(abs(i - j))
        target_precomputed = np.zeros((len(to_precompute), len(to_precompute)))
        target_precomputed[0][1] = target_precomputed[1][0] = 1.0
        target_precomputed[0][2] = target_precomputed[2][0] = 2.0
        target_precomputed[1][2] = target_precomputed[2][1] = 1.0
        calc_precomputed = get_traj_matrix_token(
            to_precompute, dist_matrix, map_area_names, dtw=False
        )
        assert (target_precomputed == calc_precomputed).all()
        assert target_precomputed.shape == calc_precomputed.shape
        calc_precomputed = get_traj_matrix_token(
            to_precompute, dist_matrix, map_area_names, dtw=True
        )
        target_precomputed[0][1] = target_precomputed[1][0] = 2 / 3
        target_precomputed[0][2] = target_precomputed[2][0] = 2.0
        target_precomputed[1][2] = target_precomputed[2][1] = 2 / 3
        assert (target_precomputed == calc_precomputed).all()
        assert target_precomputed.shape == calc_precomputed.shape

    def test_get_traj_matrix_position(self):
        """Tests get_traj_matrix_position."""
        traj_pos_state1 = np.array([[[[1, 0, 0]]], [[[2, 0, 0]]], [[[3, 0, 0]]]])
        traj_pos_state2 = np.array([[[[2, 0, 0]]], [[[3, 0, 0]]], [[[4, 0, 0]]]])
        traj_pos_state3 = np.array([[[[3, 0, 0]]], [[[4, 0, 0]]], [[[5, 0, 0]]]])
        to_precompute = np.stack([traj_pos_state1, traj_pos_state2, traj_pos_state3])
        target_precomputed = np.zeros((len(to_precompute), len(to_precompute)))
        target_precomputed[0][1] = target_precomputed[1][0] = 1.0
        target_precomputed[0][2] = target_precomputed[2][0] = 2.0
        target_precomputed[1][2] = target_precomputed[2][1] = 1.0
        calc_precomputed = get_traj_matrix_position(to_precompute, dtw=False)
        assert (target_precomputed == calc_precomputed).all()
        assert target_precomputed.shape == calc_precomputed.shape
        calc_precomputed = get_traj_matrix_position(to_precompute, dtw=True)
        target_precomputed[0][1] = target_precomputed[1][0] = 2 / 3
        target_precomputed[0][2] = target_precomputed[2][0] = 2.0
        target_precomputed[1][2] = target_precomputed[2][1] = 2 / 3
        assert (target_precomputed == calc_precomputed).all()
        assert target_precomputed.shape == calc_precomputed.shape

    def test_euclidean(self):
        """Tests euclidean."""

    for i in range(1, 10):
        rng = np.random.default_rng()
        rng.random()
        array1 = rng.random(i)
        array2 = rng.random(i)
        assert round(euclidean(array1, array2), 2) == round(
            distance.euclidean(array1, array2), 2
        )

    def test_fast_position_state_distance(self):
        """Tests fast_position_state_distance."""
        pos_state1 = np.array([[[1, 0, 0]]])
        pos_state2 = np.array([[[2, 0, 0]]])
        dist = fast_position_state_distance(pos_state1, pos_state2)
        assert isinstance(dist, float)
        assert dist == 1.0
        pos_state1 = np.array([[[0, 0, 0]]])
        pos_state2 = np.array([[[2, 0, 0]]])
        dist = fast_position_state_distance(pos_state1, pos_state2)
        assert isinstance(dist, float)
        assert dist == 2
        pos_state1 = np.array([[[1, 0, 0], [1, 0, 0]]])
        pos_state2 = np.array([[[2, 0, 0], [2, 0, 0]]])
        dist = fast_position_state_distance(pos_state1, pos_state2)
        assert isinstance(dist, float)
        assert dist == 1.0
        pos_state1 = np.array([[[1, 0, 0], [1, 0, 0]], [[1, 0, 0], [1, 0, 0]]])
        pos_state2 = np.array([[[2, 0, 0], [2, 0, 0]], [[2, 0, 0], [2, 0, 0]]])
        dist = fast_position_state_distance(pos_state1, pos_state2)
        assert isinstance(dist, float)
        assert dist == 1.0
        pos_state1 = np.array([[[1, 1, 1]]])
        pos_state2 = np.array([[[2, 2, 2]]])
        dist = fast_position_state_distance(pos_state1, pos_state2)
        assert isinstance(dist, float)
        assert round(dist, 2) == round(math.sqrt(3), 2)

    def test_fast_token_state_distance(self):
        """Tests fast_token_state_distance."""
        d1_type = types.DictType(types.string, types.float64)
        dist_matrix = typed.Dict.empty(types.string, d1_type)
        map_area_names = ["One", "Two", "Three", "Four"]
        for i in range(4):
            if map_area_names[i] not in dist_matrix:
                dist_matrix[map_area_names[i]] = typed.Dict.empty(
                    key_type=types.string,
                    value_type=types.float64,
                )
            for j in range(4):
                dist_matrix[map_area_names[i]][map_area_names[j]] = float(abs(i - j))
        dist_matrix["One"]["Four"] = dist_matrix["Four"]["One"] = inf
        token_array1 = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        token_array2 = np.array(
            [
                0.0,
                1.0,
                0.0,
                0.0,
            ]
        )
        dist = fast_token_state_distance(
            token_array1, token_array2, dist_matrix, map_area_names
        )
        assert abs(dist / sys.maxsize - 1) < 0.0001
        token_array1 = np.array(
            [
                1.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        token_array2 = np.array(
            [
                0.0,
                1.0,
                0.0,
                0.0,
            ]
        )
        dist = fast_token_state_distance(
            token_array1, token_array2, dist_matrix, map_area_names
        )
        assert isinstance(dist, float)
        assert dist == 1.0
        token_array1 = np.array(
            [
                1.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        token_array2 = np.array(
            [
                0.0,
                0.0,
                1.0,
                0.0,
            ]
        )
        dist = fast_token_state_distance(
            token_array1, token_array2, dist_matrix, map_area_names
        )
        assert isinstance(dist, float)
        assert dist == 2.0
        token_array1 = np.array(
            [
                5.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        token_array2 = np.array(
            [
                0.0,
                0.0,
                5.0,
                0.0,
            ]
        )
        dist = fast_token_state_distance(
            token_array1, token_array2, dist_matrix, map_area_names
        )
        assert isinstance(dist, float)
        assert dist == 2.0
        token_array1 = np.array(
            [
                5.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        token_array2 = np.array(
            [
                0.0,
                0.0,
                0.0,
                5.0,
            ]
        )
        dist = fast_token_state_distance(
            token_array1, token_array2, dist_matrix, map_area_names
        )
        assert isinstance(dist, float)
        assert abs(dist / (sys.maxsize / 6) - 1) < 0.0001
        token_array1 = np.array(
            [
                1.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        token_array2 = np.array(
            [
                0.0,
                1.0,
                1.0,
                0.0,
            ]
        )
        dist = fast_token_state_distance(
            token_array1, token_array2, dist_matrix, map_area_names
        )
        assert isinstance(dist, float)
        assert dist == 1.0

    def test_trajectory_distance_wrapper(self):
        """Tests trajectory_distance_wrapper."""
        traj_pos_state1 = np.array(
            [[[[1, 0, 0]]], [[[101, 0, 0]]], [[[201, 0, 0]]], [[[201, 0, 0]]]]
        )
        traj_pos_state2 = np.array([[[[101, 0, 0]]], [[[201, 0, 0]]], [[[301, 0, 0]]]])
        dtw = False
        map_name = "de_ancient"
        distance_type = "euclidean"
        assert trajectory_distance_wrapper(
            (map_name, traj_pos_state1, traj_pos_state2, distance_type, dtw)
        ) == trajectory_distance(
            map_name,
            traj_pos_state1,
            traj_pos_state2,
            distance_type,
            dtw=dtw,
        )
        dtw = True
        assert trajectory_distance_wrapper(
            (map_name, traj_pos_state1, traj_pos_state2, distance_type, dtw)
        ) == trajectory_distance(
            map_name,
            traj_pos_state1,
            traj_pos_state2,
            distance_type,
            dtw=dtw,
        )
        token_array1 = np.array(
            [
                [
                    0.0,
                    2.0,
                    0.0,
                    2.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    2.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                    1.0,
                ]
            ]
        )
        token_array2 = np.array(
            [
                [
                    0.0,
                    2.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    2.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    5.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ]
        )
        assert trajectory_distance_wrapper(
            (map_name, token_array1, token_array2, distance_type, dtw)
        ) == trajectory_distance(
            map_name,
            token_array1,
            token_array2,
            distance_type,
            dtw=dtw,
        )
        dtw = False
        assert trajectory_distance_wrapper(
            (map_name, token_array1, token_array2, distance_type, dtw)
        ) == trajectory_distance(
            map_name,
            token_array1,
            token_array2,
            distance_type,
            dtw=dtw,
        )

    def test_transform_to_traj_dimensions(self):
        """Tests transform_to_traj_dimensions."""
        time = 10
        x = 3
        test_array = np.arange(time * 5 * x).reshape(time, 5, x)
        transformed = transform_to_traj_dimensions(test_array)
        assert transformed.shape == (5, time, 1, 1, x)
        for step, player in itertools.product(range(time), range(5)):
            assert (transformed[player, step, 0, 0] == test_array[step, player]).all()

    def test_mark_areas(self):
        """Tests mark_areas."""
        areas = {74, 5, 68, 1113}
        assert (
            mark_areas(
                output_path=self.outputpath, map_name="de_ancient", areas=areas, dpi=100
            )
            is None
        )

    def test_plot_path(self):
        """Tests plot_path."""
        graph = area_distance("de_inferno", 2831, 3030, dist_type="graph")
        geodesic = area_distance("de_inferno", 2831, 3030, dist_type="geodesic")
        assert (
            plot_path(
                output_path=self.outputpath,
                map_name="de_inferno",
                graph=graph,
                geodesic=geodesic,
                dpi=100,
            )
            is None
        )
