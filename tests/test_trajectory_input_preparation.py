"""Tests for trajectory_input_preparation.py"""

import pytest
import pandas as pd
from csgoml.scripts.trajectory_input_preparation import (
    initialize_round_positions,
    build_intermediate_frames,
    get_postion_token,
    initialize_position_dataset_dict,
    check_size,
    frame_is_empty,
    get_player_id,
    pad_to_full_length,
    partial_step,
    append_to_round_positions,
    convert_winner_to_int,
    get_token_length,
    initialize_round,
    analyze_players,
    add_general_information,
    analyze_frames,
    analyze_rounds,
)


class TestTrajectoryInputPreparation:
    """Class to test trajectory_input_preparation.py"""

    def test_initialize_round_positions(self):
        """Tests initialize_round_positions"""
        round_positions = {
            "Tick": [],
            "token": [],
            "interpolated": [],
            "CTtoken": [],
            "CTPlayer1Alive": [],
            "CTPlayer1Name": [],
            "CTPlayer1x": [],
            "CTPlayer1y": [],
            "CTPlayer1z": [],
            "CTPlayer1Area": [],
            "CTPlayer2Alive": [],
            "CTPlayer2Name": [],
            "CTPlayer2x": [],
            "CTPlayer2y": [],
            "CTPlayer2z": [],
            "CTPlayer2Area": [],
            "CTPlayer3Alive": [],
            "CTPlayer3Name": [],
            "CTPlayer3x": [],
            "CTPlayer3y": [],
            "CTPlayer3z": [],
            "CTPlayer3Area": [],
            "CTPlayer4Alive": [],
            "CTPlayer4Name": [],
            "CTPlayer4x": [],
            "CTPlayer4y": [],
            "CTPlayer4z": [],
            "CTPlayer4Area": [],
            "CTPlayer5Alive": [],
            "CTPlayer5Name": [],
            "CTPlayer5x": [],
            "CTPlayer5y": [],
            "CTPlayer5z": [],
            "CTPlayer5Area": [],
            "Ttoken": [],
            "TPlayer1Alive": [],
            "TPlayer1Name": [],
            "TPlayer1x": [],
            "TPlayer1y": [],
            "TPlayer1z": [],
            "TPlayer1Area": [],
            "TPlayer2Alive": [],
            "TPlayer2Name": [],
            "TPlayer2x": [],
            "TPlayer2y": [],
            "TPlayer2z": [],
            "TPlayer2Area": [],
            "TPlayer3Alive": [],
            "TPlayer3Name": [],
            "TPlayer3x": [],
            "TPlayer3y": [],
            "TPlayer3z": [],
            "TPlayer3Area": [],
            "TPlayer4Alive": [],
            "TPlayer4Name": [],
            "TPlayer4x": [],
            "TPlayer4y": [],
            "TPlayer4z": [],
            "TPlayer4Area": [],
            "TPlayer5Alive": [],
            "TPlayer5Name": [],
            "TPlayer5x": [],
            "TPlayer5y": [],
            "TPlayer5z": [],
            "TPlayer5Area": [],
        }
        assert initialize_round_positions() == round_positions

    def test_build_intermediate_frames(self):
        """Tests build_intermediate_frames"""
        previous_frame = {
            "t": {
                "players": [
                    {
                        "steamID": 76561198049899734,
                        "x": 1000.0,
                        "y": 100.0,
                        "z": -100.0,
                        "isAlive": True,
                    },
                    {
                        "steamID": 76561198049899733,
                        "x": 2000.0,
                        "y": 200.0,
                        "z": -200.0,
                        "isAlive": True,
                    },
                ]
            },
            "ct": {
                "players": [
                    {
                        "steamID": 76561198049899732,
                        "x": -1000.0,
                        "y": -100.0,
                        "z": 100.0,
                        "isAlive": False,
                    },
                    {
                        "steamID": 76561198049899731,
                        "x": -2000.0,
                        "y": -200.0,
                        "z": 200.0,
                        "isAlive": True,
                    },
                ]
            },
        }
        current_frame = {
            "t": {
                "players": [
                    {
                        "steamID": 76561198049899733,
                        "x": 1000.0,
                        "y": 100.0,
                        "z": -100.0,
                        "isAlive": True,
                    },
                    {
                        "steamID": 76561198049899734,
                        "x": 2000.0,
                        "y": 200.0,
                        "z": -200.0,
                        "isAlive": True,
                    },
                ]
            },
            "ct": {
                "players": [
                    {
                        "steamID": 76561198049899732,
                        "x": -2000.0,
                        "y": -200.0,
                        "z": 200.0,
                        "isAlive": False,
                    },
                    {
                        "steamID": 76561198049899731,
                        "x": -4000.0,
                        "y": -400.0,
                        "z": 400.0,
                        "isAlive": False,
                    },
                ]
            },
        }
        second_difference = 1
        int_frames = build_intermediate_frames(
            current_frame=current_frame,
            previous_frame=previous_frame,
            second_difference=second_difference,
        )
        assert int_frames == [current_frame]
        second_difference = 2
        int_frames = build_intermediate_frames(
            current_frame=current_frame,
            previous_frame=previous_frame,
            second_difference=second_difference,
        )
        int_frame = {
            "t": {
                "players": [
                    {
                        "steamID": 76561198049899733,
                        "x": 1500.0,
                        "y": 150.0,
                        "z": -150.0,
                        "isAlive": True,
                    },
                    {
                        "steamID": 76561198049899734,
                        "x": 1500.0,
                        "y": 150.0,
                        "z": -150.0,
                        "isAlive": True,
                    },
                ]
            },
            "ct": {
                "players": [
                    {
                        "steamID": 76561198049899732,
                        "x": -1500.0,
                        "y": -150.0,
                        "z": 150.0,
                        "isAlive": False,
                    },
                    {
                        "steamID": 76561198049899731,
                        "x": -3000.0,
                        "y": -300.0,
                        "z": 300.0,
                        "isAlive": True,
                    },
                ]
            },
        }
        assert int_frames == [int_frame, current_frame]

    def test_get_postion_token(self):
        """Tests get_postion_token"""
        frame = {
            "t": {
                "players": [
                    {
                        "x": 1210.0,
                        "y": -544.0,
                        "z": 232.0,
                        "isAlive": True,
                    },
                    {
                        "x": 1210.0,
                        "y": -544.0,
                        "z": 232.0,
                        "isAlive": True,
                    },
                    {
                        "x": 1210.0,
                        "y": -544.0,
                        "z": 232.0,
                        "isAlive": False,
                    },
                ]
            },
            "ct": {
                "players": [
                    {
                        "x": 1210.0,
                        "y": -544.0,
                        "z": 232.0,
                        "isAlive": True,
                    },
                    {
                        "x": 2560.0,
                        "y": -197.0,
                        "z": 148.0,
                        "isAlive": True,
                    },
                ]
            },
        }
        map_name = "de_inferno"
        token_length = 25
        tokens = get_postion_token(
            frame=frame, map_name=map_name, token_length=token_length
        )
        assert tokens == {
            "tToken": "0200000000000000000000000",
            "ctToken": "0100000000000000100000000",
            "token": "01000000000000001000000000200000000000000000000000",
        }
        map_name = "de_nope"
        tokens = get_postion_token(
            frame=frame, map_name=map_name, token_length=token_length
        )
        assert tokens == {
            "tToken": "0000000000000000000000000",
            "ctToken": "0000000000000000000000000",
            "token": "00000000000000000000000000000000000000000000000000",
        }
        map_name = "de_inferno"
        frame["t"]["players"] = None
        tokens = get_postion_token(
            frame=frame, map_name=map_name, token_length=token_length
        )
        assert tokens == {
            "tToken": "0000000000000000000000000",
            "ctToken": "0000000000000000000000000",
            "token": "00000000000000000000000000000000000000000000000000",
        }
        frame["t"] = None
        tokens = get_postion_token(
            frame=frame, map_name=map_name, token_length=token_length
        )
        assert tokens == {
            "tToken": "0000000000000000000000000",
            "ctToken": "0000000000000000000000000",
            "token": "00000000000000000000000000000000000000000000000000",
        }

    def test_initialize_position_dataset_dict(self):
        """Tests initialize_position_dataset_dict"""
        position_dataset_dict = {
            "MatchID": [],
            "MapName": [],
            "Round": [],
            "Winner": [],
            "position_df": [],
        }
        assert initialize_position_dataset_dict() == position_dataset_dict

    def test_check_size(self):
        """Tests check_size"""
        my_dict = {"entry1": ["0"] * 10, "entry2": ["o"] * 10}
        assert check_size(my_dict) == 10
        my_dict["entry2"].append("o")
        with pytest.raises(AssertionError):
            check_size(my_dict)
        my_dict["entry1"].append("0")
        assert check_size(my_dict) == 11

    def test_frame_is_empty(self):
        """Tests frame_is_empty"""
        round_dict = {
            "roundNum": 10,
            "frames": [
                {
                    "t": {
                        "players": [
                            {
                                "x": 1210.0,
                                "y": -544.0,
                                "z": 232.0,
                                "isAlive": True,
                            },
                            {
                                "x": 1210.0,
                                "y": -544.0,
                                "z": 232.0,
                                "isAlive": True,
                            },
                            {
                                "x": 1210.0,
                                "y": -544.0,
                                "z": 232.0,
                                "isAlive": False,
                            },
                        ]
                    },
                    "ct": {
                        "players": [
                            {
                                "x": 1210.0,
                                "y": -544.0,
                                "z": 232.0,
                                "isAlive": True,
                            },
                            {
                                "x": 2560.0,
                                "y": -197.0,
                                "z": 148.0,
                                "isAlive": True,
                            },
                        ]
                    },
                }
            ],
        }
        assert not frame_is_empty(round_dict)
        round_dict["frames"] = []
        assert frame_is_empty(round_dict)
        round_dict["frames"] = None
        assert frame_is_empty(round_dict)

    def test_get_player_id(self):
        """Tests get_player_id"""
        player_dict = {"steamID": 76561198049899734, "name": "JanEric1"}
        my_id = get_player_id(player_dict)
        assert isinstance(my_id, int)
        assert my_id == 76561198049899734
        player_dict = {"steamID": 0, "name": "Bot John"}
        my_id = get_player_id(player_dict)
        assert isinstance(my_id, str)
        assert my_id == "Bot John"

    def test_pad_to_full_length(self):
        """Tests pad_to_full_length"""
        round_pos_dict = {"Tick": [], "Alive": [], "PlayerX": [], "PlayerName": []}
        padded_pos_dict = {"Tick": [], "Alive": [], "PlayerX": [], "PlayerName": []}
        pad_to_full_length(round_positions=round_pos_dict)
        assert round_pos_dict == padded_pos_dict
        round_pos_dict = {
            "Tick": [1, 2, 3],
            "Alive": [],
            "PlayerX": [],
            "PlayerName": [],
        }
        padded_pos_dict = {
            "Tick": [1, 2, 3],
            "Alive": [0, 0, 0],
            "PlayerX": [0.0, 0.0, 0.0],
            "PlayerName": ["Nobody", "Nobody", "Nobody"],
        }
        pad_to_full_length(round_positions=round_pos_dict)
        assert round_pos_dict == padded_pos_dict
        round_pos_dict = {
            "Tick": [1, 2, 3],
            "Alive": [1],
            "PlayerX": [1000],
            "PlayerName": ["JanEric1"],
        }
        padded_pos_dict = {
            "Tick": [1, 2, 3],
            "Alive": [1, 0, 0],
            "PlayerX": [1000, 1000, 1000],
            "PlayerName": ["JanEric1", "JanEric1", "JanEric1"],
        }
        pad_to_full_length(round_positions=round_pos_dict)
        assert round_pos_dict == padded_pos_dict

    def test_partial_step(self):
        """Tests partial_step"""
        current = 200
        previous = 100
        second_difference = 1
        step_value = 1
        assert partial_step(current, previous, second_difference, step_value) == current
        second_difference = 2
        assert partial_step(current, previous, second_difference, step_value) == current
        step_value = 2
        assert partial_step(current, previous, second_difference, step_value) == 150
        second_difference = 10
        step_value = 10
        assert partial_step(current, previous, second_difference, step_value) == 110
        step_value = 7
        assert partial_step(current, previous, second_difference, step_value) == 140
        step_value = 1
        assert partial_step(current, previous, second_difference, step_value) == current

    def test_append_to_round_positions(self):
        """Tests append_to_round_positions"""
        round_positions = initialize_round_positions()
        exp_round_positions = initialize_round_positions()
        side = "t"
        id_number_dict = {"t": {"5": "3"}}
        player_id = 5
        player_dict = {
            "name": "JanEric1",
            "isAlive": True,
            "x": 800,
            "y": 350,
            "z": 150,
        }  # Area 63
        second_difference = 1
        map_name = "de_inferno"
        exp_round_positions["TPlayer3Alive"].append(1)
        exp_round_positions["TPlayer3x"].append(800)
        exp_round_positions["TPlayer3y"].append(350)
        exp_round_positions["TPlayer3z"].append(150)
        exp_round_positions["TPlayer3Name"].append("JanEric1")
        exp_round_positions["TPlayer3Area"].append(63)
        append_to_round_positions(
            round_positions=round_positions,
            side=side,
            id_number_dict=id_number_dict,
            player_id=player_id,
            player=player_dict,
            second_difference=second_difference,
            map_name=map_name,
        )
        assert round_positions == exp_round_positions

    def test_convert_winner_to_int(self):
        """Tests convert_winner_to_int"""
        assert convert_winner_to_int("CT") == 1
        assert convert_winner_to_int("T") == 0
        assert convert_winner_to_int("Nope") is None

    def test_get_token_length(self):
        """Tests get_token_length"""
        assert get_token_length("de_inferno") == 25
        assert get_token_length("de_nuke") == 30
        assert get_token_length("de_ancient") == 20
        assert get_token_length("cs_rush") == 1

    def test_initialize_round(self):
        """Tests initialize_round"""
        round_dict = {"winningSide": "CT", "roundNum": 17}
        assert initialize_round(round_dict) == (
            False,
            1,
            {"t": {}, "ct": {}},
            {"t": False, "ct": False},
            initialize_round_positions(),
            [0, 0],
        )
        round_dict = {"winningSide": "T", "roundNum": 17}
        assert initialize_round(round_dict) == (
            False,
            1,
            {"t": {}, "ct": {}},
            {"t": False, "ct": False},
            initialize_round_positions(),
            [0, 0],
        )
        round_dict = {"winningSide": "Nope", "roundNum": 17}
        assert initialize_round(round_dict) == (
            False,
            1,
            {"t": {}, "ct": {}},
            {"t": False, "ct": False},
            initialize_round_positions(),
            [0, 0],
        )

    def test_analyze_players(self):
        """Tests analyze_players"""
        id_number_dict = {"t": {}, "ct": {}}
        dict_initialized = {"t": False, "ct": False}
        round_positions = initialize_round_positions()
        second_difference = 1
        map_name = "de_inferno"
        side = "ct"
        frame = {
            "ct": {
                "players": [
                    {
                        "steamID": 111,
                        "name": "JanEric1",
                        "x": 1210.0,
                        "y": -544.0,
                        "z": 232.0,
                        "isAlive": True,
                    },
                    {
                        "steamID": 222,
                        "name": "JanEric2",
                        "x": 2560.0,
                        "y": -197.0,
                        "z": 148.0,
                        "isAlive": True,
                    },
                ]
            },
        }
        analyze_players(
            frame,
            dict_initialized,
            id_number_dict,
            side,
            round_positions,
            second_difference,
            map_name,
        )
        assert len(round_positions["CTPlayer1Name"]) == 1
        assert len(round_positions["CTPlayer2Name"]) == 1
        assert id_number_dict[side] == {"111": "1", "222": "2"}
        assert id_number_dict["t"] == {}
        assert dict_initialized[side] is True
        assert dict_initialized["t"] is False
        frame = {
            "ct": {
                "players": [
                    {
                        "steamID": 112,
                        "name": "JanEric1",
                        "x": 1210.0,
                        "y": -544.0,
                        "z": 232.0,
                        "isAlive": True,
                    },
                    {
                        "steamID": 222,
                        "name": "JanEric2",
                        "x": 2560.0,
                        "y": -197.0,
                        "z": 148.0,
                        "isAlive": True,
                    },
                ]
            },
        }
        analyze_players(
            frame,
            dict_initialized,
            id_number_dict,
            side,
            round_positions,
            second_difference,
            map_name,
        )
        assert len(round_positions["CTPlayer1Name"]) == 1
        assert len(round_positions["CTPlayer2Name"]) == 2

    def test_add_general_information(self):
        """Tests add_general_information"""
        current_frame = {
            "t": {
                "players": [
                    {
                        "steamID": 76561198049899733,
                        "x": 1000.0,
                        "y": 100.0,
                        "z": -100.0,
                        "isAlive": True,
                    },
                    {
                        "steamID": 76561198049899734,
                        "x": 2000.0,
                        "y": 200.0,
                        "z": -200.0,
                        "isAlive": True,
                    },
                ]
            },
            "ct": {
                "players": [
                    {
                        "steamID": 76561198049899732,
                        "x": -2000.0,
                        "y": -200.0,
                        "z": 200.0,
                        "isAlive": False,
                    },
                    {
                        "steamID": 76561198049899731,
                        "x": -4000.0,
                        "y": -400.0,
                        "z": 400.0,
                        "isAlive": False,
                    },
                ]
            },
        }
        current_round = {}
        second_difference = 1
        last_good_frame = 1
        token_length = 25
        map_name = "de_inferno"
        round_positions = initialize_round_positions()
        ticks = [128, 256, 128]
        index = 0
        for value in round_positions.values():
            assert isinstance(value, list)
            assert len(value) == 0
        add_general_information(
            current_frame,
            current_round,
            second_difference,
            token_length,
            round_positions,
            ticks,
            index,
            map_name,
            last_good_frame,
        )
        for key, value in round_positions.items():
            assert isinstance(value, list)
            if key in ["token", "Ttoken", "CTtoken", "interpolated", "Tick"]:
                assert len(value) == 1
                if key == "interpolated":
                    assert value == [0]
                if key == "Tick":
                    assert value == [128.0]
            else:
                assert len(value) == 0

    def test_analyze_frames(self):
        """Tests analyze_frames"""
        current_round = {
            "roundNum": 10,
            "frames": [
                {
                    "tick": 128,
                    "t": {
                        "alivePlayers": 2,
                        "players": [
                            {
                                "name": "JanEric1",
                                "steamID": 76561198049899733,
                                "x": 1000.0,
                                "y": 100.0,
                                "z": -100.0,
                                "isAlive": True,
                            },
                            {
                                "name": "JanEric2",
                                "steamID": 76561198049899734,
                                "x": 2000.0,
                                "y": 200.0,
                                "z": -200.0,
                                "isAlive": True,
                            },
                        ],
                    },
                    "ct": {
                        "alivePlayers": 2,
                        "players": [
                            {
                                "name": "JanEric1",
                                "steamID": 76561198049899732,
                                "x": -2000.0,
                                "y": -200.0,
                                "z": 200.0,
                                "isAlive": False,
                            },
                            {
                                "name": "JanEric1",
                                "steamID": 76561198049899731,
                                "x": -4000.0,
                                "y": -400.0,
                                "z": 400.0,
                                "isAlive": False,
                            },
                        ],
                    },
                },
                {
                    "tick": 256,
                    "t": {
                        "alivePlayers": 2,
                        "players": [
                            {
                                "name": "JanEric1",
                                "steamID": 76561198049899733,
                                "x": 1000.0,
                                "y": 100.0,
                                "z": -100.0,
                                "isAlive": True,
                            },
                            {
                                "name": "JanEric1",
                                "steamID": 76561198049899734,
                                "x": 2000.0,
                                "y": 200.0,
                                "z": -200.0,
                                "isAlive": True,
                            },
                        ],
                    },
                    "ct": {
                        "alivePlayers": 2,
                        "players": [
                            {
                                "name": "JanEric1",
                                "steamID": 76561198049899732,
                                "x": -2000.0,
                                "y": -200.0,
                                "z": 200.0,
                                "isAlive": False,
                            },
                            {
                                "name": "JanEric1",
                                "steamID": 76561198049899731,
                                "x": -4000.0,
                                "y": -400.0,
                                "z": 400.0,
                                "isAlive": False,
                            },
                        ],
                    },
                },
            ],
        }
        map_name = "de_inferno"
        token_length = 25
        tick_rate = 128
        skip_round, round_positions = analyze_frames(
            current_round, map_name, token_length, tick_rate
        )
        assert not skip_round
        assert len(round_positions["Tick"]) == 2
        assert round_positions["Tick"] == [128.0, 256.0]
        current_round = {
            "roundNum": 10,
            "frames": [
                {
                    "tick": 128,
                    "t": {
                        "alivePlayers": 2,
                        "players": [
                            {
                                "name": "JanEric1",
                                "steamID": 76561198049899733,
                                "x": 1000.0,
                                "y": 100.0,
                                "z": -100.0,
                                "isAlive": True,
                            },
                            {
                                "name": "JanEric1",
                                "steamID": 76561198049899734,
                                "x": 2000.0,
                                "y": 200.0,
                                "z": -200.0,
                                "isAlive": True,
                            },
                        ],
                    },
                    "ct": {
                        "alivePlayers": 2,
                        "players": [
                            {
                                "name": "JanEric1",
                                "steamID": 76561198049899732,
                                "x": -2000.0,
                                "y": -200.0,
                                "z": 200.0,
                                "isAlive": False,
                            },
                            {
                                "name": "JanEric1",
                                "steamID": 76561198049899731,
                                "x": -4000.0,
                                "y": -400.0,
                                "z": 400.0,
                                "isAlive": False,
                            },
                        ],
                    },
                },
                {
                    "tick": 256,
                    "t": {
                        "alivePlayers": 6,
                        "players": [
                            {
                                "name": "JanEric1",
                                "steamID": 76561198049899733,
                                "x": 1000.0,
                                "y": 100.0,
                                "z": -100.0,
                                "isAlive": True,
                            },
                            {
                                "name": "JanEric1",
                                "steamID": 76561198049899734,
                                "x": 2000.0,
                                "y": 200.0,
                                "z": -200.0,
                                "isAlive": True,
                            },
                        ],
                    },
                    "ct": {
                        "alivePlayers": 2,
                        "players": [
                            {
                                "name": "JanEric1",
                                "steamID": 76561198049899732,
                                "x": -2000.0,
                                "y": -200.0,
                                "z": 200.0,
                                "isAlive": False,
                            },
                            {
                                "name": "JanEric1",
                                "steamID": 76561198049899731,
                                "x": -4000.0,
                                "y": -400.0,
                                "z": 400.0,
                                "isAlive": False,
                            },
                        ],
                    },
                },
            ],
        }

        skip_round, round_positions = analyze_frames(
            current_round, map_name, token_length, tick_rate
        )
        assert skip_round
        assert len(round_positions["Tick"]) == 1
        assert round_positions["Tick"] == [128.0]
        current_round = {
            "roundNum": 10,
            "frames": [
                {
                    "tick": 128,
                    "t": {
                        "alivePlayers": 2,
                        "players": None,
                    },
                    "ct": {
                        "alivePlayers": 2,
                        "players": [
                            {
                                "name": "JanEric1",
                                "steamID": 76561198049899732,
                                "x": -2000.0,
                                "y": -200.0,
                                "z": 200.0,
                                "isAlive": False,
                            },
                            {
                                "name": "JanEric1",
                                "steamID": 76561198049899731,
                                "x": -4000.0,
                                "y": -400.0,
                                "z": 400.0,
                                "isAlive": False,
                            },
                        ],
                    },
                },
                {
                    "tick": 256,
                    "t": {
                        "alivePlayers": 2,
                        "players": [
                            {
                                "name": "JanEric1",
                                "steamID": 76561198049899733,
                                "x": 1000.0,
                                "y": 100.0,
                                "z": -100.0,
                                "isAlive": True,
                            },
                            {
                                "name": "JanEric1",
                                "steamID": 76561198049899734,
                                "x": 2000.0,
                                "y": 200.0,
                                "z": -200.0,
                                "isAlive": True,
                            },
                        ],
                    },
                    "ct": {
                        "alivePlayers": 2,
                        "players": [
                            {
                                "name": "JanEric1",
                                "steamID": 76561198049899732,
                                "x": -2000.0,
                                "y": -200.0,
                                "z": 200.0,
                                "isAlive": False,
                            },
                            {
                                "name": "JanEric1",
                                "steamID": 76561198049899731,
                                "x": -4000.0,
                                "y": -400.0,
                                "z": 400.0,
                                "isAlive": False,
                            },
                        ],
                    },
                },
            ],
        }
        skip_round, round_positions = analyze_frames(
            current_round, map_name, token_length, tick_rate
        )
        assert not skip_round
        assert len(round_positions["Tick"]) == 1
        assert round_positions["Tick"] == [256.0]

    def test_analyze_rounds(self):
        """Tests analyze_rounds"""
        data = {
            "mapName": "de_inferno",
            "tickRate": 128,
            "gameRounds": [
                {
                    "winningSide": "CT",
                    "endTScore": 5,
                    "endCTScore": 5,
                    "roundNum": 10,
                    "frames": [
                        {
                            "tick": 128,
                            "t": {
                                "alivePlayers": 2,
                                "players": [
                                    {
                                        "name": "JanEric1",
                                        "steamID": 76561198049899733,
                                        "x": 1000.0,
                                        "y": 100.0,
                                        "z": -100.0,
                                        "isAlive": True,
                                    },
                                    {
                                        "name": "JanEric1",
                                        "steamID": 76561198049899734,
                                        "x": 2000.0,
                                        "y": 200.0,
                                        "z": -200.0,
                                        "isAlive": True,
                                    },
                                ],
                            },
                            "ct": {
                                "alivePlayers": 2,
                                "players": [
                                    {
                                        "name": "JanEric1",
                                        "steamID": 76561198049899732,
                                        "x": -2000.0,
                                        "y": -200.0,
                                        "z": 200.0,
                                        "isAlive": False,
                                    },
                                    {
                                        "name": "JanEric1",
                                        "steamID": 76561198049899731,
                                        "x": -4000.0,
                                        "y": -400.0,
                                        "z": 400.0,
                                        "isAlive": False,
                                    },
                                ],
                            },
                        },
                        {
                            "tick": 256,
                            "t": {
                                "alivePlayers": 6,
                                "players": [
                                    {
                                        "name": "JanEric1",
                                        "steamID": 76561198049899733,
                                        "x": 1000.0,
                                        "y": 100.0,
                                        "z": -100.0,
                                        "isAlive": True,
                                    },
                                    {
                                        "name": "JanEric1",
                                        "steamID": 76561198049899734,
                                        "x": 2000.0,
                                        "y": 200.0,
                                        "z": -200.0,
                                        "isAlive": True,
                                    },
                                ],
                            },
                            "ct": {
                                "alivePlayers": 2,
                                "players": [
                                    {
                                        "name": "JanEric1",
                                        "steamID": 76561198049899732,
                                        "x": -2000.0,
                                        "y": -200.0,
                                        "z": 200.0,
                                        "isAlive": False,
                                    },
                                    {
                                        "name": "JanEric1",
                                        "steamID": 76561198049899731,
                                        "x": -4000.0,
                                        "y": -400.0,
                                        "z": 400.0,
                                        "isAlive": False,
                                    },
                                ],
                            },
                        },
                    ],
                },
                {
                    "winningSide": "CT",
                    "endTScore": 6,
                    "endCTScore": 5,
                    "roundNum": 11,
                    "frames": [
                        {
                            "tick": 128,
                            "t": {
                                "alivePlayers": 2,
                                "players": None,
                            },
                            "ct": {
                                "alivePlayers": 2,
                                "players": [
                                    {
                                        "name": "JanEric1",
                                        "steamID": 76561198049899732,
                                        "x": -2000.0,
                                        "y": -200.0,
                                        "z": 200.0,
                                        "isAlive": False,
                                    },
                                    {
                                        "name": "JanEric1",
                                        "steamID": 76561198049899731,
                                        "x": -4000.0,
                                        "y": -400.0,
                                        "z": 400.0,
                                        "isAlive": False,
                                    },
                                ],
                            },
                        },
                        {
                            "tick": 256,
                            "t": {
                                "alivePlayers": 2,
                                "players": [
                                    {
                                        "name": "JanEric1",
                                        "steamID": 76561198049899733,
                                        "x": 1000.0,
                                        "y": 100.0,
                                        "z": -100.0,
                                        "isAlive": True,
                                    },
                                    {
                                        "name": "JanEric1",
                                        "steamID": 76561198049899734,
                                        "x": 2000.0,
                                        "y": 200.0,
                                        "z": -200.0,
                                        "isAlive": True,
                                    },
                                ],
                            },
                            "ct": {
                                "alivePlayers": 2,
                                "players": [
                                    {
                                        "name": "JanEric1",
                                        "steamID": 76561198049899732,
                                        "x": -2000.0,
                                        "y": -200.0,
                                        "z": 200.0,
                                        "isAlive": False,
                                    },
                                    {
                                        "name": "JanEric1",
                                        "steamID": 76561198049899731,
                                        "x": -4000.0,
                                        "y": -400.0,
                                        "z": 400.0,
                                        "isAlive": False,
                                    },
                                ],
                            },
                        },
                    ],
                },
                {
                    "winningSide": "T",
                    "endTScore": 6,
                    "endCTScore": 6,
                    "roundNum": 12,
                    "frames": [
                        {
                            "tick": 128,
                            "t": {
                                "alivePlayers": 2,
                                "players": None,
                            },
                            "ct": {
                                "alivePlayers": 2,
                                "players": [
                                    {
                                        "name": "JanEric1",
                                        "steamID": 76561198049899732,
                                        "x": -2000.0,
                                        "y": -200.0,
                                        "z": 200.0,
                                        "isAlive": False,
                                    },
                                    {
                                        "name": "JanEric1",
                                        "steamID": 76561198049899731,
                                        "x": -4000.0,
                                        "y": -400.0,
                                        "z": 400.0,
                                        "isAlive": False,
                                    },
                                ],
                            },
                        },
                        {
                            "tick": 256,
                            "t": {
                                "alivePlayers": 2,
                                "players": [
                                    {
                                        "name": "JanEric1",
                                        "steamID": 76561198049899733,
                                        "x": 1000.0,
                                        "y": 100.0,
                                        "z": -100.0,
                                        "isAlive": True,
                                    },
                                    {
                                        "name": "JanEric1",
                                        "steamID": 76561198049899734,
                                        "x": 2000.0,
                                        "y": 200.0,
                                        "z": -200.0,
                                        "isAlive": True,
                                    },
                                ],
                            },
                            "ct": {
                                "alivePlayers": 2,
                                "players": [
                                    {
                                        "name": "JanEric1",
                                        "steamID": 76561198049899732,
                                        "x": -2000.0,
                                        "y": -200.0,
                                        "z": 200.0,
                                        "isAlive": False,
                                    },
                                    {
                                        "name": "JanEric1",
                                        "steamID": 76561198049899731,
                                        "x": -4000.0,
                                        "y": -400.0,
                                        "z": 400.0,
                                        "isAlive": False,
                                    },
                                ],
                            },
                        },
                    ],
                },
                {
                    "winningSide": "CT",
                    "endTScore": 5,
                    "endCTScore": 5,
                    "roundNum": 10,
                    "frames": None,
                },
                {
                    "winningSide": "JanEric1",
                    "endTScore": 5,
                    "endCTScore": 5,
                    "roundNum": 10,
                    "frames": [1],
                },
            ],
        }
        position_dataset_dict = initialize_position_dataset_dict()
        match_id = "Test_id"
        analyze_rounds(data, position_dataset_dict, match_id)
        for value in position_dataset_dict.values():
            assert isinstance(value, list)
            assert len(value) == 2
        assert position_dataset_dict["MatchID"] == ["Test_id", "Test_id"]
        assert position_dataset_dict["Winner"] == [1, 0]
        assert isinstance(position_dataset_dict["position_df"][0], pd.DataFrame)
