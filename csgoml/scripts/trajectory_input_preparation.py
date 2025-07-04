#!/usr/bin/env python

r"""Runs over each demo by map and produces a dataframe tracking player positions.

For each map folder in the provided directory this script produces
a dataframe tracking player movement in each round.
The produced dataframe has one row per round and for each
round columns for the matchid, mapname, roundnumber,
winning side and a dataframe containing the trajectory data.
The trajectory data contains for each parsed frame (every 128th recorded tick
Ideally 1 frame per second but can be less depending on server configuration)
each players x,y,z coordinates, name and alive status.
Additionally three position tokens are generated.
One encapsulating the positions of each t side player,
one of each ct side player and a combined one.
The dataframe is finally exported to a json file.


Example::
        python tensorflow_input_preparation --dir \\
            'E:\\PhD\\MachineLearning\\CSGOData\\ParsedDemos'

"""


import argparse
import copy
import json
import logging
import os
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Literal, overload

import polars as pl
from awpy.analytics.nav import find_closest_area, generate_position_token
from awpy.data import NAV
from awpy.types import (
    Game,
    GameFrame,
    GameRound,
    PlayerInfo,
    Token,
    int_to_string_n_players,
    is_valid_side,
    proper_player_number,
    upper_side,
)

from csgoml.helpers import setup_logging
from csgoml.types import DictInitialized, IDNumberDict, RoundPositions

N_PLAYERS = 5


def initialize_round_positions(
    match_id: str, map_name: str, round_number: int, winner: Literal[0, 1]
) -> RoundPositions:
    """Initializes dictionary of lists for one rounds positions.

    The positions dictionary contains the following keys:
    Tick, token, interpolated, CTtoken, Ttoken and for each player on each side
    side+"Player"+number+feature where side is in [CT,T],
    number in range(1,6) and feature in ["Alive", "Name", "x", "y", "z"]

    Args:
        match_id (str): Name of the match containing the round
        map_name (str): Map that the round was played on
        round_number (int): Number of the round in the game
        winner (Literal[0, 1]): Winner of the round (1 for CT 0 for T)

    Returns:
        Partially initialized round_positions dictionary
    """
    return {
        "MatchID": match_id,
        "MapName": map_name,
        "Round": round_number,
        "Winner": winner,
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


def build_intermediate_frames(
    current_frame: GameFrame, previous_frame: GameFrame, second_difference: int
) -> list[GameFrame]:
    """Builds intermediate frames if fewer than 1 were recorded per second.

    Linearly interpolates the players x,y,z positions from one frame to the next
    while keeping everything else as it is in the first frame.
    This has to be done to be able to generate intermediate position tokens.

    Args:
        current_frame (GameFrame): Dictionary the contains all information about player
            positions at the most recent timestep
        previous_frame (GameFrame): Dictionary the contains all information about player
            positions one timestep previously
        second_difference (int): Difference in the number of seconds between
            the previous and current frames

    Returns:
        A list of intermediate frames from previous_frame to including current_frame
    """
    intermdiate_frames: list[GameFrame] = []
    for i in range(second_difference, 0, -1):
        this_frame = copy.deepcopy(current_frame)
        for side in ["t", "ct"]:
            if current_frame[side]["players"] is None:
                logging.debug(
                    "Side['players'] is none. "
                    "Skipping this side for frame interpolation!",
                )
                continue
            for index, player in enumerate(current_frame[side]["players"]):
                for prev_player in previous_frame[side]["players"]:
                    if prev_player["steamID"] == player["steamID"]:
                        this_frame[side]["players"][index]["isAlive"] = (
                            player["isAlive"] if i == 1 else prev_player["isAlive"]
                        )
                        this_frame[side]["players"][index]["x"] = partial_step(
                            player["x"], prev_player["x"], second_difference, i
                        )
                        this_frame[side]["players"][index]["y"] = partial_step(
                            player["y"], prev_player["y"], second_difference, i
                        )
                        this_frame[side]["players"][index]["z"] = partial_step(
                            player["z"], prev_player["z"], second_difference, i
                        )
        intermdiate_frames.append(this_frame)
    return intermdiate_frames


def get_position_token(frame: GameFrame, map_name: str, token_length: int) -> Token:
    """Generate a dictionary of position tokens from frame dictionary and map_name.

    The position token is a string of integers representing the
    number of players in a given unique named area.
    If token generation fails because of empty players or
    unsupported map then strings of 0 are returned instead.

    Args:
        frame (GameFrame): Dictionary containing all information about both teams
            status and each players position and status
        map_name (str): A string of the maps name
        token_length (int): Integer of the length of
            one sides position token on the played map

    Returns:
        Dict of three position tokens.
            One for each side and additionally a combined one.
    """
    try:
        tokens: Token = generate_position_token(map_name, frame)
    except TypeError:
        tokens = {
            "tToken": token_length * "0",
            "ctToken": token_length * "0",
            "token": 2 * token_length * "0",
        }
        logging.debug(
            "Got TypeError when trying to generate position token. "
            "This is due to one sides 'player' entry being none."
        )
    except KeyError:
        tokens = {
            "tToken": token_length * "0",
            "ctToken": token_length * "0",
            "token": 2 * token_length * "0",
        }
        logging.debug(
            "Got KeyError when trying to generate position token. "
            "This is due to the map not being supported."
        )
    except ValueError:
        tokens = {
            "tToken": token_length * "0",
            "ctToken": token_length * "0",
            "token": 2 * token_length * "0",
        }
        logging.debug(
            "Got ValueError when trying to generate position token. "
            "This is due to the map not being supported."
        )
    return tokens


# Tighten this to Mapping[str, list] if that ever properly type checks
# with @final TypedDicts. -> Make them final if that is the case.
def check_size(dictionary: Mapping[str, Any]) -> int:
    """Checks that the size of each list behind every dictionary key is the same.

    The input dictionary is expected to have a
    list corresponding to each key and that each list has the same size.

    Args:
        dictionary (dict[str, list]): Dict of lists with the
            expectation that each list has the same size

    Returns:
        length (int): Integer corresponding to the length of every list in the dict.

    Raises:
        AssertionError: If not all lists have the same size
    """
    length = 0
    same_size = True
    for key, value in dictionary.items():
        if not isinstance(value, list):
            continue
        logging.debug("Length of key %s is %s", key, len(value))
        if length == 0:
            length = len(value)
        else:
            same_size = length == len(value)
            if not same_size:
                logging.error(
                    "Key %s has size %s while %s is expected!",
                    key,
                    len(value),
                    length,
                )
                logging.error(dictionary)
                msg = (
                    "Not all elements in dict have the same size. "
                    "Something has gone wrong."
                )
                raise AssertionError(msg)
    return length


def frame_is_empty(current_round: GameRound) -> bool:
    """Checks whether a round dictionary contains None or an empty list frames.

    None or empty frames will raise exceptions when
    trying to extract player information out of them.
    This method checks if a frame is empty or None and
    logs an error message if either is the case.

    Args:
        current_round (GameRound): Dict containing info about a single CS:GO round.

    Returns:
        Whether the the frames list is None or has a length of zero.
    """
    if current_round["frames"] is None:
        logging.error(current_round)
        logging.error("Found none frames in round %s!", current_round["roundNum"])
        return True
    if len(current_round["frames"]) == 0:
        logging.error("Found empty frames in round %s!", current_round["roundNum"])
        return True
    return False


def get_player_id(player: PlayerInfo) -> int | str:
    """Extracts a players steamID from their player dictionary in a given frame.

    Each player has a unique steamID by which they can be identified.
    Bots, however, do not naturally have a steamID and are instead all assigned 0.
    To avoid overlap bots are instead identified by
    their name which is unique in any given game.

    Args:
        player (dict): Dictionary about a players position and status in a given frame

    Returns:
        Integer corresponding to their steamID if they are a player or
        a string corresponding to the bots name if not.
    """
    return player["name"] if player["steamID"] == 0 else player["steamID"]


def pad_to_full_length(round_positions: RoundPositions) -> None:
    """Pads each entry in a given round_positions dictionary to the full length.

    For every player their name, status and position
    should be stored for every timestep in the round.
    If a player leaves mid round it can happen that their entries are incomplete.
    To avoid problems further down the line their entries
    are padded the the full length of the round.
    If they left mid round their most recent information is used.
    If they left before the round started dummy data is produced instead.
    Afterwards a check is performed to assert that
    every entry in the dictionary has the same length.

    Args:
        round_positions (RoundPositions): Dictionary containing information
            about all players for each timestep

    Returns:
        None. Dictionary is padded in place.

    """
    if len(round_positions["Tick"]) == 0:
        return
    for key in round_positions:
        if "Alive" in key:
            # If the Alive list is completely empty fill it with a dead player
            # If the player left mid round he is considered dead
            # for the time after leaving, so pad it to full length with False
            if len(round_positions[key]) == 0:
                logging.debug("An alive key has length 0. Padding to length of tick!")
                logging.debug("Start tick: %s", round_positions["Tick"][0])
                round_positions[key] = [0] * len(round_positions["Tick"])
            else:
                round_positions[key] += [0] * (
                    len(round_positions["Tick"]) - len(round_positions[key])
                )
        elif "Player" in key:
            # If a player wasnt there for the whole round
            # set his name as Nobody and position as 0,0,0.
            if len(round_positions[key]) == 0:
                if "Name" in key:
                    round_positions[key] = ["Nobody"] * len(round_positions["Tick"])
                else:
                    round_positions[key] = (
                        [0] * len(round_positions["Tick"])
                        if "Area" in key
                        else [0.0] * len(round_positions["Tick"])
                    )
            # If a player left mid round pad his name and position
            # with the last values from when he was there.
            # Exactly like it would be if he had died "normally"
            round_positions[key] += [round_positions[key][-1]] * (
                len(round_positions["Tick"]) - len(round_positions[key])
            )
    check_size(round_positions)


def partial_step(
    current: float, previous: float, second_difference: int, step_value: int
) -> float:
    """Calculates intermediate values between two positions.

    Calculates the step_value'th step between previous and
    current with a total of second_difference steps needed.

    Args:
        current (float): Most recent value to interpolate towards
        previous (float): Value at the previous time step to interpolate away from
        second_difference (int): How many intermediary steps are needed
        step_value (int): How many'th intermediate step is to be calculated

    Returns:
        A float corresponding to the needed intermediate step
    """
    return (current - previous) / second_difference * (
        second_difference - step_value + 1
    ) + previous


def append_to_round_positions(
    round_positions: RoundPositions,
    side: Literal["ct", "t"],
    id_number_dict: IDNumberDict,
    player_id: int | str,
    player: PlayerInfo,
    second_difference: int,
    map_name: str,
) -> None:
    """Append a players information from the most recent frame to round_position dict..

    If the time difference between the most recent and the previous frame is larger
    than expected also add interpolated values for the missing time steps.

    Args:
        round_positions (dict[str, list]): Dictionary containing information
            about all players for each timestep
        side (str): Describing the side the given player is currently playing on.
            Should be either "ct" or "t"
        id_number_dict (dict): Dict correlating a players steamID with their ranking.
        player_id (int | str]): Integer or string containing a players steamID
            or bots name
        player (dict): Dict containing the players position and status
            for the most recent timestep
        second_difference (int): Time difference between this
            and the previous frame in seconds.
            If it is larger than 1 then interpolation has to be done.
        map_name (str): A string of the maps name

    Returns:
        None (Dictionary is appended to in place)
    """
    # Add the relevant information of this player to the rounds dict.
    # Add name of the player.
    # Mainly for debugging purposes. Can be removed for actual analysis
    for i in range(second_difference, 0, -1):
        append_step_to_round_positions(
            round_positions,
            side,
            id_number_dict,
            player_id,
            player,
            second_difference,
            map_name,
            step=i,
        )


def _bool_int(value: bool) -> Literal[0, 1]:  # noqa: FBT001
    """Typeguarded conversion from bool to Literal[0, 1]."""
    return 1 if value else 0


def append_step_to_round_positions(
    round_positions: RoundPositions,
    side: Literal["ct", "t"],
    id_number_dict: IDNumberDict,
    player_id: int | str,
    player: PlayerInfo,
    second_difference: int,
    map_name: str,
    step: int,
) -> None:  # sourcery skip: use-fstring-for-concatenation
    """Append a players information from the most recent frame to round_position dict..

    If the time difference between the most recent and the previous frame is larger
    than expected also add interpolated values for the missing time steps.

    Args:
        round_positions (dict[str, list]): Dictionary containing information
            about all players for each timestep
        side (str): Describing the side the given player is currently playing on.
            Should be either "ct" or "t"
        id_number_dict (dict): Dict correlating a players steamID with their ranking.
        player_id (int | str]): Integer or string containing a players steamID
            or bots name
        player (dict): Dict containing the players position and status
            for the most recent timestep
        second_difference (int): Time difference between this
            and the previous frame in seconds.
            If it is larger than 1 then interpolation has to be done.
        map_name (str): A string of the maps name
        step (int): The

    Returns:
        None (Dictionary is appended to in place)
    """
    # Add the relevant information of this player to the rounds dict.
    # Add name of the player.
    # Mainly for debugging purposes. Can be removed for actual analysis
    name_val = (
        player["name"]
        if step == 1
        else round_positions[
            upper_side(side) + "Player" + id_number_dict[side][str(player_id)] + "Name"
        ][-1]
    )
    alive_val = (
        _bool_int(player["isAlive"])
        if step == 1
        else round_positions[
            upper_side(side) + "Player" + id_number_dict[side][str(player_id)] + "Alive"
        ][-1]
    )
    x_val = (
        player["x"]
        if step == 1
        else partial_step(
            player["x"],
            round_positions[
                upper_side(side) + "Player" + id_number_dict[side][str(player_id)] + "x"
            ][-1],
            second_difference,
            step,
        )
    )
    y_val = (
        player["y"]
        if step == 1
        else partial_step(
            player["y"],
            round_positions[
                upper_side(side) + "Player" + id_number_dict[side][str(player_id)] + "y"
            ][-1],
            second_difference,
            step,
        )
    )
    z_val = (
        player["z"]
        if step == 1
        else partial_step(
            player["z"],
            round_positions[
                upper_side(side) + "Player" + id_number_dict[side][str(player_id)] + "z"
            ][-1],
            second_difference,
            step,
        )
    )
    area_val = (
        find_closest_area(map_name, (x_val, y_val, z_val))["areaId"]
        if map_name in NAV
        else -1
    )
    round_positions[
        upper_side(side) + "Player" + id_number_dict[side][str(player_id)] + "Name"
    ].append(name_val)
    # Alive status so the model doesn't have to learn from stopping trajectories.
    round_positions[
        upper_side(side) + "Player" + id_number_dict[side][str(player_id)] + "Alive"
    ].append(alive_val)
    round_positions[
        upper_side(side) + "Player" + id_number_dict[side][str(player_id)] + "x"
    ].append(x_val)
    round_positions[
        upper_side(side) + "Player" + id_number_dict[side][str(player_id)] + "y"
    ].append(y_val)
    round_positions[
        upper_side(side) + "Player" + id_number_dict[side][str(player_id)] + "z"
    ].append(z_val)
    round_positions[
        upper_side(side) + "Player" + id_number_dict[side][str(player_id)] + "Area"
    ].append(area_val)


@overload
def convert_winner_to_int(winner_string: Literal["CT"]) -> Literal[1]: ...


@overload
def convert_winner_to_int(winner_string: Literal["T"]) -> Literal[0]: ...


def convert_winner_to_int(winner_string: str) -> Literal[0, 1] | None:
    """Converts the string of the winning side into 0 or 1.

    CT -> 1
    T -> 0

    Args:
        winner_string: String indicating the winner of a round

    Returns:
        The winner of a round. If the input was invalid then None is returned instead
    """
    if winner_string == "CT":
        return 1
    if winner_string == "T":
        return 0
    logging.error("Winner has to be either CT or T, but was %s instead!", winner_string)
    return None


def get_token_length(map_name: str) -> int:
    """Determine the length of player position tokens for this map.

    The length of the position token is the number of unique named areas.
    Determine it by looping through all areas and
    adding the named area it belongs to to a set
    The number of unique named areas is then the length of the set

    Args:
        map_name: String of the maps name

    Returns:
        Integer of the length of position tokens for this map
    """
    if map_name not in NAV:
        return 1
    area_names: set[str] = {NAV[map_name][area]["areaName"] for area in NAV[map_name]}
    return len(area_names)


def initialize_round(
    current_round: GameRound, match_id: str, map_name: str, winner_id: Literal[0, 1]
) -> tuple[
    bool,
    int,
    IDNumberDict,
    DictInitialized,
    RoundPositions,
    list[int],
]:
    """Initializes the variables for the current round.

    Args:
        current_round (GameRound): Dict containing all the
            information about a single CS:GO round.
        match_id (str): Name of the match the round is played in
        map_name (str): Name of the map the round is being played on
        winner_id (Literal[0,1]): Winner of the current round

    Returns:
        A tuple of:
            skip_round: False, handles whether the round should be skipped entirely
            last_good_frame: 1, handles how many frames ago the last good round was
            id_number_dict: Contains the ID -> player mapping for both sides
            dict_initialized: Contains whether or not the id_number_dict
                for each side has been fully initialized
            round_positions: Will contain player positions for this round
            ticks: Will contain the current and previous tick
            winner_id: Whether or not the CTs won the current round
    """
    skip_round = False
    last_good_frame = 1
    # Dict for mapping players steamID to player number for each round
    id_number_dict: IDNumberDict = {
        "t": {},
        "ct": {},
    }
    # Dict to check if mapping has already been initialized this round
    dict_initialized: DictInitialized = {"t": False, "ct": False}
    # Initialize the dict that tracks player position, status and name for each round
    round_positions = initialize_round_positions(
        match_id,
        map_name,
        current_round["endTScore"] + current_round["endCTScore"],
        winner_id,
    )
    logging.debug("Round number %s", current_round["roundNum"])
    # Iterate over each frame in the round
    # current_tick, last_tick
    ticks = [0, 0]
    # Convert the winning side into a boolean. 1 for CT and 0 for T
    return (
        skip_round,
        last_good_frame,
        id_number_dict,
        dict_initialized,
        round_positions,
        ticks,
    )


def analyze_players(
    frame: GameFrame,
    dict_initialized: DictInitialized,
    id_number_dict: IDNumberDict,
    side: Literal["ct", "t"],
    round_positions: RoundPositions,
    second_difference: int,
    map_name: str,
) -> None:
    """Analyzes the players in a given frame and side.

    Args:
        frame (GameFrame): Dicitionary containing all information
            about the current round
        dict_initialized (dict[str, bool]): Dict containing information about
            whether or not a given side has been initialized already
        id_number_dict (dict[str, dict[str, str]]): Dict mapping player to a number
        side (Literal["ct","t"]): String of the current side
        round_positions (dict[str, list]): Dict containing all player trajectories
            of the current round
        second_difference (int): Time difference from last frame to the current one
        map_name (str): Name of the map under consideration

    Returns:
        None (round_positions is appended to in place)
    """
    for player_index, player in enumerate(frame[side]["players"] or []):
        _analyze_player(
            player,
            player_index,
            dict_initialized,
            id_number_dict,
            side,
            round_positions,
            second_difference,
            map_name,
        )
    # After looping over each player in the team once
    # the steamID matching has been initialized
    dict_initialized[side] = True


def _analyze_player(
    player: PlayerInfo,
    player_index: int,
    dict_initialized: DictInitialized,
    id_number_dict: IDNumberDict,
    side: Literal["ct", "t"],
    round_positions: RoundPositions,
    second_difference: int,
    map_name: str,
) -> None:
    if not proper_player_number(player_index) or player_index == N_PLAYERS:
        return
    player_id = get_player_id(player)
    # If the dict of the team has not been initialized add that player.
    # Should only happen once per player per team per round
    # But for each team can happen on different rounds in some rare cases.
    if dict_initialized[side] is False:
        id_number_dict[side][str(player_id)] = int_to_string_n_players(player_index + 1)
    # If a player joins mid round
    # (either a bot due to player leaving or player (re)joining)
    # do not use him for this round.
    if str(player_id) not in id_number_dict[side]:
        return
    append_to_round_positions(
        round_positions,
        side,
        id_number_dict,
        player_id,
        player,
        second_difference,
        map_name,
    )


def add_general_information(
    frame: GameFrame,
    current_round: GameRound,
    second_difference: int,
    token_length: int,
    round_positions: RoundPositions,
    ticks: list[int],
    index: int,
    map_name: str,
    last_good_frame: int,
) -> None:
    """Adds general information to round_positions.

    Args:
        frame (GameFrame): Dict containing all information about the current round
        current_round (GameRound): Dict containing all information
            about the current round
        second_difference (int): Time difference from last frame to the current one
        token_length: (int): Length of the position token for the current map
        round_positions (dict[str, list]): Dictionary containing all player
            trajectories of the current round
        ticks (list[int]): Will contain the current and previous tick
        index (int): Index of the current frame
        map_name (str): Name of the map under consideration
        last_good_frame (int): Indicates how many frames ago the last good round was.

    Returns:
        None (round_positions)
    """
    if not current_round["frames"] or (index - last_good_frame) >= len(
        current_round["frames"]
    ):
        return
    token_frames = (
        [frame]
        if second_difference == 1
        else build_intermediate_frames(
            frame,
            current_round["frames"][index - last_good_frame],
            second_difference,
        )
    )
    for i in range(second_difference, 0, -1):
        tokens = get_position_token(
            token_frames[second_difference - i],
            map_name,
            token_length,
        )
        round_positions["Tick"].append(
            round(
                partial_step(
                    ticks[0],
                    ticks[1],
                    second_difference,
                    i,
                )
            )
        )
        round_positions["token"].append(tokens["token"])
        round_positions["CTtoken"].append(tokens["ctToken"])
        round_positions["Ttoken"].append(tokens["tToken"])
        round_positions["interpolated"].append(0 if i == 1 else 1)


def analyze_frames(
    current_round: GameRound,
    map_name: str,
    token_length: int,
    tick_rate: int,
    general_information: tuple[str, Literal[0, 1]],
) -> tuple[bool, RoundPositions]:
    """Analyzes the frames in a given round.

    Args:
        current_round (GameRound): Dict containing all information
            about the current round
        map_name (str): Name of the map under consideration
        token_length: (int): Length of the position token for the current map
        tick_rate (int): Tickrate of the current game
        general_information (tuple): Tuple of
            match_id (str): Name of the match the round is played in
            winner_id (Literal[0,1]): Winner of the current round

    Returns:
        A tuple of:
            skip_round (bool): Whether the current round should be skipped
            round_positions (dict): A dictionary of all the players positions
                throughout the round
    """
    (
        skip_round,
        last_good_frame,
        id_number_dict,
        dict_initialized,
        round_positions,
        ticks,
    ) = initialize_round(
        current_round, general_information[0], map_name, general_information[1]
    )
    for index, frame in enumerate(current_round["frames"] or []):
        # There should never be more than 5 players alive in a team.
        # If that does happen completely skip the round.
        # Propagate that information past the loop by setting skip_round to true
        if (
            frame["ct"]["alivePlayers"] > N_PLAYERS
            or frame["t"]["alivePlayers"] > N_PLAYERS
        ):
            logging.error(
                "Found frame with more than %s players alive in a team in round %s !",
                N_PLAYERS,
                current_round["roundNum"],
            )
            skip_round = True
            return skip_round, round_positions
        if frame["ct"]["players"] is None or frame["t"]["players"] is None:
            logging.debug(
                "Side['players'] is none. Skipping this frame from round %s !",
                current_round["roundNum"],
            )
            last_good_frame += 1
            continue
        # Loop over both sides
        ticks[0] = int(frame["tick"])
        if ticks[1] == 0:
            second_difference = 1
        else:
            second_difference = int((ticks[0] - ticks[1]) / tick_rate)
        for side in ("ct", "t"):
            analyze_players(
                frame,
                dict_initialized,
                id_number_dict,
                side,
                round_positions,
                second_difference,
                map_name,
            )
        # If at least one side has been initialized the round can be used for analysis,
        # so add the tick value used for tracking.
        # Will also removed for the eventual analysis.
        # But you do not want to set it for frames where you have no player data
        # which should only ever happen in the first frame of a round at worst.
        if True in dict_initialized.values():
            add_general_information(
                frame,
                current_round,
                second_difference,
                token_length,
                round_positions,
                ticks,
                index,
                map_name,
                last_good_frame,
            )
            last_good_frame = 1
            ticks[1] = ticks[0]
    return skip_round, round_positions  # False


def analyze_rounds(
    data: Game, position_dataset: list[RoundPositions], match_id: str
) -> None:
    """Analyzes all rounds in a game and adds their relevant data.

    Loops over every round in "data, every frame in each rounds,
    every side in each frame and every player in each side
    and adds their position as well as auxiliary information to dictionary.
    This dictionary and more auxiliary information is then appended to the overall
    dictionary containing all information about matches on this map.

    Args:
        data (dict): Dictionary containing all information about a CS:GO game
        position_dataset (list): List containing trajectory
            information about rounds on a given map
        match_id (str): String representing the name of an input demo file

    Returns:
        None position_dataset_dict is modified inplace
    """
    map_name = data["mapName"]
    token_length = get_token_length(map_name)
    tick_rate = 1 << (data["tickRate"] - 1).bit_length()
    for current_round in data["gameRounds"] or []:
        # If there are no frames in the round skip it.
        if frame_is_empty(current_round):
            continue
        winning_side = current_round["winningSide"]
        if not is_valid_side(winning_side):
            continue
        winner_id = convert_winner_to_int(winning_side)
        skip_round, round_positions = analyze_frames(
            current_round, map_name, token_length, tick_rate, (match_id, winner_id)
        )
        # Skip the rest of the loop if the whole round should be skipped.
        # Pad to full length in case a player left
        pad_to_full_length(round_positions)
        if skip_round or len(round_positions["Tick"]) == 0:
            continue
        position_dataset.append(round_positions)
        # Check that each entry in the dataset has the same length.
        # Especially that for each round there is a trajectory dataframe.
        logging.debug(
            "Finished another round and appended to dataset. Now at size %s",
            len(position_dataset),
        )


def _map_directories(base_dir: str) -> Iterator[Path]:
    """Crawl all desired directories and filter them.

    Args:
        base_dir (str): Path of the base directory from which subdirectories
            should be crawled.

    Yields:
        Paths of all the desired files.
    """
    # List of maps already done -> Do not do them again
    done: set[str] = {"cache", "cbble", "dust2", "inferno", "mirage", "nuke"}
    # List of maps to specifically do -> only do those
    to_do: set[str] = {"ancient"}
    for directory in Path(base_dir).iterdir():
        if directory.is_dir():
            logging.info("Looking at directory %s", directory)
            if directory.name in done:
                logging.info("Skipping this directory as it has already been analyzed.")
                continue
            if to_do and directory.name not in to_do:
                logging.info(
                    "Skipping this directory as it not one "
                    "of those that should be analyzed."
                )
                continue
            yield directory


def main(args: list[str]) -> None:
    """Runs over each demo by map and produces a dataframe tracking player positions."""
    parser = argparse.ArgumentParser("Analyze the early mid fight on inferno")
    parser.add_argument(
        "-d", "--debug", action="store_true", default=False, help="Enable debug output."
    )
    parser.add_argument(
        "--dir",
        default=r"D:\CSGO\Demos\Maps",
        help="Path to directory containing the individual map directories.",
    )
    parser.add_argument(
        "-l",
        "--log",
        default=r"D:\CSGO\ML\CSGOML\logs\Tensorflow_Input_Preparation.log",
        help="Path to output log.",
    )
    parser.add_argument(
        "-r",
        "--reanalyze",
        default=False,
        action="store_true",
        help="Reanalyze all demos for each map."
        " Otherwise only analyze those created after the "
        "existing results json has been created and append those.",
    )
    options = parser.parse_args(args)

    setup_logging(options)

    logging.info("Starting")

    for directory in _map_directories(options.dir):
        output_json_path = (
            str(directory)
            + r"\Analysis\Prepared_Input_Tensorflow_"
            + directory.name
            + "_polars.json"
        )
        seen_match_ids: set[str] = set()
        if not options.reanalyze and os.path.exists(output_json_path):
            prev_dataframe = pl.read_json(output_json_path)
            seen_match_ids = set(prev_dataframe.get_column("MatchID").unique())
        else:
            prev_dataframe = pl.DataFrame()
        logging.info(
            "The following %s MatchIDs are already included "
            "in the existing json file and will be skipped: %s",
            len(seen_match_ids),
            seen_match_ids,
        )
        position_dataset: list[RoundPositions] = []
        for files_done, file_path in enumerate(directory.iterdir()):
            match_id = file_path.stem
            # checking if it is a file
            if (
                match_id not in seen_match_ids
                and os.path.isfile(file_path)
                and file_path.suffix == ".json"
            ):
                logging.info("Analyzing file %s", file_path.name)
                with open(file_path, encoding="utf-8") as demo_json:
                    data = json.load(demo_json)
                analyze_rounds(data, position_dataset, match_id)
                if files_done > 0 and files_done % 50 == 0:
                    position_dataset_df = pl.from_dicts(
                        position_dataset  # pyright: ignore [reportGeneralTypeIssues]
                    )
                    position_dataset_df = pl.concat(
                        [prev_dataframe, position_dataset_df], how="vertical"
                    )
                    position_dataset_df.write_ndjson(output_json_path)
                    logging.info("Wrote output json to: %s", output_json_path)
                    position_dataset = []
                    prev_dataframe = position_dataset_df
        # Transform to dataset and write it to file as json
        position_dataset_df = pl.from_dicts(
            position_dataset  # pyright: ignore [reportGeneralTypeIssues]
        )
        position_dataset_df = pl.concat(
            [prev_dataframe, position_dataset_df], how="vertical"
        )
        position_dataset_df.write_ndjson(output_json_path)
        logging.info("Wrote output json to: %s", output_json_path)


if __name__ == "__main__":
    main(sys.argv[1:])
