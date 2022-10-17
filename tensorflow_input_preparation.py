"""Runs over each demo separated by map and produces a dataframe tracking player positions.

For each map folder in the provided directory this scripts produces a dataframe trackining player movement in each round.
The produced dataframe has one row per round and for each round columns for the matchid, mapname, roundnumber, winning side and a dataframe containing the trajectory data.
The trajectory data contains for each parsed frame (every 128th recorded tick. Ideally 1 frame per second but can be less depending on server configuration) each players
x,y,z coordinates, name and alive status. additionally three position tokens are generated. one encapsulating the positions of each t side player, one of each ct side player and a combined one.
The dataframe is finally exported to a json file.


    Typical usage example:
        python tensorflow_input_preparation --dir 'E:\\PhD\\MachineLearning\\CSGOData\\ParsedDemos'

"""

#!/usr/bin/env python

import os
import sys
import logging
import argparse
import json
import copy
import pandas as pd
from awpy.analytics.nav import generate_position_token, find_closest_area
from awpy.data import NAV


def initialize_round_positions():
    """Initializes dictionary of lists for one rounds positions.

    The positions dictinary contains the following keys:
    Tick, token, interpolated, CTtoken, Ttoken and for each player on each side
    side+"Player"+number+feature where side is in [CT,T], number in range(1,6) and feature in ["Alive", "Name", "x", "y", "z"]

    Args:
        None

    Returns:
        Empty round_positions dictionary
    """
    round_positions = {}
    round_positions["Tick"] = []
    round_positions["token"] = []
    round_positions["interpolated"] = []
    for side in ["CT", "T"]:
        round_positions[side + "token"] = []
        for number in range(1, 6):
            for feature in ["Alive", "Name", "x", "y", "z", "Area"]:
                round_positions[side + "Player" + str(number) + feature] = []
    return round_positions


def build_intermediate_frames(current_frame, previous_frame, second_difference):
    """Builds intermediate frames if fewer than 1 were recorded per second due to the server configuration.

    Linearly interpolates the players x,y,z positions from one frame to the next while keeping everything else as it is in the first frame.
    This has to be done to be able to generate intermediate position tokens.

    Args:
        current_frame: Dictionary the contains all information about player positions at the most recent timestep
        previous_frame: Dictionary the contains all information about player positions one timestep previously
        second_difference: Difference in the number of seconds between the previous and current frames

    Returns:
        A list of intermediate frames from after previous_frame to including current_frame
    """
    intermdiate_frames = []
    for i in range(second_difference, 0, -1):
        this_frame = copy.deepcopy(current_frame)
        for side in ["t", "ct"]:
            if current_frame[side]["players"] is None:
                logging.debug(
                    "Side['players'] is none. Skipping this side for frame interpolation !",
                )
                continue
            for index, player in enumerate(current_frame[side]["players"]):
                for prev_player in previous_frame[side]["players"]:
                    if prev_player["steamID"] == player["steamID"]:
                        this_frame[side]["players"][index]["isAlive"] = (
                            int(player["isAlive"])
                            if i == 1
                            else int(prev_player["isAlive"])
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


def get_postion_token(frame, map_name, token_length):
    """Generate a dictionary of position tokens from frame dictionary and map_name.

    The position token is a string of integers representing the number of players in a given unique named area.
    If token generation fails because of empty players or unsupported map then strings of 0 are returned instead.

    Args:
        frame: Dictionary containing all information about both teams status and each players position and status
        map_name: A string of the maps name
        token_length: An integer of the length of one sides position token on the played map

    Returns:
        Dictionary of three position tokens. One for each side and aditionally a combined one
    """
    try:
        tokens = generate_position_token(map_name, frame)
    except TypeError:
        tokens = {
            "tToken": token_length * "0",
            "ctToken": token_length * "0",
            "token": 2 * token_length * "0",
        }
        logging.debug(
            "Got TypeError when trying to generate position token. This is due to one sides 'player' entry being none."
        )
    except KeyError:
        tokens = {
            "tToken": token_length * "0",
            "ctToken": token_length * "0",
            "token": 2 * token_length * "0",
        }
        logging.debug(
            "Got KeyError when trying to generate position token. This is due to the map not being supported."
        )
    except ValueError:
        tokens = {
            "tToken": token_length * "0",
            "ctToken": token_length * "0",
            "token": 2 * token_length * "0",
        }
        logging.debug(
            "Got KeyError when trying to generate position token. This is due to the map not being supported."
        )
    return tokens


def initialize_position_dataset_dict():
    """Initializes the dictionary of lists containing one entry per round.

    The dictinary contains the following keys:
    MatchID, MapName, Round, Winner, position_df

    Args:
        None

    Returns:
        Empty position_dataset_dict dictionary
    """
    position_dataset_dict = {}
    position_dataset_dict["MatchID"] = []
    position_dataset_dict["MapName"] = []
    position_dataset_dict["Round"] = []
    position_dataset_dict["Winner"] = []
    position_dataset_dict["position_df"] = []
    return position_dataset_dict


def check_size(dictionary):
    """Checks that the size of each list behind every dictionary key is the same.

    The input dictionary is expected to have a list corresponding to each key and that each list has the same size.

    Args:
        dictionary: dictionary of lists with the expactation that each list has the same size

    Returns:
        length: An integer corresponding to the length of every list in the dictionary.

    Raises:
        sys.exit if not all lists have the same size
    """
    length = 0
    same_size = True
    for key in dictionary:
        logging.debug("Length of key %s is %s", key, len(dictionary[key]))
        if length == 0:
            length = len(dictionary[key])
        else:
            same_size = length == len(dictionary[key])
            if not same_size:
                logging.error(
                    "Key %s has size %s while %s is expected!",
                    key,
                    len(dictionary[key]),
                    length,
                )
                logging.error(dictionary)
                sys.exit(
                    "Not all elements in dict have the same size. Something has gone wrong."
                )
    return length


def frame_is_empty(current_round):
    """Checks whether a round dicionary contains None or an empty list frames.

    None or empty frames will raise exceptions when trying to extract player information out of them.
    This method checks if a frame is empty or None and logs and error message if either is the case.

    Args:
        current_round: A dictionary containing all the information about a single CS:GO round.

    Returns:
        A boolean whether the value behind the frames key is None or the list has a length of zero.
    """
    if current_round["frames"] is None:
        logging.error(current_round)
        logging.error("Found none frames in round %s!", current_round["roundNum"])
        return True
    if len(current_round["frames"]) == 0:
        logging.error("Found empty frames in round %s!", current_round["roundNum"])
        return True
    return False


def get_player_id(player):
    """Extracts a players steamID from their player dictionary in a given frame.

    Each player has a unique steamID by which they can be identified.
    Bots, however, do not naturally have a steamID and are instead all assigned 0.
    To avoid overlap bots are instead identified by their name which is unique in any given game.

    Args:
        player: Dictionary about a players position and status in a given frame

    Returns:
        An integer corresponding to their steamID if they are a player or a string corresponding to the bots name if not.
    """
    if player["steamID"] == 0:
        return player["name"]
    return player["steamID"]


def pad_to_full_length(round_positions):
    """Pads each entry in a given round_positions dictionary to the full length.

    For every player their name, status and position should be stored for every timestep in the round.
    If a player leaves mid round it can happen that their entries are incomplete.
    To avoid problems further down the line their entries are padded the the full length of the round.
    If they left mid round their most recent information is used. If they left before the round started dummy data is produced instead.
    Afterwards a check is performed to assert that every entry in the dictionary has the same length.

    Args:
        round_positions: Dictionary containing information about all players for each timestep

    Returns:
        None. Dictionary is padded in place.

    """
    if len(round_positions["Tick"]) == 0:
        return
    for key in round_positions:
        if "Alive" in key:
            # If the Alive list is completely empty fill it with a dead player
            # If the player left mid round he is considered dead for the time after leaving, so pad it to full length with False
            if len(round_positions[key]) == 0:
                logging.debug("An alive key has length 0. Padding to length of tick!")
                logging.debug("Start tick: %s", round_positions["Tick"][0])
                round_positions[key] = [0] * len(round_positions["Tick"])
            else:
                round_positions[key] += [0] * (
                    len(round_positions["Tick"]) - len(round_positions[key])
                )
        elif "Player" in key:
            # If a player wasnt there for the whole round set his name as Nobody and position as 0,0,0.
            if len(round_positions[key]) == 0:
                if "Name" in key:
                    round_positions[key] = ["Nobody"] * len(round_positions["Tick"])
                else:
                    round_positions[key] = [0.0] * len(round_positions["Tick"])
            # If a player left mid round pad his name and position with the last values from when he was there. Exactly like it would be if he had died "normally"
            round_positions[key] += [round_positions[key][-1]] * (
                len(round_positions["Tick"]) - len(round_positions[key])
            )
    _ = check_size(round_positions)


def partial_step(current, previous, second_difference, step_value):
    """Calculates intermediate values between two positions.

    Calculates the step_value'th step between previous and current with a total of second_difference steps needed.

    Args:
        current: Float corresponding to the most recent value to interpolate towards
        previous: Float corresponding to the value at the previous time step to interpolate away from
        second_difference: Integer determining how many intermediary steps are needed
        step_value: Integer describing the how many'th intermediate step is to be calculated

    Returns:
        A float corresponding to the needed intermediate step
    """
    return (current - previous) / second_difference * (
        second_difference - step_value + 1
    ) + previous


def append_to_round_positions(
    round_positions,
    side,
    id_number_dict,
    player_id,
    player,
    second_difference,
    map_name,
):
    """Append a players information from the most recent frame to their list entries in the round_position dictionary

    If the time difference between the most recent and the previous frame is larger than expected also add interpolated values for the missing time steps.

    Args:
        round_positions: Dictionary containing information about all players for each timestep
        side: A string describing the side the given player is currently playing on. Should be either "ct" or "t"
        id_number_dict: A dictionary correlating a players steamID with their ranking number.
        player_id: An integer or string containing a players steamID or bots name
        player: A dictionary containing the players position and status for the most recent timestep
        second_difference: An integer corresponding the the time difference between this and the previous frame in seconds. If it is larger than 1 then interpolation has to be done.
        map_name: A string of the maps name

    Returns:
        None (Dictionary is appended to in place)
    """
    # Add the relevant information of this player to the rounds dict.
    # Add name of the player. Mainly for debugging purposes. Can be removed for actual analysis
    for i in range(second_difference, 0, -1):
        name_val = (
            player["name"]
            if i == 1
            else round_positions[
                side.upper() + "Player" + id_number_dict[side][str(player_id)] + "Name"
            ][-1]
        )
        alive_val = (
            int(player["isAlive"])
            if i == 1
            else round_positions[
                side.upper() + "Player" + id_number_dict[side][str(player_id)] + "Alive"
            ][-1]
        )
        x_val = (
            player["x"]
            if i == 1
            else partial_step(
                player["x"],
                round_positions[
                    side.upper() + "Player" + id_number_dict[side][str(player_id)] + "x"
                ][-1],
                second_difference,
                i,
            )
        )
        y_val = (
            player["y"]
            if i == 1
            else partial_step(
                player["y"],
                round_positions[
                    side.upper() + "Player" + id_number_dict[side][str(player_id)] + "y"
                ][-1],
                second_difference,
                i,
            )
        )
        z_val = (
            player["z"]
            if i == 1
            else partial_step(
                player["z"],
                round_positions[
                    side.upper() + "Player" + id_number_dict[side][str(player_id)] + "z"
                ][-1],
                second_difference,
                i,
            )
        )
        area_val = (
            find_closest_area(map_name, [x_val, y_val, z_val])["areaId"]
            if map_name in NAV
            else -1
        )
        round_positions[
            side.upper() + "Player" + id_number_dict[side][str(player_id)] + "Name"
        ].append(name_val)
        # Is alive status so the model does not have to learn that from stopping trajectories
        round_positions[
            side.upper() + "Player" + id_number_dict[side][str(player_id)] + "Alive"
        ].append(alive_val)
        round_positions[
            side.upper() + "Player" + id_number_dict[side][str(player_id)] + "x"
        ].append(x_val)
        round_positions[
            side.upper() + "Player" + id_number_dict[side][str(player_id)] + "y"
        ].append(y_val)
        round_positions[
            side.upper() + "Player" + id_number_dict[side][str(player_id)] + "z"
        ].append(z_val)
        round_positions[
            side.upper() + "Player" + id_number_dict[side][str(player_id)] + "Area"
        ].append(area_val)


def convert_winner_to_int(winner_string):
    """Converts the string of the winning side into 0 or 1.

    CT -> 1
    T -> 0

    Args:
        winner_string: String indicating the winner of a round

    Returns:
        A boolean indicating the winner of a round. If the input was invalid then None is returned instead
    """
    if winner_string == "CT":
        return 1
    elif winner_string == "T":
        return 0
    else:
        logging.error(
            "Winner has to be either CT or T, but was %s instead!", winner_string
        )
        return None


def get_token_length(map_name):
    """Determine the lenght of player position tokens for this map.

    The length of the position token is the number of unique named areas.
    Determine it by looping through all areas and add the named area it belongs to to a set
    The number of unique named areas is then the length of the set

    Args:
        map_name: String of the maps name

    Returns:
        Integer of the length of position tokens for this map
    """
    area_names = set()
    if map_name not in NAV:
        return 1
    for area in NAV[map_name]:
        area_names.add(NAV[map_name][area]["areaName"])
    return len(area_names)


def analyze_rounds(data, position_dataset_dict, match_id):
    """Analyzes all rounds in a game and adds their relevant data to position_dataset_dict.

    Loops over every round in "data, every frame in each rounds, every side in each frame and every player in each side
    and adds their position as well as auxilliary information to dictionary. This dictionary and more auxilliary information
    is then appended to the overall dictionary containing all information about matches on this map.

    Args:
        data: Dictionary containing all information about a CS:GO game
        position_dataset_dict: Dictionary containing trajectory information about rounds on a given map
        match_id: String representing the name of an input demo file

    Returns:
        None position_dataset_dict is modified inplace
    """
    map_name = data["mapName"]
    token_length = get_token_length(map_name)
    for current_round in data["gameRounds"]:
        skip_round = False
        last_good_frame = 1
        # If there are no frames in the round skip it.
        if frame_is_empty(current_round):
            continue
        # Dict for mapping players steamID to player number for each round
        id_number_dict = {"t": {}, "ct": {}}
        # Dict to check if mapping has already been initialized this round
        dict_initialized = {"t": False, "ct": False}
        # Initialize the dict that tracks player position, status and name for each round
        round_positions = initialize_round_positions()
        logging.debug("Round number %s", current_round["roundNum"])
        # Iterate over each frame in the round
        # current_tick, last_tick
        ticks = [0, 0]
        # Convert the winning side into a boolean. 1 for CT and 0 for T
        winner_id = convert_winner_to_int(current_round["winningSide"])
        # If winner is neither then None is returned and the round should be skipped.
        if winner_id is None:
            continue
        for index, frame in enumerate(current_round["frames"]):
            # There should never be more than 5 players alive in a team.
            # If that does happen completely skip the round.
            # Propagate that information past the loop by setting skip_round to true
            if frame["ct"]["alivePlayers"] > 5 or frame["t"]["alivePlayers"] > 5:
                logging.error(
                    "Found frame with more than 5 players alive in a team in round %s !",
                    current_round["roundNum"],
                )
                skip_round = True
                break
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
                second_difference = int((ticks[0] - ticks[1]) / 128)
            for side in ["ct", "t"]:
                # If the side does not contain any players for that frame skip it

                # Loop over each player in the team.
                for player_index, player in enumerate(frame[side]["players"]):
                    # logging.info(f)
                    player_id = get_player_id(player)
                    # If the dict of the team has not been initialized add that player. Should only happen once per player per team per round
                    # But for each team can happen on different rounds in some rare cases.
                    if dict_initialized[side] is False:
                        id_number_dict[side][str(player_id)] = str(player_index + 1)
                    # logging.debug(id_number_dict[side])
                    # If a player joins mid round (either a bot due to player leaving or player (re)joining)
                    # do not use him for this round.
                    if str(player_id) not in id_number_dict[side]:
                        continue
                    append_to_round_positions(
                        round_positions,
                        side,
                        id_number_dict,
                        player_id,
                        player,
                        second_difference,
                        map_name,
                    )
                # After looping over each player in the team once the steamID matching has been initialized
                dict_initialized[side] = True
            # If at least one side has been initialized the round can be used for analysis, so add the tick value used for tracking.
            # Will also removed for the eventual analysis.
            # But you do not want to set it for frames where you have no player data which should only ever happen in the first frame of a round at worst.
            if True in dict_initialized.values():
                token_frames = (
                    [frame]
                    if second_difference == 1
                    else build_intermediate_frames(
                        frame,
                        current_round["frames"][index - last_good_frame],
                        second_difference,
                    )
                )
                last_good_frame = 1
                for i in range(second_difference, 0, -1):
                    tokens = get_postion_token(
                        token_frames[second_difference - i],
                        map_name,
                        token_length,
                    )
                    round_positions["Tick"].append(
                        partial_step(
                            *ticks,
                            second_difference,
                            i,
                        )
                    )
                    round_positions["token"].append(
                        tokens["token"] if i == 1 else round_positions["token"][-1]
                    )
                    round_positions["CTtoken"].append(
                        tokens["ctToken"] if i == 1 else round_positions["CTtoken"][-1]
                    )
                    round_positions["Ttoken"].append(
                        tokens["tToken"] if i == 1 else round_positions["Ttoken"][-1]
                    )
                    round_positions["interpolated"].append(0 if i == 1 else 1)
                ticks[1] = ticks[0]
        # Skip the rest of the loop if the whole round should be skipped.
        # Pad to full length in case a player left
        pad_to_full_length(round_positions)
        if skip_round or len(round_positions["Tick"]) == 0:
            continue
        # Append demo id, map name and round number to the final dataset dict.)
        position_dataset_dict["MatchID"].append(match_id)
        position_dataset_dict["MapName"].append(map_name)
        position_dataset_dict["Round"].append(
            current_round["endTScore"] + current_round["endCTScore"]
        )
        position_dataset_dict["Winner"].append(winner_id)

        # Make sure each entry in the round_positions has the same size now. Especially that nothing is longer than the Tick entry which would indicate multiple players filling on player number
        # Transform to dataframe
        round_positions_df = pd.DataFrame(round_positions)
        # Add the rounds trajectory information to the overall dataset.
        position_dataset_dict["position_df"].append(round_positions_df)
        # Check that each entry in the dataset has the same length. Especially that for each round there is a trajectory dataframe.
        logging.debug(
            "Finished another round and appended to dataset. Now at size %s",
            check_size(position_dataset_dict),
        )


# def GetMinMaxFromFirst(reference_position_df):
#     minimum={"x":sys.maxsize,"y":sys.maxsize,"z":sys.maxsize}
#     maximum={"x":-sys.maxsize,"y":-sys.maxsize,"z":-sys.maxsize}
#     for feature in ["x","y","z"]:
#         for side in ["CT","T"]:
#             for number in range(1,6):
#                 maximum[feature]=max(reference_position_df[side+"Player"+str(number)+feature].max(),maximum[feature])
#                 minimum[feature]=min(reference_position_df[side+"Player"+str(number)+feature].min(),minimum[feature])
#     return minimum,maximum

# "D:\CSGO\Demos\Maps"
def main(args):
    """Runs over each demo separated by map and produces a dataframe tracking player positions."""
    parser = argparse.ArgumentParser("Analyze the early mid fight on inferno")
    parser.add_argument(
        "-d", "--debug", action="store_true", default=False, help="Enable debug output."
    )
    parser.add_argument(
        "--dir",
        default=r"E:\PhD\MachineLearning\CSGOData\ParsedDemos",
        # default=r"D:\CSGO\Demos\Maps",
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
        help="Reanalyze all demos for each map. Otherwise only analyze those created after the existing results json has been created and append those.",
    )
    options = parser.parse_args(args)

    if options.debug:
        logging.basicConfig(
            filename=options.log,
            encoding="utf-8",
            level=logging.DEBUG,
            filemode="w",
            format="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        logging.basicConfig(
            filename=options.log,
            encoding="utf-8",
            level=logging.INFO,
            filemode="w",
            format="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    logging.info("Starting")
    # done={"ancient","cache","cbble","cs_rush","dust2","facade","inferno","marquis","mirage","mist","nuke","overpass","resort","santorini","santorini_playtest","season","train","vertigo"}
    # List of maps already done -> Do not do them again
    done = {"ancient", "cache", "cbble", "dust2", "inferno", "mirage", "nuke"}
    # List of maps to specifically do -> only do those
    to_do = set()
    # to_do = []
    for directoryname in os.listdir(options.dir):
        directory = os.path.join(options.dir, directoryname)
        if os.path.isdir(directory):
            logging.info("Looking at directory %s", directory)
            if directoryname in done:
                logging.info("Skipping this directory as it has already been analyzed.")
                continue
            if (len(to_do) > 0) and (directoryname not in to_do):
                logging.info(
                    "Skipping this directory as it not one of those that should be analyzed."
                )
                continue
            output_json_path = (
                directory
                + r"\Analysis\Prepared_Input_Tensorflow_"
                + directoryname
                + ".json"
            )
            seen_match_ids = set()
            if not options.reanalyze and os.path.exists(output_json_path):
                with open(output_json_path, encoding="utf-8") as pre_analyzed:
                    prev_dataframe = pd.read_json(pre_analyzed)
                seen_match_ids = set(prev_dataframe["MatchID"].unique())
            else:
                prev_dataframe = pd.DataFrame()
            logging.info(
                "The following %s MatchIDs are already included in the existing json file and will be skipped: %s",
                len(seen_match_ids),
                seen_match_ids,
            )
            position_dataset_dict = initialize_position_dataset_dict()
            for files_done, filename in enumerate(os.listdir(directory)):
                file_path = os.path.join(directory, filename)
                match_id = filename.rsplit(".", 1)[0]
                # checking if it is a file
                if (
                    match_id not in seen_match_ids
                    and os.path.isfile(file_path)
                    and filename.endswith(".json")
                ):
                    logging.info("Analyzing file %s", filename)
                    with open(file_path, encoding="utf-8") as demo_json:
                        data = json.load(demo_json)
                    analyze_rounds(data, position_dataset_dict, match_id)
                    if files_done > 0 and files_done % 50 == 0:
                        position_dataset_df = pd.DataFrame(position_dataset_dict)
                        position_dataset_df = pd.concat(
                            [prev_dataframe, position_dataset_df], ignore_index=True
                        )
                        position_dataset_df.to_json(output_json_path)  # , indent=2)
                        logging.info("Wrote output json to: %s", output_json_path)
                        position_dataset_dict = initialize_position_dataset_dict()
                        with open(output_json_path, encoding="utf-8") as pre_analyzed:
                            prev_dataframe = pd.read_json(pre_analyzed)
            # Transform to dataset and write it to file as json
            position_dataset_df = pd.DataFrame(position_dataset_dict)
            position_dataset_df = pd.concat(
                [prev_dataframe, position_dataset_df], ignore_index=True
            )
            position_dataset_df.to_json(output_json_path)  # , indent=2)
            logging.info("Wrote output json to: %s", output_json_path)
            # Has to be read back in like
            # with open("D:\CSGO\Demos\Maps\vertigo\Analysis\Prepared_Input_Tensorflow_vertigo.json", encoding='utf-8') as PreAnalyzed:
            #   dataframe=pd.read_json(PreAnalyzed)
            #   round_df=pd.DataFrame(dataframe.iloc[30]["position_df"])
            #   logging.info(dataframe)
            #   logging.info(pd.DataFrame(dataframe.iloc[30]["position_df"]))


if __name__ == "__main__":
    main(sys.argv[1:])
