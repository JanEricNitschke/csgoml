"""Determine whether a given engagement is favourable to the T or CT side.

    Typical usage example:

    analyzer = FightAnalyzer(
    options.debug,
    times=[options.starttime, options.endtime],
    directory=options.dir,
    log=options.log,
    )
    dataframe = analyzer.analyze_demos()
    analyzer.print_ct_win_percentage_for_time_cutoffs(dataframe)
"""
#!/usr/bin/env python

import os
import argparse
import sys
import json
import logging
import pandas as pd
import numpy as np
from awpy.analytics.nav import find_closest_area
from awpy.data import NAV


class FightAnalyzer:
    """Determine whether a given engagement is favourable to the T or CT side.

    Attributes:
        json: Path to save the resulting dataframe to
        positions: A list [CTPositions, TPositions] where CT- and TPositions are lists of map locations that should be considered for the analysis.
        times: A list containing the time window (start- and endtime) to consider for the analysis.
        directory: A path to the directory where the demos to be analyzed reside.
        weapons: A set of weapon names. Both sides should have had at least one of these weapons.
        n_analyzed: An integer of the number of demos that have been analyzed
    """

    def __init__(
        self,
        debug=False,
        times=None,
        my_json=r"D:\CSGO\Demos\Maps\inferno\Analysis\Inferno_kills_mid.json",
        directory=r"D:\CSGO\Demos\Maps\inferno",
        log=r"D:\CSGO\ML\CSGOML\FightAnalyzer.log",
        positions=None,
        weapons=None,
    ):
        if debug:
            logging.basicConfig(
                filename=log,
                encoding="utf-8",
                level=logging.DEBUG,
                filemode="w",
                format="%(asctime)s %(levelname)-8s %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        else:
            logging.basicConfig(
                filename=log,
                encoding="utf-8",
                level=logging.INFO,
                filemode="w",
                format="%(asctime)s %(levelname)-8s %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        FightAnalyzer.logger = logging.getLogger("FightAnalyzer")
        if not positions:
            self.positions = {"CT": ["TopofMid", "Middle"], "T": ["Middle", "TRamp"]}
        else:
            self.positions = positions
        if not times:
            self.times = [5, 25]
        else:
            self.times = times
        self.directory = directory
        if weapons is None:
            self.weapons = {
                "M4A4",
                "AWP",
                "AK-47",
                "Galil AR",
                "M4A1",
                "SG 553",
                "SSG 08",
                "G3SG1",
                "SCAR-20",
                "FAMAS",
            }
        else:
            self.weapons = weapons
        self.n_analyzed = 0
        self.json = my_json

    def get_area_from_pos(self, map_name, pos):
        """Determine the area name for a given position.

        Args:
            map: A string of the name of the map being considered
            pos: A list of x, y and z coordinates

        Return:
            A string of the name of the area that contains pos on map "map"
            If None can be found returns None
        """
        if None in pos:
            logging.debug("No area found for pos:")
            logging.debug(pos)
            return "No area found"
        closest_area = find_closest_area(map_name, pos)
        area_id = closest_area["areaId"]
        if area_id is None:
            logging.debug("No area found for pos:")
            logging.debug(map_name)
            logging.debug(pos)
            return None
        return NAV[map_name][area_id]["areaName"]

    def check_position(self, event, map_name):
        """Check whether the attacker and victim in an event were in the correct areas

        Checks the attacker and victim positions and side against the predefined positions for CTs and Ts

        Args:
            event: Dictionary containing information about the kill or damage event
            map_name: A string of the name of the map being considered

        Return:
            A boolean indicating whether or not the event is of relevance for the analysis
        """
        logging.debug("Checking Position")
        if not self.positions:
            return True
        attacker_area = self.get_area_from_pos(
            map_name, [event["attackerX"], event["attackerY"], event["attackerZ"]]
        )
        victim_area = self.get_area_from_pos(
            map_name, [event["victimX"], event["victimY"], event["victimZ"]]
        )
        logging.debug("AttackerArea: %s", attacker_area)
        logging.debug("AttackerSide: %s", event["attackerSide"])
        logging.debug("AttackerPositions: %s", self.positions[event["attackerSide"]])
        logging.debug("VictimArea: %s", victim_area)
        logging.debug("VictimSide: %s", event["victimSide"])
        logging.debug("VictimPositions: %s", self.positions[event["victimSide"]])
        logging.debug(
            "Attacker area matches: %s",
            attacker_area in self.positions[event["attackerSide"]],
        )
        logging.debug(
            "Victim area matches: %s",
            victim_area in self.positions[event["victimSide"]],
        )
        if (
            not self.positions[event["attackerSide"]]
            or attacker_area in self.positions[event["attackerSide"]]
        ) and (
            not self.positions[event["victimSide"]]
            or victim_area in self.positions[event["victimSide"]]
        ):
            return True
        return False

    def get_game_time(self, event, ticks):
        """Convert event tick to seconds since roundstart

        Args:
            event: Dictionary containing information about the kill or damage event
            ticks: Dictionary containing the tick at which the round started and tickrate of the server

        Return:
            An integer of the number of seconds since the round started
        """
        game_time = (event["tick"] - ticks["roundStartTick"]) / ticks["tickRate"]
        return game_time

    def check_weapons(self, current_round, event):
        """Checks whether both sides were using an allowed weapons.

        Goes through all frames that happend before the engagement and produces a list of weapons
        the victim had in inventory.
        Only consider engagements where the victim held at least on of the allowed weapons
        and the attacker used on of them.

        Args:
            current_round: A dictionary containing all the information about the round the event occured in.
            event: A dictionary containg all the information about the kill/damage event in question.

        Returns:
            A boolean whether the event is relevant to the analysis based on the used/held weapons.
        """
        logging.debug("Checking weapons!")
        if not self.weapons:
            return True
        weapons_set = set()
        for frame in current_round["frames"]:
            if frame["seconds"] > event["seconds"]:
                break
            else:
                for player in frame[event["victimSide"].lower()]["players"]:
                    if player["steamID"] == event["victimSteamID"]:
                        if (not player["isAlive"]) or player["inventory"] is None:
                            continue
                        for weapon in player["inventory"]:
                            weapons_set.add(weapon["weaponName"])
        logging.debug("Allowed weapons: %s", " ".join(self.weapons))
        logging.debug("Attacker weapon: %s", event["weapon"])
        logging.debug("Victim weapons: %s", " ".join(weapons_set))
        if event["weapon"] not in self.weapons:
            return False
        for weapon in weapons_set:
            if weapon in self.weapons:
                return True
        return False

    def summarize_round(
        self, event, game_time, current_round, results, match_id, map_name
    ):
        """Appends information of the round to the lists of the results dictionary

        Args:
            event: A dictionary containg all the information about the kill/damage event in question.
            game_time: An integer indicating how many seconds have passed since the start of the round
            current_round: A dictionary containing all the information about the round the event occured in.
            results: A dictionaory if lists holding the information of all relevant engagements
            match_id: A string containing the name of the demo file
            map_name: The name of the map that the current game was played on

        Returns:
            None (lists in dictionary are appended to in place)
        """
        results["Weapon"].append(event["weapon"])
        results["Round"].append(
            current_round["endTScore"] + current_round["endCTScore"]
        )
        results["WinnerSide"].append(event["attackerSide"])
        results["Time"].append(game_time)
        results["AttackerArea"].append(
            self.get_area_from_pos(
                map_name, [event["attackerX"], event["attackerY"], event["attackerZ"]]
            )
        )
        results["VictimArea"].append(
            self.get_area_from_pos(
                map_name, [event["victimX"], event["victimY"], event["victimZ"]]
            )
        )
        results["MatchID"].append(match_id)

    def initialize_results(self):
        """Initializes the results dictionary

        The dictionary contains the following keys:
        Weapons, Round, WinnerSide, Time, AttackerArea, VictimArea, MatchID

        Args:
            None

        Returns:
            Empty results dictionary
        """
        results = {}
        results["Weapon"] = []
        results["Round"] = []
        results["WinnerSide"] = []
        results["Time"] = []
        results["AttackerArea"] = []
        results["VictimArea"] = []
        results["MatchID"] = []
        return results

    def analyze_map(self, data, results, map_name):
        """Loop over all rounds and their events and add those that fullfill all criteria to the results dict.

        Args:
            data: Json object of the parsed demo file
            results: Dict of lists containing all relevant events
            map_name: String of the name of the map

        Returns:
            None (results is modified in place)
        """
        # Loop over rounds and each event in them and check if they fulfill all criteria
        match_id = data["matchID"]
        ticks = {}
        # Round tickrate to power of two that is equal or larger
        ticks["tickRate"] = 1 << (data["tickRate"] - 1).bit_length()
        for current_round in data["gameRounds"]:
            ticks["roundStartTick"] = current_round["startTick"]
            logging.debug("Round:")
            logging.debug(current_round)
            # Go through all kill events
            logging.debug("kills of that round:")
            logging.debug(current_round["kills"])
            if current_round["kills"] is None:
                logging.debug("Round does not have kills recorded")
                continue
            for event in current_round["kills"]:
                if (
                    event["victimSide"] not in self.positions
                    or event["attackerSide"] not in self.positions
                ):
                    # This happens either in POV demos or when a player kills themselves
                    continue
                game_time = self.get_game_time(event, ticks)
                # logging.info(game_time)
                # logging.info(self.times)
                if (
                    (not self.times or self.times[0] < game_time < self.times[1])
                    and self.check_position(event, map_name)
                    and self.check_weapons(current_round, event)
                ):
                    self.summarize_round(
                        event,
                        game_time,
                        current_round,
                        results,
                        match_id,
                        map_name,
                    )

    def analyze_demos(self):
        """Loops over the specified directory and analyzes each demo file.

        Builds a dictionary of lists containing information about each kill event.
        This information is written to a json file and returned in form of a dataframe.

        Args:
            None

        Returns:
            dataframe containing "Weapon", "Winner", "Round", "Time", "AttackerArea", "VictimArea", "MatchID" for each kill event
        """
        results = self.initialize_results()
        for filename in os.listdir(self.directory):
            if filename.endswith(".json"):
                logging.info("Working on file %s", filename)
                file_path = os.path.join(self.directory, filename)
                # checking if it is a file
                if os.path.isfile(file_path):
                    with open(file_path, encoding="utf-8") as demo_json:
                        demo_data = json.load(demo_json)
                    map_name = demo_data["mapName"]
                    self.analyze_map(demo_data, results, map_name)
                    self.n_analyzed += 1
        logging.info("Analyzed a total of %s demos!", self.n_analyzed)
        dataframe = pd.DataFrame(results)
        dataframe.to_json(self.json)
        return dataframe

    def calculate_CT_win_percentage(self, dataframe):
        """Calculates CT win percentage from CT and T kills

        Args:
            dataframe: dataframe containing winner side of each kill event

        Returns:
            Tuples consisting of total number of kills and CT win percentage.
            If not kills happend then return (0, 0)
        """
        CTWin = (dataframe.WinnerSide == "CT").sum()
        TWin = (dataframe.WinnerSide == "T").sum()
        if (CTWin + TWin) > 0:
            return (CTWin + TWin, round(100 * CTWin / (CTWin + TWin)))
        else:
            return (0, 0)

    def print_ct_win_percentage_for_time_cutoffs(self, dataframe):
        """Prints the total number of kills and CT win percentage for time slices starting from the lowest and always going to the lates

        Args:
            dataframe: A dataframe containing the time and winner of each kill event

        Returns:
            None (just prints)
        """
        # Todo: Also plot
        game_times = np.linspace(
            self.times[0], self.times[1], self.times[1] - self.times[0] + 1
        )

        logging.info("CTWinPercentages:")

        for time in game_times:
            time_allowed = dataframe["Time"] < time
            logging.info("Event times less than %s:", time)
            kills, ct_win_perc = self.calculate_CT_win_percentage(
                dataframe[time_allowed]
            )
            logging.info("CT Win: %s %% Total Kills: %s", ct_win_perc, kills)


def main(args):
    """Determine whether the early inferno mid fight on a buy round is favourable to the T or CT side."""
    parser = argparse.ArgumentParser("Analyze the early mid fight on inferno")
    parser.add_argument(
        "-d", "--debug", action="store_true", default=False, help="Enable debug output."
    )
    parser.add_argument(
        "-a",
        "--analyze",
        action="store_true",
        default=True,
        help="Reanalyze demos instead of reading from existing json file.",
    )
    parser.add_argument(
        "-j",
        "--json",
        default=r"D:\CSGO\Demos\Maps\inferno\Analysis\Inferno_kills_mid.json",
        help="Path of json containting preanalyzed results.",
    )
    parser.add_argument(
        "--starttime",
        type=int,
        default=5,
        help="Lower end of the clock time range that should be analyzed",
    )
    parser.add_argument(
        "--endtime",
        type=int,
        default=25,
        help="Upper end of the clock time range that should be analyzed",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=r"D:\CSGO\Demos\Maps\inferno",
        help="Directoy containting the demos to be analyzed.",
    )
    parser.add_argument(
        "-l",
        "--log",
        default=r"D:\CSGO\ML\CSGOML\FightAnalyzer.log",
        help="Path to output log.",
    )
    options = parser.parse_args(args)

    if options.debug:
        logging.basicConfig(
            filename=options.log, encoding="utf-8", level=logging.DEBUG, filemode="w"
        )
    else:
        logging.basicConfig(
            filename=options.log, encoding="utf-8", level=logging.INFO, filemode="w"
        )
    analyzer = FightAnalyzer(
        options.debug,
        times=[options.starttime, options.endtime],
        directory=options.dir,
        log=options.log,
    )
    if options.analyze:
        dataframe = analyzer.analyze_demos()
    else:
        with open(options.json, encoding="utf-8") as pre_analyzed:
            dataframe = pd.read_json(pre_analyzed)

    logging.info(dataframe)

    remove_Tramp = (
        (dataframe["WinnerSide"] == "T") & (dataframe["AttackerArea"] != "TRamp")
    ) | ((dataframe["WinnerSide"] == "CT") & (dataframe["VictimArea"] != "TRamp"))

    logging.info("With TRamp forbidden:")
    analyzer.print_ct_win_percentage_for_time_cutoffs(dataframe[remove_Tramp])
    logging.info("\nWith TRamp allowed:")
    analyzer.print_ct_win_percentage_for_time_cutoffs(dataframe)


if __name__ == "__main__":
    main(sys.argv[1:])
