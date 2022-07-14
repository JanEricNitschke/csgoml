"""Determine whether a given engagement is favourable to the T or CT side.

    Typical usage example:

    analyzer = FightAnalyzer(
    debug=False,
    times=[5, 25],
    directory="D:\\CSGO\\Demos\\Maps\\inferno",
    log="D:\\CSGO\\ML\\CSGOML\\logs\\FightAnalyzer.log",
    )
    dataframe = analyzer.analyze_demos()
    analyzer.print_ct_win_percentage_for_time_cutoffs(dataframe)
"""
#!/usr/bin/env python

from dataclasses import dataclass
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
        my_json=r"D:\CSGO\ML\CSGOML\Analysis\FightAnalyzer.json",
        directory=r"D:\CSGO\Demos\Maps",
        log=r"D:\CSGO\ML\CSGOML\logs\FightAnalyzer.log",
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

        self.directory = directory
        self.current_frame_index = 0
        self.n_analyzed = 0
        self.json = my_json
        # Pistols = [
        #    "CZ75 Auto",
        #    "Desert Eagle",
        #    "Dual Berettas",
        #    "Five-SeveN",
        #    "Glock-18",
        #    "P2000",
        #    "P250",
        #    "R8 Revolver",
        #    "Tec-9",
        #    "USP-S",
        # ]
        # Heavy = ["MAG-7", "Nova", "Sawed-Off", "XM1014", "M249", "Negev"]
        # SMG = ["MAC-10", "MP5-SD", "MP7", "MP9", "P90", "PP-Bizon", "UMP-45"]
        # Rifles = [
        #    "AK-47",
        #    "AUG",
        #    "FAMAS",
        #    "Galil AR",
        #    "M4A1",
        #    "M4A4",
        #    "SG 553",
        #    "AWP",
        #    "G3SG1",
        #    "SCAR-20",
        #    "SSG 08",
        # ]
        # Grenades = [
        #    "Smoke Grenade",
        #    "Flashbang",
        #    "HE Grenade",
        #    "Incendiary Grenade",
        #    "Molotov",
        #    "Decoy Grenade",
        # ]
        # Equipment = ["Knife", "Zeus x27"]
        # WeaponNames = Pistols + Heavy + SMG + Rifles + Grenades + Equipment
        # WeaponClasses = (
        #    ["Pistols"] * len(Pistols)
        #    + ["Heavy"] * len(Heavy)
        #    + ["SMG"] * len(SMG)
        #    + ["Rifle"] * len(Rifles)
        #    + ["Grenade"] * len(Grenades)
        #    + ["Equipment"] * len(Equipment)
        # )
        # self.weapon_df = pd.DataFrame(
        #    list(zip(WeaponNames, WeaponClasses)), columns=["Name", "Class"]
        # )
        # weapon_dict = self.weapon_df["Name"].to_dict()
        # self.weapon_dict = dict((v, k) for k, v in weapon_dict.items())

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
        """Grabs the CT and T areas of the event

        Args:
            event: Dictionary containing information about the kill or damage event
            map_name: A string of the name of the map being considered

        Return:
            A tuple of strings containing the area that the CT and T player were in when the event occured
        """
        logging.debug("Checking Position")
        attacker_area = self.get_area_from_pos(
            map_name, [event["attackerX"], event["attackerY"], event["attackerZ"]]
        )
        victim_area = self.get_area_from_pos(
            map_name, [event["victimX"], event["victimY"], event["victimZ"]]
        )
        if event["attackerSide"] == "CT" and event["victimSide"] == "T":
            return attacker_area, victim_area
        elif event["attackerSide"] == "T" and event["victimSide"] == "CT":
            return victim_area, attacker_area
        logging.debug(
            "Unknown sides: attackerSide %s; victimSide %s!. One should be 'CT' and the other should be 'T'.",
            event["attackerSide"],
            event["victimSide"],
        )
        return None, None

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
        """Grabs the attacker weapon and the weapons of the victim in the last frame before the event

        Args:
            current_round: A dictionary containing all the information about the round the event occured in.
            event: A dictionary containg all the information about the kill/damage event in question.

        Returns:
            A tuple of a string and list of tuples containing the attacker weapon and victim weapons
        """
        logging.debug("Checking weapons!")
        victim_weapons = set()
        while (
            self.current_frame_index < len(current_round["frames"])
            and current_round["frames"][self.current_frame_index]["tick"]
            <= event["tick"]
        ):
            for player in current_round["frames"][self.current_frame_index][
                event["victimSide"].lower()
            ]["players"]:
                if (
                    event["victimSteamID"]
                    and player["steamID"] == event["victimSteamID"]
                ) or (
                    (not event["victimSteamID"])
                    and event["victimName"] == player["name"]
                ):
                    if player["inventory"] is None:
                        self.current_frame_index += 1
                        continue
                    victim_weapons = {
                        weapon["weaponName"]
                        for weapon in player["inventory"]
                        if weapon["weaponClass"] != "Grenade"
                    }
            self.current_frame_index += 1
        self.current_frame_index -= 1

        logging.debug("Attacker weapon: %s", event["weapon"])
        logging.debug("Victim weapons: %s", " ".join(victim_weapons))
        return event["weapon"], victim_weapons

    def summarize_round(
        self,
        event,
        game_time,
        attacker_weapon,
        victim_weapons,
        CT_position,
        T_position,
        current_round,
        results,
        match_id,
        map_name,
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

        EventID = (
            match_id
            + "-"
            + str(current_round["endTScore"] + current_round["endCTScore"])
            + "-"
            + str(game_time)
        )
        results["EventID"].append(EventID)
        results["AttackerWeapon"].append(attacker_weapon)
        for weapon in victim_weapons:
            results["VictimWeapons"]["EventID"].append(EventID)
            results["VictimWeapons"]["VictimWeapon"].append(weapon)
        results["Round"].append(
            current_round["endTScore"] + current_round["endCTScore"]
        )
        results["CTWon"].append(int(event["attackerSide"] == "CT"))
        results["Time"].append(game_time)
        results["TArea"].append(T_position)
        results["CTArea"].append(CT_position)
        results["MatchID"].append(match_id)
        results["Pro"].append(False)
        results["MapName"].append(map_name)

    def initialize_results(self):
        """Initializes the results dictionary

        The dictionary contains the following keys:
        AttackerWeapon, VictimWeapons, Round, CTWon, Time, CTArea, TArea, MatchID, Pro, MapName

        Args:
            None

        Returns:
            Empty results dictionary
        """
        results = {}
        results["EventID"] = []
        results["MatchID"] = []
        results["Round"] = []
        results["Pro"] = []
        results["MapName"] = []
        results["Time"] = []
        results["CTWon"] = []
        results["CTArea"] = []
        results["TArea"] = []
        results["AttackerWeapon"] = []
        results["VictimWeapons"] = {"EventID": [], "VictimWeapon": []}
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
            self.current_frame_index = 0
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
                game_time = self.get_game_time(event, ticks)
                CT_position, T_position = self.check_position(event, map_name)
                if not CT_position or not T_position:
                    continue
                attacker_weapon, victim_weapons = self.check_weapons(
                    current_round, event
                )
                self.summarize_round(
                    event,
                    game_time,
                    attacker_weapon,
                    victim_weapons,
                    CT_position,
                    T_position,
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
        for directory in os.listdir(self.directory):
            if "de_" + directory not in NAV and directory not in NAV:
                logging.info(
                    "Map %s|%s is not in supplied navigation files. Skipping it.",
                    "de_" + directory,
                    directory,
                )
                continue
            directory = os.path.join(self.directory, directory)
            logging.info("Scanning directory: %s", directory)
            for filename in os.listdir(directory):
                if filename.endswith(".json"):
                    logging.debug("Working on file %s", filename)
                    file_path = os.path.join(directory, filename)
                    # checking if it is a file
                    if os.path.isfile(file_path):
                        with open(file_path, encoding="utf-8") as demo_json:
                            demo_data = json.load(demo_json)
                        map_name = demo_data["mapName"]
                        self.analyze_map(demo_data, results, map_name)
                        self.n_analyzed += 1
        logging.info("Analyzed a total of %s demos!", self.n_analyzed)
        event_weapon_mapping = pd.DataFrame(results["VictimWeapons"])
        del results["VictimWeapons"]
        dataframe = pd.DataFrame(results)
        dict_dataframe = dataframe.to_dict()
        dict_event_weapon_mapping = event_weapon_mapping.to_dict()
        # dict_weapon_df = self.weapon_df.to_dict()
        result_dict = {
            "Events": dict_dataframe,
            "Mapping": dict_event_weapon_mapping,
            # "Weapons": dict_weapon_df,
        }
        dataframe.to_json(self.json)
        with open(self.json, "w", encoding="utf-8") as outfile:
            json.dump(result_dict, outfile)
        return dataframe, event_weapon_mapping

    def calculate_CT_win_percentage(self, dataframe):
        """Calculates CT win percentage from CT and T kills

        Args:
            dataframe: dataframe containing winner side of each kill event

        Returns:
            Tuples consisting of total number of kills and CT win percentage.
            If not kills happend then return (0, 0)
        """
        logging.debug(dataframe)
        if len(dataframe) > 0:
            return len(dataframe), round(100 * dataframe["CTWon"].mean())
        return 0, 0

    def print_ct_win_percentage_for_time_cutoffs(self, dataframe, times):
        """Prints the total number of kills and CT win percentage for time slices starting from the lowest and always going to the lates

        Args:
            dataframe: A dataframe containing the time and winner of each kill event

        Returns:
            None (just prints)
        """
        # Todo: Also plot
        game_times = np.linspace(times[0], times[1], times[1] - times[0] + 1)

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
        default=False,
        help="Reanalyze demos instead of reading from existing json file.",
    )
    parser.add_argument(
        "-j",
        "--json",
        default=r"D:\CSGO\ML\CSGOML\Analysis\FightAnalyzer.json",
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
        default=r"D:\CSGO\Demos\Maps",
        help="Directoy containting the demos to be analyzed.",
    )
    parser.add_argument(
        "-l",
        "--log",
        default=r"D:\CSGO\ML\CSGOML\logs\FightAnalyzer.log",
        help="Path to output log.",
    )
    parser.add_argument(
        "-m",
        "--map",
        default="de_inferno",
        help="Path to output log.",
    )
    options = parser.parse_args(args)

    # if options.debug:
    #    logging.basicConfig(
    #        filename=options.log, encoding="utf-8", level=logging.DEBUG, filemode="w"
    #    )
    # else:
    #    logging.basicConfig(
    #        filename=options.log, encoding="utf-8", level=logging.INFO, filemode="w"
    #    )
    times = [options.starttime, options.endtime]
    positions = {"CT": {"TopofMid", "Middle"}, "T": {"Middle", "TRamp"}}
    weapons = {
        "Attacker": {
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
        },
        "Victim": {
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
        },
    }
    analyzer = FightAnalyzer(
        debug=options.debug,
        directory=options.dir,
        log=options.log,
        my_json=options.json,
    )
    if options.analyze:
        dataframe, mapping_df = analyzer.analyze_demos()
    else:
        with open(options.json, encoding="utf-8") as pre_analyzed:
            data = json.load(pre_analyzed)
        dataframe = pd.DataFrame(data["Events"])
        mapping_df = pd.DataFrame(data["Mapping"])
        # weapons_df = pd.DataFrame(data["Weapons"])

    logging.info(dataframe)
    logging.info(mapping_df)
    # logging.info(weapons_df)

    # SELECT AVG(d.CTWon)
    # FROM (
    #  SELECT DISTINCT e.ID, e.CTWon
    # FROM dataframe d JOIN VictimWeapons vw
    # ON e.EventID = vw.EventID
    # WHERE d.MapName = 'options.map'
    # AND d.CTArea in positions["CT"]
    # AND d.TArea in positions["T"]
    # AND d.AttackerWeapon in weapons["Attacker"]
    # AND vw.VictimWeapon in weapons["Victim"] ) t
    # ---
    # SELECT AVG(value1)
    # FROM (
    # SELECT DISTINCT e.ID, e.value1
    # FROM events e JOIN MtoM m
    # ON e.ID = m.EventID
    # WHERE e.value2 IN (1,3) AND m.value3 IN (0,2,3,4,7,8)
    # ) t
    dataframe = dataframe[dataframe["MapName"] == options.map]
    dataframe = dataframe[dataframe["CTArea"].isin(positions["CT"])]
    dataframe = dataframe[dataframe["TArea"].isin(positions["T"])]
    dataframe = dataframe[dataframe["AttackerWeapon"].isin(weapons["Attacker"])]
    dataframe = dataframe[
        dataframe["EventID"].apply(
            lambda x: mapping_df.loc[mapping_df["EventID"] == x]
            .loc[:, "VictimWeapon"]
            .isin(weapons["Victim"])
            .any()
        )
    ]
    remove_Tramp = dataframe["TArea"] != "TRamp"

    logging.info("With TRamp forbidden:")
    analyzer.print_ct_win_percentage_for_time_cutoffs(dataframe[remove_Tramp], times)
    logging.info("\nWith TRamp allowed:")
    analyzer.print_ct_win_percentage_for_time_cutoffs(dataframe, times)


if __name__ == "__main__":
    main(sys.argv[1:])
