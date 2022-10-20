"""Determine whether a given engagement is favourable to the T or CT side.

    Typical usage example:

    analyzer = FightAnalyzer(
        connection,
        cursor,
        options.dirs,
        debug=options.debug,
        log=options.log,
    )
    analyzer.analyze_demos()
    analyzer.print_ct_win_percentage_for_time_cutoffs(times, positions, weapons, classes, use_weapons_classes, options.map)
"""
#!/usr/bin/env python
# pylint: disable=invalid-name
import os
import argparse
import sys
import math
import json
import logging
from typing import Optional, Union
import numpy as np
import boto3
from awpy.analytics.nav import find_closest_area
from awpy.data import NAV
import pymysql


class FightAnalyzer:
    """Determine whether a given engagement is favourable to the T or CT side.

    Attributes:
        directories (list[str]): A list of paths to the directories where the demos to be analyzed reside.
        connection: Connection to a mysql database
        cursor: A mysql cursor to the connection
        n_analyzed (int): An integer of the number of demos that have been analyzed
        current_frame_index (int): An integer of the index of the current frame in the current round
    """

    def __init__(
        self,
        connection: pymysql.connections.Connection,
        cursor: pymysql.cursors.Cursor,
        directories: Optional[list[str]] = None,
        debug: bool = False,
        log: str = r"D:\CSGO\ML\CSGOML\logs\FightAnalyzer.log",
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

        if directories is None:
            self.directories = [
                r"E:\PhD\MachineLearning\CSGOData\ParsedDemos",
                r"D:\CSGO\Demos\Maps",
            ]
        else:
            self.directories = directories
        self.current_frame_index = 0
        self.n_analyzed = 0
        self.cursor = cursor
        self.connection = connection

    def get_area_from_pos(self, map_name: str, pos: list[float]) -> Optional[str]:
        """Determine the area name for a given position.

        Args:
            map (str): A string of the name of the map being considered
            pos (list[float]): A list of x, y and z coordinates

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

    def check_position(
        self, event: dict, map_name: str
    ) -> Union[tuple[str, str], tuple[None, None]]:
        """Grabs the CT and T areas of the event

        Args:
            event (dict): Dictionary containing information about the kill or damage event
            map_name (str): A string of the name of the map being considered

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

    def get_game_time(self, event: dict, ticks: dict) -> int:
        """Convert event tick to seconds since roundstart

        Args:
            event (dict): Dictionary containing information about the kill or damage event
            ticks (dict): Dictionary containing the tick at which the round started and tickrate of the server

        Return:
            An integer of the number of seconds since the round started
        """
        game_time = (event["tick"] - ticks["freezeTimeEndTick"]) / ticks["tickRate"]
        return game_time

    def check_weapons(
        self, current_round: dict, event: dict
    ) -> tuple[str, list[str], list[str]]:
        """Grabs the attacker weapon and the weapons of the victim in the last frame before the event

        Args:
            current_round (dict): A dictionary containing all the information about the round the event occured in.
            event (dict): A dictionary containg all the information about the kill/damage event in question.

        Returns:
            A tuple of a string and list of tuples containing the attacker weapon and victim weapons
        """
        logging.debug("Checking weapons!")
        victim_weapons = set()
        attacker_weapons = set()
        while (
            self.current_frame_index < len(current_round["frames"])
            and current_round["frames"][self.current_frame_index]["tick"]
            <= event["tick"]
        ):
            if (
                current_round["frames"][self.current_frame_index][
                    event["victimSide"].lower()
                ]["players"]
                is not None
            ):
                for player in current_round["frames"][self.current_frame_index][
                    event["victimSide"].lower()
                ]["players"]:
                    if (
                        event["victimSteamID"]
                        and player["steamID"] == event["victimSteamID"]
                    ):
                        if player["inventory"] is None:
                            continue
                        victim_weapons = {
                            weapon["weaponName"] for weapon in player["inventory"]
                        }
            if (
                current_round["frames"][self.current_frame_index][
                    event["attackerSide"].lower()
                ]["players"]
                is not None
            ):
                for player in current_round["frames"][self.current_frame_index][
                    event["attackerSide"].lower()
                ]["players"]:
                    if (
                        event["attackerSteamID"]
                        and player["steamID"] == event["attackerSteamID"]
                    ):
                        if player["inventory"] is None:
                            continue
                        attacker_weapons = {
                            weapon["weaponName"] for weapon in player["inventory"]
                        }
            self.current_frame_index += 1
        self.current_frame_index -= 1
        if event["attackerSide"] == "T":
            CT_weapons, T_weapons = victim_weapons, attacker_weapons
        elif event["attackerSide"] == "CT":
            CT_weapons, T_weapons = attacker_weapons, victim_weapons
        logging.debug("Attacker weapon: %s", event["weapon"])
        logging.debug("Victim weapons: %s", " ".join(victim_weapons))
        return event["weapon"], CT_weapons, T_weapons

    def summarize_round(
        self,
        event: dict,
        game_time: int,
        kill_weapon: str,
        CT_weapons: list[str],
        T_weapons: list[str],
        CT_position: str,
        T_position: str,
        current_round: dict,
        match_id: str,
        map_name: str,
        is_pro_game: bool,
    ) -> None:
        """Appends information of the round to MYSQL database

        Args:
            event (dict): A dictionary containg all the information about the kill/damage event in question.
            game_time (int): An integer indicating how many seconds have passed since the start of the round
            kill_weapon (str): A string of the weapon used by the attacker of the event
            CT_weapons (list[str]): A list of strings of the weapons the CT of the event had in their inventory
            T_weapons (list[str]): A list of strings of the weapons the T of the event had in their inventory
            CT_position (str): A string of the position of the CT in the event
            T_position (str): A string of the position of the T in the event
            current_round (dict): A dictionary containing all the information about the round the event occured in.
            match_id (str): A string containing the name of the demo file
            map_name (str): The name of the map that the current game was played on
            is_pro_game (bool): A boolean indicating whether the current round is from a pro game or not

        Returns:
            None (DB is appended to in place)
        """
        sql = "INSERT INTO Events (MatchID, Round, Pro, MapName, Time, CTWon, CTArea, TArea, KillWeapon) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
        val = (
            match_id,
            current_round["endTScore"] + current_round["endCTScore"],
            int(is_pro_game),
            map_name,
            game_time,
            int(event["attackerSide"] == "CT"),
            CT_position,
            T_position,
            kill_weapon,
        )
        logging.debug("sql: %s, val:%s", sql, *val)
        self.cursor.execute(sql, val)
        event_id = self.cursor.lastrowid
        for weapon in CT_weapons:
            sql = "INSERT INTO CTWeapons (EventID, CTWeapon) VALUES (%s, %s)"
            val = (event_id, weapon)
            logging.debug("sql: %s, val:%s", sql, *val)
            self.cursor.execute(sql, val)
        for weapon in T_weapons:
            sql = "INSERT INTO TWeapons (EventID, TWeapon) VALUES (%s, %s)"
            val = (event_id, weapon)
            logging.debug("sql: %s, val:%s", sql, *val)
            self.cursor.execute(sql, val)

    def analyze_map(
        self, data: dict, map_name: str, match_id: str, is_pro_game: bool
    ) -> None:
        """Loop over all rounds and their events and add them to a MYSQL database.

        Args:
            data (json/dict): Json object of the parsed demo file
            map_name (str): String of the name of the map
            match_id (str): String of the demo file name
            is_pro_game (bool): A boolean indicating whether the current map is from a pro game

        Returns:
            None (DB is modified in place)
        """
        # Loop over rounds and each event in them and check if they fulfill all criteria
        ticks = {}
        # Round tickrate to power of two that is equal or larger
        ticks["tickRate"] = 1 << (data["tickRate"] - 1).bit_length()
        for current_round in data["gameRounds"]:
            self.current_frame_index = 0
            ticks["roundStartTick"] = current_round["startTick"]
            logging.debug("Round:")
            logging.debug(current_round)
            # Go through all kill events
            # logging.debug("kills of that round:")
            # logging.debug(current_round["kills"])
            if current_round["kills"] is None:
                logging.debug("Round does not have kills recorded")
                continue
            for event in current_round["kills"]:
                if not event["victimSteamID"] or not event["attackerSteamID"]:
                    continue
                game_time = self.get_game_time(event, ticks)
                CT_position, T_position = self.check_position(event, map_name)
                if not CT_position or not T_position:
                    continue
                kill_weapon, CT_weapons, T_weapons = self.check_weapons(
                    current_round, event
                )
                self.summarize_round(
                    event,
                    game_time,
                    kill_weapon,
                    CT_weapons,
                    T_weapons,
                    CT_position,
                    T_position,
                    current_round,
                    match_id,
                    map_name,
                    is_pro_game,
                )

    def check_exists(self, match_id: str) -> bool:
        """Checks if the MYSQL DB already contains events from this match
        Args:
            match_id (str): String of the demo file name

        Returns:
            A boolean indicating whether there are already events from this match in the DB
        """
        sql = "SELECT COUNT(e.MatchID) FROM Events e WHERE e.MatchID = %s"
        logging.debug(sql)
        self.cursor.execute(sql, (match_id,))
        result = list(self.cursor.fetchone())
        logging.debug(result)
        return int(result[0]) > 0

    def analyze_demos(self) -> None:
        """Loops over the specified directory and analyzes each demo file.

        Collects information about each kill event.
        This information is written to a MYSQL DB

        Args:
            None

        Returns:
            None (Modifies external MYSQL DB)
        """
        sql = "SELECT DISTINCT e.MatchID FROM Events e"
        self.cursor.execute(sql)
        done_set = set()
        for x in self.cursor.fetchall():
            done_set.add(x[0])
        for maps_dir in self.directories:
            for map_dir in os.listdir(maps_dir):
                if "de_" + map_dir not in NAV and map_dir not in NAV:
                    logging.info(
                        "Map %s|%s is not in supplied navigation files. Skipping it.",
                        "de_" + map_dir,
                        map_dir,
                    )
                    continue
                map_dir = os.path.join(maps_dir, map_dir)
                logging.info("Scanning directory: %s", map_dir)
                for filename in os.listdir(map_dir):
                    if filename.endswith(".json"):
                        logging.debug("Working on file %s", filename)
                        file_path = os.path.join(map_dir, filename)
                        # checking if it is a file
                        if os.path.isfile(file_path):
                            with open(file_path, encoding="utf-8") as demo_json:
                                demo_data = json.load(demo_json)
                            match_id = os.path.splitext(filename)[0]
                            if match_id in done_set:
                                logging.debug(
                                    "There are already events from the file with the matchid %s in the database. Skipping it.",
                                )
                                continue
                            logging.info(match_id)
                            is_pro_game = maps_dir != r"D:\CSGO\Demos\Maps"
                            map_name = demo_data["mapName"]
                            self.analyze_map(demo_data, map_name, match_id, is_pro_game)
                            self.n_analyzed += 1
                self.connection.commit()
        logging.info("Analyzed a total of %s demos!", self.n_analyzed)

    def calculate_CT_win_percentage(
        self,
        times: list[float],
        positions: dict,
        weapons: dict,
        classes: dict,
        use_weapons_classes: dict,
        map_name: str,
    ) -> tuple[float, float]:
        """Calculates CT win percentage from CT and T kills

        Queries information from a database to determine CT win percentage of events fitting the criteria

        Args:
            times (list[float]): A list of two floats indicating between which times the event should have occured
            positions (dict): A dicitionary of positions that are allowed/forbidden for each side of an event
            weapons (dict): A dictionary of weapons that are allowed/forbidden for each side of an event
            classes (dict): A dictionary of weapon classes that are allowed/forbidden for each side of an event
            use_weapons_classes (dict): A dictionary determining if weapons or classes should be used for each side
            map_name (str): A string of the map that should be used for the query

        Returns:
            Tuples consisting of total number of kills and CT win percentage.
            If no kills happend then return (0, 0)
        """
        ct_pos = ", ".join(f'"{val}"' for val in positions["CT"]["Allowed"])
        t_pos = ", ".join(f'"{val}"' for val in positions["T"]["Allowed"])
        T_weapon = ", ".join(f'"{val}"' for val in weapons["T"]["Allowed"])
        not_T_weapon = ", ".join(f'"{val}"' for val in weapons["T"]["Forbidden"])
        CT_weapon = ", ".join(f'"{val}"' for val in weapons["CT"]["Allowed"])
        not_CT_weapon = ", ".join(f'"{val}"' for val in weapons["CT"]["Forbidden"])
        Kill_weapon = ", ".join(f'"{val}"' for val in weapons["Kill"])
        # not_ct_pos = ", ".join(f'"{val}"' for val in positions["CT"]["Forbidden"])
        # not_t_pos = ", ".join(f'"{val}"' for val in positions["T"]["Forbidden"])

        CT_classes = ", ".join(f'"{val}"' for val in classes["CT"]["Allowed"])
        T_classes = ", ".join(f'"{val}"' for val in classes["T"]["Allowed"])
        Kill_classes = ", ".join(f'"{val}"' for val in classes["Kill"])
        not_CT_classes = ", ".join(f'"{val}"' for val in classes["CT"]["Forbidden"])
        not_T_classes = ", ".join(f'"{val}"' for val in classes["T"]["Forbidden"])

        # not sql injection save, but this only gets executed with my own input, so it should be fine.
        # The actual lambda function is injection save
        sql = (
            f"""SELECT AVG(t.CTWon), COUNT(t.CTWon) """
            f"""FROM ( """
            f"""SELECT DISTINCT e.EventID, e.CTWon """
            f"""FROM Events e JOIN CTWeapons ctw """
            f"""ON e.EventID = ctw.EventID """
            f"""JOIN TWeapons tw """
            f"""ON e.EventID = tw.EventID """
            f"""JOIN WeaponClasses wcct """
            f"""ON ctw.CTWeapon = wcct.WeaponName """
            f"""JOIN WeaponClasses wct """
            f"""ON tw.TWeapon = wct.WeaponName """
            f"""JOIN WeaponClasses wck """
            f"""ON e.KillWeapon = wck.WeaponName """
            f"""WHERE e.MapName = '{map_name}' """
            f"""AND e.Time BETWEEN {times[0]} AND {times[1]} """
        )
        if ct_pos != "":
            sql += f"""AND e.CTArea in ({ct_pos}) """
        if t_pos != "":
            sql += f"""AND e.TArea in ({t_pos}) """
        # if not_ct_pos != "":
        #     sql += f"""AND e.CTArea NOT in ({not_ct_pos}) """
        # if not_t_pos != "":
        #     sql += f"""AND e.TArea NOT in ({not_t_pos}) """

        if use_weapons_classes["CT"] == "weapons":
            if CT_weapon != "":
                sql += f"""AND ctw.CTWeapon in ({CT_weapon}) """
            if not_CT_weapon != "":
                sql += f"""AND ctw.CTWeapon NOT in ({not_CT_weapon}) """
        elif use_weapons_classes["CT"] == "classes":
            if CT_classes != "":
                sql += f"""AND wcct.Class in ({CT_classes}) """
            if not_CT_classes != "":
                sql += f"""AND wcct.Class NOT in ({not_CT_classes}) """

        if use_weapons_classes["T"] == "weapons":
            if T_weapon != "":
                sql += f"""AND tw.TWeapon in ({T_weapon}) """
            if not_T_weapon != "":
                sql += f"""AND tw.TWeapon NOT in ({not_T_weapon}) """
        elif use_weapons_classes["T"] == "classes":
            if T_classes != "":
                sql += f"""AND wct.Class in ({T_classes}) """
            if not_T_classes != "":
                sql += f"""AND wct.Class NOT in ({not_T_classes}) """

        if use_weapons_classes["Kill"] == "weapons":
            if Kill_weapon != "":
                sql += f"""AND e.KillWeapon in ({Kill_weapon}) """
        elif use_weapons_classes["Kill"] == "classes":
            if Kill_classes != "":
                sql += f"""AND wck.Class in ({Kill_classes}) """

        sql += """) t"""

        logging.info(sql)
        self.cursor.execute(sql)
        result = list(self.cursor.fetchone())

        if result[1] > 0:
            return result[1], round(100 * result[0])
        return 0, 0

    def print_ct_win_percentage_for_time_cutoffs(
        self,
        edge_times: list[float],
        positions: dict,
        weapons: dict,
        classes: dict,
        use_weapons_classes: dict,
        map_name: str,
    ) -> None:
        """Prints the total number of kills and CT win percentage for time slices starting from the lowest and always going to the lates

         Args:
            edge_times (list[float]): A list of two floats determining the times that should be considered for the event
            positions (dict): A dicitionary of positions that are allowed/forbidden for each side of an event
            weapons (dict): A dictionary of weapons that are allowed/forbidden for each side of an event
            classes (dict): A dictionary of weapon classes that are allowed/forbidden for each side of an event
            use_weapons_classes (dict): A dictionary determining if weapons or classes should be used for each side
            map_name (str): A string of the map that should be used for the query

        Returns:
            None (just prints)
        """
        if edge_times[0] == 0 and edge_times[1] == 175:
            game_times = [edge_times]
        elif edge_times[0] == 0:
            game_times = zip(
                [0] * (edge_times[1] - edge_times[0]),
                np.linspace(1, edge_times[1], edge_times[1] - edge_times[0]),
            )
        elif edge_times[1] == 175:
            game_times = zip(
                np.linspace(edge_times[0], 174, edge_times[1] - edge_times[0]),
                [175] * (edge_times[1] - edge_times[0]),
            )
        else:
            mean = (edge_times[1] + edge_times[0]) / 2
            diff = edge_times[1] - edge_times[0]
            if diff % 2:
                game_times = zip(
                    np.linspace(edge_times[0], math.floor(mean), math.ceil(diff / 2)),
                    np.linspace(edge_times[1], math.ceil(mean), math.ceil(diff / 2)),
                )
            else:
                game_times = zip(
                    np.linspace(edge_times[0], int(mean - 1), int(diff / 2)),
                    np.linspace(edge_times[1], int(mean + 1), int(diff / 2)),
                )
        logging.info("CTWinPercentages:")
        for times in game_times:
            logging.info("Event times less than %s:", times)
            kills, ct_win_perc = self.calculate_CT_win_percentage(
                times, positions, weapons, classes, use_weapons_classes, map_name
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
        "--starttime",
        type=int,
        default=0,
        help="Lower end of the clock time range that should be analyzed",
    )
    parser.add_argument(
        "--endtime",
        type=int,
        default=25,
        help="Upper end of the clock time range that should be analyzed",
    )
    parser.add_argument(
        "--dirs",
        nargs="*",
        default=[
            # r"E:\PhD\MachineLearning\CSGOData\ParsedDemos",
            r"D:\CSGO\Demos\Maps",
        ],
        help="All the directories that should be scanned for demos.",
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

    times = [options.starttime, options.endtime]
    positions = {
        "CT": {"Allowed": {"TopofMid", "Middle"}, "Forbidden": {}},
        "T": {"Allowed": {"Middle", "TRamp"}, "Forbidden": {}},
    }
    use_weapons_classes = {"CT": "classes", "T": "classes", "Kill": "classes"}
    weapons = {
        "CT": {
            "Allowed": {
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
                "AUG",
            },
            "Forbidden": {},
        },
        "Kill": [
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
            "AUG",
        ],
        "T": {
            "Allowed": {
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
                "AUG",
            },
            "Forbidden": {},
        },
    }

    classes = {
        "CT": {
            "Allowed": {
                "Rifle",
                "Heavy",
            },
            "Forbidden": {},
        },
        "T": {
            "Allowed": {
                "Rifle",
                "Heavy",
            },
            "Forbidden": {},
        },
        "Kill": [
            "Rifle",
            "Heavy",
        ],
    }

    host = "fightanalyzer.ctox3zthjpph.eu-central-1.rds.amazonaws.com"
    user = "IAM_USER"
    database = "FightAnalyzerDB"
    port = 3306
    region = "eu-central-1"
    os.environ["LIBMYSQL_ENABLE_CLEARTEXT_PLUGIN"] = "1"
    session = boto3.Session()
    client = session.client("rds")
    token = client.generate_db_auth_token(
        DBHostname=host, Port=port, DBUsername=user, Region=region
    )
    connection = pymysql.connect(
        host=host,
        user=user,
        password=token,
        database=database,
        ssl_ca=r"D:\\CSGO\\ML\\CSGOML\AWS_Steps\\Certs\\global-bundle.pem",
    )

    cursor = connection.cursor()

    analyzer = FightAnalyzer(
        connection,
        cursor,
        directories=options.dirs,
        debug=options.debug,
        log=options.log,
    )

    if options.analyze:
        analyzer.analyze_demos()

    logging.info("With TRamp allowed:")
    analyzer.print_ct_win_percentage_for_time_cutoffs(
        times, positions, weapons, classes, use_weapons_classes, options.map
    )

    cursor.close()
    connection.close()


if __name__ == "__main__":
    main(sys.argv[1:])
