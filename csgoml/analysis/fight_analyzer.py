#!/usr/bin/env python
"""Determine whether a given engagement is favourable to the T or CT side.

Example::

    analyzer = FightAnalyzer(
        connection,
        cursor,
        options.dirs,
        debug=options.debug,
        log=options.log,
    )
    analyzer.analyze_demos()
    analyzer.calculate_ct_win_percentage(
        event,
        cursor,
    )
"""

import argparse
import json
import logging
import math
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import boto3
import pymysql
from awpy.analytics.nav import find_closest_area
from awpy.analytics.stats import lower_side
from awpy.data import NAV
from awpy.types import Game, GameFrame, GameRound, KillAction, PlayerInfo
from pymysql.cursors import Cursor

from csgoml.types import (
    FightSpecification,
    PositionSpecification,
    QueryResult,
    TickInformation,
)


class FightAnalyzer:
    """Determine whether a given engagement is favourable to the T or CT side.

    Attributes:
        directories (list[str]): A list of paths to the directories where
            the demos to be analyzed reside.
        connection: Connection to a mysql database
        cursor: A mysql cursor to the connection
        n_analyzed (int): An integer of the number of demos that have been analyzed
        current_frame_index (int): An integer of the index
            of the current frame in the current round
    """

    def __init__(
        self,
        connection: pymysql.connections.Connection,
        cursor: Cursor,
        directories: list[str] | None = None,
    ) -> None:
        """Initialize the class.

        Args:
            connection (pymysql.connections.Connection): Connection a a database
            cursor (Cursor): Cursor for that connection.
            directories (list[str] | None, optional): Directories to check.
                Defaults to None.
        """
        if directories is None:
            self.directories: list[str] = [
                r"E:\PhD\MachineLearning\CSGOData\ParsedDemos",
                r"D:\CSGO\Demos\Maps",
            ]
        else:
            self.directories = directories
        self.current_frame_index: int = 0
        self.n_analyzed: int = 0
        self.cursor: Cursor = cursor
        self.connection: pymysql.connections.Connection = connection

    def get_area_from_pos(self, map_name: str, pos: list[float | None]) -> str | None:
        """Determine the area name for a given position.

        Args:
            map_name (str): A string of the name of the map being considered
            pos (list[Optional[float]]): A list of x, y and z coordinates

        Return:
            A string of the name of the area that contains pos on map "map"
            If None can be found returns None
        """
        if None in pos:
            logging.debug("No area found for pos:")
            logging.debug(pos)
            return None
        closest_area = find_closest_area(map_name, pos)
        area_id = closest_area["areaId"]
        return NAV[map_name][area_id]["areaName"]

    def check_position(
        self, event: KillAction, map_name: str
    ) -> tuple[str | None, str | None]:
        """Grabs the CT and T areas of the event.

        Args:
            event (dict): Dictionary containing information about the kill event
            map_name (str): A string of the name of the map being considered

        Return:
            A tuple of strings containing the area that the
            CT and T player were in when the event occured
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
        if event["attackerSide"] == "T" and event["victimSide"] == "CT":
            return victim_area, attacker_area
        logging.debug(
            "Unknown sides: attackerSide %s; victimSide %s!. "
            "One should be 'CT' and the other should be 'T'.",
            event["attackerSide"],
            event["victimSide"],
        )
        return None, None

    def get_game_time(self, event: KillAction, ticks: TickInformation) -> int:
        """Convert event tick to seconds since roundstart.

        Args:
            event (KillAction): Dictionary containing information about the kill event
            ticks (TickInformation): Dictionary containing the tick at
                which the round started and tickrate of the server

        Return:
            An integer of the number of seconds since the round started
        """
        return (event["tick"] - ticks["freezeTimeEndTick"]) / ticks["tickRate"]

    def _update_player_weapons(
        self, players: list[PlayerInfo], player_steamid: int | None
    ) -> set[str] | None:
        """Update the a weapons set.

        Loop through all the players and if there is one that matches the attackerid
        and they have an inventory then update the attacker weapons set with their
        inventory.

        If no matching player is found or they have no inventory then return None.

        Args:
            players (list[PlayerInfo]): List of players to check.
            player_steamid (int | None): SteamID of the attacker.

        Returns:
            set[str] | None: Set of weapons the player had in their inventory
                if a match was found. `None` otherwise.
        """
        weapons: set[str] | None = None
        for player in players:
            if (
                player_steamid
                and player["steamID"] == player_steamid
                and player["inventory"] is not None
            ):
                weapons = {weapon["weaponName"] for weapon in player["inventory"]}
        return weapons

    def check_weapons(
        self, round_frames: list[GameFrame], event: KillAction
    ) -> tuple[str, set[str], set[str]] | None:
        """Grabs the weapons of attacker and victim in the last frame before the event.

        Args:
            round_frames (list[GameFrame]): A dictionary containing all the information
                about the round the event occured in.
            event (KillAction): A dictionary containg all the information about
                the kill/damage event in question.

        Returns:
            A tuple of a string (killing weapon) and two sets
                containing the attacker weapons and victim weapons
        """
        logging.debug("Checking weapons!")
        victim_weapons: set[str] = set()
        attacker_weapons: set[str] = set()
        if (victim_side := event["victimSide"]) not in ("CT", "T") or (
            attacker_side := event["attackerSide"]
        ) not in (
            "CT",
            "T",
        ):
            return None
        # Loop as long as the current frames is before our
        # event of interest and before the last frame
        while (
            self.current_frame_index < len(round_frames or [])
            and round_frames[self.current_frame_index]["tick"] <= event["tick"]
        ):
            # Update weapons of an inventory for the relevant player can be found
            # in this frame.
            victim_weapons = (
                self._update_player_weapons(
                    round_frames[self.current_frame_index][lower_side(victim_side)][
                        "players"
                    ]
                    or [],
                    event["victimSteamID"],
                )
                or victim_weapons
            )
            attacker_weapons = (
                self._update_player_weapons(
                    round_frames[self.current_frame_index][lower_side(attacker_side)][
                        "players"
                    ]
                    or [],
                    event["attackerSteamID"],
                )
                or attacker_weapons
            )
            # Iterate through the frames
            self.current_frame_index += 1
        # Step back one frame because we can have
        # multiple events associated with one frame
        self.current_frame_index -= 1
        # Assign CT and T weapons based on victim and attacker weapons
        if event["attackerSide"] == "T":
            ct_weapons, t_weapons = victim_weapons, attacker_weapons
        else:  # event["attackerSide"] == "CT":
            ct_weapons, t_weapons = attacker_weapons, victim_weapons
        logging.debug("Attacker weapon: %s", event["weapon"])
        logging.debug("Victim weapons: %s", " ".join(victim_weapons))
        return event["weapon"], ct_weapons, t_weapons

    def summarize_round(
        self,
        event: KillAction,
        game_time: int,
        kill_weapon: str,
        ct_weapons: set[str],
        t_weapons: set[str],
        ct_position: str,
        t_position: str,
        current_round: GameRound,
        match_id: str,
        map_name: str,
        *,
        is_pro_game: bool = False,
    ) -> None:
        """Appends information of the round to MYSQL database.

        Args:
            event (dict): AFnformationabout the kill/damage event in question.
            game_time (int): Wow many seconds have passed since the start of the round
            kill_weapon (str):Weapon used by the attacker of the event
            ct_weapons (set[str]): Strings of the weapons the CT
                of the event had in their inventory
            t_weapons (set[str]): Strings of the weapons the T
                of the event had in their inventory
            ct_position (str): A string of the position of the CT in the event
            t_position (str): A string of the position of the T in the event
            current_round (dict):Information about the round the event occured in.
            match_id (str): Name of the demo file
            map_name (str): Name of the map that the current game was played on
            is_pro_game (bool): Whether the current round is from a pro game or not

        Returns:
            None (DB is appended to in place)
        """
        val_event = (
            match_id,
            current_round["endTScore"] + current_round["endCTScore"],
            int(is_pro_game),
            map_name,
            game_time,
            int(event["attackerSide"] == "CT"),
            ct_position,
            t_position,
            kill_weapon,
        )
        logging.debug("Adding event with characteristics: %s", val_event)
        self.cursor.execute(
            (
                "INSERT INTO Events (MatchID, Round, Pro, MapName, Time, CTWon, "
                "CTArea, TArea, KillWeapon) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
            ),
            val_event,
        )
        event_id = self.cursor.lastrowid
        for weapon in ct_weapons:
            self.cursor.execute(
                "INSERT INTO CTWeapons (EventID, CTWeapon) VALUES (%s, %s)",
                (event_id, weapon),
            )
        for weapon in t_weapons:
            self.cursor.execute(
                "INSERT INTO TWeapons (EventID, TWeapon) VALUES (%s, %s)",
                (event_id, weapon),
            )

    def analyze_map(self, data: Game, match_id: str, *, is_pro_game: bool) -> None:
        """Loop over all rounds and their events and add them to a MYSQL database.

        Args:
            data (json/dict): Json object of the parsed demo file
            match_id (str): String of the demo file name
            is_pro_game (bool): Whether the current map is from a pro game

        Returns:
            None (DB is modified in place)
        """
        # Loop over rounds and each event in them and check if they fulfill all criteria
        ticks: TickInformation = {
            "tickRate": 1 << (data["tickRate"] - 1).bit_length(),
            "freezeTimeEndTick": 0,
        }
        for current_round in data["gameRounds"] or []:
            self.current_frame_index = 0
            ticks["freezeTimeEndTick"] = current_round["freezeTimeEndTick"]
            logging.debug("Round:")
            logging.debug(current_round)
            # Go through all kill events
            if current_round["kills"] is None:
                logging.debug("Round does not have kills recorded")
                continue
            for event in current_round["kills"]:
                if not event["victimSteamID"] or not event["attackerSteamID"]:
                    continue
                game_time = self.get_game_time(event, ticks)
                ct_position, t_position = self.check_position(event, data["mapName"])
                if not ct_position or not t_position:
                    continue
                if not (round_frames := current_round["frames"]):
                    continue
                used_weapons = self.check_weapons(round_frames, event)
                if used_weapons is None:
                    continue
                kill_weapon, ct_weapons, t_weapons = used_weapons
                self.summarize_round(
                    event,
                    game_time,
                    kill_weapon,
                    ct_weapons,
                    t_weapons,
                    ct_position,
                    t_position,
                    current_round,
                    match_id,
                    data["mapName"],
                    is_pro_game=is_pro_game,
                )

    def _demo_files(self, done_set: set[Any]) -> Iterator[Path]:
        """Crawl all desired directories and filter maps and files.

        Also logs the traversed and skipped files and folders and commits
        on the connection after each map_dir.

        Args:
            done_set (set): Set of matchids that should be skipped as they
                have already been analyzed.

        Yields:
            Paths of all the desired files.
        """
        logging.debug("Excluding matches with id: %s", done_set)
        for path in self.directories:
            for map_dir in Path(path).iterdir():
                logging.info("Scanning directory: %s", map_dir)
                if f"de_{map_dir.name}" not in NAV and map_dir.name not in NAV:
                    logging.info(
                        "Map %s|%s is not in supplied navigation files. Skipping it.",
                        f"de_{map_dir.name}",
                        map_dir.name,
                    )
                    continue
                for file in map_dir.glob("*.json"):
                    if file.stem in done_set:
                        logging.debug(
                            "There are already events from the file with "
                            "the matchid %s in the database. Skipping it.",
                            file.stem,
                        )
                        continue
                    logging.info("Working on file %s", file)
                    yield file
                self.connection.commit()

    def analyze_demos(self) -> None:
        """Loops over the specified directory and analyzes each demo file.

        Collects information about each kill event.
        This information is written to a MYSQL DB

        Args:
            None

        Returns:
            None (Modifies external MYSQL DB)
        """
        self.cursor.execute("SELECT DISTINCT e.MatchID FROM Events e")
        done_set = {x[0] for x in self.cursor.fetchall()}

        for demo_file in self._demo_files(done_set):
            logging.info(match_id := demo_file.stem)
            with open(demo_file, encoding="utf-8") as demo_json:
                demo_data: Game = json.load(demo_json)
            self.analyze_map(
                demo_data,
                match_id,
                is_pro_game=r"D:\CSGO\Demos\Maps" in demo_file.parents,
            )
            self.n_analyzed += 1
        logging.info("Analyzed a total of %s demos!", self.n_analyzed)

    def get_wilson_interval(
        self, success_percent: float, total_n: int, z: float
    ) -> tuple[float, float, float]:
        """Calculates the Wilson score interval of success-failure experiments.

        Calcualtes the Wilson score interval as an approximation
        of the binomial proportion confidence interval.

        Args:
            success_percent (float): Percentage of experiments that ended in success
            total_n (int): Total number of experiments
            z (float): Number of standard deviations that the interval should cover
        Returns:
            Tuplet of floats of the form:
            lower_bound_of_interval, success_percent, upper_bound_of_interval
        """
        lower = (
            (success_percent + z**2 / (2 * total_n))
            - (
                z
                * math.sqrt(
                    (success_percent * (1 - success_percent) + z**2 / (4 * total_n))
                    / total_n
                )
            )
        ) / (1 + z**2 / total_n)
        upper = (
            (success_percent + z**2 / (2 * total_n))
            + (
                z
                * math.sqrt(
                    (success_percent * (1 - success_percent) + z**2 / (4 * total_n))
                    / total_n
                )
            )
        ) / (1 + z**2 / total_n)
        return lower, success_percent, upper

    def _add_position_information_to_query(
        self,
        sql: str,
        param: list[Any],
        position_selection: PositionSpecification,
    ) -> str:
        """Add position related information to query string and params list.

        Args:
            sql (str): Current query string
            param (list[str |int]): Current list of query parameters.
            position_selection (PositionSpecification): Information about position
                selection.
        """
        if position_selection["CT"]:
            sql += """AND e.CTArea in %s """
            param.append(position_selection["CT"])
        if position_selection["T"]:
            sql += """AND e.TArea in %s """
            param.append(position_selection["T"])
        return sql

    def _add_ct_equip_information_to_query(
        self,
        sql: str,
        param: list[Any],
        event: FightSpecification,
    ) -> str:
        """Add ct equip related information to query string and params list.

        Args:
            sql (str): Current query string
            param (list[str |int]): Current list of query parameters.
            event (FightSpecification): Dict that holds information about
                which fight scenarios to query.
        """
        if event["use_weapons_classes"]["CT"] == "weapons":
            if event["weapons"]["CT"]["Allowed"]:
                sql += """AND ctw.CTWeapon in %s """
                param.append(event["weapons"]["CT"]["Allowed"])
            if event["weapons"]["CT"]["Forbidden"]:
                sql += """AND ctw.CTWeapon NOT in %s """
                param.append(event["weapons"]["CT"]["Forbidden"])
        elif event["use_weapons_classes"]["CT"] == "classes":
            if event["classes"]["CT"]["Allowed"]:
                sql += """AND wcct.Class in %s """
                param.append(event["classes"]["CT"]["Allowed"])
            if event["classes"]["CT"]["Forbidden"]:
                sql += """AND wcct.Class NOT in %s """
                param.append(event["classes"]["CT"]["Forbidden"])
        return sql

    def _add_t_equip_information_to_query(
        self,
        sql: str,
        param: list[Any],
        event: FightSpecification,
    ) -> str:
        """Add t equip related information to query string and params list.

        Args:
            sql (str): Current query string
            param (list[str |int]): Current list of query parameters.
            event (FightSpecification): Dict that holds information about
                which fight scenarios to query.
        """
        if event["use_weapons_classes"]["T"] == "weapons":
            if event["weapons"]["T"]["Allowed"]:
                sql += """AND tw.TWeapon in %s """
                param.append(event["weapons"]["T"]["Allowed"])
            if event["weapons"]["T"]["Forbidden"]:
                sql += """AND tw.TWeapon NOT in %s """
                param.append(event["weapons"]["T"]["Forbidden"])
        elif event["use_weapons_classes"]["T"] == "classes":
            if event["classes"]["T"]["Allowed"]:
                sql += """AND wct.Class in %s """
                param.append(event["classes"]["T"]["Allowed"])
            if event["classes"]["T"]["Forbidden"]:
                sql += """AND wct.Class NOT in %s """
                param.append(event["classes"]["T"]["Forbidden"])
        return sql

    def _add_kill_equip_information_to_query(
        self,
        sql: str,
        param: list[Any],
        event: FightSpecification,
    ) -> str:
        """Add kill equip related information to query string and params list.

        Args:
            sql (str): Current query string
            param (list[str |int]): Current list of query parameters.
            event (FightSpecification): Dict that holds information about
                which fight scenarios to query.
        """
        if event["use_weapons_classes"]["Kill"] == "weapons":
            if event["weapons"]["Kill"]:
                sql += """AND e.KillWeapon in %s """
                param.append(event["weapons"]["Kill"])
        elif event["use_weapons_classes"]["Kill"] == "classes":  # noqa: SIM102
            if event["classes"]["Kill"]:
                sql += """AND wck.Class in %s """
                param.append(event["classes"]["Kill"])
        return sql

    def calculate_ct_win_percentage(
        self, event: FightSpecification, cursor: Cursor
    ) -> QueryResult:
        """Calculates CT win percentage from CT and T kills.

        Queries information from a database to
        determine CT win percentage of events fitting the criteria

        Args:
            event (FightSpecification): Dict that holds information about
                which fight scenarios to query.
            cursor: (Cursor): Cursor to execute query with.

        Returns:
            QueryResult dict containing information about total number of events
            found as well as ct win percentage with uncertainties.
        """
        sql = (
            """SELECT AVG(t.CTWon), COUNT(t.CTWon) """
            """FROM ( """
            """SELECT DISTINCT e.EventID, e.CTWon """
            """FROM Events e JOIN CTWeapons ctw """
            """ON e.EventID = ctw.EventID """
            """JOIN TWeapons tw """
            """ON e.EventID = tw.EventID """
            """JOIN WeaponClasses wcct """
            """ON ctw.CTWeapon = wcct.WeaponName """
            """JOIN WeaponClasses wct """
            """ON tw.TWeapon = wct.WeaponName """
            """JOIN WeaponClasses wck """
            """ON e.KillWeapon = wck.WeaponName """
            """WHERE e.MapName = %s """
            """AND e.Time BETWEEN %s AND %s """
        )
        param: list[Any] = [
            event["map_name"],
            event["times"]["start"],
            event["times"]["end"],
        ]

        sql = self._add_position_information_to_query(sql, param, event["positions"])

        sql = self._add_ct_equip_information_to_query(sql, param, event)
        sql = self._add_t_equip_information_to_query(sql, param, event)
        sql = self._add_kill_equip_information_to_query(sql, param, event)

        sql += """) t"""

        logging.info(sql)
        logging.info(param)
        res: QueryResult = {"situations_found": 0, "ct_win_percentage": (0, 0, 0)}
        cursor.execute(sql, param)
        result = list(cursor.fetchone() or [])

        if len(result) == 2 and result[1] > 0:  # noqa: PLR2004
            res["situations_found"], res["ct_win_percentage"] = (
                result[1],
                tuple(
                    round(100 * x)
                    for x in self.get_wilson_interval(float(result[0]), result[1], 1.0)
                ),
            )
        return res


def main(args: list[str]) -> None:
    """Determines whether a fight is favourable to the T or CT side."""
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

    event: FightSpecification = {
        "map_name": "de_inferno",
        "weapons": {
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
            "CT": {
                "Allowed": [
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
                "Forbidden": [],
            },
            "T": {
                "Allowed": [
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
                "Forbidden": [],
            },
        },
        "classes": {
            "Kill": ["Rifle", "Heavy"],
            "CT": {"Allowed": ["Rifle", "Heavy"], "Forbidden": []},
            "T": {"Allowed": ["Rifle", "Heavy"], "Forbidden": []},
        },
        "positions": {"CT": ["TopofMid", "Middle"], "T": ["Middle", "TRamp"]},
        "use_weapons_classes": {"CT": "weapons", "Kill": "weapons", "T": "weapons"},
        "times": {"start": 0, "end": 10},
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
        ssl_ca=r"D:\\CSGO\\ML\\csgoml\\auxiliary\\AWS_Steps\\Certs\\global-bundle.pem",
    )

    cursor = connection.cursor()

    analyzer = FightAnalyzer(
        connection,
        cursor,
        directories=options.dirs,
    )

    if options.analyze:
        analyzer.analyze_demos()

    logging.info("With TRamp allowed:")
    logging.info(
        analyzer.calculate_ct_win_percentage(
            event,
            cursor,
        )
    )

    cursor.close()
    connection.close()


if __name__ == "__main__":
    main(sys.argv[1:])
