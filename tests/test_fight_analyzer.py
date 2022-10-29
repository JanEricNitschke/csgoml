"""Tests for fight_analyzer.py"""
# pylint: disable=attribute-defined-outside-init,invalid-name

import os
import shutil
import json
import requests
import pymysql
from csgoml.analysis.fight_analyzer import FightAnalyzer


class TestFightAnalyzer:
    """Class to test FightAnalyzer"""

    def setup_class(self):
        """Setup class by defining loading dictionary of test json files"""
        try:
            mysql2 = pymysql.connect(
                host="127.0.0.1",
                user="root",
                passwd="testpassword",
                db="analyzertest",
                port=3800,
            )
            my_cursor = mysql2.cursor()
            my_cursor.execute(
                "CREATE TABLE Events (EventID int NOT NULL AUTO_INCREMENT, MatchID  TEXT, Round BIGINT, Pro  TINYINT(1), MapName VARCHAR(20), Time double, CTWon tinyint(1), CTArea VARCHAR(30), TArea VARCHAR(30), KillWeapon VARCHAR(30), PRIMARY KEY (EventID), INDEX (MapName, Time, CTArea, TArea, KillWeapon, Pro))"
            )
            my_cursor.execute(
                "CREATE TABLE CTWeapons (EventID int, CTWeapon VARCHAR(30), FOREIGN KEY (EventID) REFERENCES Events(EventID), INDEX (EventID, CTWeapon))"
            )
            my_cursor.execute(
                "CREATE TABLE TWeapons (EventID int, TWeapon VARCHAR(30), FOREIGN KEY (EventID) REFERENCES Events(EventID), INDEX (EventID, TWeapon))"
            )
            my_cursor.execute(
                "CREATE TABLE WeaponClasses (WeaponName text, Class text)"
            )
            weapons = {
                "Pistols": [
                    "CZ75 Auto",
                    "Desert Eagle",
                    "Dual Berettas",
                    "Five-SeveN",
                    "Glock-18",
                    "P2000",
                    "P250",
                    "R8 Revolver",
                    "Tec-9",
                    "USP-S",
                ],
                "Heavy": ["MAG-7", "Nova", "Sawed-Off", "XM1014", "M249", "Negev"],
                "SMG": ["MAC-10", "MP5-SD", "MP7", "MP9", "P90", "PP-Bizon", "UMP-45"],
                "Rifles": [
                    "AK-47",
                    "AUG",
                    "FAMAS",
                    "Galil AR",
                    "M4A1",
                    "M4A4",
                    "SG 553",
                    "AWP",
                    "G3SG1",
                    "SCAR-20",
                    "SSG 08",
                ],
                "Grenades": [
                    "Smoke Grenade",
                    "Flashbang",
                    "HE Grenade",
                    "Incendiary Grenade",
                    "Molotov",
                    "Decoy Grenade",
                ],
                "Equipment": ["Knife", "Zeus x27"],
            }
            for my_class, weapons_list in weapons.items():
                for weapon in weapons_list:
                    sql = (
                        "INSERT INTO WeaponClasses (WeaponName, Class) VALUES (%s, %s)"
                    )
                    val = (weapon, my_class)
                    my_cursor.execute(sql, val)
            mysql2.commit()
            with open("tests/test_jsons.json", encoding="utf-8") as f:
                self.json_data = json.load(f)
            for file in self.json_data:
                self._get_jsonfile(json_link=self.json_data[file]["url"])
            for file in os.listdir():
                if file.endswith(".json"):
                    if file.startswith("356"):
                        target_path = os.path.join(os.getcwd(), "Maps", "inferno")
                    elif file.startswith("733"):
                        target_path = os.path.join(os.getcwd(), "Maps", "mirage")
                    else:
                        target_path = os.path.join(os.getcwd(), "Maps", "cs_rush")
                    if not os.path.exists(target_path):
                        os.makedirs(target_path)
                    shutil.move(file, os.path.join(target_path, file))
        except Exception as e:
            print(
                f"Caught exception {e} when trying to connect to database. Skipping tests that rely on it."
            )
            mysql2 = None
            my_cursor = None
        self.analyzer = FightAnalyzer(
            connection=mysql2,
            cursor=my_cursor,
            directories=[os.path.join(os.getcwd(), "Maps")],
        )

    def teardown_class(self):
        """Set sorter to none, deletes all demofiles, JSON and directories"""
        if self.analyzer.connection is not None:
            files_in_directory = os.listdir()
            filtered_files = [
                file for file in files_in_directory if file.endswith(".json")
            ]
            if len(filtered_files) > 0:
                for f in filtered_files:
                    os.remove(f)
            for directory in self.analyzer.directories:
                shutil.rmtree(directory)
            self.analyzer.cursor.close()
            self.analyzer.connection.close()
        self.analyzer = None

    @staticmethod
    def _get_jsonfile(json_link):
        print("Requesting " + json_link)
        r = requests.get(json_link, timeout=20)
        open(json_link.split(r"/")[-1], "wb").write(r.content)

    def test_get_area_from_pos(self):
        """Tests get_area_from_pos"""
        assert (
            self.analyzer.get_area_from_pos("de_dust2", [None, 1000.0, -100])
            == "No area found"
        )
        assert self.analyzer.get_area_from_pos("de_dust2", [750.0, -150.0, 73.0]) == ""
        assert (
            self.analyzer.get_area_from_pos("de_dust2", [-464.0, 2010.0, -60.0])
            == "MidDoors"
        )
        assert (
            self.analyzer.get_area_from_pos("de_dust2", [295.0, 2422.0, -57.35])
            == "CTSpawn"
        )

    def test_check_position(self):
        """Test check_position"""
        event_dict = {
            "attackerX": -464.0,
            "attackerY": 2010.0,
            "attackerZ": -60.0,
            "victimX": 295.0,
            "victimY": 2422.0,
            "victimZ": -57.35,
            "attackerSide": "CT",
            "victimSide": "T",
        }
        assert self.analyzer.check_position(event_dict, "de_dust2") == (
            "MidDoors",
            "CTSpawn",
        )
        event_dict["attackerSide"], event_dict["victimSide"] = (
            event_dict["victimSide"],
            event_dict["attackerSide"],
        )
        assert self.analyzer.check_position(event_dict, "de_dust2") == (
            "CTSpawn",
            "MidDoors",
        )
        event_dict["attackerSide"] = "Nope"
        assert self.analyzer.check_position(event_dict, "de_dust2") == (None, None)

    def test_get_game_time(self):
        """Tests get_game_time"""
        event_dict = {"tick": 1128}
        ticks_dict = {"freezeTimeEndTick": 1000, "tickRate": 128}
        assert self.analyzer.get_game_time(event=event_dict, ticks=ticks_dict) == 1.0
        ticks_dict["tickRate"] = 64
        assert self.analyzer.get_game_time(event=event_dict, ticks=ticks_dict) == 2.0
        ticks_dict["freezeTimeEndTick"] = 936
        assert self.analyzer.get_game_time(event=event_dict, ticks=ticks_dict) == 3.0
        event_dict["tick"] = 1192
        assert self.analyzer.get_game_time(event=event_dict, ticks=ticks_dict) == 4.0

    def test_check_weapons(self):
        """Tests check_weapons"""
        round_dict = {
            "frames": [
                {
                    "tick": 1,
                    "t": {
                        "players": [
                            {
                                "steamID": 76561198243884024,
                                "inventory": [
                                    {
                                        "weaponName": "P90",
                                    }
                                ],
                            }
                        ]
                    },
                    "ct": {
                        "players": [
                            {
                                "steamID": 76561198063005604,
                                "inventory": [
                                    {
                                        "weaponName": "Glock-18",
                                    }
                                ],
                            }
                        ]
                    },
                },
                {
                    "tick": 2,
                    "t": {
                        "players": [
                            {
                                "steamID": 76561198243884024,
                                "inventory": [
                                    {
                                        "weaponName": "Glock-18",
                                    },
                                    {
                                        "weaponName": "P90",
                                    },
                                ],
                            }
                        ]
                    },
                    "ct": {
                        "players": [
                            {
                                "steamID": 76561198063005604,
                                "inventory": [
                                    {
                                        "weaponName": "Glock-18",
                                    },
                                    {
                                        "weaponName": "AK-47",
                                    },
                                ],
                            }
                        ]
                    },
                },
            ]
        }
        event_dict = {
            "tick": 1,
            "attackerSteamID": 76561198243884024,
            "attackerSide": "T",
            "victimSteamID": 76561198063005604,
            "weapon": "P90",
            "victimSide": "CT",
        }

        assert self.analyzer.check_weapons(
            current_round=round_dict, event=event_dict
        ) == ("P90", {"Glock-18"}, {"P90"})
        event_dict["tick"] = 2
        assert self.analyzer.check_weapons(
            current_round=round_dict, event=event_dict
        ) == ("P90", {"Glock-18", "AK-47"}, {"Glock-18", "P90"})

        self.analyzer.current_frame_index = 0
        event_dict["tick"] = 2
        event_dict["attackerSide"], event_dict["victimSide"] = (
            event_dict["victimSide"],
            event_dict["attackerSide"],
        )
        event_dict["attackerSteamID"], event_dict["victimSteamID"] = (
            event_dict["victimSteamID"],
            event_dict["attackerSteamID"],
        )
        event_dict["weapon"] = "AK-47"
        round_dict = {
            "frames": [
                {
                    "tick": 1,
                    "t": {
                        "players": [
                            {
                                "steamID": 76561198243884024,
                                "inventory": [
                                    {
                                        "weaponName": "P90",
                                    }
                                ],
                            }
                        ]
                    },
                    "ct": {
                        "players": [
                            {
                                "steamID": 76561198063005604,
                                "inventory": [
                                    {
                                        "weaponName": "Glock-18",
                                    }
                                ],
                            }
                        ]
                    },
                },
                {
                    "tick": 2,
                    "t": {
                        "players": [{"steamID": 76561198243884024, "inventory": None}]
                    },
                    "ct": {
                        "players": [
                            {
                                "steamID": 76561198063005604,
                                "inventory": None,
                            }
                        ]
                    },
                },
                {
                    "tick": 3,
                    "t": {
                        "players": [
                            {
                                "steamID": 76561198243884024,
                                "inventory": [
                                    {
                                        "weaponName": "Glock-18",
                                    },
                                    {
                                        "weaponName": "P90",
                                    },
                                ],
                            }
                        ]
                    },
                    "ct": {
                        "players": [
                            {
                                "steamID": 76561198063005604,
                                "inventory": [
                                    {
                                        "weaponName": "Glock-18",
                                    },
                                    {
                                        "weaponName": "AK-47",
                                    },
                                    {"weaponName": "HE-Grenade"},
                                ],
                            }
                        ]
                    },
                },
            ]
        }
        assert self.analyzer.check_weapons(
            current_round=round_dict, event=event_dict
        ) == ("AK-47", {"Glock-18"}, {"P90"})
        event_dict["tick"] = 3
        assert self.analyzer.check_weapons(
            current_round=round_dict, event=event_dict
        ) == ("AK-47", {"Glock-18", "AK-47", "HE-Grenade"}, {"Glock-18", "P90"})

    def test_summarize_round(self):
        """Tests summatize_round"""
        if self.analyzer.cursor is None:
            print(
                "Skipping 'test_summarize_round' because it relies on a database connection."
            )
            return
        self.analyzer.cursor.execute("SELECT EXISTS (SELECT 1 FROM `Events`);")
        result = self.analyzer.cursor.fetchone()
        assert result == (0,)
        event_dict = {"attackerSide": "CT"}
        game_time = 30.0
        kill_weapon = "P90"
        CT_weapons = ["P90", "Glock-18"]
        T_weapons = ["Ak-47", "HE Grenade", "Flashbang"]
        CT_position = "MidDoors"
        T_position = "CTSpawn"
        current_rounds = {"endTScore": 10, "endCTScore": 6}
        match_id = "560"
        map_name = "de_inferno"
        is_pro = False
        self.analyzer.summarize_round(
            event=event_dict,
            game_time=game_time,
            kill_weapon=kill_weapon,
            CT_weapons=CT_weapons,
            T_weapons=T_weapons,
            CT_position=CT_position,
            T_position=T_position,
            current_round=current_rounds,
            match_id=match_id,
            map_name=map_name,
            is_pro_game=is_pro,
        )
        self.analyzer.cursor.execute("SELECT COUNT(*) FROM `Events`")
        result = self.analyzer.cursor.fetchone()
        assert result == (1,)
        self.analyzer.cursor.execute("SELECT * FROM `Events`")
        result = self.analyzer.cursor.fetchone()
        assert result == (
            1,
            "560",
            16,
            0,
            "de_inferno",
            30.0,
            1,
            "MidDoors",
            "CTSpawn",
            "P90",
        )
        self.analyzer.cursor.execute("SELECT COUNT(*) FROM `CTWeapons`")
        result = self.analyzer.cursor.fetchone()
        assert result == (2,)
        self.analyzer.cursor.execute("SELECT COUNT(*) FROM `TWeapons`")
        result = self.analyzer.cursor.fetchone()
        assert result == (3,)
        self.analyzer.cursor.execute("SET FOREIGN_KEY_CHECKS = 0;")
        self.analyzer.cursor.execute("TRUNCATE TABLE `CTWeapons`;")
        self.analyzer.cursor.execute("TRUNCATE TABLE `TWeapons`;")
        self.analyzer.cursor.execute("TRUNCATE TABLE `Events`;")
        self.analyzer.cursor.execute("SET FOREIGN_KEY_CHECKS = 1;")

    def test_analyze_demos(self):
        """Tests analyze_demos"""
        if self.analyzer.cursor is None:
            print(
                "Skipping 'test_analyze_demos' because it relies on a database connection."
            )
            return
        self.analyzer.analyze_demos()
        assert self.analyzer.n_analyzed == 2
        self.analyzer.cursor.execute(
            "SELECT COUNT(*) FROM `Events` where MapName = %s", "de_inferno"
        )
        result = self.analyzer.cursor.fetchone()
        assert result == (6,)
        self.analyzer.cursor.execute(
            "SELECT COUNT(*) FROM `Events` where MapName = %s and Round = %s",
            ("de_inferno", 1),
        )
        result = self.analyzer.cursor.fetchone()
        assert result == (3,)
        self.analyzer.cursor.execute(
            "SELECT COUNT(*) FROM `Events` where MapName = %s", "de_mirage"
        )
        result = self.analyzer.cursor.fetchone()
        assert result == (3,)

    def test_calculate_CT_win_percentage(self):
        """Tests calculate_CT_win_percentage"""
        if self.analyzer.cursor is None:
            print(
                "Skipping 'test_calculate_CT_win_percentage' because it relies on a database connection."
            )
            return
        times = [0, 10000]
        positions = {
            "CT": {"Allowed": {}, "Forbidden": {}},
            "T": {"Allowed": {}, "Forbidden": {}},
        }
        use_weapons_classes = {"CT": "weapons", "T": "weapons", "Kill": "weapons"}
        weapons = {
            "CT": {
                "Allowed": {},
                "Forbidden": {},
            },
            "Kill": [],
            "T": {
                "Allowed": {},
                "Forbidden": {},
            },
        }

        classes = {
            "CT": {
                "Allowed": {},
                "Forbidden": {},
            },
            "T": {
                "Allowed": {},
                "Forbidden": {},
            },
            "Kill": [],
        }
        assert self.analyzer.calculate_CT_win_percentage(
            times=times,
            positions=positions,
            weapons=weapons,
            classes=classes,
            use_weapons_classes=use_weapons_classes,
            map_name="de_inferno",
        ) == (6, round(4 / 6 * 100))
        assert self.analyzer.calculate_CT_win_percentage(
            times=times,
            positions=positions,
            weapons=weapons,
            classes=classes,
            use_weapons_classes=use_weapons_classes,
            map_name="cs_rush",
        ) == (0, 0)
