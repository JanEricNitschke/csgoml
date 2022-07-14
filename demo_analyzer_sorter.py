"""Runs awpy demo parser on multiple demo files and organizes the results by map.

    Typical usage example:

    analyzer = DemoAnalyzerSorter(
        indentation=False,
        dirs=["D:\\CSGO\\Demos",r"C:\\Program Files (x86)\\Steam\\steamapps\\common\\Counter-Strike Global Offensive\\csgo\\replays"],
        log=None
        start_id=1,
        end_id=999999,
        mm_id=10000,
        maps_dir="D:\\CSGO\\Demos\\Maps",
        json_ending="",
    )
    analyzer.parse_demos()
"""
#!/usr/bin/env python

import os
import sys
import logging
import shutil
import argparse
from awpy.parser import DemoParser


class DemoAnalyzerSorter:
    """Runs awpy demo parser on multiple demo files and organizes the results by map.

    Attributes:
        indentation: A boolean indicating if json files should be indented
        dirs: A list of directory paths to scan for demo files
        log: Path of the log file
        start_id: An integer that determines at which demo parsing should start
        end_id: An integer indicating at which demo parsing should stop
        mm_id: An integer that determines how demos that do not have an id should be treated
        maps_dir: A directory path determining where the parsed json files should be stored.
        json_ending: A string that will be appended to the end of the parsed json.
        n_analyzed: An integer keeping track of the number of demos that have been analyzed.
    """

    def __init__(
        self,
        indentation=False,
        dirs=None,
        log=r"D:\CSGO\ML\CSGOML\logs\DemoAnalyzerSorter.log",
        start_id=1,
        end_id=99999,
        mm_id=10000,
        maps_dir=r"D:\CSGO\Demos\Maps",
        json_ending="",
        debug=False,
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
        DemoAnalyzerSorter.logger = logging.getLogger("DemoAnalyzerSorter")
        self.indentation = indentation
        if dirs is None:
            self.dirs = [
                r"D:\CSGO\Demos",
                r"C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\csgo\replays",
            ]
        else:
            self.dirs = dirs
        self.log = log
        self.start_id = start_id
        self.end_id = end_id
        self.mm_id = mm_id
        self.maps_dir = maps_dir
        self.json_ending = json_ending
        self.n_analyzed = 0

    def get_ids(self, filename):
        """Fetches map ID from filename.

        For file names of the type '510.dem' is grabs the ID before the file ending.
        If that is not an integer it instead returns a default defined at class initialization.

        Args:
            filename: A string corresponding to the filename of a demo.

        Returns:
            ID: The file name without the file ending.
            NumberID: The ID converted into an integer or a default if that is not possible.
        """
        name = filename.split(".")[0]
        logging.debug(("Using ID: %s", id))
        try:
            number_id = int(name)
        except ValueError:
            number_id = self.mm_id
        return name, number_id

    def clean_rounds(self, demo_parser):
        """Cleans rounds and handles exceptions

        Tells the demo parser to clean the rounds of its currently active demo and catches exceptions.

        Args:
            demo_parser: Awpy demo parser object.

        Returns:
            None (Rounds are cleaned in place)
        """
        try:
            demo_parser.clean_rounds()
        except AttributeError:
            logging.exception("This demo has an attribute error while cleaning.")
        except TypeError:
            logging.exception("This demo has a type error while cleaning.")

    def get_map_name(self, data):
        """Extracts the map name from CS:GO demo parsed to json.

        Most of the commonly used CS:GO maps have a name of the format de_XYZ
        This extracts the map name after the 'de_'.
        If the map in question does not follow this format the whole name is used instead.

        Args:
            data: json object produced from awpy parsing a csgo demo file.

        Returns:
            A string corresponding to the (shortend) name of the map of the parsed game.
        """
        if data["mapName"].startswith("de_"):
            return data["mapName"].split("_")[1]
        return data["mapName"]

    def move_json(self, source, destination):
        """Moves json files from source to destination while handling exceptions.

        Args:
            source: String corresponding to the current path of the json file
            destination: String corresponding to the path the json file should be moved to.

        Returns:
            None
        """
        logging.info("Source: %s", source)
        logging.info("Destination: %s", destination)
        try:
            os.rename(source, destination)
        except FileExistsError:
            os.remove(destination)
            os.rename(source, destination)
        except OSError:
            shutil.move(source, destination)
        logging.info("Moved json to: %s", destination)

    def parse_demos(self):
        """Run awpy demo parser on all demos in dirs and move them to their corresponding directory in map_dirs.

        Args:
            None (Everything is taken from class variables)

        Returns:
            None
        """
        for directory in self.dirs:
            logging.info("Scanning directory: %s", directory)
            for filename in os.listdir(directory):
                if filename.endswith(".dem"):
                    file_path = os.path.join(directory, filename)
                    # checking if it is a file
                    if os.path.isfile(file_path):
                        logging.info("At file: %s", filename)
                        name, number_id = self.get_ids(filename)
                        if int(number_id) >= self.start_id:
                            if int(number_id) > self.end_id:
                                logging.info("Parsed last relevant demo.")
                                break
                            demo_parser = DemoParser(
                                demofile=file_path,
                                demo_id=name,
                                parse_rate=128,
                                buy_style="hltv",
                                dmg_rolled=True,
                                parse_frames=True,
                                json_indentation=self.indentation,
                                outpath=directory,
                            )
                            data = demo_parser.parse(clean=False)
                            self.clean_rounds(demo_parser)
                            data = demo_parser.json
                            map_name = self.get_map_name(data)
                            logging.debug("Scanned map name: %s", map_name)
                            source = os.path.join(directory, name + ".json")
                            if not os.path.exists(
                                os.path.join(self.maps_dir, map_name)
                            ):
                                os.makedirs(os.path.join(self.maps_dir, map_name))
                            destination = os.path.join(
                                self.maps_dir,
                                map_name,
                                name + self.json_ending + ".json",
                            )
                            self.move_json(source, destination)
                            self.n_analyzed += 1
                            if self.n_analyzed > 5:
                                break
        logging.info("Analyzed a total of %s demos!", self.n_analyzed)


def main(args):
    """Runs awpy demo parser on multiple demo files and organizes the results by map."""
    parser = argparse.ArgumentParser("Analyze the early mid fight on inferno")
    parser.add_argument(
        "-d", "--debug", action="store_true", default=False, help="Enable debug output."
    )
    parser.add_argument(
        "--noindentation",
        action="store_true",
        default=False,
        help="Turn off indentation of json file.",
    )
    parser.add_argument(
        "--dirs",
        nargs="*",
        default=[
            r"D:\CSGO\Demos",
            r"C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\csgo\replays",
        ],
        help="All the directories that should be scanned for demos.",
    )
    parser.add_argument(
        "-l",
        "--log",
        default=r"D:\CSGO\ML\CSGOML\logs\DemoAnalyzerSorter.log",
        help="Path to output log.",
    )
    parser.add_argument(
        "--startid",
        type=int,
        default=1,
        help="Analyze demos with a name above this id",
    )
    parser.add_argument(
        "--endid",
        type=int,
        default=2,
        help="Analyze demos with a name below this id",
    )
    parser.add_argument(
        "--mmid",
        type=int,
        default=10000,
        help="Set id value that should be used for mm demos that normally do not have one.",
    )
    parser.add_argument(
        "-m",
        "--mapsdir",
        default=r"D:\CSGO\Demos\Maps",
        help="Path to directory that contains the folders for the maps that should be included in the analysis.",
    )
    parser.add_argument(
        "--jsonending",
        default="",
        help="What should be added at the end of the name of the produced json files (between the id and the .json). Default is nothing.",
    )
    options = parser.parse_args(args)

    analyzer = DemoAnalyzerSorter(
        indentation=(not options.noindentation),
        dirs=options.dirs,
        log=options.log,
        start_id=options.startid,
        end_id=options.endid,
        mm_id=options.mmid,
        maps_dir=options.mapsdir,
        json_ending=options.jsonending,
    )
    analyzer.parse_demos()


if __name__ == "__main__":
    main(sys.argv[1:])
