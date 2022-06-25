"""Testing awpy plotting"""
#!/usr/bin/env python

import itertools
import os
import sys
import logging
import json
import shutil
import argparse
from awpy.parser import DemoParser
import matplotlib.pyplot as plt
from awpy.visualization.plot import plot_map
from awpy.visualization.plot import plot_round
from awpy.visualization.plot import plot_rounds_same_players
from awpy.visualization.plot import plot_rounds_different_players
from awpy.visualization.plot import plot_nades
from awpy.visualization.plot import plot_map, position_transform
from matplotlib import patches
from awpy.data import NAV
import numpy as np
import collections
from awpy.analytics.nav import generate_position_token

# import shapely


def stepped_hull(points, is_sorted=False, return_sections=False):
    # May be equivalent to the orthogonal convex hull

    if not is_sorted:
        points = sorted(set(points))

    if len(points) <= 1:
        return points

    # Get extreme y points
    min_y = min(points, key=lambda p: p[1])
    max_y = max(points, key=lambda p: p[1])

    # Create upper section
    upper_left = build_stepped_upper(
        sorted(points, key=lambda tup: (tup[0], tup[1])), max_y
    )
    upper_right = build_stepped_upper(
        sorted(points, key=lambda tup: (-tup[0], tup[1])), max_y
    )

    # Create lower section
    lower_left = build_stepped_lower(
        sorted(points, key=lambda tup: (tup[0], -tup[1])), min_y
    )
    lower_right = build_stepped_lower(
        sorted(points, key=lambda tup: (-tup[0], -tup[1])), min_y
    )

    # Correct the ordering
    lower_right.reverse()
    upper_left.reverse()

    if return_sections:
        return lower_left, lower_right, upper_right, upper_left

    # Remove duplicate points
    hull = list(dict.fromkeys(lower_left + lower_right + upper_right + upper_left))
    hull.append(hull[0])
    return hull


def build_stepped_upper(points, max_y):
    # Steps towards the highest y point

    section = [points[0]]

    if max_y != points[0]:
        for point in points:
            if point[1] >= section[-1][1]:
                section.append(point)
            if max_y == point:
                break
    return section


def build_stepped_lower(points, min_y):
    # Steps towards the lowest y point

    section = [points[0]]

    if min_y != points[1]:
        for point in points:
            if point[1] <= section[-1][1]:
                section.append(point)

            if min_y == point:
                break
    return section


def get_ids(filename, mmid):
    """get id"""
    name = filename.split(".")[0]
    logging.debug(("Using ID: %s", name))
    try:
        number_id = int(name)
    except ValueError:
        number_id = mmid
    return name, number_id


def main(args):
    """Runs awpy demo parser on multiple demo files and organizes the results by map."""
    parser = argparse.ArgumentParser("Analyze the early mid fight on inferno")
    parser.add_argument(
        "-d", "--debug", action="store_true", default=False, help="Enable debug output."
    )
    parser.add_argument(
        "-l",
        "--log",
        default=r"D:\CSGO\ML\CSGOML\logs\plotting_tests.log",
        help="Path to output log.",
    )
    parser.add_argument(
        "-f",
        "--file",
        default=r"D:\CSGO\Demos\713.dem",  # "D:\CSGO\Demos\Maps\dust2\713.json"
        help="Path to output log.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=r"D:\CSGO\ML\CSGOML\Plots",
        help="Path to output log.",
    )
    options = parser.parse_args(args)

    if options.log == "None":
        options.log = None
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
    logging.info("file %s", options.file)
    name, number_id = get_ids(os.path.basename(options.file), -1)
    logging.info("name %s", name)
    logging.info("number_id %s", number_id)
    json_path = os.path.join(options.output, name + ".json")
    logging.info("Path to json?: %s", json_path)
    if os.path.isfile(json_path):
        logging.info("Reading json from file.")
        with open(json_path, encoding="utf-8") as demo_json:
            data = json.load(demo_json)
    else:
        logging.info("Running DemoParser to generate json file.")
        demo_parser = DemoParser(
            demofile=options.file,
            demo_id=name,
            log=True,
            parse_rate=128,
            buy_style="hltv",
            dmg_rolled=True,
            parse_frames=True,
            json_indentation=True,
            outpath=r"D:\CSGO\ML\CSGOML\Plots",
        )
        data = demo_parser.parse(clean=True)
    # map_name = data["mapName"]
    # f, ax = plt.subplots()
    # plt.gca().invert_yaxis()
    # logging.info(NAV["de_dust2"])
    # token = generate_position_token(map_name, data["gameRounds"][2]["frames"][5])[
    #    "tToken"
    # ]

    for map_name, map_dict in NAV.items():
        area_points = collections.defaultdict(list)
        fig2, ax2 = plot_map(map_name=map_name, map_type="simpleradar", dark=True)
        fig2.set_size_inches(19.2, 10.8)
        for a in map_dict:
            area = map_dict[a]
            if area["areaName"] == "":
                continue
            try:
                cur_x = []
                cur_y = []
                cur_x.append(position_transform(map_name, area["southEastX"], "x"))
                cur_x.append(position_transform(map_name, area["northWestX"], "x"))
                cur_y.append(position_transform(map_name, area["southEastY"], "y"))
                cur_y.append(position_transform(map_name, area["northWestY"], "y"))
            except KeyError:
                cur_x = []
                cur_y = []
                cur_x.append(area["southEastX"])
                cur_x.append(area["northWestX"])
                cur_y.append(area["southEastY"])
                cur_y.append(area["northWestY"])
            for x, y in itertools.product(cur_x, cur_y):
                area_points[area["areaName"]].append((x, y))
        for i, area in enumerate(sorted(area_points)):
            if not area or len(area_points[area]) < 5 * 4:
                continue
            # Orthogonal convex hull
            points_array = np.array(area_points[area])
            hull = np.array(stepped_hull(area_points[area]))
            points_array = np.array(area_points[area])
            ax2.plot(hull[:, 0], hull[:, 1], "-", lw=2, label=area)  #
            center = points_array.mean(axis=0)
            ax2.text(center[0], center[1], 0, c="#EB0841")
            handles, labels = ax2.get_legend_handles_labels()
        lgd = ax2.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1.05))
        # plt.legend()  # - (center[0] - hull[0][0]) / 2 # area + "_" +
        plt.savefig(
            os.path.join(options.output, f"hull_with_token_res_{map_name}.png"),
            bbox_inches="tight",
            bbox_extra_artists=(lgd,),
            dpi=100,
        )
        plt.figure().clear()
        plt.close()
        plt.cla()
        plt.clf()
    for map_name, map_dict in NAV.items():
        logging.info(map_name)
        fig1, ax1 = plot_map(map_name=map_name, map_type="simpleradar", dark=True)
        for a in map_dict:
            area = map_dict[a]
            try:
                area["southEastX"] = position_transform(
                    map_name, area["southEastX"], "x"
                )
                area["northWestX"] = position_transform(
                    map_name, area["northWestX"], "x"
                )
                area["southEastY"] = position_transform(
                    map_name, area["southEastY"], "y"
                )
                area["northWestY"] = position_transform(
                    map_name, area["northWestY"], "y"
                )
            except KeyError:
                pass
            width = area["southEastX"] - area["northWestX"]
            height = area["northWestY"] - area["southEastY"]
            southwest_x = area["northWestX"]
            southwest_y = area["southEastY"]
            rect = patches.Rectangle(
                (southwest_x, southwest_y),
                width,
                height,
                linewidth=1,
                edgecolor="yellow",
                facecolor="None",
            )
            ax1.add_patch(rect)
        plt.savefig(
            os.path.join(options.output, f"tiles_{map_name}.png"), bbox_inches="tight"
        )
        plt.figure().clear()
        plt.close()
        plt.cla()
        plt.clf()

    """
    map_file = os.path.join(options.output, name + "_map.png")
    nades_file = os.path.join(options.output, name + "_nades.png")
    rounds_file_diff_play = os.path.join(options.output, name + "_rounds_diff_play.gif")
    rounds_file_same_play = os.path.join(options.output, name + "_rounds_same_play.gif")
    round_file = os.path.join(options.output, name + "_round.gif")
    logging.info("map_file: %s", map_file)
    logging.info("nades_file: %s", nades_file)
    logging.info("rounds_file_diff_play: %s", rounds_file_diff_play)
    logging.info("rounds_file_same_play: %s", rounds_file_same_play)
    logging.info("round_file: %s", round_file)
    if not os.path.isfile(map_file):
        logging.info("Generating map_file")
        fig, ax = plot_map(map_name=map_name, map_type="simpleradar", dark=True)
        plt.savefig(map_file)
        plt.show()
    if not os.path.isfile(nades_file):
        logging.info("Generating nades_file")
        fig, ax = plot_nades(
            rounds=data["gameRounds"][7:10],
            nades=[
                "Flashbang",
                "HE Grenade",
                "Smoke Grenade",
                "Molotov",
                "Incendiary Grenade",
            ],
            side="CT",
            map_name=map_name,
        )
        plt.savefig(nades_file)
        plt.show()
    # if not os.path.isfile(rounds_file):
    import matplotlib

    logging.info("matplotlib: {}".format(matplotlib.__version__))
    if not os.path.isfile(rounds_file_same_play):
        os.chdir(options.output)
        logging.info("Generating rounds_file")
        plot_rounds_same_players(
            rounds_file_same_play,
            [data["gameRounds"][i + 5]["frames"] for i in range(10)],
            map_name=map_name,
            map_type="simpleradar",
            sides=["ct", "t"],
        )
    if not os.path.isfile(rounds_file_diff_play):
        os.chdir(options.output)
        logging.info("Generating rounds_file")
        plot_rounds_different_players(
            rounds_file_diff_play,
            [data["gameRounds"][i + 5]["frames"] for i in range(10)],
            map_name=map_name,
            map_type="simpleradar",
            sides=["ct", "t"],
        )
    if not os.path.isfile(round_file):
        os.chdir(options.output)
        logging.info("Generating rounds_file")
        plot_round(
            round_file,
            data["gameRounds"][8]["frames"],
            map_name=map_name,
            map_type="simpleradar",
            dark=False,
        )"""


if __name__ == "__main__":
    main(sys.argv[1:])
