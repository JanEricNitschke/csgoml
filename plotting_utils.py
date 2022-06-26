"""Collection of functions to extend awpy plotting capabilites

    Typical usage example:

    plot_round_tokens(
        filename=plot_token_file,
        frames=frames,
        map_name=demo_map_name,
        map_type="simpleradar",
        dark=False,
        fps=2,
    )
    for map_name in NAV:
        plot_map_areas(
            output_path=options.output,
            map_name=map_name,
            map_type="simpleradar",
            dark=False,
        )
        plot_map_tiles(
            output_path=options.output,
            map_name=map_name,
            map_type="simpleradar",
            dark=False,
        )
"""
#!/usr/bin/env python

import collections
import itertools
import os
import sys
import logging
import json
import shutil
import argparse
import numpy as np
from matplotlib import patches
import imageio
from tqdm import tqdm
import matplotlib.pyplot as plt
from awpy.parser import DemoParser
from awpy.visualization.plot import *
from awpy.analytics.nav import generate_position_token
from awpy.data import NAV


def stepped_hull(points, is_sorted=False, return_sections=False):
    """Takes a set of points and produces an approximation of their orthogonal convex hull

    Args:
        points: A list of points given as tuples (x, y)
        is_sorted: A boolean specifying if the input points are already sorted
        return_sections: A boolean determining if the sections are returned separetely or combined

    Returns:
        A list of points making up the hull or four lists of points making up the four quadrants of the hull"""
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
    """Builds builds towards the upper part of the hull based on starting point and maximum y value.

    Args:
        points: A list of points to build the upper left hull section from
        max_y: The point with the highest y

    Returns:
        A list of points making up the upper part of the hull"""
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
    """Builds builds towards the lower part of the hull based on starting point and maximum y value.

    Args:
        points: A list of points to build the upper left hull section from
        min_y: The point with the lowest y

    Returns:
        A list of points making up the lower part of the hull"""
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
    logging.debug(("Using ID: %s", name))
    try:
        number_id = int(name)
    except ValueError:
        number_id = mmid
    return name, number_id


def get_areas_hulls_centers(map_name):
    """Gets the sets of points making up the named areas of a map. Then builds their hull and centroid.

    Args:
        map_name: A string specifying the map for which the features should be build.

    Returns:
        Three dictionary containing the points, hull and centroid of each named area in the map
    """
    hulls = {}
    centers = {}
    area_points = collections.defaultdict(list)
    for a in NAV[map_name]:
        area = NAV[map_name][a]
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
    for area in area_points:
        points_array = np.array(area_points[area])
        hull = np.array(stepped_hull(area_points[area]))
        hulls[area] = hull
        centers[area] = points_array.mean(axis=0)
    return area_points, hulls, centers


def plot_round_tokens(
    filename, frames, map_name="de_ancient", map_type="original", dark=False, fps=10
):
    """Plots the position tokens of a round and saves as a .gif. CTs are blue, Ts are orange. Only use untransformed coordinates.

    Args:
        filename (string): Filename to save the gif
        frames (list): List of frames from a parsed demo
        map_name (string): Map to search
        map_type (string): "original" or "simpleradar"
        dark (boolean): Only for use with map_type="simpleradar". Indicates if you want to use the SimpleRadar dark map type
        fps (integer): Number of frames per second in the gif

    Returns:
        matplotlib fig and ax, saves .gif
    """
    if os.path.isdir("csgo_tmp"):
        shutil.rmtree("csgo_tmp/")
    os.mkdir("csgo_tmp")
    # Colors of borders of each area depending on number of t's/ct's that reside in it
    colors = {
        "t": ["lightsteelblue", "#300000", "#7b0000", "#b10000", "#c80000", "#ff0000"],
        "ct": ["lightsteelblue", "#360CCD", "#302DD9", "#2A4EE5", "#246FF0", "#1E90FC"],
    }
    image_files = []
    # Grab points, hull and centroid for each named area of the map
    area_points, hulls, centers = get_areas_hulls_centers(map_name)
    # Loop over each frame of the round
    for frame_id, frame in tqdm(enumerate(frames)):
        fig, axis = plot_map(map_name=map_name, map_type=map_type, dark=dark)
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)
        # Grab the position token
        tokens = generate_position_token(map_name, frame)
        # Loop over each named area
        for area_id, area in enumerate(sorted(area_points)):
            text = ""
            # Plot each area twice. Once for each side. If both sides have players in one tile the colors get added as
            # the alphas arent 1.
            for side in ["t", "ct"]:
                # Dont plot the "" area as it stretches across the whole map and messes with clarity
                if not area:
                    # axis.plot(np.NaN, np.NaN, label="None_" + tokens[side + "Token"][i])
                    continue
                # Plot the hull of the area
                axis.plot(
                    hulls[area][:, 0],
                    hulls[area][:, 1],
                    "-",
                    # Set alpha and color depending on players inside.
                    # 0 players from this team means low alpha gray
                    # Otherwise blue/red with darkness depending on number of players
                    alpha=0.8 if int(tokens[side + "Token"][area_id]) > 0 else 0.2,
                    c=colors[side][int(tokens[side + "Token"][area_id])],
                    lw=3,
                )
                if tokens[side + "Token"][area_id] != "0":
                    text += side + ":" + tokens[side + "Token"][area_id] + " "
            # If there are any players inside the area plot a text
            if int(tokens["ctToken"][area_id]) + int(tokens["tToken"][area_id]) > 0:
                axis.text(
                    centers[area][0] - (centers[area][0] - hulls[area][0][0]) / 1.5,
                    centers[area][1],
                    text,
                    c="#EB0841",
                    size="x-small",
                )
        image_files.append(f"csgo_tmp/{frame_id}.png")
        fig.savefig(image_files[-1], dpi=300, bbox_inches="tight")
        plt.close()
    images = []
    for file in image_files:
        images.append(imageio.imread(file))
    imageio.mimsave(filename, images, fps=fps)
    shutil.rmtree("csgo_tmp/")
    return True


def plot_map_areas(output_path, map_name="de_ancient", map_type="original", dark=False):
    """Plot all named areas in the given map.

    Args:
        output_path (string): Path to the output folder
        map_name (string): Map to search
        map_type (string): "original" or "simpleradar"
        dark (boolean): Only for use with map_type="simpleradar". Indicates if you want to use the SimpleRadar dark map type

    Returns:
        None, saves .png
    """
    logging.info(f"Plotting areas for {map_name}")
    area_points = collections.defaultdict(list)
    fig, axis = plot_map(map_name=map_name, map_type=map_type, dark=dark)
    fig.set_size_inches(19.2, 10.8)
    # Grab points, hull and centroid for each named area
    area_points, hulls, centers = get_areas_hulls_centers(map_name)
    for area in area_points:
        # Dont plot the "" area as it stretches across the whole map and messes with clarity
        # but add it to the legend
        if not area:
            axis.plot(np.NaN, np.NaN, label="None")
            continue
        axis.plot(hulls[area][:, 0], hulls[area][:, 1], "-", lw=3, label=area)
        # Add name of area into its middle
        axis.text(
            centers[area][0] - (centers[area][0] - hulls[area][0][0]) / 1.5,
            centers[area][1],
            area,
            c="#EB0841",
        )
    # Place legend outside of the plot
    handles, labels = axis.get_legend_handles_labels()
    lgd = axis.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1.01))
    plt.savefig(
        os.path.join(output_path, f"hulls_{map_name}.png"),
        bbox_inches="tight",
        bbox_extra_artists=(lgd,),
        dpi=1000,
    )
    plt.close()


def plot_map_tiles(output_path, map_name="de_ancient", map_type="original", dark=False):
    """Plot all navigation mesh tiles in a given map.

    Args:
        output_path (string): Path to the output folder
        map_name (string): Map to search
        map_type (string): "original" or "simpleradar"
        dark (boolean): Only for use with map_type="simpleradar". Indicates if you want to use the SimpleRadar dark map type

    Returns:
        None, saves .png
    """
    logging.info(f"Plotting tiles for {map_name}")
    fig, axis = plot_map(map_name=map_name, map_type=map_type, dark=dark)
    fig.set_size_inches(19.2, 10.8)
    # Loop over each nav mesh tile
    for a in NAV[map_name]:
        area = NAV[map_name][a]
        try:
            area["southEastX"] = position_transform(map_name, area["southEastX"], "x")
            area["northWestX"] = position_transform(map_name, area["northWestX"], "x")
            area["southEastY"] = position_transform(map_name, area["southEastY"], "y")
            area["northWestY"] = position_transform(map_name, area["northWestY"], "y")
        except KeyError:
            pass
        # Get its lower left points, height and width
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
        axis.add_patch(rect)
    plt.savefig(
        os.path.join(output_path, f"tiles_{map_name}.png"),
        bbox_inches="tight",
        dpi=1000,
    )
    plt.close()


def main(args):
    """Uses awpy and the extension functions defined in this module to plots player positions in 1 or 10 rounds, position tokens in 1 round
    and each maps nav mesh and named areas."""
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
    os.chdir(options.output)
    logging.info("file %s", options.file)
    name, number_id = get_ids(os.path.basename(options.file), -1)
    logging.info("name %s", name)
    logging.info("number_id %s", number_id)
    json_path = os.path.join(options.output, name + ".json")
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
    demo_map_name = data["mapName"]

    round_num = 10
    frames = data["gameRounds"][round_num]["frames"]
    base_name = f"{name}_{round_num}"
    plot_round_file = os.path.join(options.output, f"{base_name}_normal_positions.gif")
    plot_token_file = os.path.join(options.output, f"{base_name}__token_positions.gif")

    logging.info("Generating token round file.")
    plot_round_tokens(
        filename=plot_token_file,
        frames=frames,
        map_name=demo_map_name,
        map_type="simpleradar",
        dark=False,
        fps=2,
    )
    logging.info("Generating round file.")
    plot_round(
        filename=plot_round_file,
        frames=frames,
        map_name=demo_map_name,
        map_type="simpleradar",
        dark=False,
        fps=2,
    )

    n_rounds = 10
    rounds_file_diff_play = os.path.join(
        options.output, f"{base_name}_{n_rounds}_rounds_different_players.gif"
    )
    rounds_file_same_play = os.path.join(
        options.output, f"{base_name}_{n_rounds}_rounds_same_players.gif"
    )
    nades_file = os.path.join(
        options.output, f"{base_name}_{n_rounds}_rounds_nades.png"
    )
    logging.info("Generating rounds_same_players.")
    plot_rounds_same_players(
        rounds_file_same_play,
        [
            data["gameRounds"][round_num + i]["frames"]
            for i in range(-n_rounds // 2, n_rounds // 2)
        ],
        map_name=demo_map_name,
        map_type="simpleradar",
        sides=["ct", "t"],
        fps=1,
    )
    logging.info("Generating rounds_different_players.")
    plot_rounds_different_players(
        rounds_file_diff_play,
        [
            data["gameRounds"][round_num + i]["frames"]
            for i in range(-n_rounds // 2, n_rounds // 2)
        ],
        map_name=demo_map_name,
        map_type="simpleradar",
        sides=["ct", "t"],
        fps=1,
    )

    logging.info("Generating nades_file.")
    _, _ = plot_nades(
        rounds=[
            data["gameRounds"][round_num + i]
            for i in range(-n_rounds // 2, n_rounds // 2)
        ],
        nades=[
            "Flashbang",
            "HE Grenade",
            "Smoke Grenade",
            "Molotov",
            "Incendiary Grenade",
        ],
        side="CT",
        map_name=demo_map_name,
    )
    plt.savefig(nades_file, bbox_inches="tight", dpi=300)
    plt.close()

    logging.info("Generating tile and area files for all maps.")
    for map_name in NAV:
        plot_map_areas(
            output_path=options.output,
            map_name=map_name,
            map_type="simpleradar",
            dark=False,
        )
        plot_map_tiles(
            output_path=options.output,
            map_name=map_name,
            map_type="simpleradar",
            dark=False,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
