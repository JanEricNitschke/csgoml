#!/usr/bin/env python
"""Example script utilizing trajectory_[handler,clusterer,predictor].

Reads in inputs from  tensorflow_input_preparation.py and builds/trains DNNs to predict
the round winner based on player trajectory data
as well as clusters rounds by trajectory/high level strategies.
"""


import argparse
import logging
import os
import sys
from typing import TYPE_CHECKING

from csgoml.helpers import setup_logging
from csgoml.trajectories import trajectory_clusterer, trajectory_handler

if TYPE_CHECKING:
    from csgoml.types import UserClusteringConfig, UserTrajectoryConfig


def main(args: list[str]) -> None:
    """Read input prepared by tensorflow_input_preparation.py.

    And builds/trains DNNs to predict the round winner based on player trajectory data.
    """
    parser = argparse.ArgumentParser("Analyze the early mid fight on inferno")
    parser.add_argument(
        "-d", "--debug", action="store_true", default=False, help="Enable debug output."
    )
    parser.add_argument("-m", "--map", default="cache", help="Map to analyze")
    parser.add_argument(
        "-l",
        "--log",
        default=r"D:\CSGO\ML\CSGOML\logs\ReadTensorflowInput.log",
        help="Path to output log.",
    )
    parser.add_argument(
        "--randomstate", type=int, default=123, help="Random state for train_test_split"
    )
    options = parser.parse_args(args)

    setup_logging(options)

    random_state = options.randomstate

    file = (
        "D:\\CSGO\\Demos\\Maps\\"
        + options.map
        + "\\Analysis\\Prepared_Input_Tensorflow_"
        + options.map
        + ".json"
    )

    analysis_path = os.path.join(r"D:\CSGO\Demos\Maps", options.map, "Analysis")
    map_name = f"de_{options.map}"

    handler = trajectory_handler.TrajectoryHandler(
        json_path=file, random_state=random_state, map_name=map_name
    )

    for key, info in handler.aux.items():
        logging.debug(key)
        logging.debug(info)
    logging.debug(handler.datasets["token"].shape)
    logging.debug(handler.datasets["token"])
    logging.debug(handler.datasets["position"].shape)
    logging.debug(handler.datasets["position"])

    clusterer = trajectory_clusterer.TrajectoryClusterer(
        analysis_path=analysis_path,
        trajectory_handler=handler,
        random_state=random_state,
        map_name=map_name,
    )
    traj_config: UserTrajectoryConfig = {
        "coordinate_type_for_distance": "area",
        "n_rounds": 1000,
        "side": "T",
        "time": 10,
        "dtw": False,
    }
    clust_config: UserClusteringConfig = {
        "do_histogram": False,
        "n_bins": 50,
        "do_knn": False,
        "plot_all_trajectories": False,
        "knn_ks": [2, 3, 4, 5, 10, 20, 50, 100, 200, 400, 500, 600],
        "do_dbscan": False,
        "dbscan_eps": 500,
        "dbscan_minpt": 2,
        "do_kmed": False,
        "kmed_n_clusters": 3,
    }
    logging.info(
        clusterer.do_clustering(traj_config=traj_config, clust_config=clust_config)
    )


if __name__ == "__main__":
    main(sys.argv[1:])
