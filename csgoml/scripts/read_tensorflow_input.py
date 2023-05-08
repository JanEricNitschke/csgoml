"""Example script utilizing trajectory_[handler,clusterer,predictor] to read in inputs from  tensorflow_input_preparation.py
and builds/trains DNNs to predict the round winner based on player trajectory data as well as clusters rounds by trajectory/high level strategies.
"""
#!/usr/bin/env python

import os
import logging
import argparse
import sys
from csgoml.trajectories import trajectory_handler, trajectory_clusterer

# import trajectory_predictor


def main(args):
    """Read input prepared by tensorflow_input_preparation.py and builds/trains DNNs to predict the round winner based on player trajectory data."""
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
            format="%(asctime)s %(levelname)-8s - %(funcName)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    random_state = options.randomstate

    # Read in the prepared json file.
    # file = (
    #     r"E:\PhD\MachineLearning\CSGOData\ParsedDemos\\"
    #     + options.map
    #     + r"\\Analysis\\Prepared_Input_Tensorflow_"
    #     + options.map
    #     + ".json"
    # )
    file = (
        "D:\\CSGO\\Demos\\Maps\\"
        + options.map
        + "\\Analysis\\Prepared_Input_Tensorflow_"
        + options.map
        + ".json"
    )

    # analysis_path = os.path.join(
    #     r"E:\PhD\MachineLearning\CSGOData\ParsedDemos", options.map, "Analysis"
    # )
    analysis_path = os.path.join(
        r"D:\CSGO\Demos\Maps", options.map, "Analysis"
    )
    map_name = "de_" + options.map

    handler = trajectory_handler.TrajectoryHandler(
        json_path=file, random_state=random_state, map_name=map_name
    )

    for info in handler.aux:
        logging.debug(info)
        logging.debug(handler.aux[info])
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
    traj_config = ("area", 1000, 10, "T", False)
    # clust_config = {
    #     "do_histogram": False,
    #     "n_bins": 50,
    #     "do_knn": False,
    #     "knn_ks": [2, 3, 4, 5, 10, 20, 50, 100, 200, 400, 500, 600],
    #     "plot_all_trajectories": False,
    #     "do_dbscan": True,
    #     "dbscan_eps": 1,
    #     "dbscan_minpt": 400,
    #     "do_kmed": False,
    #     "kmed_n_clusters": 6,
    # }
    clust_config = {
        "do_histogram": False,
        "n_bins": 50,
        "do_knn": False,
        "knn_ks": [2, 3, 4, 5, 10, 20, 50, 100, 200, 400, 500, 600],
        "plot_all_trajectories": False,
        "do_dbscan": False,
        "dbscan_eps": 500,
        "dbscan_minpt": 2,
        "do_kmed": False,
        "kmed_n_clusters": 3,
    }
    logging.info(
        clusterer.do_clustering(
            trajectory_config=traj_config, clustering_config=clust_config
        )
    )

    # predictor = trajectory_predictor.TrajectoryPredictor(
    #     analysis_path=analysis_path,
    #     trajectory_handler=handler,
    #     random_state=random_state,
    #     map_name=map_name,
    # )
    # traj_config = ("position", 20, "T", False)
    # dnn_config = {
    #     "batch_size": 32,
    #     "learning_rate": 0.00007,
    #     "epochs": 50,
    #     "patience": 5,
    #     "nodes_per_layer": 32,
    # }
    # dnn_config = {}
    # predictor.do_predicting(trajectory_config=traj_config, dnn_config=dnn_config)


if __name__ == "__main__":
    main(sys.argv[1:])
