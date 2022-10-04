"""Read input prepared by tensorflow_input_preparation.py and builds/trains DNNs to predict the round winner based on player trajectory data.

    Requires as input a dataframe/json that for each round contains the winner and a dataframe of the players trajectories/ position token trajectories.

    Typical usage example:

    predictor = TrajectoryPredictor(
        file,
        times=[10, 160, 10],
        sides=["CT","T","BOTH"],
        random_state=123,
        example_id=22,
    )
    predictor.generate_data_sets()
    predictor.plot_all_models()
"""
#!/usr/bin/env python

import math
import itertools
import shutil
import collections
import os
import logging
import argparse
import sys
import random
from collections import defaultdict
import pandas as pd
import numpy as np
import json
import imageio
from sympy.utilities.iterables import multiset_permutations
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt
import matplotlib as mpl
from awpy.data import NAV
from multiprocessing import Pool
from awpy.visualization.plot import (
    plot_map,
    position_transform,
    get_player_id,
    get_shortest_distances_mapping,
    plot_positions,
)
from awpy.analytics.nav import (
    area_distance,
    find_closest_area,
    position_state_distance,
    token_state_distance,
    tree,
)
from tqdm import tqdm
from nav_utils import (
    generate_centroids,
    get_area_distance_matrix,
)


class TrajectoryPredictor:
    """Read input prepared by tensorflow_input_preparation.py and builds/trains DNNs to predict the round winner based on player trajectory data.

    Requires as input a dataframe/json that for each round contains the winner and a dataframe of the players trajectories/ position token trajectories.

    Attributes:
        input: Path to the json input file containing all the trajectory data for every round on a given map.
        datasets: Dictionary of datasets derived from input. split by Pos/Token, CT,T,Both, time, train/test/val. Todo: supervised or unsupervised and move this to a proper database
        models: Dictionary of models. split by Pos/Token, CT,T,Both, time Todo: supervised or unsupervised and move this to a proper database
        times: List of time information. (lowest, highest, stepsize) Default is (10,160,10)
        sides: List of side configurations to consider. Possible are BOTH, CT and T
        random_state: Integer for random_states
        complete_dataframe: Pandas dataframe generated from input.
        example_id: Integer used for debugging output.
    """

    def __init__(
        self,
        prepared_input,
        times=None,
        sides=None,
        random_state=None,
        example_id=None,
        map_name="de_ancient",
    ):
        self.map_name = map_name
        if random_state is None:
            self.random_state = random.randint(1, 10**8)
        else:
            self.random_state = random_state

        if times is None:
            self.times = [20, 20, 10]  # [10,160,10]
        else:
            self.times = times

        self.sides = ["CT", "T", "BOTH"]
        if sides:
            self.sides = sides

        self.input = prepared_input
        if os.path.isfile(self.input):
            with open(self.input, encoding="utf-8") as pre_analyzed:
                self.complete_dataframe = pd.read_json(pre_analyzed)
        else:
            logging.error("File %s does not exist!", self.input)
            raise FileNotFoundError("File does not exist!")

        logging.debug("Initial dataframe:")
        logging.debug(self.complete_dataframe)

        def tree():
            def the_tree():
                return defaultdict(the_tree)

            return the_tree()

        # Todo: Store this in a proper database (spanner?)
        # Todo: Also save models to file and read already existing models back in
        self.datasets = tree()
        self.models = tree()

        self.example_id = example_id

    def __get_configurations(self):
        # coordinates = ["tokens", "positions"]
        coordinates = ["positions"]
        time = np.arange(
            self.times[0], self.times[1] + self.times[2] // 2, self.times[2]
        )
        return itertools.product(self.sides, time, coordinates)

    def __transform_ticks_to_seconds(self, tick, first_tick):
        """Transforms a tick value to a corresponding second value given a start_tick

        There is a tick every 128 seconds. Given the start tick the second value of the current tick is calculated as
        (tick-first_tick)/128

        Args:
            tick: An integer corresponding to the current tick
            first_tick: An integer corresponding to the reference tick from which the time in seconds should be calculated

        Returns:
            An integer corresponding to the seconds passed between first_tick and tick
        """
        result = int((int(tick) - int(first_tick)) / 128)
        return result

    def __transform_to_data_frame(self, json_format_dict):
        """Transforms a json dictionary to a pandas dataframe and converts ticks to seconds.

        Args:
            json_format_dict: A dictionary in json format holding information player positions during a CS:GO round

        Returns:
            A pandas dataframe corresponding to the input json with the tick values transformed to seconds.
        """
        return_dataframe = pd.DataFrame(json_format_dict)
        first_tick = int(return_dataframe.iloc[0]["Tick"])
        return_dataframe["Tick"] = return_dataframe["Tick"].apply(
            self.__transform_ticks_to_seconds, args=(first_tick,)
        )
        # return_dataframe = return_dataframe.apply(self.regularize_coordinatesdf)
        return return_dataframe

    def __pad_to_length(self, position_df, time):
        """Pad or shorten a pandas dataframe to the given length.

        If the given time is larger than the current size the dataframe is expanded to that size.
        Values of last valid column are used to pad

        If the time is smaller then the dataframe is cut off at that length.

        Args:
            position_df: A dataframe of player positions during a round
            time: Length the dataframe should have after padding/shortening

        Returns:
            A dataframe padded/cut off to have the given length
        """
        if time > len(position_df):
            # Pad with last entry
            # idx = np.minimum(np.arange(time), len(position_df) - 1)
            # position_df=position_df.iloc[idx]
            position_df = position_df.reindex(range(time), fill_value=0.0, method="pad")
        else:
            # Cut if the required size is smaller.
            position_df = position_df.head(time)
        return position_df

    def __generate_array_for_tokens(self, position_df, token, time):
        """Transforms a dataframe of player positions and tokens into an array corresponding to the token through the time steps.

                Input dataframe is of the shape:
                DEBUG:root:    Tick                                     token               CTtoken  CTPlayer1Alive      CTPlayer1Name  CTPlayer1x  CTPlayer1y  CTPlayer1z  CTPlayer2Alive  ... TPlayer4Name  TPlayer4x  TPlayer4y
        0      0  0000050000000000000000000000000000005000  00000500000000000000               1  EIQ-nickelback666    0.067664    0.924986    0.089216               1  ...    1WIN.Polt  -0.080275  -0.849314  -0.923190
        1      1  0000050000000000000000000000000000005000  00000500000000000000               1  EIQ-nickelback666    0.094994    0.811096    0.247608               1  ...    1WIN.Polt  -0.093063  -0.810903  -0.878734
        2      2  1000004000000000000000000000010000004000  10000040000000000000               1  EIQ-nickelback666    0.122323    0.697205    0.406001               1  ...    1WIN.Polt  -0.105850  -0.772492  -0.834277
        3      3  1000004000000000000000000000010000004000  10000040000000000000               1  EIQ-nickelback666    0.225975    0.684219    0.430871               1  ...    1WIN.Polt  -0.116979  -0.745519  -0.905904
        4      4  1310000000000000000000000000030000002000  13100000000000000000               1  EIQ-nickelback666    0.329626    0.671234    0.455740               1  ...    1WIN.Polt  -0.128107  -0.718546  -0.977531
        ..   ...                                       ...                   ...             ...                ...         ...         ...         ...             ...  ...          ...        ...        ...        ...
        84    84  0001000000000000000000000000000000000000  00010000000000000000               0  EIQ-nickelback666   -0.025643    0.032743    0.189914               0  ...    1WIN.Polt  -0.918829   0.537522   0.252183
        85    85  0001000000000000000000000000000000000000  00010000000000000000               0  EIQ-nickelback666   -0.025643    0.032743    0.189914               0  ...    1WIN.Polt  -0.918829   0.537522   0.252183
        86    86  0001000000000000000000000000000000000000  00010000000000000000               0  EIQ-nickelback666   -0.025643    0.032743    0.189914               0  ...    1WIN.Polt  -0.918829   0.537522   0.252183
        87    87  0001000000000000000000000000000000000000  00010000000000000000               0  EIQ-nickelback666   -0.025643    0.032743    0.189914               0  ...    1WIN.Polt  -0.918829   0.537522   0.252183
        88    88  0000000100000000000000000000000000000000  00000001000000000000               0  EIQ-nickelback666   -0.025643    0.032743    0.189914               0  ...    1WIN.Polt  -0.918829   0.537522   0.252183

                The dataframe is reduced the the column of the given token (token,CTtoken,Ttoken)
                DEBUG:root:      token
        0      0000050000000000000000000000000000005000
        1      0000050000000000000000000000000000005000
        2      1000004000000000000000000000010000004000
        3      1000004000000000000000000000010000004000
        4      1310000000000000000000000000030000002000
        ..   ...                                       ...
        84     0001000000000000000000000000000000000000
        85     0001000000000000000000000000000000000000
        86     0001000000000000000000000000000000000000
        87     0001000000000000000000000000000000000000
        88     0000000100000000000000000000000000000000

                Each row is then transformed into a id array of integers:
        0    [0,0,0,0,0,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,0,0,0]
                Then each value is divided by 5, the df padded to a given length and returned back into a multy dimensional array.

                Args:
                    position_df: A dataframe of player positions during a round.
                    token: A string clarifying which token should be used.
                    time: The number of rows the final array should have.

                Returns:
                    A two dimensional numpy array containing the transformed token at each timestep
        """
        # Split the token strings into a columns so that there is one column for each integer in the token string
        # Only work with the token entry of the df
        position_df = position_df[[token]]
        # Initialize the numpy array
        # First value is number of rows, second number of columns
        return_array = np.zeros(tuple([time, len(position_df.iloc[0][token])]))
        # Transform the string into a list of the individual letters(ints) in the string
        position_df = position_df[token].apply(list)
        # Convert the dataframe to a list and back into a df to have one column for each entry of the string
        position_df = pd.DataFrame(position_df.tolist())
        # Convert the individual string into the respective integers
        # Divide numbers by 5 so they are all from 0 to 1
        # TODO: renable this?
        # position_df = position_df.applymap(lambda x: float(x) / 5)
        # Pad the df to the specified length
        position_df = self.__pad_to_length(position_df, time)
        # Convert to numpy array
        # return_array=position_df.to_numpy()
        for ind, column in enumerate(position_df.columns):
            return_array[:, ind] = position_df[column].to_numpy()
        return return_array

    def __modify_data_frame_shape(self, position_df, configuration):
        """Transforms data frame by throwing away all unnecessary features
        and then turning it into a (multidimensional) array.

        If coordinates is true then the result is a 4d array where for each timestep there is a 3d array representing team, player number and the feature.
        If it is false then for each time step there is an array corresponding to all token characters

        Args:
            position_df: A dataframe of player positions during a round.
            configuration: tuple of:
                team: A string indicating whether to include positions for players on the CT side, T side or both sides
                time: An integer indicating
                coordinates: A string indicating whether player coordinates should be used directly ("positions") or the summarizing tokens ("tokens") instead.

        Returns:
            A multi-dimensional numpy array containing split tokens or player positions (organized by team, playnumber, coordinate)
        """
        team, time, coordinates = configuration
        # Throw away all unneeded columns from the dataframe and then convert it into a numpy array
        # If coordinates is false than it is a 1-D array of the tokens for each timestep
        # If coordinates is true than it is a 4-D array with the first index representing the timestep, the second the team, the third the playernumber in that team and the fourth the feature.
        position_df.reset_index(drop=True, inplace=True)
        # Set length of dataframe to make sure all have the same size
        # Pad each column if set size is larger than dataframe size
        # If the full coordinates should be used then for each player their alive status as well as their x,y,z coordinates are kept, everything else (player names and tokens) is discarded.
        if coordinates == "positions":
            featurelist = ["x", "y", "z"]
            # featurelist=["Alive","x","y","z"]
            position_df = self.__pad_to_length(position_df, time)
            if team == "BOTH":
                sides = ["CT", "T"]
                # Dimension with size 2 for teams
                dimensions = [
                    time,
                    2,
                    5,
                    len(featurelist),
                ]  # time for timesteps, 5 for players, len(featurelist) for features
            else:
                sides = [team]
                # Dimension with size 1 for team
                dimensions = [time, 1, 5, len(featurelist)]

            # Only keep the information for the side(s) that should be used.
            return_array = np.zeros(tuple(dimensions))
            for side in sides:
                for number in range(1, 6):
                    for feature in featurelist:
                        return_array[
                            :, sides.index(side), number - 1, featurelist.index(feature)
                        ] = position_df[
                            side + "Player" + str(number) + feature
                        ].to_numpy()
        else:
            # Remove all columns from the dataframe except for the column of interest
            # Split the token strings into a columns so that there is one column for each integer in the token string
            if team == "BOTH":
                return_array = self.__generate_array_for_tokens(
                    position_df, "token", time
                )
            else:
                return_array = self.__generate_array_for_tokens(
                    position_df, team + "token", time
                )
        return return_array

    def __get_maximum_length(self, position_dfs):
        """Get the maximum length of all rounds for a given map.

        For a given maps dataframe containing information about every round recorded determine the length of eachof the rounds.
        Get the maximum of the lengths of all rounds.

        Args:
            position_dfs: A dataframe containing information about every recorded round on the map including another dataframe of all players trajectories.

        Returns:
            The maximum of the lengths of all recorded rounds
        """
        lengths = position_dfs["position_df"].apply(len)
        return lengths.max()

    def __transform_dataframe_to_arrays(self, dataframe):
        """Transforms the dataframe containing one entry per recorded round into separate arrays for the prediction label and feature

        Args:
            dataframe: a dataframe containing on entry per recorded round. Amongst others the winner of the round and a dataframe containing trajectory data.

        Returns:
            Two separate numpy arrays label_array and feature_array. The first is a 1-D array containing the winner of each round.
            The second is a multi-dimensional one containing the desired features across all timesteps for each round.
        """
        label_array = dataframe["Winner"].to_numpy()
        feature_array = np.zeros(
            tuple(
                [len(dataframe["position_df"])]
                + list(dataframe.iloc[0]["position_df"].shape)
            )
        )
        for index, array in dataframe["position_df"].iteritems():
            feature_array[index] = array
        return label_array, feature_array

    def get_token_model(self, nodes_per_layer, input_shape):
        """Generate a LSTM network to predict the winner of a round based on position-token trajectory

        Args:
            nodes_per_layer: An integer determining how many nodes each network layer should have
            input_shape: The exact shape of the network input will have

        Returns:
            The sequantial tf.keras LSTM model
        """
        # Todo more flexibility when defining models
        model = tf.keras.Sequential(
            [
                layers.LSTM(nodes_per_layer, input_shape=input_shape),
                layers.Dense(
                    nodes_per_layer, activation="relu"
                ),  # ,kernel_regularizer=tf.keras.regularizers.l2(0.001)
                layers.Dropout(0.2),
                layers.Dense(
                    nodes_per_layer, activation="relu"
                ),  # ,kernel_regularizer=tf.keras.regularizers.l2(0.001)
                layers.Dropout(0.2),
                layers.Dense(1),
            ]
        )
        return model

    def get_coordinate_model(self, nodes_per_layer, input_shape):
        """Generate a LSTM network to predict the winner of a round based on player trajectories

        Args:
            nodes_per_layer: An integer determining how many nodes each network layer should have
            input_shape: The exact shape of the network input will have

        Returns:
            The sequantial tf.keras CONV2D + LSTM model
        """
        # Todo more flexibility when defining models
        pooling_size = (2, 1)
        model = tf.keras.Sequential(
            [
                layers.TimeDistributed(
                    layers.Conv2D(
                        nodes_per_layer / 2,
                        pooling_size,
                        activation="elu",
                        padding="same",
                    ),
                    input_shape=input_shape,
                ),
                layers.TimeDistributed(
                    layers.AveragePooling2D(pool_size=pooling_size, strides=(1, 1))
                ),
                layers.TimeDistributed(
                    layers.Conv2D(
                        nodes_per_layer, pooling_size, activation="elu", padding="same"
                    ),
                    input_shape=input_shape,
                ),
                layers.TimeDistributed(
                    layers.AveragePooling2D(pool_size=pooling_size, strides=(1, 1))
                ),
                layers.TimeDistributed(layers.Flatten()),
                layers.LSTM(nodes_per_layer),
                layers.Dense(
                    nodes_per_layer, activation="relu"
                ),  # ,kernel_regularizer=tf.keras.regularizers.l2(0.001)
                layers.Dropout(0.2),
                layers.Dense(
                    nodes_per_layer, activation="relu"
                ),  # ,kernel_regularizer=tf.keras.regularizers.l2(0.001)
                layers.Dropout(0.2),
                layers.Dense(1),
            ]
        )
        return model

    def regularize_coordinates(self, coordinate, minimum, maximum):
        """Regularizes coordinates to be between -1 and 1

        If the map is in the awpy NAV data then minimum and maximum correspondt to actual min and max values.
        If the map is not then they are flatly taken to be +-2000/2000/200 for x, y and z respectively.
        In that case the coordinate can end up outside of -1 to 1 in some cases.

        Args:
            coordinate: Float representing a player coordinate
            minimum: The minimal possible value of this coordinate
            maximum: The maximal possible value of thos coordinate

        Returns:
            A float corresponding to a rescaled coordinate that is always between -1 and 1
        """
        shift = (maximum + minimum) / 2
        scaling = (maximum - minimum) / 2
        return (coordinate - shift) / scaling

    def get_extremes_from_NAV(self):
        """Determines the maximal and mininmal possible x, y and z values for a given map

        Look through the awpy NAV data of the given map and search all recorded areas.
        Keep track of minimum and maximum value for each coordinate.
        If the map does not exist in the NAV data then use default values of +-2000/2000/200 for x, y and z respectively

        Args:
            map_name: String of the currently investigated map

        Returns:
            Two dictionary containing the minimum and maximum determined values for each coordinate.
        """
        if self.map_name not in NAV:
            minimum = {"x": -2000, "y": -2000, "z": -200}
            maximum = {"x": 2000, "y": 2000, "z": 200}
        else:
            minimum = {"x": sys.maxsize, "y": sys.maxsize, "z": sys.maxsize}
            maximum = {"x": -sys.maxsize, "y": -sys.maxsize, "z": -sys.maxsize}
            for area in NAV[self.map_name]:
                for feature in ["x", "y", "z"]:
                    for corner in ["northWest", "southEast"]:
                        maximum[feature] = max(
                            NAV[self.map_name][area][corner + feature.upper()],
                            maximum[feature],
                        )
                        minimum[feature] = min(
                            NAV[self.map_name][area][corner + feature.upper()],
                            minimum[feature],
                        )
        return minimum, maximum

    def regularize_coordinatesdf(self, position_df):
        """Apply coordinate regularization to all coordinate positions in the dataframe.

        Gather minimum and maximum values and regularize each coordinate to be between -1 and 1

        Args:
            position_df: Dataframe containing every players position and status for each time step
            map_name: String of the current maps name

        Returns:
            dataframe after player coordinate regularization
        """
        minimum, maximum = self.get_extremes_from_NAV()
        for side in ["CT", "T"]:
            for number in range(1, 6):
                for feature in ["x", "y", "z"]:
                    position_df[side + "Player" + str(number) + feature] = position_df[
                        side + "Player" + str(number) + feature
                    ].apply(
                        self.regularize_coordinates,
                        args=(minimum[feature], maximum[feature]),
                    )
        return position_df

    def generate_train_test_val_datasets(
        self, this_dataframe, configuration, batch_size=32
    ):
        """Generate the train, test and validation sub datasets from the total dataset based on which features, side and time window should be considered.

        Args:
            this_dataframe: A pandas dataframe containing a row for each round and columns for the winning side and a dataframe of player/token trajectories
            configuration: A tuple of (side,time,coordinate) determining what should be in the output dataframe.
                Side: A string clarifying about which side information should be included: CT, T, or BOTH
                time: An integer determining up to which second in the round trajectory information should be included
                coordinates: A string determining if individual players positions ("positions") or aggregate tokens ("tokens) should be used
            batch_size: An integer determining the size that the datasets should be batched to

        Returns:
            None (datasets are added to self.datasets nested dict)
        """
        logging.debug("Generating dataset for configuration %s.", configuration)
        # Shuffle and split dataframe into training, test and validation set
        # set aside 20% of train and test data for evaluation
        side, time, coordinate = configuration
        train_df, test_df = train_test_split(
            this_dataframe,
            test_size=0.2,
            shuffle=True,
            random_state=self.random_state,
        )
        # Use the same function above for the validation set  0.25 x 0.8 = 0.2
        train_df, val_df = train_test_split(
            train_df, test_size=0.25, shuffle=True, random_state=self.random_state
        )

        train_df["position_df"] = train_df["position_df"].apply(
            self.__modify_data_frame_shape, args=(configuration,)
        )
        val_df["position_df"] = val_df["position_df"].apply(
            self.__modify_data_frame_shape, args=(configuration,)
        )
        test_df["position_df"] = test_df["position_df"].apply(
            self.__modify_data_frame_shape, args=(configuration,)
        )

        logging.debug("Dataframe after cleanup")
        logging.debug(train_df)
        if self.example_id is not None:
            logging.debug("Example for position_df dataframe entry after cleaning.")
            logging.debug(train_df.iloc[self.example_id]["position_df"])

        train_df.reset_index(drop=True, inplace=True)
        val_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)

        test_labels, test_features = self.__transform_dataframe_to_arrays(test_df)
        val_labels, val_features = self.__transform_dataframe_to_arrays(val_df)
        train_labels, train_features = self.__transform_dataframe_to_arrays(train_df)

        logging.info(
            self.trajectory_distance(train_features[0], train_features[1], "geodesic")
        )
        precomputed_matrix_path = os.path.join(
            os.path.dirname(os.path.abspath(self.input)),
            f"pre_computed_round_distances_{self.map_name}.npy",
        )
        if os.path.exists(precomputed_matrix_path):
            logging.info("Loading precomputed distances from file")
            precomputed = np.load(precomputed_matrix_path)
        else:
            logging.info("Precomputing areas")
            plot_array = self.precompute_areas(train_features)
            logging.info(
                "Precomputing all round distances for %s combinations.",
                len(plot_array) ** 2,
            )
            precomputed = np.zeros((len(train_features), len(train_features)))
            for i in range(len(train_features)):
                logging.info(i)
                for j in range(i + 1, len(train_features)):
                    precomputed[i][j] = self.trajectory_distance(
                        plot_array[i],
                        plot_array[j],
                        "geodesic",
                        precomputed_areas=True,
                    )
            np.save(
                precomputed_matrix_path,
                precomputed,
            )
            logging.info("Saved distances to file.")
        precomputed += precomputed.T

        # logging.info("Plotting histogram of distances")
        # self.plot_histogram(
        #     precomputed,
        #     os.path.join(
        #         os.path.dirname(os.path.abspath(self.input)),
        #         f"hist_round_distances_{self.map_name}.png",
        #     ),
        #     n_bins=20,
        # )

        # for k in [2, 3, 4, 5, 10, 20]:
        #     self.plot_knn(k, precomputed)

        distance_variant = "geodesic"
        name = "ClusterTesting_KMed_Geo"
        output = r"D:\CSGO\ML\CSGOML\Plots\multi_round_testing"
        base_name = f"{name}_{configuration[0]}_{configuration[1]}"
        distance_variant = "geodesic"
        rounds_file_diff_play_img_traj = os.path.join(
            output, f"All_rounds_different_players_img_traj.png"
        )

        logging.info("Generating rounds_different_players image trajectory")
        self.plot_rounds_different_players(
            rounds_file_diff_play_img_traj,
            train_features,
            map_name=self.map_name,
            map_type="simpleradar",
            fps=1,
            dist_type=distance_variant,
            image=True,
            trajectory=True,
        )

        # cluster_dict = self.run_kmed(5, precomputed)
        # # self.run_dbscan(1500, 20, precomputed)
        # # cluster_dict = self.run_dbscan(600, 4, precomputed)
        # for cluster_id, rounds in cluster_dict.items():
        #     logging.info(cluster_id)
        #     base_name = f"{name}_{cluster_id}"
        #     rounds_file_diff_play_img_traj = os.path.join(
        #         output, f"{base_name}_rounds_different_players_img_traj.png"
        #     )
        #     logging.info("Generating rounds_different_players image trajectory")
        #     self.plot_rounds_different_players(
        #         rounds_file_diff_play_img_traj,
        #         [train_features[i] for i in rounds],
        #         map_name=self.map_name,
        #         map_type="simpleradar",
        #         fps=1,
        #         dist_type=distance_variant,
        #         image=True,
        #         trajectory=True,
        #     )

        self.datasets[side][time][coordinate]["val"]["features"] = val_features
        self.datasets[side][time][coordinate]["train"]["features"] = train_features
        self.datasets[side][time][coordinate]["test"]["features"] = test_features
        self.datasets[side][time][coordinate]["val"]["labels"] = val_labels
        self.datasets[side][time][coordinate]["train"]["labels"] = train_labels
        self.datasets[side][time][coordinate]["test"]["labels"] = test_labels

        logging.info("Input preparation done for configuration %s", configuration)
        logging.info(train_labels.shape)
        logging.info(train_features.shape)

        self.datasets[side][time][coordinate]["val"][
            "dataset"
        ] = tf.data.Dataset.from_tensor_slices((val_features, val_labels)).batch(
            batch_size
        )
        self.datasets[side][time][coordinate]["train"][
            "dataset"
        ] = tf.data.Dataset.from_tensor_slices((train_features, train_labels)).batch(
            batch_size
        )
        self.datasets[side][time][coordinate]["test"][
            "dataset"
        ] = tf.data.Dataset.from_tensor_slices((test_features, test_labels)).batch(
            batch_size
        )

    def get_model(self, configuration, nodes_per_layer=32, override=False):
        """Grabs or generates a model usable for the specified configuration.

        Args:
            configuration: configuration: A tuple of (side,time,coordinate) determining what dataset the model should be applicable to.
                Side: A string clarifying about which side information is included in the dataset: CT, T, or BOTH
                time: An integer determining up to which second in the round trajectory information is included
                coordinates: A string determining if individual players positions ("positions") or aggregate tokens ("tokens) were used
            nodes_per_layer: An integer specifying how many nodes each layer in the LSTM should have. Currently the same for all layers
            override: A boolean that determines if a new model should be created even though one already exists for the given configuration.
                      Usefull when nodes_per_layer should be updated.
        Returns:
            LSTM network model that is applicable to datasets produced according to the given configuration
        """
        # Todo: More flexibility when defining models
        side, time, coordinate = configuration
        if coordinate in self.models[side][time] and not override:
            model = self.models[side][time][coordinate]
        else:
            if coordinate == "positions":
                model = self.get_coordinate_model(
                    nodes_per_layer,
                    self.datasets[side][time][coordinate]["train"]["features"][0].shape,
                )
            else:
                model = self.get_token_model(
                    nodes_per_layer,
                    self.datasets[side][time][coordinate]["train"]["features"][0].shape,
                )
            self.models[side][time][coordinate] = model
        # Todo: Also save model to file
        return model

    def compile_fit_and_evaluate_model(self, configuration, epochs=50, patience=5):
        """Compiles, fits and evaluates a LSTM network model useable on a dataset generated according to the configuration

        Args:
            configuration: configuration: A tuple of (side,time,coordinate) determining what dataset the model should be applicable to.
                Side: A string clarifying about which side information is included in the dataset: CT, T, or BOTH
                time: An integer determining up to which second in the round trajectory information is included
                coordinates: A string determining if individual players positions ("positions") or aggregate tokens ("tokens) were used
            epochs: An integer indicating for how many epochs the model should be trained
            patience: An integer indicating the early stopping patience that should be used during training

        Returns:
            A tuple of (History, loss, accuracy, entropy)
                History: History object from model.fit
                loss: A float of the loss from model.evaluate on the test dataset
                accuracy: A float of the accuracy from model.evaluate on the test dataset
                entropy: A float of binary crossentropy from model.evaluate on the test dataset
        """
        logging.debug(
            "Compiling, fitting and evaluating model for configuration %s.",
            configuration,
        )
        side, time, coordinate = configuration
        model = self.get_model(configuration)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.00007),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[
                "accuracy",
                tf.keras.losses.BinaryCrossentropy(
                    from_logits=True, name="binary_crossentropy"
                ),
            ],
        )

        model.summary()

        history = model.fit(
            self.datasets[side][time][coordinate]["train"]["dataset"],
            epochs=epochs,
            validation_data=self.datasets[side][time][coordinate]["val"]["dataset"],
            callbacks=tf.keras.callbacks.EarlyStopping(
                monitor="val_binary_crossentropy", patience=patience
            ),
        )

        loss, accuracy, entropy = model.evaluate(
            self.datasets[side][time][coordinate]["test"]["dataset"]
        )
        return (history, loss, accuracy, entropy)

    def plot_model(self, configuration, plot_path=r"D:\CSGO\ML\CSGOML\Plots"):
        """Plots and logs results for training and evaluating the model corresponding to the configuration

        Args:
            configuration: configuration: A tuple of (side,time,coordinate) determining which model should be used.
                Side: A string clarifying about which side information is included in the dataset that the model is applicable to: CT, T, or BOTH
                time: An integer determining up to which second in the round trajectory information is included in the dataset that the model is applicable to
                coordinates: A string determining if individual players positions ("positions") or aggregate tokens ("tokens) were used in the dataset that the model is applicable to
            plot_path: A string of the path of the directory where the resultant plots should be saved to
        Returns:
            None (Logs evaluation loss and accuarcy from the test set and produces plots of training vs val loss and accuracy during training)
        """
        history, loss, accuracy, _ = self.compile_fit_and_evaluate_model(configuration)
        logging.info("Loss: %s", loss)
        logging.info("Accuracy: %s", accuracy)

        history_dict = history.history

        acc = history_dict["accuracy"]
        val_acc = history_dict["val_accuracy"]
        loss = history_dict["loss"]
        val_loss = history_dict["val_loss"]

        epochs = range(1, len(acc) + 1)

        # "bo" is for "blue dot"
        plt.plot(epochs, loss, "bo", label="Training loss")
        # b is for "solid blue line"
        plt.plot(epochs, val_loss, "b", label="Validation loss")
        plt.title("Training and validation loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(
            f"{plot_path}\\train_val_loss_{configuration[0]}_{configuration[1]}_{configuration[2]}.png"
        )
        plt.show()

        plt.plot(epochs, acc, "bo", label="Training acc")
        plt.plot(epochs, val_acc, "b", label="Validation acc")
        plt.title("Training and validation accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend(loc="lower right")
        plt.savefig(
            f"{plot_path}\\train_val_acc_{configuration[0]}_{configuration[1]}_{configuration[2]}.png"
        )
        plt.show()

    def generate_data_sets(self, batch_size=32):
        """Generate datasets for each possible combination of side, time and coordinates settings and add them to self.datasets

        For each configuration entries for val,train,test x labels,features,dataset are created and added to the self.datasets entry of the given configuration.

        Args:
            batch_size: An integer determining the batch_size of the resultant fitting ready dataset.

        Returns:
            None (datasets are directly added to self.datasets)
        """
        # Only keep the label (Winner) and training feature (position_df) and discard the MatchID, Map_name and round number.
        dataframe = self.complete_dataframe[["Winner", "position_df"]]
        # Transform the position_df from a json dict to df and also transform ticks to seconds
        dataframe["position_df"] = dataframe["position_df"].apply(
            self.__transform_to_data_frame
        )
        if self.example_id is not None:
            logging.debug("Example for position_df dataframe entry before cleaning.")
            logging.debug(dataframe.iloc[self.example_id]["position_df"])

        # Todo: Do this completely separately for train, test and validation datasets
        max_time = min(self.__get_maximum_length(dataframe), 160)
        self.times[0] = min(self.times[0], max_time)
        self.times[1] = min(self.times[1], max_time)
        for configuration in self.__get_configurations():
            this_dataframe = dataframe.copy()
            self.generate_train_test_val_datasets(
                this_dataframe, configuration, batch_size
            )
        logging.info("Done generating all possible datasets.")
        logging.debug(self.datasets)

    def generate_all_models(self, nodes_per_layer=32, override=False):
        """Add models for all possible configurations to self.models

        Args:
            nodes_per_layer: An interger indicating how many nodes each layer of the models should have.
            override: A boolean indicating whether already existing models should be overriden. (Used when changing nodes_per_layer)

        Returns:
            None (models are directly added to self.models)
        """
        for configuration in self.__get_configurations():
            _ = self.get_model(
                configuration, nodes_per_layer=nodes_per_layer, override=override
            )

    def plot_all_models(self):
        """Calls self.plot_model for each possible configuration

        Args:
            None (configratuions are taken from self.__get_configurations)

        Returns:
            None (Only logs and produces plots)
        """
        for configuration in self.__get_configurations():
            self.plot_model(configuration)

    def trajectory_distance(
        self,
        trajectory_array_1,
        trajectory_array_2,
        distance_type="geodesic",
        precomputed_areas=False,
    ):
        """Calculates a distance distance between two trajectories

        Args:
            trajectory_array_1: Numpy array with shape (n_Time,2|1, 5, 3) with the first index indicating the team, the second the player and the third the coordinate
            trajectory_array_2: Numpy array with shape (n_Time,2|1, 5, 3) with the first index indicating the team, the second the player and the third the coordinate
            distance_type: String indicating how the distance between two player positions should be calculated. Options are "geodesic", "graph", "euclidean" and "edit_distance"
            precomputed_areas (boolean): Indicates whether the position arrays already contain the precomputed areas in the x coordinate of the position

        Returns:
            A float representing the distance between these two trajectories
        """
        distance = 0
        length = max(len(trajectory_array_1), len(trajectory_array_2))
        if len(trajectory_array_1.shape) > 2.5:
            for time in range(length):
                distance += (
                    position_state_distance(
                        map_name=self.map_name,
                        position_array_1=trajectory_array_1[time]
                        if time in range(len(trajectory_array_1))
                        else trajectory_array_1[-1],
                        position_array_2=trajectory_array_2[time]
                        if time in range(len(trajectory_array_2))
                        else trajectory_array_2[-1],
                        distance_type=distance_type,
                        precomputed_areas=precomputed_areas,
                    )
                    / length
                )
        else:
            for time in range(length):
                distance += (
                    token_state_distance(
                        map_name=self.map_name,
                        token_array_1=trajectory_array_1[time]
                        if time in range(len(trajectory_array_1))
                        else trajectory_array_1[-1],
                        token_array_2=trajectory_array_2[time]
                        if time in range(len(trajectory_array_2))
                        else trajectory_array_2[-1],
                        distance_type=distance_type,
                    )
                    / length
                )
        return distance

    def plot_rounds_different_players(
        self,
        filename,
        frames_list,
        map_name="de_ancient",
        map_type="original",
        dark=False,
        fps=10,
        n_frames=9000,
        dist_type="geodesic",
        image=False,
        trajectory=False,
    ):
        """Plots a list of rounds and saves as a .gif. Each player in the first round is assigned a separate color. Players in the other rounds are matched by proximity.
        Only use untransformed coordinates.

        Args:
            filename (string): Filename to save the gif
            frames_list (list): List of np arrays each frame should have shape (Time_steps,2|1(sides),5,3)
            map_name (string): Map to search
            map_type (string): "original" or "simpleradar"
            dark (boolean): Only for use with map_type="simpleradar". Indicates if you want to use the SimpleRadar dark map type
            fps (integer): Number of frames per second in the gif
            n_frames (integer): The first how many frames should be plotted
            dist_type (string): String indicating the type of distance to use. Can be graph, geodesic, euclidean, manhattan, canberra or cosine
            image (boolean): Boolean indicating whether a gif of positions or a singular image of trajectories should be produced
            trajectory (boolean): Indicates whether the clustering of players should be done for the whole trajectories instead of each individual time step

        Returns:
            True, saves .gif
        """
        if image and not trajectory:
            frame_positions = collections.defaultdict(
                lambda: collections.defaultdict(lambda: collections.defaultdict(list))
            )
            frame_colors = collections.defaultdict(
                lambda: collections.defaultdict(lambda: collections.defaultdict(list))
            )
        elif not image:
            if os.path.isdir("csgo_tmp"):
                shutil.rmtree("csgo_tmp/")
            os.mkdir("csgo_tmp")
            image_files = []
            if trajectory:
                mappings = defaultdict(list)
        else:  # image and trajectory
            f, a = plot_map(map_name=map_name, map_type=map_type, dark=dark)
        # Needed to check if leaders have been fully initialized
        dict_initialized = {0: False, 1: False}
        # Used to store values for each round separately. Needed when a round ends.

        colors_list = {
            1: ["cyan", "yellow", "fuchsia", "lime", "orange"],
            0: ["red", "green", "black", "white", "gold"],
        }

        if trajectory:
            reference_traj = {}
            for side in range(frames_list[0].shape[1]):
                reference_traj[side] = self.precompute_areas(
                    self.transform_to_traj_dimensions(frames_list[0][:, side, :, :])
                )
            for frame_index, frames in enumerate(tqdm(frames_list)):
                # Initialize lists used to store values for this round for this frame
                for side in range(frames.shape[1]):
                    mapping = get_shortest_distances_mapping(
                        self.map_name,
                        reference_traj[side],
                        self.precompute_areas(
                            self.transform_to_traj_dimensions(frames[:, side, :, :])
                        ),
                        dist_type=dist_type,
                        trajectory=True,
                        precomputed_areas=True,
                    )
                    if image:
                        for player in range(frames.shape[2]):
                            a.plot(
                                [
                                    position_transform(map_name, x, "x")
                                    for x in frames[:, side, player, 0]
                                ],
                                [
                                    position_transform(map_name, y, "y")
                                    for y in frames[:, side, player, 1]
                                ],
                                c=colors_list[side][mapping[player]],
                                linestyle="-",
                                linewidth=0.5,
                            )
                    else:
                        mappings[side].append(mapping)
            if image:
                a.get_xaxis().set_visible(False)
                a.get_yaxis().set_visible(False)
                f.savefig(filename, dpi=1000, bbox_inches="tight")
                plt.close()
                return True

        # Determine how many frames are there in total
        max_frames = max(frames.shape[0] for frames in frames_list)
        # Build tree data structure for leaders
        # Currently leaders={"t":{},"ct":{}} would probably do as well
        # For each side the keys are a players steamd id + "_" + frame_number in case the same steamid occurs in multiple rounds
        leaders = tree()
        for i in tqdm(range(min(max_frames, int(n_frames)))):
            # Initialize lists used to store things from all rounds to plot for each frame
            if not image:
                positions = []
                colors = []
                markers = []
                alphas = []
                sizes = []
            if not trajectory:
                # Is used to determine if a specific leader has already been seen. Needed when a leader drops out because their round has already ended
                checked_in = set()
                # Loop over all the rounds and update the position and status of all leaders
                for frame_index, frames in enumerate(frames_list):
                    # Check if the current frame has already ended
                    if i in range(frames.shape[0]):
                        for side in range(frames.shape[1]):
                            # Do not do this if leaders has not been fully initialized
                            if dict_initialized[side] is True:
                                for player_index, p in enumerate(frames[i][side]):
                                    player_id = f"{player_index}_{frame_index}_{side}"
                                    if (
                                        player_id not in leaders[side]
                                        or player_id in checked_in
                                    ):
                                        continue
                                    leaders[side][player_id]["pos"] = p
            # Now do another loop to add all players in all frames with their appropriate colors.
            for frame_index, frames in enumerate(frames_list):
                # Initialize lists used to store values for this round for this frame
                if i in range(len(frames)):
                    for side in range(frames.shape[1]):
                        if not trajectory:
                            # If we have already initialized leaders
                            if dict_initialized[side] is True:
                                # Get the positions of all players in the current frame and round
                                current_positions = []
                                for player_index, p in enumerate(frames[i][side]):
                                    current_positions.append(p)
                                # Find the best mapping between current players and leaders
                                mapping = get_shortest_distances_mapping(
                                    self.map_name,
                                    leaders[side],
                                    current_positions,
                                    dist_type=dist_type,
                                )
                        # Now do the actual plotting
                        for player_index, p in enumerate(frames[i][side]):
                            pos = p
                            if trajectory:
                                markers.append(rf"$ {frame_index} $")
                                colors.append(
                                    colors_list[side][
                                        mappings[side][frame_index][player_index]
                                    ]
                                )
                                positions.append(pos)
                                # If we are an alive leader we get opaque and big markers
                                if frame_index == 0:
                                    alphas.append(1)
                                    sizes.append(mpl.rcParams["lines.markersize"] ** 2)
                                # If not we get partially transparent and small ones
                                else:
                                    alphas.append(0.5)
                                    sizes.append(
                                        0.3 * mpl.rcParams["lines.markersize"] ** 2
                                    )
                            else:
                                player_id = f"{player_index}_{frame_index}_{side}"

                                # If the leaders have not been initialized yet, do so
                                if dict_initialized[side] is False:
                                    leaders[side][player_id]["index"] = player_index
                                    leaders[side][player_id]["pos"] = pos

                                # This is relevant for all subsequent frames
                                # If we are a leader we update our values
                                # Should be able to be dropped as we already updated leaders in the earlier loop
                                if player_id in leaders[side]:
                                    # Grab our current player_index from what it was the previous round to achieve color consistency
                                    player_index = leaders[side][player_id]["index"]
                                    # Update our position
                                    leaders[side][player_id]["pos"] = pos
                                # If not a leader
                                else:
                                    # Grab the id of the leader assigned to this player
                                    assigned_leader = mapping[player_index]
                                    # If the assigned leader is now dead or has not been assigned (happens when his round is already over)
                                    # Then we take over that position if we are not also dead
                                    if assigned_leader not in checked_in:
                                        # Remove the previous leaders entry from the dict
                                        old_index = leaders[side][assigned_leader][
                                            "index"
                                        ]
                                        del leaders[side][assigned_leader]
                                        # Fill with our own values but use their prior index to keep color consistency when switching leaders
                                        leaders[side][player_id]["index"] = old_index
                                        leaders[side][player_id]["pos"] = pos
                                        player_index = leaders[side][player_id]["index"]
                                    # If the leader is alive and present or if we are also dead
                                    else:
                                        # We just grab our color
                                        player_index = leaders[side][assigned_leader][
                                            "index"
                                        ]
                                if image:
                                    frame_colors[frame_index][side][
                                        player_index
                                    ].append(colors_list[side][player_index])
                                    frame_positions[frame_index][side][
                                        player_index
                                    ].append(pos)
                                else:
                                    markers.append(rf"$ {frame_index} $")
                                    colors.append(colors_list[side][player_index])
                                    positions.append(pos)
                                    # If we are an alive leader we get opaque and big markers
                                    if (
                                        player_id in leaders[side]
                                        and not player_id in checked_in
                                    ):
                                        alphas.append(1)
                                        sizes.append(
                                            mpl.rcParams["lines.markersize"] ** 2
                                        )
                                    # If not we get partially transparent and small ones
                                    else:
                                        alphas.append(0.5)
                                        sizes.append(
                                            0.3 * mpl.rcParams["lines.markersize"] ** 2
                                        )
                                # If we are a leader we are now checked in so everyone knows our round has not ended yet
                                if player_id in leaders[side]:
                                    checked_in.add(player_id)
                        if not trajectory:
                            # Once we have done our first loop over a side we are initialized
                            dict_initialized[side] = True
            if not image:
                f, _ = plot_positions(
                    positions=positions,
                    colors=colors,
                    markers=markers,
                    alphas=alphas,
                    sizes=sizes,
                    map_name=map_name,
                    map_type=map_type,
                    dark=dark,
                    apply_transformation=True,
                )
                image_files.append(f"csgo_tmp/{i}.png")
                f.savefig(image_files[-1], dpi=300, bbox_inches="tight")
                plt.close()
        if image:
            f, a = plot_map(map_name=map_name, map_type=map_type, dark=dark)
            for frame in frame_positions:
                for side in frame_positions[frame]:
                    for player in frame_positions[frame][side]:
                        a.plot(
                            [
                                position_transform(map_name, x[0], "x")
                                for x in frame_positions[frame][side][player]
                            ],
                            [
                                position_transform(map_name, x[1], "y")
                                for x in frame_positions[frame][side][player]
                            ],
                            c=frame_colors[frame][side][player][0],
                            linestyle="-",
                            linewidth=0.5,
                        )
            a.get_xaxis().set_visible(False)
            a.get_yaxis().set_visible(False)
            f.savefig(filename, dpi=1000, bbox_inches="tight")
            plt.close()
        else:
            images = []
            for file in image_files:
                images.append(imageio.imread(file))
            imageio.mimsave(filename, images, fps=fps)
            # shutil.rmtree("csgo_tmp/")
        return True

    def transform_to_traj_dimensions(self, pos_array):
        """Transforms a numpy array of shape (Time,5,3) to (5,Time,1,1,3) to allow for individual trajectory distances.
        Only needed when trying to get the distance between single player trajectories within "get_shortest_distance_mapping"

        Args:
            pos_array (numpy array): Numpy array with shape (Time, 2|1, 5, 3)

            Returns:
                numpy array
        """
        dimensions = [5, len(pos_array), 1, 1, 3]
        return_array = np.zeros(tuple(dimensions))
        for i in range(5):
            return_array[i, :, :, :, :] = pos_array[:, np.newaxis, np.newaxis, i, :]
        return return_array

    def plot_histogram(self, distance_matrix, path, n_bins=10):
        """Plots a histogram of the distances in the precomputed distance matrix"""
        plt.hist(
            distance_matrix.flatten(), density=False, bins=n_bins
        )  # density=False would make counts
        plt.ylabel("Probability")
        plt.xlabel("Data")
        plt.savefig(path)
        plt.close()

    def plot_knn(self, n_nn, precomputed):
        """Plot k-distance with k = n_nn for precomputed distance matrix"""
        logging.info("NN %s", n_nn)
        neighbors = NearestNeighbors(n_neighbors=n_nn, metric="precomputed")
        neighbors_fit = neighbors.fit(precomputed)
        distances, indices = neighbors_fit.kneighbors(precomputed)
        # logging.info(distances)
        # logging.info(indices)
        distances = np.sort(distances, axis=0)
        distance = distances[:, n_nn - 1]
        plt.plot(distance)
        plt.savefig(
            os.path.join(
                os.path.dirname(os.path.abspath(self.input)),
                f"nearest_neighbors_{n_nn}_distances_{self.map_name}.png",
            )
        )
        plt.close()

    def run_dbscan(self, eps, minpt, precomputed):
        """Run dbscan on the precomputed matrix with the given parameters"""
        dbscan = DBSCAN(eps=eps, min_samples=minpt, metric="precomputed").fit(
            precomputed
        )  # fitting the model
        labels = dbscan.labels_  # getting the labels
        logging.info(labels)
        cluster_dict = defaultdict(list)
        for round_num, cluster in enumerate(labels):
            cluster_dict[cluster].append(round_num)
        logging.info(cluster_dict)
        return cluster_dict

    def run_kmed(self, n_cluster, precomputed):
        """Run dbscan on the precomputed matrix with the given parameters"""
        kmed = KMedoids(n_clusters=n_cluster, metric="precomputed").fit(
            precomputed
        )  # fitting the model
        labels = kmed.labels_  # getting the labels
        logging.info(labels)
        cluster_dict = defaultdict(list)
        for round_num, cluster in enumerate(labels):
            cluster_dict[cluster].append(round_num)
        logging.info(cluster_dict)
        return cluster_dict

    def precompute_areas(self, rounds_array):
        """Precompute the area for every position in the trajectory array"""
        return_array = rounds_array.copy()
        for trajectory_id, trajectory_array in enumerate(rounds_array):
            if len(trajectory_array.shape) > 2.5:
                for time, position_array in enumerate(trajectory_array):
                    for team_id, team in enumerate(position_array):
                        for player_id, player in enumerate(team):
                            return_array[trajectory_id][time][team_id][player_id][
                                0
                            ] = find_closest_area(self.map_name, player)["areaId"]
        return return_array


def main(args):
    """Read input prepared by tensorflow_input_preparation.py and builds/trains DNNs to predict the round winner based on player trajectory data."""
    parser = argparse.ArgumentParser("Analyze the early mid fight on inferno")
    parser.add_argument(
        "-d", "--debug", action="store_true", default=False, help="Enable debug output."
    )
    parser.add_argument("-m", "--map", default="ancient", help="Map to analyze")
    parser.add_argument(
        "-l",
        "--log",
        default=r"D:\CSGO\ML\CSGOML\logs\ReadTensorflowInput.log",
        help="Path to output log.",
    )
    parser.add_argument(
        "--exampleid",
        type=int,
        default=22,
        help="For which round the position_df should be printed when debugging.",
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

    example_index = options.exampleid
    random_state = options.randomstate

    # Time should either an integer or MAX.
    # If it is an integer n that means that the first n timesteps will be considered for each round and the rest discarded.
    # If it is 'MAX' then we look what the longest round in the dataset is and pad all other rounds to that length.
    # Should probably do the Training,Test split before and only look at the max length of the training set. TODO

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
    predictor = TrajectoryPredictor(
        file,
        times=[20, 20, 2],
        sides=["CT"],
        random_state=random_state,
        example_id=example_index,
        map_name="de_" + options.map,
    )
    predictor.generate_data_sets()
    # predictor.plot_all_models()


if __name__ == "__main__":
    main(sys.argv[1:])
