"""Read input prepared by tensorflow_input_preparation.py and builds/trains DNNs to predict the round winner based on player trajectory data.

    Requires as input a dataframe/json that for each round contains the winner and a dataframe of the players trajectories/ position token trajectories.

    Typical usage example:

    todo
"""
#!/usr/bin/env python

import itertools
import os
import logging
import argparse
import sys
import random
from collections import defaultdict
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class TrajectoryPredictor:
    """Read input prepared by tensorflow_input_preparation.py and builds/trains DNNs to predict the round winner based on player trajectory data.

    Requires as input a dataframe/json that for each round contains the winner and a dataframe of the players trajectories/ position token trajectories.

    Attributes:
        input: Path to the json input file containing all the trajectory data for every round on a given map.
        datasets: dictionary of datasets derived from input. split by Pos/Token, CT,T,Both, time, train/test/val. Todo: supervised or unsupervised and move this to a proper database
        models: dictionary of models. split by Pos/Token, CT,T,Both, time Todo: supervised or unsupervised and move this to a proper database
        times: tuple of time information. (lowest, highest, stepsize) Default is (10,160,10)
        random_state: Integer for random_states
    """

    def __init__(self, prepared_input, times=None, random_state=None):
        if random_state is None:
            self.random_state = random.randint(1, 10**8)
        else:
            self.random_state = random_state
        if times is None:
            self.times = [10, 160, 10]
        else:
            self.times = times
        self.input = prepared_input
        nested_dict = lambda: defaultdict(nested_dict)
        self.datasets = nested_dict()
        self.models = nested_dict()

    def __get_configurations(self):
        sides = ["CT", "T", "BOTH"]
        coordinates = [True, False]
        time = np.arange(self.times[0], self.times[1], self.times[2])
        return itertools.product(sides, time, coordinates)

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
        return return_dataframe

    def __pad_to_length(self, position_df, time):
        """Pad or shorten a pandas dataframe to the given length.

        If the given time is larger than the current size the dataframe is expanded to that size.
        Values of last valid column are used to pad

        If the stime is smaller then the dataframe is cut off at that length.

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
        position_df.applymap(lambda x: float(x) / 5)
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
                coordinates: A boolean indicating whether player coordinates should be used directly (true) or the summarizing tokens instead.

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
        if coordinates:
            featurelist = ["x", "y", "z"]
            # featurelist=["Alive","x","y","z"]
            dimensions = [
                time,
                5,
                len(featurelist),
            ]  # time for timesteps, 5 for players, len(feautrelist) for features
            position_df = self.__pad_to_length(position_df, time)
            if team == "BOTH":
                sides = ["CT", "T"]
                # Dimension with size 2 for teams
                dimensions.append(2)
            else:
                sides = [team]
                # Dimension with size 1 for team
                dimensions.append(1)

            # Only keep the information for the side(s) that should be used.
            return_array = np.zeros(tuple(dimensions))
            for side in sides:
                for number in range(1, 6):
                    for feature in featurelist:
                        return_array[
                            :, number - 1, featurelist.index(feature), sides.index(side)
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
        model = tf.keras.Sequential(
            [
                layers.LSTM(nodes_per_layer, input_shape=input_shape),
                layers.Dense(
                    nodes_per_layer, activation="relu"
                ),  # ,kernel_regularizer=tf.keras.regularizers.l2(0.001)
                layers.Dropout(0.4),
                layers.Dense(
                    nodes_per_layer, activation="relu"
                ),  # ,kernel_regularizer=tf.keras.regularizers.l2(0.001)
                layers.Dropout(0.4),
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
                layers.Dropout(0.4),
                layers.Dense(
                    nodes_per_layer, activation="relu"
                ),  # ,kernel_regularizer=tf.keras.regularizers.l2(0.001)
                layers.Dropout(0.4),
                layers.Dense(1),
            ]
        )
        return model

    def generate_data_sets(self):
        """Perform generation of different datasets. Currently also build the models and train them. but that will be moved to a new function soon."""
        if os.path.isfile(self.input):
            with open(self.input, encoding="utf-8") as pre_analyzed:
                complete_dataframe = pd.read_json(pre_analyzed)
        else:
            logging.error("File %s does not exist!", self.input)
            raise FileNotFoundError("File does not exist!")
        # Only keep the label (Winner) and training feature (position_df) and discard the MatchID, Map_name and round number.
        dataframe = complete_dataframe[["Winner", "position_df"]]
        # Transform the position_df from a json dict to df and also transform ticks to seconds
        dataframe["position_df"] = dataframe["position_df"].apply(
            self.__transform_to_data_frame
        )
        max_time = self.__get_maximum_length(dataframe)
        if max_time > 160:
            max_time = 160
        self.times[1] = max_time
        for configuration in self.__get_configurations():
            side, time, coordinate = configuration
            this_dataframe = dataframe.copy()
            # Shuffle and split dataframe into training, test and validation set
            # set aside 20% of train and test data for evaluation
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

            train_df.reset_index(drop=True, inplace=True)
            val_df.reset_index(drop=True, inplace=True)
            test_df.reset_index(drop=True, inplace=True)

            test_labels, test_features = self.__transform_dataframe_to_arrays(test_df)
            val_labels, val_features = self.__transform_dataframe_to_arrays(val_df)
            train_labels, train_features = self.__transform_dataframe_to_arrays(
                train_df
            )

            train_dataset = tf.data.Dataset.from_tensor_slices(
                (train_features, train_labels)
            )
            test_dataset = tf.data.Dataset.from_tensor_slices(
                (test_features, test_labels)
            )
            val_dataset = tf.data.Dataset.from_tensor_slices((val_features, val_labels))
            BATCH_SIZE = 32
            SHUFFLE_BUFFER_SIZE = 40

            train_dataset = train_dataset.batch(BATCH_SIZE)
            test_dataset = test_dataset.batch(BATCH_SIZE)
            val_dataset = val_dataset.batch(BATCH_SIZE)

            nodes_per_layer = 32

            self.datasets[side][time][coordinate]["val"] = val_dataset
            self.datasets[side][time][coordinate]["train"] = train_dataset
            self.datasets[side][time][coordinate]["test"] = test_dataset
            if coordinate:
                model = self.get_coordinate_model(
                    nodes_per_layer, train_features[0].shape
                )
            else:
                model = self.get_token_model(nodes_per_layer, train_features[0].shape)

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
                train_dataset,
                epochs=50,
                batch_size=BATCH_SIZE,
                validation_data=val_dataset,
                callbacks=tf.keras.callbacks.EarlyStopping(
                    monitor="val_binary_crossentropy", patience=5
                ),
            )

            loss, accuracy, entropy = model.evaluate(test_dataset)

            print("Loss: ", loss)
            print("Accuracy: ", accuracy)

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

            plt.show()

            plt.plot(epochs, acc, "bo", label="Training acc")
            plt.plot(epochs, val_acc, "b", label="Validation acc")
            plt.title("Training and validation accuracy")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.legend(loc="lower right")

            plt.show()


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
        default=r"D:\CSGO\ML\CSGOML\ReadTensorflowInput.log",
        help="Path to output log.",
    )
    parser.add_argument(
        "-s",
        "--side",
        default="BOTH",
        help="Which side to include in analysis (CT,T,BOTH) .",
    )
    parser.add_argument(
        "--coordinates",
        action="store_true",
        default=False,
        help="Whether to use full coordinate or just tokens.",
    )
    parser.add_argument(
        "-t",
        "--time",
        default="MAX",
        help="How many position snapshots should be used. Either an integer or MAX.",
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
            format="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    example_index = options.exampleid
    random_state = options.randomstate

    if options.side not in ["CT", "T", "BOTH"]:
        logging.error(
            "Side %s unknown. Has to be one of 'CT','T','BOTH'!", options.side
        )
        sys.exit()

    # Time should either an integer or MAX.
    # If it is an integer n that means that the first n timesteps will be considered for each round and the rest discarded.
    # If it is 'MAX' then we look what the longest round in the dataset is and pad all other rounds to that length.
    # Should probably do the Training,Test split before and only look at the max length of the training set. TODO

    # Read in the prepared json file.
    # File="D:\CSGO\Demos\Maps\\"+options.map+"\Analysis\Prepared_Input_Tensorflow_"+options.map+".json"
    File = (
        r"E:\PhD\MachineLearning\CSGOData\ParsedDemos\\"
        + options.map
        + r"\Analysis\Prepared_Input_Tensorflow_"
        + options.map
        + ".json"
    )

    predictor = TrajectoryPredictor(File, random_state=random_state)
    predictor.generate_data_sets()


if __name__ == "__main__":
    main(sys.argv[1:])
