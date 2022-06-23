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
        datasets: Dictionary of datasets derived from input. split by Pos/Token, CT,T,Both, time, train/test/val. Todo: supervised or unsupervised and move this to a proper database
        models: Dictionary of models. split by Pos/Token, CT,T,Both, time Todo: supervised or unsupervised and move this to a proper database
        times: List of time information. (lowest, highest, stepsize) Default is (10,160,10)
        sides: List of side configurations to consider. Possible are BOTH, CT and T
        random_state: Integer for random_states
        complete_dataframe: Pandas dataframe generated from input.
        example_id: Integer used for debugging output.
    """

    def __init__(
        self, prepared_input, times=None, sides=None, random_state=None, example_id=None
    ):
        if random_state is None:
            self.random_state = random.randint(1, 10**8)
        else:
            self.random_state = random_state
        if times is None:
            self.times = [10, 160, 10]
        else:
            self.times = times

        self.sides = ["CT", "T", "BOTH"]
        for side in list(sides):
            if side not in self.sides:
                sides.remove(side)
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

        self.datasets = tree()
        self.models = tree()
        self.example_id = example_id

    def __get_configurations(self):
        sides = ["BOTH"]  #
        coordinates = ["tokens", "positions"]
        time = np.arange(
            self.times[0], self.times[1] + self.times[2] // 2, self.times[2]
        )
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
        if coordinates == "positions":
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
        self.datasets[side][time][coordinate]["val"]["features"] = val_features
        self.datasets[side][time][coordinate]["train"]["features"] = train_features
        self.datasets[side][time][coordinate]["test"]["features"] = test_features
        self.datasets[side][time][coordinate]["val"]["labels"] = val_labels
        self.datasets[side][time][coordinate]["train"]["labels"] = train_labels
        self.datasets[side][time][coordinate]["test"]["labels"] = test_labels

        logging.info("Input preparation done for configuration %s", configuration)
        logging.info(train_labels.shape)
        logging.info(train_features.shape)

        train_dataset = tf.data.Dataset.from_tensor_slices(
            (train_features, train_labels)
        )
        test_dataset = tf.data.Dataset.from_tensor_slices((test_features, test_labels))
        val_dataset = tf.data.Dataset.from_tensor_slices((val_features, val_labels))

        train_dataset = train_dataset.batch(batch_size)
        test_dataset = test_dataset.batch(batch_size)
        val_dataset = val_dataset.batch(batch_size)
        self.datasets[side][time][coordinate]["val"]["dataset"] = val_dataset
        self.datasets[side][time][coordinate]["train"]["dataset"] = train_dataset
        self.datasets[side][time][coordinate]["test"]["dataset"] = test_dataset

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


def main(args):
    """Read input prepared by tensorflow_input_preparation.py and builds/trains DNNs to predict the round winner based on player trajectory data."""
    parser = argparse.ArgumentParser("Analyze the early mid fight on inferno")
    parser.add_argument(
        "-d", "--debug", action="store_true", default=True, help="Enable debug output."
    )
    parser.add_argument("-m", "--map", default="ancient", help="Map to analyze")
    parser.add_argument(
        "-l",
        "--log",
        default=r"D:\CSGO\ML\CSGOML\ReadTensorflowInput.log",
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
            format="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    example_index = options.exampleid
    random_state = options.randomstate

    # Time should either an integer or MAX.
    # If it is an integer n that means that the first n timesteps will be considered for each round and the rest discarded.
    # If it is 'MAX' then we look what the longest round in the dataset is and pad all other rounds to that length.
    # Should probably do the Training,Test split before and only look at the max length of the training set. TODO

    # Read in the prepared json file.
    # File="D:\CSGO\Demos\Maps\\"+options.map+"\Analysis\Prepared_Input_Tensorflow_"+options.map+".json"
    file = (
        r"E:\PhD\MachineLearning\CSGOData\ParsedDemos\\"
        + options.map
        + r"\Analysis\Prepared_Input_Tensorflow_"
        + options.map
        + ".json"
    )

    predictor = TrajectoryPredictor(
        file,
        times=[160, 160, 10],
        sides=["BOTH"],
        random_state=random_state,
        example_id=example_index,
    )
    predictor.generate_data_sets()
    predictor.plot_all_models()


if __name__ == "__main__":
    main(sys.argv[1:])
