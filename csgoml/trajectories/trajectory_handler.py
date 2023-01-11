"""
This module contains the TrajectoryHandler that reads in the precomputed trajectories and makes them easily availalbe in the form of numpy arrays.

Typical usage example:

    handler = TrajectoryHandler(
        json_path=file, random_state=random_state, map_name="de_" + options.map
    )
    print(handler.datasets["token"])
"""

import os
import random
import logging
from typing import Optional, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# from awpy.data import NAV


class TrajectoryHandler:
    """Read input prepared by tensorflow_input_preparation.py and perform final cleaning, transforming and shuffling.

    Requires as input a dataframe/json that for each round contains the winner and a dataframe of the players trajectories/ position token trajectories.

    Attributes:
        input (string): Path to the json input file containing all the trajectory data for every round on a given map.
        datasets (dict[str, np.ndarray]): Dictionary of position and token trajectory numpy arrays
        aux (dict[str, np.ndarray]): Dictionary of auxilliary information about about trajectories.
        random_state (int): Integer for random_states
        time (int): Maximum time that is reasonable for a round to have
        map_name (string): Name of the map under consideration
    Raises:
        FileNotFoundError: If there is no file at the path of the input string
    """

    def __init__(
        self,
        json_path: str,
        random_state: Optional[int] = None,
        map_name: str = "de_ancient",
        time: int = 175,
    ):
        logging.info("Starting init")
        self.map_name: str = map_name
        if random_state is None:
            self.random_state: int = random.randint(1, 10**8)
        else:
            self.random_state = random_state
        self.input: str = json_path
        if os.path.isfile(self.input):
            with open(self.input, encoding="utf-8") as pre_analyzed:
                complete_dataframe = pd.read_json(pre_analyzed)
        else:
            logging.error("File %s does not exist!", self.input)
            raise FileNotFoundError("File does not exist!")
        self.time: int = time

        logging.debug("Initial dataframe:")
        logging.debug(complete_dataframe)

        self.datasets: dict[str, np.ndarray] = {}
        self.aux: dict[str, np.ndarray] = {}
        for column in complete_dataframe:
            if column != "position_df":
                self.aux[column] = complete_dataframe[column].to_numpy()
        # Transform the position_df from a json dict to df and also transform ticks to seconds
        dataframe = complete_dataframe[["position_df"]]
        dataframe["position_df"] = dataframe["position_df"].apply(
            self.__transform_to_data_frame
        )

        logging.debug("Example for position_df dataframe entry before cleaning.")
        logging.debug(dataframe.iloc[6]["position_df"])

        dataframe["token_array"] = dataframe["position_df"].apply(
            self.__get_token_array
        )
        dataframe["position_array"] = dataframe["position_df"].apply(
            self.__get_position_array
        )
        # Shape of the numpy array is #Round,self.time,len(token(self.map_name)) First half of the token length is CT second is T
        self.datasets["token"] = np.stack(dataframe["token_array"].to_numpy())
        # Shape of the numpy array is #Rounds,self.time,side(2),player(5),feature(5[x,y,z,area,alive])
        self.datasets["position"] = np.stack(dataframe["position_array"].to_numpy())
        logging.info("Finished init")

    def __transform_to_data_frame(self, json_format_dict: dict) -> pd.DataFrame:
        """Transforms a json dictionary to a pandas dataframe and converts ticks to seconds.

        Args:
            json_format_dict: A dictionary in json format holding information about player positions during a CS:GO round

        Returns:
            A pandas dataframe corresponding to the input json with the tick values transformed to seconds.
        """
        return_dataframe = pd.DataFrame(json_format_dict)
        first_tick = int(return_dataframe.iloc[0]["Tick"])
        return_dataframe["Tick"] = return_dataframe["Tick"].apply(
            self.__transform_ticks_to_seconds, args=(first_tick,)
        )
        # If reintroduced it should only be applied to the position feature array used for DNN predictions
        # return_dataframe = return_dataframe.apply(self.regularize_coordinatesdf)
        return return_dataframe

    def __transform_ticks_to_seconds(self, tick: int, first_tick: int) -> int:
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

    def __get_token_array(self, position_df: pd.DataFrame) -> np.ndarray:
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

        Returns:
            A two dimensional numpy array containing the transformed token at each timestep
        """
        position_df.reset_index(drop=True, inplace=True)
        # Split the token strings into a columns so that there is one column for each integer in the token string
        # Only work with the token entry of the df
        position_df = position_df[["token"]]
        # Initialize the numpy array
        # First value is number of rows, second number of columns
        return_array = np.zeros(
            tuple([self.time, len(position_df.iloc[0]["token"])]), dtype=np.int64
        )
        # Transform the string into a list of the individual letters(ints) in the string
        position_df = position_df["token"].apply(list)
        # Convert the dataframe to a list and back into a df to have one column for each entry of the string
        position_df = pd.DataFrame(position_df.tolist())
        # Convert the individual string into the respective integers
        # Pad the df to the specified length
        position_df = position_df.reindex(range(self.time), fill_value=0, method="pad")
        # Convert to numpy array
        # return_array=position_df.to_numpy()
        for ind, column in enumerate(position_df.columns):
            return_array[:, ind] = position_df[column].to_numpy()
        return return_array

    def __get_position_array(self, position_df: pd.DataFrame) -> np.ndarray:
        """Transforms data frame by throwing away all unnecessary features
        and then turning it into a (multidimensional) array.

        The result is a 4d array where for each timestep there is a 3d array representing team, player number and the feature.

        Args:
            position_df: A dataframe of player positions during a round.

        Returns:
            A multi-dimensional numpy array containing split tokens or player positions (organized by team, playnumber, coordinate)
        """
        # Throw away all unneeded columns from the dataframe and then convert it into a numpy array
        # If coordinates is false than it is a 1-D array of the tokens for each timestep
        # If coordinates is true than it is a 4-D array with the first index representing the timestep, the second the team, the third the playernumber in that team and the fourth the feature.
        position_df.reset_index(drop=True, inplace=True)
        # Set length of dataframe to make sure all have the same size
        # Pad each column if set size is larger than dataframe size
        # If the full coordinates should be used then for each player their alive status as well as their x,y,z coordinates are kept, everything else (player names and tokens) is discarded.
        featurelist = ["x", "y", "z", "Area", "Alive"]
        position_df = position_df.reindex(
            range(self.time), fill_value=0.0, method="pad"
        )
        sides = ["CT", "T"]
        # Dimension with size 2 for teams
        dimensions = [
            self.time,
            2,
            5,
            len(featurelist),
        ]  # time for timesteps, 5 for players, len(featurelist) for features

        # Only keep the information for the side(s) that should be used.
        return_array = np.zeros(tuple(dimensions))
        for side_index, side in enumerate(sides):
            for number in range(1, 6):
                for feature_index, feature in enumerate(featurelist):
                    return_array[
                        :, side_index, number - 1, feature_index
                    ] = position_df[side + "Player" + str(number) + feature].to_numpy()
        return return_array

    def get_predictor_input(
        self,
        coordinate_type: str,
        side: str,
        time: int,
        consider_alive: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get the input for the DNN training to predict end of round result.

        First gets the array for the correct coordinate type and then slices it by the desired side and time.
        Finally the dataset is split into train, val and test sets and returned together with the labels.
        Shape of the position numpy array is #Round,self.time,side(2),player(5),feature(5[x,y,z,area,alive])
        Shape of the token numpy array is #Round,self.time,len(token(self.map_name)) First half of the token length is CT second is T
        Shape of the aux arrays is #Round

        Args:
            coordinate_type (string): A string indicating whether player coordinates should be used directly ("position") or the summarizing tokens ("token") instead.
            side (string): A string indicating whether to include positions for players on the CT side ('CT'), T  side ('T') or both sides ('BOTH')
            time (integer): An integer indicating the first how many seconds should be considered
            consider_alive (boolean): A boolean indicating whether the alive status of each player should be considered. Only relevant together with coordinate_type of "position"

        Returns:
            Numpy arrays for train, val and test labels and features. Shapes depend on desired configuration. Order is train_label, val_label, test_label, train_features, val_features, test_features"""
        label = self.aux["Winner"]
        if coordinate_type == "position":
            side_conversion: Union[dict[str, list[int]], dict[str, tuple[int, int]]] = {
                "CT": [0],
                "T": [1],
                "BOTH": [0, 1],
            }
            features_of_interest = [0, 1, 2]
            if consider_alive:
                features_of_interest.append(4)
            indices: tuple[np.typing.NDArray[np.int8], ...] = np.ix_(
                range(self.datasets["position"].shape[0]),
                range(time),
                side_conversion[side],
                range(self.datasets["position"].shape[3]),
                features_of_interest,
            )
            features = self.datasets["position"][indices]
        else:  # coordinate_type == "token"
            start, end = 0, self.datasets["token"].shape[-1]
            mid = end // 2
            side_conversion = {
                "CT": (start, mid),
                "T": (mid, end),
                "BOTH": (start, end),
            }
            first, last = side_conversion[side]
            features = self.datasets["token"][:, :time, first:last]

        train_labels, test_labels, train_features, test_features = train_test_split(
            label,
            features,
            test_size=0.20,
            shuffle=True,
            random_state=self.random_state,
        )
        # Use the same function above for the validation set  0.25 x 0.8 = 0.2
        train_labels, val_labels, train_features, val_features = train_test_split(
            train_labels,
            train_features,
            test_size=0.25,
            shuffle=True,
            random_state=self.random_state,
        )
        return (
            train_labels,
            val_labels,
            test_labels,
            train_features,
            val_features,
            test_features,
        )

    def get_clustering_input(
        self, n_rounds: int, coordinate_type_for_distance: str, side: str, time: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the input clustering of round.

        First gets the array for the correct coordinate type and then slices it by the desired side and time.
        A shuffle of the relevant array for clustering is returned together with the matching positions (shuffled coherently) needed for plotting the clusters.
        Shape of the position numpy array is #Round,self.time,side(2),player(5),feature(5[x,y,z,area,alive]).
        Shape of the token numpy array is #Round,self.time,len(token(self.map_name)) First half of the token length is CT second is T.

        Args:
            n_rounds (int): How many rounds should be in the final output. Can be necessary to not use all of them due to time constraints.
            coordinate_type_for_distance (string): A string indicating whether player coordinates should be used directly ("position"), the areas ("area") or the summarizing tokens ("token") instead.
            side (string): A string indicating whether to include positions for players on the CT side ('CT'), T  side ('T') or both sides ('BOTH')
            time (integer): An integer indicating the first how many seconds should be considered

        Returns:
            Numpy arrays to use for plotting and clustering. Shape of the plotting array is (#Rounds,time,side(1/2),player(5),features(4/3 for x,y,z,{area}))
            Shape of the clustering array depend son desired configuration. Order is array_for_plotting, array_for_clustering"""
        side_conversion = {"CT": (0,), "T": (1,), "BOTH": (0, 1)}
        array_for_plotting = self.datasets["position"][
            :n_rounds, :time, side_conversion[side], :, :4
        ]
        if coordinate_type_for_distance == "position":
            array_for_plotting = array_for_plotting[:, :, :, :, :3]
            array_for_clustering = array_for_plotting
        elif coordinate_type_for_distance == "area":
            indices: tuple[np.typing.NDArray[np.int8], ...] = np.ix_(
                range(min(self.datasets["position"].shape[0], n_rounds)),
                range(time),
                side_conversion[side],
                range(self.datasets["position"].shape[3]),
                (3,),
            )
            array_for_clustering = self.datasets["position"][indices]
        else:  # coordinate_type == "token"
            start, end = 0, self.datasets["token"].shape[-1]
            mid = end // 2
            side_conversion = {
                "CT": (start, mid),
                "T": (mid, end),
                "BOTH": (start, end),
            }
            first, last = side_conversion[side]
            array_for_clustering = self.datasets["token"][:n_rounds, :time, first:last]
        return shuffle(
            array_for_plotting, array_for_clustering, random_state=self.random_state
        )

    # def regularize_coordinates(self, coordinate, minimum, maximum):
    #     """Regularizes coordinates to be between -1 and 1

    #     If the map is in the awpy NAV data then minimum and maximum correspondt to actual min and max values.
    #     If the map is not then they are flatly taken to be +-2000/2000/200 for x, y and z respectively.
    #     In that case the coordinate can end up outside of -1 to 1 in some cases.

    #     Args:
    #         coordinate: Float representing a player coordinate
    #         minimum: The minimal possible value of this coordinate
    #         maximum: The maximal possible value of thos coordinate

    #     Returns:
    #         A float corresponding to a rescaled coordinate that is always between -1 and 1
    #     """
    #     shift = (maximum + minimum) / 2
    #     scaling = (maximum - minimum) / 2
    #     return (coordinate - shift) / scaling

    # def get_extremes_from_NAV(self):
    #     """Determines the maximal and mininmal possible x, y and z values for a given map

    #     Look through the awpy NAV data of the given map and search all recorded areas.
    #     Keep track of minimum and maximum value for each coordinate.
    #     If the map does not exist in the NAV data then use default values of +-2000/2000/200 for x, y and z respectively

    #     Args:
    #         map_name: String of the currently investigated map

    #     Returns:
    #         Two dictionary containing the minimum and maximum determined values for each coordinate.
    #     """
    #     if self.map_name not in NAV:
    #         minimum = {"x": -2000, "y": -2000, "z": -200}
    #         maximum = {"x": 2000, "y": 2000, "z": 200}
    #     else:
    #         minimum = {"x": sys.maxsize, "y": sys.maxsize, "z": sys.maxsize}
    #         maximum = {"x": -sys.maxsize, "y": -sys.maxsize, "z": -sys.maxsize}
    #         for area in NAV[self.map_name]:
    #             for feature in ["x", "y", "z"]:
    #                 for corner in ["northWest", "southEast"]:
    #                     maximum[feature] = max(
    #                         NAV[self.map_name][area][corner + feature.upper()],
    #                         maximum[feature],
    #                     )
    #                     minimum[feature] = min(
    #                         NAV[self.map_name][area][corner + feature.upper()],
    #                         minimum[feature],
    #                     )
    #     return minimum, maximum

    # def regularize_coordinatesdf(self, position_df):
    #     """Apply coordinate regularization to all coordinate positions in the dataframe.

    #     Gather minimum and maximum values and regularize each coordinate to be between -1 and 1

    #     Args:
    #         position_df: Dataframe containing every players position and status for each time step
    #         map_name: String of the current maps name

    #     Returns:
    #         dataframe after player coordinate regularization
    #     """
    #     minimum, maximum = self.get_extremes_from_NAV()
    #     for side in ["CT", "T"]:
    #         for number in range(1, 6):
    #             for feature in ["x", "y", "z"]:
    #                 position_df[side + "Player" + str(number) + feature] = position_df[
    #                     side + "Player" + str(number) + feature
    #                 ].apply(
    #                     self.regularize_coordinates,
    #                     args=(minimum[feature], maximum[feature]),
    #                 )
    #     return position_df
