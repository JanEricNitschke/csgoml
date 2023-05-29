"""This module contains the TrajectoryHandler.

That reads in the precomputed trajectories and
makes them easily availalbe in the form of numpy arrays.

Example::

    handler = TrajectoryHandler(
        json_path=file, random_state=random_state, map_name="de_" + options.map
    )
    print(handler.datasets["token"])
"""

import logging
import os
import random

import numpy as np
import numpy.typing as npt
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


class TrajectoryHandler:
    """Read preparedby input perform final cleaning, transforming and shuffling.

    Requires as input a dataframe/json that for each round contains
    the winner and a dataframe of the players trajectories/ position token trajectories.

    Attributes:
        input (string): Path to the json input file containing all
            the trajectory data for every round on a given map.
        datasets (dict[str, npt.NDArray]): Dictionary of position and
            token trajectory numpy arrays
        aux (dict[str, npt.NDArray]): Dictionary of auxilliary information
            about about trajectories.
        random_state (int): Integer for random_states
        time (int): Maximum time that is reasonable for a round to have
        map_name (string): Name of the map under consideration
    Raises:
        FileNotFoundError: If there is no file at the path of the input string
    """

    def __init__(
        self,
        json_path: str,
        random_state: int | None = None,
        map_name: str = "de_ancient",
        time: int = 175,
    ) -> None:
        """Initialize an instance.

        Args:
            json_path (str): Path to the prepared json input.
            random_state (int | None, optional):Integer for random_states.
                Defaults to None.
            map_name (str, optional): _description_. Defaults to "de_ancient".
            time (int, optional): _description_. Defaults to 175.

        Raises:
            FileNotFoundError: If the file path is invalid.
        """
        logging.info("Starting init")
        self.map_name: str = map_name
        if random_state is None:
            # We are not doing cryptography
            self.random_state: int = random.randint(1, 10**8)  # noqa: S311
        else:
            self.random_state = random_state
        self.input: str = json_path
        if os.path.isfile(self.input):
            complete_dataframe = pl.scan_ndjson(self.input).collect()
        else:
            logging.error("File %s does not exist!", self.input)
            msg = "File does not exist!"
            raise FileNotFoundError(msg)
        self.time: int = time

        logging.debug("Initial dataframe:")
        logging.debug(complete_dataframe)

        self.datasets: dict[str, npt.NDArray] = {}
        self.aux: dict[str, npt.NDArray] = {}

        self._fill_aux(complete_dataframe)
        complete_dataframe = self._apply_transformations(complete_dataframe)
        logging.info(complete_dataframe)
        self.datasets["token"] = self._get_token_dataset(complete_dataframe)
        self.datasets["position"] = self._get_position_dataset(complete_dataframe)
        logging.info(self.aux)
        logging.info(self.datasets)

    def _fill_aux(self, trajectories: pl.DataFrame) -> None:
        """Fill the auxiliary information dictionary."""
        for column in trajectories.select(
            pl.exclude([pl.List(pl.Int64), pl.List(pl.Utf8), pl.List(pl.Float64)])
        ):
            self.aux[column.name] = column.to_numpy(use_pyarrow=False)

    def _apply_transformations(self, trajectories: pl.DataFrame) -> pl.DataFrame:
        """Apply transformations to trajectories dataframe.

        Go from ticks to seconds by substracting the first tick value and
        dividing by 128.

        Transform tokens from string to list of integers.

        Pad all time series data to the specified length.
        """
        trajectories = trajectories.with_columns(
            [
                (
                    pl.col("Tick")
                    .cast(pl.List(pl.Float64))
                    .arr.eval((pl.element() - pl.element().first()) / 128)
                ),
                pl.col(["token", "CTtoken", "Ttoken"]).arr.eval(
                    pl.element().str.extract_all(r"\d").cast(pl.List(pl.Int64))
                ),
            ]
        )
        cols = pl.col(
            pl.List(pl.Int64), pl.List(pl.Float64), pl.List(pl.List(pl.Int64))
        )
        trajectories = trajectories.with_columns(
            cols.arr.take(pl.arange(0, self.time), null_on_oob=True).arr.eval(
                pl.element().forward_fill()
            )
        )
        return trajectories

    def _get_token_dataset(self, trajectories: pl.DataFrame) -> npt.NDArray:
        """Get the token dataset by transforming the dataframe column to a np array."""
        return np.stack(trajectories.get_column("token").to_list())

    def _get_position_dataset(self, trajectories: pl.DataFrame) -> npt.NDArray:
        """Get position dataset from trajectory dataframe.

        Transform each feature column into a 2D numpy array and fill
        the respective entries of the total position array with them.
        """
        featurelist = ["x", "y", "z", "Area", "Alive"]
        sides = ["CT", "T"]
        dimensions = [
            len(trajectories),
            self.time,
            2,
            5,
            len(featurelist),
        ]
        return_array = np.zeros(tuple(dimensions))
        for side_index, side in enumerate(sides):
            for number in range(1, 6):
                for feature_index, feature in enumerate(featurelist):
                    return_array[
                        :, :, side_index, number - 1, feature_index
                    ] = np.stack(
                        trajectories.get_column(
                            side + "Player" + str(number) + feature
                        ).to_list()
                    )
        return return_array

    def _get_position_predictor_input(
        self,
        side: str,
        time: int,
        *,
        consider_alive: bool = False,
    ) -> npt.NDArray:
        """Get the input for the DNN training with positions.

        First gets the array for positions.
        Slices it by the desired side and time.
        Finally the dataset is split into train, val and test sets and
        returned together with the labels.
        Shape of the position numpy array is
        #Round,self.time,side(2),player(5),feature(5[x,y,z,area,alive])

        Args:
            side (string): Whether to include positions for players on
                the CT side ('CT'), T  side ('T') or both sides ('BOTH')
            time (integer): An integer indicating the first
                how many seconds should be considered
            consider_alive (boolean): Indicating whether the alive status of each player
                should be considered.

        Returns:
            Numpy arrays for features.
        """
        side_conversion: dict[str, list[int]] | dict[str, tuple[int, int]] = {
            "CT": [0],
            "T": [1],
            "BOTH": [0, 1],
        }
        features_of_interest = [0, 1, 2]
        if consider_alive:
            features_of_interest.append(4)
        indices: tuple[npt.NDArray[np.int8], ...] = np.ix_(
            range(self.datasets["position"].shape[0]),
            range(time),
            side_conversion[side],
            range(self.datasets["position"].shape[3]),
            features_of_interest,
        )
        return self.datasets["position"][indices]

    def _get_token_predictor_input(
        self,
        side: str,
        time: int,
    ) -> npt.NDArray:
        """Get the input for the DNN training with tokens.

        First gets the array for tokens
        then slices it by the desired side and time.
        Finally the dataset is split into train, val and test sets and
        returned together with the labels.
        Shape of the token numpy array is
        #Round,self.time,len(token(self.map_name))
        First half of the token length is CT second is T

        Args:
            side (string): Whether to include positions for players on
                the CT side ('CT'), T  side ('T') or both sides ('BOTH')
            time (integer): An integer indicating the first
                how many seconds should be considered


        Returns:
            Numpy arrays for features.
        """
        start, end = 0, self.datasets["token"].shape[-1]
        mid = end // 2
        side_conversion = {
            "CT": (start, mid),
            "T": (mid, end),
            "BOTH": (start, end),
        }
        first, last = side_conversion[side]
        return self.datasets["token"][:, :time, first:last]

    def get_predictor_input(
        self,
        coordinate_type: str,
        side: str,
        time: int,
        *,
        consider_alive: bool = False,
    ) -> tuple[
        npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray
    ]:
        """Get the input for the DNN training to predict end of round result.

        First gets the array for the correct coordinate type and
        then slices it by the desired side and time.
        Finally the dataset is split into train, val and test sets and
        returned together with the labels.
        Shape of the position numpy array is
        #Round,self.time,side(2),player(5),feature(5[x,y,z,area,alive])
        Shape of the token numpy array is
        #Round,self.time,len(token(self.map_name))
        First half of the token length is CT second is T
        Shape of the aux arrays is #Round

        Args:
            coordinate_type (string): Whether player coordinates should be used
                directly ("position") or the summarizing tokens ("token") instead.
            side (string): Whether to include positions for players on
                the CT side ('CT'), T  side ('T') or both sides ('BOTH')
            time (integer): An integer indicating the first
                how many seconds should be considered
            consider_alive (boolean): Indicating whether the alive status of each player
                should be considered.
                Only relevant together with coordinate_type of "position"

        Returns:
            Numpy arrays for train, val and test labels and features.
            Shapes depend on desired configuration.Order is:
            train_label, val_label, test_label,
            train_features, val_features, test_features
        """
        label = self.aux["Winner"]
        if coordinate_type == "position":
            features = self._get_position_predictor_input(
                side, time, consider_alive=consider_alive
            )
        else:  # coordinate_type == "token"
            features = self._get_token_predictor_input(side, time)

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
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Get the input clustering of round.

        First gets the array for the correct coordinate type and
        then slices it by the desired side and time.
        A shuffle of the relevant array for clustering is returned together with the
        matching positions (shuffled coherently) needed for plotting the clusters.
        Shape of the position numpy array is
        #Round,self.time,side(2),player(5),feature(5[x,y,z,area,alive]).
        Shape of the token numpy array is #Round,self.time,len(token(self.map_name))
        First half of the token length is CT second is T.

        Args:
            n_rounds (int): How many rounds should be in the final output.
                Can be necessary to not use all of them due to time constraints.
            coordinate_type_for_distance (string): A string indicating whether
                player coordinates should be used directly ("position"),
                the areas ("area") or the summarizing tokens ("token") instead.
            side (string): A string indicating whether to include positions for
                players on the CT side ('CT'), T  side ('T') or both sides ('BOTH')
            time (integer): An integer indicating the first how
                many seconds should be considered

        Returns:
            Numpy arrays to use for plotting and clustering.
            Shape of the plotting array is
            (#Rounds,time,side(1/2),player(5),features(4/3 for x,y,z,{area}))
            Shape of the clustering array depend son desired configuration.
            Order is array_for_plotting, array_for_clustering
        """
        side_conversion = {"CT": (0,), "T": (1,), "BOTH": (0, 1)}
        array_for_plotting = self.datasets["position"][
            -n_rounds:, :time, side_conversion[side], :, :4
        ]
        if coordinate_type_for_distance == "position":
            array_for_plotting = array_for_plotting[:, :, :, :, :3]
            array_for_clustering = array_for_plotting
        elif coordinate_type_for_distance == "area":
            indices: tuple[npt.NDArray[np.int8], ...] = np.ix_(
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
            array_for_clustering = self.datasets["token"][-n_rounds:, :time, first:last]
        return shuffle(
            array_for_plotting, array_for_clustering, random_state=self.random_state
        )
