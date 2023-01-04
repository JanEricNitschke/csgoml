"""
This module contains the TrajectoryPredictor.
It gets its inputs properly formatted from a TrajectoryHandler and then builds/trains DNNs to predict the round winner based on player trajectory data.
"""

import os
from typing import Optional, Literal
import random
import logging
import matplotlib.pyplot as plt
from numpy import ndarray
import tensorflow as tf
from tensorflow import keras
from csgoml.trajectories.trajectory_handler import TrajectoryHandler


class TrajectoryPredictor:
    """Builds and trains DNNs to predict the winner of a round based on trajectories of different configurations by grabbing them from its TrajetoryHandler.

    Attributes:
        analysis_input (string): Path to where the results (distance matrix and plots) should be stored
        trajectory_handler (trajectory_handler.TrajectoryHandler): trajectory_handler.TrajectoryHandler from which to grab requested datasets
        random_state (int): Integer for random_states
        map_name (string): Name of the map under consideration
    """

    def __init__(
        self,
        analysis_path: str,
        trajectory_handler: TrajectoryHandler,
        random_state: Optional[int] = None,
        map_name: str = "de_ancient",
    ):
        self.analysis_path: str = os.path.join(analysis_path, "predicting")
        if not os.path.exists(self.analysis_path):
            os.makedirs(self.analysis_path)
        self.map_name: str = map_name
        if random_state is None:
            self.random_state: int = random.randint(1, 10**8)
        else:
            self.random_state = random_state
        self.trajectory_handler: TrajectoryHandler = trajectory_handler

    def do_predicting(
        self, trajectory_config: tuple[str, int, str, bool], dnn_config: dict
    ) -> Literal[True]:
        """Does everything needed to cluster a configuration and plot the results

        Args:
            trajectory_config (tuple): Tuple of (coordinate_type, time, side, consider_alive) where:
                coordinate_type (string): A string indicating whether player coordinates should be used directly ("position") or the summarizing tokens ("token") instead.
                side (string): A string indicating whether to include positions for players on the CT side ('CT'), T  side ('T') or both sides ('BOTH')
                time (integer): An integer indicating the first how many seconds should be considered
                consider_alive (boolean): A boolean indicating whether the alive status of each player should be considered. Only relevant together with coordinate_type of "position"
            dnn_config (dict): Dictionary containing settings for clustering. Contents:
                'batch_size' (int): A tf.int64 scalar tf.Tensor, representing the number of consecutive elements of this dataset to combine in a single batch.
                'learning_rate' (float): A floating point value. The learning rate.
                'epochs' (int): Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided.
                                The model is not trained for a number of iterations given by epochs, but merely until the epoch of index epochs is reached.
                'patience' (int): Number of epochs with no improvement after which training will be stopped.
                'nodes_per_layer' (int): Number of nodes per DNN layer
        Returns:
            w.i.p."""
        coordinate_type, time, side, consider_alive = trajectory_config
        config_snippet = f"{self.map_name}_{side}_{time}_{consider_alive}_{coordinate_type}_{self.random_state}"
        config_path = os.path.join(self.analysis_path, config_snippet)
        if not os.path.exists(config_path):
            os.makedirs(config_path)

        self.set_dnn_config_defaults(dnn_config)

        # Grab dataset corresponding to config
        # train_label, val_label, test_label, train_features, val_features, test_features
        tensors = self.trajectory_handler.get_predictor_input(
            coordinate_type, side, time, consider_alive
        )

        # Set input_shape properly
        dnn_config["input_shape"] = tensors[3][0].shape  # train_features[0].shape
        # Generate batched datasets
        batch_size = dnn_config["batch_size"]
        train_dataset, val_dataset, test_dataset = self.get_datasets_from_tensors(
            tensors, batch_size
        )
        #  model, history, test_loss, test_accuracy, test_entropy,
        (model, history, _, _, _,) = self.get_compile_train_evaluate_model(
            coordinate_type, (train_dataset, val_dataset, test_dataset), dnn_config
        )

        dnn_snippet = "_".join([str(val) for val in dnn_config.values()])
        model_path = os.path.join(config_path, f"tf_model_{dnn_snippet}")
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model.save(os.path.join(model_path, "SavedModel"))
        model.save(os.path.join(model_path, "HDF5Model.h5"))

        plot_path = os.path.join(config_path, "plots")
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        self.plot_model(history, plot_path, dnn_snippet, config_snippet)

        return True

    def set_dnn_config_defaults(self, dnn_config: dict) -> None:
        """Sets the defaults in dnn_config in case they have not been set
        Args:
            dnn_config (dict): Dictionary containing settings for clustering.
        Returns:
            None, modified dnn_config in place"""
        if "batch_size" not in dnn_config:
            dnn_config["batch_size"] = 32
        if "learning_rate" not in dnn_config:
            dnn_config["learning_rate"] = 0.00007
        if "epochs" not in dnn_config:
            dnn_config["epochs"] = 50
        if "patience" not in dnn_config:
            dnn_config["patience"] = 5
        if "nodes_per_layer" not in dnn_config:
            dnn_config["nodes_per_layer"] = 32

    def get_compile_train_evaluate_model(
        self,
        coordinate_type: str,
        datasets: tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset],
        dnn_config: dict,
    ) -> tuple[keras.Sequential, tf.keras.callbacks.History, float, float, float]:
        """Wrapper function to get, compile, train and evaluate a model
        Args:
            coordinate_type (string): A string indicating whether player coordinates should be used directly ("position") or the summarizing tokens ("token") instead.
            datasets (tuple[tf.data.Dataset]: train, val and test datasets)
            dnn_config (dict): Dictionary containing settings for clustering.

        Returns:
            tuple of history, loss, accuracy, entropy from model.fit and model.evaluate"""
        train_dataset, val_dataset, test_dataset = datasets
        # Get model
        logging.info(
            "Grabbing model for coordinate_type: %s with dnn_config: %s",
            coordinate_type,
            dnn_config,
        )
        model = self.get_model(coordinate_type, dnn_config)

        # Compile mode
        logging.info("Compiling model")
        learning_rate = dnn_config["learning_rate"]  # 0.00007
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[
                "accuracy",
                tf.keras.losses.BinaryCrossentropy(
                    from_logits=True, name="binary_crossentropy"
                ),
            ],
        )
        model.summary()

        # Train model
        # epochs, patience = 50, 5
        logging.info("Training model")
        epochs, patience = dnn_config["epochs"], dnn_config["patience"]
        history = model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=tf.keras.callbacks.EarlyStopping(
                monitor="val_binary_crossentropy", patience=patience
            ),
        )

        # Evaluate model
        logging.info("Evaluating model")
        loss, accuracy, entropy = model.evaluate(test_dataset)
        logging.info("Loss: %s", loss)
        logging.info("Accuracy: %s", accuracy)
        logging.info("Entropy: %s", entropy)

        return model, history, loss, accuracy, entropy

    def get_datasets_from_tensors(
        self,
        tensors: tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray],
        batch_size: int,
    ) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Generates batched datasets from tensors

        Args:
            tensors (tuple[ndarray]): tuple of numpy arrays of train_labels, val_labels, test_labels, train_features, val_features, test_features
            batch_size (int):A tf.int64 scalar tf.Tensor, representing the number of consecutive elements of this dataset to combine in a single batch.
        Returns:
            A tuple of tf.datasets (train_dataset, val_dataset, test_dataset)"""
        (
            train_labels,
            val_labels,
            test_labels,
            train_features,
            val_features,
            test_features,
        ) = tensors
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (train_features, train_labels)
        ).batch(batch_size)
        val_dataset = tf.data.Dataset.from_tensor_slices(
            (val_features, val_labels)
        ).batch(batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices(
            (test_features, test_labels)
        ).batch(batch_size)
        return train_dataset, val_dataset, test_dataset

    def get_model(self, coordinate: str, model_config: dict) -> keras.Sequential:
        """Grabs or generates a model usable for the specified configuration.

        Args:
            coordinates (string): A string determining if individual players positions ("position") or aggregate tokens ("token) were used
            model_config (dict): Dictionary of all configuration settings of the model
        Returns:
            LSTM network model that is applicable to datasets produced according to the given configuration
        """
        if coordinate == "position":
            model = self.get_coordinate_model(model_config)
        else:  # coordinate == "token"
            model = self.get_token_model(model_config)
        return model

    def get_token_model(self, model_config: dict) -> keras.Sequential:
        """Generate a LSTM network to predict the winner of a round based on position-token trajectory

        Args:
            model_config (dict): Dictionary of all configuration settings of the model
                nodes_per_layer: An integer determining how many nodes each network layer should have
                input_shape (tuple): Tuple of the shape oof the network input

        Returns:
            The sequantial tf.keras LSTM model
        """
        nodes_per_layer = model_config["nodes_per_layer"]
        input_shape = model_config["input_shape"]
        model = tf.keras.Sequential(
            [
                keras.layers.LSTM(nodes_per_layer, input_shape=input_shape),
                keras.layers.Dense(
                    nodes_per_layer, activation="relu"
                ),  # ,kernel_regularizer=tf.keras.regularizers.l2(0.001)
                keras.layers.Dropout(0.2),
                keras.layers.Dense(
                    nodes_per_layer, activation="relu"
                ),  # ,kernel_regularizer=tf.keras.regularizers.l2(0.001)
                keras.layers.Dropout(0.2),
                keras.layers.Dense(1),
            ]
        )
        return model

    def get_coordinate_model(self, model_config: dict) -> keras.Sequential:
        """Generate a LSTM network to predict the winner of a round based on player trajectories

        Args:
            model_config (dict): Dictionary of all configuration settings of the model
                nodes_per_layer (int): An integer determining how many nodes each network layer should have
                input_shape (tuple): Tuple of the shape oof the network input

        Returns:
            The sequantial tf.keras CONV2D + LSTM model
        """
        nodes_per_layer = model_config["nodes_per_layer"]
        input_shape = model_config["input_shape"]
        logging.info(input_shape)
        pooling_size_1 = (2, 2)
        if input_shape[1] == 2:
            pooling_size_2 = (2, 2)
        else:
            pooling_size_2 = (1, 2)
        model = tf.keras.Sequential(
            [
                keras.layers.TimeDistributed(
                    keras.layers.BatchNormalization(),
                    input_shape=input_shape,
                ),
                keras.layers.TimeDistributed(
                    keras.layers.Conv2D(
                        nodes_per_layer / 2,
                        pooling_size_1,
                        activation="elu",
                        padding="same",
                    ),
                ),
                keras.layers.TimeDistributed(
                    keras.layers.AveragePooling2D(
                        pool_size=(1, 2),
                        strides=(1, 1),
                        padding="same",
                    )
                ),
                keras.layers.TimeDistributed(
                    keras.layers.Conv2D(
                        nodes_per_layer,
                        pooling_size_1,
                        activation="elu",
                        padding="same",
                    ),
                    input_shape=input_shape,
                ),
                keras.layers.TimeDistributed(
                    keras.layers.AveragePooling2D(
                        pool_size=pooling_size_2,
                        strides=(1, 1),
                        padding="same",
                    )
                ),
                keras.layers.TimeDistributed(keras.layers.Flatten()),
                keras.layers.LSTM(nodes_per_layer),
                keras.layers.Dense(
                    nodes_per_layer, activation="relu"
                ),  # ,kernel_regularizer=tf.keras.regularizers.l2(0.001)
                keras.layers.Dropout(0.2),
                keras.layers.Dense(
                    nodes_per_layer, activation="relu"
                ),  # ,kernel_regularizer=tf.keras.regularizers.l2(0.001)
                keras.layers.Dropout(0.2),
                keras.layers.Dense(1),
            ]
        )
        return model

    def plot_model(
        self,
        history: tf.keras.callbacks.History,
        plot_path: str,
        dnn_snippet: str,
        config_snippet: str,
    ):
        """Plots and logs results for training and evaluating the model corresponding to the configuration

        Args:
            History: History object from model.fit
            plot_path: A string of the path of the directory where the resultant plots should be saved to
            dnn_snippet (str): String containing all dnn model related configurations to include in the file name
            config_snippet (str): String containing all dataset configurations to include in the file name

        Returns:
            None (Logs evaluation loss and accuarcy from the test set and produces plots of training vs val loss and accuracy during training)
        """
        logging.info("Plotting model training progession")
        # Plot training progress
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
            os.path.join(
                plot_path, f"train_val_loss_{dnn_snippet}_{config_snippet}.png"
            )
        )
        plt.close()

        plt.plot(epochs, acc, "bo", label="Training acc")
        plt.plot(epochs, val_acc, "b", label="Validation acc")
        plt.title("Training and validation accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend(loc="lower right")
        plt.savefig(
            os.path.join(plot_path, f"train_val_acc_{dnn_snippet}_{config_snippet}.png")
        )
        plt.close()
