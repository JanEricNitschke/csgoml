# CSGOML
`CSGOML` is a collection of [python](https://www.python.org/downloads/) scripts to do [CS:GO](https://store.steampowered.com/app/730/CounterStrike_Global_Offensive/?l=german) data analysis utilizing the [awpy](https://github.com/pnxenopoulos/awpy) package for data parsing.

# demo_analyzer_sort.py
Contains a class automating the parsing of multiple files in succession with awpy and sorting the resulting json files by map.

Useful when you have accumulated a large collection of your own demos and/or when doing map specific data analysis.

# fight_analyzer.py
Contains a class for analyzing a specifically defined engagement for whether it is T or CT favoured.

Running the script adds every kill as well as information about the map, players locations and weapons as well as the time to a MySQL database.

It also supports the ability to query specific situations (by map, player weapons and positions and game time) for their CT win percentage.

The latter functionality has also been given a front-end either [here](https://github.com/JanEricNitschke/CSGOML/tree/main/AWS_Steps/FightAnalyzer) or updated [here](https://github.com/JanEricNitschke/AngularFightAnalyzer)

# tensorflow_input_preparation.py
A script that produces, for each map separately, a json file containing different configurations of player trajectory data for each round played on that map.

This is in perparation of further analysis to separate the extensive cleaning neccessary from the final analysis.

# read_tensorflow_input.py, trajectory_handler.py, trajectory_predictor.py and trajectory_clusterer.py
Contain classes designed to read in the json files produced by tensorflow_input_preparation.py and train LSTM networks to predict a winner of a round based on player trajectory data or cluster rounds based on player trajectories.

It supports the option to chose between which side(s) to consider, limit the data to only contain the first n seconds and to chose
between using each players full x, y and z coordinates or a tokenized version as described in [ggViz: Accelerating Large-Scale Esports Game Analysis](https://arxiv.org/pdf/2107.06495.pdf) and implemented in [awpy](https://github.com/pnxenopoulos/awpy).

# download_demos.py and demo_watchdog.py
Two scripts used to build a dataset large enough to enable machine learning techniques to fulfill their potential.

download_demos.py downloads the demos from professional CS:GO games tracked on [hltv](https://www.hltv.org/).

demo_watchdog.py then unpacks the resulting rar file
and calls demo_analyzer_sort.py to parse the demos to json files and store them based on the map played.

The full demos is subsequently deleted as hard disk requirement needed to store all demos in full are currently infeasible for me.

Currently more than 1000 matches (>2000 maps with over 50000 rounds) have been accumulated.

# plot_utils.py

This is a module containing various functions that augment already existing plotting functions present in [awpy](https://github.com/pnxenopoulos/awpy).
Specifically the plotting of position tokens, visualization of named areas and multi-round plotting.
Run as a script it illustrates the basic functionality of these functions as well as the basic ones directly from awpy.

# nav_utils.py
This is a module containing functions augmenting navigation capabilities of [awpy](https://github.com/pnxenopoulos/awpy).
It contains functions to precompute distance matrices for both areas and places of each map.
Included are also plotting functions to validate the results.
The matrices for the places and for same maps for the area are included in this repo while for other maps the area matrices are too large and thus stored separately [here](https://cernbox.cern.ch/index.php/s/T1BJ69qKLlK1fSu).