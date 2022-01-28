#!/usr/bin/env python

import os
import logging
import pandas as pd
import numpy as np
import argparse
import sys
from csgo.data import NAV
import tensorflow as tf
from sklearn.model_selection import train_test_split

def transformToDataFrame(JsonFormatDict):
    return pd.DataFrame(JsonFormatDict)


def modifyDataFrameShape(position_df,team,time,coordinates):
    # Throw away all unneeded columns from the dataframe and then convert it into a numpy array
    # If coordinates is false than it is a 1-D array of the tokens for each timestep
    # If coordinates is true than it is a 4-D array with the first index representing the timestep, the second the team, the third the playernumber in that team and the fourth the feature.
    # Set length of dataframe to make sure all have the same size
    # Pad each column with its last entry if set size is larger than dataframe size
    if time>len(position_df):
        idx = np.minimum(np.arange(time), len(position_df) - 1)
        position_df=position_df.iloc[idx]
    else:
        # Cut if the required size is smaller.
        position_df=position_df.head(time)

    # If the full coordinates should be used then for each player their alive status as well as their x,y,z coordinates are kept, everything else (player names and tokens) is discarded.
    if coordinates:
        #ColumnsToKeep=[]
        dimensions=[time]
        featurelist=["Alive","x","y","z"]
        if team=="BOTH":
            sides=["CT","T"]
            # Dimension with size 2 for teams
            dimensions.append(2)
        else:
            sides=[team]
            # Dimension with size 1 for team
            dimensions.append(1)
        # Dimension with size 5 for players
        dimensions.append(5)
        # Dimensions with size 4 for features
        dimensions.append(4)
        # Only keep the information for the side(s) that should be used.
        return_array=np.zeros(tuple(dimensions))
        for side in sides:
            for number in range(1,6):
                for feature in featurelist:
                    return_array[:,sides.index(side),number-1,featurelist.index(feature)]=position_df[side+"Player"+str(number)+feature].to_numpy()
    else:
        # Remove all columns from the dataframe except for the column of interest
        if team=="BOTH":
            return_array=position_df["token"].to_numpy()
        else:
            return_array=position_df[team+"token"].to_numpy()
    return return_array

def getMaximumLength(position_dfs):
    lengths=position_dfs["position_df"].apply(len)
    return lengths.max()

def TransFormDataframeToArrays(dataframe):
    label_array=dataframe["Winner"].to_numpy()
    feature_array=np.zeros(tuple([len(dataframe["position_df"])]+list(dataframe.iloc[0]["position_df"].shape)))
    for index, array in dataframe["position_df"].iteritems():
        feature_array[index]=array
    return label_array, feature_array


def main(args):
    parser = argparse.ArgumentParser("Analyze the early mid fight on inferno")
    parser.add_argument("-d", "--debug",  action='store_true', default=False, help="Enable debug output.")
    parser.add_argument("-m", "--map",  default="overpass", help="Map to analyze")
    parser.add_argument("-l", "--log",  default='D:\CSGO\ML\CSGOML\Test.log', help="Path to output log.")
    parser.add_argument("-s", "--side",  default='BOTH', help="Which side to include in analysis (CT,T,BOTH) .")
    parser.add_argument("--coordinates",  action='store_true', default=False, help="Whether to use full coordinate or just tokens.")
    parser.add_argument("-t", "--time",  default='MAX', help="How many position snapshots should be used. Either an integer or MAX.")
    parser.add_argument("--exampleid", type=int, default=22, help="For which round the position_df should be printed when debugging.")
    parser.add_argument("--randomstate", type=int, default=8, help="Random state for train_test_split")
    options = parser.parse_args(args)

    if options.debug:
        logging.basicConfig(filename=options.log, encoding='utf-8', level=logging.DEBUG,filemode='w')
    else:
        logging.basicConfig(filename=options.log, encoding='utf-8', level=logging.INFO,filemode='w')



    ExampleIndex=options.exampleid
    RandomState=options.randomstate

    if options.side not in ["CT","T","BOTH"]:
        logging.error("Side "+options.side+" unknown. Has to be one of 'CT','T','BOTH'!")
        sys.exit

    # Time should either an integer or MAX.
    # If it is an integer n that means that the first n timesteps will be considered for each round and the rest discarded.
    # If it is 'MAX' then we look what the longest round in the dataset is and pad all other rounds to that length.
    # Should probably do the Training,Test split before and only look at the max length of the training set. TODO
    try:
        int(options.time)
    except ValueError:
        if  not options.time =="MAX":
            logging.error("Time "+options.time+" unknown. Has to be either an integer or 'MAX'!")
            sys.exit

    # Read in the prepared json file.
    File="D:\CSGO\Demos\Maps\\"+options.map+"\Analysis\Prepared_Input_Tensorflow_"+options.map+".json"
    if os.path.isfile(File):
        with open(File, encoding='utf-8') as PreAnalyzed:
            dataframe=pd.read_json(PreAnalyzed)
    else:
        logging.error("File "+File+" does not exist! Probably because map "+options.map+" does not exist!")

    logging.debug("Initial dataframe:")
    logging.debug(dataframe)


    # Only keep the laben (Winner) and training feature (position_df) and discard the MatchID, Map_name and round number.
    dataframe=dataframe[["Winner","position_df"]]
    dataframe["position_df"]=dataframe["position_df"].apply(transformToDataFrame)

    logging.debug("Example for position_df dataframe entry before cleaning.")
    logging.debug(dataframe.iloc[ExampleIndex]["position_df"])

    # Get the time length to be used.
    if options.time =="MAX":
        time=getMaximumLength(dataframe)
    else:
        time=int(options.time)
    logging.debug("Time has been set to "+str(time)+"!")

    # Transform the feature to the proper length and discard all columns that should not be used.
    dataframe["position_df"]=dataframe["position_df"].apply(modifyDataFrameShape,args=(options.side,time,options.coordinates))

    logging.debug("Dataframe after cleanup")
    logging.debug(dataframe)
    logging.debug("Example for position_df dataframe entry after cleaning.")
    logging.debug(dataframe.iloc[ExampleIndex]["position_df"])

    # Shuffle and split dataframe into training, test and validation set
    # set aside 20% of train and test data for evaluation
    train_df, test_df = train_test_split(dataframe, test_size=0.2, shuffle = True, random_state = RandomState)
    # Use the same function above for the validation set  0.25 x 0.8 = 0.2
    train_df, val_df = train_test_split(train_df, test_size=0.25, shuffle = True, random_state = RandomState)
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    test_labels, test_features = TransFormDataframeToArrays(test_df)
    val_labels, val_features = TransFormDataframeToArrays(val_df)
    train_labels, train_features = TransFormDataframeToArrays(train_df)



    logging.info("Input preparation done!")
    logging.info(train_labels.shape)
    logging.info(train_features.shape)


    #input_tensor=tf.convert_to_tensor(dataframe)
    #logging.info(input_tensor)


if __name__ == '__main__':
    main(sys.argv[1:])