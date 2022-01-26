#!/usr/bin/env python

from turtle import pos
from csgo.analytics.nav import find_closest_area
from csgo.parser import DemoParser
import os
import json
import logging
import pandas as pd
import numpy as np
import argparse
import sys
from csgo.analytics import nav
from csgo.data import NAV

def transformToDataFrame(JsonFormatDict):
    return pd.DataFrame(JsonFormatDict)


def modifyDataFrameShape(position_df,team,time,coordinates):
    if coordinates:
        ColumnsToKeep=[]
        if team=="BOTH":
            sides=["CT","T"]
        else:
            sides=[team]
        for side in sides:
            for number in range(1,6):
                for feature in ["Alive","x","y","z"]:
                    ColumnsToKeep.append(side+"Player"+str(number)+feature)
        position_df=position_df[ColumnsToKeep]

        # Still need to regularize the coordinates
    else:
        # Remove all columns from the dataframe except for the column of interest
        if team=="BOTH":
            position_df=position_df[["token"]]
        else:
            position_df=position_df[[team+"token"]]
    # Set length of dataframe to make sure all have the same size
    # Pad each column with its last entry if set size is larger than dataframe size
    if time>len(position_df):
        idx = np.minimum(np.arange(time), len(position_df) - 1)
        position_df=position_df.iloc[idx]
    else:
        # Cut if the required size is smaller.
        position_df=position_df.head(time)
    position_df.reset_index(drop=True, inplace=True)
    return position_df

def getMaximumLength(position_dfs):
    lengths=position_dfs["position_df"].apply(len)
    return lengths.max()



def main(args):
    parser = argparse.ArgumentParser("Analyze the early mid fight on inferno")
    parser.add_argument("-d", "--debug",  action='store_true', default=False, help="Enable debug output.")
    parser.add_argument("-m", "--map",  default="overpass", help="Map to analyze")
    parser.add_argument("-l", "--log",  default='D:\CSGO\ML\CSGOML\Test.log', help="Path to output log.")
    parser.add_argument("-s", "--side",  default='BOTH', help="Which side to include in analysis (CT,T,BOTH) .")
    parser.add_argument("--coordinates",  action='store_true', default=False, help="Whether to use full coordinate or just tokens.")
    parser.add_argument("-t", "--time",  default='MAX', help="How many position snapshots should be used. Either an integer or MAX.")
    options = parser.parse_args(args)

    if options.debug:
        logging.basicConfig(filename=options.log, encoding='utf-8', level=logging.DEBUG,filemode='w')
    else:
        logging.basicConfig(filename=options.log, encoding='utf-8', level=logging.INFO,filemode='w')

    if options.side not in ["CT","T","BOTH"]:
        logging.info("Side "+options.side+" unknown. Has to be one of 'CT','T','BOTH'!")
        sys.exit

    try:
        int(options.time)
    except ValueError:
        if  not options.time =="MAX":
            logging.info("Time "+options.time+" unknown. Has to be either an integer or 'MAX'!")
            sys.exit

    File="D:\CSGO\Demos\Maps\\"+options.map+"\Analysis\Prepared_Input_Tensorflow_"+options.map+".json"
    if os.path.isfile(File):
        with open(File, encoding='utf-8') as PreAnalyzed:
            dataframe=pd.read_json(PreAnalyzed)
            logging.info("Initial dataframe:")
            logging.info(dataframe)
            dataframe=dataframe[["Winner","position_df"]]
            dataframe["position_df"]=dataframe["position_df"].apply(transformToDataFrame)
            logging.info("Example for position_df dataframe entry before cleaning.")
            logging.info(dataframe.iloc[30]["position_df"])
            if options.time =="MAX":
                time=getMaximumLength(dataframe)
            else:
                time=int(options.time)
            logging.info("Time has been set to "+str(time)+"!")
            dataframe["position_df"]=dataframe["position_df"].apply(modifyDataFrameShape,args=(options.side,time,options.coordinates))
            logging.info("Dataframe after cleanup")
            logging.info(dataframe)
            logging.info("Example for position_df dataframe entry after cleaning.")
            logging.info(dataframe.iloc[30]["position_df"])
    else:
        logging.info("File "+File+" does not exist! Probably because map "+options.map+" does not exist!")


if __name__ == '__main__':
    main(sys.argv[1:])