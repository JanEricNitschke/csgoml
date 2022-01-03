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

Debug=False
if Debug:
    logging.basicConfig(filename='D:\CSGO\ML\CSGOML\Test.log', encoding='utf-8', level=logging.DEBUG,filemode='w')
else:
    logging.basicConfig(filename='D:\CSGO\ML\CSGOML\Test.log', encoding='utf-8', level=logging.INFO,filemode='w')

with open("D:\CSGO\ML\CSGOML\Prepared_Input_Tensorflow.json", encoding='utf-8') as PreAnalyzed:
    dataframe=pd.read_json(PreAnalyzed)
    round_trajectory=pd.DataFrame(dataframe.iloc[30]["position_df"])
    logging.info(dataframe)
    logging.info(round_trajectory)
    