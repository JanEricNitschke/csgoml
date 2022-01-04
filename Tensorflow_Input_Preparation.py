#!/usr/bin/env python

from csgo.parser import DemoParser
import os
import sys
import logging
import shutil
import argparse
from csgo.visualization.plot import position_transform
import pandas as pd
from csgo.data import MAP_DATA

def Initialize_round_positions():
    round_positions={}
    round_positions["Tick"]=[]
    for side in ["CT","T"]:
        for number in range(1,6):
            for feature in ["Alive","Name","x","y","z"]:
                round_positions[side+"Player"+str(number)+feature]=[]
    return round_positions

def Initialize_position_dataset_dict():
    position_dataset_dict={}
    position_dataset_dict["MatchID"]=[]
    position_dataset_dict["MapName"]=[]
    position_dataset_dict["Round"]=[]
    position_dataset_dict["position_df"]=[]
    return position_dataset_dict

def CheckSize(dict):
    length=0
    SameSize=True
    for key in dict:
        logging.debug("Length of key " + key + " is " + str(len(dict[key])))
        if length==0:
            length=len(dict[key])
        else:
            SameSize=(length==len(dict[key]))
            if not SameSize:
                logging.error("Key " + key + " has size " + str(len(dict[key])) + " while " + str(length) + " is expected!")
                logging.error(dict)
                sys.exit("Not all elements in dict have the same size. Something has gone wrong.")
    return length

def EmptyFrames(round):
    emptyFrames=(round["frames"]==None or len(round["frames"])==0)
    if emptyFrames:
        logging.error("Found empty frames in round "+str(round["roundNum"])+"!")
    return emptyFrames

def GetPlayerID(player):
    # Bots do not have a steamID.
    # Use their name instead to not cause conflicts for rounds that have multiple bots in a team.
    if player["steamID"]==0:
        return player["name"]
    else:
        return player["steamID"]

def PadToFullLength(round_positions):
    for key in round_positions:
        if "Alive" in key:
            # If the Alive list is completely empty fill it with a dead player
            # If the player left mid round he is considered dead for the time after leaving, so pad it to full length with False
            if len(round_positions[key])==0:
                logging.error("An alive key has length 0. Padding to length of tick!")
                round_positions[key]=[False]*len(round_positions["Tick"])
            round_positions[key] += [False]*(len(round_positions["Tick"])-len(round_positions[key]))
        elif "Player" in key:
            # If a player wasnt there for the whole round set his name as Nobody and position as 0,0,0.
            if len(round_positions[key])==0:
                if "Name" in key:
                    round_positions[key]=["Nobody"]*len(round_positions["Tick"])
                else:
                    round_positions[key]=[0.0]*len(round_positions["Tick"])
            #logging.info(key)
            # If a player left mid round pad his name and position with the last values from when he was there. Exactly like it would be if he had died "normally"
            round_positions[key] += [round_positions[key][-1]]*(len(round_positions["Tick"])-len(round_positions[key]))
    length=CheckSize(round_positions)

def AppendToRoundPositions(round_positions,side,id_number_dict,PlayerID,p,map_name):
    # Add the relevant information of this player to the rounds dict.
    # Add name of the player. Mainly for debugging purposes. Will be removed for actual analysis
    round_positions[side.upper()+"Player"+id_number_dict[side][str(PlayerID)]+"Name"].append(p["name"])
    # Is alive status so the model does not have to learn that from stopping trajectories
    round_positions[side.upper()+"Player"+id_number_dict[side][str(PlayerID)]+"Alive"].append(p["isAlive"])
    # Transform coordinates, not sure if actually needed or helpful
    if map_name in MAP_DATA:
        pos = (
            position_transform(map_name, p["x"], "x"),
            position_transform(map_name, p["y"], "y"),
        )
    else:
        pos = (p["x"],p["y"])
    round_positions[side.upper()+"Player"+id_number_dict[side][str(PlayerID)]+"x"].append(pos[0])
    round_positions[side.upper()+"Player"+id_number_dict[side][str(PlayerID)]+"y"].append(pos[1])
    round_positions[side.upper()+"Player"+id_number_dict[side][str(PlayerID)]+"z"].append(p["z"])


def main(args):
    parser = argparse.ArgumentParser("Analyze the early mid fight on inferno")
    parser.add_argument("-d", "--debug",  action='store_true', default=False, help="Enable debug output.")
    parser.add_argument("--dir",  default="D:\CSGO\Demos\Maps", help="Path to directory containing the individual map directories.")
    parser.add_argument("-l", "--log",  default='D:\CSGO\ML\CSGOML\Preperation_Tensorflow.log', help="Path to output log.")
    options = parser.parse_args(args)

    if options.debug:
        logging.basicConfig(filename=options.log, encoding='utf-8', level=logging.DEBUG,filemode='w')
    else:
        logging.basicConfig(filename=options.log, encoding='utf-8', level=logging.INFO,filemode='w')

    logging.info("Starting")

    dir=options.dir
    os.chdir(dir)

    demo_parser = DemoParser(parse_rate=128, buy_style="hltv",dmg_rolled=True,parse_frames=True)

    # More comments and split stuff into functions
    for directoryname in os.listdir(dir):
        directory=os.path.join(dir,directoryname)
        if os.path.isdir(directory):
            os.chdir(directory)
            demo_parser.outpath=directory
            logging.info("Looking at directory "+directory)
            position_dataset_dict=Initialize_position_dataset_dict()
            for filename in os.listdir(directory):
                f = os.path.join(directory, filename)
                # checking if it is a file
                if os.path.isfile(f):
                    if filename.endswith(".json"):
                        logging.info("Analyzing file "+filename)
                        MatchID=filename.rsplit(".",1)[0]
                        demo_parser.output_file=MatchID+".json"
                        data = demo_parser._read_json()
                        map_name=data["mapName"]
                        for round in data["gameRounds"]:
                            SkipRound=False
                            # If there are no frames in the round skip it.
                            if EmptyFrames(round):
                                continue
                            # Dict for mapping players steamID to player number for each round
                            id_number_dict={"t":{},"ct":{}}
                            # dict to check if mapping has already been initialized this round
                            dict_initialized={"t": False, "ct": False}
                            # initialize the dict that tracks player position,status,name for each round
                            round_positions = Initialize_round_positions()
                            logging.debug("Round number "+str(round["roundNum"]))
                            #logging.info(round["roundNum"])
                            # Iterate over each frame in the round
                            for f in round["frames"]:
                                #logging.info(f)
                                # There should never be more than 5 players alive in a team.
                                # If that does happen completely skip the round.
                                # Propagate that information past the loop by setting SkipRound to true
                                if f["ct"]["alivePlayers"]>5 or f["t"]["alivePlayers"]>5:
                                    logging.error("Found frame with more than 5 players alive in a team in round "+str(round["roundNum"])+"!")
                                    SkipRound=True
                                    break
                                # Loop over both sides
                                for side in ["ct", "t"]:
                                    # If the side does not contain any players for that frame skip it
                                    if f[side]["players"]==None:
                                        logging.error("Side['players'] is none. Skipping this frame from round "+str(round["roundNum"])+"!")
                                        continue
                                    # Loop over each player in the team.
                                    for n, p in enumerate(f[side]["players"]):
                                        #logging.info(f)
                                        PlayerID=GetPlayerID(p)
                                        # If the dict of the team has not been initialized add that player. Should only happen once per player per team per round
                                        # But for each team can happen on different rounds in some rare cases.
                                        if dict_initialized[side]==False:
                                            id_number_dict[side][str(PlayerID)]=str(n+1)
                                        #logging.debug(id_number_dict[side])
                                        # If a player joins mid round (either a bot due to player leaving or player (re)joining)
                                        # do not use him for this round.
                                        if str(PlayerID) not in id_number_dict[side]:
                                            continue
                                        #logging.info(id_number_dict)
                                        AppendToRoundPositions(round_positions,side,id_number_dict,PlayerID,p,map_name)
                                    # After looping over each player in the team once the steamID matching has been initialized
                                    dict_initialized[side]=True
                                # If at least one side has been initialized the round can be used for analysis, so add the tick value used for tracking.
                                # Will also removed for the eventual analysis.
                                # But you do not want to set it for frames where you have no player data which should only ever happen in the first frame of a round at worst.
                                if True in dict_initialized.values():
                                    round_positions["Tick"].append(f["tick"])

                            # Skip the rest of the loop if the whole round should be skipped.
                            if SkipRound:
                                continue
                            # Append demo id, map name and round number to the final dataset dict.
                            position_dataset_dict["MatchID"].append(MatchID)
                            position_dataset_dict["MapName"].append(map_name)
                            position_dataset_dict["Round"].append(round["endTScore"]+round["endCTScore"])
                            #logging.info(round_positions)
                            # Pad to full length in case a player left
                            PadToFullLength(round_positions)
                            # Make sure each entry in the round_positions has the same size now. Especially that nothing is longer than the Tick entry which would indicate multiple players filling on player number
                            # Transform to dataframe
                            round_positions_df=pd.DataFrame(round_positions)
                            #logging.info(round_positions_df)
                            # Add the rounds trajectory information to the overall dataset.
                            position_dataset_dict["position_df"].append(round_positions_df)
                            # Check thateach entry in the dataset has the same length. Especially that for each round there is a trajectory dataframe.
                            length=CheckSize(position_dataset_dict)
                            logging.debug("Finished another round and appended to dataset. Now at size "+str(length))
                        #logging.info(position_dataset_dict)
            # Transform to dataset and write it to file as json
            position_dataset_df=pd.DataFrame(position_dataset_dict)
            #logging.info(position_dataset_df)
            position_dataset_df.to_json(directory+"\Analysis\Prepared_Input_Tensorflow_"+directoryname+".json")
            logging.info("Wrote output json to: "+directory+"\Analysis\Prepared_Input_Tensorflow_"+directoryname+".json")
            # Has to be read back in like
            # with open("D:\CSGO\Demos\Maps\vertigo\Analysis\Prepared_Input_Tensorflow_vertigo.json", encoding='utf-8') as PreAnalyzed:
            #   dataframe=pd.read_json(PreAnalyzed)
            #   round_df=pd.DataFrame(dataframe.iloc[30]["position_df"])
            #   logging.info(dataframe)
            #   logging.info(pd.DataFrame(dataframe.iloc[30]["position_df"]))

if __name__ == '__main__':
    main(sys.argv[1:])