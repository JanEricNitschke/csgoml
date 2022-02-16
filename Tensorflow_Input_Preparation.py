#!/usr/bin/env python

from curses import pair_content
import os
import sys
import logging
import argparse
from awpy.analytics.nav import generate_position_token
import pandas as pd
from awpy.data import NAV
import json

def Initialize_round_positions():
    round_positions={}
    round_positions["Tick"]=[]
    round_positions["token"]=[]
    for side in ["CT","T"]:
        round_positions[side+"token"]=[]
        for number in range(1,6):
            for feature in ["Alive","Name","x","y","z"]:
                round_positions[side+"Player"+str(number)+feature]=[]
    return round_positions

def Initialize_position_dataset_dict():
    position_dataset_dict={}
    position_dataset_dict["MatchID"]=[]
    position_dataset_dict["MapName"]=[]
    position_dataset_dict["Round"]=[]
    position_dataset_dict["Winner"]=[]
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
    NoneFrames=(round["frames"]==None)
    if NoneFrames:
        logging.error(round)
        logging.error("Found none frames in round "+str(round["roundNum"])+"!")
        return True
    EmptyFrames=(len(round["frames"])==0)
    if EmptyFrames:
        logging.error("Found empty frames in round "+str(round["roundNum"])+"!")
        return True
    return False

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
                logging.debug("An alive key has length 0. Padding to length of tick!")
                logging.debug("Start tick: "+str(round_positions["Tick"][0]))
                round_positions[key]=[0]*len(round_positions["Tick"])
            round_positions[key] += [0]*(len(round_positions["Tick"])-len(round_positions[key]))
        elif "Player" in key:
            # If a player wasnt there for the whole round set his name as Nobody and position as 0,0,0.
            if len(round_positions[key])==0:
                if "Name" in key:
                    round_positions[key]=["Nobody"]*len(round_positions["Tick"])
                else:
                    round_positions[key]=[0.0]*len(round_positions["Tick"])
            # If a player left mid round pad his name and position with the last values from when he was there. Exactly like it would be if he had died "normally"
            round_positions[key] += [round_positions[key][-1]]*(len(round_positions["Tick"])-len(round_positions[key]))
    length=CheckSize(round_positions)


def PartialStep(current,previous,secondDifference,stepValue):
    return ((current-previous)/secondDifference*(secondDifference-stepValue+1)+previous)

def AppendToRoundPositions(round_positions,side,id_number_dict,PlayerID,player,secondDifference):
    # Add the relevant information of this player to the rounds dict.
    # Add name of the player. Mainly for debugging purposes. Will be removed for actual analysis
    for i in range(secondDifference,0,-1):
        round_positions[side.upper()+"Player"+id_number_dict[side][str(PlayerID)]+"Name"].append(player["name"] if i==1 else round_positions[side.upper()+"Player"+id_number_dict[side][str(PlayerID)]+"Name"][-1])
        # Is alive status so the model does not have to learn that from stopping trajectories
        round_positions[side.upper()+"Player"+id_number_dict[side][str(PlayerID)]+"Alive"].append(int(player["isAlive"]) if i==1 else round_positions[side.upper()+"Player"+id_number_dict[side][str(PlayerID)]+"Alive"][-1])
        round_positions[side.upper()+"Player"+id_number_dict[side][str(PlayerID)]+"x"].append(PartialStep(player["x"],round_positions[side.upper()+"Player"+id_number_dict[side][str(PlayerID)]+"x"][-1],secondDifference,i))
        round_positions[side.upper()+"Player"+id_number_dict[side][str(PlayerID)]+"y"].append(PartialStep(player["y"],round_positions[side.upper()+"Player"+id_number_dict[side][str(PlayerID)]+"y"][-1],secondDifference,i))
        round_positions[side.upper()+"Player"+id_number_dict[side][str(PlayerID)]+"z"].append(PartialStep(player["z"],round_positions[side.upper()+"Player"+id_number_dict[side][str(PlayerID)]+"z"][-1],secondDifference,i))

def ConvertWinnerToInt(WinnerString):
    if WinnerString=="CT":
        return 1
    elif WinnerString=="T":
        return 0
    else:
        logging.error("Winner has to be either CT or T, but was "+WinnerString)
        sys.exit

def RegularizeCoordinates(Coordinate,minimum,maximum):
    shift=(maximum+minimum)/2
    scaling=(maximum-minimum)/2
    return (Coordinate-shift)/scaling

def GetExtremesFromNAV(map_name):
    minimum={"x":sys.maxsize,"y":sys.maxsize,"z":sys.maxsize}
    maximum={"x":-sys.maxsize,"y":-sys.maxsize,"z":-sys.maxsize}
    if map_name not in NAV.keys():
        minimum={"x":-2000,"y":-2000,"z":-200}
        maximum={"x":2000,"y":2000,"z":200}
    else:
        for area in NAV[map_name]:
            for feature in ["x","y","z"]:
                for corner in ["northWest","southEast"]:
                    maximum[feature]=max(NAV[map_name][area][corner+feature.upper()],maximum[feature])
                    minimum[feature]=min(NAV[map_name][area][corner+feature.upper()],minimum[feature])
    return minimum, maximum


def RegularizeCoordinatesdf(position_df,map_name):
    minimum, maximum = GetExtremesFromNAV("de_"+map_name)
    for side in ["CT","T"]:
        for number in range(1,6):
            for feature in ["x","y","z"]:
                    position_df[side+"Player"+str(number)+feature]=position_df[side+"Player"+str(number)+feature].apply(RegularizeCoordinates,args=(minimum[feature],maximum[feature]))
    return position_df

def GetTokenLength(map):
    area_names=[]
    if map not in NAV.keys():
        return 1
    for area in NAV[map]:
        if NAV[map][area]["areaName"] not in area_names:
            area_names.append(NAV[map][area]["areaName"])
    return len(area_names)


# def GetMinMaxFromFirst(reference_position_df):
#     minimum={"x":sys.maxsize,"y":sys.maxsize,"z":sys.maxsize}
#     maximum={"x":-sys.maxsize,"y":-sys.maxsize,"z":-sys.maxsize}
#     for feature in ["x","y","z"]:
#         for side in ["CT","T"]:
#             for number in range(1,6):
#                 maximum[feature]=max(reference_position_df[side+"Player"+str(number)+feature].max(),maximum[feature])
#                 minimum[feature]=min(reference_position_df[side+"Player"+str(number)+feature].min(),minimum[feature])
#     return minimum,maximum


def main(args):
    parser = argparse.ArgumentParser("Analyze the early mid fight on inferno")
    parser.add_argument("-d", "--debug",  action='store_true', default=False, help="Enable debug output.")
    parser.add_argument("--dir",  default="D:\CSGO\Demos\Maps", help="Path to directory containing the individual map directories.")
    parser.add_argument("-l", "--log",  default='D:\CSGO\ML\CSGOML\Tensorflow_Input_Preparation.log', help="Path to output log.")
    options = parser.parse_args(args)

    if options.debug:
        logging.basicConfig(filename=options.log, encoding='utf-8', level=logging.DEBUG,filemode='w')
    else:
        logging.basicConfig(filename=options.log, encoding='utf-8', level=logging.INFO,filemode='w')

    logging.info("Starting")
    #done=["ancient","cache","cbble","cs_rush","dust2","facade","inferno","marquis","mirage","mist","nuke","overpass","resort","santorini","santorini_playtest","season"]
    done=[]
    do=["cs_rush"]
    # More comments and split stuff into functions
    for directoryname in os.listdir(options.dir):
        directory=os.path.join(options.dir,directoryname)
        if os.path.isdir(directory):
            logging.info("Looking at directory "+directory)
            if directoryname in done:
                logging.info("Skipping this directory as it has already been analyzed.")
                continue
            if  ((len(do) > 0) and (directoryname not in do)):
                logging.info("Skipping this directory as it not one of those that should be analyzed.")
                continue
            position_dataset_dict=Initialize_position_dataset_dict()
            for filename in os.listdir(directory):
                f = os.path.join(directory, filename)
                # checking if it is a file
                if os.path.isfile(f):
                    if filename.endswith(".json"):
                        logging.info("Analyzing file "+filename)
                        MatchID=filename.rsplit(".",1)[0]
                        with open(f, encoding='utf-8') as f:
                            data = json.load(f)
                        map_name=data["mapName"]
                        tokenlength=GetTokenLength(map_name)
                        for round in data["gameRounds"]:
                            SkipRound=False
                            # If there are no frames in the round skip it.
                            if EmptyFrames(round):
                                continue
                            # Dict for mapping players steamID to player number for each round
                            id_number_dict={"t":{},"ct":{}}
                            # Dict to check if mapping has already been initialized this round
                            dict_initialized={"t": False, "ct": False}
                            # Initialize the dict that tracks player position,status,name for each round
                            round_positions = Initialize_round_positions()
                            logging.debug("Round number "+str(round["roundNum"]))
                            # Iterate over each frame in the round
                            currentTick=0
                            lastTick=0
                            for frame in round["frames"]:
                                # There should never be more than 5 players alive in a team.
                                # If that does happen completely skip the round.
                                # Propagate that information past the loop by setting SkipRound to true
                                if frame["ct"]["alivePlayers"]>5 or frame["t"]["alivePlayers"]>5:
                                    logging.error("Found frame with more than 5 players alive in a team in round "+str(round["roundNum"])+"!")
                                    SkipRound=True
                                    break
                                # Loop over both sides
                                currentTick=int(frame["tick"])
                                if lastTick==0:
                                    secondDifference=1
                                else:
                                    secondDifference=int((currentTick-lastTick)/128)
                                for side in ["ct", "t"]:
                                    # If the side does not contain any players for that frame skip it
                                    if frame[side]["players"]==None:
                                        logging.debug("Side['players'] is none. Skipping this frame from round "+str(round["roundNum"])+"!")
                                        continue
                                    # Loop over each player in the team.
                                    for n, player in enumerate(frame[side]["players"]):
                                        #logging.info(f)
                                        PlayerID=GetPlayerID(player)
                                        # If the dict of the team has not been initialized add that player. Should only happen once per player per team per round
                                        # But for each team can happen on different rounds in some rare cases.
                                        if dict_initialized[side]==False:
                                            id_number_dict[side][str(PlayerID)]=str(n+1)
                                        #logging.debug(id_number_dict[side])
                                        # If a player joins mid round (either a bot due to player leaving or player (re)joining)
                                        # do not use him for this round.
                                        if str(PlayerID) not in id_number_dict[side]:
                                            continue
                                        AppendToRoundPositions(round_positions,side,id_number_dict,PlayerID,player,secondDifference)
                                    # After looping over each player in the team once the steamID matching has been initialized
                                    dict_initialized[side]=True
                                # If at least one side has been initialized the round can be used for analysis, so add the tick value used for tracking.
                                # Will also removed for the eventual analysis.
                                # But you do not want to set it for frames where you have no player data which should only ever happen in the first frame of a round at worst.
                                if True in dict_initialized.values():
                                    for i in range(secondDifference,0,-1):
                                        round_positions["Tick"].append(PartialStep(currentTick,lastTick,secondDifference,i))
                                        try:
                                            tokens=generate_position_token(map_name, frame)
                                        except TypeError:
                                            tokens={'tToken': tokenlength*'0','ctToken': tokenlength*'0','token': 2*tokenlength*'0'}
                                            logging.debug("Got TypeError when trying to generate position token. This is due to one sides 'player' entry being none.")
                                        except KeyError:
                                            tokens={'tToken': tokenlength*'0','ctToken': tokenlength*'0','token': 2*tokenlength*'0'}
                                            logging.debug("Got KeyError when trying to generate position token. This is due to the map not being supported.")
                                        except ValueError:
                                            tokens={'tToken': tokenlength*'0','ctToken': tokenlength*'0','token': 2*tokenlength*'0'}
                                            logging.debug("Got KeyError when trying to generate position token. This is due to the map not being supported.")
                                        round_positions["token"].append(tokens["token"] if i==1 else round_positions["token"][-1])
                                        round_positions["CTtoken"].append(tokens["ctToken"] if i==1 else round_positions["ctToken"][-1])
                                        round_positions["Ttoken"].append(tokens["tToken"] if i==1 else round_positions["tToken"][-1])
                                    else:
                                        pass
                                    lastTick=currentTick
                            # Skip the rest of the loop if the whole round should be skipped.
                            if SkipRound:
                                continue
                            # Append demo id, map name and round number to the final dataset dict.
                            position_dataset_dict["MatchID"].append(MatchID)
                            position_dataset_dict["MapName"].append(map_name)
                            position_dataset_dict["Round"].append(round["endTScore"]+round["endCTScore"])
                            position_dataset_dict["Winner"].append(ConvertWinnerToInt(round["winningSide"]))
                            # Pad to full length in case a player left
                            PadToFullLength(round_positions)
                            # Make sure each entry in the round_positions has the same size now. Especially that nothing is longer than the Tick entry which would indicate multiple players filling on player number
                            # Transform to dataframe
                            round_positions_df=pd.DataFrame(round_positions)
                            # Add the rounds trajectory information to the overall dataset.
                            position_dataset_dict["position_df"].append(round_positions_df)
                            # Check thateach entry in the dataset has the same length. Especially that for each round there is a trajectory dataframe.
                            length=CheckSize(position_dataset_dict)
                            logging.debug("Finished another round and appended to dataset. Now at size "+str(length))
            # Transform to dataset and write it to file as json
            position_dataset_df=pd.DataFrame(position_dataset_dict)
            position_dataset_df["position_df"]=position_dataset_df["position_df"].apply(RegularizeCoordinatesdf,args=(directoryname,))
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