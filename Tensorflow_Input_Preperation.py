from csgo.parser import DemoParser
import os
import sys
import logging
import shutil
from csgo.visualization.plot import position_transform
import pandas as pd
from csgo.data import MAP_DATA

def Initialize_round_positions():
    round_positions={}
    round_positions["Tick"]=[]
    round_positions["CTPlayer1Alive"]=[]
    round_positions["CTPlayer1Name"]=[]
    round_positions["CTPlayer1x"]=[]
    round_positions["CTPlayer1y"]=[]
    round_positions["CTPlayer1z"]=[]
    round_positions["CTPlayer2Alive"]=[]
    round_positions["CTPlayer2Name"]=[]
    round_positions["CTPlayer2x"]=[]
    round_positions["CTPlayer2y"]=[]
    round_positions["CTPlayer2z"]=[]
    round_positions["CTPlayer3Alive"]=[]
    round_positions["CTPlayer3Name"]=[]
    round_positions["CTPlayer3x"]=[]
    round_positions["CTPlayer3y"]=[]
    round_positions["CTPlayer3z"]=[]
    round_positions["CTPlayer4Alive"]=[]
    round_positions["CTPlayer4Name"]=[]
    round_positions["CTPlayer4x"]=[]
    round_positions["CTPlayer4y"]=[]
    round_positions["CTPlayer4z"]=[]
    round_positions["CTPlayer5Alive"]=[]
    round_positions["CTPlayer5Name"]=[]
    round_positions["CTPlayer5x"]=[]
    round_positions["CTPlayer5y"]=[]
    round_positions["CTPlayer5z"]=[]
    round_positions["TPlayer1Alive"]=[]
    round_positions["TPlayer1Name"]=[]
    round_positions["TPlayer1x"]=[]
    round_positions["TPlayer1y"]=[]
    round_positions["TPlayer1z"]=[]
    round_positions["TPlayer2Alive"]=[]
    round_positions["TPlayer2Name"]=[]
    round_positions["TPlayer2x"]=[]
    round_positions["TPlayer2y"]=[]
    round_positions["TPlayer2z"]=[]
    round_positions["TPlayer3Alive"]=[]
    round_positions["TPlayer3Name"]=[]
    round_positions["TPlayer3x"]=[]
    round_positions["TPlayer3y"]=[]
    round_positions["TPlayer3z"]=[]
    round_positions["TPlayer4Alive"]=[]
    round_positions["TPlayer4Name"]=[]
    round_positions["TPlayer4x"]=[]
    round_positions["TPlayer4y"]=[]
    round_positions["TPlayer4z"]=[]
    round_positions["TPlayer5Alive"]=[]
    round_positions["TPlayer5Name"]=[]
    round_positions["TPlayer5x"]=[]
    round_positions["TPlayer5y"]=[]
    round_positions["TPlayer5z"]=[]
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

Debug=False
if Debug:
    logging.basicConfig(filename='D:\CSGO\ML\CSGOML\Preperation_Tensorflow.log', encoding='utf-8', level=logging.DEBUG,filemode='w')
else:
    logging.basicConfig(filename='D:\CSGO\ML\CSGOML\Preperation_Tensorflow.log', encoding='utf-8', level=logging.INFO,filemode='w')


dir="D:\CSGO\Demos\Maps"
os.chdir(dir)

demo_parser = DemoParser(parse_rate=128, buy_style="hltv",dmg_rolled=True,parse_frames=True)

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
                    dataframe=demo_parser._parse_json()
                    map_name=dataframe["mapName"]
                    for round in data["gameRounds"]:
                        SkipRound=False
                        if round["frames"]==None or len(round["frames"])==0:
                            continue
                        id_number_dict={"t":{},"ct":{}}
                        dict_initialized={"t": False, "ct": False}
                        round_positions = Initialize_round_positions()
                        logging.debug("Round number "+str(round["roundNum"]))
                        #logging.info(round["roundNum"])
                        for i, f in enumerate(round["frames"]):
                            #logging.info(f)
                            if f["ct"]["alivePlayers"]>5 or f["t"]["alivePlayers"]>5:
                                SkipRound=True
                                break
                            for side in ["ct", "t"]:
                                if f[side]["players"]==None:
                                    logging.debug("Side['players'] is none. Skipping this frame.")
                                    continue
                                for n, p in enumerate(f[side]["players"]):
                                    #logging.info(f)
                                    if p["steamID"]==0:
                                        PlayerID=p["name"]
                                    else:
                                        PlayerID=p["steamID"]
                                    if dict_initialized[side]==False:
                                        id_number_dict[side][str(PlayerID)]=str(n+1)
                                    #logging.debug(id_number_dict[side])
                                    if str(PlayerID) not in id_number_dict[side]:
                                        continue
                                        # if PlayerID==0:
                                        #     continue
                                        # else:
                                        #     id_number_dict[side][str(PlayerID)]=id_number_dict[side][str(0)]
                                        #     del id_number_dict[side][str(0)]      
                                    #logging.info(id_number_dict)                           
                                    round_positions[side.upper()+"Player"+id_number_dict[side][str(PlayerID)]+"Name"].append(p["name"])
                                    round_positions[side.upper()+"Player"+id_number_dict[side][str(PlayerID)]+"Alive"].append(p["isAlive"])
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
                                dict_initialized[side]=True
                            if True in dict_initialized.values():
                                round_positions["Tick"].append(f["tick"])
                        # Pad to full length in case a player left
                        if SkipRound:
                            continue
                        position_dataset_dict["MatchID"].append(MatchID)
                        position_dataset_dict["MapName"].append(map_name)
                        position_dataset_dict["Round"].append(round["endTScore"]+round["endCTScore"])
                        #logging.info(round_positions)
                        for key in round_positions:
                            if "Alive" in key:
                                if len(round_positions[key])==0:
                                    round_positions[key]=[False]*len(round_positions["Tick"])
                                round_positions[key] += [False]*(len(round_positions["Tick"])-len(round_positions[key]))
                            elif "Player" in key:
                                if len(round_positions[key])==0:
                                    if "Name" in key:
                                        round_positions[key]=["Nobody"]*len(round_positions["Tick"])
                                    else:
                                        round_positions[key]=[0.0]*len(round_positions["Tick"])
                                #logging.info(key)
                                round_positions[key] += [round_positions[key][-1]]*(len(round_positions["Tick"])-len(round_positions[key]))
                        length=CheckSize(round_positions)
                        round_positions_df=pd.DataFrame(round_positions)
                        #logging.info(round_positions_df.to_string())
                        position_dataset_dict["position_df"].append(round_positions_df)
                        length=CheckSize(position_dataset_dict)
                        logging.debug("Finished another round and appended to dataset. Now at size "+str(length))
                    #logging.info(position_dataset_dict)
        position_dataset_df=pd.DataFrame(position_dataset_dict)
        position_dataset_df.to_json(directory+"\Analysis\Prepared_Input_Tensorflow_"+directoryname+".json")
        logging.info("Wrote output json to: "+directory+"\Analysis\Prepared_Input_Tensorflow_"+directoryname+".json")
        # Has to be read back in like
        # with open("D:\CSGO\Demos\Maps\vertigo\Analysis\Prepared_Input_Tensorflow_vertigo.json", encoding='utf-8') as PreAnalyzed:
        #   dataframe=pd.read_json(PreAnalyzed)
        #   round_df=pd.DataFrame(dataframe.iloc[30]["position_df"])
        #   logging.info(dataframe)
        #   logging.info(pd.DataFrame(dataframe.iloc[30]["position_df"]))
# logging.info(position_dataset_df.to_string())
