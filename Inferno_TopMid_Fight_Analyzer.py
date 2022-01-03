#!/usr/bin/env python

from logging import BASIC_FORMAT, NullHandler

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

def getAreaFromPos(map,pos):
    if None in pos:
        logging.debug("No area found for pos:")
        logging.debug(pos)
        return "No area found"
    ClosestArea = find_closest_area(map, pos)
    AreaID=ClosestArea["areaId"]
    if AreaID==None:
        logging.debug("No area found for pos:")
        logging.debug(map)
        logging.debug(pos)
        return "No area found"
    return NAV[map][AreaID]["areaName"]

def checkPosition(dict,map):
    # Make sure that attacker and victim are in one of the desired positions.
    logging.debug("Checking Position")
    AttackerArea=getAreaFromPos(map,[dict["attackerX"],dict["attackerY"],dict["attackerZ"]])
    VictimArea=getAreaFromPos(map,[dict["victimX"],dict["victimY"],dict["victimZ"]])
    logging.debug("AttackerArea: "+str(AttackerArea))
    logging.debug("AttackerSide: "+str(dict["attackerSide"]))
    logging.debug("VictimArea: "+str(VictimArea))
    logging.debug("VictimSide: "+str(dict["victimSide"]))
    if ((AttackerArea=="TopofMid" or AttackerArea=="Middle") and dict["attackerSide"]=="CT"):
    # Filter for victim Position and bottom mid and side of T
        if ((VictimArea=="Middle") and dict["victimSide"]=="T") or VictimArea=="TRamp":
            return True
    if ((VictimArea=="TopofMid" or VictimArea=="Middle") and dict["victimSide"]=="CT"):
                # Filter for victim Position at top mid and side of CT
        if ((AttackerArea=="Middle") and dict["attackerSide"]=="T") or AttackerArea=="TRamp":
            return True
    return False

def getGameTime(dict):
    # Convert the clocktime to seconds.
    logging.debug("Getting game time!")
    TimeList=dict["clockTime"].split(":")
    logging.debug("ClockTime: "+dict["clockTime"])
    try:
        return int(TimeList[0])*60+int(TimeList[1])
    except ValueError:
        return -int(TimeList[1])


def printInfo(dict,gameTime,round,map):
    if "hpDamageTaken" in dict:
        logging.info("Damage event:")
    else:
        logging.info("Kill event:")
    logging.info("Attacker: "+dict["attackerSide"])
    logging.info("Round: "+str(round["endTScore"]+round["endCTScore"]))
    logging.info("Time: "+str(gameTime))
    logging.info("Weapon: "+dict["weapon"])
    logging.info("AttackerArea: "+str(getAreaFromPos(map,[dict["attackerX"],dict["attackerY"],dict["attackerZ"]])))
    logging.info("AttackerSide: "+str(dict["attackerSide"]))
    logging.info("VictimArea: "+str(getAreaFromPos(map,[dict["victimX"],dict["victimY"],dict["victimZ"]])))
    logging.info("VictimSide: "+str(dict["victimSide"]))
    if "hpDamageTaken" in dict:
       logging.info("Damage: "+str(dict["hpDamageTaken"]))
    logging.info("")
      

def checkWeapons(round,dict,FastCheck):
    # Change check weapons to look at inventory of the player for frames before his death
    logging.debug("Checking weapons!")
    # Fast check logic
    logging.debug("Doing fast check with buyType!")
    logging.debug("victimBuy: "+round[dict["victimSide"].lower()+"BuyType"])
    logging.debug("attackerBuy: "+round[dict["attackerSide"].lower()+"BuyType"])
    VictimFullBuy=round[dict["victimSide"].lower()+"BuyType"]=="Full Buy"
    BothHalfBuy=(round[dict["victimSide"].lower()+"BuyType"]=="Half Buy" and round[dict["attackerSide"].lower()+"BuyType"]=="Half Buy")
    FastCheckResult=(VictimFullBuy or BothHalfBuy)
    if FastCheck:
        return FastCheckResult
    logging.debug("Doing slow check with active weapons!")
    weaponslist=[]
    for frame in round["frames"]:
        if frame["seconds"]>dict["seconds"]:
            break
        else:
            for player in frame[dict["victimSide"].lower()]["players"]:
                if player["steamID"]==dict["victimSteamID"]:
                    if not player["isAlive"]:
                        continue
                    for weapon in player["inventory"]:
                        if weapon["weaponClass"]=="Rifle":
                            if weapon["weaponName"] not in weaponslist:
                                weaponslist.append(weapon["weaponName"])
    allowedWeapons=["M4A4","AWP","AK-47","Galil AR","M4A1","SG 553","SSG 08","G3SG1","SCAR-20"] #"FAMAS",
    logging.debug("Allowed weapons: " +" ".join(allowedWeapons))
    logging.debug("Attacker weapon: "+dict["weapon"])
    logging.debug("Victim weapons: "+" ".join(weaponslist))
    if dict["weapon"] not in allowedWeapons:
        return False
    for weapon in weaponslist:
        if weapon in allowedWeapons:
            return (True and FastCheckResult)
    return False

def RoundAllowed(round):
    logging.debug("Checking if round should be analyzed!")
    logging.debug("Round is warmup: "+str(round["isWarmup"]))
    logging.debug("Winning side: "+round["winningSide"])
    logging.debug("At least one side spend money: "+str(round["ctSpend"]>1 or round["tSpend"]>1))
    #return ((round["isWarmup"]==False) and (round["winningSide"]!="") and (round["ctSpend"]>0 or round["tSpend"]>0))
    return (round["ctSpend"]>1 or round["tSpend"]>1)




def SummarizeRound(dict,gameTime,round,Results,MatchID,map):
    # Get relevant information from each event
    Results["Weapon"].append(dict["weapon"])
    Results["Round"].append(round["endTScore"]+round["endCTScore"])
    if "hpDamageTaken" in dict:
        Results["type"].append("Damage")
        Results["DamageTaken"].append(dict["hpDamageTaken"])
    else:
        Results["type"].append("Kill")
        Results["DamageTaken"].append(0)
    Results["WinnerSide"].append(dict["attackerSide"])
    Results["Time"].append(gameTime)
    Results["AttackerArea"].append(getAreaFromPos(map,[dict["attackerX"],dict["attackerY"],dict["attackerZ"]]))
    Results["VictimArea"].append(getAreaFromPos(map,[dict["victimX"],dict["victimY"],dict["victimZ"]]))
    Results["MatchID"].append(MatchID)


def InitializeResults():
    Results={}
    Results["Weapon"]=[]
    Results["Round"]=[]
    Results["type"]=[]
    Results["DamageTaken"]=[]
    Results["WinnerSide"]=[]
    Results["Time"]=[]
    Results["AttackerArea"]=[]
    Results["VictimArea"]=[]
    Results["MatchID"]=[]
    return Results


def AnalyzeMap(data,FastWeaponCheck,Results,map):
    # Loop over rounds and each event in them and check if they fulfill all criteria
    events=["kills","damages"]
    #events=["kills"]
    MatchID=data["matchID"]
    for round in data["gameRounds"]:
        # Throw away warump or reset round.
        # Proper rounds are neither warump but always have a winningSide
        if RoundAllowed(round):
            logging.debug("Round:")
            logging.debug(round)
            #Go through all damage events
            for event in events:
                logging.debug(event+" of that round:")
                logging.debug(round[event])
                if round[event]==None:
                    logging.debug("Round does not have damages recorded")
                    continue
                for dict in round[event]:
                    gameTime=getGameTime(dict)
                    if gameTime>100:
                        if checkPosition(dict,map):
                            if checkWeapons(round,dict,FastWeaponCheck):
                                #printInfo(dict,gameTime,round)
                                SummarizeRound(dict,gameTime,round,Results,MatchID)


Debug=False

if Debug:
    logging.basicConfig(filename='D:\CSGO\ML\CSGOML\Inferno_Analyzer.log', encoding='utf-8', level=logging.DEBUG,filemode='w')
else:
    logging.basicConfig(filename='D:\CSGO\ML\CSGOML\Inferno_Analyzer.log', encoding='utf-8', level=logging.INFO,filemode='w')
FastWeaponCheck=False
dataframe=None
Results=InitializeResults()
NumberOfDemosAnalyzed=0
Number=""
dir="D:\CSGO\Demos\Maps\inferno"
for filename in os.listdir(dir):
    try:
        int(filename[0:3])
        NumberStart=True
    except ValueError:
        NumberStart=False
    if filename.endswith(".json") and NumberStart and filename.startswith(Number):
        logging.info("Working on file "+filename)
        f = os.path.join(dir, filename)
        # checking if it is a file
        if os.path.isfile(f):
            with open(f, encoding='utf-8') as f:
                demo_data = json.load(f)
                AnalyzeMap(demo_data,FastWeaponCheck,Results)
                NumberOfDemosAnalyzed+=1
dataframe=pd.DataFrame(Results)
output_path="D:\CSGO\Demos\Maps\inferno"
dataframe.to_json(output_path+"\Inferno_kills_damages_mid.json")
logging.info(dataframe)
logging.info("Number of demos analyzed: "+str(NumberOfDemosAnalyzed))
# Combine MapResults to total Result
#os.chdir(dir)
#print(os.path.abspath(os.getcwd()))

#demo_parser = DemoParser(demofile=f,demo_id=Number,parse_rate=128, buy_style="csgo",dmg_rolled=True)
#data = demo_parser.parse()
# demo_parser.output_file=Number+".json"
# data = demo_parser._read_json()
# Loop over all Rounds
#AnalyzeMap(data,FastWeaponCheck)



# def UpdateMapResult(MapResult,RoundResults,MatchID):
#     if RoundResults["Empty"]:
#         return
#     MapResult["TKillStamps"].extend([MatchID+"_"+str(Round) for Round in RoundResults["TKillStamps"]])
#     MapResult["CTKillStamps"].extend([MatchID+"_"+str(Round) for Round in RoundResults["CTKillStamps"]])
#     MapResult["TKills"]+=RoundResults["TKills"]
#     MapResult["CTKills"]+=RoundResults["CTKills"]
#     MapResult["TDamage"]+=RoundResults["TDamage"]
#     MapResult["CTDamage"]+=RoundResults["CTDamage"]
#     MapResult["NetDamage"].append(RoundResults["CTDamage"]-RoundResults["TDamage"]) # >0 Means more Dmg done by CT
#     MapResult["NetKills"].append(RoundResults["CTKills"]-RoundResults["TKills"])  # >0 Means more kills done by CT
#     MapResult["Times"].extend(RoundResults["Times"])
#     MapResult["CTKillWeapons"].extend(RoundResults["CTKillWeapons"])
#     MapResult["CTDamageWeapons"].extend(RoundResults["CTDamageWeapons"])
#     MapResult["TKillWeapons"].extend(RoundResults["TKillWeapons"])
#     MapResult["TDamageWeapons"].extend(RoundResults["TDamageWeapons"])
#     MapResult["Rounds"].append(RoundResults["Round"])
#    # MapResult["RoundResults"].append(RoundResults)
#     if len(MapResult["Events"])==0:
#         MapResult["Events"]=RoundResults["Events"]
#     else:
#         for key in MapResult["Events"]:
#             MapResult["Events"][key].extend(RoundResults["Events"][key])


# def UpdateTotalResult(TotalResult,MapResults):
#     TotalResult["TKills"]+=MapResults["TKills"]
#     TotalResult["CTKills"]+=MapResults["CTKills"]
#     TotalResult["TDamage"]+=MapResults["TDamage"]
#     TotalResult["CTDamage"]+=MapResults["CTDamage"]
#     TotalResult["NetDamage"].extend(MapResults["NetDamage"]) # >0 Means more Dmg done by CT
#     TotalResult["NetKills"].extend(MapResults["NetKills"])  # >0 Means more kills done by CT
#     TotalResult["Times"].extend(MapResults["Times"])
#     TotalResult["CTKillStamps"].extend(MapResults["CTKillStamps"])
#     TotalResult["CTKillWeapons"].extend(MapResults["CTKillWeapons"])
#     TotalResult["CTDamageWeapons"].extend(MapResults["CTDamageWeapons"])
#     TotalResult["TKillStamps"].extend(MapResults["TKillStamps"])
#     TotalResult["TKillWeapons"].extend(MapResults["TKillWeapons"])
#     TotalResult["TDamageWeapons"].extend(MapResults["TDamageWeapons"])
#     TotalResult["Rounds"].extend(MapResults["Rounds"])
#     for NetDamage in Map["NetDamage"]:
#         if NetDamage>0:
#             TotalResult["RoundWithCTDamageAdvantage"]+=1
#         elif NetDamage<0:
#             TotalResult["RoundWithTDamageAdvantage"]+=1
#     for NetKills in Map["NetKills"]:
#         if NetKills>0:
#             TotalResult["RoundWithCTKillAdvantage"]+=1
#         elif NetKills<0:
#             TotalResult["RoundWithTKillAdvantage"]+=1
#     #TotalResult["RoundResults"].append(MapResults)
#     if len(TotalResult["Events"])==0:
#         TotalResult["Events"]=MapResults["Events"]
#     else:
#         if len(MapResults["Events"])==0:
#             return
#         for key in TotalResult["Events"]:
#             TotalResult["Events"][key].extend(MapResults["Events"][key])

# def InitializeMapResult():
#     MapResult={}
#     MapResult["TKills"]=0
#     MapResult["CTKills"]=0
#     MapResult["TDamage"]=0
#     MapResult["CTDamage"]=0
#     MapResult["NetDamage"]=[] # >0 Means more Dmg done by CT
#     MapResult["NetKills"]=[]  # >0 Means more kills done by CT
#     MapResult["Times"]=[]
#     MapResult["CTKillWeapons"]=[]
#     MapResult["CTDamageWeapons"]=[]
#     MapResult["TKillWeapons"]=[]
#     MapResult["TDamageWeapons"]=[]
#     MapResult["Rounds"]=[]
#     MapResult["RoundResults"]=[]
#     return MapResult

# def InitializeTotalResult():
#     TotalResult={}
#     TotalResult["RoundWithCTDamageAdvantage"]=0
#     TotalResult["RoundWithTDamageAdvantage"]=0
#     TotalResult["RoundWithCTKillAdvantage"]=0
#     TotalResult["RoundWithTKillAdvantage"]=0
#     TotalResult["TKills"]=0
#     TotalResult["CTKills"]=0
#     TotalResult["TDamage"]=0
#     TotalResult["CTDamage"]=0
#     TotalResult["NetDamage"]=[] # >0 Means more Dmg done by CT
#     TotalResult["NetKills"]=[]  # >0 Means more kills done by CT
#     TotalResult["Times"]=[]
#     TotalResult["CTKillWeapons"]=[]
#     TotalResult["CTDamageWeapons"]=[]
#     TotalResult["TKillWeapons"]=[]
#     TotalResult["TDamageWeapons"]=[]
#     TotalResult["Rounds"]=[]
#     TotalResult["RoundResults"]=[]
#     return TotalResult

# def InitializeRoundResults():
#     RoundResults={}
#     RoundResults["TKills"]=0
#     RoundResults["CTKills"]=0
#     RoundResults["TDamage"]=0
#     RoundResults["CTDamage"]=0
#     RoundResults["Times"]=[]
#     RoundResults["CTKillWeapons"]=[]
#     RoundResults["CTDamageWeapons"]=[]
#     RoundResults["TKillWeapons"]=[]
#     RoundResults["TDamageWeapons"]=[]
#     RoundResults["Events"]=[]
#     RoundResults["Round"]=0
#     RoundResults["Empty"]=True
#     return RoundResults