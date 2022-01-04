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
    # Check if the round was a proper buy and that both the attacker and victim had weapons that would be used for an actual mid fight.
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
                    if (not player["isAlive"]) or player["inventory"]==None:
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
    # Some round cleaning, not neccesarily needed anymore due to the jsons having already been cleaned at creation.
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
                                #printInfo(dict,gameTime,round,map)
                                SummarizeRound(dict,gameTime,round,Results,MatchID,map)

def CalculateCTWinPercentage(df):
        CTWin=(df.WinnerSide == "CT").sum()
        TWin=(df.WinnerSide == "T").sum()
        if (CTWin+TWin)>0:
            return ("CT Win: "+str(round(100*CTWin/(CTWin+TWin)))+"%", "Total Kills: "+str(CTWin+TWin))
        else:
            return "No Kills found!"

def CheckMMStatus(filename, includemm):
    if includemm:
        return True
    else:
        try:
            int(filename[0:3])
            return True
        except ValueError:
            return False
         

def main(args):
    parser = argparse.ArgumentParser("Analyze the early mid fight on inferno")
    parser.add_argument("-d", "--debug",  action='store_true', default=False, help="Enable debug output.")
    parser.add_argument("-a", "--analyze",  action='store_true', default=True, help="Reanalyze demos instead of reading from existing json file.")
    parser.add_argument("-j", "--json",  default="D:\CSGO\Demos\Maps\inferno\Analysis\Inferno_kills_damages_mid.json", help="Path of json containting preanalyzed results.")
    parser.add_argument("-f", "--fastcheck",  action='store_true', default=False,  help="When analyzing demos only do the fast check of looking at the victim teams buy status.")
    parser.add_argument("-n", "--number", type=str, default="", help="Only analyze demos that start with this string.")
    parser.add_argument("--includemm", action='store_true', default=False,  help="Require demos to start with a number (so exclude mm demos.")
    parser.add_argument("--starttime", type=int, default=90, help="Lower end of the clock time range that should be analyzed")
    parser.add_argument("--endtime", type=int, default=110, help="Upper end of the clock time range that should be analyzed")
    parser.add_argument("--dir", type=str, default="D:\CSGO\Demos\Maps\inferno", help="Directoy containting the demos to be analyzed.")
    parser.add_argument("-l", "--log",  default='D:\CSGO\ML\CSGOML\Inferno_Analyzer.log', help="Path to output log.")
    options = parser.parse_args(args)

    if options.debug:
        logging.basicConfig(filename=options.log, encoding='utf-8', level=logging.DEBUG,filemode='w')
    else:
        logging.basicConfig(filename=options.log, encoding='utf-8', level=logging.INFO,filemode='w')

    FastWeaponCheck=options.fastcheck
    dataframe=None
    Results=InitializeResults()
    
    NumberOfDemosAnalyzed=0
    dir=options.dir
    if options.analyze:
        for filename in os.listdir(dir):
            PassesIncludeMM=CheckMMStatus(filename, options.includemm)
            if filename.endswith(".json") and PassesIncludeMM and filename.startswith(options.number):
                logging.info("Working on file "+filename)
                f = os.path.join(dir, filename)
                # checking if it is a file
                if os.path.isfile(f):
                    with open(f, encoding='utf-8') as f:
                        demo_data = json.load(f)
                        map=demo_data["mapName"]
                        AnalyzeMap(demo_data,FastWeaponCheck,Results,map)
                        NumberOfDemosAnalyzed+=1
        logging.info("Analyzed a total of "+str(NumberOfDemosAnalyzed)+" demos!")
        dataframe=pd.DataFrame(Results)
        dataframe.to_json(options.json)
    else:
        with open(options.json, encoding='utf-8') as PreAnalyzed:
            dataframe=pd.read_json(PreAnalyzed)

    logging.info(dataframe)

    RemoveDamage=(dataframe["type"]=="Kill")
    RemoveTramp=(((dataframe["WinnerSide"]=="T") & (dataframe["AttackerArea"]!="TRamp")) | ((dataframe["WinnerSide"]=="CT") & (dataframe["VictimArea"]!="TRamp")))
    StartTime=options.starttime
    EndTime=options.endtime
    gameTimes=np.linspace(StartTime,EndTime,EndTime-StartTime+1)

    logging.info("CTWinPercentages:")

    logging.info("With TRamp forbidden:")
    for Time in gameTimes:
        timeAllowed=(dataframe["Time"]>Time)
        logging.info(str(Time)+":")
        logging.info(CalculateCTWinPercentage(dataframe[timeAllowed & RemoveTramp & RemoveDamage]))

    logging.info("\nWith TRamp allowed:")
    for Time in gameTimes:
        timeAllowed=(dataframe["Time"]>Time)
        logging.info(str(Time)+":")
        logging.info(CalculateCTWinPercentage(dataframe[timeAllowed & RemoveDamage]))

if __name__ == '__main__':
    main(sys.argv[1:])