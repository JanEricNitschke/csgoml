from logging import BASIC_FORMAT, NullHandler
from csgo.parser import DemoParser
import os
import sys
import json
import logging

def checkPosition(dict):
    logging.debug("Checking Position")
    logging.debug("AttackerArea: "+str(dict["attackerAreaName"]))
    logging.debug("AttackerSide: "+str(dict["attackerSide"]))
    logging.debug("VictimArea: "+str(dict["victimAreaName"]))
    logging.debug("VictimSide: "+str(dict["victimSide"]))
    if ((dict["attackerAreaName"]=="TopofMid" or dict["attackerAreaName"]=="Middle") and dict["attackerSide"]=="CT"):
    # Filter for victim Position and bottom mid and side of T
        if ((dict["victimAreaName"]=="Middle") and dict["victimSide"]=="T"): # or dict["victimAreaName"]=="TRamp"
            return True
    if ((dict["victimAreaName"]=="TopofMid" or dict["victimAreaName"]=="Middle") and dict["victimSide"]=="CT"):
                # Filter for victim Position at top mid and side of CT
        if ((dict["attackerAreaName"]=="Middle") and dict["attackerSide"]=="T"): #dict["attackerAreaName"]=="TRamp" or 
            return True
    return False

def getGameTime(dict):
    logging.debug("Getting game time!")
    TimeList=dict["clockTime"].split(":")
    logging.debug("ClockTime: "+dict["clockTime"])
    try:
        return int(TimeList[0])*60+int(TimeList[1])
    except ValueError:
        return -int(TimeList[1])
    


def printInfo(dict,gameTime,round):
    print("Attacker: "+dict["attackerSide"])
    print("Round: "+str(round["endTScore"]+round["endCTScore"]))
    print("Time: "+str(gameTime))
    print("Weapon: "+dict["weapon"])
    if "hpDamageTaken" in dict:
        print("Damage: "+str(dict["hpDamageTaken"]))
      

def checkWeapons(round,dict,FastCheck):
    # Change check weapons to look at inventory of the player for frames before his death
    logging.debug("Checking weapons!")
    if FastCheck:
        logging.debug("Doing fast check with buyType!")
        logging.debug("victimBuy: "+round[dict["victimSide"].lower()+"BuyType"])
        logging.debug("attackerBuy: "+round[dict["attackerSide"].lower()+"BuyType"])
        VictimFullBuy=round[dict["victimSide"].lower()+"BuyType"]=="Full Buy"
        BothHalfBuy=(round[dict["victimSide"].lower()+"BuyType"]=="Half Buy" and round[dict["attackerSide"].lower()+"BuyType"]=="Half Buy")
        return (VictimFullBuy or BothHalfBuy)
    logging.debug("Doing slow check with active weapons!")
    weaponslist=[]
    for frame in round["frames"]:
        if frame["seconds"]>dict["seconds"]:
            break
        else:
            for player in frame[dict["victimSide"].lower()]["players"]:
                if player["steamID"]==dict["victimSteamID"]:
                    for weapon in player["inventory"]:
                        if weapon["weaponClass"]=="Rifle":
                            if weapon["weaponName"] not in weaponslist:
                                weaponslist.append(weapon["weaponName"])
    allowedWeapons=["M4A4","AWP","AK-47","M4A1","FAMAS","Galil AR","SG 553","SSG 08","G3SG1","SCAR-20"]
    logging.debug("Allowed weapons: " +" ".join(allowedWeapons))
    logging.debug("Attacker weapon: "+dict["weapon"])
    logging.debug("Victim weapons: "+" ".join(weaponslist))
    if dict["weapon"] not in allowedWeapons:
        return False
    for weapon in weaponslist:
        if weapon in allowedWeapons:
            return True
    return False

def RoundAllowed(round):
    logging.debug("Checking if round should be analyzed!")
    logging.debug("Round is warmup: "+str(round["isWarmup"]))
    logging.debug("Winning side: "+round["winningSide"])
    logging.debug("At least one side spend money: "+str(round["ctSpend"]>1 or round["tSpend"]>1))
    #return ((round["isWarmup"]==False) and (round["winningSide"]!="") and (round["ctSpend"]>0 or round["tSpend"]>0))
    return (round["ctSpend"]>1 or round["tSpend"]>1)

def InitializeRoundResults():
    RoundResults={}
    RoundResults["TKills"]=0
    RoundResults["CTKills"]=0
    RoundResults["TDamage"]=0
    RoundResults["CTDamage"]=0
    RoundResults["Times"]=[]
    RoundResults["CTKillWeapons"]=[]
    RoundResults["CTDamageWeapons"]=[]
    RoundResults["TKillWeapons"]=[]
    RoundResults["TDamageWeapons"]=[]
    RoundResults["Events"]=[]
    RoundResults["Round"]=0
    RoundResults["Empty"]=True
    return RoundResults

def InitializeMapResult():
    MapResult={}
    MapResult["TKills"]=0
    MapResult["CTKills"]=0
    MapResult["TDamage"]=0
    MapResult["CTDamage"]=0
    MapResult["NetDamage"]=[] # >0 Means more Dmg done by CT
    MapResult["NetKills"]=[]  # >0 Means more kills done by CT
    MapResult["Times"]=[]
    MapResult["CTKillWeapons"]=[]
    MapResult["CTDamageWeapons"]=[]
    MapResult["TKillWeapons"]=[]
    MapResult["TDamageWeapons"]=[]
    MapResult["Rounds"]=[]
    MapResult["RoundResults"]=[]
    return MapResult

def InitializeTotalResult():
    TotalResult={}
    TotalResult["RoundWithCTDamageAdvantage"]=0
    TotalResult["RoundWithTDamageAdvantage"]=0
    TotalResult["RoundWithCTKillAdvantage"]=0
    TotalResult["RoundWithTKillAdvantage"]=0
    TotalResult["TKills"]=0
    TotalResult["CTKills"]=0
    TotalResult["TDamage"]=0
    TotalResult["CTDamage"]=0
    TotalResult["NetDamage"]=[] # >0 Means more Dmg done by CT
    TotalResult["NetKills"]=[]  # >0 Means more kills done by CT
    TotalResult["Times"]=[]
    TotalResult["CTKillWeapons"]=[]
    TotalResult["CTDamageWeapons"]=[]
    TotalResult["TKillWeapons"]=[]
    TotalResult["TDamageWeapons"]=[]
    TotalResult["Rounds"]=[]
    TotalResult["RoundResults"]=[]
    return TotalResult

def SummarizeRound(dict,gameTime,round,RoundResults):
    EventResult={}
    if "hpDamageTaken" in dict:
        EventResult["type"]="Damage"
        EventResult["DamageTaken"]=dict["hpDamageTaken"]
    else:
        EventResult["type"]="Kill"
    EventResult["WinnerSide"]=dict["attackerSide"]
    if EventResult["type"]=="Kill":
        if EventResult["WinnerSide"]=="CT":
            RoundResults["CTKillWeapons"].append(dict["weapon"])
            RoundResults["CTKills"]+=1
        elif EventResult["WinnerSide"]=="T":
            RoundResults["TKillWeapons"].append(dict["weapon"])
            RoundResults["TKills"]+=1
    elif EventResult["type"]=="Damage":
        if EventResult["WinnerSide"]=="CT":
            RoundResults["CTDamageWeapons"].append(dict["weapon"])
            RoundResults["CTDamage"]+=EventResult["DamageTaken"]
        elif EventResult["WinnerSide"]=="T":
            RoundResults["TDamageWeapons"].append(dict["weapon"])
            RoundResults["TDamage"]+=EventResult["DamageTaken"]
    RoundResults["Round"]=round["endTScore"]+round["endCTScore"]
    EventResult["Time"]=gameTime
    RoundResults["Times"].append(gameTime)
    #RoundResults["Events"].append(EventResult)
    RoundResults["Empty"]=False

def UpdateMapResult(MapResult,RoundResults):
    if RoundResults["Empty"]:
        return
    MapResult["TKills"]+=RoundResults["TKills"]
    MapResult["CTKills"]+=RoundResults["CTKills"]
    MapResult["TDamage"]+=RoundResults["TDamage"]
    MapResult["CTDamage"]+=RoundResults["CTDamage"]
    MapResult["NetDamage"].append(RoundResults["CTDamage"]-RoundResults["TDamage"]) # >0 Means more Dmg done by CT
    MapResult["NetKills"].append(RoundResults["CTKills"]-RoundResults["TKills"])  # >0 Means more kills done by CT
    MapResult["Times"].extend(RoundResults["Times"])
    MapResult["CTKillWeapons"].extend(RoundResults["CTKillWeapons"])
    MapResult["CTDamageWeapons"].extend(RoundResults["CTDamageWeapons"])
    MapResult["TKillWeapons"].extend(RoundResults["TKillWeapons"])
    MapResult["TDamageWeapons"].extend(RoundResults["TDamageWeapons"])
    MapResult["Rounds"].append(RoundResults["Round"])
    #MapResult["RoundResults"].append(RoundResults)

def UpdateTotalResult(TotalResult,MapResults):
    TotalResult["TKills"]+=MapResults["TKills"]
    TotalResult["CTKills"]+=MapResults["CTKills"]
    TotalResult["TDamage"]+=MapResults["TDamage"]
    TotalResult["CTDamage"]+=MapResults["CTDamage"]
    TotalResult["NetDamage"].extend(MapResults["NetDamage"]) # >0 Means more Dmg done by CT
    TotalResult["NetKills"].extend(MapResults["NetKills"])  # >0 Means more kills done by CT
    TotalResult["Times"].extend(MapResults["Times"])
    TotalResult["CTKillWeapons"].extend(MapResults["CTKillWeapons"])
    TotalResult["CTDamageWeapons"].extend(MapResults["CTDamageWeapons"])
    TotalResult["TKillWeapons"].extend(MapResults["TKillWeapons"])
    TotalResult["TDamageWeapons"].extend(MapResults["TDamageWeapons"])
    TotalResult["Rounds"].extend(MapResults["Rounds"])
    for NetDamage in Map["NetDamage"]:
        if NetDamage>0:
            TotalResult["RoundWithCTDamageAdvantage"]+=1
        elif NetDamage<0:
            TotalResult["RoundWithTDamageAdvantage"]+=1
    for NetKills in Map["NetKills"]:
        if NetKills>0:
            TotalResult["RoundWithCTKillAdvantage"]+=1
        elif NetKills<0:
            TotalResult["RoundWithTKillAdvantage"]+=1
    #TotalResult["RoundResults"].append(MapResults)


def AnalyzeMap(data,FastWeaponCheck):
    MapResult=InitializeMapResult()
    events=["kills","damages"]
    for round in data["gameRounds"]:
        # Throw away warump or reset round.
        # Proper rounds are neither warump but always have a winningSide
        if RoundAllowed(round):
            logging.debug("Round:")
            logging.debug(round)
            RoundResults=InitializeRoundResults()
            #Go through all damage events
            for event in events:
                logging.debug(event+" of that round:")
                logging.debug(round[event])
                if round[event]==None:
                    logging.debug("Round does not have damages recorded")
                    continue
                for dict in round[event]:
                    if checkPosition(dict): 
                        gameTime=getGameTime(dict)
                        if gameTime>100:
                            if checkWeapons(round,dict,FastWeaponCheck):
                                #printInfo(dict,gameTime,round)
                                SummarizeRound(dict,gameTime,round,RoundResults)
            UpdateMapResult(MapResult,RoundResults)
    #Combine Round Results to MapResult
    return MapResult


Debug=False

if Debug:
    logging.basicConfig(encoding='utf-8', level=logging.DEBUG)
else:
    logging.basicConfig(encoding='utf-8', level=logging.INFO)
FastWeaponCheck=True
TotalResults=InitializeTotalResult()
MapResults=[]
NumberOfDemosAnalyzed=0
#Number="669"
dir="D:\CSGO\Demos\Maps\inferno"
for filename in os.listdir(dir):
    try:
        int(filename[0:3])
        NumberStart=True
    except ValueError:
        NumberStart=False
    if filename.endswith(".json") and NumberStart: #and filename.startswith(Number):
        logging.info("Working on file "+filename)
        f = os.path.join(dir, filename)
        # checking if it is a file
        if os.path.isfile(f):
            with open(f, encoding='utf-8') as f:
                demo_data = json.load(f)
                MapResults.append(AnalyzeMap(demo_data,FastWeaponCheck))
                NumberOfDemosAnalyzed+=1
for Map in MapResults:
    UpdateTotalResult(TotalResults,Map)
print(TotalResults)
# Combine MapResults to total Result
#os.chdir(dir)
#print(os.path.abspath(os.getcwd()))

#demo_parser = DemoParser(demofile=f,demo_id=Number,parse_rate=128, buy_style="csgo",dmg_rolled=True)
#data = demo_parser.parse()
# demo_parser.output_file=Number+".json"
# data = demo_parser._read_json()
# Loop over all Rounds
#AnalyzeMap(data,FastWeaponCheck)



                        