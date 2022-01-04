#!/usr/bin/env python

from csgo.parser import DemoParser
import os
import sys
import logging
import shutil
import argparse


def getIDs(filename,mmid):
    ID=filename.split(".")[0]
    logging.debug(("Using ID: "+ID))
    try:    
        NumberID=int(ID)
    except ValueError:
        NumberID=mmid
    return ID, NumberID

def clean_rounds(demo_parser):
    try:
        demo_parser.clean_rounds()
        pass
    except AttributeError:
        logging.error("This demo has an error while cleaning.")
        logging.exception('')

def getMapName(data):
    if data["mapName"].startswith("de_"):
        return data["mapName"].split("_")[1]
    else:
        return data["mapName"]

def MoveJson(source, destination):
    logging.info("Source: "+source)
    logging.info("Destination: "+destination)
    try:
        os.rename(source, destination)
    except FileExistsError:
        os.remove(destination)
        os.rename(source, destination)
    except OSError:
        shutil.move(source, destination)
        logging.info("Moved json to: "+destination)
    return


def main(args):
    parser = argparse.ArgumentParser("Analyze the early mid fight on inferno")
    parser.add_argument("-d", "--debug",  action='store_true', default=False, help="Enable debug output.")
    parser.add_argument('--dirs', nargs='*', default=["D:\CSGO\Demos","C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\csgo\\replays"], help='All the directories that should be scanned for demos.')
    parser.add_argument("-l", "--log",  default='D:\CSGO\ML\CSGOML\Sorter.log', help="Path to output log.")
    parser.add_argument("--startid", type=int, default=1, help="Analyze demos with a name above this id")
    parser.add_argument("--endid", type=int, default=99999, help="Analyze demos with a name below this id")
    parser.add_argument("--mmid", type=int, default=10000, help="Set id value that should be used for mm demos that normally do not have one.")
    parser.add_argument("-m", "--mapsdir", default="D:\CSGO\Demos\Maps", help="Path to directory that contains the folders for the maps that should be included in the analysis.")
    options = parser.parse_args(args)

    if options.debug:
        logging.basicConfig(filename=options.log, encoding='utf-8', level=logging.DEBUG,filemode='w')
    else:
        logging.basicConfig(filename=options.log, encoding='utf-8', level=logging.INFO,filemode='w')

    # Build list of maps that are considered.
    existing_maps=[]
    for dir in os.listdir(options.mapsdir):
        existing_maps.append(dir)
    logging.info("Maps considered: "+", ".join(existing_maps))
    NumberOfDemosAnalyzed=0

    for dir in options.dirs:
        os.chdir(dir)
        logging.debug("Changing directoy now.")
        logging.info("Scanning directory: "+dir)
        for filename in os.listdir(dir):
            if filename.endswith(".dem"):
                f = os.path.join(dir, filename)
                # checking if it is a file
                if os.path.isfile(f):
                    logging.info("At file: "+filename)
                    ID, NumberID = getIDs(filename,options.mmid)
                    if int(NumberID)>=options.startid:
                        if int(NumberID)>options.endid:
                            logging.info("Parsed last relevant demo.")
                            break
                        demo_parser = DemoParser(demofile=f,demo_id=ID,parse_rate=128, buy_style="hltv",dmg_rolled=True,parse_frames=True)
                        data = demo_parser.parse()
                        clean_rounds(demo_parser)
                        data=demo_parser.json
                        MapName=getMapName(data)
                        logging.debug("Scanned map name: "+MapName)
                        if MapName not in existing_maps:
                            logging.error("Map name "+MapName+" does not exist. Not moving json file to maps folder.")
                            continue
                        source=os.path.join(dir,ID+".json")
                        destination=os.path.join(options.mapsdir, MapName,ID+".json")
                        MoveJson(source, destination)
                        NumberOfDemosAnalyzed+=1
    logging.info("Analyzed a total of "+str(NumberOfDemosAnalyzed)+" demos!")

if __name__ == '__main__':
    main(sys.argv[1:])