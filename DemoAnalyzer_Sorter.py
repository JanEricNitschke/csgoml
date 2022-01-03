from csgo.parser import DemoParser
import os
import sys
import logging
import shutil


Debug=True

if Debug:
    logging.basicConfig(filename='D:\CSGO\ML\CSGOML\Sorter.log', encoding='utf-8', level=logging.DEBUG,filemode='w')
else:
    logging.basicConfig(filename='D:\CSGO\ML\CSGOML\Sorter.log', encoding='utf-8', level=logging.INFO,filemode='w')



dirs=["D:\CSGO\Demos","C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\csgo\\replays"]
#dir="C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\csgo\\replays"
StartID=1
EndID=99999

existing_maps=["vertigo","ancient","cache","cbble","dust2","inferno","mirage","nuke","overpass","train","santorini_playtest","santorini"]
logging.info("Maps considered: "+", ".join(existing_maps))



#print(os.path.abspath(os.getcwd()))
for dir in dirs:
    os.chdir(dir)
    logging.debug("Changing directoy now.")
    logging.info("Scanning directory: "+dir)
    logging.info("Looping over demos in directory.")
    for filename in os.listdir(dir):
        if filename.endswith(".dem"):
            f = os.path.join(dir, filename)
            # checking if it is a file
            if os.path.isfile(f):
                logging.info("At file: "+filename)
                try:
                    int(filename[0:3])
                    ID=filename[0:3]
                    NumberID=int(ID)
                except ValueError:
                    ID=filename.split(".")[0]
                    NumberID=10000
                logging.debug(("Using ID: "+ID))
                if int(NumberID)>=StartID: #True and int(ID)<=EndID:
                    if int(NumberID)>EndID:
                        logging.info("Parsed last relevant demo.")
                        break
                    demo_parser = DemoParser(demofile=f,demo_id=ID,parse_rate=128, buy_style="hltv",dmg_rolled=True,parse_frames=True)
                    data = demo_parser.parse()
                    try:
                        demo_parser.clean_rounds()
                        pass
                    except AttributeError:
                        logging.info("This demo has an error while cleaning.")
                        logging.exception('')
                        pass
                    data=demo_parser.json
                    if data["mapName"].startswith("de_"):
                        MapName=data["mapName"].split("_")[1]
                    else:
                        MapName=data["mapName"]
                    logging.debug("Scanned map name: "+MapName)
                    if MapName not in existing_maps:
                        logging.info("Map name "+MapName+" does not exist. Not moving json file to maps folder.")
                        continue
                    source=os.path.join(dir,ID+".json")
                    destination="D:\CSGO\Demos\Maps\\" + MapName +"\\" + os.path.basename(f).split(".")[0]+".json"
                    logging.debug("Source: "+source)
                    logging.debug("Desitnation: "+destination)
                    try:
                        os.rename(source, destination)
                    except FileExistsError:
                        os.remove(destination)
                        os.rename(source, destination)
                    except OSError:
                        shutil.move(source, destination)
                    logging.info("Moved json to: "+destination)