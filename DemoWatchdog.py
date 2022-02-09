#!/usr/bin/env python

from ast import Interactive
import logging
import shutil
import argparse
import time
import sys
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
import os
import patoolib
import subprocess
from io import StringIO

def wait_till_file_is_created(source_path):
    historicalSize = -1
    while (historicalSize != os.path.getsize(source_path)):
        historicalSize = os.path.getsize(source_path)
        time.sleep(1)

def log_subprocess_output(pipe):
    #output=pipe.decode("ascii")
    for line in pipe.split(b'\n'): # b'\n'-separated lines
        logging.info('got line from subprocess: %r', line)

def on_created(event):
    logging.info(f"{event.src_path} has been created!")
 
def on_deleted(event):
    logging.info(f"{event.src_path} has been deleted!")
 
def on_modified(event):
    logging.info(f"{event.src_path} has been modified!")
    wait_till_file_is_created(event.src_path)
    path=event.src_path[:-4]+"\\"
    logging.info(path)
    if not os.path.exists(path):
        os.makedirs(path)
        ending="_"+os.path.basename(os.path.normpath(path))
        patoolib.extract_archive(event.src_path, outdir=path, interactive=False)
        os.remove(event.src_path)
        cmd = 'python d:/CSGO/ML/CSGOML/DemoAnalyzer_Sorter.py -l None --dirs {} -m "D:\CSGO\Demos\Pro\Maps" --jsonending {}'.format(path, ending)
        logging.info(cmd)
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
        out, _ = p.communicate() 
        log_subprocess_output(out)
        if path.startswith("D:\Downloads\\"):
            shutil.rmtree(path)

 
def on_moved(event):
    logging.info(f"{event.src_path} has been moved!")

def main(args):
    parser = argparse.ArgumentParser("Analyze the early mid fight on inferno")
    parser.add_argument("-d", "--debug",  action='store_true', default=False, help="Enable debug output.")
    parser.add_argument('--dir', nargs='*', default="D:\Downloads\Demos", help='All the directories that should be scanned for demos.')
    parser.add_argument("-l", "--log",  default='D:\CSGO\ML\CSGOML\DemoWatchdog.log', help="Path to output log.")
    options = parser.parse_args(args)

    if options.debug:
        logging.basicConfig(filename=options.log, encoding='utf-8', level=logging.DEBUG,filemode='w',format='%(asctime)s %(levelname)-8s %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
    else:
        logging.basicConfig(filename=options.log, encoding='utf-8', level=logging.INFO,filemode='w',format='%(asctime)s %(levelname)-8s %(message)s',datefmt='%Y-%m-%d %H:%M:%S')


    patterns = ["*.rar"]
    ignore_patterns = None
    ignore_directories = True
    case_sensitive = False
    my_event_handler = PatternMatchingEventHandler(patterns, ignore_patterns, ignore_directories, case_sensitive)

    my_event_handler.on_created = on_created
    my_event_handler.on_deleted = on_deleted
    my_event_handler.on_modified = on_modified
    my_event_handler.on_moved = on_moved

    path = options.dir
    go_recursively = False
    my_observer = Observer()
    my_observer.schedule(my_event_handler, path, recursive=go_recursively)

    my_observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        my_observer.stop()
        my_observer.join()


    # Have watchdog check for new rar files
    # If it finds one it shoud unpack it and run DemoAnalyzer_Sorter.py over the resulting folder and have each json names as DemoName_RarNumber.json
    # Then move the json to a destination folder



if __name__ == '__main__':
    main(sys.argv[1:])