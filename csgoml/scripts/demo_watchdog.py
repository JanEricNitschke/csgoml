"""Watch directory for new demo files to unpack and parse.

    Have watchdog check for new rar files
    If it finds one it shoud unpack it and run DemoAnalyzer_Sorter.py over the resulting folder and have each json names as DemoName_RarNumber.json
    Then move the json to a destination folder

    Typical usage example:

    python demo_watchdog.py --dir "D:\\Downloads\\Demos"
"""
#!/usr/bin/env python

import os
import logging
import shutil
import time
import sys
import argparse
import patoolib
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
from csgoml.preparation import demo_analyzer_sorter


def wait_till_file_is_created(source_path: str) -> None:
    """Waits until creation of a file is finished.

    Keeps checking a files size every second. Only if it has not changed during that time this function returns.

    Args:
        source_path (str): A string of the patch of the file to be monitored for finishing of creation.

    Returns:
        None (when creation has finished)
    """
    historical_size = -1
    while historical_size != os.path.getsize(source_path):
        historical_size = os.path.getsize(source_path)
        time.sleep(1)


def log_subprocess_output(pipe):
    """Logs output received from subprocess"""
    # output=pipe.decode("ascii")
    for line in pipe.split(b"\n"):  # b'\n'-separated lines
        logging.info("got line from subprocess: %r", line)


def on_created(event):
    """Function triggered when a new file is created.

    Logs the path of the file that has been created.
    """
    logging.info("%s has been created!", event.src_path)


def on_deleted(event):
    """Function triggered when a file is deleted.

    Logs the path of the file that has been deleted.
    """
    logging.info("%s has been deleted!", event.src_path)


def on_modified(event):
    """Function triggered when a file is modified.

    Logs the path of the file that has been modified.
    Extracts the demos from the .rar file and then calls demo_analyzer_sorter on the created directory containing the demos.
    """
    logging.info("%s has been modified!", event.src_path)
    if not os.path.isfile(event.src_path):
        return
    wait_till_file_is_created(event.src_path)
    path = event.src_path[:-4] + "\\"
    if not os.path.exists(path):
        os.makedirs(path)
        ending = "_" + os.path.basename(os.path.normpath(path))
        try:
            patoolib.extract_archive(event.src_path, outdir=path, interactive=False)
        except Exception as e:
            logging.info(
                "Could not extract from archive due to exception %s. Skipping!", e
            )
            return
        os.remove(event.src_path)
        analyzer = demo_analyzer_sorter.DemoAnalyzerSorter(
            indentation=True,
            dirs=[path],
            maps_dir=r"E:\PhD\MachineLearning\CSGOData\ParsedDemos",
            json_ending=ending,
        )
        logging.info(
            "Calling analyzer_sorter: analyzer = demo_analyzer_sorter.DemoAnalyzerSorter(indentation=True,dirs=[%s],log=None,maps_dir='E:\\PhD\\MachineLearning\\CSGOData\\ParsedDemos',json_ending=%s)",
            path,
            ending,
        )
        analyzer.parse_demos()
        if path.startswith("D:\\Downloads\\") and path != "D:\\Downloads\\":
            shutil.rmtree(path)


def on_moved(event):
    """Function triggered when a file is moved.

    Logs the path of the file that has been moved.
    """
    logging.info("%s has been moved!", event.src_path)


def main(args):
    """Downloads demos from hltv.org"""
    parser = argparse.ArgumentParser("Analyze the early mid fight on inferno")
    parser.add_argument(
        "-d", "--debug", action="store_true", default=False, help="Enable debug output."
    )
    parser.add_argument(
        "--dir", default=r"D:\Downloads\Demos", help="Directory to monitor for changes"
    )
    parser.add_argument(
        "-l",
        "--log",
        default=r"D:\CSGO\ML\CSGOML\logs\DemoWatchdog.log",
        help="Path to output log.",
    )
    options = parser.parse_args(args)

    if options.debug:
        logging.basicConfig(
            filename=options.log,
            encoding="utf-8",
            level=logging.DEBUG,
            filemode="w",
            format="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        logging.basicConfig(
            filename=options.log,
            encoding="utf-8",
            level=logging.INFO,
            filemode="w",
            format="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    patterns = ["*.rar"]
    ignore_patterns = None
    ignore_directories = True
    case_sensitive = False
    my_event_handler = PatternMatchingEventHandler(
        patterns, ignore_patterns, ignore_directories, case_sensitive
    )

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


if __name__ == "__main__":
    main(sys.argv[1:])
