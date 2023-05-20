#!/usr/bin/env python
r"""Downloads demos from hltv.org.

Example::

    python download_demos.py --dir "D:\\Downloads\\Demos" --startid 68899 --endid 68799
"""


import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path

import requests

from csgoml.helpers import setup_logging


def find_missing(my_set: set[int]) -> list[int]:
    """Finds missing elements in a set.

    Args:
        my_set: A set of integers

    Returns:
        A list of all integers that are missing in the original list.
        (between the start and end of it)
    """
    if not my_set:
        return []
    lst = sorted(my_set)
    logging.info("Demo indices that have already been downloaded:")
    logging.info("Done : %s", lst)
    logging.info("Minimum: %s", lst[0])
    logging.info("Maximum: %s", lst[-1])
    missing = [x for x in range(lst[0], lst[-1] + 1) if x not in my_set]
    logging.info("Missing: %s", missing)
    return missing


def _get_done_indices(pro_path: str) -> set[int]:
    """Get the indices of all parsed demos in a directory.

    Args:
        pro_path (str): Path to the directory to check

    Returns:
        set[int]: Set of all already parsed demo indices.
    """
    done_indices: set[int] = set()
    # check already processed demos:
    for directoryname in Path(pro_path).iterdir():
        if os.path.isdir(directoryname):
            for filename in directoryname.iterdir():
                if filename.suffix == ".json":
                    search_result = re.search(r".+_(\d{5}).json", filename.name)
                    if search_result is None:
                        continue
                    match_id = search_result[1]
                    if match_id not in done_indices:
                        done_indices.add(int(match_id))
    return done_indices


def main(args: list[str]) -> None:
    """Downloads demos from hltv.org."""
    parser = argparse.ArgumentParser("Analyze the early mid fight on inferno")
    parser.add_argument(
        "-d", "--debug", action="store_true", default=False, help="Enable debug output."
    )
    parser.add_argument(
        "-l",
        "--log",
        default=r"D:\CSGO\ML\CSGOML\logs\DownloadDemos.log",
        help="Path to output log.",
    )
    parser.add_argument(
        "--dir",
        default="D:\\Downloads\\Demos\\",
        help="Directory that the downloaded files should be saved to",
    )
    parser.add_argument(
        "--startid",
        type=int,
        default=70501,
        help="Analyze demos with a name above this id",
    )
    parser.add_argument(
        "--endid",
        type=int,
        default=70601,
        help="Analyze demos with a name below this id",
    )
    options = parser.parse_args(args)

    # Done are: 68900-70500;

    setup_logging(options)

    # check already processed demos:
    pro_path = r"E:\PhD\MachineLearning\CSGOData\ParsedDemos"

    done_indices = _get_done_indices(pro_path=pro_path)

    find_missing(done_indices)

    session = requests.Session()

    urls = [
        f"https://www.hltv.org/download/demo/{x!s}"
        for x in range(
            options.startid,
            options.endid,
            1 if (options.endid > options.startid) else -1,
        )
        if x not in done_indices
    ]

    logging.info(urls)
    logging.info("Will download demos for %s matches.", len(urls))
    timeout = 10
    for url in urls:
        filename = options.dir + url.split("/")[-1] + ".rar"
        logging.info(filename)
        try:
            with session.get(url, stream=True, timeout=timeout) as raw:
                # with requests.get(url, stream=True, timeout=timeout) as raw:
                logging.info("Status code: %s", raw.status_code)
                logging.info("Headers: %s", raw.headers)
                with open(filename, "wb") as file:
                    for chunk in raw.iter_content(
                        chunk_size=1024 * 1024
                    ):  # 1024*1024, 128
                        file.write(chunk)
            time.sleep(5)
        except requests.exceptions.ConnectionError:
            logging.exception("Got time out")


if __name__ == "__main__":
    main(sys.argv[1:])
