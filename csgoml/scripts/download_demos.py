"""Downloads demos from hltv.org

Typical usage example:

    python download_demos.py --dir "D:\\Downloads\\Demos\\" --startid 68899 --endid 68799
"""
#!/usr/bin/env python

import shutil
import argparse
import time
import sys
import os
import re
import logging
import requests
from requests_ip_rotator import ApiGateway, EXTRA_REGIONS


def find_missing(my_set: set[int]) -> list[int]:
    """Finds missing elements in a sorted list

    Args:
        lst: A sorted list of integers

    Returns:
        A list of all integers that are missing in the original list (between the start and end of it)
    """
    if len(my_set) < 0.5:
        return []
    lst = sorted(my_set)
    logging.info("Demo indices that have already been downloaded:")
    logging.info("Done : %s", lst)
    logging.info("Minimum: %s", lst[0])
    logging.info("Maximum: %s", lst[-1])
    missing = [x for x in range(lst[0], lst[-1] + 1) if x not in my_set]
    logging.info("Missing: %s", missing)
    return missing


def main(args):
    """Downloads demos from hltv.org"""
    parser = argparse.ArgumentParser("Analyze the early mid fight on inferno")
    parser.add_argument(
        "-d", "--debug", action="store_true", default=False, help="Enable debug output."
    )
    parser.add_argument(
        "--dir",
        default="D:\\Downloads\\Demos\\",
        help="Directory that the downloaded files should be saved to",
    )
    parser.add_argument(
        "-l",
        "--log",
        default=r"D:\CSGO\ML\CSGOML\logs\DownloadDemos.log",
        help="Path to output log.",
    )
    parser.add_argument(
        "--startid",
        type=int,
        default=70418,
        help="Analyze demos with a name above this id",
    )
    parser.add_argument(
        "--endid",
        type=int,
        default=70501,
        help="Analyze demos with a name below this id",
    )
    options = parser.parse_args(args)

    # Done are: 68900-70417;

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

    done_indices = set()
    # check already processed demos:
    pro_path = r"E:\PhD\MachineLearning\CSGOData\ParsedDemos"
    for directoryname in os.listdir(pro_path):
        directory = os.path.join(pro_path, directoryname)
        if os.path.isdir(directory):
            for filename in os.listdir(directory):
                if filename.endswith(".json"):
                    search_result = re.search(r".+_(\d{5}).json", filename)
                    if search_result is None:
                        continue
                    match_id = search_result.group(1)
                    if match_id not in done_indices:
                        done_indices.add(int(match_id))

    _ = find_missing(done_indices)

    # gateway = ApiGateway("https://www.hltv.org/", regions=EXTRA_REGIONS)
    # gateway.start()

    session = requests.Session()
    # session.mount("https://www.hltv.org/", gateway)

    urls = [
        "https://www.hltv.org/download/demo/" + str(x)
        for x in range(
            options.startid,
            options.endid,
            1 if (options.endid > options.startid) else -1,
        )
        if x not in done_indices
    ]

    logging.info(urls)
    logging.info("Will download demos for %s matches.", len(urls))
    timeout = 100
    for url in urls:
        time.sleep(5)
        filename = options.dir + url.split("/")[-1] + ".rar"
        logging.info(filename)
        with session.get(url, stream=True, timeout=timeout) as raw:
            # with requests.get(url, stream=True, timeout=timeout) as raw:
            logging.info("Status code: %s", raw.status_code)
            logging.info("Headers: %s", raw.headers)
            # logging.info("Content: %s", raw.content)
            # with open(filename, "wb") as file:
            #     shutil.copyfileobj(raw.raw, file)
            # logging.info("Content: %s", raw.content)
            with open(filename, "wb") as file:
                for chunk in raw.iter_content(chunk_size=1024 * 1024):  # 1024*1024, 128
                    file.write(chunk)

    # Only run this line if you are no longer going to run the script, as it takes longer to boot up again next time.
    # gateway.shutdown()


if __name__ == "__main__":
    main(sys.argv[1:])


# CONNECTIONS=5
# TIMEOUT=5
# out = []

# with concurrent.futures.ThreadPoolExecutor(max_workers=CONNECTIONS) as executor:
#     future_to_url = (executor.submit(request_demo, url, TIMEOUT) for url in urls)
#     for future in concurrent.futures.as_completed(future_to_url):
#         try:
#             data = future.result()
#         except Exception as exc:
#             data = str(type(exc))
#         finally:
#             out.append(data)
#             print(str(len(out)),end="\r")

# def request_demo(url, timeout):
#     logging.info(url)
