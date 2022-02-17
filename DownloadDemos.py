#!/usr/bin/env python

import logging
import requests
from requests_ip_rotator import ApiGateway, EXTRA_REGIONS
import shutil
import concurrent.futures
import time
import argparse
import sys
import os
import re

def find_missing(lst):
    return [x for x in range(lst[0], lst[-1]+1) if x not in lst]

def main(args):
    parser = argparse.ArgumentParser("Analyze the early mid fight on inferno")
    parser.add_argument("-d", "--debug",  action='store_true', default=False, help="Enable debug output.")
    parser.add_argument('--dir',  default="D:\Downloads\Demos\\", help='Directory that the downloaded files should be saved to')
    parser.add_argument("-l", "--log",  default='D:\CSGO\ML\CSGOML\DownloadDemos.log', help="Path to output log.")
    parser.add_argument("--startid", type=int, default=69399, help="Analyze demos with a name above this id")
    parser.add_argument("--endid", type=int, default=69299, help="Analyze demos with a name below this id")
    options = parser.parse_args(args)

    # Done are: 69400-69899;

    if options.debug:
        logging.basicConfig(filename=options.log, encoding='utf-8', level=logging.DEBUG,filemode='w',format='%(asctime)s %(levelname)-8s %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
    else:
        logging.basicConfig(filename=options.log, encoding='utf-8', level=logging.INFO,filemode='w',format='%(asctime)s %(levelname)-8s %(message)s',datefmt='%Y-%m-%d %H:%M:%S')

    DoneIndices=[]
    # check already processed demos:
    ProPath="E:\PhD\MachineLearning\CSGOData\ParsedDemos"
    for directoryname in os.listdir(ProPath):
        directory=os.path.join(ProPath,directoryname)
        if os.path.isdir(directory):
            for filename in os.listdir(directory):
                id=re.search(".+_(\d{5}).json",filename).group(1)
                if id not in DoneIndices:
                    DoneIndices.append(int(id))
    DoneIndices.sort()
    logging.info("Demo indices that have already been downloaded:")
    logging.info("Minimum: {}".format(DoneIndices[0]))
    logging.info("Maximum: {}".format(DoneIndices[-1]))
    logging.info("Missing: {}".format(find_missing(DoneIndices)))

    gateway = ApiGateway("https://www.hltv.org/", regions=EXTRA_REGIONS)
    gateway.start()

    session = requests.Session()
    session.mount("https://www.hltv.org/", gateway)
    urls=["https://www.hltv.org/download/demo/"+str(x) for x in range(options.startid,options.endid,1 if (options.endid > options.startid) else -1) if x not in DoneIndices]

    logging.info(urls)
    logging.info("Will download demos for {} match.".format(len(urls)))
    timeout=5
    for url in urls:
        Filename=options.dir+url.split("/")[-1]+".rar"
        logging.info(Filename)
        with session.get(url, stream=True, timeout=timeout) as raw:
            with open(Filename, "wb") as file:
                shutil.copyfileobj(raw.raw, file)

    # Only run this line if you are no longer going to run the script, as it takes longer to boot up again next time.
    gateway.shutdown()



if __name__ == '__main__':
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