import logging
import requests
from requests_ip_rotator import ApiGateway, EXTRA_REGIONS
import shutil
import concurrent.futures
import time

logging.basicConfig(filename='D:\CSGO\ML\CSGOML\DownloadDemos.log', encoding='utf-8', level=logging.DEBUG,filemode='w')


gateway = ApiGateway("https://www.hltv.org/")
gateway.start()

session = requests.Session()
session.mount("https://www.hltv.org/", gateway)
urls=["https://www.hltv.org/download/demo/"+str(x) for x in range(69877,69882)]


logging.info(urls)
timeout=5
# def request_demo(url, timeout):
#     logging.info(url)
for url in urls:
    Filename="D:\Downloads\\"+url.split("/")[-1]+".rar"
    logging.info(Filename)
    with session.get(url, stream=True, timeout=timeout) as raw:
        with open(Filename, "wb") as file:
            shutil.copyfileobj(raw.raw, file)

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
# Only run this line if you are no longer going to run the script, as it takes longer to boot up again next time.
gateway.shutdown() 