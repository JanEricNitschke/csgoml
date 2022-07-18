"""DB testing
"""
#!/usr/bin/env python

import boto3
import os
import logging
import json
import pandas as pd
import pymysql
from sqlalchemy import create_engine


logging.basicConfig(
    filename=r"D:\CSGO\ML\CSGOML\logs\db_testing.log",
    encoding="utf-8",
    level=logging.INFO,
    filemode="w",
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# with open(
#     r"D:\CSGO\ML\CSGOML\Analysis\FightAnalyzer.json", encoding="utf-8"
# ) as pre_analyzed:
#     data = json.load(pre_analyzed)
# dataframe = pd.DataFrame(data["Events"])
# mapping_df = pd.DataFrame(data["Mapping"])
# dataframe.to_sql(con=connection, name="Events", index=False, if_exists="replace")
# mapping_df.to_sql(
#    con=connection, name="VictimWeapons", index=False, if_exists="replace"
# )

host = "fightanalyzer.ctox3zthjpph.eu-central-1.rds.amazonaws.com"
user = "IAM_USER"
database = "FightAnalyzerDB"
port = 3306
region = "eu-central-1"
os.environ["LIBMYSQL_ENABLE_CLEARTEXT_PLUGIN"] = "1"
session = boto3.Session()
client = session.client("rds")
token = client.generate_db_auth_token(
    DBHostname=host, Port=port, DBUsername=user, Region=region
)
connection = pymysql.connect(
    host=host,
    user=user,
    password=token,
    database=database,
    ssl_ca=r"D:\\CSGO\\ML\\CSGOML\AWS_Steps\\Certs\\global-bundle.pem",
)
# engine = create_engine(
#     "mysql://{0}:{1}@{2}:{3}/{4}?charset=utf8".format(
#         user, password, host, port, database
#     )
# )

# Pistols = [
#     "CZ75 Auto",
#     "Desert Eagle",
#     "Dual Berettas",
#     "Five-SeveN",
#     "Glock-18",
#     "P2000",
#     "P250",
#     "R8 Revolver",
#     "Tec-9",
#     "USP-S",
# ]
# Heavy = ["MAG-7", "Nova", "Sawed-Off", "XM1014", "M249", "Negev"]
# SMG = ["MAC-10", "MP5-SD", "MP7", "MP9", "P90", "PP-Bizon", "UMP-45"]
# Rifles = [
#     "AK-47",
#     "AUG",
#     "FAMAS",
#     "Galil AR",
#     "M4A1",
#     "M4A4",
#     "SG 553",
#     "AWP",
#     "G3SG1",
#     "SCAR-20",
#     "SSG 08",
# ]
# Grenades = [
#     "Smoke Grenade",
#     "Flashbang",
#     "HE Grenade",
#     "Incendiary Grenade",
#     "Molotov",
#     "Decoy Grenade",
# ]
# Equipment = ["Knife", "Zeus x27"]
# WeaponNames = Pistols + Heavy + SMG + Rifles + Grenades + Equipment
# WeaponClasses = (
#     ["Pistols"] * len(Pistols)
#     + ["Heavy"] * len(Heavy)
#     + ["SMG"] * len(SMG)
#     + ["Rifle"] * len(Rifles)
#     + ["Grenade"] * len(Grenades)
#     + ["Equipment"] * len(Equipment)
# )
# weapon_df = pd.DataFrame(
#     list(zip(WeaponNames, WeaponClasses)), columns=["WeaponName", "Class"]
# )
# logging.info(weapon_df)
# connection = engine.connect()
# weapon_df.to_sql(con=connection, name="WeaponClasses", index=False, if_exists="replace")
# connection.close()
# engine.dispose()
with connection:
    cursor = connection.cursor()
    cursor.execute("show grants for admin;")
    myresult = cursor.fetchall()
    for x in myresult:
        logging.info(x)
    cursor.execute("show grants for IAM_USER;")
    myresult = cursor.fetchall()
    for x in myresult:
        logging.info(x)

    sql = "select user,plugin,host from mysql.user;"
    cursor.execute(sql)
    myresult = cursor.fetchall()
    for x in myresult:
        logging.info(x)

    cursor.execute("SELECT VERSION()")
    version = cursor.fetchone()
    print(f"Database version: {version[0]} ")
    logging.info("Database version: %s ", version[0])
    cursor.execute("Show tables;")
    myresult = cursor.fetchall()
    for x in myresult:
        logging.info(x)

    sql = "DESCRIBE `Events`"
    cursor.execute(sql)
    result = cursor.fetchall()
    for i in result:
        logging.info(i)
    sql = "SELECT COUNT(`EventID`) FROM `Events`"
    cursor.execute(sql)
    # Fetch all the records
    result = cursor.fetchall()
    for i in result:
        logging.info(i)
    sql = "SELECT MAX(`CTWon`) FROM `Events`"
    cursor.execute(sql)
    # Fetch all the records
    result = cursor.fetchall()
    for i in result:
        logging.info(i)
    sql = "SELECT  * FROM `Events` LIMIT 100"
    cursor.execute(sql)
    # Fetch all the records
    result = cursor.fetchall()
    for i in result:
        logging.info(i)

    sql = "DESCRIBE `VictimWeapons`"
    cursor.execute(sql)
    result = cursor.fetchall()
    for i in result:
        logging.info(i)

    sql = "SELECT COUNT(`EventID`) FROM `VictimWeapons`"
    cursor.execute(sql)
    # Fetch all the records
    result = cursor.fetchall()
    for i in result:
        logging.info(i)
    sql = "SELECT * FROM `VictimWeapons` LIMIT 100"
    cursor.execute(sql)

    # Fetch all the records
    result = cursor.fetchall()
    for i in result:
        logging.info(i)

    sql = "DESCRIBE `WeaponClasses`"
    cursor.execute(sql)
    result = cursor.fetchall()
    for i in result:
        logging.info(i)

    sql = "SELECT COUNT(`WeaponName`) FROM `WeaponClasses`"
    cursor.execute(sql)
    # Fetch all the records
    result = cursor.fetchall()
    for i in result:
        logging.info(i)
    sql = "SELECT * FROM `WeaponClasses` LIMIT 100"
    cursor.execute(sql)

    # Fetch all the records
    result = cursor.fetchall()
    for i in result:
        logging.info(i)
