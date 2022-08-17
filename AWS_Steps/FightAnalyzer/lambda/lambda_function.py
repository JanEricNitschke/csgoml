import sys
import logging
import rds_config
import pymysql
import json
import math

# rds settings
rds_host = "fightanalyzer.ctox3zthjpph.eu-central-1.rds.amazonaws.com"
name = rds_config.db_username
password = rds_config.db_password
db_name = rds_config.db_name

logger = logging.getLogger()
logger.setLevel(logging.INFO)
try:
    conn = pymysql.connect(
        host=rds_host, user=name, passwd=password, db=db_name, connect_timeout=5
    )
except pymysql.MySQLError as e:
    logger.error("ERROR: Unexpected error: Could not connect to MySQL instance.")
    logger.error(e)

logger.info("SUCCESS: Connection to RDS MySQL instance succeeded")


def get_wilson_interval(success_percent, total_n, z):
    success_percent = float(success_percent)
    lower = (
        success_percent
        + z * z / (2 * total_n)
        - z
        * math.sqrt(
            (success_percent * (1 - success_percent) + z * z / (4 * total_n)) / total_n
        )
    ) / (1 + z * z / total_n)
    upper = (
        success_percent
        + z * z / (2 * total_n)
        + z
        * math.sqrt(
            (success_percent * (1 - success_percent) + z * z / (4 * total_n)) / total_n
        )
    ) / (1 + z * z / total_n)
    return [lower, success_percent, upper]


def lambda_handler(event, context):
    """
    This function fetches content from MySQL RDS instance
    """
    try:
        sql = (
            f"""SELECT AVG(t.CTWon), COUNT(t.CTWon) """
            f"""FROM ( """
            f"""SELECT DISTINCT e.EventID, e.CTWon """
            f"""FROM Events e JOIN CTWeapons ctw """
            f"""ON e.EventID = ctw.EventID """
            f"""JOIN TWeapons tw """
            f"""ON e.EventID = tw.EventID """
            f"""JOIN WeaponClasses wcct """
            f"""ON ctw.CTWeapon = wcct.WeaponName """
            f"""JOIN WeaponClasses wct """
            f"""ON tw.TWeapon = wct.WeaponName """
            f"""JOIN WeaponClasses wck """
            f"""ON e.KillWeapon = wck.WeaponName """
            f"""WHERE e.MapName = %s """
            f"""AND e.Time BETWEEN %s AND %s """
        )
        param = [event["map_name"], event["times"]["start"], event["times"]["end"]]

        if event["positions"]["CT"]:
            sql += f"""AND e.CTArea in %s """
            param.append(event["positions"]["CT"])
        if event["positions"]["T"]:
            sql += f"""AND e.TArea in %s """
            param.append(event["positions"]["T"])
        if event["use_weapons_classes"]["CT"] == "weapons":
            if event["weapons"]["CT"]["Allowed"]:
                sql += f"""AND ctw.CTWeapon in %s """
                param.append(event["weapons"]["CT"]["Allowed"])
            if event["weapons"]["CT"]["Forbidden"]:
                sql += f"""AND ctw.CTWeapon NOT in %s """
                param.append(event["weapons"]["CT"]["Forbidden"])
        elif event["use_weapons_classes"]["CT"] == "classes":
            if event["classes"]["CT"]["Allowed"]:
                sql += f"""AND wcct.Class in %s """
                param.append(event["classes"]["CT"]["Allowed"])
            if event["classes"]["CT"]["Forbidden"]:
                sql += f"""AND wcct.Class NOT in %s """
                param.append(event["classes"]["CT"]["Forbidden"])

        if event["use_weapons_classes"]["T"] == "weapons":
            if event["weapons"]["T"]["Allowed"]:
                sql += f"""AND tw.TWeapon in %s """
                param.append(event["weapons"]["T"]["Allowed"])
            if event["weapons"]["T"]["Forbidden"]:
                sql += f"""AND tw.TWeapon NOT in %s """
                param.append(event["weapons"]["T"]["Forbidden"])
        elif event["use_weapons_classes"]["T"] == "classes":
            if event["classes"]["T"]["Allowed"]:
                sql += f"""AND wct.Class in %s """
                param.append(event["classes"]["T"]["Allowed"])
            if event["classes"]["T"]["Forbidden"]:
                sql += f"""AND wct.Class NOT in %s """
                param.append(event["classes"]["T"]["Forbidden"])

        if event["use_weapons_classes"]["Kill"] == "weapons":
            if event["weapons"]["Kill"]:
                sql += f"""AND e.KillWeapon in %s """
                param.append(event["weapons"]["Kill"])
        elif event["use_weapons_classes"]["Kill"] == "classes":
            if event["classes"]["Kill"]:
                sql += f"""AND wck.Class in %s """
                param.append(event["classes"]["Kill"])

        sql += """) t"""

        logger.info(sql)
        logging.info(param)
        res = {}
        with conn.cursor() as cursor:
            cursor.execute(sql, param)
            result = list(cursor.fetchone())
            logging.info(cursor._last_executed)
            res["sql"] = cursor._last_executed

        if result[1] > 0:
            res["Situations_found"], res["CT_win_percentage"] = (
                result[1],
                [
                    round(100 * x)
                    for x in get_wilson_interval(result[0], result[1], 1.0)
                ],  # number of standard deviations
            )
        else:
            res["Situations_found"], res["CT_win_percentage"], res["sql"] = (
                0,
                [0, 0, 0],
                sql,
            )

        return {"statusCode": 200, "body": json.dumps(res)}
    except Exception as err:
        return {
            "statusCode": 500,
            "body": json.dumps(f"Unexpected {err=}, {type(err)=}"),
        }
