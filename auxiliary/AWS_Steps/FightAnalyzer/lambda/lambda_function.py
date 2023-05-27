"""AWS Lambda function to call to query database for fight scenarios."""

# pylint: disable=invalid-name

import json
import logging
import math

import pymysql
import rds_config

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
except pymysql.MySQLError:
    logger.exception("ERROR: Unexpected error: Could not connect to MySQL instance.")

logger.info("SUCCESS: Connection to RDS MySQL instance succeeded")


def get_wilson_interval(
    success_percent: float, total_n: int, z: float
) -> tuple[float, float, float]:
    """Calculates the Wilson score interval for a series of success-failure experiments.

    Calcualtes the Wilson score interval as anapproximation of
    the binomial proportion confidence interval.

    Args:
        success_percent (float): Percentage of experiments that ended in success
        total_n (int): Total number of experiments
        z (float): Number of standard deviations that the interval should cover
    Returns:
        A list of floats of the form:
        [lower_bound_of_interval, success_percent, upper_bound_of_interval]
    """
    lower = (
        (success_percent + z**2 / (2 * total_n))
        - (
            z
            * math.sqrt(
                (success_percent * (1 - success_percent) + z**2 / (4 * total_n))
                / total_n
            )
        )
    ) / (1 + z**2 / total_n)
    upper = (
        (success_percent + z**2 / (2 * total_n))
        + (
            z
            * math.sqrt(
                (success_percent * (1 - success_percent) + z**2 / (4 * total_n))
                / total_n
            )
        )
    ) / (1 + z**2 / total_n)
    return [lower, success_percent, upper]


def lambda_handler(event: dict, context) -> dict:
    """This function fetches content from MySQL RDS instance."""
    try:
        sql = (
            """SELECT AVG(t.CTWon), COUNT(t.CTWon) """
            """FROM ( """
            """SELECT DISTINCT e.EventID, e.CTWon """
            """FROM Events e JOIN CTWeapons ctw """
            """ON e.EventID = ctw.EventID """
            """JOIN TWeapons tw """
            """ON e.EventID = tw.EventID """
            """JOIN WeaponClasses wcct """
            """ON ctw.CTWeapon = wcct.WeaponName """
            """JOIN WeaponClasses wct """
            """ON tw.TWeapon = wct.WeaponName """
            """JOIN WeaponClasses wck """
            """ON e.KillWeapon = wck.WeaponName """
            """WHERE e.MapName = %s """
            """AND e.Time BETWEEN %s AND %s """
        )
        param = [event["map_name"], event["times"]["start"], event["times"]["end"]]

        if event["positions"]["CT"]:
            sql += """AND e.CTArea in %s """
            param.append(event["positions"]["CT"])
        if event["positions"]["T"]:
            sql += """AND e.TArea in %s """
            param.append(event["positions"]["T"])
        if event["use_weapons_classes"]["CT"] == "weapons":
            if event["weapons"]["CT"]["Allowed"]:
                sql += """AND ctw.CTWeapon in %s """
                param.append(event["weapons"]["CT"]["Allowed"])
            if event["weapons"]["CT"]["Forbidden"]:
                sql += """AND ctw.CTWeapon NOT in %s """
                param.append(event["weapons"]["CT"]["Forbidden"])
        elif event["use_weapons_classes"]["CT"] == "classes":
            if event["classes"]["CT"]["Allowed"]:
                sql += """AND wcct.Class in %s """
                param.append(event["classes"]["CT"]["Allowed"])
            if event["classes"]["CT"]["Forbidden"]:
                sql += """AND wcct.Class NOT in %s """
                param.append(event["classes"]["CT"]["Forbidden"])

        if event["use_weapons_classes"]["T"] == "weapons":
            if event["weapons"]["T"]["Allowed"]:
                sql += """AND tw.TWeapon in %s """
                param.append(event["weapons"]["T"]["Allowed"])
            if event["weapons"]["T"]["Forbidden"]:
                sql += """AND tw.TWeapon NOT in %s """
                param.append(event["weapons"]["T"]["Forbidden"])
        elif event["use_weapons_classes"]["T"] == "classes":
            if event["classes"]["T"]["Allowed"]:
                sql += """AND wct.Class in %s """
                param.append(event["classes"]["T"]["Allowed"])
            if event["classes"]["T"]["Forbidden"]:
                sql += """AND wct.Class NOT in %s """
                param.append(event["classes"]["T"]["Forbidden"])

        if event["use_weapons_classes"]["Kill"] == "weapons":
            if event["weapons"]["Kill"]:
                sql += """AND e.KillWeapon in %s """
                param.append(event["weapons"]["Kill"])
        elif event["use_weapons_classes"]["Kill"] == "classes":  # noqa: SIM102
            if event["classes"]["Kill"]:
                sql += """AND wck.Class in %s """
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
                    for x in get_wilson_interval(float(result[0]), result[1], 1.0)
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
