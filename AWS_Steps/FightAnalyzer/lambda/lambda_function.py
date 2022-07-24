import sys
import logging
import rds_config
import pymysql
import json

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


def lambda_handler(event, context):
    """
    This function fetches content from MySQL RDS instance
    """
    try:
        ct_pos = ", ".join(f'"{val}"' for val in event["positions"]["CT"])
        t_pos = ", ".join(f'"{val}"' for val in event["positions"]["T"])
        T_weapon = ", ".join(f'"{val}"' for val in event["weapons"]["T"]["Allowed"])
        not_T_weapon = ", ".join(
            f'"{val}"' for val in event["weapons"]["T"]["Forbidden"]
        )
        CT_weapon = ", ".join(f'"{val}"' for val in event["weapons"]["CT"]["Allowed"])
        not_CT_weapon = ", ".join(
            f'"{val}"' for val in event["weapons"]["CT"]["Forbidden"]
        )
        Kill_weapon = ", ".join(f'"{val}"' for val in event["weapons"]["Kill"])

        CT_classes = ", ".join(f'"{val}"' for val in event["classes"]["CT"]["Allowed"])
        T_classes = ", ".join(f'"{val}"' for val in event["classes"]["T"]["Allowed"])
        Kill_classes = ", ".join(f'"{val}"' for val in event["classes"]["Kill"])
        not_CT_classes = ", ".join(
            f'"{val}"' for val in event["classes"]["CT"]["Forbidden"]
        )
        not_T_classes = ", ".join(
            f'"{val}"' for val in event["classes"]["T"]["Forbidden"]
        )

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
            f"""WHERE e.MapName = '{event["map_name"]}' """
            f"""AND e.Time BETWEEN {event["times"]["start"]} AND {event["times"]["end"]} """
        )
        if ct_pos != "":
            sql += f"""AND e.CTArea in ({ct_pos}) """
        if t_pos != "":
            sql += f"""AND e.TArea in ({t_pos}) """

        if event["use_weapons_classes"]["CT"] == "weapons":
            if CT_weapon != "":
                sql += f"""AND ctw.CTWeapon in ({CT_weapon}) """
            if not_CT_weapon != "":
                sql += f"""AND ctw.CTWeapon NOT in ({not_CT_weapon}) """
        elif event["use_weapons_classes"]["CT"] == "classes":
            if CT_classes != "":
                sql += f"""AND wcct.Class in ({CT_classes}) """
            if not_CT_classes != "":
                sql += f"""AND wcct.Class NOT in ({not_CT_classes}) """

        if event["use_weapons_classes"]["T"] == "weapons":
            if T_weapon != "":
                sql += f"""AND tw.TWeapon in ({T_weapon}) """
            if not_T_weapon != "":
                sql += f"""AND tw.TWeapon NOT in ({not_T_weapon}) """
        elif event["use_weapons_classes"]["T"] == "classes":
            if T_classes != "":
                sql += f"""AND wct.Class in ({T_classes}) """
            if not_T_classes != "":
                sql += f"""AND wct.Class NOT in ({not_T_classes}) """

        if event["use_weapons_classes"]["Kill"] == "weapons":
            if Kill_weapon != "":
                sql += f"""AND e.KillWeapon in ({Kill_weapon}) """
        elif event["use_weapons_classes"]["Kill"] == "classes":
            if Kill_classes != "":
                sql += f"""AND wck.Class in ({Kill_classes}) """

        sql += """) t"""

        logger.info(sql)
        with conn.cursor() as cursor:
            cursor.execute(sql)
            result = list(cursor.fetchone())

        res = {}
        if result[1] > 0:
            res["Situations_found"], res["CT_win_percentage"] = result[1], round(
                100 * result[0]
            )
        else:
            res["Situations_found"], res["CT_win_percentage"] = 0, 0
        return {"statusCode": 200, "body": json.dumps(res)}
    except Exception as err:
        return {
            "statusCode": 500,
            "body": json.dumps(f"Unexpected {err=}, {type(err)=}"),
        }
