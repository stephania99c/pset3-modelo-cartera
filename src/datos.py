"""Lectura de la OBT del PSet 2 desde Snowflake."""

import os
from pathlib import Path

import snowflake.connector
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

TABLA = "CARTERA.GOLD.OBT_CARTERA_DIARIA"


def conectar():
    return snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PAT"],
        role=os.environ["SNOWFLAKE_ROLE"],
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
    )


def cargar_obt():
    con = conectar()
    df = con.cursor().execute(f"SELECT * FROM {TABLA}").fetch_pandas_all()
    con.close()
    df.columns = [c.lower() for c in df.columns]
    return df