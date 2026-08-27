"""
결과 CSV를 Snowflake 테이블로 적재하는 스크립트.

사전 준비:
    1. .env 파일에 Snowflake 연결 정보 작성 (SNOWFLAKE_USER, PASSWORD, ACCOUNT, WAREHOUSE, DATABASE, SCHEMA)
    2. pip install snowflake-connector-python python-dotenv

실행:
    python src/load_to_snowflake.py
"""
import os

import pandas as pd
from dotenv import load_dotenv

from config import MERGED_OUTPUT_PATH

TABLE_NAME = "카페_입지추천_전체결과"

DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    상권_코드 STRING,
    상권_코드_명 STRING,
    자치구_코드_명 STRING,
    행정동_코드_명 STRING,
    종합점수 FLOAT,
    성과잠재력_점수 FLOAT,
    수요공급갭_점수 FLOAT,
    경쟁카페수 FLOAT,
    점포당_매출 FLOAT,
    총_유동인구_수 FLOAT,
    총_상주인구_수 FLOAT,
    총_직장_인구_수 FLOAT
)
"""


def main():
    load_dotenv()
    import snowflake.connector
    from snowflake.connector.pandas_tools import write_pandas

    df = pd.read_csv(MERGED_OUTPUT_PATH, encoding="utf-8-sig")

    conn = snowflake.connector.connect(
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        database=os.environ["SNOWFLAKE_DATABASE"],
        schema=os.environ["SNOWFLAKE_SCHEMA"],
    )
    conn.cursor().execute(DDL)

    success, n_chunks, n_rows, _ = write_pandas(conn, df, TABLE_NAME.upper())
    print(f"업로드 {'성공' if success else '실패'}: {n_rows}행, {n_chunks}청크")

    conn.close()


if __name__ == "__main__":
    main()
