"""
서울시 카페 입지 추천 대시보드 (Streamlit)

실행:
    streamlit run dashboard/app.py

기본적으로 로컬 CSV(data/processed/카페_입지추천_전체결과.csv)를 읽습니다.
Snowflake 대시보드로 전환하려면 로컬 실행부 대신 load_from_snowflake()를 사용하세요.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import MERGED_OUTPUT_PATH  # noqa: E402

st.set_page_config(page_title="서울시 카페 입지 추천", page_icon="☕", layout="wide")


@st.cache_data
def load_from_local() -> pd.DataFrame:
    return pd.read_csv(MERGED_OUTPUT_PATH, encoding="utf-8-sig")


def load_from_snowflake() -> pd.DataFrame:
    """Snowflake 테이블에서 결과를 직접 조회한다. secrets.toml에 연결 정보 필요."""
    import snowflake.connector

    conn = snowflake.connector.connect(
        user=st.secrets["snowflake"]["user"],
        password=st.secrets["snowflake"]["password"],
        account=st.secrets["snowflake"]["account"],
        warehouse=st.secrets["snowflake"]["warehouse"],
        database=st.secrets["snowflake"]["database"],
        schema=st.secrets["snowflake"]["schema"],
    )
    query = "SELECT * FROM 카페_입지추천_전체결과"
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def main():
    st.title("☕ 서울시 카페 창업 입지 추천")
    st.caption("서울시 상권분석서비스 공공데이터 기반 · TEAM 커머셜")

    df = load_from_local()

    districts = ["전체"] + sorted(df["자치구_코드_명"].dropna().unique().tolist())
    col1, col2 = st.columns([1, 3])
    with col1:
        selected_district = st.selectbox("자치구 선택", districts)
        top_n = st.slider("표시할 상위 상권 수", min_value=5, max_value=50, value=15, step=5)

    filtered = df if selected_district == "전체" else df[df["자치구_코드_명"] == selected_district]
    filtered = filtered.sort_values("종합점수", ascending=False).head(top_n)

    with col2:
        st.subheader(f"TOP {top_n} 추천 상권")
        st.dataframe(
            filtered[["상권_코드_명", "자치구_코드_명", "행정동_코드_명",
                      "종합점수", "성과잠재력_점수", "수요공급갭_점수", "경쟁카페수"]],
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("종합점수 vs 경쟁카페수")
    st.scatter_chart(filtered, x="경쟁카페수", y="종합점수", size="점포당_매출")

    st.caption(
        "종합점수 = 0.5 × 성과잠재력 점수 + 0.5 × 수요-공급 갭 점수 (0~100 백분위)"
    )


if __name__ == "__main__":
    main()
