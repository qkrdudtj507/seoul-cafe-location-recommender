"""
1단계. 데이터 준비
- 원본(분기 × 상권, wide-format) 데이터를 커피-음료 업종으로 필터링
- 상권 코드 기준 4개 분기 평균 집계 (분기 × 상권 → 상권 단위)
- 소비지출 · 상주인구 결측 상권 제외
- 파생 변수 생성 (점포당 매출, 유동인구 비율 등)
"""
import pandas as pd

from config import (
    RAW_DATA_PATH, TARGET_INDUSTRY, ID_COL, META_COLS,
    SALES_COL, STORE_COUNT_COL,
)


def load_raw(path=RAW_DATA_PATH) -> pd.DataFrame:
    """원본 병합 데이터 로드 (분기 x 상권 wide format)."""
    return pd.read_csv(path, encoding="utf-8-sig")


def filter_target_industry(df: pd.DataFrame, industry: str = TARGET_INDUSTRY) -> pd.DataFrame:
    """분석 대상 업종(커피-음료)만 남긴다."""
    col = "서비스_업종_코드_명"
    out = df[df[col] == industry].copy()
    return out


def aggregate_by_area(df: pd.DataFrame) -> pd.DataFrame:
    """상권 코드 기준으로 4개 분기를 평균 집계해 상권 단위 행으로 축소한다."""
    meta = df[META_COLS].drop_duplicates(subset=[ID_COL])

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != ID_COL]

    agg = df.groupby(ID_COL)[numeric_cols].mean().reset_index()
    merged = meta.merge(agg, on=ID_COL, how="right")
    return merged


def drop_missing_areas(df: pd.DataFrame) -> pd.DataFrame:
    """소비지출·상주인구 등 핵심 지표가 결측인 상권(약 2%)을 제외한다."""
    key_cols = [c for c in df.columns if "지출_총금액" == c or "총_상주인구_수" == c]
    return df.dropna(subset=key_cols).reset_index(drop=True)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """점포당 매출(타깃)과 유동인구 비율 등 파생 변수를 추가한다."""
    out = df.copy()

    # 타깃: 점포당 매출 (0으로 나누기 방지)
    out["점포당_매출"] = out[SALES_COL] / out[STORE_COUNT_COL].replace(0, pd.NA)

    total_flow = out["총_유동인구_수"].replace(0, pd.NA)
    out["주말유동인구_비율"] = (out["토요일_유동인구_수"] + out["일요일_유동인구_수"]) / total_flow
    out["오전유동인구_비율"] = out["시간대_06_11_유동인구_수"] / total_flow
    out["2030유동인구_비율"] = (out["연령대_20_유동인구_수"] + out["연령대_30_유동인구_수"]) / total_flow

    return out


def run_preprocessing(path=RAW_DATA_PATH) -> pd.DataFrame:
    """전처리 파이프라인 전체 실행."""
    df = load_raw(path)
    df = filter_target_industry(df)
    df = aggregate_by_area(df)
    df = drop_missing_areas(df)
    df = add_derived_features(df)
    return df


if __name__ == "__main__":
    result = run_preprocessing()
    print(f"전처리 완료: {result.shape[0]}개 상권 x {result.shape[1]}개 변수")
    print(result[["상권_코드_명", "자치구_코드_명", "점포당_매출"]].head())
