"""
1-3단계. 종합 추천 점수 산출
- 성과 잠재력 점수 (50%): 선택 변수로 학습한 RF 모델의 out-of-fold 예측 점포당 매출 → 백분위
- 수요-공급 갭 점수 (50%): (유동/상주/직장인구·주말유동비율 표준화 평균) - (경쟁점포수·카페밀도 표준화 평균)
- 종합점수 = 0.5 × 성과잠재력 + 0.5 × 수요공급갭 (0~100 백분위 스케일)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

DEMAND_COLS = ["총_유동인구_수", "총_상주인구_수", "총_직장_인구_수", "주말유동인구_비율"]
COMPETITION_COLS = ["similr_induty_stor_co", "카페_밀도"]


def oof_performance_score(df: pd.DataFrame, features: list, target_col: str, n_splits: int = 5) -> pd.Series:
    """5-fold CV로 out-of-fold 예측을 만들고, 0~100 백분위 점수로 변환한다."""
    X = df[features].fillna(df[features].median()).values
    y = df[target_col].fillna(df[target_col].median()).values

    oof_pred = np.zeros(len(df))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, valid_idx in kf.split(X):
        model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        model.fit(X[train_idx], y[train_idx])
        oof_pred[valid_idx] = model.predict(X[valid_idx])

    return pd.Series(oof_pred, index=df.index).rank(pct=True) * 100


def demand_supply_gap_score(df: pd.DataFrame) -> pd.Series:
    """수요 지표 표준화 평균 - 경쟁 지표 표준화 평균 → 0~100 백분위 점수."""
    out = df.copy()
    out["카페_밀도"] = out["stor_co"] / out["영역_면적"].replace(0, np.nan)

    scaler = StandardScaler()

    demand = out[DEMAND_COLS].fillna(out[DEMAND_COLS].median())
    demand_z = pd.DataFrame(scaler.fit_transform(demand), index=out.index, columns=DEMAND_COLS)
    demand_score = demand_z.mean(axis=1)

    competition = out[COMPETITION_COLS].fillna(out[COMPETITION_COLS].median())
    competition_z = pd.DataFrame(scaler.fit_transform(competition), index=out.index, columns=COMPETITION_COLS)
    competition_score = competition_z.mean(axis=1)

    gap = demand_score - competition_score
    return gap.rank(pct=True) * 100


def compute_final_score(df: pd.DataFrame, final_features: list, target_col: str = "점포당_매출") -> pd.DataFrame:
    """성과잠재력 + 수요공급갭을 절반씩 결합해 종합점수를 산출한다."""
    out = df.copy()
    out["성과잠재력_점수"] = oof_performance_score(out, final_features, target_col)
    out["수요공급갭_점수"] = demand_supply_gap_score(out)
    out["종합점수"] = 0.5 * out["성과잠재력_점수"] + 0.5 * out["수요공급갭_점수"]
    out["경쟁카페수"] = out["similr_induty_stor_co"]
    return out.sort_values("종합점수", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    from preprocessing import run_preprocessing
    from feature_selection import select_features

    df = run_preprocessing()
    fs = select_features(df, target_col="점포당_매출")
    scored = compute_final_score(df, fs["final_features"])

    top15 = scored[["상권_코드_명", "자치구_코드_명", "행정동_코드_명",
                     "종합점수", "성과잠재력_점수", "수요공급갭_점수", "경쟁카페수"]].head(15)
    print(top15.to_string(index=False))
