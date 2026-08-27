"""
1-2단계. 변수 선택 (3단계)
1차: 상관계수 필터 (|r| > 0.9 쌍 중 1개 제거)
2차: 다중공선성(VIF) 필터 (VIF > 10 제거)
3차: Random Forest + Lasso 중요도 결합 → 상위 N개 선택
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from config import CANDIDATE_FEATURES, CORR_THRESHOLD, VIF_THRESHOLD, N_FINAL_FEATURES


def correlation_filter(X: pd.DataFrame, threshold: float = CORR_THRESHOLD) -> list:
    """상관계수가 threshold를 넘는 쌍 중 평균 상관도가 더 높은 변수를 제거한다."""
    corr = X.corr().abs()
    mean_corr = corr.mean()
    to_drop = set()

    cols = corr.columns.tolist()
    for i, c1 in enumerate(cols):
        if c1 in to_drop:
            continue
        for c2 in cols[i + 1:]:
            if c2 in to_drop:
                continue
            if corr.loc[c1, c2] > threshold:
                # 다른 변수들과 평균적으로 더 많이 겹치는 쪽을 제거
                drop_col = c1 if mean_corr[c1] >= mean_corr[c2] else c2
                to_drop.add(drop_col)

    return [c for c in cols if c not in to_drop]


def compute_vif(X: pd.DataFrame) -> pd.Series:
    """statsmodels 없이 R² 기반으로 VIF를 직접 계산한다. VIF = 1 / (1 - R²)."""
    vif_values = {}
    for col in X.columns:
        y = X[col].values
        others = X.drop(columns=[col]).values
        r2 = LinearRegression().fit(others, y).score(others, y)
        r2 = min(r2, 0.999999)  # 완전 공선성으로 인한 division by zero 방지
        vif_values[col] = 1 / (1 - r2)
    return pd.Series(vif_values).sort_values(ascending=False)


def vif_filter(X: pd.DataFrame, threshold: float = VIF_THRESHOLD) -> list:
    """VIF가 threshold를 넘는 변수를 한 번에 하나씩 반복 제거한다."""
    cols = X.columns.tolist()
    while True:
        vif = compute_vif(X[cols])
        worst = vif.idxmax()
        if vif[worst] <= threshold or len(cols) <= 2:
            break
        cols.remove(worst)
    return cols


def rf_lasso_importance(X: pd.DataFrame, y: pd.Series, n_final: int = N_FINAL_FEATURES) -> list:
    """RF feature_importance와 Lasso 표준화 계수를 결합한 순위로 상위 n_final개를 선택한다."""
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_scaled, y)
    rf_rank = pd.Series(rf.feature_importances_, index=X.columns).rank(ascending=False)

    lasso = LassoCV(cv=5, random_state=42, max_iter=5000).fit(X_scaled, y)
    lasso_rank = pd.Series(np.abs(lasso.coef_), index=X.columns).rank(ascending=False)

    combined_rank = (rf_rank + lasso_rank).sort_values()
    return combined_rank.head(n_final).index.tolist()


def select_features(df: pd.DataFrame, target_col: str, candidates: list = CANDIDATE_FEATURES) -> dict:
    """3단계 변수 선택 파이프라인 전체 실행. 각 단계 결과를 dict로 반환한다."""
    X_full = df[candidates].fillna(df[candidates].median())
    y = df[target_col].fillna(df[target_col].median())

    step1 = correlation_filter(X_full, CORR_THRESHOLD)
    step2 = vif_filter(X_full[step1], VIF_THRESHOLD)
    step3 = rf_lasso_importance(X_full[step2], y, N_FINAL_FEATURES)

    return {
        "candidates": candidates,
        "after_corr_filter": step1,
        "after_vif_filter": step2,
        "final_features": step3,
    }


if __name__ == "__main__":
    from preprocessing import run_preprocessing

    df = run_preprocessing()
    result = select_features(df, target_col="점포당_매출")

    print(f"후보 변수: {len(result['candidates'])}개")
    print(f"1차 상관계수 필터 후: {len(result['after_corr_filter'])}개")
    print(f"2차 VIF 필터 후: {len(result['after_vif_filter'])}개")
    print(f"3차 RF+Lasso 최종 선택: {len(result['final_features'])}개")
    print(result["final_features"])
