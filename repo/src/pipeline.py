"""
서울시 카페 입지 추천 시스템 - 엔드투엔드 파이프라인

실행:
    python src/pipeline.py

결과:
    data/processed/카페_입지추천_전체결과.csv 에 전체 상권 점수 저장
"""
from preprocessing import run_preprocessing
from feature_selection import select_features
from scoring import compute_final_score
from config import MERGED_OUTPUT_PATH, PROCESSED_DIR

OUTPUT_COLS = [
    "상권_코드", "상권_코드_명", "자치구_코드_명", "행정동_코드_명",
    "종합점수", "성과잠재력_점수", "수요공급갭_점수",
    "경쟁카페수", "점포당_매출", "총_유동인구_수", "총_상주인구_수", "총_직장_인구_수",
]


def run():
    print("[1/3] 데이터 전처리 중...")
    df = run_preprocessing()
    print(f"      -> {df.shape[0]}개 상권, {df.shape[1]}개 변수")

    print("[2/3] 변수 선택 중 (상관계수 → VIF → RF+Lasso)...")
    fs = select_features(df, target_col="점포당_매출")
    print(f"      -> 최종 선택 변수: {fs['final_features']}")

    print("[3/3] 스코어링 중 (성과잠재력 + 수요공급갭)...")
    scored = compute_final_score(df, fs["final_features"])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    scored[OUTPUT_COLS].to_csv(MERGED_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n완료! 결과 저장: {MERGED_OUTPUT_PATH}")

    print("\n=== TOP 15 추천 상권 ===")
    print(scored[["상권_코드_명", "자치구_코드_명", "종합점수", "경쟁카페수"]].head(15).to_string(index=False))

    return scored


if __name__ == "__main__":
    run()
