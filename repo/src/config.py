"""
프로젝트 전역 설정: 경로, 원본 컬럼명, 분석 파라미터
서울시 상권분석서비스 데이터 컬럼명을 그대로 사용합니다.
"""
from pathlib import Path

# --- 경로 ---
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "final_merged_data.csv"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MERGED_OUTPUT_PATH = PROCESSED_DIR / "카페_입지추천_전체결과.csv"

# --- 분석 대상 업종 ---
TARGET_INDUSTRY = "커피-음료"

# --- 상권 식별 / 메타 컬럼 ---
ID_COL = "상권_코드"
META_COLS = ["상권_코드", "상권_코드_명", "자치구_코드_명", "행정동_코드_명"]

# --- 매출 관련 ---
SALES_COL = "당월_매출_금액"
STORE_COUNT_COL = "stor_co"                     # 동일 업종 전체 점포수
SIMILAR_STORE_COL = "similr_induty_stor_co"      # 유사 업종 경쟁 점포수
CLOSE_RATE_COL = "clsbiz_rt"                     # 폐업률

# --- 데이터 누수 위험 컬럼 (요일/시간대/성별/연령별 매출) : 후보에서 제외 ---
LEAKAGE_PATTERNS = [
    "매출_금액", "매출_건수",  # 당월_매출_금액 자체는 별도로 타깃 계산에만 사용, 이후 후보에서 제외
]

# --- 변수 선택 파라미터 ---
CORR_THRESHOLD = 0.9
VIF_THRESHOLD = 10
N_FINAL_FEATURES = 10

# --- 후보 변수 (수요 / 공급 / 인프라 3개 축에서 파생) ---
CANDIDATE_FEATURES = [
    # 수요 - 유동인구
    "총_유동인구_수", "연령대_20_유동인구_수", "연령대_30_유동인구_수",
    "시간대_06_11_유동인구_수", "토요일_유동인구_수", "일요일_유동인구_수",
    # 수요 - 상주인구 / 직장인구
    "총_상주인구_수", "총_직장_인구_수", "총_가구_수",
    # 수요 - 소비력
    "지출_총금액", "여가_지출_총금액", "문화_지출_총금액",
    # 공급 - 경쟁
    "stor_co", "similr_induty_stor_co", "frc_stor_co", "opbiz_rt", "clsbiz_rt",
    # 인프라 - 시설/교통
    "집객시설_수", "은행_수", "지하철_역_수", "버스_정거장_수",
    "종합병원_수", "일반_병원_수", "대학교_수", "백화점_수",
]
