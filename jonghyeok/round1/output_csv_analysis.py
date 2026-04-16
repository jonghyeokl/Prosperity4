from pathlib import Path
import pandas as pd

# 입력 / 출력 파일 경로
INPUT_CSV = Path("jonghyeok/round1/output/ash_alpha_all_results.csv")
OUTPUT_CSV = Path("jonghyeok/round1/output/ash_alpha_total_profit_over_56900.csv")

THRESHOLD = 56900

def main():
    df = pd.read_csv(INPUT_CSV)

    # total_profit > 56900 인 행만 필터링
    filtered = df[df["total_profit"] > THRESHOLD].copy()

    # alpha1 ~ alpha10, day별 profit, total_profit 순서로 정렬해서 보고 싶으면 아래 사용
    preferred_cols = [
        "alpha1", "alpha2", "alpha3", "alpha4", "alpha5",
        "alpha6", "alpha7", "alpha8", "alpha9", "alpha10",
        "day_-2_profit", "day_-1_profit", "day_0_profit", "total_profit",
    ]
    existing_cols = [col for col in preferred_cols if col in filtered.columns]
    remaining_cols = [col for col in filtered.columns if col not in existing_cols]
    filtered = filtered[existing_cols + remaining_cols]

    # total_profit 내림차순 정렬
    filtered = filtered.sort_values(by="total_profit", ascending=False).reset_index(drop=True)

    # 콘솔 출력
    if filtered.empty:
        print(f"total_profit > {THRESHOLD} 인 행이 없습니다.")
    else:
        print(filtered.to_string(index=False))

    # 별도 파일 저장
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\nSaved filtered rows to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()