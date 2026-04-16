from pathlib import Path
import pandas as pd

INPUT_CSV = Path("jonghyeok/round1/output/ash_alpha_all_results.csv")
OUTPUT_DIR = Path("jonghyeok/round1/output/custom_alpha_group_means")

# 원하는 alpha 인덱스 조합들
GROUPS_TO_ANALYZE = [
    (8, 9, 10)
]


def validate_groups(df: pd.DataFrame, groups: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    alpha_cols = set(df.columns)
    validated = []

    for group in groups:
        if len(group) == 0:
            raise ValueError("빈 그룹은 허용되지 않습니다.")

        if len(set(group)) != len(group):
            raise ValueError(f"중복 인덱스가 있는 그룹입니다: {group}")

        for idx in group:
            col = f"alpha{idx}"
            if col not in alpha_cols:
                raise ValueError(f"{col} 컬럼이 없습니다. 문제의 그룹: {group}")

        validated.append(tuple(group))

    return validated


def group_name(group: tuple[int, ...]) -> str:
    return "_".join([f"alpha{i}" for i in group])


def main():
    df = pd.read_csv(INPUT_CSV)

    if "total_profit" not in df.columns:
        raise ValueError("total_profit 컬럼이 없습니다.")

    groups = validate_groups(df, GROUPS_TO_ANALYZE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    combined_rows = []

    for group in groups:
        cols = [f"alpha{i}" for i in group]

        grouped = (
            df.groupby(cols, as_index=False)
            .agg(
                mean_total_profit=("total_profit", "mean"),
                count=("total_profit", "size"),
                max_total_profit=("total_profit", "max"),
                min_total_profit=("total_profit", "min"),
                std_total_profit=("total_profit", "std"),
            )
            .sort_values(
                by=["mean_total_profit"] + cols,
                ascending=[False] + [True] * len(cols),
            )
            .reset_index(drop=True)
        )

        name = group_name(group)

        print(f"\n=== {name} ===")
        print(grouped.to_string(index=False))

        output_csv = OUTPUT_DIR / f"{name}_mean_total_profit.csv"
        grouped.to_csv(output_csv, index=False, encoding="utf-8-sig")

        tmp = grouped.copy()
        tmp.insert(0, "group", name)
        combined_rows.append(tmp)

    combined_df = pd.concat(combined_rows, ignore_index=True)
    combined_csv = OUTPUT_DIR / "all_custom_alpha_groups_mean_total_profit.csv"
    combined_df.to_csv(combined_csv, index=False, encoding="utf-8-sig")

    print(f"\nSaved combined result to: {combined_csv}")


if __name__ == "__main__":
    main()