# jonghyeok/data_analysis/round3_check_theo_diff_lag1_reversal.py

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


INPUT_FILENAME = "round3_theo_diff_5000_5300.csv"
PRODUCTS = ["VEV_5000", "VEV_5100", "VEV_5200", "VEV_5300"]


def infer_base_step(ts: pd.Series) -> int:
    diffs = ts.diff().dropna()
    diffs = diffs[diffs > 0]

    if len(diffs) == 0:
        return 100

    mode = diffs.mode()
    if len(mode) == 0:
        return int(diffs.iloc[0])

    return int(mode.iloc[0])


def load_theo_diff_data() -> pd.DataFrame:
    project_root = Path(__file__).resolve().parents[2]
    input_path = (
        project_root
        / "jonghyeok"
        / "data_analysis"
        / "output"
        / INPUT_FILENAME
    )

    if not input_path.exists():
        raise FileNotFoundError(f"파일을 찾지 못했습니다: {input_path}")

    df = pd.read_csv(input_path)

    required = {"product", "day_num", "timestamp", "combined_timestamp", "theo_diff"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}")

    df = df[df["product"].isin(PRODUCTS)].copy()

    for col in ["day_num", "timestamp", "combined_timestamp", "theo_diff"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["product", "day_num", "timestamp", "theo_diff"])
    df = df.sort_values(["product", "day_num", "timestamp"]).reset_index(drop=True)

    return df


def make_lag1_pairs(df: pd.DataFrame) -> pd.DataFrame:
    pairs = []

    for (product, day_num), g in df.groupby(["product", "day_num"], sort=False):
        g = g.sort_values("timestamp").copy()

        step = infer_base_step(g["timestamp"])

        x = g["theo_diff"]
        ts = g["timestamp"]

        r_t = x.diff()
        gap1 = ts.diff()

        # 연속 timestamp만 사용
        r_t = r_t.where(gap1 == step)

        sub = pd.DataFrame(
            {
                "product": product,
                "day_num": day_num,
                "timestamp": ts,
                "r_t": r_t,
            }
        ).dropna()

        sub["r_t1"] = sub["r_t"].shift(-1)
        future_gap = sub["timestamp"].shift(-1) - sub["timestamp"]

        sub = sub[(sub["r_t1"].notna()) & (future_gap == step)].copy()
        pairs.append(sub[["product", "day_num", "timestamp", "r_t", "r_t1"]])

    if not pairs:
        raise RuntimeError("lag-1 pair를 만들 수 없습니다.")

    return pd.concat(pairs, ignore_index=True)


def fit_line(x: np.ndarray, y: np.ndarray):
    # y = a + b x
    b, a = np.polyfit(x, y, 1)
    corr = np.corrcoef(x, y)[0, 1]

    n = len(x)
    if n >= 3 and abs(corr) < 1:
        t_stat = corr * math.sqrt((n - 2) / (1 - corr**2))
    else:
        t_stat = np.nan

    y_hat = a + b * x
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return a, b, corr, t_stat, r2


def analyze_by_product(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for product, g in pairs.groupby("product"):
        x = g["r_t"].to_numpy(dtype=float)
        y = g["r_t1"].to_numpy(dtype=float)

        a, b, corr, t_stat, r2 = fit_line(x, y)

        rows.append(
            {
                "product": product,
                "n": len(g),
                "intercept": a,
                "slope": b,
                "corr": corr,
                "t_stat": t_stat,
                "r2": r2,
                "mean_reversion_score": -b,
            }
        )

    return pd.DataFrame(rows).sort_values("product").reset_index(drop=True)


def plot_by_product(pairs: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> None:
    for product, g in pairs.groupby("product"):
        info = summary[summary["product"] == product].iloc[0]

        sample_n = min(5000, len(g))
        plot_df = g.sample(sample_n, random_state=42) if len(g) > sample_n else g

        x_plot = plot_df["r_t"].to_numpy(dtype=float)
        y_plot = plot_df["r_t1"].to_numpy(dtype=float)

        a = float(info["intercept"])
        b = float(info["slope"])

        x_line = np.linspace(x_plot.min(), x_plot.max(), 200)
        y_line = a + b * x_line

        plt.figure(figsize=(8, 8))
        plt.scatter(x_plot, y_plot, s=6, alpha=0.25)
        plt.plot(x_line, y_line)
        plt.axhline(0, linewidth=1)
        plt.axvline(0, linewidth=1)
        plt.xlabel("r_t = theo_diff_t - theo_diff_(t-1)")
        plt.ylabel("r_(t+1) = theo_diff_(t+1) - theo_diff_t")
        plt.title(
            f"{product}: theo_diff lag-1 change relation\n"
            f"corr={info['corr']:.4f}, slope={b:.4f}, "
            f"t={info['t_stat']:.2f}, r2={info['r2']:.4f}"
        )
        plt.tight_layout()

        out_path = out_dir / f"round3_theo_diff_lag1_reversal_{product}.png"
        plt.savefig(out_path, dpi=160)
        plt.close()


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    out_dir = (
        project_root
        / "jonghyeok"
        / "data_analysis"
        / "output"
        / "theo_diff_lag1_reversal"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_theo_diff_data()
    pairs = make_lag1_pairs(df)
    summary = analyze_by_product(pairs)

    pairs_path = out_dir / "round3_theo_diff_lag1_pairs.csv"
    summary_path = out_dir / "round3_theo_diff_lag1_summary.csv"

    pairs.to_csv(pairs_path, index=False)
    summary.to_csv(summary_path, index=False)

    plot_by_product(pairs, summary, out_dir)

    print("\n=== Theo Diff Lag-1 Reversal Summary ===")
    print(
        summary[
            [
                "product",
                "n",
                "slope",
                "corr",
                "t_stat",
                "r2",
                "mean_reversion_score",
            ]
        ].to_string(index=False)
    )

    print("\n[SAVED]", pairs_path)
    print("[SAVED]", summary_path)
    print("[SAVED]", out_dir)


if __name__ == "__main__":
    main()