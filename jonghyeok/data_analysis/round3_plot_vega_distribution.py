# jonghyeok/data_analysis/round3_plot_vega_distribution.py

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TARGET_PRODUCTS = [
    "VEV_5000",
    "VEV_5100",
    "VEV_5200",
    "VEV_5300",
]

LOW_VEGA_CUTOFF = 0.5
BINS = 80


def load_cached_feature_df(input_path: Path) -> pd.DataFrame:
    data = json.loads(input_path.read_text())

    if isinstance(data, dict) and "features" in data:
        features = data["features"]
    else:
        features = data

    if isinstance(features, dict):
        rows = list(features.values())
    elif isinstance(features, list):
        rows = features
    else:
        raise ValueError("지원하지 않는 feature 파일 구조입니다.")

    df = pd.DataFrame(rows)

    if "product" not in df.columns or "vega" not in df.columns:
        raise ValueError("feature 파일에 product 또는 vega 컬럼이 없습니다.")

    df["vega"] = pd.to_numeric(df["vega"], errors="coerce")
    df = df[np.isfinite(df["vega"])].copy()

    return df


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for product, g in df.groupby("product"):
        v = g["vega"].to_numpy(dtype=float)

        if len(v) == 0:
            continue

        rows.append(
            {
                "product": product,
                "n": len(v),
                "mean": np.mean(v),
                "std": np.std(v),
                "min": np.min(v),
                "p01": np.quantile(v, 0.01),
                "p05": np.quantile(v, 0.05),
                "p10": np.quantile(v, 0.10),
                "p25": np.quantile(v, 0.25),
                "p50": np.quantile(v, 0.50),
                "p75": np.quantile(v, 0.75),
                "p90": np.quantile(v, 0.90),
                "p95": np.quantile(v, 0.95),
                "p99": np.quantile(v, 0.99),
                "max": np.max(v),
                "low_vega_ratio": np.mean(v <= LOW_VEGA_CUTOFF),
                "low_vega_count": int(np.sum(v <= LOW_VEGA_CUTOFF)),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("product").reset_index(drop=True)

    return out


def plot_histograms(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for product in TARGET_PRODUCTS:
        g = df[df["product"] == product].copy()

        if g.empty:
            print(f"[SKIP] {product}: no rows")
            continue

        vega = g["vega"].to_numpy(dtype=float)

        plt.figure(figsize=(8, 5))
        plt.hist(vega, bins=BINS)
        plt.axvline(LOW_VEGA_CUTOFF, linestyle="--", linewidth=1.5)

        plt.title(f"{product} vega distribution")
        plt.xlabel("vega")
        plt.ylabel("count")
        plt.tight_layout()

        out_path = out_dir / f"{product}_vega_hist.png"
        plt.savefig(out_path, dpi=150)
        plt.close()

        print(f"[SAVED] {out_path}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    input_path = project_root / "jonghyeok" / "data_analysis" / "output" / "round3_main_cached_features.json"
    out_dir = project_root / "jonghyeok" / "data_analysis" / "output" / "vega_distribution"

    df = load_cached_feature_df(input_path)

    if TARGET_PRODUCTS:
        df = df[df["product"].isin(TARGET_PRODUCTS)].copy()

    summary = build_summary(df)
    summary_path = out_dir / "vega_summary.csv"

    plot_histograms(df, out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)

    print("\n=== Vega Summary ===")
    if not summary.empty:
        print(
            summary[
                [
                    "product",
                    "n",
                    "mean",
                    "p05",
                    "p10",
                    "p25",
                    "p50",
                    "p75",
                    "p90",
                    "p95",
                    "max",
                    "low_vega_ratio",
                    "low_vega_count",
                ]
            ].to_string(index=False)
        )

    print("\n[SAVED]", summary_path)


if __name__ == "__main__":
    main()