# jonghyeok/data_analysis/round3_plot_theo_diff_distribution.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


INPUT_FILENAME = "round3_theo_diff_5000_5300.csv"

PRODUCTS = ["VEV_5000", "VEV_5100", "VEV_5200", "VEV_5300"]

BINS = 120
CLIP_PERCENTILE_LOW = 0.5
CLIP_PERCENTILE_HIGH = 99.5


def load_data() -> pd.DataFrame:
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

    required = {"product", "theo_diff"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}")

    df = df[df["product"].isin(PRODUCTS)].copy()
    df["theo_diff"] = pd.to_numeric(df["theo_diff"], errors="coerce")
    df = df.dropna(subset=["product", "theo_diff"])

    return df


def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for product, g in df.groupby("product"):
        x = g["theo_diff"].to_numpy(dtype=float)

        rows.append(
            {
                "product": product,
                "n": len(x),
                "mean": float(np.mean(x)),
                "std": float(np.std(x, ddof=1)),
                "median": float(np.median(x)),
                "min": float(np.min(x)),
                "p01": float(np.percentile(x, 1)),
                "p05": float(np.percentile(x, 5)),
                "p25": float(np.percentile(x, 25)),
                "p75": float(np.percentile(x, 75)),
                "p95": float(np.percentile(x, 95)),
                "p99": float(np.percentile(x, 99)),
                "max": float(np.max(x)),
            }
        )

    return pd.DataFrame(rows).sort_values("product").reset_index(drop=True)


def plot_each_product(df: pd.DataFrame, out_dir: Path) -> None:
    for product, g in df.groupby("product"):
        x = g["theo_diff"].to_numpy(dtype=float)

        lo = np.percentile(x, CLIP_PERCENTILE_LOW)
        hi = np.percentile(x, CLIP_PERCENTILE_HIGH)
        x_clip = x[(x >= lo) & (x <= hi)]

        mean = np.mean(x)
        median = np.median(x)

        plt.figure(figsize=(10, 6))
        plt.hist(x_clip, bins=BINS, density=True, alpha=0.75)
        plt.axvline(0, linewidth=1, linestyle="--", label="0")
        plt.axvline(mean, linewidth=1, linestyle="--", label=f"mean={mean:.4f}")
        plt.axvline(median, linewidth=1, linestyle="--", label=f"median={median:.4f}")

        plt.xlabel("theo_diff = option_valid_mid - theo")
        plt.ylabel("density")
        plt.title(
            f"{product}: theo_diff distribution\n"
            f"clipped to p{CLIP_PERCENTILE_LOW}~p{CLIP_PERCENTILE_HIGH}"
        )
        plt.legend()
        plt.tight_layout()

        out_path = out_dir / f"round3_theo_diff_distribution_{product}.png"
        plt.savefig(out_path, dpi=160)
        plt.close()


def plot_combined(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, product in zip(axes, PRODUCTS):
        g = df[df["product"] == product]
        x = g["theo_diff"].to_numpy(dtype=float)

        lo = np.percentile(x, CLIP_PERCENTILE_LOW)
        hi = np.percentile(x, CLIP_PERCENTILE_HIGH)
        x_clip = x[(x >= lo) & (x <= hi)]

        mean = np.mean(x)
        median = np.median(x)

        ax.hist(x_clip, bins=BINS, density=True, alpha=0.75)
        ax.axvline(0, linewidth=1, linestyle="--")
        ax.axvline(mean, linewidth=1, linestyle="--")
        ax.axvline(median, linewidth=1, linestyle="--")

        ax.set_title(f"{product}\nmean={mean:.4f}, median={median:.4f}")
        ax.set_xlabel("theo_diff")
        ax.set_ylabel("density")

    plt.suptitle("Round 3 theo_diff distributions by voucher", y=1.02)
    plt.tight_layout()

    out_path = out_dir / "round3_theo_diff_distribution_by_product.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    out_dir = (
        project_root
        / "jonghyeok"
        / "data_analysis"
        / "output"
        / "theo_diff_distribution"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data()
    summary = make_summary(df)

    summary_path = out_dir / "round3_theo_diff_distribution_summary.csv"
    summary.to_csv(summary_path, index=False)

    plot_each_product(df, out_dir)
    plot_combined(df, out_dir)

    print("\n=== Theo Diff Distribution Summary ===")
    print(summary.to_string(index=False))

    print("\n[SAVED]", summary_path)
    print("[SAVED]", out_dir)


if __name__ == "__main__":
    main()