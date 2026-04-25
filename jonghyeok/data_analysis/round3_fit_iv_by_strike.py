from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_PRODUCTS = [
    "VEV_5000",
    "VEV_5100",
    "VEV_5200",
    "VEV_5300",
    "VEV_5400",
    "VEV_5500",
]


def calc_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    resid = y - pred
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "n": int(len(y)),
    }


def fit_strike_mean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = df.copy()

    strike_iv = out.groupby("product")["raw_iv"].mean().to_dict()
    out["fit_iv"] = out["product"].map(strike_iv)
    out["residual"] = out["raw_iv"] - out["fit_iv"]

    metrics = calc_metrics(out["raw_iv"].to_numpy(), out["fit_iv"].to_numpy())
    return out, {
        "fit_type": "strike_mean",
        "strike_iv": {k: float(v) for k, v in strike_iv.items()},
        **metrics,
    }


def fit_strike_median(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = df.copy()

    strike_iv = out.groupby("product")["raw_iv"].median().to_dict()
    out["fit_iv"] = out["product"].map(strike_iv)
    out["residual"] = out["raw_iv"] - out["fit_iv"]

    metrics = calc_metrics(out["raw_iv"].to_numpy(), out["fit_iv"].to_numpy())
    return out, {
        "fit_type": "strike_median",
        "strike_iv": {k: float(v) for k, v in strike_iv.items()},
        **metrics,
    }


def fit_quadratic_plus_strike_bias(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = df.copy()

    x = out["m"].to_numpy(dtype=float)
    y = out["raw_iv"].to_numpy(dtype=float)

    coeffs = np.polyfit(x, y, deg=2)
    base_fit = np.polyval(coeffs, x)

    out["base_fit_iv"] = base_fit
    out["base_residual"] = out["raw_iv"] - out["base_fit_iv"]

    strike_bias = out.groupby("product")["base_residual"].mean().to_dict()

    out["strike_bias"] = out["product"].map(strike_bias)
    out["fit_iv"] = out["base_fit_iv"] + out["strike_bias"]
    out["residual"] = out["raw_iv"] - out["fit_iv"]

    metrics = calc_metrics(out["raw_iv"].to_numpy(), out["fit_iv"].to_numpy())

    return out, {
        "fit_type": "quadratic_plus_strike_bias",
        "formula": "fair_iv = a*m^2 + b*m + c + strike_bias[product]",
        "coeffs": [float(v) for v in coeffs],
        "strike_bias": {k: float(v) for k, v in strike_bias.items()},
        **metrics,
    }


def by_product_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for product, g in df.groupby("product"):
        y = g["raw_iv"].to_numpy()
        pred = g["fit_iv"].to_numpy()
        m = calc_metrics(y, pred)

        rows.append(
            {
                "product": product,
                "n": len(g),
                "mean_raw_iv": float(g["raw_iv"].mean()),
                "mean_fit_iv": float(g["fit_iv"].mean()),
                "mean_residual": float(g["residual"].mean()),
                "rmse": m["rmse"],
                "mae": m["mae"],
                "r2": m["r2"],
            }
        )

    return pd.DataFrame(rows)


def plot_fit(df: pd.DataFrame, out_dir: Path, name: str) -> None:
    plt.figure(figsize=(11, 7))

    for product, g in df.groupby("product"):
        plt.scatter(g["m"], g["raw_iv"], s=8, alpha=0.35, label=product)

    if "base_fit_iv" in df.columns:
        xs = np.linspace(df["m"].min(), df["m"].max(), 500)
        coeffs = np.polyfit(df["m"], df["base_fit_iv"], deg=2)
        plt.plot(xs, np.polyval(coeffs, xs), linewidth=2, label="base quadratic")

    plt.scatter(df["m"], df["fit_iv"], s=4, alpha=0.25, label="fit_iv")

    plt.xlabel("m")
    plt.ylabel("raw_iv / fit_iv")
    plt.title(name)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / f"{name}_fit.png", dpi=160)
    plt.close()


def plot_residual_boxplot(df: pd.DataFrame, out_dir: Path, name: str) -> None:
    products = list(df["product"].drop_duplicates())
    data = [df.loc[df["product"] == p, "residual"].to_numpy() for p in products]

    plt.figure(figsize=(11, 6))
    plt.boxplot(data, labels=products, showfliers=False)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("raw_iv - fit_iv")
    plt.title(f"{name} residual by product")
    plt.tight_layout()
    plt.savefig(out_dir / f"{name}_residual_by_product.png", dpi=160)
    plt.close()


def run_model(df: pd.DataFrame, out_dir: Path, model_name: str):
    if model_name == "strike_mean":
        fitted, info = fit_strike_mean(df)
    elif model_name == "strike_median":
        fitted, info = fit_strike_median(df)
    elif model_name == "quadratic_bias":
        fitted, info = fit_quadratic_plus_strike_bias(df)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    fitted.to_csv(out_dir / f"{model_name}_points.csv", index=False)

    product_summary = by_product_metrics(fitted)
    product_summary.to_csv(out_dir / f"{model_name}_by_product.csv", index=False)

    (out_dir / f"{model_name}.json").write_text(
        json.dumps(info, indent=2),
        encoding="utf-8",
    )

    plot_fit(fitted, out_dir, model_name)
    plot_residual_boxplot(fitted, out_dir, model_name)

    print(f"\n========== {model_name} ==========")
    print(json.dumps(info, indent=2))

    print("\nBY PRODUCT")
    with pd.option_context("display.max_rows", 100, "display.width", 200):
        print(product_summary.to_string(index=False))


def main():
    repo_root = Path(__file__).resolve().parents[2]

    default_input = (
        repo_root
        / "jonghyeok"
        / "data_analysis"
        / "outputs"
        / "round3_iv_fit"
        / "round3_iv_points_with_fit.csv"
    )

    default_out_dir = (
        repo_root
        / "jonghyeok"
        / "data_analysis"
        / "outputs"
        / "round3_iv_by_strike"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--out-dir", type=Path, default=default_out_dir)
    parser.add_argument("--products", nargs="+", default=DEFAULT_PRODUCTS)
    parser.add_argument("--min-iv", type=float, default=0.1)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)

    df = df[df["product"].isin(args.products)].copy()
    df = df[df["raw_iv"] >= args.min_iv].copy()
    df = df[np.isfinite(df["raw_iv"]) & np.isfinite(df["m"])].copy()

    print(f"[INFO] input rows: {len(df)}")
    print(f"[INFO] products: {args.products}")

    run_model(df, args.out_dir, "strike_mean")
    run_model(df, args.out_dir, "strike_median")
    run_model(df, args.out_dir, "quadratic_bias")

    print(f"\n[SAVED] {args.out_dir}")


if __name__ == "__main__":
    main()