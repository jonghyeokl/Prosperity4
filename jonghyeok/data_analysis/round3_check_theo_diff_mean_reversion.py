# jonghyeok/data_analysis/round3_check_theo_diff_mean_reversion.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


EMA_WINDOWS = [
    15, 20, 25, 30, 35,
    40, 45, 50, 55, 60, 65
]

HORIZONS = [1, 2, 3, 4, 5]


def ema_prev(values: pd.Series, window: int) -> pd.Series:
    """
    t 시점 residual 계산에 t 이전 EMA만 사용.
    즉 lookahead 방지용.
    """
    alpha = 2.0 / (window + 1.0)

    out = []
    ema = None

    for v in values:
        out.append(ema)

        if ema is None:
            ema = v
        else:
            ema = alpha * v + (1.0 - alpha) * ema

    return pd.Series(out, index=values.index)


def regression_stats(x: np.ndarray, y: np.ndarray) -> dict:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    n = len(x)
    if n < 5:
        return {
            "n": n,
            "alpha": np.nan,
            "beta": np.nan,
            "corr": np.nan,
            "t_beta": np.nan,
            "r2": np.nan,
        }

    x_mean = x.mean()
    y_mean = y.mean()

    var_x = np.sum((x - x_mean) ** 2)
    cov_xy = np.sum((x - x_mean) * (y - y_mean))

    beta = cov_xy / var_x if var_x > 0 else np.nan
    alpha = y_mean - beta * x_mean

    y_hat = alpha + beta * x
    resid = y - y_hat

    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    corr = np.corrcoef(x, y)[0, 1] if n >= 2 else np.nan

    sigma2 = ss_res / max(n - 2, 1)
    se_beta = np.sqrt(sigma2 / var_x) if var_x > 0 else np.nan
    t_beta = beta / se_beta if se_beta and se_beta > 0 else np.nan

    return {
        "n": n,
        "alpha": alpha,
        "beta": beta,
        "corr": corr,
        "t_beta": t_beta,
        "r2": r2,
    }


def bucket_stats(g: pd.DataFrame, horizon: int) -> pd.DataFrame:
    col = f"future_change_{horizon}"

    tmp = g[["residual", col]].dropna().copy()
    if len(tmp) < 20:
        return pd.DataFrame()

    tmp["bucket"] = pd.qcut(tmp["residual"], q=5, labels=False, duplicates="drop")

    out = (
        tmp.groupby("bucket")
        .agg(
            residual_mean=("residual", "mean"),
            future_change_mean=(col, "mean"),
            future_change_median=(col, "median"),
            count=(col, "size"),
        )
        .reset_index()
    )

    return out


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    input_path = project_root / "jonghyeok" / "data_analysis" / "output" / "round3_theo_diff_5000_5300.csv"
    out_dir = project_root / "jonghyeok" / "data_analysis" / "output" / "theo_diff_mean_reversion"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df = df.sort_values(["product", "combined_timestamp"]).reset_index(drop=True)

    rows = []
    bucket_frames = []

    for product, base_g in df.groupby("product"):
        base_g = base_g.sort_values("combined_timestamp").copy()

        for ema_window in EMA_WINDOWS:
            g = base_g.copy()

            g["ema_prev"] = ema_prev(g["theo_diff"], ema_window)
            g["residual"] = g["theo_diff"] - g["ema_prev"]

            for h in HORIZONS:
                g[f"future_theo_diff_{h}"] = g["theo_diff"].shift(-h)
                g[f"future_change_{h}"] = g[f"future_theo_diff_{h}"] - g["theo_diff"]

                stats = regression_stats(
                    x=g["residual"].to_numpy(dtype=float),
                    y=g[f"future_change_{h}"].to_numpy(dtype=float),
                )

                rows.append(
                    {
                        "product": product,
                        "ema_window": ema_window,
                        "horizon": h,
                        **stats,
                        "mean_reversion_score": -stats["beta"] if np.isfinite(stats["beta"]) else np.nan,
                    }
                )

                b = bucket_stats(g, h)
                if not b.empty:
                    b.insert(0, "product", product)
                    b.insert(1, "ema_window", ema_window)
                    b.insert(2, "horizon", h)
                    bucket_frames.append(b)

    reg_df = pd.DataFrame(rows)

    reg_df = reg_df.sort_values(
        ["product", "horizon", "ema_window"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    bucket_df = pd.concat(bucket_frames, ignore_index=True) if bucket_frames else pd.DataFrame()

    if not bucket_df.empty:
        bucket_df = bucket_df.sort_values(
            ["product", "horizon", "ema_window", "bucket"],
            ascending=[True, True, True, True],
        ).reset_index(drop=True)

    reg_path = out_dir / "theo_diff_mean_reversion_regression_grid.csv"
    bucket_path = out_dir / "theo_diff_mean_reversion_buckets_grid.csv"

    reg_df.to_csv(reg_path, index=False)
    bucket_df.to_csv(bucket_path, index=False)

    print("\n=== Regression Summary ===")
    print(
        reg_df[
            [
                "product",
                "ema_window",
                "horizon",
                "n",
                "beta",
                "corr",
                "t_beta",
                "r2",
                "mean_reversion_score",
            ]
        ].to_string(index=False)
    )

    print("\n=== Best by product/horizon based on r2 ===")
    best = (
        reg_df.sort_values(["product", "horizon", "r2"], ascending=[True, True, False])
        .groupby(["product", "horizon"])
        .head(1)
        .reset_index(drop=True)
    )
    print(
        best[
            [
                "product",
                "horizon",
                "ema_window",
                "beta",
                "corr",
                "t_beta",
                "r2",
            ]
        ].to_string(index=False)
    )

    print("\n=== Best overall per product based on r2 ===")
    best_product = (
        reg_df.sort_values(["product", "r2"], ascending=[True, False])
        .groupby("product")
        .head(1)
        .reset_index(drop=True)
    )
    print(
        best_product[
            [
                "product",
                "ema_window",
                "horizon",
                "beta",
                "corr",
                "t_beta",
                "r2",
            ]
        ].to_string(index=False)
    )

    print("\n[SAVED]", reg_path)
    print("[SAVED]", bucket_path)


if __name__ == "__main__":
    main()