# jonghyeok/data_analysis/round3_check_theo_diff_mean_reversion_product_iv_curve.py

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


INPUT_FILENAME = "round3_theo_diff_5000_5300.csv"

PRODUCTS = ["VEV_5000", "VEV_5100", "VEV_5200", "VEV_5300"]

EMA_WINDOWS = [
    15, 20, 25, 30, 35,
    40, 45, 50, 55, 60, 65,
]

HORIZONS = [1, 2, 3, 4, 5]

DAYS_PER_YEAR = 365.0

IV_HISTORY_WINDOW = 300
IV_MIN_POINTS_FOR_FIT = 100

MIN_IV = 0.001
MAX_IV = 3.0


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_call_with_greeks(S: float, K: float, T: float, sigma: float):
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return max(S - K, 0.0), (1.0 if S > K else 0.0), 0.0

    sqrt_t = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t

    price = S * _norm_cdf(d1) - K * _norm_cdf(d2)
    delta = _norm_cdf(d1)
    vega = S * _norm_pdf(d1) * sqrt_t

    return price, delta, vega


def implied_vol(
    V: float,
    S: float,
    K: float,
    T: float,
    tol: float = 1e-4,
    max_iter: int = 60,
) -> Optional[float]:
    if T <= 0 or V <= 0 or S <= 0 or K <= 0:
        return None

    intrinsic = max(S - K, 0.0)

    if V < intrinsic - 1e-2 or V > S + 1e-2:
        return None

    lo = MIN_IV
    hi = MAX_IV

    f_lo = bs_call_with_greeks(S, K, T, lo)[0] - V
    f_hi = bs_call_with_greeks(S, K, T, hi)[0] - V

    if f_lo * f_hi > 0:
        return None

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = bs_call_with_greeks(S, K, T, mid)[0] - V

        if abs(f_mid) < tol:
            return mid

        if f_lo * f_mid < 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid

    return 0.5 * (lo + hi)


def fit_quadratic(points: list[tuple[float, float]]) -> Optional[tuple[float, float, float]]:
    if len(points) < IV_MIN_POINTS_FOR_FIT:
        return None

    arr = np.asarray(points, dtype=float)
    x = arr[:, 0]
    y = arr[:, 1]

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < IV_MIN_POINTS_FOR_FIT:
        return None

    try:
        a, b, c = np.polyfit(x, y, deg=2)
        return float(a), float(b), float(c)
    except Exception:
        return None


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


def recompute_theo_diff_with_product_iv_curve(df: pd.DataFrame) -> pd.DataFrame:
    """
    기존 df의 S_mid, option_mid, K, T_days를 사용.
    product별로만 rolling quadratic IV curve를 fit해서 theo_diff 재계산.

    중요:
    - t 시점 theo_diff 계산에는 t 이전 history만 사용.
    - 계산 후 현재 raw_iv를 product별 history에 추가.
    """
    rows = []

    for product, g in df.groupby("product", sort=True):
        g = g.sort_values("combined_timestamp").copy()
        hist: list[tuple[float, float]] = []

        for _, row in g.iterrows():
            S = float(row["S_mid"])
            K = float(row["K"])
            V = float(row["option_mid"])
            T = float(row["T_days"]) / DAYS_PER_YEAR

            if S <= 0 or K <= 0 or V <= 0 or T <= 0:
                continue

            m = math.log(K / S) / math.sqrt(T)

            raw_iv = implied_vol(V=V, S=S, K=K, T=T)

            # 1. 현재 시점 이전 history로 product별 IV curve fitting
            coeffs = fit_quadratic(hist)

            if coeffs is not None:
                a, b, c = coeffs
                fair_iv = a * m * m + b * m + c

                if fair_iv > 0 and math.isfinite(fair_iv):
                    theo, delta, vega = bs_call_with_greeks(S, K, T, fair_iv)
                    theo_diff = V - theo

                    new_row = row.to_dict()
                    new_row["m"] = m
                    new_row["raw_iv"] = raw_iv
                    new_row["product_iv_a"] = a
                    new_row["product_iv_b"] = b
                    new_row["product_iv_c"] = c
                    new_row["fair_iv"] = fair_iv
                    new_row["theo"] = theo
                    new_row["delta"] = delta
                    new_row["vega"] = vega
                    new_row["theo_diff"] = theo_diff
                    new_row["iv_curve_type"] = "product_rolling_quadratic"
                    rows.append(new_row)

            # 2. 현재 raw_iv를 이후 timestamp용 history에 추가
            if raw_iv is not None and MIN_IV <= raw_iv <= MAX_IV and math.isfinite(raw_iv):
                hist.append((m, raw_iv))
                if len(hist) > IV_HISTORY_WINDOW:
                    hist = hist[-IV_HISTORY_WINDOW:]

    out = pd.DataFrame(rows)

    if out.empty:
        raise RuntimeError("product별 IV curve로 theo_diff를 만들 수 없습니다.")

    out = out.sort_values(["product", "combined_timestamp"]).reset_index(drop=True)
    return out


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    input_path = (
        project_root
        / "jonghyeok"
        / "data_analysis"
        / "output"
        / INPUT_FILENAME
    )

    out_dir = (
        project_root
        / "jonghyeok"
        / "data_analysis"
        / "output"
        / "theo_diff_mean_reversion_product_iv_curve"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df = df[df["product"].isin(PRODUCTS)].copy()

    required = {
        "product",
        "combined_timestamp",
        "day_num",
        "timestamp",
        "K",
        "S_mid",
        "option_mid",
        "T_days",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}")

    for col in [
        "combined_timestamp",
        "day_num",
        "timestamp",
        "K",
        "S_mid",
        "option_mid",
        "T_days",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=[
            "product",
            "combined_timestamp",
            "day_num",
            "timestamp",
            "K",
            "S_mid",
            "option_mid",
            "T_days",
        ]
    )

    df = df.sort_values(["product", "combined_timestamp"]).reset_index(drop=True)

    # 핵심 변경: theo_diff를 product별 IV curve 기준으로 다시 계산
    df = recompute_theo_diff_with_product_iv_curve(df)

    points_path = out_dir / "round3_theo_diff_product_iv_curve_points.csv"
    df.to_csv(points_path, index=False)

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

    reg_path = out_dir / "theo_diff_mean_reversion_product_iv_curve_regression_grid.csv"
    bucket_path = out_dir / "theo_diff_mean_reversion_product_iv_curve_buckets_grid.csv"

    reg_df.to_csv(reg_path, index=False)
    bucket_df.to_csv(bucket_path, index=False)

    print("\n=== Product IV Curve Mean Reversion Summary ===")
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

    print("\n[SAVED]", points_path)
    print("[SAVED]", reg_path)
    print("[SAVED]", bucket_path)


if __name__ == "__main__":
    main()