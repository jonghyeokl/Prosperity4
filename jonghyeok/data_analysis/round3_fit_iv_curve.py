# jonghyeok/data_analysis/fit_round3_iv_curve.py

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistics import NormalDist


N = NormalDist()

VALID_BID_ASK_VOLUME = {
    "VELVETFRUIT_EXTRACT": 15,
    "VEV_4000": 6,
    "VEV_4500": 6,
    "VEV_5000": 6,
    "VEV_5100": 6,
    "VEV_5200": 6,
    "VEV_5300": 5,
    "VEV_5400": 5,
    "VEV_5500": 5,
    "VEV_6000": 5,
    "VEV_6500": 5,
}

UNDERLYING = "VELVETFRUIT_EXTRACT"

VOUCHERS = [
    "VEV_5000",
    "VEV_5100",
    "VEV_5200",
    "VEV_5300",
    "VEV_5400",
    "VEV_5500",
    "VEV_6000",
    "VEV_6500",
]

STRIKES = {
    "VEV_5000": 5000,
    "VEV_5100": 5100,
    "VEV_5200": 5200,
    "VEV_5300": 5300,
    "VEV_5400": 5400,
    "VEV_5500": 5500,
    # "VEV_6000": 6000,
    # "VEV_6500": 6500,
}

DAYS_PER_YEAR = 365.0


@dataclass
class FitResult:
    coeffs: np.ndarray
    rmse: float
    mae: float
    r2: float
    n: int


def read_csv_auto(path: Path) -> pd.DataFrame:
    """Prosperity csv는 보통 ';' 구분자지만, 혹시 몰라 fallback."""
    df = pd.read_csv(path, sep=";")
    if len(df.columns) == 1:
        df = pd.read_csv(path)
    return df


def parse_day_from_filename(path: Path) -> int:
    m = re.search(r"day_(-?\d+)", path.name)
    if not m:
        raise ValueError(f"Cannot parse day from filename: {path.name}")
    return int(m.group(1))


def get_valid_mid(row: pd.Series) -> float | None:
    valid_bid = None
    valid_ask = None
    valid_volume = VALID_BID_ASK_VOLUME[row["product"]]
    for i in range(1, 4):
        bid_price = row.get(f"bid_price_{i}", np.nan)
        bid_vol = row.get(f"bid_volume_{i}", np.nan)
        if pd.notna(bid_price) and pd.notna(bid_vol):
            if bid_vol >= valid_volume:
                valid_bid = bid_price
                break
    for i in range(1, 4):
        ask_price = row.get(f"ask_price_{i}", np.nan)
        ask_vol = row.get(f"ask_volume_{i}", np.nan)
        if pd.notna(ask_price) and pd.notna(ask_vol):
            if ask_vol >= valid_volume:
                valid_ask = ask_price
                break
    
    if valid_bid is None:
        valid_bid = row.get("bid_price_1", np.nan)
    if valid_ask is None:
        valid_ask = row.get("ask_price_1", np.nan)
    
    if valid_bid is None or valid_ask is None:
        return None
    return 0.5 * (valid_bid + valid_ask)


def norm_cdf(x: float) -> float:
    return N.cdf(x)


def bs_call_price(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return max(S - K, 0.0)

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)


def implied_vol_bisect(
    S: float,
    K: float,
    T: float,
    price: float,
    r: float = 0.0,
    lo: float = 1e-6,
    hi: float = 5.0,
    max_iter: int = 80,
    tol: float = 1e-8,
) -> float | None:
    """
    BS(S,K,T,sigma)=price 를 만족하는 sigma를 이분탐색으로 찾음.
    no-arbitrage 범위 밖이면 None.
    """
    if S <= 0 or K <= 0 or T <= 0 or price <= 0:
        return None

    intrinsic = max(S - K * math.exp(-r * T), 0.0)

    # call price는 intrinsic 이상, S 이하가 정상 범위.
    if price < intrinsic - 1e-6:
        return None
    if price > S + 1e-6:
        return None

    price_lo = bs_call_price(S, K, T, lo, r)
    price_hi = bs_call_price(S, K, T, hi, r)

    # hi=5로도 price를 못 맞추면 너무 높은 IV/outlier로 보고 제외.
    if price < price_lo - 1e-6 or price > price_hi + 1e-6:
        return None

    left, right = lo, hi
    for _ in range(max_iter):
        mid = 0.5 * (left + right)
        p = bs_call_price(S, K, T, mid, r)

        if abs(p - price) < tol:
            return mid

        if p < price:
            left = mid
        else:
            right = mid

    return 0.5 * (left + right)


def timestamp_progress(timestamp: float) -> float:
    """
    Prosperity timestamp는 보통 0, 100, ..., 999900.
    하루 진행률을 약 0~1로 환산.
    """
    return float(timestamp) / 1_000_000.0


def tte_years(tte_start_days: float, timestamp: float) -> float:
    """
    historical day 시작 시점의 TTE에서 하루 진행분을 뺀 값.
    예: day1 시작 8d, day 끝 근처 7d.
    """
    remaining_days = tte_start_days - timestamp_progress(timestamp)
    return max(remaining_days / DAYS_PER_YEAR, 1e-9)


def build_mid_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        product = row["product"]
        if product != UNDERLYING and product not in STRIKES:
            continue

        mid = get_valid_mid(row)
        if mid is None or not np.isfinite(mid):
            continue

        bid = row.get("bid_price_1", np.nan)
        ask = row.get("ask_price_1", np.nan)

        rows.append(
            {
                "timestamp": int(row["timestamp"]),
                "product": product,
                "mid": float(mid),
                "bid": float(bid) if pd.notna(bid) else np.nan,
                "ask": float(ask) if pd.notna(ask) else np.nan,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["timestamp", "product", "mid", "bid", "ask"])

    return pd.DataFrame(rows)


def collect_iv_points(data_dir: Path) -> pd.DataFrame:
    price_files = sorted(data_dir.glob("prices_round_3_day_*.csv"), key=parse_day_from_filename)

    if not price_files:
        raise FileNotFoundError(f"No prices_round_3_day_*.csv found in {data_dir}")

    all_points = []

    # 파일명 day가 -2,-1,0이든 1,2,3이든, 정렬된 historical 3일을 8d,7d,6d로 매핑.
    tte_start_by_file_idx = {0: 8.0, 1: 7.0, 2: 6.0}

    for file_idx, path in enumerate(price_files):
        day = parse_day_from_filename(path)
        tte_start = tte_start_by_file_idx.get(file_idx, 8.0 - file_idx)

        print(f"[LOAD] {path.name}: parsed_day={day}, tte_start={tte_start}d")

        raw = read_csv_auto(path)
        mids = build_mid_table(raw)

        if mids.empty:
            print(f"[WARN] no valid mid rows in {path.name}")
            continue

        underlying = mids[mids["product"] == UNDERLYING][["timestamp", "mid"]]
        underlying = underlying.rename(columns={"mid": "S"})

        for voucher, K in STRIKES.items():
            opt = mids[mids["product"] == voucher][["timestamp", "mid", "bid", "ask"]]
            if opt.empty:
                print(f"[WARN] no rows for {voucher} in {path.name}")
                continue

            opt = opt.rename(columns={"mid": "V", "bid": "opt_bid", "ask": "opt_ask"})
            joined = opt.merge(underlying, on="timestamp", how="inner")

            for _, r in joined.iterrows():
                ts = float(r["timestamp"])
                S = float(r["S"])
                V = float(r["V"])
                T = tte_years(tte_start, ts)

                # 너무 말도 안 되는 가격 제외
                intrinsic = max(S - K, 0.0)
                if V < intrinsic - 1e-6:
                    continue

                iv = implied_vol_bisect(S=S, K=K, T=T, price=V)
                if iv is None or not np.isfinite(iv):
                    continue

                m = math.log(K / S) / math.sqrt(T)

                all_points.append(
                    {
                        "file": path.name,
                        "parsed_day": day,
                        "file_idx": file_idx,
                        "timestamp": int(ts),
                        "product": voucher,
                        "K": K,
                        "S": S,
                        "V": V,
                        "T_years": T,
                        "T_days": T * DAYS_PER_YEAR,
                        "m": m,
                        "raw_iv": iv,
                        "intrinsic": intrinsic,
                        "spread": (
                            float(r["opt_ask"]) - float(r["opt_bid"])
                            if pd.notna(r["opt_ask"]) and pd.notna(r["opt_bid"])
                            else np.nan
                        ),
                    }
                )

    points = pd.DataFrame(all_points)
    if points.empty:
        raise RuntimeError("No valid IV points collected.")

    return points


def filter_points(points: pd.DataFrame, min_iv: float, max_iv: float, max_abs_m: float | None) -> pd.DataFrame:
    df = points.copy()

    df = df[np.isfinite(df["raw_iv"])]
    df = df[(df["raw_iv"] >= min_iv) & (df["raw_iv"] <= max_iv)]

    if max_abs_m is not None:
        df = df[df["m"].abs() <= max_abs_m]

    return df.reset_index(drop=True)


def fit_quadratic(df: pd.DataFrame, weighted: bool) -> FitResult:
    x = df["m"].to_numpy(dtype=float)
    y = df["raw_iv"].to_numpy(dtype=float)

    if weighted:
        # spread가 좁고 ATM에 가까울수록 큰 weight.
        spread = df["spread"].to_numpy(dtype=float)
        spread = np.where(np.isfinite(spread) & (spread > 0), spread, np.nanmedian(spread[np.isfinite(spread)]))
        spread_weight = 1.0 / np.maximum(spread, 1.0)

        atm_weight = 1.0 / (1.0 + np.abs(x))
        w = spread_weight * atm_weight

        coeffs = np.polyfit(x, y, deg=2, w=w)
    else:
        coeffs = np.polyfit(x, y, deg=2)

    pred = np.polyval(coeffs, x)
    resid = y - pred

    rmse = float(np.sqrt(np.mean(resid * resid)))
    mae = float(np.mean(np.abs(resid)))
    ss_res = float(np.sum(resid * resid))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return FitResult(coeffs=coeffs, rmse=rmse, mae=mae, r2=r2, n=len(df))


def add_predictions(df: pd.DataFrame, coeffs: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    out["fit_iv"] = np.polyval(coeffs, out["m"].to_numpy(dtype=float))
    out["residual"] = out["raw_iv"] - out["fit_iv"]
    return out


def print_metrics(df: pd.DataFrame, fit: FitResult) -> None:
    a, b, c = fit.coeffs

    print("\n========== GLOBAL FIT ==========")
    print(f"fair_iv(m) = {a:.10f} * m^2 + {b:.10f} * m + {c:.10f}")
    print(f"coeffs = [{a:.10f}, {b:.10f}, {c:.10f}]")
    print(f"N    = {fit.n}")
    print(f"RMSE = {fit.rmse:.8f}")
    print(f"MAE  = {fit.mae:.8f}")
    print(f"R^2  = {fit.r2:.8f}")

    print("\n========== BY PRODUCT ==========")
    by_product = (
        df.groupby("product")
        .agg(
            n=("raw_iv", "size"),
            mean_iv=("raw_iv", "mean"),
            mean_fit_iv=("fit_iv", "mean"),
            mean_resid=("residual", "mean"),
            mae=("residual", lambda x: float(np.mean(np.abs(x)))),
            rmse=("residual", lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2)))),
            min_m=("m", "min"),
            max_m=("m", "max"),
        )
        .reset_index()
    )

    with pd.option_context("display.max_rows", 50, "display.width", 200):
        print(by_product.to_string(index=False))


def save_outputs(df: pd.DataFrame, fit: FitResult, out_dir: Path, weighted: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    a, b, c = [float(x) for x in fit.coeffs]

    payload = {
        "fit_type": "weighted_quadratic" if weighted else "quadratic",
        "formula": "fair_iv(m) = a*m^2 + b*m + c",
        "a": a,
        "b": b,
        "c": c,
        "coeffs": [a, b, c],
        "rmse": fit.rmse,
        "mae": fit.mae,
        "r2": fit.r2,
        "n": fit.n,
        "underlying": UNDERLYING,
        "strikes": STRIKES,
    }

    (out_dir / "round3_iv_fit_coeffs.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    df.to_csv(out_dir / "round3_iv_points_with_fit.csv", index=False)

    # Plot 1: raw IV scatter + fitted curve
    xs = np.linspace(df["m"].min(), df["m"].max(), 500)
    ys = np.polyval(fit.coeffs, xs)

    plt.figure(figsize=(11, 7))
    for product, g in df.groupby("product"):
        plt.scatter(g["m"], g["raw_iv"], s=8, alpha=0.35, label=product)
    plt.plot(xs, ys, linewidth=2.5, label="quadratic fit")
    plt.xlabel("m = log(K / S) / sqrt(TTE)")
    plt.ylabel("raw implied volatility")
    plt.title("Round 3 VEV IV Smile Fit")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "round3_iv_fit_scatter.png", dpi=160)
    plt.close()

    # Plot 2: residual by m
    plt.figure(figsize=(11, 6))
    plt.scatter(df["m"], df["residual"], s=8, alpha=0.35)
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("m")
    plt.ylabel("raw_iv - fit_iv")
    plt.title("IV Fit Residuals")
    plt.tight_layout()
    plt.savefig(out_dir / "round3_iv_fit_residuals.png", dpi=160)
    plt.close()

    # Plot 3: residual by product boxplot
    plt.figure(figsize=(11, 6))
    products = [p for p in VOUCHERS if p in set(df["product"])]
    data = [df.loc[df["product"] == p, "residual"].to_numpy() for p in products]
    plt.boxplot(data, labels=products, showfliers=False)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("raw_iv - fit_iv")
    plt.title("Residual Distribution by Voucher")
    plt.tight_layout()
    plt.savefig(out_dir / "round3_iv_fit_residuals_by_product.png", dpi=160)
    plt.close()

    print(f"\n[SAVED] {out_dir / 'round3_iv_fit_coeffs.json'}")
    print(f"[SAVED] {out_dir / 'round3_iv_points_with_fit.csv'}")
    print(f"[SAVED] {out_dir / 'round3_iv_fit_scatter.png'}")
    print(f"[SAVED] {out_dir / 'round3_iv_fit_residuals.png'}")
    print(f"[SAVED] {out_dir / 'round3_iv_fit_residuals_by_product.png'}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    default_data_dir = repo_root / "data_capsule" / "round3"
    default_out_dir = Path(__file__).resolve().parent / "outputs" / "round3_iv_fit"

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=default_data_dir)
    parser.add_argument("--out-dir", type=Path, default=default_out_dir)
    parser.add_argument("--min-iv", type=float, default=0.001)
    parser.add_argument("--max-iv", type=float, default=2.0)
    parser.add_argument("--max-abs-m", type=float, default=None)
    parser.add_argument("--weighted", action="store_true")
    args = parser.parse_args()

    points = collect_iv_points(args.data_dir)
    print(f"\n[INFO] collected raw points: {len(points)}")

    filtered = filter_points(
        points,
        min_iv=args.min_iv,
        max_iv=args.max_iv,
        max_abs_m=args.max_abs_m,
    )
    print(f"[INFO] after filtering: {len(filtered)}")

    if len(filtered) < 20:
        raise RuntimeError(f"Too few points after filtering: {len(filtered)}")

    fit = fit_quadratic(filtered, weighted=args.weighted)
    fitted = add_predictions(filtered, fit.coeffs)

    print_metrics(fitted, fit)
    save_outputs(fitted, fit, args.out_dir, weighted=args.weighted)


if __name__ == "__main__":
    main()