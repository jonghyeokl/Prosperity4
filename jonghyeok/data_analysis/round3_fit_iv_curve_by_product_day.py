from __future__ import annotations

import argparse
import json
import math
import re
import warnings
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

STRIKES = {
    "VEV_5000": 5000,
    "VEV_5100": 5100,
    "VEV_5200": 5200,
    "VEV_5300": 5300,
    "VEV_5400": 5400,
    "VEV_5500": 5500,
}

VOUCHERS = list(STRIKES.keys())

DAYS_PER_YEAR = 365.0


@dataclass
class FitResult:
    coeffs: np.ndarray
    rmse: float
    mae: float
    r2: float
    n: int


def read_csv_auto(path: Path) -> pd.DataFrame:
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
    if pd.isna(valid_bid) or pd.isna(valid_ask):
        return None

    return 0.5 * (float(valid_bid) + float(valid_ask))


def norm_cdf(x: float) -> float:
    return N.cdf(x)


def bs_call_price(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return max(S - K, 0.0)

    sqrt_t = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t

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
    if S <= 0 or K <= 0 or T <= 0 or price <= 0:
        return None

    intrinsic = max(S - K * math.exp(-r * T), 0.0)

    if price < intrinsic - 1e-6:
        return None
    if price > S + 1e-6:
        return None

    price_lo = bs_call_price(S, K, T, lo, r)
    price_hi = bs_call_price(S, K, T, hi, r)

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
    return float(timestamp) / 1_000_000.0


def tte_years(tte_start_days: float, timestamp: float) -> float:
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
    tte_start_by_file_idx = {0: 8.0, 1: 7.0, 2: 6.0}

    for file_idx, path in enumerate(price_files):
        day = parse_day_from_filename(path)
        tte_start = tte_start_by_file_idx.get(file_idx, 8.0 - file_idx)

        print(f"[LOAD] {path.name}: parsed_day={day}, file_idx={file_idx}, tte_start={tte_start}d")

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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if weighted:
            spread = df["spread"].to_numpy(dtype=float)
            finite_spread = spread[np.isfinite(spread) & (spread > 0)]
            fallback_spread = np.nanmedian(finite_spread) if len(finite_spread) > 0 else 1.0
            spread = np.where(np.isfinite(spread) & (spread > 0), spread, fallback_spread)

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


def fit_by_product_day(filtered: pd.DataFrame, weighted: bool) -> tuple[pd.DataFrame, dict]:
    fitted_frames = []
    results = {}

    for product, pg in filtered.groupby("product"):
        results[product] = {}

        for file_idx, g in pg.groupby("file_idx"):
            if len(g) < 20:
                print(f"[SKIP] {product}, file_idx={file_idx}: too few points {len(g)}")
                continue

            fit = fit_quadratic(g, weighted=weighted)
            fitted = add_predictions(g, fit.coeffs)
            fitted_frames.append(fitted)

            a, b, c = [float(x) for x in fit.coeffs]
            parsed_day = int(g["parsed_day"].iloc[0])
            file_name = str(g["file"].iloc[0])
            day_key = f"day_{parsed_day}"

            results[product][day_key] = {
                "file_idx": int(file_idx),
                "parsed_day": parsed_day,
                "file": file_name,
                "fit_type": "weighted_quadratic_by_product_day" if weighted else "quadratic_by_product_day",
                "formula": "fair_iv_product_day(m) = a*m^2 + b*m + c",
                "product": product,
                "K": int(g["K"].iloc[0]),
                "a": a,
                "b": b,
                "c": c,
                "coeffs": [a, b, c],
                "rmse": fit.rmse,
                "mae": fit.mae,
                "r2": fit.r2,
                "n": fit.n,
            }

            print(f"\n========== {product} / {day_key} ==========")
            print(f"fair_iv(m) = {a:.10f} * m^2 + {b:.10f} * m + {c:.10f}")
            print(f"N    = {fit.n}")
            print(f"RMSE = {fit.rmse:.8f}")
            print(f"MAE  = {fit.mae:.8f}")
            print(f"R^2  = {fit.r2:.8f}")

    if not fitted_frames:
        raise RuntimeError("No product-day fits produced.")

    return pd.concat(fitted_frames, ignore_index=True), results


def safe_name(s: str) -> str:
    return s.replace("/", "_").replace(" ", "_")


def save_outputs(fitted: pd.DataFrame, results: dict, out_dir: Path, weighted: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "fit_type": "weighted_quadratic_by_product_day" if weighted else "quadratic_by_product_day",
        "formula": "fair_iv_product_day(m) = a*m^2 + b*m + c",
        "underlying": UNDERLYING,
        "strikes": STRIKES,
        "products": results,
    }

    coeff_path = out_dir / "round3_iv_fit_coeffs_by_product_day.json"
    points_path = out_dir / "round3_iv_points_with_fit_by_product_day.csv"

    coeff_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    fitted.to_csv(points_path, index=False)

    # 1. product별: day fitting curve 비교
    for product, day_infos in results.items():
        product_df = fitted[fitted["product"] == product].copy()
        if product_df.empty:
            continue

        plt.figure(figsize=(11, 7))

        for day_key, info in sorted(day_infos.items(), key=lambda x: x[1]["file_idx"]):
            day_df = product_df[product_df["file_idx"] == info["file_idx"]]
            if day_df.empty:
                continue

            coeffs = np.array(info["coeffs"], dtype=float)
            xs = np.linspace(day_df["m"].min(), day_df["m"].max(), 400)
            ys = np.polyval(coeffs, xs)

            label = (
                f"{day_key}: "
                f"a={coeffs[0]:.4f}, b={coeffs[1]:.4f}, c={coeffs[2]:.4f}"
            )
            plt.plot(xs, ys, linewidth=2.5, label=label)

        plt.xlabel("m = log(K / S) / sqrt(TTE)")
        plt.ylabel("fitted fair IV")
        plt.title(f"Round 3 Quadratic IV Curves by Day - {product}")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / f"round3_iv_fit_curves_by_day_{safe_name(product)}.png", dpi=180)
        plt.close()

    # 2. product-day별 scatter + fitted curve
    for product, day_infos in results.items():
        for day_key, info in day_infos.items():
            g = fitted[
                (fitted["product"] == product)
                & (fitted["file_idx"] == info["file_idx"])
            ].copy()

            if g.empty:
                continue

            coeffs = np.array(info["coeffs"], dtype=float)
            xs = np.linspace(g["m"].min(), g["m"].max(), 400)
            ys = np.polyval(coeffs, xs)

            plt.figure(figsize=(11, 7))
            plt.scatter(g["m"], g["raw_iv"], s=8, alpha=0.35, label=f"{product} raw_iv")
            plt.plot(xs, ys, linewidth=2.5, label=f"{product} {day_key} quadratic fit")
            plt.xlabel("m = log(K / S) / sqrt(TTE)")
            plt.ylabel("raw implied volatility")
            plt.title(f"Round 3 IV Fit - {product} / {day_key}")
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(out_dir / f"round3_iv_fit_scatter_{safe_name(product)}_{day_key}.png", dpi=160)
            plt.close()

            plt.figure(figsize=(11, 6))
            plt.scatter(g["m"], g["residual"], s=8, alpha=0.35)
            plt.axhline(0.0, linewidth=1)
            plt.xlabel("m")
            plt.ylabel("raw_iv - fit_iv")
            plt.title(f"IV Fit Residuals - {product} / {day_key}")
            plt.tight_layout()
            plt.savefig(out_dir / f"round3_iv_fit_residuals_{safe_name(product)}_{day_key}.png", dpi=160)
            plt.close()

    # 3. residual boxplot: product-day
    labels = []
    data = []

    for product in VOUCHERS:
        if product not in results:
            continue

        for day_key, info in sorted(results[product].items(), key=lambda x: x[1]["file_idx"]):
            g = fitted[
                (fitted["product"] == product)
                & (fitted["file_idx"] == info["file_idx"])
            ]
            if g.empty:
                continue
            labels.append(f"{product}\n{day_key}")
            data.append(g["residual"].to_numpy())

    if data:
        plt.figure(figsize=(14, 6))
        plt.boxplot(data, labels=labels, showfliers=False)
        plt.axhline(0.0, linewidth=1)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("raw_iv - fit_iv")
        plt.title("Residual Distribution by Product-Day")
        plt.tight_layout()
        plt.savefig(out_dir / "round3_iv_fit_residuals_by_product_day.png", dpi=160)
        plt.close()

    print(f"\n[SAVED] {coeff_path}")
    print(f"[SAVED] {points_path}")
    print(f"[SAVED] product-day comparison plots in {out_dir}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    default_data_dir = repo_root / "data_capsule" / "round3"
    default_out_dir = Path(__file__).resolve().parent / "outputs" / "round3_iv_fit_by_product_day"

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

    fitted, results = fit_by_product_day(filtered, weighted=args.weighted)
    save_outputs(fitted, results, args.out_dir, weighted=args.weighted)


if __name__ == "__main__":
    main()