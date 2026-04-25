from __future__ import annotations

import math
import re
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


N = NormalDist()

UNDERLYING = "VELVETFRUIT_EXTRACT"

STRIKES = {
    "VEV_4000": 4000,
    "VEV_4500": 4500,
    "VEV_5000": 5000,
    "VEV_5100": 5100,
    "VEV_5200": 5200,
    "VEV_5300": 5300,
    "VEV_5400": 5400,
    "VEV_5500": 5500,
    "VEV_6000": 6000,
    "VEV_6500": 6500,
}

DAYS_PER_YEAR = 365.0
DAY_TIMESTAMP_SCALE = 1_000_000


def read_csv_auto(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    if len(df.columns) == 1:
        df = pd.read_csv(path)
    return df


def parse_day(path: Path) -> int:
    m = re.search(r"day_(-?\d+)", path.name)
    if not m:
        raise ValueError(f"Cannot parse day from filename: {path.name}")
    return int(m.group(1))


def get_mid(row: pd.Series) -> float | None:
    mid = row.get("mid_price", np.nan)
    if pd.notna(mid):
        return float(mid)

    bid = row.get("bid_price_1", np.nan)
    ask = row.get("ask_price_1", np.nan)
    if pd.notna(bid) and pd.notna(ask):
        return 0.5 * (float(bid) + float(ask))

    return None


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
    lo: float = 1e-6,
    hi: float = 5.0,
    max_iter: int = 80,
) -> float | None:
    if S <= 0 or K <= 0 or T <= 0 or price <= 0:
        return None

    intrinsic = max(S - K, 0.0)
    if price < intrinsic - 1e-6 or price > S + 1e-6:
        return None

    price_lo = bs_call_price(S, K, T, lo)
    price_hi = bs_call_price(S, K, T, hi)

    if price < price_lo - 1e-6 or price > price_hi + 1e-6:
        return None

    left, right = lo, hi
    for _ in range(max_iter):
        mid = 0.5 * (left + right)
        p = bs_call_price(S, K, T, mid)

        if p < price:
            left = mid
        else:
            right = mid

    return 0.5 * (left + right)


def tte_years(day_num: int, timestamp: int) -> float:
    # day 0 -> 8d, day 1 -> 7d, day 2 -> 6d
    tte_start_days = 8.0 - day_num
    progress_days = timestamp / DAY_TIMESTAMP_SCALE
    remaining_days = max(tte_start_days - progress_days, 1e-9)
    return remaining_days / DAYS_PER_YEAR


def collect_raw_iv(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob("prices_round_3_day_*.csv"), key=parse_day)
    rows = []

    for path in files:
        day_num = parse_day(path)
        print(f"[LOAD] {path.name}, day={day_num}")

        df = read_csv_auto(path)

        mids = []
        for _, row in df.iterrows():
            product = row["product"]
            if product != UNDERLYING and product not in STRIKES:
                continue

            mid = get_mid(row)
            if mid is None or not np.isfinite(mid):
                continue

            mids.append(
                {
                    "timestamp": int(row["timestamp"]),
                    "global_timestamp": day_num * DAY_TIMESTAMP_SCALE + int(row["timestamp"]),
                    "product": product,
                    "mid": mid,
                }
            )

        mid_df = pd.DataFrame(mids)
        if mid_df.empty:
            continue

        underlying = (
            mid_df[mid_df["product"] == UNDERLYING][["timestamp", "mid"]]
            .rename(columns={"mid": "S"})
        )

        for voucher, K in STRIKES.items():
            opt = (
                mid_df[mid_df["product"] == voucher][["timestamp", "global_timestamp", "mid"]]
                .rename(columns={"mid": "V"})
            )

            joined = opt.merge(underlying, on="timestamp", how="inner")

            for _, r in joined.iterrows():
                ts = int(r["timestamp"])
                global_ts = int(r["global_timestamp"])
                S = float(r["S"])
                V = float(r["V"])
                T = tte_years(day_num, ts)

                iv = implied_vol_bisect(S, K, T, V)
                if iv is None or not np.isfinite(iv):
                    continue

                rows.append(
                    {
                        "day": day_num,
                        "timestamp": ts,
                        "global_timestamp": global_ts,
                        "product": voucher,
                        "K": K,
                        "S": S,
                        "V": V,
                        "T_days": T * DAYS_PER_YEAR,
                        "raw_iv": iv,
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data_capsule" / "round3"
    out_dir = Path(__file__).resolve().parent / "outputs" / "round3_raw_iv_timeseries"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = collect_raw_iv(data_dir)
    df.to_csv(out_dir / "round3_raw_iv_timeseries.csv", index=False)

    plt.figure(figsize=(15, 8))

    for product in STRIKES:
        g = df[df["product"] == product].sort_values("global_timestamp")
        if g.empty:
            continue

        plt.plot(
            g["global_timestamp"],
            g["raw_iv"],
            linewidth=1.0,
            alpha=0.85,
            label=product,
        )

    plt.axvline(1_000_000, linestyle="--", linewidth=1)
    plt.axvline(2_000_000, linestyle="--", linewidth=1)

    plt.xlabel("global timestamp")
    plt.ylabel("raw implied volatility")
    plt.title("Round 3 Raw Implied Volatility by Voucher")
    plt.legend(ncol=2)
    plt.tight_layout()

    out_path = out_dir / "round3_raw_iv_timeseries.png"
    plt.savefig(out_path, dpi=180)
    plt.close()

    print(f"[SAVED] {out_dir / 'round3_raw_iv_timeseries.csv'}")
    print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()