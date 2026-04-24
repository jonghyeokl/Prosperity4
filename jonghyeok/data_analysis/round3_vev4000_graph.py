from __future__ import annotations

import math
import re
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio


N = NormalDist()

UNDERLYING = "VELVETFRUIT_EXTRACT"
PRODUCT = "VEV_4000"
K = 4000

DAYS_PER_YEAR = 365.0
DAY_LENGTH = 1_000_000
IV_THRESHOLD = 0.4

VALID_BID_ASK_VOLUME = {
    "VELVETFRUIT_EXTRACT": 30,
    "VEV_4000": 5,
}

DAY_RE = re.compile(r"day_(-?\d+)")


def find_project_root(start: Path) -> Path:
    start = start.resolve()
    for candidate in [start.parent, *start.parents]:
        if (candidate / "data_capsule").exists() and (candidate / "jonghyeok").exists():
            return candidate
    return start.parents[2]


def parse_day_from_filename(path: Path) -> int:
    m = DAY_RE.search(path.name)
    if not m:
        raise ValueError(f"Cannot parse day from filename: {path.name}")
    return int(m.group(1))


def read_csv_auto(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    if len(df.columns) == 1:
        df = pd.read_csv(path)
    return df


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


def bs_call_price(S: float, K: float, T: float, sigma: float) -> float:
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return max(S - K, 0.0)

    sqrt_t = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return S * norm_cdf(d1) - K * norm_cdf(d2)


def implied_vol_bisect(
    S: float,
    K: float,
    T: float,
    price: float,
    lo: float = 1e-6,
    hi: float = 5.0,
    max_iter: int = 80,
    tol: float = 1e-8,
) -> float | None:
    if S <= 0 or K <= 0 or T <= 0 or price <= 0:
        return None

    intrinsic = max(S - K, 0.0)

    if price < intrinsic - 1e-6:
        return None
    if price > S + 1e-6:
        return None

    price_lo = bs_call_price(S, K, T, lo)
    price_hi = bs_call_price(S, K, T, hi)

    if price < price_lo - 1e-6 or price > price_hi + 1e-6:
        return None

    left, right = lo, hi

    for _ in range(max_iter):
        mid = 0.5 * (left + right)
        p = bs_call_price(S, K, T, mid)

        if abs(p - price) < tol:
            return mid

        if p < price:
            left = mid
        else:
            right = mid

    return 0.5 * (left + right)


def tte_years(day: int, timestamp: int) -> float:
    # day 0,1,2 시작 TTE = 8d,7d,6d
    tte_start_days = 8.0 - day
    progress_days = timestamp / DAY_LENGTH
    remaining_days = max(tte_start_days - progress_days, 1e-9)
    return remaining_days / DAYS_PER_YEAR


def collect_points(data_dir: Path) -> pd.DataFrame:
    price_files = sorted(data_dir.glob("prices_round_3_day_*.csv"), key=parse_day_from_filename)
    rows = []

    for path in price_files:
        day = parse_day_from_filename(path)
        if day not in {0, 1, 2}:
            continue

        print(f"[LOAD] {path.name}")

        raw = read_csv_auto(path)
        tmp = []

        for _, row in raw.iterrows():
            product = row["product"]
            if product not in {UNDERLYING, PRODUCT}:
                continue

            mid = get_valid_mid(row)
            if mid is None or not np.isfinite(mid):
                continue

            tmp.append(
                {
                    "day": day,
                    "timestamp": int(row["timestamp"]),
                    "product": product,
                    "mid": float(mid),
                }
            )

        table = pd.DataFrame(tmp)
        if table.empty:
            continue

        pivot = table.pivot(index="timestamp", columns="product", values="mid").sort_index()

        for ts, r in pivot.iterrows():
            if UNDERLYING not in r or PRODUCT not in r:
                continue
            if pd.isna(r[UNDERLYING]) or pd.isna(r[PRODUCT]):
                continue

            S = float(r[UNDERLYING])
            V = float(r[PRODUCT])
            T = tte_years(day, int(ts))

            iv = implied_vol_bisect(S=S, K=K, T=T, price=V)
            if iv is None or not np.isfinite(iv):
                continue

            rows.append(
                {
                    "combined_timestamp": day * DAY_LENGTH + int(ts),
                    "day": day,
                    "timestamp": int(ts),
                    "S": S,
                    "mid": V,
                    "raw_iv": iv,
                    "tte_days": T * DAYS_PER_YEAR,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No points collected.")

    return df.sort_values("combined_timestamp").reset_index(drop=True)


def plot_html(df: pd.DataFrame, out_path: Path) -> None:
    low_iv = df[df["raw_iv"] < IV_THRESHOLD]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["combined_timestamp"],
            y=df["mid"],
            mode="lines",
            name="VEV_4000 valid mid",
            customdata=df[["day", "timestamp", "raw_iv", "S", "tte_days"]].to_numpy(),
            hovertemplate=(
                "mid line<br>"
                "combined_ts=%{x}<br>"
                "mid=%{y:.4f}<br>"
                "day=%{customdata[0]}<br>"
                "timestamp=%{customdata[1]}<br>"
                "raw_iv=%{customdata[2]:.6f}<br>"
                "S=%{customdata[3]:.4f}<br>"
                "TTE_days=%{customdata[4]:.4f}<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=low_iv["combined_timestamp"],
            y=low_iv["mid"],
            mode="markers",
            name=f"raw_iv < {IV_THRESHOLD}",
            marker={"size": 7, "symbol": "circle"},
            customdata=low_iv[["day", "timestamp", "raw_iv", "S", "tte_days"]].to_numpy(),
            hovertemplate=(
                "low raw_iv point<br>"
                "combined_ts=%{x}<br>"
                "mid=%{y:.4f}<br>"
                "day=%{customdata[0]}<br>"
                "timestamp=%{customdata[1]}<br>"
                "raw_iv=%{customdata[2]:.6f}<br>"
                "S=%{customdata[3]:.4f}<br>"
                "TTE_days=%{customdata[4]:.4f}<extra></extra>"
            ),
        )
    )

    for boundary in [1_000_000, 2_000_000]:
        fig.add_vline(x=boundary, line_dash="dash", opacity=0.5)

    fig.update_layout(
        title="VEV_4000 valid mid price with raw IV < 0.4 points",
        xaxis_title="combined timestamp",
        yaxis_title="VEV_4000 valid mid price",
        dragmode="zoom",
        hovermode="closest",
        template="plotly_white",
        height=700,
        legend={"itemclick": "toggle", "itemdoubleclick": "toggleothers"},
    )

    fig.update_xaxes(range=[0, 3_000_000], showspikes=True, spikemode="across")
    fig.update_yaxes(showspikes=True, spikemode="across")

    html = pio.to_html(
        fig,
        include_plotlyjs="cdn",
        full_html=True,
        config={"scrollZoom": True, "displaylogo": False},
    )

    out_path.write_text(html, encoding="utf-8")


def main() -> None:
    project_root = find_project_root(Path(__file__))
    data_dir = project_root / "data_capsule" / "round3"
    out_dir = project_root / "jonghyeok" / "data_analysis" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = collect_points(data_dir)

    csv_path = out_dir / "round3_vev4000_mid_with_raw_iv.csv"
    html_path = out_dir / "round3_vev4000_mid_with_low_iv_points.html"

    df.to_csv(csv_path, index=False)
    plot_html(df, html_path)

    print(f"[SAVED] {csv_path}")
    print(f"[SAVED] {html_path}")
    print(f"[INFO] raw_iv < {IV_THRESHOLD}: {int((df['raw_iv'] < IV_THRESHOLD).sum())} / {len(df)}")


if __name__ == "__main__":
    main()