# jonghyeok/data_analysis/round3_plot_raw_iv_html.py

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

VOUCHERS = {
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

VALID_BID_ASK_VOLUME = {
    "VELVETFRUIT_EXTRACT": 30,
    "VEV_4000": 6,
    "VEV_4500": 6,
    "VEV_5000": 13,
    "VEV_5100": 13,
    "VEV_5200": 13,
    "VEV_5300": 13,
    "VEV_5400": 13,
    "VEV_5500": 13,
    "VEV_6000": 5,
    "VEV_6500": 5,
}

DAYS_PER_YEAR = 365.0
DAY_LENGTH = 1_000_000

DAY_RE = re.compile(r"day_(-?\d+)")


def find_project_root(start: Path) -> Path:
    start = start.resolve()
    for candidate in [start.parent, *start.parents]:
        if (candidate / "data_capsule").exists() and (candidate / "jonghyeok").exists():
            return candidate
    return start.parents[2]


def parse_day_from_filename(path: Path) -> int:
    match = DAY_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse day from filename: {path.name}")
    return int(match.group(1))


def read_csv_auto(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    if len(df.columns) == 1:
        df = pd.read_csv(path)
    return df


def get_valid_mid(row: pd.Series) -> float | None:
    product = row["product"]
    valid_volume = VALID_BID_ASK_VOLUME.get(product)

    if valid_volume is None:
        return None

    valid_bid = None
    valid_ask = None

    for i in range(1, 4):
        bid_price = row.get(f"bid_price_{i}", np.nan)
        bid_vol = row.get(f"bid_volume_{i}", np.nan)

        if pd.notna(bid_price) and pd.notna(bid_vol):
            if bid_vol >= valid_volume:
                valid_bid = float(bid_price)
                break

    for i in range(1, 4):
        ask_price = row.get(f"ask_price_{i}", np.nan)
        ask_vol = row.get(f"ask_volume_{i}", np.nan)

        if pd.notna(ask_price) and pd.notna(ask_vol):
            if abs(float(ask_vol)) >= valid_volume:
                valid_ask = float(ask_price)
                break

    if valid_bid is None:
        bid_price_1 = row.get("bid_price_1", np.nan)
        if pd.notna(bid_price_1):
            valid_bid = float(bid_price_1)

    if valid_ask is None:
        ask_price_1 = row.get("ask_price_1", np.nan)
        if pd.notna(ask_price_1):
            valid_ask = float(ask_price_1)

    if valid_bid is None or valid_ask is None:
        return None

    return 0.5 * (valid_bid + valid_ask)


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
    """
    day 0,1,2의 시작 TTE를 각각 8d,7d,6d로 둠.
    """
    tte_start_days = 8.0 - day
    progress_days = timestamp / DAY_LENGTH
    remaining_days = max(tte_start_days - progress_days, 1e-9)
    return remaining_days / DAYS_PER_YEAR


def collect_raw_iv_points(data_dir: Path) -> pd.DataFrame:
    price_files = sorted(data_dir.glob("prices_round_3_day_*.csv"), key=parse_day_from_filename)
    rows = []

    for path in price_files:
        day = parse_day_from_filename(path)

        if day not in {0, 1, 2}:
            continue

        print(f"[LOAD] {path.name}")

        raw = read_csv_auto(path)
        needed = {UNDERLYING, *VOUCHERS.keys()}

        tmp_rows = []

        for _, row in raw.iterrows():
            product = row["product"]

            if product not in needed:
                continue

            mid = get_valid_mid(row)

            if mid is None or not np.isfinite(mid):
                continue

            tmp_rows.append(
                {
                    "timestamp": int(row["timestamp"]),
                    "product": product,
                    "mid": float(mid),
                }
            )

        table = pd.DataFrame(tmp_rows)

        if table.empty:
            continue

        pivot = table.pivot(index="timestamp", columns="product", values="mid").sort_index()

        for ts, r in pivot.iterrows():
            if UNDERLYING not in r or pd.isna(r[UNDERLYING]):
                continue

            S = float(r[UNDERLYING])

            if not np.isfinite(S) or S <= 0:
                continue

            T = tte_years(day, int(ts))
            combined_ts = day * DAY_LENGTH + int(ts)

            for product, K in VOUCHERS.items():
                if product not in r or pd.isna(r[product]):
                    continue

                V = float(r[product])

                if not np.isfinite(V) or V <= 0:
                    continue

                iv = implied_vol_bisect(S=S, K=K, T=T, price=V)

                if iv is None or not np.isfinite(iv):
                    continue

                rows.append(
                    {
                        "combined_timestamp": combined_ts,
                        "day": day,
                        "timestamp": int(ts),
                        "product": product,
                        "S": S,
                        "K": K,
                        "voucher_price": V,
                        "tte_days": T * DAYS_PER_YEAR,
                        "raw_iv": iv,
                    }
                )

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("No raw IV points collected.")

    return df


def build_html(df: pd.DataFrame, title: str) -> str:
    chunks = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "  <meta charset='utf-8'>",
        f"  <title>{title}</title>",
        "  <style>",
        "    body { font-family: Arial, sans-serif; margin: 24px; }",
        "    .chart-block { margin-bottom: 56px; }",
        "  </style>",
        "</head>",
        "<body>",
        f"<h1>{title}</h1>",
        "<p>Drag to zoom. Double click to reset. Legend click toggles traces. Scroll wheel zoom is enabled.</p>",
    ]

    first = True

    for product in VOUCHERS:
        g = df[df["product"] == product].sort_values("combined_timestamp")

        if g.empty:
            continue

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=g["combined_timestamp"],
                y=g["raw_iv"],
                mode="markers",
                name=product,
                marker={"size": 4, "opacity": 0.55},
                customdata=g[["day", "timestamp", "S", "voucher_price", "tte_days"]].to_numpy(),
                hovertemplate=(
                    f"{product}<br>"
                    "combined_ts=%{x}<br>"
                    "raw_iv=%{y:.6f}<br>"
                    "day=%{customdata[0]}<br>"
                    "timestamp=%{customdata[1]}<br>"
                    "S=%{customdata[2]:.2f}<br>"
                    "voucher_price=%{customdata[3]:.2f}<br>"
                    "TTE_days=%{customdata[4]:.4f}<extra></extra>"
                ),
            )
        )

        for boundary in [1_000_000, 2_000_000]:
            fig.add_vline(x=boundary, line_dash="dash", opacity=0.5)

        fig.update_layout(
            title=f"{product} raw implied volatility",
            xaxis_title="combined timestamp",
            yaxis_title="raw implied volatility",
            dragmode="zoom",
            hovermode="closest",
            template="plotly_white",
            height=650,
            margin={"l": 70, "r": 30, "t": 70, "b": 60},
            legend={"itemclick": "toggle", "itemdoubleclick": "toggleothers"},
        )

        fig.update_xaxes(range=[0, 3_000_000], showspikes=True, spikemode="across")
        fig.update_yaxes(showspikes=True, spikemode="across")

        html = pio.to_html(
            fig,
            include_plotlyjs="cdn" if first else False,
            full_html=False,
            config={
                "scrollZoom": True,
                "displaylogo": False,
            },
        )
        first = False

        chunks.append(f"<div class='chart-block'>{html}</div>")

    chunks.extend(["</body>", "</html>"])
    return "\n".join(chunks)


def main() -> None:
    project_root = find_project_root(Path(__file__))
    data_dir = project_root / "data_capsule" / "round3"
    out_dir = project_root / "jonghyeok" / "data_analysis" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = collect_raw_iv_points(data_dir)

    csv_path = out_dir / "round3_raw_iv_points_all_vouchers.csv"
    html_path = out_dir / "round3_raw_iv_timeseries_by_voucher.html"

    df.to_csv(csv_path, index=False)

    html = build_html(df, title="Round 3 Raw Implied Volatility by Voucher")
    html_path.write_text(html, encoding="utf-8")

    print(f"[SAVED] {csv_path}")
    print(f"[SAVED] {html_path}")


if __name__ == "__main__":
    main()