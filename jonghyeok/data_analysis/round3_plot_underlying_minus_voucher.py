from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio


UNDERLYING = "VELVETFRUIT_EXTRACT"

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

DAY_LENGTH = 1_000_000
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


def collect_points(data_dir: Path, voucher: str) -> pd.DataFrame:
    if voucher not in VALID_BID_ASK_VOLUME:
        raise ValueError(f"Unknown voucher: {voucher}")

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
            if product not in {UNDERLYING, voucher}:
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
            if UNDERLYING not in r or voucher not in r:
                continue
            if pd.isna(r[UNDERLYING]) or pd.isna(r[voucher]):
                continue

            underlying_mid = float(r[UNDERLYING])
            voucher_mid = float(r[voucher])

            rows.append(
                {
                    "combined_timestamp": day * DAY_LENGTH + int(ts),
                    "day": day,
                    "timestamp": int(ts),
                    "underlying_mid": underlying_mid,
                    "voucher_mid": voucher_mid,
                    "diff": underlying_mid - voucher_mid,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No points collected.")

    return df.sort_values("combined_timestamp").reset_index(drop=True)


def plot_html(df: pd.DataFrame, voucher: str, out_path: Path) -> None:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["combined_timestamp"],
            y=df["diff"],
            mode="lines",
            name=f"{UNDERLYING} - {voucher}",
            customdata=df[["day", "timestamp", "underlying_mid", "voucher_mid"]].to_numpy(),
            hovertemplate=(
                f"{UNDERLYING} - {voucher}<br>"
                "combined_ts=%{x}<br>"
                "diff=%{y:.4f}<br>"
                "day=%{customdata[0]}<br>"
                "timestamp=%{customdata[1]}<br>"
                f"{UNDERLYING}_mid=%{{customdata[2]:.4f}}<br>"
                f"{voucher}_mid=%{{customdata[3]:.4f}}<extra></extra>"
            ),
        )
    )

    for boundary in [1_000_000, 2_000_000]:
        fig.add_vline(x=boundary, line_dash="dash", opacity=0.5)

    fig.update_layout(
        title=f"{UNDERLYING} mid - {voucher} mid",
        xaxis_title="combined timestamp",
        yaxis_title=f"{UNDERLYING} - {voucher}",
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--voucher", type=str, default="VEV_4000")
    args = parser.parse_args()

    project_root = find_project_root(Path(__file__))
    data_dir = project_root / "data_capsule" / "round3"
    out_dir = project_root / "jonghyeok" / "data_analysis" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = collect_points(data_dir, args.voucher)

    csv_path = out_dir / f"round3_{args.voucher}_underlying_minus_voucher.csv"
    html_path = out_dir / f"round3_{args.voucher}_underlying_minus_voucher.html"

    df.to_csv(csv_path, index=False)
    plot_html(df, args.voucher, html_path)

    print(f"[SAVED] {csv_path}")
    print(f"[SAVED] {html_path}")


if __name__ == "__main__":
    main()