# jonghyeok/data_analysis/round3_plot_theo_diff_5000_5400.py

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio


UNDERLYING = "VELVETFRUIT_EXTRACT"

ATM_VOUCHERS = {
    "VEV_5000": 5000,
    "VEV_5100": 5100,
    "VEV_5200": 5200,
    "VEV_5300": 5300,
    "VEV_5400": 5400,
}

SMILE_WINDOW_PER_VOUCHER = 300
SMILE_MIN_POINTS_FOR_FIT = 600

IV_A_FALLBACK = 0.13797259576052961
IV_B_FALLBACK = 0.04060700203856375
IV_C_FALLBACK = 0.24224047421616432

VALID_BID_ASK_VOLUME = {
    "VELVETFRUIT_EXTRACT": 15,
    "VEV_5000": 6,
    "VEV_5100": 6,
    "VEV_5200": 6,
    "VEV_5300": 5,
    "VEV_5400": 5,
    "VEV_5500": 5,
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
    m = DAY_RE.search(path.name)
    if not m:
        raise ValueError(f"Cannot parse day from filename: {path.name}")
    return int(m.group(1))


def read_csv_auto(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    if len(df.columns) == 1:
        df = pd.read_csv(path)
    return df


def get_best_valid_bid_ask(row: pd.Series, valid_volume: int):
    best_valid_bid = None
    best_valid_ask = None

    for i in range(1, 4):
        bid_price = row.get(f"bid_price_{i}", np.nan)
        bid_vol = row.get(f"bid_volume_{i}", np.nan)

        if pd.notna(bid_price) and pd.notna(bid_vol):
            if bid_vol >= valid_volume:
                best_valid_bid = float(bid_price)
                break

    for i in range(1, 4):
        ask_price = row.get(f"ask_price_{i}", np.nan)
        ask_vol = row.get(f"ask_volume_{i}", np.nan)

        if pd.notna(ask_price) and pd.notna(ask_vol):
            if -ask_vol >= valid_volume:
                best_valid_ask = float(ask_price)
                break

    best_bid = row.get("bid_price_1", np.nan)
    best_ask = row.get("ask_price_1", np.nan)

    if best_valid_bid is None and pd.notna(best_bid):
        best_valid_bid = float(best_bid)

    if best_valid_ask is None and pd.notna(best_ask):
        best_valid_ask = float(best_ask)

    return best_valid_bid, best_valid_ask


def get_valid_mid(row: pd.Series) -> float | None:
    product = row["product"]
    valid_volume = VALID_BID_ASK_VOLUME[product]

    bid, ask = get_best_valid_bid_ask(row, valid_volume)

    if bid is not None and ask is not None:
        return 0.5 * (bid + ask)

    return None


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


def implied_vol(V: float, S: float, K: float, T: float, tol: float = 1e-4, max_iter: int = 50):
    if T <= 0 or V <= 0 or S <= 0 or K <= 0:
        return None

    intrinsic = max(S - K, 0.0)

    if V < intrinsic - 1e-2 or V > S + 1e-2:
        return None

    lo = 1e-3
    hi = 3.0

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


def fit_quadratic_from_points(points: List[Tuple[float, float]]) -> Optional[Tuple[float, float, float]]:
    n = len(points)

    if n < 4:
        return None

    Sx4 = Sx3 = Sx2 = Sx1 = 0.0
    Sy = Syx = Syx2 = 0.0

    for m, v in points:
        m2 = m * m
        Sx4 += m2 * m2
        Sx3 += m2 * m
        Sx2 += m2
        Sx1 += m
        Sy += v
        Syx += v * m
        Syx2 += v * m2

    mat = [
        [Sx4, Sx3, Sx2, Syx2],
        [Sx3, Sx2, Sx1, Syx],
        [Sx2, Sx1, float(n), Sy],
    ]

    try:
        for i in range(3):
            pivot = mat[i][i]

            if abs(pivot) < 1e-15:
                swap = None
                for k in range(i + 1, 3):
                    if abs(mat[k][i]) > 1e-15:
                        swap = k
                        break

                if swap is None:
                    return None

                mat[i], mat[swap] = mat[swap], mat[i]
                pivot = mat[i][i]

            for j in range(i + 1, 3):
                factor = mat[j][i] / pivot
                for c in range(i, 4):
                    mat[j][c] -= factor * mat[i][c]

        coeffs = [0.0, 0.0, 0.0]

        for i in range(2, -1, -1):
            coeffs[i] = mat[i][3]
            for j in range(i + 1, 3):
                coeffs[i] -= mat[i][j] * coeffs[j]
            coeffs[i] /= mat[i][i]

        return coeffs[0], coeffs[1], coeffs[2]

    except Exception:
        return None


def get_tte_years(timestamp: int, day_num: int) -> float:
    progress_days = timestamp / 1_000_000.0
    remaining_days = max(8.0 - day_num - progress_days, 1e-9)
    return remaining_days / DAYS_PER_YEAR


def get_fair_iv(m: float, rolling_coeffs):
    if rolling_coeffs is not None:
        a, b, c = rolling_coeffs
    else:
        a = IV_A_FALLBACK
        b = IV_B_FALLBACK
        c = IV_C_FALLBACK

    return a * m * m + b * m + c


def collect_mid_table(path: Path) -> pd.DataFrame:
    raw = read_csv_auto(path)
    needed = {UNDERLYING, *ATM_VOUCHERS.keys()}

    rows = []

    for _, row in raw.iterrows():
        product = row["product"]

        if product not in needed:
            continue

        mid = get_valid_mid(row)

        if mid is None or not np.isfinite(mid):
            continue

        rows.append(
            {
                "timestamp": int(row["timestamp"]),
                "product": product,
                "mid": float(mid),
                "bid_1": float(row["bid_price_1"]) if pd.notna(row.get("bid_price_1", np.nan)) else np.nan,
                "ask_1": float(row["ask_price_1"]) if pd.notna(row.get("ask_price_1", np.nan)) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def collect_theo_diff_points(data_dir: Path) -> pd.DataFrame:
    price_files = sorted(data_dir.glob("prices_round_3_day_*.csv"), key=parse_day_from_filename)

    if not price_files:
        raise FileNotFoundError(f"No prices_round_3_day_*.csv found in {data_dir}")

    hist = {product: [] for product in ATM_VOUCHERS}
    rows = []

    for file_idx, path in enumerate(price_files):
        day_num = parse_day_from_filename(path)

        print(f"[LOAD] {path.name}, day_num={day_num}")

        mids = collect_mid_table(path)

        if mids.empty:
            continue

        pivot = mids.pivot(index="timestamp", columns="product", values="mid").sort_index()

        for ts, r in pivot.iterrows():
            if UNDERLYING not in r or pd.isna(r[UNDERLYING]):
                continue

            timestamp = int(ts)
            S_mid = float(r[UNDERLYING])

            if S_mid <= 0:
                continue

            T = get_tte_years(timestamp, day_num)

            if T <= 0:
                continue

            sqrt_t = math.sqrt(T)

            # 현재 코드처럼 먼저 smile history 업데이트
            for product, K in ATM_VOUCHERS.items():
                if product not in r or pd.isna(r[product]):
                    continue

                option_mid = float(r[product])

                iv = implied_vol(
                    V=option_mid,
                    S=S_mid,
                    K=K,
                    T=T,
                )

                if iv is None or not math.isfinite(iv):
                    continue

                m = math.log(K / S_mid) / sqrt_t
                hist[product].append((m, iv))

                if len(hist[product]) > SMILE_WINDOW_PER_VOUCHER:
                    hist[product] = hist[product][-SMILE_WINDOW_PER_VOUCHER:]

            all_points = []
            for product in ATM_VOUCHERS:
                all_points.extend(hist[product])

            rolling_coeffs = None
            if len(all_points) >= SMILE_MIN_POINTS_FOR_FIT:
                rolling_coeffs = fit_quadratic_from_points(all_points)

            used_rolling_fit = rolling_coeffs is not None

            # 현재 코드처럼 업데이트된 rolling_coeffs/fallback으로 theo_diff 계산
            for product, K in ATM_VOUCHERS.items():
                if product not in r or pd.isna(r[product]):
                    continue

                option_mid = float(r[product])
                m = math.log(K / S_mid) / sqrt_t
                sigma = get_fair_iv(m, rolling_coeffs)

                if sigma <= 0 or not math.isfinite(sigma):
                    continue

                theo, delta, vega = bs_call_with_greeks(S_mid, K, T, sigma)
                theo_diff = option_mid - theo

                rows.append(
                    {
                        "combined_timestamp": file_idx * DAY_LENGTH + timestamp,
                        "file_idx": file_idx,
                        "day_num": day_num,
                        "timestamp": timestamp,
                        "product": product,
                        "K": K,
                        "S_mid": S_mid,
                        "option_mid": option_mid,
                        "T_days": T * DAYS_PER_YEAR,
                        "m": m,
                        "sigma": sigma,
                        "theo": theo,
                        "delta": delta,
                        "vega": vega,
                        "theo_diff": theo_diff,
                        "used_rolling_fit": used_rolling_fit,
                    }
                )

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("No theo_diff points collected.")

    return df.sort_values(["combined_timestamp", "product"]).reset_index(drop=True)


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
        "<p>Drag to zoom. Double click to reset. Scroll wheel zoom is enabled.</p>",
    ]

    first = True

    for product in ATM_VOUCHERS:
        g = df[df["product"] == product].sort_values("combined_timestamp")

        if g.empty:
            continue

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=g["combined_timestamp"],
                y=g["theo_diff"],
                mode="lines+markers",
                name=product,
                marker={"size": 3, "opacity": 0.6},
                customdata=g[
                    [
                        "day_num",
                        "timestamp",
                        "S_mid",
                        "option_mid",
                        "theo",
                        "sigma",
                        "T_days",
                        "vega",
                        "used_rolling_fit",
                    ]
                ].to_numpy(),
                hovertemplate=(
                    f"{product}<br>"
                    "combined_ts=%{x}<br>"
                    "theo_diff=%{y:.6f}<br>"
                    "day_num=%{customdata[0]}<br>"
                    "timestamp=%{customdata[1]}<br>"
                    "S_mid=%{customdata[2]:.4f}<br>"
                    "option_mid=%{customdata[3]:.4f}<br>"
                    "theo=%{customdata[4]:.4f}<br>"
                    "sigma=%{customdata[5]:.6f}<br>"
                    "T_days=%{customdata[6]:.4f}<br>"
                    "vega=%{customdata[7]:.4f}<br>"
                    "used_rolling_fit=%{customdata[8]}<extra></extra>"
                ),
            )
        )

        fig.add_hline(y=0, line_dash="dash", opacity=0.6)

        for boundary in [1_000_000, 2_000_000]:
            fig.add_vline(x=boundary, line_dash="dash", opacity=0.4)

        fig.update_layout(
            title=f"{product}: timestamp - theo_diff",
            xaxis_title="combined timestamp",
            yaxis_title="theo_diff = option_valid_mid - theo",
            dragmode="zoom",
            hovermode="closest",
            template="plotly_white",
            height=650,
            margin={"l": 70, "r": 30, "t": 70, "b": 60},
        )

        fig.update_xaxes(showspikes=True, spikemode="across")
        fig.update_yaxes(showspikes=True, spikemode="across")

        html = pio.to_html(
            fig,
            include_plotlyjs="cdn" if first else False,
            full_html=False,
            config={"scrollZoom": True, "displaylogo": False},
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

    df = collect_theo_diff_points(data_dir)

    csv_path = out_dir / "round3_theo_diff_5000_5400.csv"
    html_path = out_dir / "round3_theo_diff_5000_5400.html"

    df.to_csv(csv_path, index=False)

    html = build_html(df, title="Round 3 Theo Diff by Voucher")
    html_path.write_text(html, encoding="utf-8")

    print(f"[SAVED] {csv_path}")
    print(f"[SAVED] {html_path}")


if __name__ == "__main__":
    main()