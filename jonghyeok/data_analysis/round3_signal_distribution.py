from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistics import NormalDist


N = NormalDist()

UNDERLYING = "VELVETFRUIT_EXTRACT"

STRIKES = {
    "VEV_5000": 5000,
    "VEV_5100": 5100,
    "VEV_5200": 5200,
    "VEV_5300": 5300,
    "VEV_5400": 5400,
    "VEV_5500": 5500,
    "VEV_6000": 6000,
    "VEV_6500": 6500,
}

IV_COEFFS = [
    0.12828333580313714,
    0.019425909900886495,
    0.23472826462671575,
]

DAYS_PER_YEAR = 365.0

THEO_DIFF_EMA_WINDOW = 20
SWITCH_EMA_WINDOW = 100
WARMUP_COUNT = 20


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


def norm_pdf(x: float) -> float:
    return N.pdf(x)


def bs_call_price_delta_vega(S: float, K: float, T: float, sigma: float):
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0.0)
        delta = 1.0 if S > K else 0.0
        vega = 0.0
        return intrinsic, delta, vega

    sqrt_t = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t

    price = S * norm_cdf(d1) - K * norm_cdf(d2)
    delta = norm_cdf(d1)
    vega = S * norm_pdf(d1) * sqrt_t

    return price, delta, vega


def timestamp_progress(timestamp: float) -> float:
    return float(timestamp) / 1_000_000.0


def tte_years(tte_start_days: float, timestamp: float) -> float:
    remaining_days = tte_start_days - timestamp_progress(timestamp)
    return max(remaining_days / DAYS_PER_YEAR, 1e-9)


def fair_iv(S: float, K: float, T: float) -> float:
    a, b, c = IV_COEFFS
    m = math.log(K / S) / math.sqrt(T)
    return a * m * m + b * m + c


def ema_update(old: float | None, value: float, window: int) -> float:
    if old is None:
        return value
    alpha = 2.0 / (window + 1.0)
    return alpha * value + (1.0 - alpha) * old


def build_price_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    needed = {UNDERLYING, *STRIKES.keys()}

    for _, row in df.iterrows():
        product = row["product"]
        if product not in needed:
            continue

        mid = get_mid(row)
        if mid is None or not np.isfinite(mid):
            continue

        best_bid = row.get("bid_price_1", np.nan)
        best_ask = row.get("ask_price_1", np.nan)

        if pd.isna(best_bid) or pd.isna(best_ask):
            continue

        rows.append(
            {
                "timestamp": int(row["timestamp"]),
                "product": product,
                "mid": float(mid),
                "best_bid": float(best_bid),
                "best_ask": float(best_ask),
            }
        )

    return pd.DataFrame(rows)


def collect_signal_points(data_dir: Path, reset_per_day: bool = True, include_warmup: bool = False) -> pd.DataFrame:
    price_files = sorted(data_dir.glob("prices_round_3_day_*.csv"), key=parse_day_from_filename)
    if not price_files:
        raise FileNotFoundError(f"No prices_round_3_day_*.csv found in {data_dir}")

    # 정렬된 historical 3일을 TTE 8d, 7d, 6d 시작으로 매핑
    tte_start_by_file_idx = {0: 8.0, 1: 7.0, 2: 6.0}

    states = {}
    all_rows = []

    for file_idx, path in enumerate(price_files):
        parsed_day = parse_day_from_filename(path)
        tte_start_days = tte_start_by_file_idx.get(file_idx, 8.0 - file_idx)

        print(f"[LOAD] {path.name}: parsed_day={parsed_day}, tte_start={tte_start_days}d")

        if reset_per_day:
            states = {}

        raw = read_csv_auto(path)
        table = build_price_table(raw)

        if table.empty:
            print(f"[WARN] empty table: {path.name}")
            continue

        pivot = table.pivot(index="timestamp", columns="product", values=["mid", "best_bid", "best_ask"])
        pivot = pivot.sort_index()

        for ts, row in pivot.iterrows():
            try:
                S_mid = float(row[("mid", UNDERLYING)])
            except Exception:
                continue

            if not np.isfinite(S_mid):
                continue

            T = tte_years(tte_start_days, ts)

            for product, K in STRIKES.items():
                try:
                    option_mid = float(row[("mid", product)])
                    option_bid = float(row[("best_bid", product)])
                    option_ask = float(row[("best_ask", product)])
                except Exception:
                    continue

                if not all(np.isfinite(x) for x in [option_mid, option_bid, option_ask]):
                    continue

                iv = fair_iv(S_mid, K, T)
                if not np.isfinite(iv) or iv <= 0:
                    continue

                theo, delta, vega = bs_call_price_delta_vega(S_mid, K, T, iv)
                d = option_mid - theo

                state = states.setdefault(
                    product,
                    {
                        "count": 0,
                        "mean_diff": None,
                        "switch_mean": None,
                    },
                )

                state["count"] += 1

                mean_diff = ema_update(state["mean_diff"], d, THEO_DIFF_EMA_WINDOW)
                state["mean_diff"] = mean_diff

                abs_dev = abs(d - mean_diff)
                switch_mean = ema_update(state["switch_mean"], abs_dev, SWITCH_EMA_WINDOW)
                state["switch_mean"] = switch_mean

                sell_signal = option_bid - theo - mean_diff
                buy_signal = option_ask - theo - mean_diff

                if (not include_warmup) and state["count"] < WARMUP_COUNT:
                    continue

                all_rows.append(
                    {
                        "file": path.name,
                        "parsed_day": parsed_day,
                        "file_idx": file_idx,
                        "timestamp": int(ts),
                        "product": product,
                        "K": K,
                        "S_mid": S_mid,
                        "option_mid": option_mid,
                        "option_bid": option_bid,
                        "option_ask": option_ask,
                        "T_days": T * DAYS_PER_YEAR,
                        "fair_iv": iv,
                        "theo": theo,
                        "delta": delta,
                        "vega": vega,
                        "theo_diff": d,
                        "mean_diff": mean_diff,
                        "raw_signal": d - mean_diff,
                        "switch_mean": switch_mean,
                        "sell_signal": sell_signal,
                        "buy_signal": buy_signal,
                        "count": state["count"],
                    }
                )

    out = pd.DataFrame(all_rows)
    if out.empty:
        raise RuntimeError("No signal points collected.")

    return out


def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["switch_mean", "sell_signal", "buy_signal", "vega"]

    rows = []
    for product, g in df.groupby("product"):
        row = {"product": product, "n": len(g)}
        for col in metrics:
            x = g[col].dropna().to_numpy()
            row[f"{col}_mean"] = float(np.mean(x))
            row[f"{col}_std"] = float(np.std(x))
            row[f"{col}_min"] = float(np.min(x))
            row[f"{col}_q01"] = float(np.quantile(x, 0.01))
            row[f"{col}_q05"] = float(np.quantile(x, 0.05))
            row[f"{col}_q25"] = float(np.quantile(x, 0.25))
            row[f"{col}_median"] = float(np.quantile(x, 0.50))
            row[f"{col}_q75"] = float(np.quantile(x, 0.75))
            row[f"{col}_q95"] = float(np.quantile(x, 0.95))
            row[f"{col}_q99"] = float(np.quantile(x, 0.99))
            row[f"{col}_max"] = float(np.max(x))
        rows.append(row)

    return pd.DataFrame(rows)


def plot_boxplots(df: pd.DataFrame, out_dir: Path) -> None:
    metrics = ["switch_mean", "sell_signal", "buy_signal", "vega"]
    products = list(STRIKES.keys())

    for metric in metrics:
        data = [df.loc[df["product"] == p, metric].dropna().to_numpy() for p in products]

        plt.figure(figsize=(12, 6))
        plt.boxplot(data, tick_labels=products, showfliers=False)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel(metric)
        plt.title(f"{metric} distribution by voucher")
        plt.tight_layout()
        plt.savefig(out_dir / f"round3_{metric}_boxplot.png", dpi=160)
        plt.close()


def plot_histograms(df: pd.DataFrame, out_dir: Path) -> None:
    metrics = ["switch_mean", "sell_signal", "buy_signal", "vega"]

    for metric in metrics:
        for product in STRIKES:
            x = df.loc[df["product"] == product, metric].dropna().to_numpy()
            if len(x) == 0:
                continue

            plt.figure(figsize=(9, 5))
            plt.hist(x, bins=80)
            plt.xlabel(metric)
            plt.ylabel("count")
            plt.title(f"{product} {metric} histogram")
            plt.tight_layout()
            plt.savefig(out_dir / f"round3_{product}_{metric}_hist.png", dpi=140)
            plt.close()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    default_data_dir = repo_root / "data_capsule" / "round3"
    default_out_dir = Path(__file__).resolve().parent / "outputs" / "round3_signal_distribution"

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=default_data_dir)
    parser.add_argument("--out-dir", type=Path, default=default_out_dir)
    parser.add_argument("--no-reset-per-day", action="store_true")
    parser.add_argument("--include-warmup", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = collect_signal_points(
        data_dir=args.data_dir,
        reset_per_day=not args.no_reset_per_day,
        include_warmup=args.include_warmup,
    )

    summary = make_summary(df)

    df.to_csv(args.out_dir / "round3_signal_points.csv", index=False)
    summary.to_csv(args.out_dir / "round3_signal_summary.csv", index=False)

    plot_boxplots(df, args.out_dir)
    plot_histograms(df, args.out_dir)

    print("\n========== SUMMARY ==========")
    with pd.option_context("display.max_columns", 200, "display.width", 240):
        print(summary.to_string(index=False))

    print(f"\n[SAVED] {args.out_dir / 'round3_signal_points.csv'}")
    print(f"[SAVED] {args.out_dir / 'round3_signal_summary.csv'}")
    print(f"[SAVED] boxplots/histograms in {args.out_dir}")


if __name__ == "__main__":
    main()