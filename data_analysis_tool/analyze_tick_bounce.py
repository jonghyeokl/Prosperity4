# data_analysis_tool/analyze_tick_bounce.py

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Parameters
# ============================================================
ROUND = 3
PRODUCT = "VELVETFRUIT_EXTRACT"
TICK_HORIZONS = [1, 5, 10, 20, 30, 50, 75, 100, 125, 150, 175, 200, 250, 300]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data_capsule" / f"round{ROUND}"
PRICE_FILE_GLOB = f"prices_round_{ROUND}_day_*.csv"

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
    "HYDROGEL_PACK": 10,
}

DEFAULT_VALID_VOLUME = 1
PLOT_SAMPLE_N = 8000

HEADERS = [
    "day",
    "timestamp",
    "product",
    "bid_price_1",
    "bid_volume_1",
    "bid_price_2",
    "bid_volume_2",
    "bid_price_3",
    "bid_volume_3",
    "ask_price_1",
    "ask_volume_1",
    "ask_price_2",
    "ask_volume_2",
    "ask_price_3",
    "ask_volume_3",
    "mid_price",
    "profit_and_loss",
]


def read_price_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, delimiter=";")
    if "product" not in df.columns:
        df.columns = HEADERS[: len(df.columns)]
    return df


def get_valid_mid(row: pd.Series) -> float | None:
    valid_volume = VALID_BID_ASK_VOLUME.get(row["product"], DEFAULT_VALID_VOLUME)

    valid_bid = None
    valid_ask = None

    for i in range(1, 4):
        bid_price = row.get(f"bid_price_{i}", np.nan)
        bid_vol = row.get(f"bid_volume_{i}", np.nan)
        if pd.notna(bid_price) and pd.notna(bid_vol) and bid_vol >= valid_volume:
            valid_bid = float(bid_price)
            break

    for i in range(1, 4):
        ask_price = row.get(f"ask_price_{i}", np.nan)
        ask_vol = row.get(f"ask_volume_{i}", np.nan)
        if pd.notna(ask_price) and pd.notna(ask_vol) and abs(ask_vol) >= valid_volume:
            valid_ask = float(ask_price)
            break

    if valid_bid is None:
        bid_price = row.get("bid_price_1", np.nan)
        if pd.notna(bid_price):
            valid_bid = float(bid_price)

    if valid_ask is None:
        ask_price = row.get("ask_price_1", np.nan)
        if pd.notna(ask_price):
            valid_ask = float(ask_price)

    if valid_bid is None or valid_ask is None:
        return None

    return 0.5 * (valid_bid + valid_ask)


def load_product_data() -> pd.DataFrame:
    csv_paths = sorted(DATA_DIR.glob(PRICE_FILE_GLOB))
    if not csv_paths:
        raise FileNotFoundError(f"가격 파일을 찾지 못했습니다: {DATA_DIR / PRICE_FILE_GLOB}")

    frames = []

    for path in csv_paths:
        df = read_price_file(path)
        df = df[df["product"] == PRODUCT].copy()
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    numeric_cols = [
        "day",
        "timestamp",
        "bid_price_1",
        "bid_volume_1",
        "bid_price_2",
        "bid_volume_2",
        "bid_price_3",
        "bid_volume_3",
        "ask_price_1",
        "ask_volume_1",
        "ask_price_2",
        "ask_volume_2",
        "ask_price_3",
        "ask_volume_3",
        "mid_price",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["day", "timestamp", "product"]).copy()
    df["day"] = df["day"].astype(int)
    df["timestamp"] = df["timestamp"].astype(int)

    df["analysis_mid"] = df.apply(get_valid_mid, axis=1)
    df = df.dropna(subset=["analysis_mid"]).copy()

    df = df.sort_values(["day", "timestamp"]).reset_index(drop=True)

    if df.empty:
        raise ValueError(f"조건에 맞는 데이터가 없습니다. PRODUCT={PRODUCT}")

    return df


def infer_base_step(ts: pd.Series) -> int:
    diffs = ts.diff().dropna()
    diffs = diffs[diffs > 0]

    if len(diffs) == 0:
        return 100

    mode = diffs.mode()
    if len(mode) == 0:
        return int(diffs.iloc[0])

    return int(mode.iloc[0])


def make_n_tick_pairs(df: pd.DataFrame, n_tick: int) -> pd.DataFrame:
    frames = []

    for day, g in df.groupby("day", sort=True):
        g = g.sort_values("timestamp").copy()
        step = infer_base_step(g["timestamp"])

        g["mid_t"] = g["analysis_mid"]
        g["mid_prev"] = g["analysis_mid"].shift(n_tick)
        g["mid_future"] = g["analysis_mid"].shift(-n_tick)

        g["prev_gap"] = g["timestamp"] - g["timestamp"].shift(n_tick)
        g["future_gap"] = g["timestamp"].shift(-n_tick) - g["timestamp"]

        g["past_change"] = g["mid_t"] - g["mid_prev"]
        g["future_change"] = g["mid_future"] - g["mid_t"]

        valid = (
            (g["prev_gap"] == n_tick * step)
            & (g["future_gap"] == n_tick * step)
            & np.isfinite(g["past_change"])
            & np.isfinite(g["future_change"])
        )

        frames.append(
            g.loc[
                valid,
                ["day", "timestamp", "mid_t", "past_change", "future_change"],
            ].copy()
        )

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


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
            "rmse": np.nan,
            "mae": np.nan,
            "bounce_hit_rate": np.nan,
            "mean_abs_past_change": np.nan,
            "mean_abs_future_change": np.nan,
        }

    x_mean = x.mean()
    y_mean = y.mean()

    var_x = np.sum((x - x_mean) ** 2)
    cov_xy = np.sum((x - x_mean) * (y - y_mean))

    beta = cov_xy / var_x if var_x > 0 else np.nan
    alpha = y_mean - beta * x_mean

    y_hat = alpha + beta * x
    resid = y - y_hat

    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    corr = np.corrcoef(x, y)[0, 1] if n >= 2 else np.nan

    sigma2 = ss_res / max(n - 2, 1)
    se_beta = np.sqrt(sigma2 / var_x) if var_x > 0 else np.nan
    t_beta = beta / se_beta if se_beta and se_beta > 0 else np.nan

    nonzero = np.abs(x) > 1e-12
    bounce_hit_rate = np.mean((x[nonzero] * y[nonzero]) < 0) if np.any(nonzero) else np.nan

    return {
        "n": n,
        "alpha": float(alpha),
        "beta": float(beta),
        "corr": float(corr),
        "t_beta": float(t_beta),
        "r2": float(r2),
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "mae": float(np.mean(np.abs(resid))),
        "bounce_hit_rate": float(bounce_hit_rate),
        "mean_abs_past_change": float(np.mean(np.abs(x))),
        "mean_abs_future_change": float(np.mean(np.abs(y))),
    }


def plot_scatter_with_fit(
    pairs: pd.DataFrame,
    stats: dict,
    n_tick: int,
    day_label: str,
    out_dir: Path,
) -> None:
    if pairs.empty:
        return

    plot_df = (
        pairs.sample(min(PLOT_SAMPLE_N, len(pairs)), random_state=42)
        if len(pairs) > PLOT_SAMPLE_N
        else pairs
    )

    x = plot_df["past_change"].to_numpy(dtype=float)
    y = plot_df["future_change"].to_numpy(dtype=float)

    alpha = stats["alpha"]
    beta = stats["beta"]

    x_line = np.linspace(np.nanmin(x), np.nanmax(x), 300)
    y_line = alpha + beta * x_line

    plt.figure(figsize=(9, 8))
    plt.scatter(x, y, s=6, alpha=0.25)
    plt.plot(x_line, y_line, linewidth=2)
    plt.axhline(0, linewidth=1)
    plt.axvline(0, linewidth=1)
    plt.xlabel(f"past_change = mid_t - mid_(t-{n_tick})")
    plt.ylabel(f"future_change = mid_(t+{n_tick}) - mid_t")
    plt.title(
        f"{PRODUCT} | {day_label} | {n_tick}-tick bounce\n"
        f"beta={beta:.4f}, corr={stats['corr']:.4f}, "
        f"r2={stats['r2']:.4f}, hit={stats['bounce_hit_rate']:.3f}"
    )
    plt.tight_layout()

    out_path = out_dir / f"{day_label}_{n_tick}_tick_bounce.png"
    plt.savefig(out_path, dpi=160)
    plt.close()


def analyze_one_day(df: pd.DataFrame, day_label: str, out_dir: Path) -> list[dict]:
    summary_rows = []

    for n_tick in TICK_HORIZONS:
        pairs = make_n_tick_pairs(df, n_tick)

        if pairs.empty:
            row = {
                "product": PRODUCT,
                "day": day_label,
                "n_tick": n_tick,
                "n": 0,
                "alpha": np.nan,
                "beta": np.nan,
                "corr": np.nan,
                "t_beta": np.nan,
                "r2": np.nan,
                "rmse": np.nan,
                "mae": np.nan,
                "bounce_hit_rate": np.nan,
                "mean_abs_past_change": np.nan,
                "mean_abs_future_change": np.nan,
                "mean_reversion_score": np.nan,
            }
            summary_rows.append(row)
            print(f"{day_label:>7} | {n_tick:>2}-tick | no valid pairs")
            continue

        stats = regression_stats(
            x=pairs["past_change"].to_numpy(dtype=float),
            y=pairs["future_change"].to_numpy(dtype=float),
        )

        row = {
            "product": PRODUCT,
            "day": day_label,
            "n_tick": n_tick,
            **stats,
            "mean_reversion_score": -stats["beta"] if np.isfinite(stats["beta"]) else np.nan,
        }
        summary_rows.append(row)

        plot_scatter_with_fit(pairs, stats, n_tick, day_label, out_dir)

        print(
            f"{day_label:>7} | {n_tick:>2}-tick | "
            f"n={stats['n']}, beta={stats['beta']:.4f}, corr={stats['corr']:.4f}, "
            f"r2={stats['r2']:.4f}, t={stats['t_beta']:.1f}, "
            f"hit={stats['bounce_hit_rate']:.3f}"
        )
        print(
            f"{'':>7} | {'':>7} | "
            f"mean_abs_past={stats['mean_abs_past_change']:.4f}, "
            f"mean_abs_future={stats['mean_abs_future_change']:.4f}, "
            f"score={row['mean_reversion_score']:.4f}"
        )

    return summary_rows


def main() -> None:
    df = load_product_data()

    output_root = PROJECT_ROOT / "data_analysis_tool" / "outputs" / PRODUCT
    output_root.mkdir(parents=True, exist_ok=True)

    all_rows = []

    print(f"\n=== {PRODUCT} tick bounce analysis ===")
    print(f"ROUND={ROUND}, DATA_DIR={DATA_DIR}")
    print(f"saved to: {output_root}\n")

    # all
    all_rows.extend(analyze_one_day(df, "day_all", output_root))

    # each day
    for day in sorted(df["day"].unique()):
        day_df = df[df["day"] == day].copy()
        day_label = f"day_{int(day)}"
        all_rows.extend(analyze_one_day(day_df, day_label, output_root))

    summary_df = pd.DataFrame(all_rows)
    summary_path = output_root / "tick_bounce_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n[SAVED]", summary_path)
    print("[SAVED]", output_root)


if __name__ == "__main__":
    main()