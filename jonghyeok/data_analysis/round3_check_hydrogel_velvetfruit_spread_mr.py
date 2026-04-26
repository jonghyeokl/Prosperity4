# jonghyeok/data_analysis/round3_check_hydrogel_velvetfruit_spread_mr.py

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

Y_PRODUCT = "HYDROGEL_PACK"
X_PRODUCT = "VELVETFRUIT_EXTRACT"

# HYDROGEL_mid = hedge_a * VELVET_mid + intercept_b + spread
# 기본값 global: 전체 데이터로 hedge ratio 한 번 fit 후 day별 안정성 평가
# "per_day"로 바꾸면 day별로 hedge ratio를 따로 fit
HEDGE_FIT_MODE = "global"  # "global" or "per_day"

HORIZONS = [1, 5, 10, 30, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500]

PLOT_SAMPLE_N = 8000

VALID_BID_ASK_VOLUME = {
    "HYDROGEL_PACK": 10,
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data_capsule" / f"round{ROUND}"
PRICE_FILE_GLOB = f"prices_round_{ROUND}_day_*.csv"

OUT_DIR = (
    PROJECT_ROOT
    / "jonghyeok"
    / "data_analysis"
    / "output"
    / "pair_spread_mr_hydrogel_velvetfruit"
)

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


# ============================================================
# Data loading
# ============================================================
def read_price_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, delimiter=";")
    if "product" not in df.columns:
        df.columns = HEADERS[: len(df.columns)]
    return df


def get_valid_mid(row: pd.Series) -> float | None:
    product = row["product"]
    valid_volume = VALID_BID_ASK_VOLUME[product]

    valid_bid = None
    valid_ask = None

    for i in range(1, 4):
        bid_price = row.get(f"bid_price_{i}", np.nan)
        bid_vol = row.get(f"bid_volume_{i}", np.nan)

        if pd.notna(bid_price) and pd.notna(bid_vol) and float(bid_vol) >= valid_volume:
            valid_bid = float(bid_price)
            break

    for i in range(1, 4):
        ask_price = row.get(f"ask_price_{i}", np.nan)
        ask_vol = row.get(f"ask_volume_{i}", np.nan)

        if pd.notna(ask_price) and pd.notna(ask_vol) and abs(float(ask_vol)) >= valid_volume:
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


def load_pair_data() -> pd.DataFrame:
    csv_paths = sorted(DATA_DIR.glob(PRICE_FILE_GLOB))
    if not csv_paths:
        raise FileNotFoundError(f"가격 파일을 찾지 못했습니다: {DATA_DIR / PRICE_FILE_GLOB}")

    target_products = {Y_PRODUCT, X_PRODUCT}
    frames = []

    for path in csv_paths:
        df = read_price_file(path)
        df = df[df["product"].isin(target_products)].copy()
        frames.append(df)

    raw = pd.concat(frames, ignore_index=True)

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
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    raw = raw.dropna(subset=["day", "timestamp", "product"]).copy()
    raw["day"] = raw["day"].astype(int)
    raw["timestamp"] = raw["timestamp"].astype(int)

    raw["valid_mid"] = raw.apply(get_valid_mid, axis=1)
    raw = raw.dropna(subset=["valid_mid"]).copy()

    wide = (
        raw.pivot_table(
            index=["day", "timestamp"],
            columns="product",
            values="valid_mid",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    required = {"day", "timestamp", Y_PRODUCT, X_PRODUCT}
    missing = required - set(wide.columns)
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}")

    wide = wide.dropna(subset=[Y_PRODUCT, X_PRODUCT]).copy()

    wide = wide.rename(
        columns={
            Y_PRODUCT: "y_mid",
            X_PRODUCT: "x_mid",
        }
    )

    wide = wide.sort_values(["day", "timestamp"]).reset_index(drop=True)

    day_order = {day: i for i, day in enumerate(sorted(wide["day"].unique()))}
    wide["day_order"] = wide["day"].map(day_order)
    wide["combined_timestamp"] = wide["day_order"] * 1_000_000 + wide["timestamp"]

    return wide


# ============================================================
# Stats
# ============================================================
def infer_base_step(ts: pd.Series) -> int:
    diffs = ts.diff().dropna()
    diffs = diffs[diffs > 0]

    if len(diffs) == 0:
        return 100

    mode = diffs.mode()
    if len(mode) == 0:
        return int(diffs.iloc[0])

    return int(mode.iloc[0])


def fit_pair_ols(g: pd.DataFrame) -> tuple[float, float, float]:
    """
    y_mid = hedge_a * x_mid + intercept_b
    returns hedge_a, intercept_b, r2
    """
    x = g["x_mid"].to_numpy(dtype=float)
    y = g["y_mid"].to_numpy(dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 5:
        return np.nan, np.nan, np.nan

    x_mean = x.mean()
    y_mean = y.mean()

    var_x = np.sum((x - x_mean) ** 2)
    cov_xy = np.sum((x - x_mean) * (y - y_mean))

    hedge_a = cov_xy / var_x if var_x > 0 else np.nan
    intercept_b = y_mean - hedge_a * x_mean

    pred = hedge_a * x + intercept_b
    resid = y - pred

    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return float(hedge_a), float(intercept_b), float(r2)


def add_spread(g: pd.DataFrame, hedge_a: float, intercept_b: float) -> pd.DataFrame:
    out = g.copy()
    out["hedge_a"] = hedge_a
    out["intercept_b"] = intercept_b
    out["spread"] = out["y_mid"] - (hedge_a * out["x_mid"] + intercept_b)
    return out


def make_horizon_pairs(g: pd.DataFrame, horizon: int) -> pd.DataFrame:
    frames = []

    for day, d in g.groupby("day", sort=True):
        d = d.sort_values("timestamp").copy()
        step = infer_base_step(d["timestamp"])

        d["future_spread"] = d["spread"].shift(-horizon)
        d["future_gap"] = d["timestamp"].shift(-horizon) - d["timestamp"]
        d["future_spread_change"] = d["future_spread"] - d["spread"]

        valid = (
            (d["future_gap"] == horizon * step)
            & np.isfinite(d["spread"])
            & np.isfinite(d["future_spread_change"])
        )

        frames.append(
            d.loc[
                valid,
                [
                    "day",
                    "timestamp",
                    "combined_timestamp",
                    "x_mid",
                    "y_mid",
                    "spread",
                    "future_spread",
                    "future_spread_change",
                ],
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
            "hit_rate": np.nan,
            "mean_abs_spread": np.nan,
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
    hit_rate = np.mean((x[nonzero] * y[nonzero]) < 0) if np.any(nonzero) else np.nan

    return {
        "n": int(n),
        "alpha": float(alpha),
        "beta": float(beta),
        "corr": float(corr),
        "t_beta": float(t_beta),
        "r2": float(r2),
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "mae": float(np.mean(np.abs(resid))),
        "hit_rate": float(hit_rate),
        "mean_abs_spread": float(np.mean(np.abs(x))),
        "mean_abs_future_change": float(np.mean(np.abs(y))),
    }


# ============================================================
# Plot
# ============================================================
def plot_spread_timeseries(spread_df: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(14, 6))
    plt.plot(spread_df["combined_timestamp"], spread_df["spread"], linewidth=1)
    plt.axhline(0, linewidth=1)
    plt.xlabel("combined_timestamp")
    plt.ylabel("spread = HYDROGEL - (a * VELVET + b)")
    plt.title(f"{Y_PRODUCT} vs {X_PRODUCT} spread")
    plt.tight_layout()
    plt.savefig(out_dir / "pair_spread_timeseries.png", dpi=160)
    plt.close()


def plot_scatter_with_fit(
    pairs: pd.DataFrame,
    stats: dict,
    day_label: str,
    horizon: int,
    out_dir: Path,
) -> None:
    if pairs.empty:
        return

    plot_df = (
        pairs.sample(min(PLOT_SAMPLE_N, len(pairs)), random_state=42)
        if len(pairs) > PLOT_SAMPLE_N
        else pairs
    )

    x = plot_df["spread"].to_numpy(dtype=float)
    y = plot_df["future_spread_change"].to_numpy(dtype=float)

    alpha = stats["alpha"]
    beta = stats["beta"]

    x_line = np.linspace(np.nanmin(x), np.nanmax(x), 300)
    y_line = alpha + beta * x_line

    plt.figure(figsize=(9, 8))
    plt.scatter(x, y, s=6, alpha=0.25)
    plt.plot(x_line, y_line, linewidth=2)
    plt.axhline(0, linewidth=1)
    plt.axvline(0, linewidth=1)
    plt.xlabel("spread_t")
    plt.ylabel(f"spread_(t+{horizon}) - spread_t")
    plt.title(
        f"{day_label} | {horizon}-tick spread MR\n"
        f"beta={beta:.4f}, corr={stats['corr']:.4f}, "
        f"r2={stats['r2']:.4f}, hit={stats['hit_rate']:.3f}"
    )
    plt.tight_layout()
    plt.savefig(out_dir / f"{day_label}_{horizon}_tick_spread_mr.png", dpi=160)
    plt.close()


# ============================================================
# Main analysis
# ============================================================
def analyze_group(
    label: str,
    g: pd.DataFrame,
    hedge_a: float,
    intercept_b: float,
    hedge_r2: float,
    out_dir: Path,
) -> tuple[list[dict], pd.DataFrame]:
    spread_df = add_spread(g, hedge_a, intercept_b)

    rows = []
    pairs_for_plot = {}

    for h in HORIZONS:
        pairs = make_horizon_pairs(spread_df, h)

        stats = regression_stats(
            x=pairs["spread"].to_numpy(dtype=float) if not pairs.empty else np.array([]),
            y=pairs["future_spread_change"].to_numpy(dtype=float) if not pairs.empty else np.array([]),
        )

        row = {
            "label": label,
            "horizon": h,
            "hedge_a": hedge_a,
            "intercept_b": intercept_b,
            "hedge_r2": hedge_r2,
            **stats,
            "mean_reversion_score": -stats["beta"] if np.isfinite(stats["beta"]) else np.nan,
        }

        rows.append(row)
        pairs_for_plot[h] = (pairs, stats)

        print(
            f"{label:>7} | {h:>3}-tick | "
            f"n={stats['n']}, beta={stats['beta']:.4f}, corr={stats['corr']:.4f}, "
            f"r2={stats['r2']:.4f}, t={stats['t_beta']:.1f}, "
            f"hit={stats['hit_rate']:.3f}, hedge_a={hedge_a:.4f}"
        )

    # best horizon graph
    valid_rows = [r for r in rows if np.isfinite(r["r2"])]
    if valid_rows:
        best = max(valid_rows, key=lambda r: r["r2"])
        best_h = int(best["horizon"])
        pairs, stats = pairs_for_plot[best_h]
        plot_scatter_with_fit(pairs, stats, label, best_h, out_dir)

    return rows, spread_df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_pair_data()

    print(f"\n=== Pair spread mean reversion: {Y_PRODUCT} vs {X_PRODUCT} ===")
    print(f"ROUND={ROUND}")
    print(f"HEDGE_FIT_MODE={HEDGE_FIT_MODE}")
    print(f"DATA_DIR={DATA_DIR}")
    print(f"OUT_DIR={OUT_DIR}\n")

    global_a, global_b, global_hedge_r2 = fit_pair_ols(df)
    print(f"GLOBAL hedge: y = {global_a:.6f} * x + {global_b:.6f}, hedge_r2={global_hedge_r2:.6f}\n")

    all_rows = []
    spread_frames = []

    # day_all
    rows, spread_df = analyze_group(
        label="day_all",
        g=df,
        hedge_a=global_a,
        intercept_b=global_b,
        hedge_r2=global_hedge_r2,
        out_dir=OUT_DIR,
    )
    all_rows.extend(rows)
    spread_df["label"] = "day_all"
    spread_frames.append(spread_df)

    # each day
    for day, day_df in df.groupby("day", sort=True):
        label = f"day_{int(day)}"

        if HEDGE_FIT_MODE == "per_day":
            hedge_a, intercept_b, hedge_r2 = fit_pair_ols(day_df)
        else:
            hedge_a, intercept_b, hedge_r2 = global_a, global_b, global_hedge_r2

        rows, spread_df = analyze_group(
            label=label,
            g=day_df,
            hedge_a=hedge_a,
            intercept_b=intercept_b,
            hedge_r2=hedge_r2,
            out_dir=OUT_DIR,
        )
        all_rows.extend(rows)
        spread_df["label"] = label
        spread_frames.append(spread_df)

    summary_df = pd.DataFrame(all_rows)
    points_df = pd.concat(spread_frames, ignore_index=True)

    summary_path = OUT_DIR / "pair_spread_mr_summary.csv"
    points_path = OUT_DIR / "pair_spread_points.csv"

    summary_df.to_csv(summary_path, index=False)
    points_df.to_csv(points_path, index=False)

    plot_spread_timeseries(points_df[points_df["label"] == "day_all"], OUT_DIR)

    print("\n=== Best horizon by label ===")
    best = (
        summary_df.sort_values(["label", "r2"], ascending=[True, False])
        .groupby("label")
        .head(1)
        .reset_index(drop=True)
    )

    print(
        best[
            [
                "label",
                "horizon",
                "hedge_a",
                "intercept_b",
                "hedge_r2",
                "beta",
                "corr",
                "r2",
                "hit_rate",
                "mean_reversion_score",
            ]
        ].to_string(index=False)
    )

    print("\n[SAVED]", summary_path)
    print("[SAVED]", points_path)
    print("[SAVED]", OUT_DIR)


if __name__ == "__main__":
    main()