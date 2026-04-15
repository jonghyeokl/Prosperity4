import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TARGET_SYMBOL = "ASH_COATED_OSMIUM"

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


def infer_base_step(ts: pd.Series) -> int:
    diffs = ts.diff().dropna()
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 100
    mode = diffs.mode()
    if len(mode) == 0:
        return int(diffs.iloc[0])
    return int(mode.iloc[0])


def load_symbol_data(symbol: str) -> pd.DataFrame:
    project_root = Path(__file__).resolve().parents[2]
    round_dir = project_root / "data_capsule" / "round1"

    csv_paths = sorted(round_dir.glob("prices_round_1_day_*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"가격 파일을 찾지 못했습니다: {round_dir}")

    frames = []
    for path in csv_paths:
        df = read_price_file(path)
        df = df[df["product"] == symbol].copy()
        frames.append(df)

    raw = pd.concat(frames, ignore_index=True)

    for col in ["day", "timestamp", "bid_price_1", "ask_price_1", "mid_price"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    raw = raw.sort_values(["day", "timestamp"]).reset_index(drop=True)

    valid_quote = (
        raw["bid_price_1"].notna()
        & raw["ask_price_1"].notna()
        & (raw["bid_price_1"] > 0)
        & (raw["ask_price_1"] > 0)
        & (raw["ask_price_1"] >= raw["bid_price_1"])
    )

    raw["quote_mid"] = np.where(
        valid_quote,
        (raw["bid_price_1"] + raw["ask_price_1"]) / 2.0,
        np.nan,
    )

    # quote_mid_only 사용
    raw["analysis_mid"] = raw["quote_mid"]

    base_steps = {}
    for day, g in raw.groupby("day", sort=False):
        base_steps[day] = infer_base_step(g["timestamp"])
    raw["base_step"] = raw["day"].map(base_steps)

    df = raw[raw["analysis_mid"].notna()].copy()
    df = df.sort_values(["day", "timestamp"]).reset_index(drop=True)
    return df


def make_lag1_pairs(df: pd.DataFrame) -> pd.DataFrame:
    pairs = []

    for _, g in df.groupby("day", sort=False):
        g = g.sort_values("timestamp").copy()

        p = g["analysis_mid"]
        ts = g["timestamp"]
        step = int(g["base_step"].iloc[0])

        r_t = p.diff()
        gap1 = ts.diff()
        r_t = r_t.where(gap1 == step)

        sub = pd.DataFrame({
            "timestamp": ts,
            "r_t": r_t,
        }).dropna()

        sub["r_t1"] = sub["r_t"].shift(-1)
        future_gap = sub["timestamp"].shift(-1) - sub["timestamp"]

        sub = sub[(sub["r_t1"].notna()) & (future_gap == step)].copy()
        pairs.append(sub[["r_t", "r_t1"]])

    out = pd.concat(pairs, ignore_index=True)
    return out


def fit_line(x: np.ndarray, y: np.ndarray):
    # y = a + b x
    b, a = np.polyfit(x, y, 1)
    corr = np.corrcoef(x, y)[0, 1]

    n = len(x)
    if n >= 3 and abs(corr) < 1:
        t_stat = corr * math.sqrt((n - 2) / (1 - corr**2))
    else:
        t_stat = np.nan

    return a, b, corr, t_stat


def main():
    df = load_symbol_data(TARGET_SYMBOL)
    pairs = make_lag1_pairs(df)

    x = pairs["r_t"].to_numpy()
    y = pairs["r_t1"].to_numpy()

    a, b, corr, t_stat = fit_line(x, y)

    print(f"n = {len(pairs)}")
    print(f"corr(r_t, r_t+1) = {corr:.6f}")
    print(f"t_stat = {t_stat:.6f}")
    print(f"fitted line: r_(t+1) = {a:.6f} + ({b:.6f}) * r_t")

    # 산점도가 너무 많아서 랜덤 샘플만 그림
    sample_n = min(5000, len(pairs))
    plot_df = pairs.sample(sample_n, random_state=42) if len(pairs) > sample_n else pairs

    x_plot = plot_df["r_t"].to_numpy()
    y_plot = plot_df["r_t1"].to_numpy()

    x_line = np.linspace(x_plot.min(), x_plot.max(), 200)
    y_line = a + b * x_line

    plt.figure(figsize=(8, 8))
    plt.scatter(x_plot, y_plot, s=6, alpha=0.25)
    plt.plot(x_line, y_line)
    plt.axhline(0, linewidth=1)
    plt.axvline(0, linewidth=1)
    plt.xlabel("r_t")
    plt.ylabel("r_(t+1)")
    plt.title(
        f"{TARGET_SYMBOL}: lag-1 return relation\n"
        f"corr={corr:.4f}, slope={b:.4f}, intercept={a:.4f}"
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()