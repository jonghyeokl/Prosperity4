import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TARGET_SYMBOL = "HYDROGEL_PACK"

# 과거 H틱 변화와 미래 H틱 변화를 비교
HORIZON = 250

# non-overlap 간격.
# 기본적으로 HORIZON과 같게 두면 됨.
NON_OVERLAP_STEP = HORIZON

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
    round_dir = project_root / "data_capsule" / "round3"

    csv_paths = sorted(round_dir.glob("prices_round_3_day_*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"가격 파일을 찾지 못했습니다: {round_dir}")

    frames = []

    for path in csv_paths:
        df = read_price_file(path)
        df = df[df["product"] == symbol].copy()
        frames.append(df)

    raw = pd.concat(frames, ignore_index=True)

    for col in [
        "day",
        "timestamp",
        "bid_price_1",
        "ask_price_1",
        "mid_price",
    ]:
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

    raw["analysis_mid"] = raw["quote_mid"]

    base_steps = {}
    for day, g in raw.groupby("day", sort=False):
        base_steps[day] = infer_base_step(g["timestamp"])

    raw["base_step"] = raw["day"].map(base_steps)

    df = raw[raw["analysis_mid"].notna()].copy()
    df = df.sort_values(["day", "timestamp"]).reset_index(drop=True)
    return df


def make_horizon_pairs(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    overlapping 전체 pair 생성.

    r_past[t] = price[t] - price[t-horizon]
    r_future[t] = price[t+horizon] - price[t]
    """
    pairs = []

    for _, g in df.groupby("day", sort=False):
        g = g.sort_values("timestamp").copy().reset_index(drop=True)

        p = g["analysis_mid"]
        ts = g["timestamp"]
        step = int(g["base_step"].iloc[0])

        r_past = p - p.shift(horizon)
        past_gap = ts - ts.shift(horizon)
        r_past = r_past.where(past_gap == horizon * step)

        r_future = p.shift(-horizon) - p
        future_gap = ts.shift(-horizon) - ts
        r_future = r_future.where(future_gap == horizon * step)

        sub = pd.DataFrame({
            "day": g["day"],
            "timestamp": ts,
            "price": p,
            "r_past": r_past,
            "r_future": r_future,
        }).dropna()

        pairs.append(sub)

    out = pd.concat(pairs, ignore_index=True)
    return out


def make_non_overlapping_pairs(
    df: pd.DataFrame,
    horizon: int,
    non_overlap_step: int,
) -> pd.DataFrame:
    """
    day별로 non-overlapping pair만 뽑음.

    핵심:
    t = horizon, horizon + non_overlap_step, horizon + 2*non_overlap_step, ...
    이런 식으로 중심점 t를 띄엄띄엄 선택.

    HORIZON=1000, NON_OVERLAP_STEP=1000이면:
    [t-1000, t]와 [t, t+1000] 구간이 서로 최대한 겹치지 않음.
    """
    pairs = []

    for _, g in df.groupby("day", sort=False):
        g = g.sort_values("timestamp").copy().reset_index(drop=True)

        p = g["analysis_mid"].to_numpy()
        ts = g["timestamp"].to_numpy()
        day = g["day"].to_numpy()
        step = int(g["base_step"].iloc[0])

        n = len(g)

        # t-horizon >= 0, t+horizon < n 이어야 함
        candidate_indices = range(horizon, n - horizon, non_overlap_step)

        rows = []

        for idx in candidate_indices:
            past_idx = idx - horizon
            future_idx = idx + horizon

            # timestamp가 진짜 연속적인지 확인
            if ts[idx] - ts[past_idx] != horizon * step:
                continue
            if ts[future_idx] - ts[idx] != horizon * step:
                continue

            r_past = p[idx] - p[past_idx]
            r_future = p[future_idx] - p[idx]

            rows.append({
                "day": day[idx],
                "timestamp": ts[idx],
                "price": p[idx],
                "r_past": r_past,
                "r_future": r_future,
            })

        if rows:
            pairs.append(pd.DataFrame(rows))

    if not pairs:
        return pd.DataFrame(columns=["day", "timestamp", "price", "r_past", "r_future"])

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


def print_stats(name: str, pairs: pd.DataFrame):
    print(f"\n=== {name} ===")
    print(f"n = {len(pairs)}")

    if len(pairs) < 3:
        print("Not enough samples.")
        return None

    x = pairs["r_past"].to_numpy()
    y = pairs["r_future"].to_numpy()

    if np.std(x) == 0 or np.std(y) == 0:
        print("Zero std; cannot compute correlation.")
        return None

    a, b, corr, t_stat = fit_line(x, y)

    print(f"corr(r_past, r_future) = {corr:.6f}")
    print(f"t_stat = {t_stat:.6f}")
    print(f"fitted line: r_future = {a:.6f} + ({b:.6f}) * r_past")

    if corr < 0:
        print("interpretation: negative relation -> mean reversion tendency")
    elif corr > 0:
        print("interpretation: positive relation -> momentum tendency")
    else:
        print("interpretation: near zero -> weak/no linear relation")

    return a, b, corr, t_stat


def threshold_analysis(pairs: pd.DataFrame):
    if len(pairs) == 0:
        return

    abs_r = pairs["r_past"].abs()

    thresholds = [
        abs_r.quantile(0.50),
        abs_r.quantile(0.60),
        abs_r.quantile(0.70),
        abs_r.quantile(0.80),
        abs_r.quantile(0.90),
    ]

    print("\n=== Non-overlap threshold analysis ===")

    for th in thresholds:
        up = pairs[pairs["r_past"] > th]
        down = pairs[pairs["r_past"] < -th]

        print(f"\nthreshold = {th:.3f}")

        if len(up) > 0:
            print(
                f"after UP > {th:.3f}: "
                f"n={len(up)}, "
                f"avg_future={up['r_future'].mean():.3f}, "
                f"median_future={up['r_future'].median():.3f}, "
                f"reversal_rate={(up['r_future'] < 0).mean():.3f}"
            )

        if len(down) > 0:
            print(
                f"after DOWN < -{th:.3f}: "
                f"n={len(down)}, "
                f"avg_future={down['r_future'].mean():.3f}, "
                f"median_future={down['r_future'].median():.3f}, "
                f"reversal_rate={(down['r_future'] > 0).mean():.3f}"
            )


def plot_scatter(pairs: pd.DataFrame, title_suffix: str):
    if len(pairs) < 3:
        return

    x = pairs["r_past"].to_numpy()
    y = pairs["r_future"].to_numpy()

    if np.std(x) == 0 or np.std(y) == 0:
        return

    a, b, corr, t_stat = fit_line(x, y)

    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = a + b * x_line

    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, s=25, alpha=0.7)
    plt.plot(x_line, y_line)
    plt.axhline(0, linewidth=1)
    plt.axvline(0, linewidth=1)
    plt.xlabel(f"past {HORIZON}-timeclock return: price[t] - price[t-{HORIZON}]")
    plt.ylabel(f"future {HORIZON}-timeclock return: price[t+{HORIZON}] - price[t]")
    plt.title(
        f"{TARGET_SYMBOL}: {HORIZON}-timeclock bounce test\n"
        f"{title_suffix}\n"
        f"corr={corr:.4f}, slope={b:.4f}, t={t_stat:.2f}, n={len(pairs)}"
    )
    plt.tight_layout()
    plt.show()


def main():
    df = load_symbol_data(TARGET_SYMBOL)

    print(f"symbol = {TARGET_SYMBOL}")
    print(f"horizon = {HORIZON} timeclocks")
    print(f"non_overlap_step = {NON_OVERLAP_STEP}")

    overlapping_pairs = make_horizon_pairs(df, HORIZON)
    non_overlap_pairs = make_non_overlapping_pairs(
        df=df,
        horizon=HORIZON,
        non_overlap_step=NON_OVERLAP_STEP,
    )

    print_stats("Overlapping samples", overlapping_pairs)
    print_stats("Non-overlapping samples", non_overlap_pairs)

    print("\n=== Non-overlapping samples by day ===")
    if len(non_overlap_pairs) > 0:
        print(non_overlap_pairs.groupby("day").size())

    threshold_analysis(non_overlap_pairs)

    plot_scatter(non_overlap_pairs, "non-overlapping samples")


if __name__ == "__main__":
    main()