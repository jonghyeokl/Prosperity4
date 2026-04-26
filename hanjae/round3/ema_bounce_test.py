import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TARGET_SYMBOL = "HYDROGEL_PACK"

# EMA가 의미 있는지 테스트할 파라미터
EMA_SPANS = [5, 10, 20, 40, 80, 120]
HORIZONS = [1, 5, 10, 20, 40]


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


def make_ema_pairs(df: pd.DataFrame, ema_span: int, horizon: int) -> pd.DataFrame:
    pairs = []

    for _, g in df.groupby("day", sort=False):
        g = g.sort_values("timestamp").copy().reset_index(drop=True)

        p = g["analysis_mid"]
        ts = g["timestamp"]
        step = int(g["base_step"].iloc[0])

        ema = p.ewm(span=ema_span, adjust=False).mean()

        # 현재 가격이 EMA에서 얼마나 벗어났는지
        deviation = p - ema

        # 미래 horizon 뒤 가격 변화
        future_return = p.shift(-horizon) - p
        future_gap = ts.shift(-horizon) - ts
        future_return = future_return.where(future_gap == horizon * step)

        sub = pd.DataFrame({
            "timestamp": ts,
            "price": p,
            "ema": ema,
            "deviation": deviation,
            "future_return": future_return,
        }).dropna()

        # EMA 초반 warm-up 구간 제거
        sub = sub.iloc[ema_span:].copy()

        pairs.append(sub)

    out = pd.concat(pairs, ignore_index=True)
    return out


def analyze_ema_mean_reversion(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for ema_span in EMA_SPANS:
        for horizon in HORIZONS:
            pairs = make_ema_pairs(df, ema_span, horizon)

            x = pairs["deviation"].to_numpy()
            y = pairs["future_return"].to_numpy()

            if len(pairs) < 10 or np.std(x) == 0 or np.std(y) == 0:
                continue

            a, b, corr, t_stat = fit_line(x, y)

            above = pairs[pairs["deviation"] > 0]
            below = pairs[pairs["deviation"] < 0]

            above_avg_future = above["future_return"].mean() if len(above) > 0 else np.nan
            below_avg_future = below["future_return"].mean() if len(below) > 0 else np.nan

            above_reversal_rate = (above["future_return"] < 0).mean() if len(above) > 0 else np.nan
            below_reversal_rate = (below["future_return"] > 0).mean() if len(below) > 0 else np.nan

            rows.append({
                "ema_span": ema_span,
                "horizon": horizon,
                "n": len(pairs),
                "corr": corr,
                "slope": b,
                "intercept": a,
                "t_stat": t_stat,
                "above_avg_future": above_avg_future,
                "below_avg_future": below_avg_future,
                "above_reversal_rate": above_reversal_rate,
                "below_reversal_rate": below_reversal_rate,
            })

    result = pd.DataFrame(rows)
    result = result.sort_values(["horizon", "ema_span"]).reset_index(drop=True)
    return result


def print_result_table(result: pd.DataFrame):
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.6f}")

    print("\n=== EMA mean reversion test ===")
    print(result)

    print("\n=== Strong candidates: corr < 0, slope < 0, t_stat < -3 ===")
    strong = result[
        (result["corr"] < 0)
        & (result["slope"] < 0)
        & (result["t_stat"] < -3)
    ].copy()

    strong = strong.sort_values(["corr", "t_stat"])
    print(strong)


def plot_one_ema_relation(df: pd.DataFrame, ema_span: int, horizon: int):
    pairs = make_ema_pairs(df, ema_span, horizon)

    x = pairs["deviation"].to_numpy()
    y = pairs["future_return"].to_numpy()

    a, b, corr, t_stat = fit_line(x, y)

    sample_n = min(5000, len(pairs))
    plot_df = pairs.sample(sample_n, random_state=42) if len(pairs) > sample_n else pairs

    x_plot = plot_df["deviation"].to_numpy()
    y_plot = plot_df["future_return"].to_numpy()

    x_line = np.linspace(x_plot.min(), x_plot.max(), 200)
    y_line = a + b * x_line

    plt.figure(figsize=(8, 8))
    plt.scatter(x_plot, y_plot, s=6, alpha=0.25)
    plt.plot(x_line, y_line)
    plt.axhline(0, linewidth=1)
    plt.axvline(0, linewidth=1)
    plt.xlabel(f"price[t] - EMA_{ema_span}[t]")
    plt.ylabel(f"price[t+{horizon}] - price[t]")
    plt.title(
        f"{TARGET_SYMBOL}: EMA deviation vs future return\n"
        f"EMA span={ema_span}, horizon={horizon}, "
        f"corr={corr:.4f}, slope={b:.4f}, t={t_stat:.2f}"
    )
    plt.tight_layout()
    plt.show()


def threshold_analysis(df: pd.DataFrame, ema_span: int, horizon: int):
    pairs = make_ema_pairs(df, ema_span, horizon)

    dev_abs = pairs["deviation"].abs()
    thresholds = [
        dev_abs.quantile(0.50),
        dev_abs.quantile(0.60),
        dev_abs.quantile(0.70),
        dev_abs.quantile(0.80),
        dev_abs.quantile(0.90),
        dev_abs.quantile(0.95),
    ]

    print(f"\n=== Threshold analysis: EMA span={ema_span}, horizon={horizon} ===")

    for th in thresholds:
        above = pairs[pairs["deviation"] > th]
        below = pairs[pairs["deviation"] < -th]

        print(f"\nthreshold = {th:.3f}")

        if len(above) > 0:
            print(
                f"price ABOVE EMA by > {th:.3f}: "
                f"n={len(above)}, "
                f"avg_future={above['future_return'].mean():.3f}, "
                f"median_future={above['future_return'].median():.3f}, "
                f"reversal_rate={(above['future_return'] < 0).mean():.3f}"
            )

        if len(below) > 0:
            print(
                f"price BELOW EMA by < -{th:.3f}: "
                f"n={len(below)}, "
                f"avg_future={below['future_return'].mean():.3f}, "
                f"median_future={below['future_return'].median():.3f}, "
                f"reversal_rate={(below['future_return'] > 0).mean():.3f}"
            )


def main():
    df = load_symbol_data(TARGET_SYMBOL)

    result = analyze_ema_mean_reversion(df)
    print_result_table(result)

    # 여기 원하는 조합으로 바꿔서 산점도 확인
    BEST_EMA_SPAN = 40
    BEST_HORIZON = 10

    plot_one_ema_relation(df, BEST_EMA_SPAN, BEST_HORIZON)
    threshold_analysis(df, BEST_EMA_SPAN, BEST_HORIZON)


if __name__ == "__main__":
    main()