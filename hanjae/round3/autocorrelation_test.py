import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TARGET_SYMBOL = "HYDROGEL_PACK"

# return 정의: price[t] - price[t-RETURN_HORIZON]
RETURN_HORIZON = 1

# autocorrelation을 몇 lag까지 볼지
MAX_LAG = 100

# 랜덤 비교군 몇 개 그릴지
N_RANDOM = 1000

# "shuffle" 추천
RANDOM_MODE = "shuffle"   # "shuffle" or "gaussian"

RANDOM_SEED = 42


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

    raw["analysis_mid"] = raw["quote_mid"]

    base_steps = {}
    for day, g in raw.groupby("day", sort=False):
        base_steps[day] = infer_base_step(g["timestamp"])

    raw["base_step"] = raw["day"].map(base_steps)

    df = raw[raw["analysis_mid"].notna()].copy()
    df = df.sort_values(["day", "timestamp"]).reset_index(drop=True)
    return df


def extract_returns_by_day(df: pd.DataFrame, return_horizon: int) -> list[np.ndarray]:
    out = []

    for _, g in df.groupby("day", sort=False):
        g = g.sort_values("timestamp").copy().reset_index(drop=True)

        p = g["analysis_mid"]
        ts = g["timestamp"]
        step = int(g["base_step"].iloc[0])

        r = p - p.shift(return_horizon)
        gap = ts - ts.shift(return_horizon)

        r = r.where(gap == return_horizon * step)
        r = r.dropna().to_numpy()

        if len(r) > 2:
            out.append(r)

    return out


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return np.nan
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return np.corrcoef(x, y)[0, 1]


def autocorr_curve_from_series_list(series_list: list[np.ndarray], max_lag: int) -> np.ndarray:
    corrs = []

    for lag in range(1, max_lag + 1):
        xs = []
        ys = []

        for s in series_list:
            if len(s) <= lag:
                continue

            xs.append(s[:-lag])
            ys.append(s[lag:])

        if len(xs) == 0:
            corrs.append(np.nan)
            continue

        x = np.concatenate(xs)
        y = np.concatenate(ys)

        corrs.append(safe_corr(x, y))

    return np.array(corrs)


def make_random_series_list(
    real_series_list: list[np.ndarray],
    mode: str,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    random_list = []

    for s in real_series_list:
        if mode == "shuffle":
            s_rand = rng.permutation(s)

        elif mode == "gaussian":
            mu = float(np.mean(s))
            sigma = float(np.std(s))
            if sigma == 0:
                sigma = 1e-8
            s_rand = rng.normal(loc=mu, scale=sigma, size=len(s))

        else:
            raise ValueError("mode must be 'shuffle' or 'gaussian'")

        random_list.append(s_rand)

    return random_list


def build_random_curves(
    real_series_list: list[np.ndarray],
    max_lag: int,
    n_random: int,
    mode: str,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    curves = []

    for _ in range(n_random):
        rand_series_list = make_random_series_list(real_series_list, mode, rng)
        curve = autocorr_curve_from_series_list(rand_series_list, max_lag)
        curves.append(curve)

    return np.vstack(curves)   # shape = (n_random, max_lag)


def summarize_against_random(actual_curve: np.ndarray, random_curves: np.ndarray) -> pd.DataFrame:
    rows = []
    for i in range(len(actual_curve)):
        lag = i + 1
        rand_col = random_curves[:, i]
        rand_col = rand_col[~np.isnan(rand_col)]

        if len(rand_col) == 0:
            continue

        rows.append({
            "lag": lag,
            "actual_autocorr": actual_curve[i],
            "random_mean": np.mean(rand_col),
            "random_std": np.std(rand_col),
            "random_q05": np.quantile(rand_col, 0.05),
            "random_q50": np.quantile(rand_col, 0.50),
            "random_q95": np.quantile(rand_col, 0.95),
        })

    return pd.DataFrame(rows)


def plot_autocorr_vs_random(
    actual_curve: np.ndarray,
    random_curves: np.ndarray,
    symbol: str,
    return_horizon: int,
    mode: str,
):
    lags = np.arange(1, len(actual_curve) + 1)

    fig = plt.figure(figsize=(13, 6), facecolor="#0a1220")
    ax = fig.add_axes([0.07, 0.26, 0.78, 0.62])
    ax.set_facecolor("#e8ebf2")

    # 랜덤 곡선들
    for i in range(random_curves.shape[0]):
        ax.plot(lags, random_curves[i], color="black", alpha=0.12, linewidth=1)

    # 실제 곡선
    ax.plot(lags, actual_curve, color="red", linewidth=2.2, label=symbol)

    ax.axhline(0, color="gray", linewidth=1, linestyle="--", alpha=0.8)

    ax.set_xlabel("lag", fontsize=11)
    ax.set_ylabel("Autocorrelation", fontsize=11)
    ax.set_title(
        f"Autocorrelation Plot for {symbol}",
        fontsize=16,
        color="white",
        pad=16,
    )

    # 제목이 figure 위쪽에 보이도록
    fig.text(
        0.5, 0.93,
        f"Figure: Autocorrelation Plot for {symbol}",
        ha="center", va="center",
        fontsize=17, color="white", fontweight="bold"
    )

    # 범례
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="black", lw=1.5, label=f"{mode}_random_series"),
        Line2D([0], [0], color="red", lw=2.2, label=symbol),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0))

    # 아래 설명문
    fig.text(
        0.5, 0.10,
        f"Autocorrelation of {symbol} returns compared to autocorrelations from "
        f"{mode} random sequences.",
        ha="center", va="center",
        fontsize=12, color="white", style="italic"
    )

    plt.show()


def main():
    df = load_symbol_data(TARGET_SYMBOL)
    real_series_list = extract_returns_by_day(df, RETURN_HORIZON)

    actual_curve = autocorr_curve_from_series_list(real_series_list, MAX_LAG)
    random_curves = build_random_curves(
        real_series_list=real_series_list,
        max_lag=MAX_LAG,
        n_random=N_RANDOM,
        mode=RANDOM_MODE,
        seed=RANDOM_SEED,
    )

    summary = summarize_against_random(actual_curve, random_curves)

    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.6f}")

    print("\n=== Actual vs Random autocorrelation summary ===")
    print(summary.head(20))

    # 실제 곡선이 random 5% 구간보다 더 아래면 mean reversion이 랜덤보다 강하다고 볼 수 있음
    strong_neg = summary[summary["actual_autocorr"] < summary["random_q05"]].copy()

    print("\n=== Lags where actual autocorr is below random 5% quantile ===")
    print(strong_neg[["lag", "actual_autocorr", "random_q05", "random_q95"]])

    plot_autocorr_vs_random(
        actual_curve=actual_curve,
        random_curves=random_curves,
        symbol=TARGET_SYMBOL,
        return_horizon=RETURN_HORIZON,
        mode=RANDOM_MODE,
    )


if __name__ == "__main__":
    main()