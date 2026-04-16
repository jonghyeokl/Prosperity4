from __future__ import annotations

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

# 현재 ASH 코드와 동일
MIN_HISTORY_LENGTH = 1
MAX_HISTORY_LENGTH = 5
MID_FALLBACK_OFFSET = 8
VALID_BID_ASK_VOLUME = 10


def read_price_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, delimiter=";")
    if "product" not in df.columns:
        df.columns = HEADERS[: len(df.columns)]
    return df


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

    for col in [
        "day",
        "timestamp",
        "bid_price_1",
        "ask_price_1",
        "bid_price_2",
        "ask_price_2",
        "bid_price_3",
        "ask_price_3",
        "bid_volume_1",
        "ask_volume_1",
        "bid_volume_2",
        "ask_volume_2",
        "bid_volume_3",
        "ask_volume_3",
    ]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    raw = raw.sort_values(["day", "timestamp"]).reset_index(drop=True)
    return raw


def median_of_list(xs: list[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def simulate_current_ash_fair(df: pd.DataFrame) -> pd.DataFrame:
    """
    현재 올린 ASH 코드와 동일한 방식으로
    각 시점의 mid_price, fair_price, mid_price - fair_price를 계산
    """
    rows = []

    for day, g in df.groupby("day", sort=True):
        g = g.sort_values("timestamp").copy()

        # traderData state reset per day
        past_few_mid_history: list[float] = []

        for _, row in g.iterrows():
            bid = row["bid_price_1"]
            ask = row["ask_price_1"]

            best_bid = None if pd.isna(bid) else float(bid)
            best_ask = None if pd.isna(ask) else float(ask)

            best_valid_bid = None
            best_valid_ask = None

            for i in range(1, 4):
                bid_price = row[f"bid_price_{i}"]
                bid_vol = row[f"bid_volume_{i}"]
                if not pd.isna(bid_price) and not pd.isna(bid_vol) and bid_vol >= VALID_BID_ASK_VOLUME:
                    best_valid_bid = float(bid_price)
                    break

            for i in range(1, 4):
                ask_price = row[f"ask_price_{i}"]
                ask_vol = row[f"ask_volume_{i}"]
                if not pd.isna(ask_price) and not pd.isna(ask_vol) and ask_vol >= VALID_BID_ASK_VOLUME:
                    best_valid_ask = float(ask_price)
                    break

            valid_mid_price = None
            if best_valid_bid is not None and best_valid_ask is not None:
                valid_mid_price = (best_valid_bid + best_valid_ask) / 2

            fair_price = None
            if len(past_few_mid_history) >= MIN_HISTORY_LENGTH:
                fair_price = median_of_list(past_few_mid_history)

            diff = None
            if valid_mid_price is not None and fair_price is not None:
                diff = valid_mid_price - fair_price

            rows.append(
                {
                    "day": int(row["day"]),
                    "timestamp": int(row["timestamp"]),
                    "best_bid": best_bid,
                    "best_ask": best_ask,
                    "mid_price_like_trader": valid_mid_price,
                    "fair_price_like_trader": fair_price,
                    "mid_minus_fair": diff,
                    "history_len_before_update": len(past_few_mid_history),
                }
            )

            # 현재 코드와 동일하게 fair 계산 후 현재 mid를 history에 추가
            if valid_mid_price is not None:
                if len(past_few_mid_history) < MAX_HISTORY_LENGTH:
                    past_few_mid_history.append(valid_mid_price)
                else:
                    past_few_mid_history.pop(0)
                    past_few_mid_history.append(valid_mid_price)

    return pd.DataFrame(rows)


def normal_cdf_scalar(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        return 1.0 if x >= mu else 0.0
    z = (x - mu) / (sigma * math.sqrt(2))
    return 0.5 * (1.0 + math.erf(z))


def infer_lattice_step(values: np.ndarray) -> float:
    uniq = np.unique(np.round(values, 10))
    diffs = np.diff(np.sort(uniq))
    diffs = diffs[diffs > 1e-9]

    if len(diffs) == 0:
        return 1.0

    step = float(np.min(diffs))

    for candidate in [0.5, 1.0, 0.25, 0.1]:
        if abs(step - candidate) < 1e-6:
            return candidate

    return step


def quantize_to_lattice(values: np.ndarray, step: float) -> np.ndarray:
    return np.round(values / step) * step


def build_lattice_grid(values: np.ndarray, mu: float, sigma: float, step: float) -> np.ndarray:
    if sigma <= 0:
        lo = np.min(values)
        hi = np.max(values)
    else:
        lo = min(np.min(values), mu - 6 * sigma)
        hi = max(np.max(values), mu + 6 * sigma)

    lo = math.floor(lo / step) * step
    hi = math.ceil(hi / step) * step

    n = int(round((hi - lo) / step)) + 1
    grid = lo + step * np.arange(n)
    return np.round(grid, 10)


def empirical_pmf(values: np.ndarray, grid: np.ndarray, step: float) -> np.ndarray:
    q = quantize_to_lattice(values, step)
    counts = pd.Series(q).value_counts().sort_index()

    p = np.zeros(len(grid), dtype=float)
    index_map = {float(g): i for i, g in enumerate(grid)}

    for k, v in counts.items():
        k = float(np.round(k, 10))
        if k in index_map:
            p[index_map[k]] = v

    if p.sum() > 0:
        p = p / p.sum()
    return p


def discretized_normal_pmf(grid: np.ndarray, mu: float, sigma: float, step: float) -> np.ndarray:
    if sigma <= 0:
        p = np.zeros(len(grid), dtype=float)
        idx = int(np.argmin(np.abs(grid - mu)))
        p[idx] = 1.0
        return p

    p = np.zeros(len(grid), dtype=float)
    half = 0.5 * step

    for i, g in enumerate(grid):
        left = g - half
        right = g + half
        p[i] = normal_cdf_scalar(right, mu, sigma) - normal_cdf_scalar(left, mu, sigma)

    s = p.sum()
    if s > 0:
        p /= s
    return p


def empirical_cdf_from_pmf(p: np.ndarray) -> np.ndarray:
    return np.cumsum(p)


def skewness(values: np.ndarray, mu: float, sigma: float) -> float:
    if sigma <= 0:
        return float("nan")
    z = (values - mu) / sigma
    return float(np.mean(z ** 3))


def excess_kurtosis(values: np.ndarray, mu: float, sigma: float) -> float:
    if sigma <= 0:
        return float("nan")
    z = (values - mu) / sigma
    return float(np.mean(z ** 4) - 3.0)


def total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.sum(np.abs(p - q)))


def hellinger_distance(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p2 = np.clip(p, eps, 1.0)
    q2 = np.clip(q, eps, 1.0)
    p2 /= p2.sum()
    q2 /= q2.sum()

    m = 0.5 * (p2 + q2)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sum(a * np.log(a / b)))

    return 0.5 * kl(p2, m) + 0.5 * kl(q2, m)


def similarity_score(tv: float, h: float, js: float) -> float:
    score = 100.0
    score -= min(50.0, 140.0 * tv)
    score -= min(30.0, 100.0 * h)
    score -= min(20.0, 80.0 * js)
    return max(0.0, score)


def main():
    raw = load_symbol_data(TARGET_SYMBOL)
    simulated = simulate_current_ash_fair(raw)

    valid = simulated.dropna(subset=["mid_minus_fair"]).copy()
    values = valid["mid_minus_fair"].to_numpy(dtype=float)

    mu = float(np.mean(values))
    sigma = float(np.std(values, ddof=0))
    sk = skewness(values, mu, sigma)
    exk = excess_kurtosis(values, mu, sigma)

    step = infer_lattice_step(values)
    grid = build_lattice_grid(values, mu, sigma, step)

    quantized_values = quantize_to_lattice(values, step)
    value_counts = (
        pd.Series(quantized_values)
        .value_counts()
        .sort_index()
    )

    count_map = {float(k): int(v) for k, v in value_counts.items()}
    counts_on_grid = np.array([count_map.get(float(g), 0) for g in grid], dtype=int)
    percent_on_grid = counts_on_grid / counts_on_grid.sum()

    print("[value counts of mid_price - fair_price]")
    for val, cnt in value_counts.items():
        print(f"{val:>8.3f}: {int(cnt)}")
    print()

    p_emp = empirical_pmf(values, grid, step)
    p_disc_norm = discretized_normal_pmf(grid, mu, sigma, step)

    cdf_emp = empirical_cdf_from_pmf(p_emp)
    cdf_disc_norm = empirical_cdf_from_pmf(p_disc_norm)

    tv = total_variation_distance(p_emp, p_disc_norm)
    h = hellinger_distance(p_emp, p_disc_norm)
    js = js_divergence(p_emp, p_disc_norm)
    sim_score = similarity_score(tv, h, js)

    print("=" * 80)
    print(f"SYMBOL: {TARGET_SYMBOL}")
    print("=" * 80)
    print(f"n = {len(values)}")
    print(f"mean(mid_price - fair_price) = {mu:.6f}")
    print(f"std(mid_price - fair_price)  = {sigma:.6f}")
    print(f"skewness                     = {sk:.6f}")
    print(f"excess kurtosis             = {exk:.6f}")
    print(f"inferred lattice step       = {step:.6f}")
    print()
    print("[discretized normal fit]")
    print(f"total variation distance    = {tv:.6f}")
    print(f"hellinger distance          = {h:.6f}")
    print(f"JS divergence               = {js:.6f}")
    print(f"similarity score            = {sim_score:.2f} / 100")
    print()

    if tv < 0.05 and h < 0.08:
        verdict = "이산화된 정규분포에 꽤 가까움"
    elif tv < 0.10 and h < 0.15:
        verdict = "대체로 이산화된 정규분포와 비슷함"
    elif tv < 0.18:
        verdict = "어느 정도 비슷하지만 차이가 눈에 띔"
    else:
        verdict = "이산화된 정규분포와도 꽤 다름"

    print(f"verdict: {verdict}")
    print()

    # 1) empirical PMF vs discretized normal PMF
    plt.figure(figsize=(14, 7))
    width = step * 0.42

    bars_emp = plt.bar(
        grid - width / 2,
        p_emp,
        width=width,
        alpha=0.7,
        label="empirical PMF",
    )

    plt.bar(
        grid + width / 2,
        p_disc_norm,
        width=width,
        alpha=0.7,
        label="discretized normal PMF",
    )

    # empirical 막대 위에 값 / 개수 / 퍼센티지 표시
    ymax = max(np.max(p_emp), np.max(p_disc_norm))
    text_offset = ymax * 0.01

    for x, y, cnt, pct in zip(grid - width / 2, p_emp, counts_on_grid, percent_on_grid):
        if cnt > 0:
            plt.text(
                x,
                y + text_offset,
                f"{x + width/2:.2f}\n{cnt} ({pct:.1%})",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    plt.xlabel("mid_price - fair_price")
    plt.ylabel("probability mass")
    plt.title(f"{TARGET_SYMBOL}: empirical vs discretized normal PMF")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2) CDF comparison
    plt.figure(figsize=(10, 6))
    plt.step(grid, cdf_emp, where="mid", label="empirical CDF")
    plt.step(grid, cdf_disc_norm, where="mid", label="discretized normal CDF")
    plt.axvline(mu, linewidth=1, linestyle="--", label=f"mean = {mu:.3f}")
    plt.xlabel("mid_price - fair_price")
    plt.ylabel("cumulative probability")
    plt.title(f"{TARGET_SYMBOL}: empirical vs discretized normal CDF")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()