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

    for col in ["day", "timestamp", "bid_price_1", "ask_price_1"]:
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

            # 현재 코드와 동일한 mid_price 계산
            mid_price = None
            if best_bid is not None and best_ask is not None:
                mid_price = 0.5 * (best_bid + best_ask)

            if mid_price is None:
                if best_bid is not None:
                    mid_price = best_bid + MID_FALLBACK_OFFSET
                elif best_ask is not None:
                    mid_price = best_ask - MID_FALLBACK_OFFSET

            fair_price = None
            if len(past_few_mid_history) >= MIN_HISTORY_LENGTH:
                fair_price = median_of_list(past_few_mid_history)

            diff = None
            if mid_price is not None and fair_price is not None:
                diff = mid_price - fair_price

            rows.append(
                {
                    "day": int(row["day"]),
                    "timestamp": int(row["timestamp"]),
                    "best_bid": best_bid,
                    "best_ask": best_ask,
                    "mid_price_like_trader": mid_price,
                    "fair_price_like_trader": fair_price,
                    "mid_minus_fair": diff,
                    "history_len_before_update": len(past_few_mid_history),
                }
            )

            # 현재 코드와 동일하게 fair 계산 후 현재 mid를 history에 추가
            if mid_price is not None:
                if len(past_few_mid_history) < MAX_HISTORY_LENGTH:
                    past_few_mid_history.append(mid_price)
                else:
                    past_few_mid_history.pop(0)
                    past_few_mid_history.append(mid_price)

    return pd.DataFrame(rows)


def normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.zeros_like(x, dtype=float)
    z = (x - mu) / sigma
    return np.exp(-0.5 * z * z) / (sigma * math.sqrt(2 * math.pi))


def normal_cdf_scalar(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        return 1.0 if x >= mu else 0.0
    z = (x - mu) / (sigma * math.sqrt(2))
    return 0.5 * (1 + math.erf(z))


def ks_statistic_against_fitted_normal(values: np.ndarray, mu: float, sigma: float) -> float:
    x = np.sort(values)
    n = len(x)
    if n == 0:
        return float("nan")

    empirical_upper = np.arange(1, n + 1) / n
    empirical_lower = np.arange(0, n) / n
    fitted = np.array([normal_cdf_scalar(v, mu, sigma) for v in x])

    d_plus = np.max(empirical_upper - fitted)
    d_minus = np.max(fitted - empirical_lower)
    return float(max(d_plus, d_minus))


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


def normal_similarity_score(ks: float, skew: float, ex_kurt: float) -> float:
    """
    대충 0~100 점수.
    높을수록 정규분포에 더 가까움.
    """
    score = 100.0
    score -= min(60.0, 220.0 * ks)
    score -= min(20.0, 12.0 * abs(skew))
    score -= min(20.0, 6.0 * abs(ex_kurt))
    return max(0.0, score)


def main():
    raw = load_symbol_data(TARGET_SYMBOL)
    simulated = simulate_current_ash_fair(raw)

    valid = simulated.dropna(subset=["mid_minus_fair"]).copy()
    values = valid["mid_minus_fair"].to_numpy(dtype=float)

    mu = float(np.mean(values))
    sigma = float(np.std(values, ddof=0))
    ks = ks_statistic_against_fitted_normal(values, mu, sigma)
    sk = skewness(values, mu, sigma)
    exk = excess_kurtosis(values, mu, sigma)
    similarity = normal_similarity_score(ks, sk, exk)

    print("=" * 80)
    print(f"SYMBOL: {TARGET_SYMBOL}")
    print("=" * 80)
    print(f"n = {len(values)}")
    print(f"mean(mid_price - fair_price) = {mu:.6f}")
    print(f"std(mid_price - fair_price)  = {sigma:.6f}")
    print(f"skewness                     = {sk:.6f}")
    print(f"excess kurtosis             = {exk:.6f}")
    print(f"KS statistic vs fitted N    = {ks:.6f}")
    print(f"normal similarity score     = {similarity:.2f} / 100")
    print()

    if ks < 0.03 and abs(sk) < 0.2 and abs(exk) < 0.5:
        verdict = "정규분포에 꽤 가까움"
    elif ks < 0.06 and abs(sk) < 0.5 and abs(exk) < 1.0:
        verdict = "대체로 정규분포와 비슷함"
    elif ks < 0.10:
        verdict = "어느 정도 비슷하지만 차이가 눈에 띔"
    else:
        verdict = "정규분포와 꽤 다름"

    print(f"verdict: {verdict}")
    print()

    # 히스토그램 + fitted normal overlay
    plt.figure(figsize=(10, 6))
    counts, bins, _ = plt.hist(
        values,
        bins=80,
        density=True,
        alpha=0.6,
        label="empirical distribution",
    )

    x_min = np.percentile(values, 0.5)
    x_max = np.percentile(values, 99.5)
    x_grid = np.linspace(x_min, x_max, 600)
    pdf = normal_pdf(x_grid, mu, sigma)

    plt.plot(x_grid, pdf, linewidth=2, label=f"fitted normal N({mu:.3f}, {sigma:.3f}²)")
    plt.axvline(mu, linewidth=1, linestyle="--", label=f"mean = {mu:.3f}")
    plt.xlabel("mid_price - fair_price")
    plt.ylabel("density")
    plt.title(f"{TARGET_SYMBOL}: distribution of mid_price - fair_price")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # QQ-style plot
    plt.figure(figsize=(8, 8))
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    probs = (np.arange(1, n + 1) - 0.5) / n

    # inverse normal CDF 없이 샘플 quantile 근사:
    # np.quantile of simulated normal sample
    rng = np.random.default_rng(42)
    normal_sample = rng.normal(mu, sigma, size=max(200000, n))
    theo_q = np.quantile(normal_sample, probs)

    plt.scatter(theo_q, sorted_vals, s=8, alpha=0.4)
    lo = min(theo_q.min(), sorted_vals.min())
    hi = max(theo_q.max(), sorted_vals.max())
    plt.plot([lo, hi], [lo, hi], linewidth=1)
    plt.xlabel("theoretical normal quantiles")
    plt.ylabel("empirical quantiles")
    plt.title(f"{TARGET_SYMBOL}: QQ plot of (mid_price - fair_price)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()