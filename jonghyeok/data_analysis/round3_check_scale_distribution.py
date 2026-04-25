from __future__ import annotations

import json
import math
import re
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Params: 메인 코드와 맞추기
# =========================

DATA_DIR = Path("data_capsule/round3")
OUT_DIR = Path("jonghyeok/data_analysis/output/scale_distribution")

UNDERLYING = "VELVETFRUIT_EXTRACT"

VOUCHERS = [
    "VEV_5000",
    "VEV_5100",
    "VEV_5200",
    "VEV_5300",
]

STRIKES = {
    "VEV_5000": 5000,
    "VEV_5100": 5100,
    "VEV_5200": 5200,
    "VEV_5300": 5300,
}

VALID_BID_ASK_VOLUME = {
    "VELVETFRUIT_EXTRACT": 30,
    "VEV_5000": 13,
    "VEV_5100": 13,
    "VEV_5200": 13,
    "VEV_5300": 13,
    "VEV_5400": 13,
    "VEV_5500": 13,
    "VEV_6000": 5,
    "VEV_6500": 5,
}

DAYS_PER_YEAR = 365.0
N = NormalDist()

# 메인 코드의 window와 같게 설정
THEO_DIFF_MEAN_WINDOW = 50
SCALE_WINDOW = 50

# main 코드가 mean/scale을 현재 tick 포함해서 업데이트한 뒤 z_score 계산하면 True
# main 코드가 이전 tick까지의 mean/scale로 z_score 계산한 뒤 업데이트하면 False
USE_CURRENT_IN_SIGNAL = True

DAY_RE = re.compile(r"day_(-?\d+)")


# =========================
# IO / mid
# =========================

def read_csv_auto(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    if len(df.columns) == 1:
        df = pd.read_csv(path)
    return df


def parse_day(path: Path) -> int:
    m = DAY_RE.search(path.name)
    if not m:
        raise ValueError(f"Cannot parse day from filename: {path.name}")
    return int(m.group(1))


def get_valid_mid(row: pd.Series) -> float | None:
    product = row["product"]
    valid_volume = VALID_BID_ASK_VOLUME[product]

    valid_bid = None
    valid_ask = None

    for i in range(1, 4):
        price = row.get(f"bid_price_{i}", np.nan)
        vol = row.get(f"bid_volume_{i}", np.nan)
        if pd.notna(price) and pd.notna(vol) and vol >= valid_volume:
            valid_bid = float(price)
            break

    for i in range(1, 4):
        price = row.get(f"ask_price_{i}", np.nan)
        vol = row.get(f"ask_volume_{i}", np.nan)
        if pd.notna(price) and pd.notna(vol) and vol >= valid_volume:
            valid_ask = float(price)
            break

    if valid_bid is None or valid_ask is None:
        return None

    return 0.5 * (valid_bid + valid_ask)


def load_price_data() -> pd.DataFrame:
    files = sorted(DATA_DIR.glob("prices_round_3_day_*.csv"), key=parse_day)
    if not files:
        raise FileNotFoundError(f"No prices_round_3_day_*.csv in {DATA_DIR}")

    dfs = []
    for file_idx, path in enumerate(files):
        df = read_csv_auto(path)
        df["parsed_day"] = parse_day(path)
        df["file_idx"] = file_idx
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


# =========================
# IV / BS
# =========================

def load_iv_coeffs() -> list[float]:
    candidates = [
        Path("jonghyeok/data_analysis/outputs/round3_iv_fit/round3_iv_fit_coeffs.json"),
        Path("jonghyeok/data_analysis/output/round3_iv_fit/round3_iv_fit_coeffs.json"),
        Path("jonghyeok/data_analysis/output/round3_iv_fit_coeffs.json"),
    ]

    for path in candidates:
        if path.exists():
            payload = json.loads(path.read_text())
            return [float(x) for x in payload["coeffs"]]

    raise FileNotFoundError(
        "round3_iv_fit_coeffs.json을 못 찾았습니다. "
        "IV fit 코드를 먼저 돌리거나, candidates 경로를 수정하세요."
    )


def bs_call_price(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return max(S - K, 0.0)

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    return S * N.cdf(d1) - K * math.exp(-r * T) * N.cdf(d2)


def timestamp_progress(timestamp: float) -> float:
    return float(timestamp) / 1_000_000.0


def tte_years(file_idx: int, timestamp: float) -> float:
    # sorted historical files: 8d, 7d, 6d
    tte_start_days = 8.0 - file_idx
    remaining_days = tte_start_days - timestamp_progress(timestamp)
    return max(remaining_days / DAYS_PER_YEAR, 1e-9)


def fair_iv(S: float, K: float, T: float, coeffs: list[float]) -> float:
    m = math.log(K / S) / math.sqrt(T)
    return float(np.polyval(coeffs, m))


# =========================
# Main scale distribution
# =========================

def build_theo_diff_df() -> pd.DataFrame:
    coeffs = load_iv_coeffs()
    raw = load_price_data()

    target_products = [UNDERLYING] + VOUCHERS
    raw = raw[raw["product"].isin(target_products)].copy()

    raw["valid_mid"] = raw.apply(get_valid_mid, axis=1)
    raw = raw.dropna(subset=["valid_mid"])

    underlying = raw[raw["product"] == UNDERLYING][
        ["file_idx", "parsed_day", "timestamp", "valid_mid"]
    ].rename(columns={"valid_mid": "S"})

    rows = []

    for voucher in VOUCHERS:
        K = STRIKES[voucher]

        opt = raw[raw["product"] == voucher][
            ["file_idx", "parsed_day", "timestamp", "valid_mid"]
        ].rename(columns={"valid_mid": "V"})

        joined = opt.merge(
            underlying,
            on=["file_idx", "parsed_day", "timestamp"],
            how="inner",
        )

        for _, r in joined.iterrows():
            S = float(r["S"])
            V = float(r["V"])
            timestamp = float(r["timestamp"])
            file_idx = int(r["file_idx"])

            T = tte_years(file_idx, timestamp)
            sigma = fair_iv(S, K, T, coeffs)
            theo = bs_call_price(S, K, T, sigma)

            rows.append({
                "product": voucher,
                "file_idx": file_idx,
                "day": int(r["parsed_day"]),
                "timestamp": int(timestamp),
                "S": S,
                "V": V,
                "K": K,
                "T": T,
                "fair_iv": sigma,
                "theo_price": theo,
                "theo_diff": V - theo,
            })

    return pd.DataFrame(rows)


def add_main_scale(df: pd.DataFrame) -> pd.DataFrame:
    out = []

    alpha_mean = 2 / (THEO_DIFF_MEAN_WINDOW + 1)
    alpha_scale = 2 / (SCALE_WINDOW + 1)

    for product, pdf in df.groupby("product"):
        for day, g in pdf.groupby("day"):
            g = g.sort_values("timestamp").copy()

            x = g["theo_diff"].astype(float)

            if USE_CURRENT_IN_SIGNAL:
                mean = x.ewm(alpha=alpha_mean, adjust=False).mean()
                dev = x - mean
                raw_scale = dev.abs().ewm(alpha=alpha_scale, adjust=False).mean()
            else:
                mean_after = x.ewm(alpha=alpha_mean, adjust=False).mean()
                mean = mean_after.shift(1)

                dev = x - mean
                scale_after = dev.abs().ewm(alpha=alpha_scale, adjust=False).mean()
                raw_scale = scale_after.shift(1)

            g["theo_diff_mean"] = mean
            g["theo_diff_dev"] = dev
            g["raw_scale"] = raw_scale
            out.append(g)

    return pd.concat(out, ignore_index=True)


def summarize_scale(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for product, g in df.groupby("product"):
        s = g["raw_scale"].replace([np.inf, -np.inf], np.nan).dropna()

        rows.append({
            "product": product,
            "n": len(s),
            "mean": s.mean(),
            "std": s.std(),
            "min": s.min(),
            "q001": s.quantile(0.001),
            "q005": s.quantile(0.005),
            "q01": s.quantile(0.01),
            "q05": s.quantile(0.05),
            "q10": s.quantile(0.10),
            "q25": s.quantile(0.25),
            "median": s.quantile(0.50),
            "q75": s.quantile(0.75),
            "q90": s.quantile(0.90),
            "q95": s.quantile(0.95),
            "q99": s.quantile(0.99),
            "max": s.max(),
        })

    return pd.DataFrame(rows)


def plot_histograms(df: pd.DataFrame) -> None:
    for product, g in df.groupby("product"):
        s = g["raw_scale"].replace([np.inf, -np.inf], np.nan).dropna()

        plt.figure(figsize=(10, 6))
        plt.hist(s, bins=100)
        plt.xlabel("raw_scale")
        plt.ylabel("count")
        plt.title(f"{product} raw_scale distribution")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"{product}_raw_scale_hist.png", dpi=160)
        plt.close()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = build_theo_diff_df()
    df = add_main_scale(df)

    detail_path = OUT_DIR / "main_scale_detail.csv"
    summary_path = OUT_DIR / "main_scale_summary.csv"

    df.to_csv(detail_path, index=False)

    summary = summarize_scale(df)
    summary.to_csv(summary_path, index=False)

    plot_histograms(df)

    print("\n=== MAIN RAW SCALE DISTRIBUTION ===")
    print(summary.to_string(index=False))
    print(f"\nSaved detail : {detail_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved plots  : {OUT_DIR}/*_raw_scale_hist.png")


if __name__ == "__main__":
    main()