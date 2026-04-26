# jonghyeok/data_analysis/round3_make_day_voucher_history_map.py

from __future__ import annotations

import math
from pathlib import Path
from pprint import pformat

import numpy as np
import pandas as pd


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

UNDERLYING = "VELVETFRUIT_EXTRACT"

IV_CURVE_FIT_VOUCHERS = {
    "VEV_5000": 5000,
    "VEV_5100": 5100,
    "VEV_5200": 5200,
    "VEV_5300": 5300,
    "VEV_5400": 5400,
    "VEV_5500": 5500,
}

VALID_BID_ASK_VOLUME = {
    "VELVETFRUIT_EXTRACT": 15,
    "VEV_5000": 6,
    "VEV_5100": 6,
    "VEV_5200": 6,
    "VEV_5300": 5,
    "VEV_5400": 5,
    "VEV_5500": 5,
}

SMILE_WINDOW_PER_VOUCHER = 300
PRELOAD_POINTS_PER_VOUCHER = 100

DAYS_PER_YEAR = 365.0
MIN_IV = 1e-3
MAX_IV = 3.0


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_call_with_greeks(S: float, K: float, T: float, sigma: float):
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return max(S - K, 0.0), (1.0 if S > K else 0.0), 0.0

    sqrt_t = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t

    price = S * _norm_cdf(d1) - K * _norm_cdf(d2)
    delta = _norm_cdf(d1)
    vega = S * _norm_pdf(d1) * sqrt_t

    return price, delta, vega


def implied_vol(
    V: float,
    S: float,
    K: float,
    T: float,
    tol: float = 1e-4,
    max_iter: int = 50,
) -> float | None:
    if T <= 0 or V <= 0 or S <= 0 or K <= 0:
        return None

    intrinsic = max(S - K, 0.0)

    if V < intrinsic - 1e-2 or V > S + 1e-2:
        return None

    lo, hi = MIN_IV, MAX_IV

    f_lo = bs_call_with_greeks(S, K, T, lo)[0] - V
    f_hi = bs_call_with_greeks(S, K, T, hi)[0] - V

    if f_lo * f_hi > 0:
        return None

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = bs_call_with_greeks(S, K, T, mid)[0] - V

        if abs(f_mid) < tol:
            return mid

        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid

    return 0.5 * (lo + hi)


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


def get_tte_years(timestamp: int, day_num: int) -> float:
    progress_days = timestamp / 1_000_000.0
    remaining_days = max(8.0 - day_num - progress_days, 1e-9)
    return remaining_days / DAYS_PER_YEAR


def load_round3_prices() -> pd.DataFrame:
    project_root = Path(__file__).resolve().parents[2]

    candidate_dirs = [
        project_root / "data_capsule" / "round3",
        project_root / "data" / "round3",
        project_root / "jonghyeok" / "data" / "round3",
        project_root / "jonghyeok" / "data_analysis" / "data" / "round3",
    ]

    csv_paths = []
    for d in candidate_dirs:
        if d.exists():
            csv_paths = sorted(d.glob("prices_round_3_day_*.csv"))
            if csv_paths:
                break

    if not csv_paths:
        raise FileNotFoundError(
            "prices_round_3_day_*.csv 파일을 찾지 못했습니다. "
            "candidate_dirs를 현재 데이터 위치에 맞게 수정하세요."
        )

    frames = []
    target_products = {UNDERLYING, *IV_CURVE_FIT_VOUCHERS.keys()}

    for path in csv_paths:
        df = read_price_file(path)
        df = df[df["product"].isin(target_products)].copy()
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
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["day", "timestamp", "product"]).copy()
    df["day"] = df["day"].astype(int)
    df["timestamp"] = df["timestamp"].astype(int)

    df["valid_mid"] = df.apply(get_valid_mid, axis=1)
    df = df.dropna(subset=["valid_mid"]).copy()

    df = df.sort_values(["day", "timestamp", "product"]).reset_index(drop=True)
    return df


def build_history_from_source_day(df: pd.DataFrame, source_day: int) -> dict[str, list[tuple[float, float]]]:
    day_df = df[df["day"] == source_day].copy()

    underlying_df = (
        day_df[day_df["product"] == UNDERLYING][["timestamp", "valid_mid"]]
        .rename(columns={"valid_mid": "underlying_mid"})
        .copy()
    )

    history: dict[str, list[tuple[float, float]]] = {
        product: [] for product in IV_CURVE_FIT_VOUCHERS
    }

    for product, K in IV_CURVE_FIT_VOUCHERS.items():
        voucher_df = (
            day_df[day_df["product"] == product][["timestamp", "valid_mid"]]
            .rename(columns={"valid_mid": "option_mid"})
            .copy()
        )

        merged = pd.merge(voucher_df, underlying_df, on="timestamp", how="inner")
        merged = merged.sort_values("timestamp")

        points = []

        for _, row in merged.iterrows():
            timestamp = int(row["timestamp"])
            S = float(row["underlying_mid"])
            V = float(row["option_mid"])
            T = get_tte_years(timestamp, source_day)

            if S <= 0 or V <= 0 or T <= 0:
                continue

            iv = implied_vol(V=V, S=S, K=float(K), T=T)

            if iv is None or not math.isfinite(iv):
                continue

            sqrt_t = math.sqrt(T)
            m = math.log(float(K) / S) / sqrt_t

            if not math.isfinite(m):
                continue

            points.append((round(float(m), 10), round(float(iv), 10)))

        max_len = min(PRELOAD_POINTS_PER_VOUCHER, SMILE_WINDOW_PER_VOUCHER)
        history[product] = points[-max_len:]

    return history


def make_day_voucher_history_map(df: pd.DataFrame) -> dict[int, dict[str, list[tuple[float, float]]]]:
    days = sorted(df["day"].unique())

    if not days:
        return {}

    out = {}

    # 첫 day는 이전 day가 없으므로 key 생성 안 함.
    for i in range(1, len(days)):
        target_day = int(days[i])
        source_day = int(days[i - 1])
        out[target_day] = build_history_from_source_day(df, source_day)

    # 실전/다음 day용: 마지막 historical day를 이용해 max_day + 1 history 생성.
    last_day = int(days[-1])
    out[last_day + 1] = build_history_from_source_day(df, last_day)

    return out


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    out_dir = project_root / "jonghyeok" / "data_analysis" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_round3_prices()
    history_map = make_day_voucher_history_map(df)

    output_path = out_dir / "round3_day_voucher_history_map.py"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Auto-generated by round3_make_day_voucher_history_map.py\n")
        f.write("# Structure: DAY_VOUCHER_HISTORY_MAP[day_num][voucher] = list[(m, iv)]\n\n")
        f.write("DAY_VOUCHER_HISTORY_MAP = ")
        f.write(pformat(history_map, width=120, sort_dicts=False))
        f.write("\n")

    print("[SAVED]", output_path)
    print()
    print("생성된 파일에서 아래 변수 전체를 복사해서 Trader 코드에 붙여넣으면 됩니다.")
    print("DAY_VOUCHER_HISTORY_MAP keys:", sorted(history_map.keys()))

    for day, voucher_history in history_map.items():
        sizes = {p: len(v) for p, v in voucher_history.items()}
        print(f"day {day}: {sizes}")


if __name__ == "__main__":
    main()