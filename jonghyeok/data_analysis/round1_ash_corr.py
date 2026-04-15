import math
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_SYMBOL = "ASH_COATED_OSMIUM"
ANALYSIS_PRICE_MODE = "quote_mid_only"

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

WINDOWS = [80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
EPSILONS = [0.25, 0.5, 0.75]

BOUNCE_INTERCEPT = 0.008043
BOUNCE_SLOPE = -0.496676


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


def load_symbol_data(symbol: str) -> tuple[pd.DataFrame, dict]:
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

    numeric_cols = ["day", "timestamp", "bid_price_1", "ask_price_1", "mid_price"]
    for col in numeric_cols:
        if col in raw.columns:
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

    raw["provided_mid_clean"] = raw["mid_price"].where(raw["mid_price"] > 0, np.nan)

    if ANALYSIS_PRICE_MODE == "quote_mid_only":
        raw["analysis_mid"] = raw["quote_mid"]
    elif ANALYSIS_PRICE_MODE == "all_valid_mid":
        raw["analysis_mid"] = raw["quote_mid"].fillna(raw["provided_mid_clean"])
    else:
        raise ValueError(f"Unknown ANALYSIS_PRICE_MODE: {ANALYSIS_PRICE_MODE}")

    base_steps = {}
    for day, g in raw.groupby("day", sort=False):
        base_steps[day] = infer_base_step(g["timestamp"])
    raw["base_step"] = raw["day"].map(base_steps)

    valid = raw[raw["analysis_mid"].notna()].copy()
    valid = valid.sort_values(["day", "timestamp"]).reset_index(drop=True)

    quality = {
        "analysis_price_mode": ANALYSIS_PRICE_MODE,
        "total_rows": int(len(raw)),
        "analysis_rows": int(len(valid)),
    }

    return valid, quality


def compute_micro_bounce_scan(df: pd.DataFrame, lags=(1, 2, 3, 5, 8, 10)) -> pd.DataFrame:
    rows = []

    for lag in lags:
        xs = []
        ys = []

        for _, g in df.groupby("day", sort=False):
            g = g.sort_values("timestamp").copy()

            p = g["analysis_mid"]
            ts = g["timestamp"]
            step = int(g["base_step"].iloc[0])

            ret1 = p.diff()
            gap1 = ts.diff()
            ret1 = ret1.where(gap1 == step)

            sub = pd.DataFrame({
                "timestamp": ts,
                "ret1": ret1,
            }).dropna()

            future_ret = sub["ret1"].shift(-lag)
            future_gap = sub["timestamp"].shift(-lag) - sub["timestamp"]

            mask = future_ret.notna() & (future_gap == lag * step)
            if mask.any():
                xs.append(sub.loc[mask, "ret1"].to_numpy())
                ys.append(future_ret.loc[mask].to_numpy())

        if not xs:
            rows.append({"lag": lag, "n": 0, "rho": np.nan, "t_stat": np.nan})
            continue

        x = np.concatenate(xs)
        y = np.concatenate(ys)

        n = len(x)
        rho = np.corrcoef(x, y)[0, 1] if n >= 3 else np.nan
        if pd.notna(rho) and abs(rho) < 1 and n >= 3:
            t_stat = rho * math.sqrt((n - 2) / (1 - rho ** 2))
        else:
            t_stat = np.nan

        rows.append({
            "lag": lag,
            "n": n,
            "rho": rho,
            "t_stat": t_stat,
        })

    return pd.DataFrame(rows)


def ema_from_alpha(x: pd.Series, alpha: float) -> pd.Series:
    out = []
    prev = None
    for v in x:
        if pd.isna(v):
            out.append(np.nan)
            continue
        if prev is None:
            prev = v
        else:
            prev = alpha * v + (1 - alpha) * prev
        out.append(prev)
    return pd.Series(out, index=x.index)


def evaluate_ash_model(df: pd.DataFrame, window: int, epsilon: float) -> dict:
    alpha = 2 / (window + 1)

    fair_rows = []

    for _, g in df.groupby("day", sort=False):
        g = g.sort_values("timestamp").copy()

        mid = g["analysis_mid"]
        ts = g["timestamp"]
        step = int(g["base_step"].iloc[0])

        ema = ema_from_alpha(mid, alpha)
        prev_mid = mid.shift(1)
        prev_gap = ts - ts.shift(1)

        r_t = (mid - prev_mid).where(prev_gap == step)

        bounce_pred = mid + BOUNCE_INTERCEPT + BOUNCE_SLOPE * r_t
        bounce_pred = bounce_pred.fillna(mid)

        fair = epsilon * ema + (1 - epsilon) * bounce_pred

        future_1 = mid.shift(-1).where(ts.shift(-1) - ts == step)
        future_2 = mid.shift(-2).where(ts.shift(-2) - ts == 2 * step)
        future_3 = mid.shift(-3).where(ts.shift(-3) - ts == 3 * step)
        future_5 = mid.shift(-5).where(ts.shift(-5) - ts == 5 * step)

        target_3 = (future_1 + future_2 + future_3) / 3

        tmp = pd.DataFrame({
            "mid": mid,
            "fair": fair,
            "future_1": future_1,
            "future_3avg": target_3,
            "future_5": future_5,
        }).dropna(subset=["mid", "fair"])

        fair_rows.append(tmp)

    all_df = pd.concat(fair_rows, ignore_index=True)

    def mae(a, b):
        z = pd.concat([a, b], axis=1).dropna()
        if len(z) == 0:
            return np.nan
        return np.mean(np.abs(z.iloc[:, 0] - z.iloc[:, 1]))

    def corr_signal(target):
        z = pd.DataFrame({
            "signal": all_df["fair"] - all_df["mid"],
            "ret": target - all_df["mid"],
        }).dropna()
        if len(z) < 3:
            return np.nan
        return z["signal"].corr(z["ret"])

    def hit_rate(target):
        z = pd.DataFrame({
            "signal": all_df["fair"] - all_df["mid"],
            "ret": target - all_df["mid"],
        }).dropna()
        if len(z) == 0:
            return np.nan
        nonzero = z[z["signal"] != 0]
        if len(nonzero) == 0:
            return np.nan
        return np.mean(np.sign(nonzero["signal"]) == np.sign(nonzero["ret"]))

    return {
        "window": window,
        "epsilon": epsilon,
        "n_rows": len(all_df),
        "mae_1": mae(all_df["fair"], all_df["future_1"]),
        "mae_3avg": mae(all_df["fair"], all_df["future_3avg"]),
        "mae_5": mae(all_df["fair"], all_df["future_5"]),
        "corr_1": corr_signal(all_df["future_1"]),
        "corr_3avg": corr_signal(all_df["future_3avg"]),
        "corr_5": corr_signal(all_df["future_5"]),
        "hit_1": hit_rate(all_df["future_1"]),
        "hit_3avg": hit_rate(all_df["future_3avg"]),
        "hit_5": hit_rate(all_df["future_5"]),
    }


def main():
    df, quality = load_symbol_data(TARGET_SYMBOL)

    print("[DATA QUALITY]")
    for k, v in quality.items():
        print(f"{k}: {v}")
    print()

    print("[MICRO BOUNCE CHECK]")
    print(compute_micro_bounce_scan(df).to_string(index=False))
    print()

    rows = []
    for window in WINDOWS:
        for epsilon in EPSILONS:
            rows.append(evaluate_ash_model(df, window, epsilon))

    result = pd.DataFrame(rows)

    # primary: mae_3avg ascending
    # secondary: corr_3avg descending
    # tertiary: hit_3avg descending
    result = result.sort_values(
        ["mae_3avg", "corr_3avg", "hit_3avg"],
        ascending=[True, False, False],
    )

    print("[WINDOW / EPSILON RANKING]")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()