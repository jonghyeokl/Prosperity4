import math
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_SYMBOL = "ASH_COATED_OSMIUM"

# 기본은 quote가 있는 row만 분석
# "quote_mid_only" / "all_valid_mid"
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

WINDOWS = [5, 8, 10, 15, 20, 30, 40, 60, 80, 120]
HORIZONS = [1, 2, 3, 5, 8, 10, 15, 20, 30]
Z_THRESHOLDS = [1.0, 1.5, 2.0, 2.5]

MIN_SAMPLE_PER_CELL = 200
MIN_EXTREME_SAMPLE = 20
MIN_ROLL_STD = 1.0
Z_CAP = 8.0
Y_CAP = 8.0
TOP_K = 20


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

    # 정상 quote가 있을 때만 quote_mid 사용
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

    # mid_price는 0 이하이면 placeholder로 간주
    raw["provided_mid_clean"] = raw["mid_price"].where(raw["mid_price"] > 0, np.nan)

    raw["mid_source"] = np.select(
        [
            raw["quote_mid"].notna(),
            raw["quote_mid"].isna() & raw["provided_mid_clean"].notna(),
        ],
        [
            "quote_mid",
            "provided_mid_only",
        ],
        default="missing_or_placeholder",
    )

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

    quality = {
        "analysis_price_mode": ANALYSIS_PRICE_MODE,
        "total_rows": int(len(raw)),
        "quote_mid_rows": int((raw["mid_source"] == "quote_mid").sum()),
        "provided_mid_only_rows": int((raw["mid_source"] == "provided_mid_only").sum()),
        "missing_or_placeholder_rows": int((raw["mid_source"] == "missing_or_placeholder").sum()),
        "analysis_rows": int(raw["analysis_mid"].notna().sum()),
        "rows_dropped_from_analysis": int(raw["analysis_mid"].isna().sum()),
    }

    valid = raw[raw["analysis_mid"].notna()].copy()
    valid = valid.sort_values(["day", "timestamp"]).reset_index(drop=True)

    return valid, quality


def ols_stats(x: np.ndarray, y: np.ndarray) -> dict:
    n = len(x)
    if n < 3:
        return {"n": n, "alpha": np.nan, "beta": np.nan, "corr": np.nan, "t_stat": np.nan}

    x_mean = x.mean()
    y_mean = y.mean()

    xc = x - x_mean
    yc = y - y_mean

    sxx = np.sum(xc ** 2)
    syy = np.sum(yc ** 2)
    sxy = np.sum(xc * yc)

    if sxx <= 0 or syy <= 0:
        return {"n": n, "alpha": np.nan, "beta": np.nan, "corr": np.nan, "t_stat": np.nan}

    beta = sxy / sxx
    alpha = y_mean - beta * x_mean

    residual = y - (alpha + beta * x)
    rss = np.sum(residual ** 2)

    if n <= 2:
        t_stat = np.nan
    else:
        sigma2 = rss / (n - 2)
        se_beta = math.sqrt(sigma2 / sxx) if sxx > 0 else np.nan
        t_stat = beta / se_beta if se_beta and se_beta > 0 else np.nan

    corr = sxy / math.sqrt(sxx * syy)

    return {
        "n": n,
        "alpha": alpha,
        "beta": beta,
        "corr": corr,
        "t_stat": t_stat,
    }


def sign_opposite(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b) < 0


def classify_cell(row: dict) -> tuple[int, str]:
    score = 0

    beta = row["beta"]
    corr = row["corr"]
    t_stat = row["t_stat"]
    full_hit_rate = row["full_hit_rate"]

    ext15 = row["extreme_hit_rate_z1.5"]
    ext15_n = row["extreme_n_z1.5"]

    ext20 = row["extreme_hit_rate_z2.0"]
    ext20_n = row["extreme_n_z2.0"]

    if pd.notna(beta) and beta < 0:
        score += 1
    else:
        return 0, "none"

    if pd.notna(corr):
        if corr <= -0.05:
            score += 2
        elif corr <= -0.03:
            score += 1

    if pd.notna(t_stat):
        if t_stat <= -3.0:
            score += 2
        elif t_stat <= -2.0:
            score += 1

    if pd.notna(full_hit_rate) and full_hit_rate >= 0.53:
        score += 1

    if pd.notna(ext15) and ext15_n >= MIN_EXTREME_SAMPLE and ext15 >= 0.58:
        score += 1

    if pd.notna(ext20) and ext20_n >= MIN_EXTREME_SAMPLE and ext20 >= 0.62:
        score += 1

    if score >= 6:
        return score, "strong"
    if score >= 4:
        return score, "medium"
    if score >= 2:
        return score, "weak"
    return score, "none"


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
            rows.append({"lag": lag, "n": 0, "rho": np.nan, "t_stat": np.nan, "label": "none"})
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
            "label": "none",
        })

    out = pd.DataFrame(rows)

    lag1 = out[out["lag"] == 1]
    if len(lag1) == 1:
        lag1_rho = lag1.iloc[0]["rho"]
        lag1_t = lag1.iloc[0]["t_stat"]

        other = out[out["lag"].isin([2, 3, 5])]
        if (
            pd.notna(lag1_rho)
            and pd.notna(lag1_t)
            and lag1_rho <= -0.30
            and lag1_t <= -10
            and other["rho"].abs().fillna(999).max() <= 0.05
        ):
            out.loc[out["lag"] == 1, "label"] = "micro_bounce"
        elif (
            pd.notna(lag1_rho)
            and pd.notna(lag1_t)
            and lag1_rho <= -0.15
            and lag1_t <= -5
        ):
            out.loc[out["lag"] == 1, "label"] = "weak_micro_bounce"

    return out


def compute_grid_scan(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for window in WINDOWS:
        for horizon in HORIZONS:
            x_all = []
            zf_all = []

            for _, g in df.groupby("day", sort=False):
                g = g.sort_values("timestamp").copy()

                p = g["analysis_mid"]
                ts = g["timestamp"]
                step = int(g["base_step"].iloc[0])

                roll_mean = p.rolling(window=window, min_periods=window).mean()
                roll_std = p.rolling(window=window, min_periods=window).std(ddof=0)

                z_now = (p - roll_mean) / roll_std

                future_price = p.shift(-horizon)
                future_mean = roll_mean.shift(-horizon)
                future_std = roll_std.shift(-horizon)
                future_gap = ts.shift(-horizon) - ts

                z_future = (future_price - future_mean) / future_std

                mask = (
                    z_now.notna()
                    & z_future.notna()
                    & roll_std.notna()
                    & future_std.notna()
                    & (roll_std >= MIN_ROLL_STD)
                    & (future_std >= MIN_ROLL_STD)
                    & (future_gap == horizon * step)
                )

                if not mask.any():
                    continue

                x = z_now.loc[mask].to_numpy()
                zf = z_future.loc[mask].to_numpy()

                x = np.clip(x, -Z_CAP, Z_CAP)
                zf = np.clip(zf, -Z_CAP, Z_CAP)

                x_all.append(x)
                zf_all.append(zf)

            if not x_all:
                row = {
                    "window": window,
                    "horizon": horizon,
                    "n": 0,
                    "alpha": np.nan,
                    "phi": np.nan,
                    "corr": np.nan,
                    "t_stat": np.nan,
                    "extreme_n": 0,
                    "shrink_rate": np.nan,
                    "flip_rate": np.nan,
                    "same_sign_rate": np.nan,
                    "median_abs_ratio": np.nan,
                    "score": 0,
                    "label": "none",
                }
                rows.append(row)
                continue

            x = np.concatenate(x_all)
            zf = np.concatenate(zf_all)

            if len(x) < MIN_SAMPLE_PER_CELL:
                row = {
                    "window": window,
                    "horizon": horizon,
                    "n": len(x),
                    "alpha": np.nan,
                    "phi": np.nan,
                    "corr": np.nan,
                    "t_stat": np.nan,
                    "extreme_n": 0,
                    "shrink_rate": np.nan,
                    "flip_rate": np.nan,
                    "same_sign_rate": np.nan,
                    "median_abs_ratio": np.nan,
                    "score": 0,
                    "label": "none",
                }
                rows.append(row)
                continue

            stats = ols_stats(x, zf)
            phi = stats["beta"]
            corr = stats["corr"]
            t_stat = stats["t_stat"]

            ext_mask = np.abs(x) >= 1.5
            extreme_n = int(ext_mask.sum())

            if extreme_n >= MIN_EXTREME_SAMPLE:
                x_ext = x[ext_mask]
                zf_ext = zf[ext_mask]

                shrink_rate = np.mean(np.abs(zf_ext) < np.abs(x_ext))
                flip_rate = np.mean(np.sign(zf_ext) == -np.sign(x_ext))
                same_sign_rate = np.mean(np.sign(zf_ext) == np.sign(x_ext))
                median_abs_ratio = np.median(
                    np.abs(zf_ext) / np.maximum(np.abs(x_ext), 1e-12)
                )
            else:
                shrink_rate = np.nan
                flip_rate = np.nan
                same_sign_rate = np.nan
                median_abs_ratio = np.nan

            # classification
            score = 0
            label = "none"

            # 1) bounce-like
            if (
                pd.notna(phi)
                and pd.notna(corr)
                and phi < -0.05
                and corr < -0.05
                and pd.notna(flip_rate)
                and flip_rate >= 0.65
            ):
                label = "bounce_like"
                score = 0

            # 2) gradual mean reversion
            elif (
                pd.notna(phi)
                and pd.notna(corr)
                and pd.notna(shrink_rate)
                and pd.notna(median_abs_ratio)
                and 0.05 <= phi < 0.85
                and corr > 0.05
                and shrink_rate >= 0.60
                and median_abs_ratio < 0.90
            ):
                score = 2

                if phi < 0.60:
                    score += 1
                if corr >= 0.20:
                    score += 1
                if shrink_rate >= 0.65:
                    score += 1
                if median_abs_ratio < 0.80:
                    score += 1
                if pd.notna(same_sign_rate) and 0.35 <= same_sign_rate <= 0.90:
                    score += 1

                if score >= 6:
                    label = "strong_gradual_mr"
                elif score >= 4:
                    label = "medium_gradual_mr"
                else:
                    label = "weak_gradual_mr"

            row = {
                "window": window,
                "horizon": horizon,
                "n": stats["n"],
                "alpha": stats["alpha"],
                "phi": phi,
                "corr": corr,
                "t_stat": t_stat,
                "extreme_n": extreme_n,
                "shrink_rate": shrink_rate,
                "flip_rate": flip_rate,
                "same_sign_rate": same_sign_rate,
                "median_abs_ratio": median_abs_ratio,
                "score": score,
                "label": label,
            }
            rows.append(row)

    return pd.DataFrame(rows)


def summarize_by_horizon(grid: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for horizon, g in grid.groupby("horizon", sort=True):
        gradual = g[g["label"].isin(["strong_gradual_mr", "medium_gradual_mr"])]
        bounce = g[g["label"] == "bounce_like"]

        best = g.sort_values(
            ["score", "corr"],
            ascending=[False, False],
        ).iloc[0]

        rows.append({
            "horizon": horizon,
            "gradual_mr_windows": int(len(gradual)),
            "bounce_like_windows": int(len(bounce)),
            "best_score": int(best["score"]) if pd.notna(best["score"]) else 0,
            "best_label": best["label"],
            "best_window": int(best["window"]),
            "best_phi": best["phi"],
            "best_corr": best["corr"],
            "best_shrink_rate": best["shrink_rate"],
            "best_flip_rate": best["flip_rate"],
            "local_gradual_mr": len(gradual) >= 3,
            "local_bounce_like": len(bounce) >= 3,
        })

    return pd.DataFrame(rows).sort_values("horizon").reset_index(drop=True)


def overall_verdict(micro: pd.DataFrame, horizon_summary: pd.DataFrame) -> str:
    micro_bounce = (micro["label"] == "micro_bounce").any()
    weak_micro_bounce = (micro["label"] == "weak_micro_bounce").any()

    n_gradual = int(horizon_summary["local_gradual_mr"].sum())
    n_bounce_like = int(horizon_summary["local_bounce_like"].sum())

    if micro_bounce and n_gradual >= 3:
        return "micro bounce + broad gradual mean reversion"
    if micro_bounce and n_gradual >= 1:
        return "micro bounce + local gradual mean reversion"
    if micro_bounce and n_bounce_like >= 1:
        return "micro bounce only"
    if micro_bounce:
        return "micro bounce only"

    if weak_micro_bounce and n_gradual >= 3:
        return "weak micro bounce + broad gradual mean reversion"
    if weak_micro_bounce and n_gradual >= 1:
        return "weak micro bounce + local gradual mean reversion"

    if n_gradual >= 3:
        return "broad gradual mean reversion"
    if n_gradual >= 1:
        return "local gradual mean reversion"
    if n_bounce_like >= 1:
        return "bounce-like only"

    return "no clear mean reversion"

def print_report(symbol: str, quality: dict, micro: pd.DataFrame, grid: pd.DataFrame, horizon_summary: pd.DataFrame, verdict: str) -> None:
    print("=" * 80)
    print(f"SYMBOL: {symbol}")
    print("=" * 80)
    print(f"OVERALL VERDICT: {verdict}")
    print()

    print("[DATA QUALITY]")
    for k, v in quality.items():
        print(f"{k}: {v}")
    print()

    print("[MICRO BOUNCE SCAN]")
    print(micro.to_string(index=False))
    print()

    print("[HORIZON SUMMARY]")
    print(horizon_summary.to_string(index=False))
    print()

    print(f"[TOP {TOP_K} GRID CELLS]")
    top = grid.sort_values(
        ["score", "corr"],
        ascending=[False, False],
    ).head(TOP_K)

    cols = [
        "window",
        "horizon",
        "n",
        "score",
        "label",
        "phi",
        "corr",
        "t_stat",
        "extreme_n",
        "shrink_rate",
        "flip_rate",
        "same_sign_rate",
        "median_abs_ratio",
    ]
    print(top[cols].to_string(index=False))
    print()

    print("[HEATMAP: SCORE]")
    score_pivot = grid.pivot(index="window", columns="horizon", values="score").sort_index()
    print(score_pivot.fillna("").to_string())
    print()

    print("[HEATMAP: PHI]")
    phi_pivot = grid.pivot(index="window", columns="horizon", values="phi").sort_index()
    print(phi_pivot.round(4).fillna("").to_string())
    print()

    print("[HEATMAP: SHRINK_RATE]")
    shrink_pivot = grid.pivot(index="window", columns="horizon", values="shrink_rate").sort_index()
    print(shrink_pivot.round(4).fillna("").to_string())
    print()


def main():
    df, quality = load_symbol_data(TARGET_SYMBOL)

    if len(df) == 0:
        raise ValueError("분석 가능한 유효 가격 데이터가 없습니다.")

    micro = compute_micro_bounce_scan(df)
    grid = compute_grid_scan(df)
    horizon_summary = summarize_by_horizon(grid)
    verdict = overall_verdict(micro, horizon_summary)

    print_report(TARGET_SYMBOL, quality, micro, grid, horizon_summary, verdict)


if __name__ == "__main__":
    main()