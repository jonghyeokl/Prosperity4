# jonghyeok/data_analysis/round3_check_theo_diff_multivariate_mr.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILENAME = "round3_theo_diff_5000_5300.csv"
PRODUCTS = ["VEV_5000", "VEV_5100", "VEV_5200", "VEV_5300"]

EMA_WINDOWS = list(range(15, 66, 5))
HORIZONS = [1, 2, 3, 4, 5]

# True: day 경계에서 EMA/lag/future를 끊음. 더 엄밀함.
# False: combined_timestamp 기준으로 day들을 이어붙임. 기존 EMA 분석과 더 비슷함.
GROUP_BY_DAY = True


def infer_base_step(ts: pd.Series) -> int:
    diffs = ts.diff().dropna()
    diffs = diffs[diffs > 0]

    if len(diffs) == 0:
        return 100

    mode = diffs.mode()
    if len(mode) == 0:
        return int(diffs.iloc[0])

    return int(mode.iloc[0])


def ema_prev(values: pd.Series, window: int) -> pd.Series:
    """
    t 시점 값을 넣기 전의 EMA.
    residual_t = x_t - EMA_prev_t 계산용.
    """
    alpha = 2.0 / (window + 1.0)

    out = []
    ema = None

    for v in values:
        out.append(ema)

        if ema is None:
            ema = v
        else:
            ema = alpha * v + (1.0 - alpha) * ema

    return pd.Series(out, index=values.index, dtype="float64")


def load_data() -> pd.DataFrame:
    project_root = Path(__file__).resolve().parents[2]
    input_path = (
        project_root
        / "jonghyeok"
        / "data_analysis"
        / "output"
        / INPUT_FILENAME
    )

    if not input_path.exists():
        raise FileNotFoundError(f"파일을 찾지 못했습니다: {input_path}")

    df = pd.read_csv(input_path)

    required = {
        "product",
        "day_num",
        "timestamp",
        "combined_timestamp",
        "theo_diff",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}")

    df = df[df["product"].isin(PRODUCTS)].copy()

    for col in ["day_num", "timestamp", "combined_timestamp", "theo_diff"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["product", "day_num", "timestamp", "combined_timestamp", "theo_diff"])
    df = df.sort_values(["product", "combined_timestamp"]).reset_index(drop=True)

    return df


def build_feature_frame(product_df: pd.DataFrame, ema_window: int, horizon: int) -> pd.DataFrame:
    frames = []

    group_cols = ["day_num"] if GROUP_BY_DAY else ["product"]

    for _, g in product_df.groupby(group_cols, sort=False):
        g = g.sort_values("combined_timestamp").copy()

        ts_col = "timestamp" if GROUP_BY_DAY else "combined_timestamp"
        step = infer_base_step(g[ts_col])

        x = g["theo_diff"]

        g["ema_prev"] = ema_prev(x, ema_window)
        g["residual"] = g["theo_diff"] - g["ema_prev"]

        g["lag1_change"] = g["theo_diff"] - g["theo_diff"].shift(1)

        prev_gap = g[ts_col] - g[ts_col].shift(1)

        g["future_theo_diff"] = g["theo_diff"].shift(-horizon)
        g["future_change"] = g["future_theo_diff"] - g["theo_diff"]

        future_gap = g[ts_col].shift(-horizon) - g[ts_col]

        valid = (
            (prev_gap == step)
            & (future_gap == horizon * step)
            & np.isfinite(g["residual"])
            & np.isfinite(g["lag1_change"])
            & np.isfinite(g["future_change"])
        )

        frames.append(
            g.loc[
                valid,
                [
                    "product",
                    "day_num",
                    "timestamp",
                    "combined_timestamp",
                    "theo_diff",
                    "ema_prev",
                    "residual",
                    "lag1_change",
                    "future_change",
                ],
            ]
        )

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def ols_stats(X: np.ndarray, y: np.ndarray) -> dict:
    """
    y = alpha + beta * residual + gamma * lag1_change
    X columns: [1, residual, lag1_change]
    """
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X = X[mask]
    y = y[mask]

    n = len(y)
    k = X.shape[1]

    if n <= k:
        return {
            "n": n,
            "alpha": np.nan,
            "beta_residual": np.nan,
            "gamma_lag1": np.nan,
            "t_alpha": np.nan,
            "t_beta_residual": np.nan,
            "t_gamma_lag1": np.nan,
            "r2": np.nan,
            "corr_pred_actual": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
        }

    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    resid = y - pred

    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))
    corr = float(np.corrcoef(y, pred)[0, 1]) if n >= 2 else np.nan

    dof = n - k
    sigma2 = ss_res / dof if dof > 0 else np.nan

    try:
        xtx_inv = np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(sigma2 * xtx_inv))
        t_vals = coef / se
    except Exception:
        t_vals = np.array([np.nan] * k)

    return {
        "n": n,
        "alpha": float(coef[0]),
        "beta_residual": float(coef[1]),
        "gamma_lag1": float(coef[2]),
        "t_alpha": float(t_vals[0]),
        "t_beta_residual": float(t_vals[1]),
        "t_gamma_lag1": float(t_vals[2]),
        "r2": float(r2),
        "corr_pred_actual": corr,
        "rmse": rmse,
        "mae": mae,
    }


def train_test_stats(f: pd.DataFrame) -> dict:
    max_day = f["day_num"].max()

    train = f[f["day_num"] < max_day].copy()
    test = f[f["day_num"] == max_day].copy()

    if len(train) <= 3 or len(test) <= 3:
        return {
            "train_n": len(train),
            "test_n": len(test),
            "test_r2": np.nan,
            "test_corr_pred_actual": np.nan,
            "test_rmse": np.nan,
            "test_mae": np.nan,
        }

    X_train = np.column_stack(
        [
            np.ones(len(train)),
            train["residual"].to_numpy(dtype=float),
            train["lag1_change"].to_numpy(dtype=float),
        ]
    )
    y_train = train["future_change"].to_numpy(dtype=float)

    coef, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)

    X_test = np.column_stack(
        [
            np.ones(len(test)),
            test["residual"].to_numpy(dtype=float),
            test["lag1_change"].to_numpy(dtype=float),
        ]
    )
    y_test = test["future_change"].to_numpy(dtype=float)

    pred = X_test @ coef
    resid = y_test - pred

    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
    test_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "train_n": len(train),
        "test_n": len(test),
        "test_r2": float(test_r2),
        "test_corr_pred_actual": float(np.corrcoef(y_test, pred)[0, 1]) if len(test) >= 2 else np.nan,
        "test_rmse": float(np.sqrt(np.mean(resid ** 2))),
        "test_mae": float(np.mean(np.abs(resid))),
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    out_dir = (
        project_root
        / "jonghyeok"
        / "data_analysis"
        / "output"
        / "theo_diff_multivariate_mr"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data()

    rows = []
    feature_frames = []

    for product, product_df in df.groupby("product", sort=True):
        product_df = product_df.sort_values("combined_timestamp").copy()

        for ema_window in EMA_WINDOWS:
            for horizon in HORIZONS:
                f = build_feature_frame(product_df, ema_window, horizon)

                if f.empty:
                    continue

                X = np.column_stack(
                    [
                        np.ones(len(f)),
                        f["residual"].to_numpy(dtype=float),
                        f["lag1_change"].to_numpy(dtype=float),
                    ]
                )
                y = f["future_change"].to_numpy(dtype=float)

                stats = ols_stats(X, y)
                test_stats = train_test_stats(f)

                rows.append(
                    {
                        "product": product,
                        "ema_window": ema_window,
                        "horizon": horizon,
                        **stats,
                        **test_stats,
                    }
                )

                f = f.copy()
                f["ema_window"] = ema_window
                f["horizon"] = horizon
                feature_frames.append(f)

    summary = pd.DataFrame(rows)

    summary = summary.sort_values(
        ["product", "horizon", "ema_window"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    features = pd.concat(feature_frames, ignore_index=True) if feature_frames else pd.DataFrame()

    summary_path = out_dir / "round3_theo_diff_multivariate_mr_summary.csv"
    features_path = out_dir / "round3_theo_diff_multivariate_mr_features.csv"

    summary.to_csv(summary_path, index=False)
    features.to_csv(features_path, index=False)

    print("\n=== Multivariate Mean Reversion Summary ===")
    print(
        summary[
            [
                "product",
                "ema_window",
                "horizon",
                "n",
                "alpha",
                "beta_residual",
                "gamma_lag1",
                "t_beta_residual",
                "t_gamma_lag1",
                "r2",
                "corr_pred_actual",
                "test_r2",
                "test_corr_pred_actual",
            ]
        ].to_string(index=False)
    )

    print("\n=== Best by product/horizon based on test_r2 ===")
    best = (
        summary.sort_values(
            ["product", "horizon", "test_r2"],
            ascending=[True, True, False],
        )
        .groupby(["product", "horizon"])
        .head(1)
        .reset_index(drop=True)
    )
    print(
        best[
            [
                "product",
                "horizon",
                "ema_window",
                "beta_residual",
                "gamma_lag1",
                "r2",
                "test_r2",
                "test_corr_pred_actual",
            ]
        ].to_string(index=False)
    )

    print("\n=== Best overall per product based on test_r2 ===")
    best_product = (
        summary.sort_values(
            ["product", "test_r2"],
            ascending=[True, False],
        )
        .groupby("product")
        .head(1)
        .reset_index(drop=True)
    )
    print(
        best_product[
            [
                "product",
                "ema_window",
                "horizon",
                "beta_residual",
                "gamma_lag1",
                "r2",
                "test_r2",
                "test_corr_pred_actual",
            ]
        ].to_string(index=False)
    )

    print("\n[SAVED]", summary_path)
    print("[SAVED]", features_path)


if __name__ == "__main__":
    main()