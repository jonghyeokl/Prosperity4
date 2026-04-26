# jonghyeok/data_analysis/round3_check_theo_diff_ar2_no_intercept.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


INPUT_FILENAME = "round3_theo_diff_5000_5300.csv"
PRODUCTS = ["VEV_5000", "VEV_5100", "VEV_5200", "VEV_5300"]


def infer_base_step(ts: pd.Series) -> int:
    diffs = ts.diff().dropna()
    diffs = diffs[diffs > 0]

    if len(diffs) == 0:
        return 100

    mode = diffs.mode()
    if len(mode) == 0:
        return int(diffs.iloc[0])

    return int(mode.iloc[0])


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

    required = {"product", "day_num", "timestamp", "combined_timestamp", "theo_diff"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}")

    df = df[df["product"].isin(PRODUCTS)].copy()

    for col in ["day_num", "timestamp", "combined_timestamp", "theo_diff"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["product", "day_num", "timestamp", "theo_diff"])
    df = df.sort_values(["product", "day_num", "timestamp"]).reset_index(drop=True)

    return df


def make_ar2_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (product, day_num), g in df.groupby(["product", "day_num"], sort=False):
        g = g.sort_values("timestamp").copy()
        step = infer_base_step(g["timestamp"])

        g["x_t"] = g["theo_diff"]
        g["x_t1"] = g["theo_diff"].shift(1)
        g["x_t2"] = g["theo_diff"].shift(2)

        gap1 = g["timestamp"] - g["timestamp"].shift(1)
        gap2 = g["timestamp"].shift(1) - g["timestamp"].shift(2)

        g = g[(gap1 == step) & (gap2 == step)].copy()
        g = g.dropna(subset=["x_t", "x_t1", "x_t2"])

        rows.append(
            g[
                [
                    "product",
                    "day_num",
                    "timestamp",
                    "combined_timestamp",
                    "x_t",
                    "x_t1",
                    "x_t2",
                ]
            ]
        )

    if not rows:
        raise RuntimeError("AR(2) row를 만들 수 없습니다.")

    return pd.concat(rows, ignore_index=True)


def fit_ar2_no_intercept(g: pd.DataFrame) -> dict:
    y = g["x_t"].to_numpy(dtype=float)
    x1 = g["x_t1"].to_numpy(dtype=float)
    x2 = g["x_t2"].to_numpy(dtype=float)

    X = np.column_stack([x1, x2])

    # x_t = a*x_(t-1) + b*x_(t-2)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = coef

    pred = X @ coef
    resid = y - pred

    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    rmse = float(np.sqrt(np.mean(resid**2)))
    mae = float(np.mean(np.abs(resid)))
    corr = float(np.corrcoef(y, pred)[0, 1]) if len(y) >= 2 else np.nan

    roots = np.roots([1.0, -a, -b])
    max_abs_root = float(np.max(np.abs(roots)))
    stable = bool(max_abs_root < 1.0)

    return {
        "n": len(g),
        "a": float(a),
        "b": float(b),
        "a_plus_b": float(a + b),
        "rmse": rmse,
        "mae": mae,
        "corr_pred_actual": corr,
        "r2": float(r2),
        "max_abs_root": max_abs_root,
        "stable_ar2": stable,
    }


def fit_train_test_ar2_no_intercept(g: pd.DataFrame) -> dict:
    max_day = g["day_num"].max()

    train = g[g["day_num"] < max_day].copy()
    test = g[g["day_num"] == max_day].copy()

    if len(train) < 10 or len(test) < 10:
        return {
            "train_n": len(train),
            "test_n": len(test),
            "test_rmse": np.nan,
            "test_mae": np.nan,
            "test_r2": np.nan,
            "test_corr_pred_actual": np.nan,
        }

    y_train = train["x_t"].to_numpy(dtype=float)
    X_train = np.column_stack(
        [
            train["x_t1"].to_numpy(dtype=float),
            train["x_t2"].to_numpy(dtype=float),
        ]
    )

    coef, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)

    y_test = test["x_t"].to_numpy(dtype=float)
    X_test = np.column_stack(
        [
            test["x_t1"].to_numpy(dtype=float),
            test["x_t2"].to_numpy(dtype=float),
        ]
    )

    pred = X_test @ coef
    resid = y_test - pred

    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    corr = float(np.corrcoef(y_test, pred)[0, 1]) if len(y_test) >= 2 else np.nan

    return {
        "train_n": len(train),
        "test_n": len(test),
        "test_rmse": float(np.sqrt(np.mean(resid**2))),
        "test_mae": float(np.mean(np.abs(resid))),
        "test_r2": float(r2),
        "test_corr_pred_actual": corr,
    }


def plot_product(g: pd.DataFrame, product: str, info: dict, out_dir: Path) -> None:
    y = g["x_t"].to_numpy(dtype=float)

    X = np.column_stack(
        [
            g["x_t1"].to_numpy(dtype=float),
            g["x_t2"].to_numpy(dtype=float),
        ]
    )

    coef = np.array([info["a"], info["b"]], dtype=float)
    pred = X @ coef

    sample_n = min(5000, len(g))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(g), size=sample_n, replace=False)

    y_plot = y[idx]
    pred_plot = pred[idx]

    lo = min(y_plot.min(), pred_plot.min())
    hi = max(y_plot.max(), pred_plot.max())

    plt.figure(figsize=(8, 8))
    plt.scatter(pred_plot, y_plot, s=6, alpha=0.25)
    plt.plot([lo, hi], [lo, hi], linewidth=1)
    plt.xlabel("predicted theo_diff_t")
    plt.ylabel("actual theo_diff_t")
    plt.title(
        f"{product}: AR(2) no-intercept theo_diff prediction\n"
        f"x_t = {info['a']:.4f} x_(t-1) + {info['b']:.4f} x_(t-2), "
        f"r2={info['r2']:.4f}"
    )
    plt.tight_layout()
    plt.savefig(out_dir / f"round3_theo_diff_ar2_no_intercept_pred_actual_{product}.png", dpi=160)
    plt.close()


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    out_dir = (
        project_root
        / "jonghyeok"
        / "data_analysis"
        / "output"
        / "theo_diff_ar2_no_intercept"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data()
    ar2 = make_ar2_rows(df)

    rows = []

    for product, g in ar2.groupby("product"):
        info = fit_ar2_no_intercept(g)
        test_info = fit_train_test_ar2_no_intercept(g)

        row = {
            "product": product,
            **info,
            **test_info,
        }
        rows.append(row)

        plot_product(g, product, info, out_dir)

    summary = pd.DataFrame(rows).sort_values("product").reset_index(drop=True)

    ar2_path = out_dir / "round3_theo_diff_ar2_no_intercept_rows.csv"
    summary_path = out_dir / "round3_theo_diff_ar2_no_intercept_summary.csv"

    ar2.to_csv(ar2_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\n=== Theo Diff AR(2) No Intercept Summary ===")
    print(
        summary[
            [
                "product",
                "n",
                "a",
                "b",
                "a_plus_b",
                "r2",
                "corr_pred_actual",
                "rmse",
                "mae",
                "max_abs_root",
                "stable_ar2",
                "test_r2",
                "test_corr_pred_actual",
                "test_rmse",
                "test_mae",
            ]
        ].to_string(index=False)
    )

    print("\n[SAVED]", ar2_path)
    print("[SAVED]", summary_path)
    print("[SAVED]", out_dir)


if __name__ == "__main__":
    main()