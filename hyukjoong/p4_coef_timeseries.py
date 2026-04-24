"""
Phase 4: Time series of smile coefficients.

For each timestamp we fit v = a*m^2 + b*m + c over all 10 vouchers (that tick)
and track the series (a_t, b_t, c_t). c_t is the "base IV" -- the key signal
Frankfurt hints at ("interesting patterns in timeseries of base IV").

Outputs:
  - p4_smile_ts.csv  (a, b, c, resid_std per tick)
  - p4_baseIV_plot.png (base IV time series + rolling mean/std)
  - p4_coef_acf.png  (autocorr of base IV)
Console: mean / std / stationarity check / lag-1 AR(1).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = os.environ.get("DATA_DIR", ".")

# Reuse IV surface
IV_CSV = "p23_iv_surface.csv"


def fit_quadratic(m, v):
    X = np.column_stack([m * m, m, np.ones_like(m)])
    coefs, *_ = np.linalg.lstsq(X, v, rcond=None)
    return tuple(coefs)


def main():
    if not os.path.exists(IV_CSV):
        print(f"ERROR: {IV_CSV} not found. Run p23_iv_smile_fit.py first.")
        return
    iv = pd.read_csv(IV_CSV)
    iv = iv.dropna(subset=["iv"])

    # Per-tick fit
    rows = []
    # Group by t_global
    for t, g in iv.groupby("t_global"):
        if len(g) < 4:
            continue
        m = g["m"].values
        v = g["iv"].values
        a, b, c = fit_quadratic(m, v)
        resid = v - (a * m * m + b * m + c)
        rows.append({
            "t_global": t, "day": int(g["day"].iloc[0]),
            "timestamp": int(g["timestamp"].iloc[0]),
            "a": a, "b": b, "c_baseIV": c,
            "resid_std": np.std(resid), "n": len(g),
            "S": g["S"].iloc[0], "TTE": g["TTE"].iloc[0],
        })
    ts = pd.DataFrame(rows).sort_values("t_global").reset_index(drop=True)
    ts.to_csv("p4_smile_ts.csv", index=False)

    # Console summary
    print("=== Smile coefficients (per tick) summary ===")
    with pd.option_context("display.float_format", "{:.6f}".format):
        print(ts[["a", "b", "c_baseIV", "resid_std"]].describe().to_string())

    # Stationarity-ish: lag-1 autocorr
    c = ts["c_baseIV"].values
    c_lag = c[:-1]; c_now = c[1:]
    rho1 = np.corrcoef(c_lag, c_now)[0, 1]
    print(f"\nBase IV lag-1 autocorr = {rho1:.6f}")
    # AR(1) coefficient via OLS on demeaned
    dc = c - c.mean()
    phi = np.dot(dc[:-1], dc[1:]) / np.dot(dc[:-1], dc[:-1])
    print(f"Base IV AR(1) phi (demeaned) = {phi:.6f}")
    print(f"Base IV mean = {c.mean():.6f}, std = {c.std():.6f}")

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for ax, col, title in zip(
        axes,
        ["c_baseIV", "a", "b"],
        ["Base IV (c)  [aka v at m=0]", "Curvature (a)", "Skew (b)"],
    ):
        ax.plot(ts["t_global"].values, ts[col].values, lw=0.5)
        # Rolling mean
        ax.plot(ts["t_global"].values,
                ts[col].rolling(200, min_periods=20).mean().values,
                color="red", lw=1.2, label="rolling mean (200)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        for d in [1, 2]:
            ax.axvline(d * 1_000_000, color="black", ls="--", lw=0.5)
    axes[-1].set_xlabel("t_global")
    fig.tight_layout()
    fig.savefig("p4_baseIV_plot.png", dpi=130)

    # ACF of base IV (lag 1..50)
    def acf(x, max_lag=50):
        x = x - x.mean()
        var = np.dot(x, x)
        out = []
        for k in range(1, max_lag + 1):
            out.append(np.dot(x[:-k], x[k:]) / var)
        return np.array(out)

    acs = acf(c, 50)
    fig2, ax = plt.subplots(figsize=(8, 3))
    ax.bar(range(1, 51), acs, width=0.8)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title("Base IV autocorrelation (lags 1-50)")
    ax.set_xlabel("lag"); ax.set_ylabel("rho")
    ax.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig("p4_coef_acf.png", dpi=130)

    print("\nSaved: p4_smile_ts.csv, p4_baseIV_plot.png, p4_coef_acf.png")


if __name__ == "__main__":
    main()