"""
Phase 2-3: IV surface + quadratic smile fit for VELVETFRUIT_EXTRACT_VOUCHER.

For each timestamp and each voucher:
    S_t = VELVETFRUIT_EXTRACT mid at same t
    V_t = voucher mid
    TTE_t = days_remaining - (timestamp_within_day / 1_000_000)
    m_t  = log(K / S_t) / sqrt(TTE_t)
    v_t  = BS implied vol (call, r=0) solving V_t = BS(S_t, K, TTE_t, v)

We then fit v = a*m^2 + b*m + c per-day and pooled.

Outputs:
  - p23_iv_surface.csv (every tick, every voucher)
  - p23_smile_fit.png  (scatter + fit per day + pooled)
  - p23_fit_coefs.csv  (a, b, c per day and pooled)
Console: residual stats (how tight the parabola is for each day).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import log, sqrt, exp, erf, pi

DATA_DIR = os.environ.get("DATA_DIR", ".")
DAYS = [0, 1, 2]

# Map day number to days-till-expiry at the START of that day.
# Problem statement: day 0 -> TTE=8, day 1 -> TTE=7, day 2 -> TTE=6.
TTE_START = {0: 8, 1: 7, 2: 6}
DAY_LEN = 1_000_000  # timestamp units per day

UNDERLYING = "VELVETFRUIT_EXTRACT"

VOUCHER_STRIKES = {
    "VEV_4000": 4000, "VEV_4500": 4500,
    "VEV_5000": 5000, "VEV_5100": 5100, "VEV_5200": 5200,
    "VEV_5300": 5300, "VEV_5400": 5400, "VEV_5500": 5500,
    "VEV_6000": 6000, "VEV_6500": 6500,
}


# ---------- Black-Scholes helpers (r=0, no dividend) ----------
def norm_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bs_call(S, K, T, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    sqrtT = sqrt(T)
    d1 = (log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return S * norm_cdf(d1) - K * norm_cdf(d2)


def implied_vol(V, S, K, T, tol=1e-6, max_iter=100):
    """Brent-style bisection for implied vol. Returns np.nan if out of range."""
    if T <= 0:
        return np.nan
    intrinsic = max(S - K, 0.0)
    if V < intrinsic - 1e-6 or V > S + 1e-6:
        return np.nan
    lo, hi = 1e-4, 5.0
    f_lo = bs_call(S, K, T, lo) - V
    f_hi = bs_call(S, K, T, hi) - V
    if f_lo * f_hi > 0:
        return np.nan
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = bs_call(S, K, T, mid) - V
        if abs(f_mid) < tol:
            return mid
        if f_lo * f_mid < 0:
            hi = mid; f_hi = f_mid
        else:
            lo = mid; f_lo = f_mid
    return 0.5 * (lo + hi)


# ---------- Data prep ----------
def load_prices():
    dfs = []
    for d in DAYS:
        p = os.path.join(DATA_DIR, f"prices_round_3_day_{d}.csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p, sep=";")
        df["t_global"] = df["day"] * DAY_LEN + df["timestamp"]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def build_iv_surface(df):
    # Pivot: mid_price by (t_global, product). Keep day & timestamp_within_day.
    pvt = df.pivot_table(
        index=["day", "timestamp", "t_global"],
        columns="product", values="mid_price",
    ).reset_index()

    # Filter to rows where underlying is present
    pvt = pvt[pvt[UNDERLYING].notna()].copy()

    records = []
    for _, row in pvt.iterrows():
        day = int(row["day"])
        ts = int(row["timestamp"])
        S = row[UNDERLYING]
        # TTE: at start of day it is TTE_START[day], decays to TTE_START[day]-1 by end
        TTE_days = TTE_START[day] - ts / DAY_LEN
        if TTE_days <= 0:
            continue
        TTE = TTE_days  # already in units of "days"; sigma will be in per-day units
        for prod, K in VOUCHER_STRIKES.items():
            if prod not in pvt.columns:
                continue
            V = row[prod]
            if pd.isna(V):
                continue
            m = log(K / S) / sqrt(TTE)
            iv = implied_vol(V, S, K, TTE)
            records.append({
                "day": day, "timestamp": ts, "t_global": row["t_global"],
                "product": prod, "K": K, "S": S, "V": V,
                "TTE": TTE, "m": m, "iv": iv,
            })
    out = pd.DataFrame(records)
    return out


# ---------- Smile fit ----------
def fit_quadratic(m, v):
    """Returns (a, b, c) for v = a*m^2 + b*m + c."""
    X = np.column_stack([m * m, m, np.ones_like(m)])
    coefs, *_ = np.linalg.lstsq(X, v, rcond=None)
    return tuple(coefs)


def main():
    print("Loading...")
    df = load_prices()
    print(f"  rows={len(df)}, products={df['product'].nunique()}")

    print("Computing IV surface (takes a moment)...")
    iv = build_iv_surface(df)
    iv_valid = iv.dropna(subset=["iv"])
    print(f"  total voucher obs: {len(iv)}, valid IVs: {len(iv_valid)}")
    print("\n  valid IV counts per voucher per day:")
    print(iv_valid.groupby(["day", "product"]).size().unstack(fill_value=0))
    iv.to_csv("p23_iv_surface.csv", index=False)

    # Per-day fit + pooled
    coef_rows = []
    for d in sorted(iv_valid["day"].unique()):
        sub = iv_valid[iv_valid["day"] == d]
        a, b, c = fit_quadratic(sub["m"].values, sub["iv"].values)
        resid = sub["iv"].values - (a * sub["m"].values ** 2 + b * sub["m"].values + c)
        coef_rows.append({
            "scope": f"day_{d}", "a": a, "b": b, "c_baseIV": c,
            "resid_std": np.std(resid), "n": len(sub),
        })
    a, b, c = fit_quadratic(iv_valid["m"].values, iv_valid["iv"].values)
    resid = iv_valid["iv"].values - (a * iv_valid["m"].values ** 2 + b * iv_valid["m"].values + c)
    coef_rows.append({
        "scope": "pooled", "a": a, "b": b, "c_baseIV": c,
        "resid_std": np.std(resid), "n": len(iv_valid),
    })
    coefs_df = pd.DataFrame(coef_rows)
    print("\n=== Smile fit (v = a*m^2 + b*m + c) ===")
    with pd.option_context("display.float_format", "{:.6f}".format):
        print(coefs_df.to_string(index=False))
    coefs_df.to_csv("p23_fit_coefs.csv", index=False)

    # Plot smile per day + pooled
    fig, axes = plt.subplots(1, len(DAYS) + 1, figsize=(5 * (len(DAYS) + 1), 4), sharey=True)
    mm = np.linspace(iv_valid["m"].min(), iv_valid["m"].max(), 100)
    colors = plt.cm.tab10.colors
    prod_order = sorted(VOUCHER_STRIKES.keys(), key=lambda x: VOUCHER_STRIKES[x])
    prod_color = {p: colors[i % 10] for i, p in enumerate(prod_order)}

    for ax, d in zip(axes[:-1], DAYS):
        sub = iv_valid[iv_valid["day"] == d]
        for p in prod_order:
            s = sub[sub["product"] == p]
            if len(s) == 0:
                continue
            ax.scatter(s["m"], s["iv"], s=3, alpha=0.3, color=prod_color[p], label=p)
        row = coefs_df[coefs_df["scope"] == f"day_{d}"].iloc[0]
        ax.plot(mm, row["a"] * mm ** 2 + row["b"] * mm + row["c_baseIV"], "k-", lw=2)
        ax.set_title(f"day {d} | baseIV={row['c_baseIV']:.4f} | resid_std={row['resid_std']:.4f}")
        ax.set_xlabel("m = log(K/S)/sqrt(T)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("implied vol (per-day-sigma)")

    # Pooled
    ax = axes[-1]
    for p in prod_order:
        s = iv_valid[iv_valid["product"] == p]
        if len(s) == 0:
            continue
        ax.scatter(s["m"], s["iv"], s=3, alpha=0.2, color=prod_color[p], label=p)
    row = coefs_df[coefs_df["scope"] == "pooled"].iloc[0]
    ax.plot(mm, row["a"] * mm ** 2 + row["b"] * mm + row["c_baseIV"], "k-", lw=2)
    ax.set_title(f"pooled | baseIV={row['c_baseIV']:.4f} | resid_std={row['resid_std']:.4f}")
    ax.set_xlabel("m")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, loc="upper right", ncol=2)

    fig.tight_layout()
    fig.savefig("p23_smile_fit.png", dpi=130)
    print("\nSaved: p23_iv_surface.csv, p23_smile_fit.png, p23_fit_coefs.csv")


if __name__ == "__main__":
    main()