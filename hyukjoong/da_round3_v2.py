"""
Round 3 analysis v2 - with voucher filtering options and better diagnostics.

Changes from v1:
  - Allow voucher subset via VOUCHER_SUBSET env var (comma-separated)
  - Default excludes VEV_6000, VEV_6500 (mid stuck at 0.5)
  - Add 'moneyness_log' = log(K/S) column (no TTE scaling) for inspection
  - Add correlation between baseIV and underlying price
  - Add per-voucher IV scatter against TTE
  - Report fit residual in absolute and relative terms

Usage:
    DATA_DIR=../data_capsule/round3 python da_round3_v2.py
    VOUCHER_SUBSET=VEV_5000,VEV_5100,VEV_5200,VEV_5300,VEV_5400,VEV_5500 python da_round3_v2.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import log, sqrt, erf

DATA_DIR = os.environ.get("DATA_DIR", ".")
OUT_DIR = os.environ.get("OUT_DIR", ".")
DAYS = [0, 1, 2]
DAY_LEN = 1_000_000
TTE_START = {0: 8, 1: 7, 2: 6}
UNDERLYING = "VELVETFRUIT_EXTRACT"

ALL_STRIKES = {
    "VEV_4000": 4000, "VEV_4500": 4500,
    "VEV_5000": 5000, "VEV_5100": 5100, "VEV_5200": 5200,
    "VEV_5300": 5300, "VEV_5400": 5400, "VEV_5500": 5500,
    "VEV_6000": 6000, "VEV_6500": 6500,
}

# Default: drop the two dead strikes (VEV_6000, VEV_6500 with mid=0.5)
DEFAULT_SUBSET = [
    "VEV_4000", "VEV_4500",
    "VEV_5000", "VEV_5100", "VEV_5200", "VEV_5300", "VEV_5400", "VEV_5500",
]
subset_env = os.environ.get("VOUCHER_SUBSET")
if subset_env:
    VOUCHER_SUBSET = [s.strip() for s in subset_env.split(",")]
else:
    VOUCHER_SUBSET = DEFAULT_SUBSET
VOUCHER_STRIKES = {k: ALL_STRIKES[k] for k in VOUCHER_SUBSET}
print(f"Using voucher subset: {VOUCHER_SUBSET}")


# ---------- BS ----------
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


def fit_quadratic(m, v):
    X = np.column_stack([m * m, m, np.ones_like(m)])
    coefs, *_ = np.linalg.lstsq(X, v, rcond=None)
    return tuple(coefs)


def load_prices():
    dfs = []
    for d in DAYS:
        path = os.path.join(DATA_DIR, f"prices_round_3_day_{d}.csv")
        if not os.path.exists(path):
            print(f"[WARN] missing {path}")
            continue
        df = pd.read_csv(path, sep=";")
        df["t_global"] = df["day"] * DAY_LEN + df["timestamp"]
        dfs.append(df)
    if not dfs:
        raise RuntimeError(f"No price files in {DATA_DIR}")
    return pd.concat(dfs, ignore_index=True)


def build_iv(df):
    pvt = df.pivot_table(
        index=["day", "timestamp", "t_global"],
        columns="product", values="mid_price",
    ).reset_index()
    pvt = pvt[pvt[UNDERLYING].notna()].copy()

    records = []
    for _, row in pvt.iterrows():
        day = int(row["day"]); ts = int(row["timestamp"])
        S = row[UNDERLYING]
        TTE = TTE_START[day] - ts / DAY_LEN
        if TTE <= 0:
            continue
        for prod, K in VOUCHER_STRIKES.items():
            if prod not in pvt.columns:
                continue
            V = row[prod]
            if pd.isna(V):
                continue
            log_moneyness = log(K / S)
            m = log_moneyness / sqrt(TTE)
            iv = implied_vol(V, S, K, TTE)
            records.append({
                "day": day, "timestamp": ts, "t_global": row["t_global"],
                "product": prod, "K": K, "S": S, "V": V,
                "TTE": TTE, "log_mny": log_moneyness, "m": m, "iv": iv,
            })
    return pd.DataFrame(records)


def phase23(iv_df):
    print("\n" + "=" * 70)
    print("  PHASE 2-3 - IV surface + smile fit (subset)")
    print("=" * 70)
    iv_valid = iv_df.dropna(subset=["iv"])
    print(f"Valid IV obs: {len(iv_valid)} / {len(iv_df)}")
    print("\nCounts per voucher per day:")
    print(iv_valid.groupby(["day", "product"]).size().unstack(fill_value=0))
    print("\nIV stats per voucher (pooled):")
    stats = iv_valid.groupby("product")["iv"].describe()[["mean", "std", "min", "max"]]
    with pd.option_context("display.float_format", "{:.5f}".format):
        print(stats.to_string())
    print("\nMean log-moneyness per voucher:")
    mny_stats = iv_valid.groupby("product")[["log_mny", "m"]].mean()
    with pd.option_context("display.float_format", "{:.5f}".format):
        print(mny_stats.to_string())

    iv_df.to_csv(os.path.join(OUT_DIR, "p23_iv_surface_v2.csv"), index=False)

    # Fit
    coef_rows = []
    for d in sorted(iv_valid["day"].unique()):
        sub = iv_valid[iv_valid["day"] == d]
        a, b, c = fit_quadratic(sub["m"].values, sub["iv"].values)
        pred = a * sub["m"].values ** 2 + b * sub["m"].values + c
        resid = sub["iv"].values - pred
        coef_rows.append({
            "scope": f"day_{d}", "a": a, "b": b, "c_baseIV": c,
            "resid_std": np.std(resid),
            "resid_std_rel": np.std(resid) / c if c != 0 else np.nan,
            "n": len(sub),
        })
    a, b, c = fit_quadratic(iv_valid["m"].values, iv_valid["iv"].values)
    pred = a * iv_valid["m"].values ** 2 + b * iv_valid["m"].values + c
    resid = iv_valid["iv"].values - pred
    coef_rows.append({
        "scope": "pooled", "a": a, "b": b, "c_baseIV": c,
        "resid_std": np.std(resid),
        "resid_std_rel": np.std(resid) / c if c != 0 else np.nan,
        "n": len(iv_valid),
    })
    coefs = pd.DataFrame(coef_rows)
    print("\nSmile fit:")
    with pd.option_context("display.float_format", "{:.6f}".format):
        print(coefs.to_string(index=False))
    coefs.to_csv(os.path.join(OUT_DIR, "p23_fit_coefs_v2.csv"), index=False)

    # Plot
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
        row = coefs[coefs["scope"] == f"day_{d}"].iloc[0]
        ax.plot(mm, row["a"] * mm ** 2 + row["b"] * mm + row["c_baseIV"], "k-", lw=2)
        ax.set_title(f"day {d} | c={row['c_baseIV']:.4f} | res={row['resid_std']:.4f}")
        ax.set_xlabel("m")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("IV")

    ax = axes[-1]
    for p in prod_order:
        s = iv_valid[iv_valid["product"] == p]
        ax.scatter(s["m"], s["iv"], s=3, alpha=0.2, color=prod_color[p], label=p)
    row = coefs[coefs["scope"] == "pooled"].iloc[0]
    ax.plot(mm, row["a"] * mm ** 2 + row["b"] * mm + row["c_baseIV"], "k-", lw=2)
    ax.set_title(f"pooled | c={row['c_baseIV']:.4f} | res={row['resid_std']:.4f}")
    ax.set_xlabel("m"); ax.grid(alpha=0.3)
    ax.legend(fontsize=6, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "p23_smile_fit_v2.png"), dpi=130)
    plt.close(fig)

    # Per-voucher IV time series (how each voucher's IV evolves over time)
    fig2, axes = plt.subplots(len(prod_order), 1,
                              figsize=(12, 1.8 * len(prod_order)), sharex=True)
    for ax, p in zip(axes, prod_order):
        s = iv_valid[iv_valid["product"] == p].sort_values("t_global")
        ax.plot(s["t_global"].values, s["iv"].values, lw=0.4, color=prod_color[p])
        ax.set_title(f"{p} (K={VOUCHER_STRIKES[p]})", fontsize=8)
        ax.grid(alpha=0.3)
        for d in [1, 2]:
            ax.axvline(d * DAY_LEN, color="red", ls="--", lw=0.5)
    axes[-1].set_xlabel("t_global")
    fig2.suptitle("Per-voucher IV time series")
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUT_DIR, "p23_per_voucher_iv.png"), dpi=130)
    plt.close(fig2)

    return iv_valid


def phase4(iv_valid):
    print("\n" + "=" * 70)
    print("  PHASE 4 - Per-tick smile coef time series")
    print("=" * 70)
    rows = []
    min_n = max(3, len(VOUCHER_SUBSET) - 1)
    for t, g in iv_valid.groupby("t_global"):
        if len(g) < min_n:
            continue
        m = g["m"].values
        v = g["iv"].values
        a, b, c = fit_quadratic(m, v)
        pred = a * m * m + b * m + c
        resid = v - pred
        rows.append({
            "t_global": t, "day": int(g["day"].iloc[0]),
            "timestamp": int(g["timestamp"].iloc[0]),
            "a": a, "b": b, "c_baseIV": c,
            "resid_std": np.std(resid), "n": len(g),
            "S": g["S"].iloc[0], "TTE": g["TTE"].iloc[0],
        })
    ts = pd.DataFrame(rows).sort_values("t_global").reset_index(drop=True)
    ts.to_csv(os.path.join(OUT_DIR, "p4_smile_ts_v2.csv"), index=False)

    print("Summary:")
    with pd.option_context("display.float_format", "{:.6f}".format):
        print(ts[["a", "b", "c_baseIV", "resid_std"]].describe().to_string())

    c = ts["c_baseIV"].values
    if len(c) > 2:
        rho1 = np.corrcoef(c[:-1], c[1:])[0, 1]
        print(f"\nBase IV lag-1 autocorr = {rho1:.6f}")
        print(f"Base IV mean={c.mean():.6f}, std={c.std():.6f}")

    # correlation of baseIV with underlying S
    corr_S = np.corrcoef(ts["c_baseIV"].values, ts["S"].values)[0, 1]
    print(f"corr(baseIV, S) = {corr_S:.4f}")

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    for ax, col, title in zip(
        axes, ["c_baseIV", "a", "b", "S"],
        ["Base IV (c)", "Curvature (a)", "Skew (b)", "Underlying S"],
    ):
        ax.plot(ts["t_global"].values, ts[col].values, lw=0.5)
        ax.plot(ts["t_global"].values,
                ts[col].rolling(200, min_periods=20).mean().values,
                color="red", lw=1.2)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        for d in [1, 2]:
            ax.axvline(d * DAY_LEN, color="black", ls="--", lw=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "p4_baseIV_plot_v2.png"), dpi=130)
    plt.close(fig)

    print("Saved: p23_smile_fit_v2.png, p23_per_voucher_iv.png, p4_baseIV_plot_v2.png")


def main():
    print(f"DATA_DIR = {os.path.abspath(DATA_DIR)}")
    df = load_prices()
    print(f"Loaded {len(df)} rows")
    iv_df = build_iv(df)
    iv_valid = phase23(iv_df)
    phase4(iv_valid)


if __name__ == "__main__":
    main()