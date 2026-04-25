"""
Refit smile using only VEV_5000~5300 (or any subset).

Outputs (IV_A, IV_B, IV_C) for annualized sigma / year TTE.

Usage:
    DATA_DIR=../data_capsule/round3 python refit_smile.py
    DATA_DIR=../data_capsule/round3 SUBSET=VEV_5000,VEV_5100,VEV_5200,VEV_5300 python refit_smile.py
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = os.environ.get("DATA_DIR", ".")
DAYS = [0, 1, 2]
DAY_LEN = 1_000_000
# Day 0 시작 시 TTE 8 days. Day d 시작 시 TTE = 8-d days.
TTE_START = {0: 8, 1: 7, 2: 6}
DAYS_PER_YEAR = 365.0
UNDERLYING = "VELVETFRUIT_EXTRACT"

STRIKES_ALL = {
    "VEV_4000": 4000, "VEV_4500": 4500,
    "VEV_5000": 5000, "VEV_5100": 5100, "VEV_5200": 5200,
    "VEV_5300": 5300, "VEV_5400": 5400, "VEV_5500": 5500,
    "VEV_6000": 6000, "VEV_6500": 6500,
}

# 기본: 5000, 5100, 5200, 5300 만 (5500 제외)
DEFAULT_SUBSET = ["VEV_5000", "VEV_5100", "VEV_5200", "VEV_5300"]
subset_env = os.environ.get("SUBSET")
if subset_env:
    SUBSET = [s.strip() for s in subset_env.split(",")]
else:
    SUBSET = DEFAULT_SUBSET
print(f"Fitting with subset: {SUBSET}")
STRIKES = {k: STRIKES_ALL[k] for k in SUBSET}


# ----- BS + IV (annualized) -----
def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_call(S, K, T, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0.0)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return S * _norm_cdf(d1) - K * _norm_cdf(d2)

def implied_vol(V, S, K, T, tol=1e-5, max_iter=100):
    if T <= 0 or V <= 0:
        return None
    intrinsic = max(S - K, 0.0)
    if V < intrinsic - 1e-3 or V > S + 1e-3:
        return None
    lo, hi = 1e-4, 5.0
    f_lo = bs_call(S, K, T, lo) - V
    f_hi = bs_call(S, K, T, hi) - V
    if f_lo * f_hi > 0:
        return None
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = bs_call(S, K, T, mid) - V
        if abs(f_mid) < tol:
            return mid
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


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


def main():
    df = load_prices()
    pvt = df.pivot_table(
        index=["day", "timestamp", "t_global"],
        columns="product", values="mid_price",
    ).reset_index()
    pvt = pvt[pvt[UNDERLYING].notna()].copy()
    print(f"Underlying ticks: {len(pvt)}")

    records = []
    for _, row in pvt.iterrows():
        day = int(row["day"])
        ts = int(row["timestamp"])
        S = row[UNDERLYING]
        tte_days = TTE_START[day] - ts / DAY_LEN
        if tte_days <= 0:
            continue
        T = tte_days / DAYS_PER_YEAR
        for prod, K in STRIKES.items():
            if prod not in pvt.columns:
                continue
            V = row[prod]
            if pd.isna(V):
                continue
            iv = implied_vol(V, S, K, T)
            if iv is None:
                continue
            m = math.log(K / S) / math.sqrt(T)
            records.append({
                "day": day, "t_global": row["t_global"],
                "product": prod, "K": K, "S": S, "V": V,
                "T": T, "m": m, "iv": iv,
            })

    iv_df = pd.DataFrame(records)
    print(f"Valid IV obs: {len(iv_df)}")

    # Quadratic fit
    ms = iv_df["m"].values
    vs = iv_df["iv"].values
    X = np.column_stack([ms * ms, ms, np.ones_like(ms)])
    coefs, *_ = np.linalg.lstsq(X, vs, rcond=None)
    a, b, c = coefs
    pred = a * ms * ms + b * ms + c
    resid = vs - pred
    print(f"\nFit: iv = a*m^2 + b*m + c  (annualized)")
    print(f"  a = {a:.15f}")
    print(f"  b = {b:.15f}")
    print(f"  c = {c:.15f}")
    print(f"  resid std = {np.std(resid):.6f}")
    print(f"  n = {len(iv_df)}")

    # Per-voucher residual stats
    print("\nPer-voucher residual stats:")
    for p in sorted(STRIKES.keys(), key=lambda x: STRIKES[x]):
        sub = iv_df[iv_df["product"] == p]
        if len(sub) == 0:
            continue
        pr = a * sub["m"].values ** 2 + b * sub["m"].values + c
        r = sub["iv"].values - pr
        print(f"  {p}  n={len(sub):6d}  mean_m={sub['m'].mean():.4f}  "
              f"mean_iv={sub['iv'].mean():.4f}  "
              f"resid_mean={r.mean():+.5f}  resid_std={r.std():.5f}")

    # Also print code snippet to paste
    print(f"\n# Paste into trader:")
    print(f"IV_A = {a}")
    print(f"IV_B = {b}")
    print(f"IV_C = {c}")

    # Plot
    mm = np.linspace(ms.min() - 0.05, ms.max() + 0.05, 200)
    fig, ax = plt.subplots(figsize=(10, 6))
    for p in sorted(STRIKES.keys(), key=lambda x: STRIKES[x]):
        sub = iv_df[iv_df["product"] == p]
        ax.scatter(sub["m"], sub["iv"], s=3, alpha=0.3, label=p)
    ax.plot(mm, a * mm * mm + b * mm + c, "k-", lw=2, label="fit")
    ax.set_xlabel("m = log(K/S)/sqrt(T)")
    ax.set_ylabel("annualized IV")
    ax.set_title(f"Smile fit with subset: {SUBSET}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("refit_smile.png", dpi=130)
    print(f"\nSaved: refit_smile.png")


if __name__ == "__main__":
    main()