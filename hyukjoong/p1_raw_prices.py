"""
Phase 1: Raw price inspection for Round 3 Prosperity data.
Inputs: prices_round_3_day_{0,1,2}.csv in DATA_DIR.
Outputs:
  - console summary (level, trend, realized vol, spread)
  - p1_mid_prices.png (3-day mid-price time series for all products)
  - p1_spread_stats.csv (per-product bid-ask spread summary)
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = os.environ.get("DATA_DIR", ".")
DAYS = [0, 1, 2]

# Prosperity standard columns (semicolon-separated): day;timestamp;product;
# bid_price_1;bid_volume_1;bid_price_2;bid_volume_2;bid_price_3;bid_volume_3;
# ask_price_1;ask_volume_1;ask_price_2;ask_volume_2;ask_price_3;ask_volume_3;
# mid_price;profit_and_loss

def load_prices():
    dfs = []
    for d in DAYS:
        path = os.path.join(DATA_DIR, f"prices_round_3_day_{d}.csv")
        if not os.path.exists(path):
            print(f"[WARN] missing {path}")
            continue
        df = pd.read_csv(path, sep=";")
        # continuous timestamp across days: day*1_000_000 + timestamp
        # (Prosperity uses 100-step ticks up to 1_000_000 per day)
        df["t_global"] = df["day"] * 1_000_000 + df["timestamp"]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def summarize_product(df, product):
    sub = df[df["product"] == product].copy()
    if sub.empty:
        return None
    sub = sub.sort_values("t_global")
    sub["spread"] = sub["ask_price_1"] - sub["bid_price_1"]
    # realized vol on mid returns (log)
    mid = sub["mid_price"].values
    logret = np.diff(np.log(mid[mid > 0]))
    realized_sigma_per_tick = np.std(logret)
    # approx: there are ~10000 ticks per day (100-step intervals in 1M window)
    ticks_per_day = 10000
    sigma_daily = realized_sigma_per_tick * np.sqrt(ticks_per_day)
    return {
        "product": product,
        "n_rows": len(sub),
        "mid_mean": sub["mid_price"].mean(),
        "mid_min": sub["mid_price"].min(),
        "mid_max": sub["mid_price"].max(),
        "mid_std": sub["mid_price"].std(),
        "spread_mean": sub["spread"].mean(),
        "spread_median": sub["spread"].median(),
        "sigma_tick": realized_sigma_per_tick,
        "sigma_daily": sigma_daily,
    }


def main():
    df = load_prices()
    print(f"Loaded {len(df)} rows across days {sorted(df['day'].unique())}")
    products = sorted(df["product"].unique())
    print(f"Products ({len(products)}): {products}")

    # Summary
    rows = [summarize_product(df, p) for p in products]
    rows = [r for r in rows if r is not None]
    summary = pd.DataFrame(rows)
    print("\n=== Per-product summary ===")
    with pd.option_context("display.width", 200, "display.max_columns", None, "display.float_format", "{:.4f}".format):
        print(summary.to_string(index=False))
    summary.to_csv("p1_spread_stats.csv", index=False)

    # Plot mid price time series
    n = len(products)
    ncol = 3
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3 * nrow), sharex=True)
    axes = axes.ravel() if nrow > 1 else axes
    for i, p in enumerate(products):
        sub = df[df["product"] == p].sort_values("t_global")
        ax = axes[i]
        ax.plot(sub["t_global"].values, sub["mid_price"].values, lw=0.6)
        ax.set_title(p, fontsize=9)
        ax.grid(alpha=0.3)
        # day boundaries
        for d in DAYS[1:]:
            ax.axvline(d * 1_000_000, color="red", lw=0.5, ls="--")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Mid prices across 3 days (red = day boundary)")
    fig.tight_layout()
    fig.savefig("p1_mid_prices.png", dpi=130)
    print("\nSaved: p1_mid_prices.png, p1_spread_stats.csv")


if __name__ == "__main__":
    main()