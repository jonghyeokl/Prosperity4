from pathlib import Path
import numpy as np
from matplotlib.colors import TwoSlopeNorm
import math

import pandas as pd
import matplotlib.pyplot as plt


INPUT_CSV = Path("jonghyeok/round1/output/pepper_all_results.csv")
OUTPUT_PATH = Path("jonghyeok/round1/output/pepper_heatmaps_all_in_one.png")

# history length마다 coeff / alpha 후보를 따로 지정
PEPPER_PARAM_GRID_BY_HISTORY = {
    19: {
        "coeffs": [0.10, 0.14, 0.18, 0.22, 0.26, 0.30, 0.34, 0.38, 0.42, 0.46],
        "alphas": [4, 6, 8, 10, 12, 14, 16, 18, 20],
    },
    39: {
        "coeffs": [0.10, 0.14, 0.18, 0.22, 0.26, 0.30, 0.34, 0.38, 0.42, 0.46],
        "alphas": [4, 6, 8, 10, 12, 14, 16, 18, 20],
    },
    69: {
        "coeffs": [0.10, 0.14, 0.18, 0.22, 0.26, 0.30, 0.34, 0.38, 0.42, 0.46],
        "alphas": [4, 6, 8, 10, 12, 14, 16, 18, 20],
    },
    99: {
        "coeffs": [0.10, 0.14, 0.18, 0.22, 0.26, 0.30, 0.34, 0.38, 0.42, 0.46],
        "alphas": [4, 6, 8, 10, 12, 14, 16, 18, 20],
    },
    199: {
        "coeffs": [0.10, 0.14, 0.18, 0.22, 0.26, 0.30, 0.34, 0.38, 0.42, 0.46],
        "alphas": [4, 6, 8, 10, 12, 14, 16, 18, 20],
    },
    299: {
        "coeffs": [0.10, 0.14, 0.18, 0.22, 0.26, 0.30, 0.34, 0.38, 0.42, 0.46],
        "alphas": [4, 6, 8, 10, 12, 14, 16, 18, 20],
    },
    399: {
        "coeffs": [0.10, 0.14, 0.18, 0.22, 0.26, 0.30, 0.34, 0.38, 0.42, 0.46],
        "alphas": [4, 6, 8, 10, 12, 14, 16, 18, 20],
    },
}


def main():
    df = pd.read_csv(INPUT_CSV)

    required_cols = {
        "PEPPER_HISTORY_LENGTH",
        "PEPPER_MUST_SELL_BUY_COEFF",
        "PEPPER_ALPHA",
        "total_profit",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {sorted(missing)}")

    df["PEPPER_MUST_SELL_BUY_COEFF"] = df["PEPPER_MUST_SELL_BUY_COEFF"].round(6)

    history_lengths = sorted(df["PEPPER_HISTORY_LENGTH"].unique())
    n = len(history_lengths)

    ncols = min(3, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(7 * ncols, 5 * nrows),
    )

    # axes를 2차원 배열처럼 맞춤
    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    elif ncols == 1:
        axes = [[ax] for ax in axes]

    for idx, history_length in enumerate(history_lengths):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]

        if history_length not in PEPPER_PARAM_GRID_BY_HISTORY:
            raise ValueError(
                f"PEPPER_PARAM_GRID_BY_HISTORY에 history_length={history_length} 설정이 없습니다."
            )

        config = PEPPER_PARAM_GRID_BY_HISTORY[history_length]
        coeffs = [round(float(x), 6) for x in config["coeffs"]]
        alphas = list(config["alphas"])

        sub = df[df["PEPPER_HISTORY_LENGTH"] == history_length].copy()

        pivot = sub.pivot_table(
            index="PEPPER_MUST_SELL_BUY_COEFF",
            columns="PEPPER_ALPHA",
            values="total_profit",
            aggfunc="mean",
        )

        pivot = pivot.reindex(index=coeffs, columns=alphas)

        # finite 값만 추출
        finite_vals = pivot.values[np.isfinite(pivot.values)]

        if len(finite_vals) == 0:
            axes[r][c].axis("off")
            continue

        # subplot별 robust color scale
        local_vmin = np.percentile(finite_vals, 5)
        local_vcenter = np.median(finite_vals)
        local_vmax = np.percentile(finite_vals, 95)

        # 혹시 값이 너무 비슷해서 같은 값이 되면 fallback
        if not (local_vmin < local_vcenter < local_vmax):
            local_vmin = float(np.min(finite_vals))
            local_vmax = float(np.max(finite_vals))
            local_vcenter = float(np.mean(finite_vals))

            if not (local_vmin < local_vcenter < local_vmax):
                # 완전히 거의 같은 값이면 아주 작은 폭 강제
                eps = 1e-6
                local_vmin -= eps
                local_vmax += eps

        norm = TwoSlopeNorm(
            vmin=local_vmin,
            vcenter=local_vcenter,
            vmax=local_vmax,
        )

        im = ax.imshow(
            pivot.values,
            aspect="auto",
            origin="lower",
            cmap="RdYlGn",
            norm=norm,
        )

        ax.set_title(f"PEPPER_HISTORY_LENGTH = {history_length}")
        ax.set_xlabel("PEPPER_ALPHA")
        ax.set_ylabel("PEPPER_MUST_SELL_BUY_COEFF")

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(x) for x in pivot.columns], rotation=45)

        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(y) for y in pivot.index])

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if pd.notna(val):
                    ax.text(j, i, f"{int(val)}", ha="center", va="center", fontsize=8)

    # 남는 subplot 숨기기
    total_slots = nrows * ncols
    for idx in range(n, total_slots):
        r = idx // ncols
        c = idx % ncols
        axes[r][c].axis("off")

    plt.suptitle("PEPPER heatmaps by history length", fontsize=18)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()