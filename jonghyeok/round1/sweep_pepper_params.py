from __future__ import annotations

import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRADER_FILE = PROJECT_ROOT / "jonghyeok" / "round1" / "JH_PEPPER_TEST.py"
DATA_DIR = PROJECT_ROOT / "data_capsule"
PYTHON_EXECUTABLE = PROJECT_ROOT / ".venv" / "bin" / "python"
PYTHONPATH_DIR = PROJECT_ROOT / "imc-prosperity-3-backtester"

OUTPUT_DIR = PROJECT_ROOT / "jonghyeok" / "round1" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP20_TXT_PATH = OUTPUT_DIR / "pepper_top20.txt"
TOP20_CSV_PATH = OUTPUT_DIR / "pepper_top20.csv"
ALL_RESULTS_CSV_PATH = OUTPUT_DIR / "pepper_all_results.csv"
FAILED_TXT_PATH = OUTPUT_DIR / "pepper_failed.txt"
FAILED_CSV_PATH = OUTPUT_DIR / "pepper_failed.csv"
HEATMAP_OUTPUT_PATH = OUTPUT_DIR / "pepper_heatmaps_all_in_one.png"

ROUND = "1"
MAX_WORKERS = int(os.getenv("PEPPER_SWEEP_WORKERS", "4"))

PEPPER_PARAM_GRID_BY_HISTORY = {
    39: {
        "coeffs": [0.32, 0.325, 0.33, 0.335, 0.34, 0.345, 0.35, 0.355, 0.36, 0.365, 0.37, 0.375, 0.38, 0.385, 0.39, 0.395, 0.40],
        "alphas": [6, 7, 8, 9, 10, 11, 12, 13, 14],
    },
}

# 39: {
#         "coeffs": [0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40, 0.41, 0.42,],
#         "alphas": [6, 7, 8, 9, 10, 11, 12, 13],
#     },

RESULT_COLUMNS = [
    "PEPPER_HISTORY_LENGTH",
    "PEPPER_MUST_SELL_BUY_COEFF",
    "PEPPER_ALPHA",
    "day_-2_profit",
    "day_-1_profit",
    "day_0_profit",
    "total_profit",
]

KEY_VALUE = "min"

VALID_KEY_VALUES = {
    "total_profit",
    "day_-2_profit",
    "day_-1_profit",
    "day_0_profit",
    "min",
}

if KEY_VALUE not in VALID_KEY_VALUES:
    raise ValueError(f"KEY_VALUE must be one of {sorted(VALID_KEY_VALUES)}, got: {KEY_VALUE}")


def metric_label() -> str:
    if KEY_VALUE == "min":
        return "min(day_-2_profit, day_-1_profit, day_0_profit)"
    return KEY_VALUE


def add_metric_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in ["day_-2_profit", "day_-1_profit", "day_0_profit", "total_profit"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if KEY_VALUE == "min":
        out["__metric_value"] = out[["day_-2_profit", "day_-1_profit", "day_0_profit"]].min(axis=1)
    else:
        out["__metric_value"] = pd.to_numeric(out[KEY_VALUE], errors="coerce")

    return out


def candidate_key(history_length: int, coeff: float, alpha: int) -> tuple[int, str, int]:
    return (int(history_length), f"{float(coeff):.10f}", int(alpha))


def generate_candidates() -> list[tuple[int, float, int]]:
    out = []
    for history_length, config in PEPPER_PARAM_GRID_BY_HISTORY.items():
        coeffs = config["coeffs"]
        alphas = config["alphas"]

        for coeff in coeffs:
            for alpha in alphas:
                out.append((history_length, coeff, alpha))
    return out


def load_existing_results() -> pd.DataFrame:
    if not ALL_RESULTS_CSV_PATH.exists():
        return pd.DataFrame(columns=RESULT_COLUMNS)

    df = pd.read_csv(ALL_RESULTS_CSV_PATH)

    # 필요한 컬럼만 없으면 에러
    missing = set(RESULT_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{ALL_RESULTS_CSV_PATH} 에 필요한 컬럼이 없습니다: {sorted(missing)}")

    return df.copy()


def existing_candidate_keys(df: pd.DataFrame) -> set[tuple[int, str, int]]:
    if df.empty:
        return set()

    return {
        candidate_key(row["PEPPER_HISTORY_LENGTH"], row["PEPPER_MUST_SELL_BUY_COEFF"], row["PEPPER_ALPHA"])
        for _, row in df.iterrows()
    }


def extract_profit_summary(stdout: str) -> tuple[int, int, int, int]:
    lines = stdout.splitlines()

    start_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "Profit summary:":
            start_idx = i + 1
            break

    if start_idx is None:
        raise ValueError("Profit summary 섹션을 찾지 못했습니다.")

    summary_lines = []
    for line in lines[start_idx:]:
        stripped = line.strip()
        if not stripped:
            break
        if stripped.startswith("Successfully saved backtest results"):
            break
        summary_lines.append(stripped)

    if len(summary_lines) < 4:
        raise ValueError(f"Profit summary 줄 수가 이상합니다: {summary_lines}")

    day_profit = {}
    total_profit = None

    day_pattern = re.compile(r"Round 1 day (-?\d+):\s*([\d,]+)")
    total_pattern = re.compile(r"Total profit:\s*([\d,]+)")

    for line in summary_lines:
        day_match = day_pattern.fullmatch(line)
        if day_match:
            day = int(day_match.group(1))
            profit = int(day_match.group(2).replace(",", ""))
            day_profit[day] = profit
            continue

        total_match = total_pattern.fullmatch(line)
        if total_match:
            total_profit = int(total_match.group(1).replace(",", ""))

    if -2 not in day_profit or -1 not in day_profit or 0 not in day_profit or total_profit is None:
        raise ValueError(f"Profit summary 파싱 실패: {summary_lines}")

    return day_profit[-2], day_profit[-1], day_profit[0], total_profit


def run_backtest(candidate: tuple[int, float, int]) -> dict:
    history_length, coeff, alpha = candidate

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PYTHONPATH_DIR)
    env["PEPPER_HISTORY_LENGTH"] = str(history_length)
    env["PEPPER_MUST_SELL_BUY_COEFF"] = f"{float(coeff):.10f}"
    env["PEPPER_ALPHA"] = str(alpha)

    cmd = [
        str(PYTHON_EXECUTABLE),
        "-m",
        "prosperity3bt",
        str(TRADER_FILE),
        ROUND,
        "--data",
        str(DATA_DIR),
    ]

    with tempfile.TemporaryDirectory(prefix="pepper_sweep_", dir=str(OUTPUT_DIR)) as tmpdir:
        completed = subprocess.run(
            cmd,
            cwd=tmpdir,
            capture_output=True,
            text=True,
            env=env,
        )

    if completed.returncode != 0:
        return {
            "success": False,
            "PEPPER_HISTORY_LENGTH": history_length,
            "PEPPER_MUST_SELL_BUY_COEFF": coeff,
            "PEPPER_ALPHA": alpha,
            "error": (
                f"backtest failed\n"
                f"STDOUT:\n{completed.stdout}\n\n"
                f"STDERR:\n{completed.stderr}"
            ),
        }

    try:
        day_m2, day_m1, day_0, total = extract_profit_summary(completed.stdout)
    except Exception as e:
        return {
            "success": False,
            "PEPPER_HISTORY_LENGTH": history_length,
            "PEPPER_MUST_SELL_BUY_COEFF": coeff,
            "PEPPER_ALPHA": alpha,
            "error": f"profit summary parse failed: {e}\n\nSTDOUT:\n{completed.stdout}",
        }

    return {
        "success": True,
        "PEPPER_HISTORY_LENGTH": history_length,
        "PEPPER_MUST_SELL_BUY_COEFF": coeff,
        "PEPPER_ALPHA": alpha,
        "day_-2_profit": day_m2,
        "day_-1_profit": day_m1,
        "day_0_profit": day_0,
        "total_profit": total,
    }


def save_top20(df: pd.DataFrame) -> None:
    top20 = df.sort_values(
        by=["__metric_value", "total_profit", "day_0_profit", "day_-1_profit", "day_-2_profit"],
        ascending=[False, False, False, False, False],
    ).head(20).copy()

    cols = [
        "PEPPER_HISTORY_LENGTH",
        "PEPPER_MUST_SELL_BUY_COEFF",
        "PEPPER_ALPHA",
        "day_-2_profit",
        "day_-1_profit",
        "day_0_profit",
        "total_profit",
        "__metric_value",
    ]
    top20 = top20[cols].rename(columns={"__metric_value": f"metric({metric_label()})"})

    top20_str = top20.to_string(index=False)
    print(f"\n=== TOP 20 by {metric_label()} ===")
    print(top20_str)
    TOP20_TXT_PATH.write_text(top20_str, encoding="utf-8")
    top20.to_csv(TOP20_CSV_PATH, index=False, encoding="utf-8-sig")


def plot_heatmaps(df: pd.DataFrame) -> None:
    """
    최근 방식:
    - 한 화면 subplot
    - history_length마다 하나씩
    - colorbar 없음
    - subplot별 local color scale
    """
    if df.empty:
        return

    history_lengths = PEPPER_PARAM_GRID_BY_HISTORY.keys()
    n = len(history_lengths)

    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(7 * ncols, 5 * nrows),
    )

    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    elif ncols == 1:
        axes = [[ax] for ax in axes]

    df = add_metric_column(df)
    df["PEPPER_HISTORY_LENGTH"] = pd.to_numeric(df["PEPPER_HISTORY_LENGTH"], errors="coerce")
    df["PEPPER_MUST_SELL_BUY_COEFF"] = pd.to_numeric(df["PEPPER_MUST_SELL_BUY_COEFF"], errors="coerce").round(6)
    df["PEPPER_ALPHA"] = pd.to_numeric(df["PEPPER_ALPHA"], errors="coerce")

    for idx, history_length in enumerate(history_lengths):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]

        if history_length not in PEPPER_PARAM_GRID_BY_HISTORY:
            raise ValueError(f"Missing config for history_length={history_length}")

        config = PEPPER_PARAM_GRID_BY_HISTORY[history_length]
        coeffs = [round(float(x), 6) for x in config["coeffs"]]
        alphas = list(config["alphas"])

        sub = df[df["PEPPER_HISTORY_LENGTH"] == history_length].copy()

        pivot = sub.pivot_table(
            index="PEPPER_MUST_SELL_BUY_COEFF",
            columns="PEPPER_ALPHA",
            values="__metric_value",
            aggfunc="mean",
        )
        pivot = pivot.reindex(index=coeffs, columns=alphas)

        pivot = pivot.apply(pd.to_numeric, errors="coerce")
        pivot_values = pivot.to_numpy(dtype=float)

        finite_vals = pivot_values[np.isfinite(pivot_values)]
        if len(finite_vals) == 0:
            ax.axis("off")
            continue

        local_vmin = np.percentile(finite_vals, 5)
        local_vcenter = np.median(finite_vals)
        local_vmax = np.percentile(finite_vals, 95)

        if not (local_vmin < local_vcenter < local_vmax):
            local_vmin = float(np.min(finite_vals))
            local_vmax = float(np.max(finite_vals))
            local_vcenter = float(np.mean(finite_vals))

            if not (local_vmin < local_vcenter < local_vmax):
                eps = 1e-6
                local_vmin -= eps
                local_vmax += eps
                local_vcenter = (local_vmin + local_vmax) / 2

        if not np.all(np.isfinite([local_vmin, local_vcenter, local_vmax])):
            ax.axis("off")
            continue

        if local_vmin == local_vmax:
            local_vmin -= 1e-6
            local_vmax += 1e-6

        if not (local_vmin < local_vcenter < local_vmax):
            local_vcenter = (local_vmin + local_vmax) / 2

        norm = TwoSlopeNorm(
            vmin=local_vmin,
            vcenter=local_vcenter,
            vmax=local_vmax,
        )

        im = ax.imshow(
            pivot_values,
            aspect="auto",
            origin="lower",
            cmap="RdYlGn",
            norm=norm,
        )

        im.get_cursor_data = lambda event: None
        im.format_cursor_data = lambda data: ""
        im.set_mouseover(False)

        norm = TwoSlopeNorm(
            vmin=local_vmin,
            vcenter=local_vcenter,
            vmax=local_vmax,
        )

        im = ax.imshow(
            pivot_values,
            aspect="auto",
            origin="lower",
            cmap="RdYlGn",
            norm=norm,
        )

        # matplotlib mouse hover cursor formatting bug 회피
        im.get_cursor_data = lambda event: None
        im.format_cursor_data = lambda data: ""
        im.set_mouseover(False)

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
                    ax.text(j, i, f"{int(val)}", ha="center", va="center", fontsize=6)

    total_slots = nrows * ncols
    for idx in range(n, total_slots):
        r = idx // ncols
        c = idx % ncols
        axes[r][c].axis("off")

    plt.suptitle(f"PEPPER heatmaps by history length ({metric_label()})", fontsize=18)
    plt.tight_layout()
    plt.savefig(HEATMAP_OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.show()


def main():
    candidates = generate_candidates()
    existing_df = load_existing_results()
    existing_keys = existing_candidate_keys(existing_df)

    candidates_to_run = [
        c for c in candidates
        if candidate_key(c[0], c[1], c[2]) not in existing_keys
    ]

    print(f"total configured candidates: {len(candidates)}")
    print(f"already existing rows: {len(existing_df)}")
    print(f"candidates to run: {len(candidates_to_run)}")
    print(f"max_workers: {MAX_WORKERS}")

    new_rows = []
    failed = []
    done = 0
    total_n = len(candidates_to_run)

    if total_n > 0:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_candidate = {
                executor.submit(run_backtest, candidate): candidate
                for candidate in candidates_to_run
            }

            for future in as_completed(future_to_candidate):
                done += 1
                candidate = future_to_candidate[future]

                try:
                    result = future.result()
                except Exception as e:
                    history_length, coeff, alpha = candidate
                    print(f"[{done}/{total_n}] FAILED: {candidate} -> {e}")
                    failed.append(
                        {
                            "PEPPER_HISTORY_LENGTH": history_length,
                            "PEPPER_MUST_SELL_BUY_COEFF": coeff,
                            "PEPPER_ALPHA": alpha,
                            "error": str(e),
                        }
                    )
                    continue

                if result["success"]:
                    print(
                        f"[{done}/{total_n}] OK: "
                        f"(H={result['PEPPER_HISTORY_LENGTH']}, "
                        f"C={result['PEPPER_MUST_SELL_BUY_COEFF']}, "
                        f"A={result['PEPPER_ALPHA']}) "
                        f"-> total={result['total_profit']}"
                    )
                    new_rows.append({k: v for k, v in result.items() if k != "success"})
                else:
                    print(f"[{done}/{total_n}] FAILED: {candidate}")
                    failed.append({k: v for k, v in result.items() if k != "success"})

    new_df = pd.DataFrame(new_rows, columns=RESULT_COLUMNS) if new_rows else pd.DataFrame(columns=RESULT_COLUMNS)

    # 기존 행 유지 + 새 행 추가
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)

    if combined_df.empty:
        raise RuntimeError("결과가 하나도 없습니다.")

    metric_df = add_metric_column(combined_df)

    metric_df = metric_df.sort_values(
        by=["__metric_value", "total_profit", "day_0_profit", "day_-1_profit", "day_-2_profit"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)

    # csv에는 helper column 없이 저장
    combined_df_sorted = metric_df.drop(columns=["__metric_value"])
    combined_df_sorted.to_csv(ALL_RESULTS_CSV_PATH, index=False, encoding="utf-8-sig")

    save_top20(metric_df)
    plot_heatmaps(metric_df)

    print(f"\nSaved all results csv to: {ALL_RESULTS_CSV_PATH}")
    print(f"Saved top 20 txt to: {TOP20_TXT_PATH}")
    print(f"Saved top 20 csv to: {TOP20_CSV_PATH}")
    print(f"Saved heatmaps to: {HEATMAP_OUTPUT_PATH}")

    if failed:
        failed_df = pd.DataFrame(failed)
        failed_df.to_csv(FAILED_CSV_PATH, index=False, encoding="utf-8-sig")
        FAILED_TXT_PATH.write_text(failed_df.to_string(index=False), encoding="utf-8")
        print(f"Saved failed cases to: {FAILED_TXT_PATH}")
        print(f"Saved failed cases csv to: {FAILED_CSV_PATH}")


if __name__ == "__main__":
    main()