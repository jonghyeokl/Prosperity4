from __future__ import annotations

import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRADER_FILE = PROJECT_ROOT / "jonghyeok" / "round1" / "round1_test.py"
DATA_DIR = PROJECT_ROOT / "data_capsule"
PYTHON_EXECUTABLE = PROJECT_ROOT / ".venv" / "bin" / "python"
PYTHONPATH_DIR = PROJECT_ROOT / "imc-prosperity-3-backtester"

OUTPUT_DIR = PROJECT_ROOT / "jonghyeok" / "round1" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP20_TXT_PATH = OUTPUT_DIR / "ash_alpha_top20.txt"
TOP20_CSV_PATH = OUTPUT_DIR / "ash_alpha_top20.csv"
ALL_RESULTS_CSV_PATH = OUTPUT_DIR / "ash_alpha_all_results.csv"
FAILED_TXT_PATH = OUTPUT_DIR / "ash_alpha_failed.txt"
FAILED_CSV_PATH = OUTPUT_DIR / "ash_alpha_failed.csv"

ROUND = "1"

# M3 4P+4E 기준: 기본은 4 권장
MAX_WORKERS = int(os.getenv("ASH_SWEEP_WORKERS", "4"))

ALPHA1_VALUES = [-80]
ALPHA2_VALUES = [-75]
ALPHA3_VALUES = [-70]
ALPHA4_VALUES = [1]
ALPHA5_VALUES = [5]
ALPHA6_VALUES = [10]
ALPHA7_VALUES = [53]
ALPHA8_VALUES = [72]
ALPHA9_VALUES = [75]
ALPHA10_VALUES = [80]


def generate_candidates() -> list[tuple[int, int, int, int, int, int, int, int, int, int]]:
    candidates = []
    for a1 in ALPHA1_VALUES:
        for a2 in ALPHA2_VALUES:
            for a3 in ALPHA3_VALUES:
                for a4 in ALPHA4_VALUES:
                    for a5 in ALPHA5_VALUES:
                        for a6 in ALPHA6_VALUES:
                            for a7 in ALPHA7_VALUES:
                                for a8 in ALPHA8_VALUES:
                                    for a9 in ALPHA9_VALUES:
                                        for a10 in ALPHA10_VALUES:
                                            arr = [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10]
                                            if all(arr[i] <= arr[i + 1] for i in range(9)):
                                                candidates.append(tuple(arr))
    return candidates


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


def run_backtest(alpha_tuple: tuple[int, int, int, int, int, int, int, int, int, int]) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PYTHONPATH_DIR)

    for i, a in enumerate(alpha_tuple, start=1):
        env[f"ASH_ALPHA_{i}"] = str(a)

    cmd = [
        str(PYTHON_EXECUTABLE),
        "-m",
        "prosperity3bt",
        str(TRADER_FILE),
        ROUND,
        "--data",
        str(DATA_DIR),
    ]

    # 각 실행을 별도 임시 디렉토리에서 돌려서 backtests 로그 누적 방지
    with tempfile.TemporaryDirectory(prefix="ash_sweep_", dir=str(OUTPUT_DIR)) as tmpdir:
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
            "alpha1": alpha_tuple[0],
            "alpha2": alpha_tuple[1],
            "alpha3": alpha_tuple[2],
            "alpha4": alpha_tuple[3],
            "alpha5": alpha_tuple[4],
            "alpha6": alpha_tuple[5],
            "alpha7": alpha_tuple[6],
            "alpha8": alpha_tuple[7],
            "alpha9": alpha_tuple[8],
            "alpha10": alpha_tuple[9],
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
            "alpha1": alpha_tuple[0],
            "alpha2": alpha_tuple[1],
            "alpha3": alpha_tuple[2],
            "alpha4": alpha_tuple[3],
            "alpha5": alpha_tuple[4],
            "alpha6": alpha_tuple[5],
            "alpha7": alpha_tuple[6],
            "alpha8": alpha_tuple[7],
            "alpha9": alpha_tuple[8],
            "alpha10": alpha_tuple[9],
            "error": f"profit summary parse failed: {e}\n\nSTDOUT:\n{completed.stdout}",
        }

    return {
        "success": True,
        "alpha1": alpha_tuple[0],
        "alpha2": alpha_tuple[1],
        "alpha3": alpha_tuple[2],
        "alpha4": alpha_tuple[3],
        "alpha5": alpha_tuple[4],
        "alpha6": alpha_tuple[5],
        "alpha7": alpha_tuple[6],
        "alpha8": alpha_tuple[7],
        "alpha9": alpha_tuple[8],
        "alpha10": alpha_tuple[9],
        "day_-2_profit": day_m2,
        "day_-1_profit": day_m1,
        "day_0_profit": day_0,
        "total_profit": total,
    }


def main():
    candidates = generate_candidates()
    total_n = len(candidates)
    print(f"total valid candidates: {total_n}")
    print(f"max_workers: {MAX_WORKERS}")

    rows = []
    failed = []
    done = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_alpha = {
            executor.submit(run_backtest, alpha_tuple): alpha_tuple
            for alpha_tuple in candidates
        }

        for future in as_completed(future_to_alpha):
            done += 1
            alpha_tuple = future_to_alpha[future]

            try:
                result = future.result()
            except Exception as e:
                print(f"[{done}/{total_n}] FAILED: {alpha_tuple} -> {e}")
                failed.append(
                    {
                        "alpha1": alpha_tuple[0],
                        "alpha2": alpha_tuple[1],
                        "alpha3": alpha_tuple[2],
                        "alpha4": alpha_tuple[3],
                        "alpha5": alpha_tuple[4],
                        "alpha6": alpha_tuple[5],
                        "alpha7": alpha_tuple[6],
                        "alpha8": alpha_tuple[7],
                        "alpha9": alpha_tuple[8],
                        "alpha10": alpha_tuple[9],
                        "error": str(e),
                    }
                )
                continue

            if result["success"]:
                print(
                    f"[{done}/{total_n}] OK: "
                    f"({result['alpha1']}, {result['alpha2']}, {result['alpha3']}, "
                    f"{result['alpha4']}, {result['alpha5']}, {result['alpha6']}, "
                    f"{result['alpha7']}, {result['alpha8']}, {result['alpha9']}, {result['alpha10']}) "
                    f"-> total={result['total_profit']}"
                )
                rows.append({k: v for k, v in result.items() if k != "success"})
            else:
                print(f"[{done}/{total_n}] FAILED: {alpha_tuple}")
                failed.append({k: v for k, v in result.items() if k != "success"})

    if not rows:
        raise RuntimeError("성공한 백테스트가 하나도 없습니다.")

    df = pd.DataFrame(rows)
    df = df.sort_values(
        by=["total_profit", "day_0_profit", "day_-1_profit", "day_-2_profit"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    df.to_csv(ALL_RESULTS_CSV_PATH, index=False, encoding="utf-8-sig")

    top20 = df.head(20).copy()

    display_columns = [
        "alpha1",
        "alpha2",
        "alpha3",
        "alpha4",
        "alpha5",
        "alpha6",
        "alpha7",
        "alpha8",
        "alpha9",
        "alpha10",
        "day_-2_profit",
        "day_-1_profit",
        "day_0_profit",
        "total_profit",
    ]

    top20_str = top20[display_columns].to_string(index=False)

    print("\n=== TOP 20 ===")
    print(top20_str)

    TOP20_TXT_PATH.write_text(top20_str, encoding="utf-8")
    top20.to_csv(TOP20_CSV_PATH, index=False, encoding="utf-8-sig")

    print(f"\nSaved all results csv to: {ALL_RESULTS_CSV_PATH}")
    print(f"Saved top 20 txt to: {TOP20_TXT_PATH}")
    print(f"Saved top 20 csv to: {TOP20_CSV_PATH}")

    if failed:
        failed_df = pd.DataFrame(failed)
        failed_df.to_csv(FAILED_CSV_PATH, index=False, encoding="utf-8-sig")
        FAILED_TXT_PATH.write_text(failed_df.to_string(index=False), encoding="utf-8")
        print(f"Saved failed cases to: {FAILED_TXT_PATH}")
        print(f"Saved failed cases csv to: {FAILED_CSV_PATH}")


if __name__ == "__main__":
    main()