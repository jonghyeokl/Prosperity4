from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 현재 백테스트할 전략 파일 경로
TRADER_FILE = PROJECT_ROOT / "jonghyeok" / "round1" / "round1_JH_test.py"

# 결과 csv 파일 경로
OUTPUT_CSV = PROJECT_ROOT / "jonghyeok" / "round1" / "ash_param_sweep.csv"

# 백테스터 관련 경로
BACKTESTER_PYTHONPATH = str(PROJECT_ROOT / "imc-prosperity-3-backtester")
PYTHON_EXECUTABLE = str(PROJECT_ROOT / ".venv" / "bin" / "python")
DATA_PATH = str(PROJECT_ROOT / "data_capsule")

ROUND = "1"

EMA_WINDOWS = list(range(150, 181, 10))
EPSILONS = [i / 20 for i in range(5, 21)]  # 0.25 ~ 0.75, step 0.05


def format_epsilon(eps: float) -> str:
    return f"{eps:.2f}"


def extract_profit_summary_lines(stdout: str) -> list[str]:
    lines = stdout.splitlines()

    start_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "Profit summary:":
            start_idx = i + 1
            break

    if start_idx is None:
        raise ValueError("stdout에서 'Profit summary:'를 찾지 못했습니다.")

    summary_lines = []
    for line in lines[start_idx:]:
        stripped = line.rstrip()

        if not stripped:
            break

        if stripped.startswith("Successfully saved backtest results"):
            break

        summary_lines.append(stripped)

    if len(summary_lines) != 4:
        raise ValueError(
            f"Profit summary 줄 수가 4가 아닙니다. actual={len(summary_lines)}\n{summary_lines}"
        )

    return summary_lines

def run_backtest(window: int, epsilon: float) -> list[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = BACKTESTER_PYTHONPATH
    env["ASH_EMA_WINDOW"] = str(window)
    env["ASH_EPSILON"] = format_epsilon(epsilon)

    cmd = [
        PYTHON_EXECUTABLE,
        "-m",
        "prosperity3bt",
        str(TRADER_FILE),
        ROUND,
        "--data",
        DATA_PATH,
    ]

    print(f"Running window={window}, epsilon={format_epsilon(epsilon)}")

    completed = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            f"backtest failed\n"
            f"window={window}, epsilon={format_epsilon(epsilon)}\n\n"
            f"STDOUT:\n{completed.stdout}\n\n"
            f"STDERR:\n{completed.stderr}"
        )

    return extract_profit_summary_lines(completed.stdout)


def main() -> None:
    table: dict[str, dict[str, list[str]]] = {}

    for epsilon in EPSILONS:
        eps_key = format_epsilon(epsilon)
        table[eps_key] = {}

        for window in EMA_WINDOWS:
            summary_lines = run_backtest(window, epsilon)
            table[eps_key][str(window)] = summary_lines

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(
            f,
            delimiter=";",
            quoting=csv.QUOTE_ALL,
            lineterminator="\n",
        )

        header = ["ASH_EPSILON"] + [str(w) for w in EMA_WINDOWS]
        writer.writerow(header)

        for eps in EPSILONS:
            eps_key = format_epsilon(eps)

            row1 = [eps_key]
            row2 = [""]
            row3 = [""]
            row4 = [""]

            for window in EMA_WINDOWS:
                lines = table[eps_key][str(window)]  # 정확히 4줄
                row1.append(lines[0])
                row2.append(lines[1])
                row3.append(lines[2])
                row4.append(lines[3])

            writer.writerow(row1)
            writer.writerow(row2)
            writer.writerow(row3)
            writer.writerow(row4)

    print(f"Saved csv to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()