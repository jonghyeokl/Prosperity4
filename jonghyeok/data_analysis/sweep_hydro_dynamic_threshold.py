from __future__ import annotations

import argparse
import csv
import itertools
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional


PRODUCT = "HYDROGEL_PACK"
PRODUCT_LINE_RE = re.compile(r"^HYDROGEL_PACK:\s*(-?[\d,]+)\s*$")

DEFAULT_WINDOWS = [1000]
DEFAULT_ALPHAS = [0.015, 0.02, 0.025, 0.03]
DEFAULT_BETAS = [0.145, 0.15, 0.155]


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_hydrogel_pnl(stdout: str) -> int:
    total = 0
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        m = PRODUCT_LINE_RE.match(line)
        if not m:
            continue
        total += int(m.group(1).replace(",", ""))
    return total


def cleanup_backtest_logs() -> None:
    backtests_dir = Path("backtests")
    if not backtests_dir.exists():
        return

    for path in backtests_dir.glob("*.log"):
        try:
            path.unlink()
        except Exception:
            pass


def run_backtest(
    *,
    python_bin: str,
    module_path: str,
    algo_path: str,
    round_num: int,
    data_dir: str,
    valid_mid_history_length: int,
    alpha: float,
    beta: float,
    timeout: Optional[int],
) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = module_path
    env["HJ_VALID_MID_HISTORY_LENGTH"] = str(valid_mid_history_length)
    env["HJ_THRESHOLD_PARAM_ALPHA"] = str(alpha)
    env["HJ_THRESHOLD_PARAM_BETA"] = str(beta)

    cmd = [
        python_bin,
        "-m",
        "prosperity3bt",
        algo_path,
        str(round_num),
        "--data",
        data_dir,
    ]

    completed = subprocess.run(
        cmd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )

    cleanup_backtest_logs()

    if completed.returncode != 0:
        raise RuntimeError(
            "Backtest failed\n"
            f"cmd={' '.join(cmd)}\n"
            f"valid_mid_history_length={valid_mid_history_length}, alpha={alpha}, beta={beta}\n"
            f"output:\n{completed.stdout}"
        )

    return {
        "valid_mid_history_length": valid_mid_history_length,
        "alpha": alpha,
        "beta": beta,
        "pnl": parse_hydrogel_pnl(completed.stdout),
    }


def maybe_write_csv(path: Optional[str], rows: list[dict]) -> None:
    if not path:
        return

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["valid_mid_history_length", "alpha", "beta", "pnl"],
        )
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict], top_n: int) -> None:
    rows = sorted(rows, key=lambda r: r["pnl"], reverse=True)

    print("\n\n==================== SUMMARY ====================")
    for i, r in enumerate(rows[:top_n], start=1):
        print(
            f"{i:>3}. pnl={r['pnl']:,} | "
            f"valid_mid_history_length={r['valid_mid_history_length']}, "
            f"alpha={r['alpha']}, beta={r['beta']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="./jonghyeok/round3/HJ_hydro_dynamic_threshold_for_sweep.py")
    parser.add_argument("--round", type=int, default=3)
    parser.add_argument("--data", default="./data_capsule")
    parser.add_argument("--python", default="./.venv/bin/python")
    parser.add_argument("--py-path", default="./imc-prosperity-3-backtester")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=0, help="0 means no timeout")
    parser.add_argument("--out", default="", help="optional csv output path; empty means do not save")
    parser.add_argument("--top-n", type=int, default=30)

    parser.add_argument("--windows", default=",".join(map(str, DEFAULT_WINDOWS)))
    parser.add_argument("--alphas", default=",".join(map(str, DEFAULT_ALPHAS)))
    parser.add_argument("--betas", default=",".join(map(str, DEFAULT_BETAS)))

    args = parser.parse_args()

    windows = parse_int_list(args.windows)
    alphas = parse_float_list(args.alphas)
    betas = parse_float_list(args.betas)

    tasks = list(itertools.product(windows, alphas, betas))
    total = len(tasks)

    print(f"total runs: {total}, workers={args.workers}")
    cleanup_backtest_logs()

    rows: list[dict] = []
    done = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}

        for idx, (window, alpha, beta) in enumerate(tasks, start=1):
            print(f"[SUBMIT {idx}/{total}] window={window}, alpha={alpha}, beta={beta}")
            fut = executor.submit(
                run_backtest,
                python_bin=args.python,
                module_path=args.py_path,
                algo_path=args.algo,
                round_num=args.round,
                data_dir=args.data,
                valid_mid_history_length=window,
                alpha=alpha,
                beta=beta,
                timeout=None if args.timeout == 0 else args.timeout,
            )
            futures[fut] = (window, alpha, beta)

        for fut in as_completed(futures):
            window, alpha, beta = futures[fut]
            done += 1
            try:
                row = fut.result()
                rows.append(row)
                print(
                    f"[DONE {done}/{total}] pnl={row['pnl']:,} | "
                    f"window={window}, alpha={alpha}, beta={beta}"
                )
            except Exception as e:
                print(f"[FAIL {done}/{total}] window={window}, alpha={alpha}, beta={beta}")
                print(e)

    rows = sorted(rows, key=lambda r: r["pnl"], reverse=True)
    maybe_write_csv(args.out, rows)
    print_summary(rows, args.top_n)

    if args.out:
        print(f"\n[SAVED] {args.out}")

    cleanup_backtest_logs()


if __name__ == "__main__":
    main()
