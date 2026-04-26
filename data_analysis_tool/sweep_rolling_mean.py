from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PRODUCT = "VELVETFRUIT_EXTRACT"
PRODUCT_RE = re.compile(f"^{PRODUCT}:\s*(-?[\d,]+)\s*$")


def parse_number_list(raw: str, cast):
    if raw.strip() == "":
        return []
    return [cast(x.strip()) for x in raw.split(",") if x.strip()]


def parse_product_pnl(stdout: str) -> int:
    total = 0
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        m = PRODUCT_RE.match(line)
        if m:
            total += int(m.group(1).replace(",", ""))
    return total


def cleanup_backtest_outputs(start_time: float) -> None:
    backtests_dir = Path("backtests")
    if not backtests_dir.exists():
        return

    for path in list(backtests_dir.iterdir()):
        try:
            if path.stat().st_mtime >= start_time - 2.0:
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink(missing_ok=True)
        except Exception:
            pass


def run_one(args, params: dict[str, Any], start_time: float) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = args.py_path

    for k, v in params["env"].items():
        env[k] = str(v)

    cmd = [
        args.python,
        "-m",
        "prosperity3bt",
        args.algo,
        str(args.round),
        "--data",
        args.data,
    ]

    completed = subprocess.run(
        cmd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=None if args.timeout == 0 else args.timeout,
    )

    cleanup_backtest_outputs(start_time)

    if completed.returncode != 0:
        raise RuntimeError(
            "Backtest failed\n"
            f"cmd={' '.join(cmd)}\n"
            f"params={params}\n"
            f"output:\n{completed.stdout}"
        )

    row = dict(params["row"])
    row[f"{PRODUCT}_pnl_3d"] = parse_product_pnl(completed.stdout)
    return row


def write_rows(path: str, rows: list[dict[str, Any]]) -> None:
    if not path:
        return

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, Any]], top_n: int) -> None:
    print("\n\n==================== SUMMARY ====================")
    rows = sorted(rows, key=lambda r: r[f"{PRODUCT}_pnl_3d"], reverse=True)

    for i, row in enumerate(rows[:top_n], start=1):
        params = ", ".join(
            f"{k}={v}" for k, v in row.items()
            if k != f"{PRODUCT}_pnl_3d"
        )
        print(f"{i:>3}. pnl={row[f'{PRODUCT}_pnl_3d']:,} | {params}")


def build_param_grid(args) -> list[dict[str, Any]]:
    lengths = parse_number_list(args.lengths, int)
    thresholds = parse_number_list(args.thresholds, float)

    out = []
    for length in lengths:
        for threshold in thresholds:
            out.append({
                "env": {
                    "HJ_VALID_MID_HISTORY_LENGTH": length,
                    "HJ_THRESHOLD": threshold,
                },
                "row": {
                    "valid_mid_history_length": length,
                    "threshold": threshold,
                },
            })
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="./data_analysis_tool/rolling_mean_for_sweep.py")
    parser.add_argument("--round", type=int, default=3)
    parser.add_argument("--data", default="./data_capsule")
    parser.add_argument("--python", default="./.venv/bin/python")
    parser.add_argument("--py-path", default="./imc-prosperity-3-backtester")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=0, help="0 means no timeout")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--out", default="", help="Optional csv path. Empty means no csv saved.")
    parser.add_argument("--lengths", default="10,20,30,40,50,70,100,150,200,300,500,750,1000") #10,20,30,40,50,70,100,150,200,300,500,750,1000
    parser.add_argument("--thresholds", default="0,0.5,1,1.5,2,3,5,8,10") #0,0.5,1,1.5,2,3,5,8,10
    args = parser.parse_args()

    params_list = build_param_grid(args)
    total = len(params_list)

    print(f"total runs: {total}, workers={args.workers}")
    start_time = time.time()

    rows: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []

        for idx, params in enumerate(params_list, start=1):
            print(f"[SUBMIT {idx}/{total}] {params['row']}")
            futures.append(executor.submit(run_one, args, params, start_time))

        done = 0
        for fut in as_completed(futures):
            done += 1
            row = fut.result()
            rows.append(row)
            params = ", ".join(
                f"{k}={v}" for k, v in row.items()
                if k != f"{PRODUCT}_pnl_3d"
            )
            print(f"[DONE {done}/{total}] pnl={row[f'{PRODUCT}_pnl_3d']:,} | {params}")

    cleanup_backtest_outputs(start_time)
    write_rows(args.out, rows)
    print_summary(rows, args.top_n)

    if args.out:
        print(f"\n[SAVED] {args.out}")
    else:
        print("\n[NO CSV SAVED] use --out to save sweep summary")


if __name__ == "__main__":
    main()
