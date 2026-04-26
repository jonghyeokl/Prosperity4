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


HYDRO_RE = re.compile(r"^HYDROGEL_PACK:\s*(-?[\d,]+)\s*$")


def parse_number_list(raw: str, cast):
    if raw.strip() == "":
        return []
    return [cast(x.strip()) for x in raw.split(",") if x.strip()]


def parse_hydrogel_pnls(stdout: str) -> list[int]:
    """
    Backtester stdout에서 HYDROGEL_PACK pnl 라인을 모두 추출한다.

    예상 예시:
        HYDROGEL_PACK: 12,345
        HYDROGEL_PACK: 23,456
        HYDROGEL_PACK: 34,567

    반환:
        [12345, 23456, 34567]
    """
    pnls = []

    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        m = HYDRO_RE.match(line)
        if m:
            pnls.append(int(m.group(1).replace(",", "")))

    return pnls


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

    pnls = parse_hydrogel_pnls(completed.stdout)

    row = dict(params["row"])

    # day별 pnl 저장
    for day_idx, pnl in enumerate(pnls):
        row[f"hydrogel_pnl_day{day_idx}"] = pnl

    # 혹시 3일보다 적게 잡히는 경우에도 컬럼 모양을 일정하게 유지
    for day_idx in range(len(pnls), 3):
        row[f"hydrogel_pnl_day{day_idx}"] = 0

    row["hydrogel_pnl_3d"] = sum(pnls)

    return row


def write_rows(path: str, rows: list[dict[str, Any]]) -> None:
    if not path:
        return

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    preferred_order = [
        "valid_mid_history_length",
        "z_score_threshold",
        "hydrogel_pnl_day0",
        "hydrogel_pnl_day1",
        "hydrogel_pnl_day2",
        "hydrogel_pnl_3d",
    ]

    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())

    fieldnames = [k for k in preferred_order if k in all_keys]
    fieldnames += [k for k in rows[0].keys() if k not in fieldnames]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, Any]], top_n: int) -> None:
    print("\n\n==================== SUMMARY ====================")
    rows = sorted(rows, key=lambda r: r["hydrogel_pnl_3d"], reverse=True)

    for i, row in enumerate(rows[:top_n], start=1):
        length = row.get("valid_mid_history_length")
        z = row.get("z_score_threshold")

        day0 = row.get("hydrogel_pnl_day0", 0)
        day1 = row.get("hydrogel_pnl_day1", 0)
        day2 = row.get("hydrogel_pnl_day2", 0)
        total = row.get("hydrogel_pnl_3d", 0)

        print(
            f"{i:>3}. total={total:,} "
            f"| day0={day0:,}, day1={day1:,}, day2={day2:,} "
            f"| length={length}, z={z}"
        )


def build_param_grid(args) -> list[dict[str, Any]]:
    lengths = parse_number_list(args.lengths, int)
    z_thresholds = parse_number_list(args.z_thresholds, float)

    out = []

    for length in lengths:
        for z in z_thresholds:
            out.append({
                "env": {
                    "HJ_VALID_MID_HISTORY_LENGTH": length,
                    "HJ_Z_SCORE_THRESHOLD": z,
                },
                "row": {
                    "valid_mid_history_length": length,
                    "z_score_threshold": z,
                },
            })

    return out


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--algo",
        default="./jonghyeok/round3/HJ_hydro_rolling_zscore_for_sweep.py",
    )
    parser.add_argument("--round", type=int, default=3)
    parser.add_argument("--data", default="./data_capsule")

    # Windows에서는 실행할 때 --python py 로 넘기면 됨
    parser.add_argument("--python", default="./.venv/bin/python")

    parser.add_argument("--py-path", default="./imc-prosperity-3-backtester")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=0, help="0 means no timeout")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--out", default="", help="Optional csv path. Empty means no csv saved.")

    parser.add_argument("--lengths", default="900,1000,1100")
    parser.add_argument("--z-thresholds", default="0.95,1.0,1.05,1.1")

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

            length = row.get("valid_mid_history_length")
            z = row.get("z_score_threshold")

            day0 = row.get("hydrogel_pnl_day0", 0)
            day1 = row.get("hydrogel_pnl_day1", 0)
            day2 = row.get("hydrogel_pnl_day2", 0)
            total_pnl = row.get("hydrogel_pnl_3d", 0)

            print(
                f"[DONE {done}/{total}] "
                f"total={total_pnl:,} "
                f"| day0={day0:,}, day1={day1:,}, day2={day2:,} "
                f"| length={length}, z={z}"
            )

    cleanup_backtest_outputs(start_time)
    write_rows(args.out, rows)
    print_summary(rows, args.top_n)

    if args.out:
        print(f"\n[SAVED] {args.out}")
    else:
        print("\n[NO CSV SAVED] use --out to save sweep summary")


if __name__ == "__main__":
    main()

"""
./.venv/bin/python jonghyeok/data_analysis/sweep_hydro_rolling_zscore.py \  --algo ./jonghyeok/round3/HJ_hydro_rolling_zscore_for_sweep.py \
  —round 3 \
  —data ./data_capsule \
  —python ./.venv/bin/python \
  —py-path ./imc-prosperity-3-backtester \
  —workers 4
"""