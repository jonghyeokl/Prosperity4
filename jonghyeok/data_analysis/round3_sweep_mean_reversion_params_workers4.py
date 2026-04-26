from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List


SMILE_WINDOW_CANDIDATES = [10, 300]

TARGET_PRODUCTS = [
    "VEV_5000",
    "VEV_5100",
    "VEV_5200",
    "VEV_5300",
]

EMA_WINDOW_CANDIDATES = [10, 20, 30, 40, 50, 60, 70]
BETA_CANDIDATES = [-1.0]

PRODUCT_LINE_RE = re.compile(r"^(VEV_5000|VEV_5100|VEV_5200|VEV_5300):\s*(-?[\d,]+)\s*$")


def parse_product_pnls(stdout: str) -> Dict[str, int]:
    """
    prosperity3bt prints each product PnL once per day.
    This returns 3-day sum per product.
    """
    sums = {p: 0 for p in TARGET_PRODUCTS}

    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        m = PRODUCT_LINE_RE.match(line)
        if not m:
            continue

        product = m.group(1)
        value = int(m.group(2).replace(",", ""))
        sums[product] += value

    return sums


def run_backtest(
    *,
    python_bin: str,
    module_path: str,
    algo_path: str,
    round_num: int,
    data_dir: str,
    smile_window: int,
    target_product: str,
    ema_window: int,
    beta: float,
    timeout: int | None,
) -> Dict[str, int]:
    env = os.environ.copy()
    env["PYTHONPATH"] = module_path
    env["HJ_SMILE_WINDOW"] = str(smile_window)
    env["HJ_TARGET_PRODUCT"] = target_product
    env["HJ_EMA_WINDOW"] = str(ema_window)
    env["HJ_BETA"] = str(beta)

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

    if completed.returncode != 0:
        raise RuntimeError(
            "Backtest failed\n"
            f"cmd={' '.join(cmd)}\n"
            f"smile_window={smile_window}, product={target_product}, "
            f"ema_window={ema_window}, beta={beta}\n"
            f"output:\n{completed.stdout}"
        )

    return parse_product_pnls(completed.stdout)


def append_result(path: Path, row: dict) -> None:
    exists = path.exists()

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "smile_window",
                "product",
                "ema_window",
                "beta",
                "product_pnl_3d",
                "VEV_5000_pnl_3d",
                "VEV_5100_pnl_3d",
                "VEV_5200_pnl_3d",
                "VEV_5300_pnl_3d",
            ],
        )

        if not exists:
            writer.writeheader()

        writer.writerow(row)


def read_existing(path: Path) -> set[tuple[int, str, int, float]]:
    if not path.exists():
        return set()

    seen = set()

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seen.add(
                (
                    int(row["smile_window"]),
                    row["product"],
                    int(row["ema_window"]),
                    float(row["beta"]),
                )
            )

    return seen


def load_results(path: Path) -> list[dict]:
    if not path.exists():
        return []

    rows = []

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "smile_window": int(row["smile_window"]),
                    "product": row["product"],
                    "ema_window": int(row["ema_window"]),
                    "beta": float(row["beta"]),
                    "product_pnl_3d": int(row["product_pnl_3d"]),
                    "VEV_5000_pnl_3d": int(row["VEV_5000_pnl_3d"]),
                    "VEV_5100_pnl_3d": int(row["VEV_5100_pnl_3d"]),
                    "VEV_5200_pnl_3d": int(row["VEV_5200_pnl_3d"]),
                    "VEV_5300_pnl_3d": int(row["VEV_5300_pnl_3d"]),
                }
            )

    return rows


def print_summary(rows: list[dict]) -> None:
    print("\n\n==================== SUMMARY ====================")

    best_by_window_product: dict[tuple[int, str], dict] = {}

    for smile_window in SMILE_WINDOW_CANDIDATES:
        print(f"\n\n===== SMILE_WINDOW_PER_VOUCHER = {smile_window} =====")

        for product in TARGET_PRODUCTS:
            sub = [
                r for r in rows
                if r["smile_window"] == smile_window and r["product"] == product
            ]

            if not sub:
                print(f"\n[{product}] no results")
                continue

            sub = sorted(sub, key=lambda r: r["product_pnl_3d"], reverse=True)
            best_by_window_product[(smile_window, product)] = sub[0]

            print(f"\n[{product}] top 5")
            for r in sub[:5]:
                print(
                    f"  ema_window={r['ema_window']:>2}, "
                    f"beta={r['beta']:>7.3f}, "
                    f"{product}_3d_pnl={r['product_pnl_3d']:,}"
                )

        best_sum = 0
        missing = False
        best_params = {}

        for product in TARGET_PRODUCTS:
            best = best_by_window_product.get((smile_window, product))
            if best is None:
                missing = True
                continue

            best_sum += best["product_pnl_3d"]
            best_params[product] = {
                "ema_window": best["ema_window"],
                "beta": best["beta"],
                "pnl": best["product_pnl_3d"],
            }

        if missing:
            print("\n[window best total] incomplete")
        else:
            print("\n[window best total: VEV_5000~VEV_5300 3d pnl sum]")
            print(f"  total={best_sum:,}")
            for product, p in best_params.items():
                print(
                    f"  {product}: ema_window={p['ema_window']}, "
                    f"beta={p['beta']}, pnl={p['pnl']:,}"
                )

    print("\n\n===== OVERALL WINDOW RANKING BY SUM OF PRODUCT-WISE BEST PNLS =====")
    totals = []
    for smile_window in SMILE_WINDOW_CANDIDATES:
        vals = [
            best_by_window_product[(smile_window, product)]["product_pnl_3d"]
            for product in TARGET_PRODUCTS
            if (smile_window, product) in best_by_window_product
        ]
        if len(vals) == len(TARGET_PRODUCTS):
            totals.append((smile_window, sum(vals)))

    for smile_window, total in sorted(totals, key=lambda x: x[1], reverse=True):
        print(f"  smile_window={smile_window:>4}: total={total:,}")



def build_tasks(seen: set[tuple[int, str, int, float]]) -> list[dict]:
    tasks = []
    run_idx = 0

    for smile_window in SMILE_WINDOW_CANDIDATES:
        for product in TARGET_PRODUCTS:
            for ema_window in EMA_WINDOW_CANDIDATES:
                for beta in BETA_CANDIDATES:
                    run_idx += 1
                    key = (smile_window, product, ema_window, float(beta))

                    if key in seen:
                        continue

                    tasks.append(
                        {
                            "run_idx": run_idx,
                            "key": key,
                            "smile_window": smile_window,
                            "product": product,
                            "ema_window": ema_window,
                            "beta": beta,
                        }
                    )

    return tasks


def run_one_task(task: dict, args: argparse.Namespace) -> dict:
    smile_window = task["smile_window"]
    product = task["product"]
    ema_window = task["ema_window"]
    beta = task["beta"]

    pnls = run_backtest(
        python_bin=args.python,
        module_path=args.py_path,
        algo_path=args.algo,
        round_num=args.round,
        data_dir=args.data,
        smile_window=smile_window,
        target_product=product,
        ema_window=ema_window,
        beta=beta,
        timeout=None if args.timeout == 0 else args.timeout,
    )

    row = {
        "smile_window": smile_window,
        "product": product,
        "ema_window": ema_window,
        "beta": beta,
        "product_pnl_3d": pnls[product],
        "VEV_5000_pnl_3d": pnls["VEV_5000"],
        "VEV_5100_pnl_3d": pnls["VEV_5100"],
        "VEV_5200_pnl_3d": pnls["VEV_5200"],
        "VEV_5300_pnl_3d": pnls["VEV_5300"],
    }

    return {
        "task": task,
        "row": row,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="./jonghyeok/round3/HJ_params_fasttest_for_sweep.py")
    parser.add_argument("--round", type=int, default=3)
    parser.add_argument("--data", default="./data_capsule")
    parser.add_argument("--python", default="./.venv/bin/python")
    parser.add_argument("--py-path", default="./imc-prosperity-3-backtester")
    parser.add_argument(
        "--out",
        default="./jonghyeok/data_analysis/output/round3_mean_reversion_sweep_results.csv",
    )
    parser.add_argument("--timeout", type=int, default=0, help="0 means no timeout")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen = read_existing(out_path) if args.resume else set()

    total_runs = (
        len(SMILE_WINDOW_CANDIDATES)
        * len(TARGET_PRODUCTS)
        * len(EMA_WINDOW_CANDIDATES)
        * len(BETA_CANDIDATES)
    )

    tasks = build_tasks(seen)
    skipped = total_runs - len(tasks)

    if skipped:
        print(f"[RESUME] skipped existing runs: {skipped}/{total_runs}")

    print(f"[START] total_runs={total_runs}, pending={len(tasks)}, workers={args.workers}")

    completed_count = skipped

    if tasks:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_task = {}

            for task in tasks:
                print(
                    f"[SUBMIT {task['run_idx']}/{total_runs}] "
                    f"smile={task['smile_window']}, product={task['product']}, "
                    f"ema={task['ema_window']}, beta={task['beta']}"
                )
                future = executor.submit(run_one_task, task, args)
                future_to_task[future] = task

            for future in as_completed(future_to_task):
                task = future_to_task[future]

                try:
                    result = future.result()
                except Exception as e:
                    print(
                        "\n[FAILED] "
                        f"run={task['run_idx']}/{total_runs}, "
                        f"smile={task['smile_window']}, product={task['product']}, "
                        f"ema={task['ema_window']}, beta={task['beta']}"
                    )
                    print(str(e))
                    raise

                row = result["row"]
                append_result(out_path, row)

                completed_count += 1
                print(
                    f"[DONE {completed_count}/{total_runs}] "
                    f"smile={row['smile_window']}, product={row['product']}, "
                    f"ema={row['ema_window']}, beta={row['beta']} "
                    f"-> {row['product']}_3d_pnl={row['product_pnl_3d']:,}"
                )

    rows = load_results(out_path)
    print_summary(rows)
    print(f"\n[SAVED] {out_path}")


if __name__ == "__main__":
    main()
