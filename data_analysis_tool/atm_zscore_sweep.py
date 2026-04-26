from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ATM_PRODUCTS = [
    "VEV_5000",
    "VEV_5100",
    "VEV_5200",
    "VEV_5300",
]

PRODUCT_RE = re.compile(r"^(VEV_\d{4}|VELVETFRUIT_EXTRACT|HYDROGEL_PACK):\s*(-?[\d,]+)\s*$")
TOTAL_RE = re.compile(r"^Total profit:\s*(-?[\d,]+)\s*$")
DAY_RE = re.compile(r"^Round\s+(\d+)\s+day\s+(\d+):\s*(-?[\d,]+)\s*$")


def parse_number_list(raw: str, cast):
    if raw.strip() == "":
        return []
    return [cast(x.strip()) for x in raw.split(",") if x.strip()]


def parse_backtest_output(stdout: str) -> dict[str, int]:
    out: dict[str, int] = {}
    product_pnls = defaultdict(int)

    for raw_line in stdout.splitlines():
        line = raw_line.strip()

        m_prod = PRODUCT_RE.match(line)
        if m_prod:
            product = m_prod.group(1)
            pnl = int(m_prod.group(2).replace(",", ""))
            product_pnls[product] += pnl
            continue

        m_day = DAY_RE.match(line)
        if m_day:
            round_num = int(m_day.group(1))
            day_num = int(m_day.group(2))
            pnl = int(m_day.group(3).replace(",", ""))
            out[f"round{round_num}_day{day_num}_pnl"] = pnl
            continue

        m_total = TOTAL_RE.match(line)
        if m_total:
            out["total_pnl"] = int(m_total.group(1).replace(",", ""))

    for product in ATM_PRODUCTS:
        out[f"{product}_pnl"] = product_pnls.get(product, 0)

    out["atm_sum_pnl"] = sum(out.get(f"{p}_pnl", 0) for p in ATM_PRODUCTS)

    return out


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


def build_param_grid(args) -> list[dict[str, Any]]:
    lengths = parse_number_list(args.lengths, int)
    z_thresholds = parse_number_list(args.z_thresholds, float)

    out = []

    for length in lengths:
        for z in z_thresholds:
            out.append({
                "env": {
                    "HJ_ATM_THEO_DIFF_HISTORY_LENGTH": length,
                    "HJ_ATM_Z_SCORE_THRESHOLD": z,
                },
                "row": {
                    "atm_theo_diff_history_length": length,
                    "atm_z_score_threshold": z,
                },
            })

    return out


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

    parsed = parse_backtest_output(completed.stdout)

    if "total_pnl" not in parsed:
        raise RuntimeError(
            "Could not parse Total profit from output\n"
            f"params={params}\n"
            f"output:\n{completed.stdout}"
        )

    row = dict(params["row"])
    row.update(parsed)

    return row


def write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    if not path or not rows:
        return

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    preferred = [
        "atm_theo_diff_history_length",
        "atm_z_score_threshold",
        "VEV_5000_pnl",
        "VEV_5100_pnl",
        "VEV_5200_pnl",
        "VEV_5300_pnl",
        "atm_sum_pnl",
        "round3_day0_pnl",
        "round3_day1_pnl",
        "round3_day2_pnl",
        "total_pnl",
    ]

    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())

    fieldnames = [k for k in preferred if k in all_keys]
    fieldnames += sorted(k for k in all_keys if k not in fieldnames)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_analysis(rows: list[dict[str, Any]], top_n: int = 5) -> dict[str, Any]:
    by_window: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        by_window[int(row["atm_theo_diff_history_length"])].append(row)

    analysis: dict[str, Any] = {
        "windows": {},
        "best_combo_by_window": [],
    }

    for window in sorted(by_window):
        window_rows = by_window[window]
        window_info: dict[str, Any] = {
            "per_product_top": {},
            "best_combo": {
                "window": window,
                "thresholds": {},
                "product_pnls": {},
                "combined_atm_pnl": 0,
            },
        }

        for product in ATM_PRODUCTS:
            pnl_key = f"{product}_pnl"
            ranked = sorted(window_rows, key=lambda r: r.get(pnl_key, 0), reverse=True)
            top = ranked[:top_n]

            top_records = []
            for r in top:
                top_records.append({
                    "threshold": r["atm_z_score_threshold"],
                    "pnl": r.get(pnl_key, 0),
                    "total_pnl": r.get("total_pnl", 0),
                    "atm_sum_pnl": r.get("atm_sum_pnl", 0),
                })

            window_info["per_product_top"][product] = top_records

            best = ranked[0]
            best_threshold = best["atm_z_score_threshold"]
            best_pnl = best.get(pnl_key, 0)

            window_info["best_combo"]["thresholds"][product] = best_threshold
            window_info["best_combo"]["product_pnls"][product] = best_pnl
            window_info["best_combo"]["combined_atm_pnl"] += best_pnl

        analysis["windows"][str(window)] = window_info
        analysis["best_combo_by_window"].append(window_info["best_combo"])

    analysis["best_combo_by_window"] = sorted(
        analysis["best_combo_by_window"],
        key=lambda x: x["combined_atm_pnl"],
        reverse=True,
    )

    return analysis


def format_analysis_text(analysis: dict[str, Any], top_n: int = 5) -> str:
    lines: list[str] = []

    for window_str, window_info in analysis["windows"].items():
        lines.append("")
        lines.append("=" * 80)
        lines.append(f"WINDOW = {window_str}")
        lines.append("=" * 80)

        for product in ATM_PRODUCTS:
            lines.append("")
            lines.append(f"[{product}] top {top_n} z thresholds")
            for i, rec in enumerate(window_info["per_product_top"][product], start=1):
                lines.append(
                    f"  {i}. z={rec['threshold']} "
                    f"| {product}_pnl={rec['pnl']:,} "
                    f"| atm_sum={rec['atm_sum_pnl']:,} "
                    f"| total={rec['total_pnl']:,}"
                )

        combo = window_info["best_combo"]
        lines.append("")
        lines.append("[BEST COMBO FOR THIS WINDOW]")
        lines.append(f"  combined_atm_pnl={combo['combined_atm_pnl']:,}")
        lines.append(f"  thresholds={combo['thresholds']}")
        lines.append(f"  product_pnls={combo['product_pnls']}")

    lines.append("")
    lines.append("=" * 80)
    lines.append("BEST COMBO BY WINDOW")
    lines.append("=" * 80)

    for i, combo in enumerate(analysis["best_combo_by_window"], start=1):
        lines.append(
            f"{i}. window={combo['window']} "
            f"| combined_atm_pnl={combo['combined_atm_pnl']:,} "
            f"| thresholds={combo['thresholds']} "
            f"| product_pnls={combo['product_pnls']}"
        )

    return "\n".join(lines)


def print_run_done(done: int, total: int, row: dict[str, Any]) -> None:
    length = row["atm_theo_diff_history_length"]
    z = row["atm_z_score_threshold"]

    product_part = " | ".join(
        f"{p}={row.get(f'{p}_pnl', 0):,}"
        for p in ATM_PRODUCTS
    )

    print(
        f"[DONE {done}/{total}] "
        f"window={length}, z={z} "
        f"| atm_sum={row.get('atm_sum_pnl', 0):,} "
        f"| total={row.get('total_pnl', 0):,} "
        f"| {product_part}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--algo", required=True)
    parser.add_argument("--round", type=int, default=3)
    parser.add_argument("--data", default="./data_capsule")
    parser.add_argument("--python", default="py")
    parser.add_argument("--py-path", default="./imc-prosperity-3-backtester")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--top-n", type=int, default=5)

    parser.add_argument("--lengths", default="20,50,100,300,500")
    parser.add_argument("--z-thresholds", default="0,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0")

    parser.add_argument("--out-csv", default="./hanjae/round3/atm_voucher_zscore_sweep_detail.csv")
    parser.add_argument("--out-summary", default="./hanjae/round3/atm_voucher_zscore_sweep_summary.txt")
    parser.add_argument("--out-json", default="./hanjae/round3/atm_voucher_zscore_sweep_summary.json")

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
            print_run_done(done, total, row)

    cleanup_backtest_outputs(start_time)

    rows = sorted(
        rows,
        key=lambda r: (r["atm_theo_diff_history_length"], r["atm_z_score_threshold"]),
    )

    write_csv(args.out_csv, rows)

    analysis = build_analysis(rows, top_n=args.top_n)
    summary_text = format_analysis_text(analysis, top_n=args.top_n)

    print("\n" + summary_text)

    if args.out_summary:
        out_summary = Path(args.out_summary)
        out_summary.parent.mkdir(parents=True, exist_ok=True)
        out_summary.write_text(summary_text, encoding="utf-8")
        print(f"\n[SAVED SUMMARY] {out_summary}")

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[SAVED JSON] {out_json}")

    if args.out_csv:
        print(f"[SAVED CSV] {args.out_csv}")


if __name__ == "__main__":
    main()