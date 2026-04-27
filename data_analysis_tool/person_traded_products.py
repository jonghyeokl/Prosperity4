from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd

TRADE_FILE_PATTERN = "trades_round_*_day_*.csv"

PRODUCT_ORDER = [
    "HYDROGEL_PACK",
    "VELVETFRUIT_EXTRACT",
    "VEV_4000",
    "VEV_4500",
    "VEV_5000",
    "VEV_5100",
    "VEV_5200",
    "VEV_5300",
    "VEV_5400",
    "VEV_5500",
    "VEV_6000",
]
PRODUCT_RANK = {product: i for i, product in enumerate(PRODUCT_ORDER)}

DAY_RE = re.compile(r"day_(-?\d+)")


def find_project_root(start: Path) -> Path:
    start = start.resolve()
    for candidate in [start.parent, *start.parents]:
        if (candidate / "data_capsule").exists():
            return candidate
    if len(start.parents) >= 2:
        return start.parents[1]
    return start.parent


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = find_project_root(THIS_FILE)
DEFAULT_DATA_CAPSULE_DIR = PROJECT_ROOT / "data_capsule"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data_analysis_tool" / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a text file listing products traded by each participant."
    )
    parser.add_argument("--round", type=int, default=4, help="Round number. Default: 4")
    parser.add_argument("--round-dir", type=Path, help="Directory containing trades CSV files")
    parser.add_argument("--trades", nargs="*", type=Path, help="Explicit trades CSV file paths")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "person_traded_products.txt",
        help="Output text path.",
    )
    parser.add_argument(
        "--unique",
        action="store_true",
        help="If set, output each product only once per person. Default repeats products by trade count.",
    )
    return parser.parse_args()


def extract_day_from_path(path: Path) -> int:
    match = DAY_RE.search(path.name)
    if not match:
        raise ValueError(f"Could not infer day from filename: {path}")
    return int(match.group(1))


def discover_trade_files(args: argparse.Namespace) -> list[Path]:
    trades = [p.resolve() for p in (args.trades or [])]
    if trades:
        return sorted(trades)

    round_dir = args.round_dir.resolve() if args.round_dir else (DEFAULT_DATA_CAPSULE_DIR / f"round{args.round}").resolve()
    if not round_dir.exists():
        raise FileNotFoundError(f"데이터 디렉터리를 찾지 못했습니다: {round_dir}")

    trades = sorted(round_dir.glob(TRADE_FILE_PATTERN))
    if not trades:
        raise FileNotFoundError(f"trades csv를 찾지 못했습니다: {round_dir / TRADE_FILE_PATTERN}")
    return trades


def read_trades(paths: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path, sep=";")
        df["day"] = extract_day_from_path(path)
        frames.append(df)
    trades = pd.concat(frames, ignore_index=True)
    trades["symbol"] = trades["symbol"].astype(str)
    trades["buyer"] = trades["buyer"].fillna("").astype(str)
    trades["seller"] = trades["seller"].fillna("").astype(str)
    trades["day"] = pd.to_numeric(trades["day"], errors="raise").astype(int)
    trades["timestamp"] = pd.to_numeric(trades["timestamp"], errors="raise").astype(int)
    return trades.sort_values(["day", "timestamp", "symbol"], kind="stable")


def product_sort_key(product: str) -> tuple[int, str]:
    return PRODUCT_RANK.get(product, len(PRODUCT_ORDER)), product


def main() -> None:
    args = parse_args()
    trades_paths = discover_trade_files(args)
    trades = read_trades(trades_paths)
    trades = trades.loc[trades["symbol"].isin(PRODUCT_ORDER)].copy()

    person_product_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for trade in trades.itertuples(index=False):
        product = str(trade.symbol)
        buyer = str(trade.buyer)
        seller = str(trade.seller)
        if buyer:
            person_product_counts[buyer][product] += 1
        if seller:
            person_product_counts[seller][product] += 1

    people = sorted(
        person_product_counts.keys(),
        key=lambda name: (-sum(person_product_counts[name].values()), name),
    )

    lines: list[str] = []
    for person in people:
        counts = person_product_counts[person]
        products: list[str] = []
        for product in sorted(counts.keys(), key=product_sort_key):
            repeat = 1 if args.unique else counts[product]
            products.extend([product] * repeat)

        lines.append(person)
        lines.append(", ".join(products))

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved person traded products to: {output_path}")


if __name__ == "__main__":
    main()
