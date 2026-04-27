from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

PRICE_FILE_PATTERN = "prices_round_*_day_*.csv"
TRADE_FILE_PATTERN = "trades_round_*_day_*.csv"
BID_PRICE_COLUMNS = ["bid_price_1", "bid_price_2", "bid_price_3"]
ASK_PRICE_COLUMNS = ["ask_price_1", "ask_price_2", "ask_price_3"]

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
    "VEV_6500",
]

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
        description="Count whether each trade price existed on the buy/sell side of the order book."
    )
    parser.add_argument("--round", type=int, default=4, help="Round number. Default: 4")
    parser.add_argument("--round-dir", type=Path, help="Directory containing prices/trades CSV files")
    parser.add_argument("--prices", nargs="*", type=Path, help="Explicit prices CSV file paths")
    parser.add_argument("--trades", nargs="*", type=Path, help="Explicit trades CSV file paths")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "order_book_side_counts.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def extract_day_from_path(path: Path) -> int:
    match = DAY_RE.search(path.name)
    if not match:
        raise ValueError(f"Could not infer day from filename: {path}")
    return int(match.group(1))


def discover_files(args: argparse.Namespace) -> tuple[list[Path], list[Path]]:
    prices = [p.resolve() for p in (args.prices or [])]
    trades = [p.resolve() for p in (args.trades or [])]

    if prices or trades:
        if not prices or not trades:
            raise ValueError("--prices와 --trades는 같이 지정하세요.")
        return sorted(prices), sorted(trades)

    round_dir = args.round_dir.resolve() if args.round_dir else (DEFAULT_DATA_CAPSULE_DIR / f"round{args.round}").resolve()
    if not round_dir.exists():
        raise FileNotFoundError(f"데이터 디렉터리를 찾지 못했습니다: {round_dir}")

    prices = sorted(round_dir.glob(PRICE_FILE_PATTERN))
    trades = sorted(round_dir.glob(TRADE_FILE_PATTERN))
    if not prices:
        raise FileNotFoundError(f"prices csv를 찾지 못했습니다: {round_dir / PRICE_FILE_PATTERN}")
    if not trades:
        raise FileNotFoundError(f"trades csv를 찾지 못했습니다: {round_dir / TRADE_FILE_PATTERN}")
    return prices, trades


def read_prices(paths: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path, sep=";")
        if "day" not in df.columns:
            df["day"] = extract_day_from_path(path)
        frames.append(df)
    prices = pd.concat(frames, ignore_index=True)
    prices["product"] = prices["product"].astype(str)
    prices["day"] = pd.to_numeric(prices["day"], errors="raise").astype(int)
    prices["timestamp"] = pd.to_numeric(prices["timestamp"], errors="raise").astype(int)
    for col in BID_PRICE_COLUMNS + ASK_PRICE_COLUMNS:
        if col in prices.columns:
            prices[col] = pd.to_numeric(prices[col], errors="coerce")
    return prices


def read_trades(paths: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path, sep=";")
        df["day"] = extract_day_from_path(path)
        frames.append(df)
    trades = pd.concat(frames, ignore_index=True)
    trades["symbol"] = trades["symbol"].astype(str)
    trades["day"] = pd.to_numeric(trades["day"], errors="raise").astype(int)
    trades["timestamp"] = pd.to_numeric(trades["timestamp"], errors="raise").astype(int)
    trades["price"] = pd.to_numeric(trades["price"], errors="coerce").astype(float)
    return trades


def build_order_book_lookup(prices: pd.DataFrame) -> dict[tuple[str, int, int], tuple[set[float], set[float]]]:
    lookup: dict[tuple[str, int, int], tuple[set[float], set[float]]] = {}

    for row in prices.itertuples(index=False):
        product = str(getattr(row, "product"))
        day = int(getattr(row, "day"))
        timestamp = int(getattr(row, "timestamp"))
        bid_prices: set[float] = set()
        ask_prices: set[float] = set()

        for col in BID_PRICE_COLUMNS:
            if hasattr(row, col):
                value = getattr(row, col)
                if pd.notna(value):
                    bid_prices.add(float(value))
        for col in ASK_PRICE_COLUMNS:
            if hasattr(row, col):
                value = getattr(row, col)
                if pd.notna(value):
                    ask_prices.add(float(value))

        lookup[(product, day, timestamp)] = (bid_prices, ask_prices)
    return lookup


def price_in_set(price: float, values: set[float]) -> bool:
    return any(math.isclose(price, value, rel_tol=0.0, abs_tol=1e-9) for value in values)


def main() -> None:
    args = parse_args()
    prices_paths, trades_paths = discover_files(args)
    prices = read_prices(prices_paths)
    trades = read_trades(trades_paths)
    trades = trades.loc[trades["symbol"].isin(PRODUCT_ORDER)].copy()

    lookup = build_order_book_lookup(prices)
    rows = []

    for product in PRODUCT_ORDER:
        product_trades = trades.loc[trades["symbol"] == product]
        buy_count = 0
        sell_count = 0
        none_count = 0

        for trade in product_trades.itertuples(index=False):
            bid_prices, ask_prices = lookup.get((product, int(trade.day), int(trade.timestamp)), (set(), set()))
            in_buy = price_in_set(float(trade.price), bid_prices)
            in_sell = price_in_set(float(trade.price), ask_prices)

            if in_buy:
                buy_count += 1
            elif in_sell:
                sell_count += 1
            else:
                none_count += 1

        rows.append(
            {
                "product": product,
                "buy_in_order_book_count": buy_count,
                "sell_in_order_book_count": sell_count,
                "none_in_order_book_count": none_count,
            }
        )

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, sep=";", index=False)
    print(f"Saved order book side counts to: {output_path}")


if __name__ == "__main__":
    main()
