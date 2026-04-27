from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

PRICE_FILE_PATTERN = "prices_round_*_day_*.csv"
TRADE_FILE_PATTERN = "trades_round_*_day_*.csv"

PRICE_COLUMNS = ["mid_price", "bid_price_1", "ask_price_1"]
BID_PRICE_COLUMNS = ["bid_price_1", "bid_price_2", "bid_price_3"]
ASK_PRICE_COLUMNS = ["ask_price_1", "ask_price_2", "ask_price_3"]
ALL_ORDER_BOOK_PRICE_COLUMNS = BID_PRICE_COLUMNS + ASK_PRICE_COLUMNS

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
        description=(
            "Create per-product participant buy/sell stats, order-book-side ratios, "
            "average marked PnL, and per-day marked PnL CSV files."
        )
    )
    parser.add_argument("--round", type=int, default=4, help="Round number. Default: 4")
    parser.add_argument("--round-dir", type=Path, help="Directory containing prices/trades CSV files")
    parser.add_argument("--prices", nargs="*", type=Path, help="Explicit prices CSV file paths")
    parser.add_argument("--trades", nargs="*", type=Path, help="Explicit trades CSV file paths")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "person_product_stats",
        help="Output directory for per-product CSV files.",
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

    # 빈 칸만 NaN으로 처리합니다. 0 가격은 VEV_6500 등에서 유효한 order book 가격일 수 있으므로 유지합니다.
    for col in set(PRICE_COLUMNS + ALL_ORDER_BOOK_PRICE_COLUMNS):
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
    trades["buyer"] = trades["buyer"].fillna("").astype(str)
    trades["seller"] = trades["seller"].fillna("").astype(str)
    trades["day"] = pd.to_numeric(trades["day"], errors="raise").astype(int)
    trades["timestamp"] = pd.to_numeric(trades["timestamp"], errors="raise").astype(int)
    trades["price"] = pd.to_numeric(trades["price"], errors="coerce").astype(float)
    trades["quantity"] = pd.to_numeric(trades["quantity"], errors="coerce").astype(float)
    return trades


def close_price(prices: pd.DataFrame, product: str, day: int) -> float | None:
    product_day = prices.loc[(prices["product"] == product) & (prices["day"] == day)].copy()
    if product_day.empty:
        return None

    exact_close = product_day.loc[product_day["timestamp"] == 999900]
    if not exact_close.empty:
        row = exact_close.sort_values("timestamp").iloc[-1]
    else:
        row = product_day.sort_values("timestamp").iloc[-1]

    for col in ["mid_price", "bid_price_1", "ask_price_1"]:
        value = row.get(col)
        if pd.notna(value):
            return float(value)
    return None


def marked_pnl_for_day(product_day_trades: pd.DataFrame, name: str, close: float) -> float:
    cash = 0.0
    position = 0.0

    for trade in product_day_trades.itertuples(index=False):
        qty = float(trade.quantity)
        price = float(trade.price)
        if str(trade.buyer) == name:
            position += qty
            cash -= price * qty
        if str(trade.seller) == name:
            position -= qty
            cash += price * qty

    return cash + position * close


def safe_mean(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return float(series.mean())


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


def compute_in_order_book_counts(product_trades: pd.DataFrame, lookup: dict[tuple[str, int, int], tuple[set[float], set[float]]]) -> dict[str, int]:
    """
    trade.price가 bid_price_1..3에 있으면 buyer가 order book 쪽입니다.
    trade.price가 ask_price_1..3에 있으면 seller가 order book 쪽입니다.
    둘 다 true이면 buyer와 seller 양쪽에 모두 +1 합니다.
    """
    counts: dict[str, int] = {}

    for trade in product_trades.itertuples(index=False):
        product = str(trade.symbol)
        bid_prices, ask_prices = lookup.get((product, int(trade.day), int(trade.timestamp)), (set(), set()))
        price = float(trade.price)

        if price_in_set(price, bid_prices):
            buyer = str(trade.buyer)
            if buyer:
                counts[buyer] = counts.get(buyer, 0) + 1

        if price_in_set(price, ask_prices):
            seller = str(trade.seller)
            if seller:
                counts[seller] = counts.get(seller, 0) + 1

    return counts


def main() -> None:
    args = parse_args()
    prices_paths, trades_paths = discover_files(args)
    prices = read_prices(prices_paths)
    trades = read_trades(trades_paths)
    trades = trades.loc[trades["symbol"].isin(PRODUCT_ORDER)].copy()

    lookup = build_order_book_lookup(prices)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    days = sorted(int(day) for day in prices["day"].unique())

    for product in PRODUCT_ORDER:
        product_trades = trades.loc[trades["symbol"] == product].copy()
        order_book_counts = compute_in_order_book_counts(product_trades, lookup)
        names = sorted((set(product_trades["buyer"]) | set(product_trades["seller"])) - {""})

        rows = []
        for name in names:
            buy_rows = product_trades.loc[product_trades["buyer"] == name]
            sell_rows = product_trades.loc[product_trades["seller"] == name]
            all_quantities = pd.concat([buy_rows["quantity"], sell_rows["quantity"]], ignore_index=True)

            day_pnls: dict[int, float] = {}
            for day in days:
                close = close_price(prices, product, day)
                if close is None:
                    continue
                product_day_trades = product_trades.loc[product_trades["day"] == day]
                day_pnls[day] = marked_pnl_for_day(product_day_trades, name, close)

            avg_pnl = sum(day_pnls.values()) / len(day_pnls) if day_pnls else 0.0

            buy_count = int(len(buy_rows))
            sell_count = int(len(sell_rows))
            total_count = buy_count + sell_count
            in_order_book_count = int(order_book_counts.get(name, 0))
            in_order_book_ratio = in_order_book_count / total_count if total_count > 0 else 0.0

            row = {
                "product": product,
                "name": name,
                "buy_count": buy_count,
                "buy_avg_quantity": safe_mean(buy_rows["quantity"]),
                "sell_count": sell_count,
                "sell_avg_quantity": safe_mean(sell_rows["quantity"]),
                "avg_quantity": safe_mean(all_quantities),
                "in_order_book_ratio": in_order_book_ratio,
                "pnl": avg_pnl,
                "_total_count": total_count,
            }
            for day in days:
                row[f"pnl_day_{day}"] = day_pnls.get(day, 0.0)
            rows.append(row)

        output_columns = [
            "product",
            "name",
            "buy_count",
            "buy_avg_quantity",
            "sell_count",
            "sell_avg_quantity",
            "avg_quantity",
            "in_order_book_ratio",
            "pnl",
            *[f"pnl_day_{day}" for day in days],
            "_total_count",
        ]
        out_df = pd.DataFrame(rows, columns=output_columns)

        if not out_df.empty:
            out_df = out_df.sort_values(["_total_count", "name"], ascending=[False, True], kind="stable")
            out_df = out_df.drop(columns=["_total_count"])
        else:
            out_df = out_df.drop(columns=["_total_count"], errors="ignore")

        output_path = output_dir / f"{product}_person_stats.csv"
        out_df.to_csv(output_path, sep=";", index=False)
        print(f"Saved {product} participant stats to: {output_path}")


if __name__ == "__main__":
    main()
