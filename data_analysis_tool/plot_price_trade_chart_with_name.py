from __future__ import annotations

import argparse
import json
import math
import re
from html import escape
from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

DEFAULT_FILL_PRICE = 0.0
PRICE_COLUMNS = ["mid_price", "bid_price_1", "ask_price_1"]
BID_PRICE_COLUMNS = ["bid_price_1", "bid_price_2", "bid_price_3"]
ASK_PRICE_COLUMNS = ["ask_price_1", "ask_price_2", "ask_price_3"]
ALL_BOOK_PRICE_COLUMNS = BID_PRICE_COLUMNS + ASK_PRICE_COLUMNS
PRICE_FILE_PATTERN = "prices_round_*_day_*.csv"
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
    "VEV_6500",
]
PRODUCT_RANK = {product: i for i, product in enumerate(PRODUCT_ORDER)}

DAY_RE = re.compile(r"day_(-?\d+)")
ROUND_RE = re.compile(r"round_(\d+)")


def find_project_root(start: Path) -> Path:
    start = start.resolve()
    for candidate in [start.parent, *start.parents]:
        if (candidate / "data_capsule").exists():
            return candidate
    # Expected local placement: <project>/data_analysis_tool/<script>.py
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
            "Plot mid/bid1/ask1 lines and trade points for Prosperity CSV data. "
            "The output is an interactive HTML file with buyer/seller trade filtering."
        )
    )
    parser.add_argument("--round", type=int, default=4, help="Round number. Default: 4")
    parser.add_argument("--round-dir", type=Path, help="Directory containing prices/trades CSV files")
    parser.add_argument("--prices", nargs="*", type=Path, help="Explicit prices CSV file paths")
    parser.add_argument("--trades", nargs="*", type=Path, help="Explicit trades CSV file paths")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output HTML path. Default: data_analysis_tool/outputs/<name>_market_charts.html",
    )
    parser.add_argument(
        "--default-fill-price",
        type=float,
        default=DEFAULT_FILL_PRICE,
        help=f"Fallback price used before any valid value appears. Default: {DEFAULT_FILL_PRICE}",
    )
    return parser.parse_args()


def extract_day_from_path(path: Path) -> int:
    match = DAY_RE.search(path.name)
    if not match:
        raise ValueError(f"Could not infer day from filename: {path}")
    return int(match.group(1))


def extract_round_from_path(path: Path) -> int | None:
    match = ROUND_RE.search(path.name)
    if not match:
        return None
    return int(match.group(1))


def discover_files(args: argparse.Namespace) -> tuple[list[Path], list[Path], str]:
    prices = [p.resolve() for p in (args.prices or [])]
    trades = [p.resolve() for p in (args.trades or [])]

    label = "custom"

    if prices or trades:
        if not prices:
            raise ValueError("--trades를 직접 주는 경우, --prices도 같이 주세요.")
        if not trades:
            raise ValueError("--prices를 직접 주는 경우, --trades도 같이 주세요.")
        first_round = extract_round_from_path(prices[0])
        if first_round is not None:
            label = f"round{first_round}"
        return sorted(prices), sorted(trades), label

    if args.round_dir:
        round_dir = args.round_dir.resolve()
        label = round_dir.name
    else:
        round_dir = (DEFAULT_DATA_CAPSULE_DIR / f"round{args.round}").resolve()
        label = f"round{args.round}"

    if not round_dir.exists():
        raise FileNotFoundError(f"데이터 디렉터리를 찾지 못했습니다: {round_dir}")

    prices = sorted(round_dir.glob(PRICE_FILE_PATTERN))
    trades = sorted(round_dir.glob(TRADE_FILE_PATTERN))

    if not prices:
        raise FileNotFoundError(f"prices csv를 찾지 못했습니다: {round_dir / PRICE_FILE_PATTERN}")
    if not trades:
        raise FileNotFoundError(f"trades csv를 찾지 못했습니다: {round_dir / TRADE_FILE_PATTERN}")

    return prices, trades, label


def read_prices(paths: Iterable[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        df = pd.read_csv(path, sep=";")
        if "day" not in df.columns:
            df["day"] = extract_day_from_path(path)
        df["source_file"] = path.name
        frames.append(df)

    prices = pd.concat(frames, ignore_index=True)

    prices["product"] = prices["product"].astype(str)
    prices["day"] = pd.to_numeric(prices["day"], errors="raise").astype(int)
    prices["timestamp"] = pd.to_numeric(prices["timestamp"], errors="raise").astype(int)

    for col in set(PRICE_COLUMNS + ALL_BOOK_PRICE_COLUMNS):
        if col in prices.columns:
            prices[col] = pd.to_numeric(prices[col], errors="coerce")
            prices.loc[prices[col] == 0.0, col] = pd.NA

    prices = prices.sort_values(
        ["product", "day", "timestamp"], kind="stable"
    ).reset_index(drop=True)
    return prices


def forward_fill_prices(prices: pd.DataFrame, default_fill_price: float) -> pd.DataFrame:
    filled = prices.copy()
    for col in PRICE_COLUMNS:
        filled[col] = (
            filled.groupby("product", sort=False)[col]
            .transform(lambda s: s.ffill().fillna(default_fill_price))
            .astype(float)
        )
    return filled


def read_trades(paths: Iterable[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        df = pd.read_csv(path, sep=";")
        df["day"] = extract_day_from_path(path)
        df["source_file"] = path.name
        frames.append(df)

    trades = pd.concat(frames, ignore_index=True)

    trades["symbol"] = trades["symbol"].astype(str)
    trades["buyer"] = trades["buyer"].fillna("").astype(str)
    trades["seller"] = trades["seller"].fillna("").astype(str)
    trades["day"] = pd.to_numeric(trades["day"], errors="raise").astype(int)
    trades["timestamp"] = pd.to_numeric(trades["timestamp"], errors="raise").astype(int)
    trades["price"] = pd.to_numeric(trades["price"], errors="coerce").astype(float)
    trades["quantity"] = pd.to_numeric(trades["quantity"], errors="coerce").astype(float)

    trades = trades.sort_values(
        ["symbol", "day", "timestamp", "price"], kind="stable"
    ).reset_index(drop=True)
    return trades


def product_sort_key(product: str) -> tuple[int, str]:
    return PRODUCT_RANK.get(product, len(PRODUCT_ORDER)), product


def participant_counts(trades: pd.DataFrame, column: str) -> list[tuple[str, int]]:
    counts = (
        trades.loc[trades[column].astype(str).str.len() > 0, column]
        .value_counts()
        .to_dict()
    )
    return sorted(counts.items(), key=lambda x: (-x[1], x[0]))


def build_order_book_lookup(prices: pd.DataFrame) -> dict[tuple[str, int, int], tuple[set[float], set[float]]]:
    lookup: dict[tuple[str, int, int], tuple[set[float], set[float]]] = {}

    for row in prices.itertuples(index=False):
        product = getattr(row, "product")
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


def annotate_trades_with_order_book_side(trades: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    annotated = trades.copy()
    lookup = build_order_book_lookup(prices)

    buyer_display: list[str] = []
    seller_display: list[str] = []
    order_book_side: list[str] = []

    for row in annotated.itertuples(index=False):
        key = (str(row.symbol), int(row.day), int(row.timestamp))
        bid_prices, ask_prices = lookup.get(key, (set(), set()))
        price = float(row.price)

        buyer_in_book = price_in_set(price, bid_prices)
        seller_in_book = price_in_set(price, ask_prices)

        buyer_name = str(row.buyer)
        seller_name = str(row.seller)

        if buyer_in_book and buyer_name:
            buyer_name = f"{buyer_name} (order book)"
        if seller_in_book and seller_name:
            seller_name = f"{seller_name} (order book)"

        if buyer_in_book and seller_in_book:
            side = "buy,sell"
        elif buyer_in_book:
            side = "buy"
        elif seller_in_book:
            side = "sell"
        else:
            side = "none"

        buyer_display.append(buyer_name)
        seller_display.append(seller_name)
        order_book_side.append(side)

    annotated["buyer_display"] = buyer_display
    annotated["seller_display"] = seller_display
    annotated["order_book_side"] = order_book_side
    return annotated


def build_product_day_figure(
    product: str,
    day: int,
    product_day_prices: pd.DataFrame,
    product_day_trades: pd.DataFrame,
) -> tuple[go.Figure, dict[str, list]]:
    fig = go.Figure()

    for col, trace_name in [
        ("mid_price", "mid price"),
        ("bid_price_1", "bid 1 price"),
        ("ask_price_1", "ask 1 price"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=product_day_prices["timestamp"],
                y=product_day_prices[col],
                mode="lines",
                name=trace_name,
                customdata=product_day_prices[["timestamp"]].to_numpy(),
                hovertemplate=(
                    f"{trace_name}<br>"
                    f"day={day}<br>"
                    "timestamp=%{customdata[0]}<br>"
                    "price=%{y}<extra></extra>"
                ),
            )
        )

    trade_payload = {
        "x": product_day_trades["timestamp"].tolist(),
        "y": product_day_trades["price"].tolist(),
        "quantity": product_day_trades["quantity"].tolist(),
        "buyer": product_day_trades["buyer"].tolist(),
        "seller": product_day_trades["seller"].tolist(),
        "customdata": product_day_trades[
            ["timestamp", "quantity", "buyer_display", "seller_display", "order_book_side"]
        ].to_numpy().tolist(),
    }

    fig.add_trace(
        go.Scatter(
            x=trade_payload["x"],
            y=trade_payload["y"],
            mode="markers",
            name="trades",
            customdata=trade_payload["customdata"],
            hovertemplate=(
                "trade<br>"
                f"day={day}<br>"
                "timestamp=%{customdata[0]}<br>"
                "price=%{y}<br>"
                "qty=%{customdata[1]}<br>"
                "buyer=%{customdata[2]}<br>"
                "seller=%{customdata[3]}<br>"
                "order_book_side=%{customdata[4]}<extra></extra>"
            ),
            marker={"size": 7, "opacity": 0.8, "symbol": "circle"},
        )
    )

    fig.update_layout(
        title=f"{product} - day {day}",
        xaxis_title="timestamp",
        yaxis_title="price",
        hovermode="closest",
        legend={"itemclick": "toggle", "itemdoubleclick": "toggleothers"},
        dragmode="zoom",
        template="plotly_white",
        margin={"l": 60, "r": 30, "t": 60, "b": 60},
        height=650,
    )

    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor")
    fig.update_yaxes(showspikes=True, spikemode="across", spikesnap="cursor")

    return fig, trade_payload


def make_output_path(label: str, explicit_output: Path | None) -> Path:
    if explicit_output is not None:
        return explicit_output.resolve()
    return (DEFAULT_OUTPUT_DIR / f"{label}_market_charts.html").resolve()


def checkbox_html(kind: str, items: list[tuple[str, int]]) -> str:
    rows = []
    for name, count in items:
        safe_name = escape(name, quote=True)
        rows.append(
            "<label class='check-row'>"
            f"<input class='{kind}-filter participant-filter' type='checkbox' data-name='{safe_name}' checked>"
            f"<span class='participant-name'>{safe_name}</span>"
            f"<span class='participant-count'>{count}</span>"
            "</label>"
        )
    return "\n".join(rows)


def build_html_document(
    figures: list[dict[str, object]],
    title: str,
    buyer_counts: list[tuple[str, int]],
    seller_counts: list[tuple[str, int]],
) -> str:
    chunks = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "  <meta charset='utf-8'>",
        f"  <title>{escape(title)}</title>",
        "  <style>",
        "    body { font-family: Arial, sans-serif; margin: 24px; margin-right: 390px; }",
        "    h1 { margin-bottom: 8px; }",
        "    .chart-block { margin: 28px 0 48px; }",
        "    .sidebar { position: fixed; top: 20px; right: 20px; width: 320px; max-height: calc(100vh - 40px); overflow-y: auto; background: #fff; border: 1px solid #d9d9d9; border-radius: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.08); padding: 16px; z-index: 1000; }",
        "    .sidebar h2 { margin: 0 0 12px; font-size: 18px; }",
        "    .sidebar h3 { margin: 18px 0 8px; font-size: 15px; border-top: 1px solid #e5e7eb; padding-top: 14px; }",
        "    .check-row { display: flex; align-items: center; gap: 8px; font-size: 13px; margin: 6px 0; color: #333; }",
        "    .check-row input { width: auto; }",
        "    .participant-name { flex: 1; }",
        "    .participant-count { color: #6b7280; font-size: 12px; }",
        "    .sidebar button { width: 48%; margin: 4px 1%; padding: 7px 8px; border: 1px solid #d1d5db; border-radius: 8px; background: #f3f4f6; color: #111827; cursor: pointer; font-size: 12px; }",
        "    .button-row { display: flex; justify-content: space-between; gap: 4px; margin-bottom: 6px; }",
        "    .hint { margin-top: 10px; font-size: 12px; line-height: 1.4; color: #555; }",
        "    .status { margin-top: 8px; font-size: 12px; color: #1f2937; }",
        "  </style>",
        "</head>",
        "<body>",
        f"  <h1>{escape(title)}</h1>",
        "  <p>Legend click: toggle traces. Drag: zoom. Double click: reset. Buyer/Seller filters apply to all trade points.</p>",
        "  <div class='sidebar'>",
        "    <h2>Chart filters</h2>",
        "    <label class='check-row'><input class='trace-filter' type='checkbox' data-trace-index='0' checked><span>mid price</span></label>",
        "    <label class='check-row'><input class='trace-filter' type='checkbox' data-trace-index='1' checked><span>bid 1 price</span></label>",
        "    <label class='check-row'><input class='trace-filter' type='checkbox' data-trace-index='2' checked><span>ask 1 price</span></label>",
        "    <label class='check-row'><input class='trace-filter' type='checkbox' data-trace-index='3' checked><span>trades</span></label>",
        "    <h3>BUYER</h3>",
        "    <div class='button-row'><button type='button' data-target='buyer' data-action='select'>Select all</button><button type='button' data-target='buyer' data-action='clear'>Clear</button></div>",
        checkbox_html("buyer", buyer_counts),
        "    <h3>SELLER</h3>",
        "    <div class='button-row'><button type='button' data-target='seller' data-action='select'>Select all</button><button type='button' data-target='seller' data-action='clear'>Clear</button></div>",
        checkbox_html("seller", seller_counts),
        "    <div class='status' id='tradeFilterStatus'>Showing all trade points.</div>",
        "    <div class='hint'>A trade is shown only when both its buyer and seller are selected.</div>",
        "  </div>",
    ]

    figure_meta: list[dict[str, object]] = []
    first = True
    for index, item in enumerate(figures):
        product = str(item["product"])
        day = int(item["day"])
        fig = item["figure"]
        div_id = f"chart_{index}"
        html = pio.to_html(fig, include_plotlyjs="cdn" if first else False, full_html=False, div_id=div_id)
        first = False
        chunks.append(f"  <div class='chart-block'><h2>{escape(product)} - day {day}</h2>{html}</div>")
        figure_meta.append(
            {
                "div_id": div_id,
                "trade_trace_index": 3,
                "trade_payload": item["trade_payload"],
                "product": product,
                "day": day,
            }
        )

    chunks.append("  <script>")
    chunks.append(f"    const FIGURE_META = {json.dumps(figure_meta)};")
    chunks.append(
        r"""
    function getSelectedNames(selector) {
      const selected = new Set();
      document.querySelectorAll(selector).forEach((el) => {
        if (el.checked) selected.add(el.dataset.name);
      });
      return selected;
    }

    function filterPayload(payload, selectedBuyers, selectedSellers) {
      const filtered = { x: [], y: [], customdata: [] };
      for (let i = 0; i < payload.x.length; i++) {
        const buyer = payload.buyer[i];
        const seller = payload.seller[i];
        if (selectedBuyers.has(buyer) && selectedSellers.has(seller)) {
          filtered.x.push(payload.x[i]);
          filtered.y.push(payload.y[i]);
          filtered.customdata.push(payload.customdata[i]);
        }
      }
      return filtered;
    }

    function applyParticipantFilter() {
      const selectedBuyers = getSelectedNames('.buyer-filter');
      const selectedSellers = getSelectedNames('.seller-filter');
      let shown = 0;
      let total = 0;

      FIGURE_META.forEach((meta) => {
        const gd = document.getElementById(meta.div_id);
        if (!gd) return;
        const filtered = filterPayload(meta.trade_payload, selectedBuyers, selectedSellers);
        shown += filtered.x.length;
        total += meta.trade_payload.x.length;
        Plotly.restyle(
          gd,
          {
            x: [filtered.x],
            y: [filtered.y],
            customdata: [filtered.customdata],
          },
          [meta.trade_trace_index]
        );
      });

      document.getElementById('tradeFilterStatus').textContent = `Showing ${shown} / ${total} trade points.`;
    }

    function applyTraceVisibility(traceIndex, visible) {
      FIGURE_META.forEach((meta) => {
        const gd = document.getElementById(meta.div_id);
        if (!gd) return;
        Plotly.restyle(gd, { visible: visible ? true : 'legendonly' }, [traceIndex]);
      });
    }

    document.querySelectorAll('.participant-filter').forEach((el) => {
      el.addEventListener('change', applyParticipantFilter);
    });

    document.querySelectorAll('.trace-filter').forEach((el) => {
      el.addEventListener('change', () => {
        applyTraceVisibility(Number(el.dataset.traceIndex), el.checked);
      });
    });

    document.querySelectorAll('button[data-target]').forEach((button) => {
      button.addEventListener('click', () => {
        const target = button.dataset.target;
        const checked = button.dataset.action === 'select';
        document.querySelectorAll(`.${target}-filter`).forEach((el) => {
          el.checked = checked;
        });
        applyParticipantFilter();
      });
    });
    """
    )
    chunks.append("  </script>")
    chunks.extend(["</body>", "</html>"])
    return "\n".join(chunks)


def main() -> None:
    args = parse_args()

    prices_paths, trades_paths, label = discover_files(args)
    raw_prices = read_prices(prices_paths)
    filled_prices = forward_fill_prices(raw_prices, default_fill_price=args.default_fill_price)
    trades = read_trades(trades_paths)
    trades = trades.loc[trades["symbol"].isin(PRODUCT_ORDER)].copy()
    trades = annotate_trades_with_order_book_side(trades, raw_prices)

    figures: list[dict[str, object]] = []
    products = [p for p in PRODUCT_ORDER if p in set(filled_prices["product"]) or p in set(trades["symbol"])]

    for product in products:
        product_prices = filled_prices.loc[filled_prices["product"] == product].sort_values(["day", "timestamp"])
        product_trades = trades.loc[trades["symbol"] == product].sort_values(["day", "timestamp", "price"])

        days = sorted(product_prices["day"].unique())
        for day in days:
            product_day_prices = product_prices.loc[product_prices["day"] == day].sort_values("timestamp")
            product_day_trades = product_trades.loc[product_trades["day"] == day].sort_values(["timestamp", "price"])
            fig, trade_payload = build_product_day_figure(product, day, product_day_prices, product_day_trades)
            figures.append(
                {
                    "product": product,
                    "day": day,
                    "figure": fig,
                    "trade_payload": trade_payload,
                }
            )

    output_path = make_output_path(label, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    html = build_html_document(
        figures,
        title=f"Prosperity market charts - {label}",
        buyer_counts=participant_counts(trades, "buyer"),
        seller_counts=participant_counts(trades, "seller"),
    )
    output_path.write_text(html, encoding="utf-8")

    print(f"Saved interactive chart to: {output_path}")


if __name__ == "__main__":
    main()
