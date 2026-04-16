from __future__ import annotations

import argparse
import json
import re
from html import escape
from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

DEFAULT_FILL_PRICE = 10000.0
PRICE_COLUMNS = ["mid_price", "bid_price_1", "ask_price_1"]
BID_VOLUME_COLUMNS = ["bid_volume_1", "bid_volume_2", "bid_volume_3"]
ASK_VOLUME_COLUMNS = ["ask_volume_1", "ask_volume_2", "ask_volume_3"]
PRICE_FILE_PATTERN = "prices_round_*_day_*.csv"
TRADE_FILE_PATTERN = "trades_round_*_day_*.csv"
IMBALANCE_THRESHOLD = 10
GREEN_BG = "rgba(144, 238, 144, 0.18)"
PINK_BG = "rgba(255, 182, 193, 0.18)"


DAY_RE = re.compile(r"day_(-?\d+)")
ROUND_RE = re.compile(r"round_(\d+)")


def find_project_root(start: Path) -> Path:
    start = start.resolve()
    for candidate in [start.parent, *start.parents]:
        if (candidate / "data_capsule").exists() and (candidate / "jonghyeok").exists():
            return candidate
    return start.parents[2]


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = find_project_root(THIS_FILE)
DEFAULT_DATA_CAPSULE_DIR = PROJECT_ROOT / "data_capsule"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "jonghyeok" / "data_analysis" / "output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot interactive Prosperity market charts from prices/trades CSV files. "
            "Creates one chart per (product, day) with mid/bid1/ask1 lines, trade markers, "
            "trade-volume filtering, and order-imbalance background highlighting."
        )
    )
    parser.add_argument("--round", type=int, help="Round number, e.g. 1 -> data_capsule/round1")
    parser.add_argument("--round-dir", type=Path, help="Directory containing prices/trades CSV files")
    parser.add_argument("--prices", nargs="*", type=Path, help="Explicit prices CSV file paths")
    parser.add_argument("--trades", nargs="*", type=Path, help="Explicit trades CSV file paths")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output HTML path. Default: jonghyeok/data_analysis/output/<name>_market_charts.html",
    )
    parser.add_argument(
        "--default-fill-price",
        type=float,
        default=DEFAULT_FILL_PRICE,
        help=f"Fallback price used before any valid value appears. Default: {DEFAULT_FILL_PRICE}",
    )
    parser.add_argument(
        "--imbalance-threshold",
        type=float,
        default=IMBALANCE_THRESHOLD,
        help=f"Threshold for total bid volume - total ask volume shading. Default: {IMBALANCE_THRESHOLD}",
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
            raise ValueError("--prices를 직접 주는 경우 --trades도 같이 주세요.")
        if not trades:
            raise ValueError("--trades를 직접 주는 경우 --prices도 같이 주세요.")
        first_round = extract_round_from_path(prices[0])
        if first_round is not None:
            label = f"round{first_round}"
        return sorted(prices), sorted(trades), label

    if args.round_dir:
        round_dir = args.round_dir.resolve()
        label = round_dir.name
    elif args.round is not None:
        round_dir = (DEFAULT_DATA_CAPSULE_DIR / f"round{args.round}").resolve()
        label = f"round{args.round}"
    else:
        raise ValueError("--round, --round-dir, 또는 --prices/--trades 중 하나는 지정해야 합니다.")

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

    for col in PRICE_COLUMNS:
        prices[col] = pd.to_numeric(prices[col], errors="coerce")
        prices.loc[prices[col] == 0.0, col] = pd.NA

    for col in BID_VOLUME_COLUMNS + ASK_VOLUME_COLUMNS:
        if col not in prices.columns:
            prices[col] = 0.0
        prices[col] = pd.to_numeric(prices[col], errors="coerce").fillna(0.0).astype(float)

    prices = prices.sort_values(["product", "day", "timestamp"], kind="stable").reset_index(drop=True)
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


def add_imbalance_columns(prices: pd.DataFrame) -> pd.DataFrame:
    enriched = prices.copy()
    enriched["total_bid_volume"] = enriched[BID_VOLUME_COLUMNS].sum(axis=1)
    enriched["total_ask_volume"] = enriched[ASK_VOLUME_COLUMNS].sum(axis=1)
    enriched["volume_imbalance"] = enriched["total_bid_volume"] - enriched["total_ask_volume"]
    return enriched


def read_trades(paths: Iterable[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        df = pd.read_csv(path, sep=";")
        df["day"] = extract_day_from_path(path)
        df["source_file"] = path.name
        frames.append(df)

    trades = pd.concat(frames, ignore_index=True)

    trades["symbol"] = trades["symbol"].astype(str)
    trades["day"] = pd.to_numeric(trades["day"], errors="raise").astype(int)
    trades["timestamp"] = pd.to_numeric(trades["timestamp"], errors="raise").astype(int)
    trades["price"] = pd.to_numeric(trades["price"], errors="coerce").astype(float)
    trades["quantity"] = pd.to_numeric(trades["quantity"], errors="coerce").astype(float)

    trades = trades.sort_values(["symbol", "day", "timestamp", "price"], kind="stable").reset_index(drop=True)
    return trades


def infer_segment_width(timestamps: pd.Series) -> float:
    ts = timestamps.drop_duplicates().sort_values().to_list()
    if len(ts) <= 1:
        return 100.0
    diffs = [float(ts[i + 1] - ts[i]) for i in range(len(ts) - 1) if ts[i + 1] > ts[i]]
    if not diffs:
        return 100.0
    return float(pd.Series(diffs).median())


def build_imbalance_shapes(product_day_prices: pd.DataFrame, threshold: float) -> list[dict]:
    if product_day_prices.empty:
        return []

    df = product_day_prices[["timestamp", "total_bid_volume", "total_ask_volume", "volume_imbalance"]].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    def regime(imbalance: float) -> int:
        if imbalance >= threshold:
            return 1
        if imbalance <= -threshold:
            return -1
        return 0

    df["regime"] = df["volume_imbalance"].map(regime)
    width = infer_segment_width(df["timestamp"])
    timestamps = df["timestamp"].to_list()

    x_ends: list[float] = []
    for i, ts in enumerate(timestamps):
        if i + 1 < len(timestamps):
            x_ends.append(float(timestamps[i + 1]))
        else:
            x_ends.append(float(ts + width))
    df["x_end"] = x_ends

    shapes: list[dict] = []
    start_idx = 0
    while start_idx < len(df):
        current_regime = int(df.loc[start_idx, "regime"])
        end_idx = start_idx
        while end_idx + 1 < len(df) and int(df.loc[end_idx + 1, "regime"]) == current_regime:
            end_idx += 1

        if current_regime != 0:
            color = GREEN_BG if current_regime == 1 else PINK_BG
            start_ts = float(df.loc[start_idx, "timestamp"])
            end_ts = float(df.loc[end_idx, "x_end"])
            shapes.append(
                {
                    "type": "rect",
                    "xref": "x",
                    "yref": "paper",
                    "x0": start_ts,
                    "x1": end_ts,
                    "y0": 0,
                    "y1": 1,
                    "fillcolor": color,
                    "line": {"width": 0},
                    "layer": "below",
                }
            )

        start_idx = end_idx + 1

    return shapes


def build_product_day_figure(
    product: str,
    day: int,
    product_day_prices: pd.DataFrame,
    product_day_trades: pd.DataFrame,
    imbalance_threshold: float,
) -> tuple[go.Figure, dict[str, list], list[dict]]:
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
                customdata=product_day_prices[["timestamp", "total_bid_volume", "total_ask_volume", "volume_imbalance"]].to_numpy(),
                hovertemplate=(
                    f"{trace_name}<br>"
                    f"day={day}<br>"
                    "timestamp=%{customdata[0]}<br>"
                    "price=%{y}<br>"
                    "total bid vol=%{customdata[1]}<br>"
                    "total ask vol=%{customdata[2]}<br>"
                    "bid-ask=%{customdata[3]}<extra></extra>"
                ),
            )
        )

    trade_payload = {
        "x": product_day_trades["timestamp"].tolist(),
        "y": product_day_trades["price"].tolist(),
        "quantity": product_day_trades["quantity"].tolist(),
        "customdata": product_day_trades[["timestamp", "quantity"]].to_numpy().tolist(),
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
                "qty=%{customdata[1]}<extra></extra>"
            ),
            marker={"size": 7, "opacity": 0.8, "symbol": "circle"},
        )
    )

    imbalance_shapes = build_imbalance_shapes(product_day_prices, threshold=imbalance_threshold)

    fig.update_layout(
        title=f"{product} - day {day}",
        xaxis_title="timestamp",
        yaxis_title="price",
        hovermode="x unified",
        legend={"itemclick": "toggle", "itemdoubleclick": "toggleothers"},
        dragmode="zoom",
        template="plotly_white",
        margin={"l": 60, "r": 30, "t": 60, "b": 60},
        height=650,
        shapes=imbalance_shapes,
    )

    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor")
    fig.update_yaxes(showspikes=True, spikemode="across", spikesnap="cursor")

    return fig, trade_payload, imbalance_shapes


def make_output_path(label: str, explicit_output: Path | None) -> Path:
    if explicit_output is not None:
        return explicit_output.resolve()
    return (DEFAULT_OUTPUT_DIR / f"{label}_market_charts.html").resolve()


def build_html_document(figures: list[dict[str, object]], title: str, imbalance_threshold: float) -> str:
    chunks = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "  <meta charset='utf-8'>",
        f"  <title>{escape(title)}</title>",
        "  <style>",
        "    body { font-family: Arial, sans-serif; margin: 24px; margin-right: 340px; }",
        "    h1 { margin-bottom: 8px; }",
        "    .chart-block { margin: 28px 0 48px; }",
        "    .sidebar { position: fixed; top: 20px; right: 20px; width: 260px; background: #fff; border: 1px solid #d9d9d9; border-radius: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.08); padding: 16px; z-index: 1000; }",
        "    .sidebar h2 { margin: 0 0 12px; font-size: 18px; }",
        "    .sidebar h3 { margin: 18px 0 10px; font-size: 15px; }",
        "    .sidebar label { display: block; font-size: 13px; margin: 10px 0 6px; color: #333; }",
        "    .sidebar input[type=number] { width: 100%; box-sizing: border-box; padding: 8px 10px; border: 1px solid #cfcfcf; border-radius: 8px; font-size: 14px; }",
        "    .sidebar button { width: 100%; margin-top: 12px; padding: 9px 10px; border: none; border-radius: 8px; background: #111827; color: #fff; cursor: pointer; font-size: 14px; }",
        "    .sidebar button.secondary { background: #f3f4f6; color: #111827; border: 1px solid #d1d5db; }",
        "    .sidebar .hint { margin-top: 10px; font-size: 12px; line-height: 1.4; color: #555; }",
        "    .sidebar .status { margin-top: 8px; font-size: 12px; color: #1f2937; }",
        "    .checkbox-row { display: flex; align-items: center; gap: 8px; margin-top: 10px; }",
        "    .checkbox-row input { width: auto; }",
        "    .legend-chip { display: flex; align-items: center; gap: 8px; margin-top: 8px; font-size: 12px; color: #444; }",
        "    .legend-swatch { width: 16px; height: 10px; border-radius: 3px; border: 1px solid rgba(0,0,0,0.08); }",
        "  </style>",
        "</head>",
        "<body>",
        f"  <h1>{escape(title)}</h1>",
        "  <p>Legend click: toggle traces. Drag: zoom. Double click: reset. Trade filter and imbalance background toggles apply to all charts.</p>",
        "  <div class='sidebar'>",
        "    <h2>Options</h2>",
        "    <h3>Trade volume filter</h3>",
        "    <label for='minTradeQty'>Min trade volume</label>",
        "    <input id='minTradeQty' type='number' step='any' placeholder='no lower bound'>",
        "    <label for='maxTradeQty'>Max trade volume</label>",
        "    <input id='maxTradeQty' type='number' step='any' placeholder='no upper bound'>",
        "    <button id='applyTradeFilter'>Apply filter</button>",
        "    <button id='resetTradeFilter' class='secondary' type='button'>Reset filter</button>",
        "    <div class='status' id='tradeFilterStatus'>Showing all trade points.</div>",
        "    <div class='hint'>Inclusive filter on trade quantity. Empty input means no bound.</div>",
        "    <h3>Order-book imbalance background</h3>",
        "    <div class='checkbox-row'>",
        "      <input id='toggleImbalanceBg' type='checkbox' checked>",
        "      <label for='toggleImbalanceBg' style='margin:0;'>Show background shading</label>",
        "    </div>",
        f"    <div class='hint'>Green: total bid volume - total ask volume ≥ {imbalance_threshold}. Pink: total ask volume - total bid volume ≥ {imbalance_threshold}.</div>",
        f"    <div class='legend-chip'><span class='legend-swatch' style='background:{GREEN_BG};'></span><span>bid dominance</span></div>",
        f"    <div class='legend-chip'><span class='legend-swatch' style='background:{PINK_BG};'></span><span>ask dominance</span></div>",
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
                "imbalance_shapes": item["imbalance_shapes"],
                "product": product,
                "day": day,
            }
        )

    chunks.append("  <script>")
    chunks.append(f"    const FIGURE_META = {json.dumps(figure_meta)};")
    chunks.append(
        """
    function parseBound(value, fallback) {
      if (value === '' || value === null || value === undefined) {
        return fallback;
      }
      const parsed = Number(value);
      return Number.isFinite(parsed) ? parsed : fallback;
    }

    function filterPayload(payload, minQty, maxQty) {
      const filtered = { x: [], y: [], customdata: [] };
      for (let i = 0; i < payload.quantity.length; i++) {
        const qty = Number(payload.quantity[i]);
        if (qty >= minQty && qty <= maxQty) {
          filtered.x.push(payload.x[i]);
          filtered.y.push(payload.y[i]);
          filtered.customdata.push(payload.customdata[i]);
        }
      }
      return filtered;
    }

    function updateStatus(minInput, maxInput) {
      const statusEl = document.getElementById('tradeFilterStatus');
      const minText = minInput === '' ? '-inf' : minInput;
      const maxText = maxInput === '' ? '+inf' : maxInput;
      statusEl.textContent = `Showing trades with volume in [${minText}, ${maxText}].`;
    }

    function resetStatus() {
      document.getElementById('tradeFilterStatus').textContent = 'Showing all trade points.';
    }

    function applyTradeFilter() {
      const minRaw = document.getElementById('minTradeQty').value.trim();
      const maxRaw = document.getElementById('maxTradeQty').value.trim();
      const minQty = parseBound(minRaw, -Infinity);
      const maxQty = parseBound(maxRaw, Infinity);

      FIGURE_META.forEach((meta) => {
        const gd = document.getElementById(meta.div_id);
        if (!gd) return;
        const filtered = filterPayload(meta.trade_payload, minQty, maxQty);
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

      updateStatus(minRaw, maxRaw);
    }

    function resetTradeFilter() {
      document.getElementById('minTradeQty').value = '';
      document.getElementById('maxTradeQty').value = '';

      FIGURE_META.forEach((meta) => {
        const gd = document.getElementById(meta.div_id);
        if (!gd) return;
        Plotly.restyle(
          gd,
          {
            x: [meta.trade_payload.x],
            y: [meta.trade_payload.y],
            customdata: [meta.trade_payload.customdata],
          },
          [meta.trade_trace_index]
        );
      });

      resetStatus();
    }

    function applyImbalanceBackgroundVisibility() {
      const checked = document.getElementById('toggleImbalanceBg').checked;
      FIGURE_META.forEach((meta) => {
        const gd = document.getElementById(meta.div_id);
        if (!gd) return;
        Plotly.relayout(gd, { shapes: checked ? meta.imbalance_shapes : [] });
      });
    }

    document.getElementById('applyTradeFilter').addEventListener('click', applyTradeFilter);
    document.getElementById('resetTradeFilter').addEventListener('click', resetTradeFilter);
    document.getElementById('toggleImbalanceBg').addEventListener('change', applyImbalanceBackgroundVisibility);

    document.getElementById('minTradeQty').addEventListener('keydown', (event) => {
      if (event.key === 'Enter') applyTradeFilter();
    });
    document.getElementById('maxTradeQty').addEventListener('keydown', (event) => {
      if (event.key === 'Enter') applyTradeFilter();
    });
    """
    )
    chunks.append("  </script>")
    chunks.extend(["</body>", "</html>"])
    return "\n".join(chunks)


def main() -> None:
    args = parse_args()

    prices_paths, trades_paths, label = discover_files(args)
    prices = read_prices(prices_paths)
    prices = forward_fill_prices(prices, default_fill_price=args.default_fill_price)
    prices = add_imbalance_columns(prices)
    trades = read_trades(trades_paths)

    figures: list[dict[str, object]] = []
    products = sorted(prices["product"].unique())

    for product in products:
        product_prices = prices.loc[prices["product"] == product].sort_values(["day", "timestamp"])
        product_trades = trades.loc[trades["symbol"] == product].sort_values(["day", "timestamp", "price"])

        for day in sorted(product_prices["day"].unique()):
            product_day_prices = product_prices.loc[product_prices["day"] == day].sort_values("timestamp")
            product_day_trades = product_trades.loc[product_trades["day"] == day].sort_values(["timestamp", "price"])
            fig, trade_payload, imbalance_shapes = build_product_day_figure(
                product=product,
                day=day,
                product_day_prices=product_day_prices,
                product_day_trades=product_day_trades,
                imbalance_threshold=args.imbalance_threshold,
            )
            figures.append(
                {
                    "product": product,
                    "day": day,
                    "figure": fig,
                    "trade_payload": trade_payload,
                    "imbalance_shapes": imbalance_shapes,
                }
            )

    output_path = make_output_path(label, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    html = build_html_document(figures, title=f"Prosperity market charts - {label}", imbalance_threshold=args.imbalance_threshold)
    output_path.write_text(html, encoding="utf-8")

    print(f"Saved interactive chart to: {output_path}")


if __name__ == "__main__":
    main()
