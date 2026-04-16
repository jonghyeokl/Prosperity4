from __future__ import annotations

from pathlib import Path
from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio


TARGET_SYMBOL = "ASH_COATED_OSMIUM"

# 현재 코드와 동일한 파라미터
ASH_EMA_WINDOW = 110
ASH_EMA_ALPHA = 2 / (ASH_EMA_WINDOW + 1)

HEADERS = [
    "day",
    "timestamp",
    "product",
    "bid_price_1",
    "bid_volume_1",
    "bid_price_2",
    "bid_volume_2",
    "bid_price_3",
    "bid_volume_3",
    "ask_price_1",
    "ask_volume_1",
    "ask_price_2",
    "ask_volume_2",
    "ask_price_3",
    "ask_volume_3",
    "mid_price",
    "profit_and_loss",
]


def read_price_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, delimiter=";")
    if "product" not in df.columns:
        df.columns = HEADERS[: len(df.columns)]
    return df


def load_ash_data() -> pd.DataFrame:
    project_root = Path(__file__).resolve().parents[2]
    round_dir = project_root / "data_capsule" / "round1"

    csv_paths = sorted(round_dir.glob("prices_round_1_day_*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"가격 파일을 찾지 못했습니다: {round_dir}")

    frames = []
    for path in csv_paths:
        df = read_price_file(path)
        df = df[df["product"] == TARGET_SYMBOL].copy()
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    for col in ["day", "timestamp", "bid_price_1", "ask_price_1"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["day", "timestamp"]).reset_index(drop=True)
    return df


def compute_mid_ema_p1_exactly_like_trader(df: pd.DataFrame) -> pd.DataFrame:
    """
    현재 트레이더 코드와 동일하게:
    - bid, ask 둘 다 있어야 mid 계산
    - mid가 있으면 EMA 업데이트
    - mid가 없으면 EMA state는 갱신하지 않음
    - 그래프용 EMA는 fallback 동작처럼 prev_ema를 표시
    - p1(t) = 0.5*mid(t-1) + 0.5*mid(t-2)
      단, mid(t-1), mid(t-2)가 둘 다 존재할 때만 계산
    - day가 바뀌면 state 초기화
    """
    out = df.copy()
    out["computed_mid"] = np.nan
    out["ema_like_trader"] = np.nan
    out["p1"] = np.nan

    for day, idx in out.groupby("day", sort=True).groups.items():
        indices = list(idx)

        prev_ema = None
        mid_history: list[float] = []

        for i in indices:
            bid = out.at[i, "bid_price_1"]
            ask = out.at[i, "ask_price_1"]

            best_bid = bid if pd.notna(bid) else None
            best_ask = ask if pd.notna(ask) else None

            if best_bid is not None and best_ask is not None:
                mid_price = (best_bid + best_ask) / 2
            else:
                mid_price = None

            if mid_price is not None:
                # EMA: 현재 trader 코드와 동일
                if prev_ema is None:
                    ema_value = mid_price
                else:
                    ema_value = ASH_EMA_ALPHA * mid_price + (1 - ASH_EMA_ALPHA) * prev_ema

                prev_ema = ema_value

                out.at[i, "computed_mid"] = mid_price
                out.at[i, "ema_like_trader"] = ema_value

                # p1(t) = 0.5*mid(t-1) + 0.5*mid(t-2)
                if len(mid_history) >= 2:
                    p1_value = 0.5 * mid_history[-1] + 0.5 * mid_history[-2]
                    out.at[i, "p1"] = p1_value
                else:
                    out.at[i, "p1"] = np.nan

                mid_history.append(mid_price)

            else:
                out.at[i, "computed_mid"] = np.nan
                out.at[i, "ema_like_trader"] = prev_ema if prev_ema is not None else np.nan
                out.at[i, "p1"] = np.nan

    return out


def build_day_figure(day_df: pd.DataFrame, day: int) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=day_df["timestamp"],
            y=day_df["computed_mid"],
            mode="lines",
            name="mid price",
            hovertemplate="timestamp=%{x}<br>mid=%{y}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=day_df["timestamp"],
            y=day_df["ema_like_trader"],
            mode="lines",
            name=f"EMA({ASH_EMA_WINDOW})",
            hovertemplate="timestamp=%{x}<br>ema=%{y}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=day_df["timestamp"],
            y=day_df["p1"],
            mode="lines",
            name="p1(t)=0.5*mid(t-1)+0.5*mid(t-2)",
            hovertemplate="timestamp=%{x}<br>p1=%{y}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"{TARGET_SYMBOL} - day {day}",
        xaxis_title="timestamp",
        yaxis_title="price",
        hovermode="x unified",
        dragmode="zoom",
        template="plotly_white",
        legend={"itemclick": "toggle", "itemdoubleclick": "toggleothers"},
        height=650,
        margin={"l": 60, "r": 30, "t": 60, "b": 60},
    )

    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor")
    fig.update_yaxes(showspikes=True, spikemode="across", spikesnap="cursor")

    return fig


def build_html_document(figures: list[tuple[int, go.Figure]], title: str) -> str:
    chunks = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "  <meta charset='utf-8'>",
        f"  <title>{escape(title)}</title>",
        "  <style>",
        "    body { font-family: Arial, sans-serif; margin: 24px; }",
        "    .chart-block { margin: 28px 0 48px; }",
        "  </style>",
        "</head>",
        "<body>",
        f"  <h1>{escape(title)}</h1>",
        "  <p>Drag to zoom, double-click to reset, legend click to show/hide traces.</p>",
    ]

    first = True
    for idx, (day, fig) in enumerate(figures):
        div_id = f"chart_{idx}"
        html = pio.to_html(
            fig,
            include_plotlyjs="cdn" if first else False,
            full_html=False,
            div_id=div_id,
        )
        first = False
        chunks.append(f"  <div class='chart-block'><h2>day {day}</h2>{html}</div>")

    chunks.extend(["</body>", "</html>"])
    return "\n".join(chunks)


def main():
    df = load_ash_data()
    df = compute_mid_ema_p1_exactly_like_trader(df)

    figures: list[tuple[int, go.Figure]] = []
    for day in sorted(df["day"].dropna().unique()):
        day = int(day)
        day_df = df[df["day"] == day].sort_values("timestamp").copy()
        fig = build_day_figure(day_df, day)
        figures.append((day, fig))

    project_root = Path(__file__).resolve().parents[2]
    output_path = project_root / "jonghyeok" / "data_analysis" / "output" / "ash_mid_ema_p1_interactive.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    html = build_html_document(
        figures,
        title=f"{TARGET_SYMBOL} mid price / EMA({ASH_EMA_WINDOW}) / p1(t)",
    )
    output_path.write_text(html, encoding="utf-8")

    print(f"Saved interactive chart to: {output_path}")


if __name__ == "__main__":
    main()