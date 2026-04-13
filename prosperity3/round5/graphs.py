import pandas as pd
import plotly.graph_objects as go

# --- CSV 파일 불러오기 ---
print("Loading trades data...")
trades_df = pd.read_csv("round-5-island-data-bottle/trades_round_5_all.csv", sep=";")

print("Loading trades data...done")
orderbook_df = pd.read_csv("round-5-island-data-bottle/prices_round_5_all.csv", sep=";")
print("Loading orderbook data...done")

# --- DJEMBES 데이터 필터링 ---
name = "VOLCANIC_ROCK"

djembes_trades = trades_df[trades_df["symbol"] == name]
djembes_orderbook = orderbook_df[orderbook_df["product"] == name]

# --- 중간 가격 정보 ---
order_timestamps = djembes_orderbook["timestamp"]
mid_prices = djembes_orderbook["mid_price"]

# --- 전체 figure 초기화 ---
fig = go.Figure()

# --- 중간 가격 라인 추가 ---
fig.add_trace(go.Scatter(
    x=order_timestamps,
    y=mid_prices,
    mode='lines',
    name='Order Book Mid Price',
    line=dict(color='blue')
))

# --- 거래자 조합별 점 추가 ---
buyers = djembes_trades["buyer"].dropna().unique()
sellers = djembes_trades["seller"].dropna().unique()

for buyer in buyers:
    for seller in sellers:
        subset = djembes_trades[
            (djembes_trades["buyer"] == buyer) &
            (djembes_trades["seller"] == seller)
        ]
        if not subset.empty:
            # if(len(subset) > 100):
            #     continue;
            fig.add_trace(go.Scatter(
                x=subset["timestamp"],
                y=subset["price"],
                mode='markers',
                name=f"{buyer} → {seller}",
                marker=dict(size=6, opacity=1),
                hovertemplate="timestamp: %{x}<br>price: %{y}<extra></extra>"
            ))

emt_subset = djembes_trades[
    (djembes_trades["buyer"].isna()) &
    (djembes_trades["seller"].isna())
]
if not emt_subset.empty:
    fig.add_trace(go.Scatter(
        x=emt_subset["timestamp"],
        y=emt_subset["price"],
        mode='markers',
        name="EMT",
        marker=dict(size=6, opacity=1),
        hovertemplate="timestamp: %{x}<br>price: %{y}<extra></extra>"
    ) )

# --- 레이아웃 설정 ---
fig.update_layout(
    title=name+": Order Book Mid Price & All Trades (by Buyer-Seller Pair)",
    xaxis_title="Timestamp",
    yaxis_title="Price",
    legend_title="Buyer → Seller",
    height=600
)

fig.show()