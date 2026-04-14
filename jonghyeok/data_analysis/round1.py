import pandas as pd
from pathlib import Path

# headers
headers = [
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

# read orders data
project_root = Path(__file__).resolve().parents[2]
csv_path = project_root / "data_capsule" / "round1" / "prices_round_1_day_-1.csv"
orders = pd.read_csv(csv_path, delimiter=';')

# dictionary 배열로 변환
orders = orders.to_dict(orient='records')

max_bid_price = 0
max_bid_price_timestamp = 0
max_bid_price_timestamp_fair_value = 0
max_bid_price_timestamp_fair_ask_value = 0
max_bid_price_timestamp_fair_bid_value = 0
recent_mid_price_history = []
recent_ask_price_history = []
recent_bid_price_history = []
cnt = 0

target_symbol = "INTARIAN_PEPPER_ROOT"
for order in orders:
    if order['product'] == target_symbol:
        fair_value = sum(recent_mid_price_history) / len(recent_mid_price_history) + 0.1 * (len(recent_mid_price_history) + 1)/2 if len(recent_mid_price_history) > 0 else 0
        fair_ask_value = sum(recent_ask_price_history) / len(recent_ask_price_history) + 0.1 * (len(recent_ask_price_history) + 1)/2 if len(recent_ask_price_history) > 0 else 0
        fair_bid_value = sum(recent_bid_price_history) / len(recent_bid_price_history) + 0.1 * (len(recent_bid_price_history) + 1)/2 if len(recent_bid_price_history) > 0 else 0
        if max_bid_price < order['bid_price_1']:
            max_bid_price = order['bid_price_1']
            max_bid_price_timestamp = order['timestamp']
            max_bid_price_timestamp_fair_value = fair_value
            max_bid_price_timestamp_fair_ask_value = fair_ask_value
            max_bid_price_timestamp_fair_bid_value = fair_bid_value
        
        if (not pd.isna(order['bid_price_1']) and not pd.isna(order['ask_price_1'])):
            mid_price = (order['bid_price_1'] + order['ask_price_1']) / 2
            recent_mid_price_history.append(mid_price)
            if len(recent_mid_price_history) > 199:
                recent_mid_price_history.pop(0)
        
        if not pd.isna(order['ask_price_1']):
            recent_ask_price_history.append(order['ask_price_1'])
            if len(recent_ask_price_history) > 199:
                recent_ask_price_history.pop(0)
        
        if not pd.isna(order['bid_price_1']):
            recent_bid_price_history.append(order['bid_price_1'])
            if len(recent_bid_price_history) > 199:
                recent_bid_price_history.pop(0)
        
        if pd.isna(order['ask_price_1']) and order['ask_price_1'] < fair_value:
            print(cnt, order['ask_price_1'], order['timestamp'], fair_value)
            cnt += 1
        
        if pd.isna(order['bid_price_1']) and order['bid_price_1'] > fair_value:
            print(cnt, order['bid_price_1'], order['timestamp'], fair_value)
            cnt += 1

        if max_bid_price > order['ask_price_1']:
            print(cnt, max_bid_price, max_bid_price_timestamp, max_bid_price_timestamp_fair_value, max_bid_price_timestamp_fair_ask_value, max_bid_price_timestamp_fair_bid_value, order['ask_price_1'], order['timestamp'], fair_value, fair_ask_value, fair_bid_value)
            cnt += 1