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

last_ask_price = 0
last_bid_price = 0
last_ask_na = False
last_bid_na = False
last_ask_na_count = 0
last_bid_na_count = 0
recent_mid_price_history = []
recent_ask_price_history = []
recent_bid_price_history = []
na_ask_map = {}
na_bid_map = {}
cnt = 0

target_symbol = "ASH_COATED_OSMIUM"
for order in orders:
    if order['product'] == target_symbol:
        limit = 80
        position = 0
        # must_sell_buy_coeff = 0.25
        beta_value = 14
        # alpha = 9.5

        # fair_value_coeff = 1 - must_sell_buy_coeff * pow(((limit + position) / (limit * 2)) , alpha)

        # avg_ask_price = sum(recent_ask_price_history) / len(recent_ask_price_history) if len(recent_ask_price_history) > 0 else 0
        # fair_ask_value = (avg_ask_price + 0.1 * (len(recent_ask_price_history) + 1)/2 + 0.1) if avg_ask_price != 0 else 0
        # avg_bid_price = sum(recent_bid_price_history) / len(recent_bid_price_history) if len(recent_bid_price_history) > 0 else 0
        # fair_bid_value = (avg_bid_price + 0.1 * (len(recent_bid_price_history) + 1)/2 + 0.1) if avg_bid_price != 0 else 0
        # fair_price = (fair_ask_value * fair_value_coeff + fair_bid_value * (1 - fair_value_coeff)) if (fair_ask_value != 0 and fair_bid_value != 0) else 0
        # if not pd.isna(order['bid_price_1']):
        #     recent_bid_price_history.append(order['bid_price_1'])
        # if len(recent_bid_price_history) > 199:
        #     recent_bid_price_history.pop(0)
        # if not pd.isna(order['ask_price_1']):
        #     recent_ask_price_history.append(order['ask_price_1'])
        # if len(recent_ask_price_history) > 199:
        #     recent_ask_price_history.pop(0)

        
        
        
        
        
        if not pd.isna(order['ask_price_1']) and not last_ask_na and last_ask_price != 0 and (last_ask_price <= order['ask_price_1'] - beta_value or last_ask_price >= order['ask_price_1'] + beta_value):
            cnt += 1
            print("ask", order['ask_price_1'], order['timestamp'])
        
        if not pd.isna(order['bid_price_1']) and not last_bid_na and last_bid_price != 0 and (last_bid_price >= order['bid_price_1'] + beta_value or last_bid_price <= order['bid_price_1'] - beta_value):
            cnt += 1
            print("bid", order['bid_price_1'], order['timestamp'])
            







        # if not pd.isna(order['ask_price_1']) and last_ask_na and order['timestamp'] != 100:
        #     # print(last_ask_na_count, last_ask_price, order['ask_price_1'], (order['ask_price_1'] - last_ask_price)/(last_ask_na_count + 1))
        #     cnt += 1
        #     if (order['ask_price_1'] - last_ask_price)/(last_ask_na_count + 1) not in na_ask_map:
        #         na_ask_map[(order['ask_price_1'] - last_ask_price)/(last_ask_na_count + 1)] = 1
        #     else:
        #         na_ask_map[(order['ask_price_1'] - last_ask_price)/(last_ask_na_count + 1)] += 1
        
        # if not pd.isna(order['bid_price_1']) and last_bid_na and order['timestamp'] != 100:
        #     # print(last_bid_na_count, last_bid_price, order['bid_price_1'], (order['bid_price_1'] - last_bid_price)/(last_bid_na_count + 1))
        #     cnt += 1
        #     if (order['bid_price_1'] - last_bid_price)/(last_bid_na_count + 1) not in na_bid_map:
        #         na_bid_map[(order['bid_price_1'] - last_bid_price)/(last_bid_na_count + 1)] = 1
        #     else:
        #         na_bid_map[(order['bid_price_1'] - last_bid_price)/(last_bid_na_count + 1)] += 1





        if not pd.isna(order['ask_price_1']):
            last_ask_price = order['ask_price_1']
            last_ask_na = False
            last_ask_na_count = 0
        else:
            last_ask_na = True
            last_ask_na_count += 1
        if not pd.isna(order['bid_price_1']):
            last_bid_price = order['bid_price_1']
            last_bid_na = False
            last_bid_na_count = 0
        else:
            last_bid_na = True
            last_bid_na_count += 1





        
print(cnt)
# print(sorted(na_ask_map.items(), key=lambda x: x[0], reverse=False))
# print(sorted(na_bid_map.items(), key=lambda x: x[0], reverse=False))