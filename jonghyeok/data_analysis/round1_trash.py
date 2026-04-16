import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
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

target_symbol = "ASH_COATED_OSMIUM"
# read prices data
project_root = Path(__file__).resolve().parents[2]
# glob all prices_round_1_day_*.csv
csv_paths = list((project_root / "data_capsule" / "round1").glob("prices_round_1_day_*.csv"))


bid_ask_volume_map = {}

for csv_path in csv_paths:
    orders = pd.read_csv(csv_path, delimiter=';')
    # dictionary 배열로 변환
    orders = orders.to_dict(orient='records')
    for order in orders:
        if order['product'] == target_symbol:
            bid_volume_1 = order['bid_volume_1']
            ask_volume_1 = order['ask_volume_1']
            if not pd.isna(bid_volume_1):
                if bid_volume_1 not in bid_ask_volume_map:
                    bid_ask_volume_map[bid_volume_1] = 0
                bid_ask_volume_map[bid_volume_1] += 1
            if not pd.isna(ask_volume_1):
                if ask_volume_1 not in bid_ask_volume_map:
                    bid_ask_volume_map[ask_volume_1] = 0
                bid_ask_volume_map[ask_volume_1] += 1

print(sorted(bid_ask_volume_map.items(), key=lambda x: x[0], reverse=False));

# draw histogram
plt.bar(bid_ask_volume_map.keys(), bid_ask_volume_map.values());
plt.xlabel('Bid Ask Volume');
plt.ylabel('Count');
plt.title('Bid Ask Volume Distribution');
plt.tight_layout();
plt.show();
