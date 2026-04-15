

import pandas as pd
from pathlib import Path

# headers
headers = [
    "timestamp",
    "buyer",
    "seller",
    "symbol",
    "currency",
    "price",
    "quantity",
]

target_symbol = "ASH_COATED_OSMIUM"
quantity_cnt_map = {}

# read orders data
project_root = Path(__file__).resolve().parents[2]
# glob all trades_round_1_day_*.csv
csv_paths = list((project_root / "data_capsule" / "round1").glob("trades_round_1_day_*.csv"))
for csv_path in csv_paths:
    orders = pd.read_csv(csv_path, delimiter=';')
    # dictionary 배열로 변환
    orders = orders.to_dict(orient='records')
    for order in orders:
        if order['symbol'] == target_symbol:
            if order['quantity'] not in quantity_cnt_map:
                quantity_cnt_map[order['quantity']] = 0
            quantity_cnt_map[order['quantity']] += 1

print(sorted(quantity_cnt_map.items(), key=lambda x: x[0], reverse=False))