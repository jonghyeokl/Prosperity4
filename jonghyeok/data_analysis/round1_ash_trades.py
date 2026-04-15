

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

# read orders data
project_root = Path(__file__).resolve().parents[2]
csv_path = project_root / "data_capsule" / "round1" / "trades_round_1_day_-1.csv"
orders = pd.read_csv(csv_path, delimiter=';')

# dictionary 배열로 변환
orders = orders.to_dict(orient='records')


target_symbol = "ASH_COATED_OSMIUM"
for order in orders:
    if order['symbol'] == target_symbol:
        if order['quantity'] >= 10:
            print(order)