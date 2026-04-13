import csv
from collections import defaultdict

# 입력/출력 파일 경로
input_path = 'round-5-island-data-bottle/trades_round_5_all.csv'
input2_path = 'round-5-island-data-bottle/prices_round_5_all.csv'
output_path = 'round5/positions_profits.csv'

best_ask = defaultdict(lambda: defaultdict(int))
best_bid = defaultdict(lambda: defaultdict(int))
positions = defaultdict(lambda: defaultdict(float))
cumulative_positions = defaultdict(lambda: defaultdict(float))
profits   = defaultdict(lambda: defaultdict(float))
max_positions = defaultdict(lambda: defaultdict(float))
min_positions = defaultdict(lambda: defaultdict(float))

clear_dict = {
    'VOLCANIC_ROCK_VOUCHER_9750': 356.5,
    'PICNIC_BASKET1': 58438.5,
    'PICNIC_BASKET2': 30384.5,
    'MAGNIFICENT_MACARONS': 742.5,
    'VOLCANIC_ROCK_VOUCHER_9500': 605.5,
    'SQUID_INK': 1898.5,
    'VOLCANIC_ROCK_VOUCHER_10000': 122.5,
    'CROISSANTS': 4270.5,
    'JAMS': 6508.0,
    'KELP': 2011.5,
    'VOLCANIC_ROCK_VOUCHER_10500': 1.0,
    'DJEMBES': 13409.5,
    'VOLCANIC_ROCK': 10105.0,
    'RAINFOREST_RESIN': 9995.5,
    'VOLCANIC_ROCK_VOUCHER_10250': 10.5
}
avg_dict = {
    "PICNIC_BASKET2": 30111.1972,
    "VOLCANIC_ROCK_VOUCHER_9750": 327.4642,
    "RAINFOREST_RESIN": 10000.0032,
    "VOLCANIC_ROCK_VOUCHER_9500": 572.2578,
    "VOLCANIC_ROCK": 10071.7165,
    "SQUID_INK": 1881.8410,
    "VOLCANIC_ROCK_VOUCHER_10250": 24.9356,
    "KELP": 2035.3802,
    "DJEMBES": 13342.1741,
    "CROISSANTS": 4265.7104,
    "MAGNIFICENT_MACARONS": 707.0886,
    "JAMS": 6486.1615,
    "VOLCANIC_ROCK_VOUCHER_10000": 124.8732,
    "VOLCANIC_ROCK_VOUCHER_10500": 3.0454,
    "PICNIC_BASKET1": 58356.6625
}
with open(input2_path, newline='') as f:
    reader = csv.DictReader(f, delimiter=';')
    for row in reader:
        symbol = row['product']
        best_ask_1  = float(row['ask_price_1']) if row['ask_price_1'] != '' else 0
        best_bid_1  = float(row['bid_price_1']) if row['bid_price_1'] != '' else 0
        timestamp = int(row['timestamp'])
        best_ask[symbol][timestamp] = best_ask_1
        best_bid[symbol][timestamp] = best_bid_1

with open(input_path, newline='') as f:
    reader = csv.DictReader(f, delimiter=';')
    for row in reader:
        buyer, seller = row['buyer'], row['seller']
        symbol        = row['symbol']
        price, qty    = float(row['price']), float(row['quantity'])
        timestamp = int(row['timestamp'])

        if best_ask[symbol][timestamp+100] == 0 or best_bid[symbol][timestamp+100] == 0:
            continue
        if buyer == seller:
            continue

        positions[symbol][buyer]  += qty
        cumulative_positions[symbol][buyer] += qty
        profits[symbol][buyer]    -= best_ask[symbol][timestamp+100] * qty

        positions[symbol][seller] -= qty
        cumulative_positions[symbol][seller] += qty
        profits[symbol][seller]   += best_bid[symbol][timestamp+100] * qty

        max_positions[symbol][buyer] =  max(max_positions[symbol][buyer], positions[symbol][buyer])
        min_positions[symbol][buyer] =  min(min_positions[symbol][buyer], positions[symbol][buyer])

# 결과를 CSV로 저장
with open(output_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['symbol', 'person', 'pnl_ratio', 'pos_rate', 'max_pos', 'min_pos', 'cleared_pnl', 'positions_all', 'pos_positions', 'neg_positions'])
    for symbol in positions:
        people = set(positions[symbol]) | set(profits[symbol])
        for person in people:
            pos = positions[symbol][person]
            pnl = profits[symbol][person]
            cleared_pnl = pnl + clear_dict.get(symbol, 0) * pos
            positions_all = cumulative_positions[symbol][person]
            if positions_all == 0:
                pnl_ratio = 0
            else:
                pnl_ratio = cleared_pnl / positions_all
            pos_positions = (cumulative_positions[symbol][person] + positions[symbol][person]) / 2
            neg_positions = (cumulative_positions[symbol][person] - positions[symbol][person]) / 2
            pos_rate = pos_positions / positions_all if positions_all != 0 else 0
            max_pos = max_positions[symbol][person]
            min_pos = min_positions[symbol][person]
            writer.writerow([symbol, person, f"{pnl_ratio:.2f}", f"{pos_rate:.2f}", f"{max_pos:.2f}", f"{min_pos:.2f}", f"{cleared_pnl:.2f}", f"{positions_all:.2f}", f"{pos_positions:.2f}", f"{neg_positions:.2f}"])

# # 결과를 CSV로 저장
# with open(output_path, 'w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(['symbol', 'person', 'position', 'profit'])
#     for symbol in positions:
#         people = set(positions[symbol]) | set(profits[symbol])
#         for person in people:
#             pos = positions[symbol][person]
#             pnl = profits[symbol][person]
#             writer.writerow([symbol, person, pos, f"{pnl:.2f}"])

