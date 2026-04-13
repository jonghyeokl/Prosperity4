import csv
from collections import defaultdict

# 입력/출력 파일 경로
input_path = 'round-5-island-data-bottle/trades_round_5_all.csv'
input2_path = 'round-5-island-data-bottle/prices_round_5_all.csv'
input3_path = 'round5/positions_profits.csv'
output_path = 'round5/positions_profits_normalized.csv'

best_ask = defaultdict(lambda: defaultdict(int))
best_bid = defaultdict(lambda: defaultdict(int))
positions = defaultdict(lambda: defaultdict(float))
cumulative_positions = defaultdict(lambda: defaultdict(float))
profits   = defaultdict(lambda: defaultdict(float))
max_positions = defaultdict(lambda: defaultdict(float))
min_positions = defaultdict(lambda: defaultdict(float))
normalized_ratio = defaultdict(lambda: defaultdict(float))

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
limit_dict = {
    "RAINFOREST_RESIN": 50,
    "KELP": 50,
    "SQUID_INK": 50,
    "CROISSANTS": 250,
    "JAMS": 350,
    "DJEMBES": 60,
    "PICNIC_BASKET1": 60,
    "PICNIC_BASKET2": 100,
    "VOLCANIC_ROCK": 400,
    "VOLCANIC_ROCK_VOUCHER_9500": 200,
    "VOLCANIC_ROCK_VOUCHER_9750": 200,
    "VOLCANIC_ROCK_VOUCHER_10000": 200,
    "VOLCANIC_ROCK_VOUCHER_10250": 200,
    "VOLCANIC_ROCK_VOUCHER_10500": 200,
    "MAGNIFICENT_MACARONS": 75,
}

with open(input3_path, newline='') as f:
    reader = csv.DictReader(f, delimiter=',')
    for row in reader:
        symbol = row['symbol']
        person = row['person']
        pos_pos = float(row['pos_positions'])
        neg_pos = float(row['neg_positions'])
        normalized_ratio[symbol][person] = abs(neg_pos) / pos_pos if pos_pos != 0 else 1

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

        positions[symbol][buyer]  += qty * normalized_ratio[symbol][buyer]
        cumulative_positions[symbol][buyer] += qty * normalized_ratio[symbol][buyer]
        profits[symbol][buyer]    -= best_ask[symbol][timestamp+100] * qty * normalized_ratio[symbol][buyer]

        positions[symbol][seller] -= qty
        cumulative_positions[symbol][seller] += qty
        profits[symbol][seller]   += best_bid[symbol][timestamp+100] * qty

        max_positions[symbol][buyer] =  max(max_positions[symbol][buyer], positions[symbol][buyer])
        min_positions[symbol][buyer] =  min(min_positions[symbol][buyer], positions[symbol][buyer])

# 결과를 CSV로 저장
with open(output_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['symbol', 'person', 'pnl_ratio', 'max_pos', 'min_pos', 'cleared_pnl', 'positions_all', 'pos_positions', 'neg_positions', 'mult_pos', 'mult_neg'])
    for symbol in positions:
        people = set(positions[symbol]) | set(profits[symbol])
        for person in people:
            pos = positions[symbol][person]
            pnl = profits[symbol][person]
            positions_all = cumulative_positions[symbol][person]
            max_pos = max_positions[symbol][person]
            min_pos = min_positions[symbol][person]
            max_all = max(abs(min_pos), abs(max_pos))
            if max_all != 0:
                mult_pos = limit_dict.get(symbol, 0) * normalized_ratio[symbol][person] / max_all
                mult_neg = limit_dict.get(symbol, 0) / max_all
                max_pos *= limit_dict.get(symbol, 0) / max_all
                min_pos *= limit_dict.get(symbol, 0) / max_all
                pos *= limit_dict.get(symbol, 0) / max_all
                pnl *= limit_dict.get(symbol, 0) / max_all
                positions_all *= limit_dict.get(symbol, 0) / max_all
            else:
                mult_pos = 1
                mult_neg = 1
            cleared_pnl = pnl + clear_dict.get(symbol, 0) * pos
            if positions_all == 0:
                pnl_ratio = 0
            else:
                pnl_ratio = cleared_pnl / positions_all
            pos_positions = (positions_all + pos) / 2
            neg_positions = (positions_all - pos) / 2
            writer.writerow([symbol, person, f"{pnl_ratio:.2f}", f"{max_pos:.2f}", f"{min_pos:.2f}", f"{cleared_pnl:.2f}", f"{positions_all:.2f}", f"{pos_positions:.2f}", f"{neg_positions:.2f}", f"{mult_pos:.2f}", f"{mult_neg:.2f}"])

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

