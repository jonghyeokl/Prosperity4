import csv
from collections import defaultdict

# 입력/출력 파일 경로
input_path = 'round-5-island-data-bottle/trades_round_5_all.csv'
input2_path = 'round-5-island-data-bottle/prices_round_5_all.csv'
input3_path = 'round5/positions_profits.csv'
output_path = 'round5/positions_profits_together.csv'

best_ask = defaultdict(lambda: defaultdict(int))
best_bid = defaultdict(lambda: defaultdict(int))
positions = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
cumulative_positions = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
profits   = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
max_positions = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
min_positions = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
normalized_ratio = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
product_buyer_list = defaultdict(list)
product_seller_list = defaultdict(list)
product_buyer_seller_list = defaultdict(list)

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
product_list = [
    "RAINFOREST_RESIN",
    "KELP",
    "SQUID_INK",
    "CROISSANTS",
    "JAMS",
    "DJEMBES",
    "PICNIC_BASKET1",
    "PICNIC_BASKET2",
    "VOLCANIC_ROCK",
    "VOLCANIC_ROCK_VOUCHER_9500",
    "VOLCANIC_ROCK_VOUCHER_9750",
    "VOLCANIC_ROCK_VOUCHER_10000",
    "VOLCANIC_ROCK_VOUCHER_10250",
    "VOLCANIC_ROCK_VOUCHER_10500",
    "MAGNIFICENT_MACARONS"
]

with open(input3_path, newline='') as f:
    reader = csv.DictReader(f, delimiter=',')
    rows = list(reader)
    for row in rows:
        symbol = row['symbol']
        person = row['person']
        pos_pos = float(row['pos_positions'])
        neg_pos = float(row['neg_positions'])
        if pos_pos > 0:
            product_buyer_list[symbol].append(person)
        if neg_pos > 0:
            product_seller_list[symbol].append(person)
    for product in product_list:
        for buyer in product_buyer_list[product]:
            for seller in product_seller_list[product]:
                if buyer != seller:
                    product_buyer_seller_list[product].append((buyer, seller))
                    normalized_ratio[product][buyer][seller] = -1
    for row in rows:
        symbol = row['symbol']
        person = row['person']
        pos_pos = float(row['pos_positions'])
        neg_pos = float(row['neg_positions'])
        if pos_pos > 0:
            for seller in product_seller_list[symbol]:
                if person != seller:
                    if normalized_ratio[symbol][person][seller] < 0:
                        normalized_ratio[symbol][person][seller] = 1 / pos_pos if pos_pos != 0 else 1
                    else:
                        normalized_ratio[symbol][person][seller] *= 1 / pos_pos if pos_pos != 0 else 1
        if neg_pos > 0:
            for buyer in product_buyer_list[symbol]:
                if person != buyer:
                    if normalized_ratio[symbol][buyer][person] < 0:
                        normalized_ratio[symbol][buyer][person] = abs(neg_pos)
                    else:
                        normalized_ratio[symbol][buyer][person] *= abs(neg_pos)

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

        for virseller in product_seller_list[symbol]:
            if buyer != virseller:
                positions[symbol][buyer][virseller]  += qty * normalized_ratio[symbol][buyer][virseller]
                cumulative_positions[symbol][buyer][virseller] += qty * normalized_ratio[symbol][buyer][virseller]
                profits[symbol][buyer][virseller]    -= best_ask[symbol][timestamp+100] * qty * normalized_ratio[symbol][buyer][virseller]
                max_positions[symbol][buyer][virseller] = max(max_positions[symbol][buyer][virseller], positions[symbol][buyer][virseller])
        
        for virbuyer in product_buyer_list[symbol]:
            if seller != virbuyer:
                positions[symbol][virbuyer][seller] -= qty
                cumulative_positions[symbol][virbuyer][seller] += qty
                profits[symbol][virbuyer][seller]   += best_bid[symbol][timestamp+100] * qty
                min_positions[symbol][virbuyer][seller] = min(min_positions[symbol][virbuyer][seller], positions[symbol][virbuyer][seller])

# 결과를 CSV로 저장
with open(output_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['symbol', 'buyer', 'seller', 'pnl_ratio', 'max_pos', 'min_pos', 'cleared_pnl', 'positions_all', 'pos_positions', 'neg_positions', 'mult_pos', 'mult_neg'])
    for symbol in product_list:
        for (buyer, seller) in product_buyer_seller_list[symbol]:
            pos = positions[symbol][buyer][seller]
            pnl = profits[symbol][buyer][seller]
            positions_all = cumulative_positions[symbol][buyer][seller]
            max_pos = max_positions[symbol][buyer][seller]
            min_pos = min_positions[symbol][buyer][seller]
            max_all = max(abs(min_pos), abs(max_pos))
            if max_all != 0:
                mult_pos = limit_dict.get(symbol, 0) * normalized_ratio[symbol][buyer][seller] / max_all
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
            pos_positions = (cumulative_positions[symbol][buyer][seller] + positions[symbol][buyer][seller]) / 2
            neg_positions = (cumulative_positions[symbol][buyer][seller] - positions[symbol][buyer][seller]) / 2
            writer.writerow([symbol, buyer, seller, f"{pnl_ratio:.2f}", f"{max_pos:.2f}", f"{min_pos:.2f}", f"{cleared_pnl:.2f}", f"{positions_all:.2f}", f"{pos_positions:.2f}", f"{neg_positions:.2f}", f"{mult_pos:.2f}", f"{mult_neg:.2f}"])


