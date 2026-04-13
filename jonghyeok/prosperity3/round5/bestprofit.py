import csv
from collections import defaultdict

# 입력/출력 파일 경로
input_path = 'round5/positions_profits_together.csv'
input2_path = 'round5/positions_profits_normalized.csv'
output_path = 'round5/positions_profits_best.csv'

best_pnl = defaultdict(lambda: defaultdict(float))

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

with open(input_path, newline='') as f:
    reader = csv.DictReader(f, delimiter=',')
    for row in reader:
        symbol = row['symbol']
        buyer = row['buyer']
        seller = row['seller']
        pnl_ratio = float(row['pnl_ratio'])
        max_pos = float(row['max_pos'])
        min_pos = float(row['min_pos'])
        cleared_pnl = float(row['cleared_pnl'])
        positions_all = float(row['positions_all'])
        pos_positions = float(row['pos_positions'])
        neg_positions = float(row['neg_positions'])
        mult_pos = float(row['mult_pos'])
        mult_neg = float(row['mult_neg'])
        if (symbol not in best_pnl) or (abs(best_pnl[symbol]['cleared_pnl']) < abs(cleared_pnl)):
            best_pnl[symbol]['symbol'] = symbol
            best_pnl[symbol]['buyer'] = buyer
            best_pnl[symbol]['seller'] = seller
            best_pnl[symbol]['pnl_ratio'] = pnl_ratio
            best_pnl[symbol]['max_pos'] = max_pos
            best_pnl[symbol]['min_pos'] = min_pos
            best_pnl[symbol]['cleared_pnl'] = cleared_pnl
            best_pnl[symbol]['positions_all'] = positions_all
            best_pnl[symbol]['pos_positions'] = pos_positions
            best_pnl[symbol]['neg_positions'] = neg_positions
            best_pnl[symbol]['mult_pos'] = mult_pos
            best_pnl[symbol]['mult_neg'] = mult_neg

with open(input2_path, newline='') as f:
    reader = csv.DictReader(f, delimiter=',')
    for row in reader:
        symbol = row['symbol']
        buyer = row['person']
        seller = row['person']
        pnl_ratio = float(row['pnl_ratio'])
        max_pos = float(row['max_pos'])
        min_pos = float(row['min_pos'])
        cleared_pnl = float(row['cleared_pnl'])
        positions_all = float(row['positions_all'])
        pos_positions = float(row['pos_positions'])
        neg_positions = float(row['neg_positions'])
        mult_pos = float(row['mult_pos'])
        mult_neg = float(row['mult_neg'])
        if (symbol not in best_pnl) or (abs(best_pnl[symbol]['cleared_pnl']) < abs(cleared_pnl)):
            best_pnl[symbol]['symbol'] = symbol
            best_pnl[symbol]['buyer'] = buyer
            best_pnl[symbol]['seller'] = seller
            best_pnl[symbol]['pnl_ratio'] = pnl_ratio
            best_pnl[symbol]['max_pos'] = max_pos
            best_pnl[symbol]['min_pos'] = min_pos
            best_pnl[symbol]['cleared_pnl'] = cleared_pnl
            best_pnl[symbol]['positions_all'] = positions_all
            best_pnl[symbol]['pos_positions'] = pos_positions
            best_pnl[symbol]['neg_positions'] = neg_positions
            best_pnl[symbol]['mult_pos'] = mult_pos
            best_pnl[symbol]['mult_neg'] = mult_neg

# 결과를 CSV로 저장
with open(output_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['symbol', 'buyer', 'seller', 'pnl_ratio', 'max_pos', 'min_pos', 'cleared_pnl', 'positions_all', 'pos_positions', 'neg_positions', 'mult_pos', 'mult_neg'])
    for symbol in product_list:
        writer.writerow([
            best_pnl[symbol]['symbol'],
            best_pnl[symbol]['buyer'],
            best_pnl[symbol]['seller'],
            f"{best_pnl[symbol]['pnl_ratio']:.2f}",
            f"{best_pnl[symbol]['max_pos']:.2f}",
            f"{best_pnl[symbol]['min_pos']:.2f}",
            f"{best_pnl[symbol]['cleared_pnl']:.2f}",
            f"{best_pnl[symbol]['positions_all']:.2f}",
            f"{best_pnl[symbol]['pos_positions']:.2f}",
            f"{best_pnl[symbol]['neg_positions']:.2f}",
            f"{best_pnl[symbol]['mult_pos']:.2f}",
            f"{best_pnl[symbol]['mult_neg']:.2f}"
        ])


