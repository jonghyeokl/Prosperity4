import csv
from collections import defaultdict

input_path = 'round-5-island-data-bottle/prices_round_5_all.csv'  # 실제 파일명으로 변경
output_path = 'round5/avg_mid_price.csv'  # 결과를 저장할 파일명

sums   = defaultdict(float)
counts = defaultdict(int)

with open(input_path, newline='') as f:
    reader = csv.DictReader(f, delimiter=';')
    for row in reader:
        product   = row['product']
        mid_price = float(row['mid_price'])
        sums[product]   += mid_price
        counts[product] += 1

with open(output_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['product', 'avg_mid_price'])
    for product in sums:
        avg = sums[product] / counts[product]
        writer.writerow([product, f"{avg:.4f}"])
# # 결과 출력
# for product in sums:
#     avg = sums[product] / counts[product]
#     print(f'{product}: 평균 mid_price = {avg:.4f}')
