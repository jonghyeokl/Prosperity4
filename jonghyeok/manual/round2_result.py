from bs4 import BeautifulSoup
import csv

html_path = "./jonghyeok/manual/round2_result.html"

with open(html_path, "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")

xy_pairs = []
for path in soup.select('path.recharts-rectangle'):
    s = round(float(path.get("speed")))
    h = round(float(path.get("height")) * 10000 / 24250 * 3)
    if s is not None and h is not None:
        xy_pairs.append((s, h))

with open("./jonghyeok/manual/round2_result.csv", "w", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(xy_pairs)