from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TARGET_SYMBOLS = [
    "HYDROGEL_PACK",
    "VELVETFRUIT_EXTRACT",
    "VEV_4000",
    "VEV_4500",
    "VEV_5000",
    "VEV_5100",
    "VEV_5200",
    "VEV_5300",
    "VEV_5400",
    "VEV_5500",
    "VEV_6000",
    "VEV_6500",
]

HEADERS = [
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


def read_price_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, delimiter=";")
    if "product" not in df.columns:
        df.columns = HEADERS[: len(df.columns)]
    return df


def load_symbol_data(symbol: str) -> pd.DataFrame:
    project_root = Path(__file__).resolve().parents[2]
    round_dir = project_root / "data_capsule" / "round3"

    csv_paths = sorted(round_dir.glob("prices_round_3_day_*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"가격 파일을 찾지 못했습니다: {round_dir}")
    
    frames = []
    for path in csv_paths:
        df = read_price_file(path)
        df = df[df["product"] == symbol].copy()
        frames.append(df)

    raw = pd.concat(frames, ignore_index=True)

    for col in [
        "day",
        "timestamp",
        "bid_price_1",
        "ask_price_1",
        "bid_price_2",
        "ask_price_2",
        "bid_price_3",
        "ask_price_3",
        "bid_volume_1",
        "ask_volume_1",
        "bid_volume_2",
        "ask_volume_2",
        "bid_volume_3",
        "ask_volume_3",
    ]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    raw = raw.sort_values(["day", "timestamp"]).reset_index(drop=True)
    return raw


def main():
    for symbol in TARGET_SYMBOLS:
        raw = load_symbol_data(symbol)
        bid_volume_map = {}
        ask_volume_map = {}
        for _, row in raw.iterrows():
            bid_volume = row["bid_volume_1"]
            ask_volume = row["ask_volume_1"]
            if bid_volume not in bid_volume_map:
                bid_volume_map[bid_volume] = 0
            bid_volume_map[bid_volume] += 1
            if ask_volume not in ask_volume_map:
                ask_volume_map[ask_volume] = 0
            ask_volume_map[ask_volume] += 1
        print(symbol, bid_volume_map)
        print(symbol, ask_volume_map)
        plt.bar(bid_volume_map.keys(), bid_volume_map.values())
        plt.xlabel('Bid Volume')
        plt.ylabel('Count')
        plt.title(f'{symbol} Bid Volume Distribution')
        plt.tight_layout()
        plt.show()
        plt.bar(ask_volume_map.keys(), ask_volume_map.values())
        plt.xlabel('Ask Volume')
        plt.ylabel('Count')
        plt.title(f'{symbol} Ask Volume Distribution')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()