import pandas as pd

# --- 각 파일 불러오기 및 타임프레임 조정 ---
df0 = pd.read_csv("round-5-island-data-bottle/trades_round_5_day_2.csv", sep=";")
df0["timestamp"] += 0 * 1_000_000

df1 = pd.read_csv("round-5-island-data-bottle/trades_round_5_day_3.csv", sep=";")
df1["timestamp"] += 1 * 1_000_000

df2 = pd.read_csv("round-5-island-data-bottle/trades_round_5_day_4.csv", sep=";")
df2["timestamp"] += 2 * 1_000_000

# --- 이어붙이기 ---
trades_df = pd.concat([df0, df1, df2], ignore_index=True)

# --- 저장 ---
trades_df.to_csv("trades_round_5_all.csv", sep=";", index=False)
print("✅ timestamp shift 완료 → 저장: trades_round_5_all.csv")


# --- 각 파일 불러오기 및 타임프레임 조정 ---
df0 = pd.read_csv("round-5-island-data-bottle/prices_round_5_day_2.csv", sep=";")
df0["timestamp"] += 0 * 1_000_000

df1 = pd.read_csv("round-5-island-data-bottle/prices_round_5_day_3.csv", sep=";")
df1["timestamp"] += 1 * 1_000_000

df2 = pd.read_csv("round-5-island-data-bottle/prices_round_5_day_4.csv", sep=";")
df2["timestamp"] += 2 * 1_000_000

# --- 이어붙이기 ---
trades_df = pd.concat([df0, df1, df2], ignore_index=True)

# --- 저장 ---
trades_df.to_csv("prices_round_5_all.csv", sep=";", index=False)
print("✅ timestamp shift 완료 → 저장: trades_round_5_all.csv")


df0 = pd.read_csv("round-5-island-data-bottle/observations_round_5_day_2.csv", sep=",")
df0["timestamp"] += 0 * 1_000_000
df1 = pd.read_csv("round-5-island-data-bottle/observations_round_5_day_3.csv", sep=",")
df1["timestamp"] += 1 * 1_000_000
df2 = pd.read_csv("round-5-island-data-bottle/observations_round_5_day_4.csv", sep=",")
df2["timestamp"] += 2 * 1_000_000
# --- 이어붙이기 ---
observations_df = pd.concat([df0, df1, df2], ignore_index=True)
# --- 저장 ---
observations_df.to_csv("observations_round_5_all.csv", sep=";", index=False)
print("✅ timestamp shift 완료 → 저장: observations_round_5_all.csv")
