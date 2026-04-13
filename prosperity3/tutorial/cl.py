import json
import matplotlib.pyplot as plt

# log.json 파일은 JSON 객체들의 배열이어야 함
with open("log.json", "r") as f:
    log_data = json.load(f)

timestamps = []
resin_positions = []

for log_entry in log_data:
    # 각 로그 항목은 JSON 객체이고, 압축된 로그 정보는 "lambdaLog" 필드에 JSON 문자열 형태로 저장됨
    lambda_log_str = log_entry["lambdaLog"]
    lambda_log = json.loads(lambda_log_str)  # 압축된 로그 배열 파싱
    
    # lambda_log의 첫 번째 요소가 compress_state로 압축된 상태 정보입니다.
    state_info = lambda_log[0]
    timestamp = state_info[0]           # 인덱스 0: 타임스탬프
    position_dict = state_info[6]         # 인덱스 6: 포지션 정보 (딕셔너리)
    resin_position = position_dict.get("RAINFOREST_RESIN", 0)
    
    timestamps.append(timestamp)
    resin_positions.append(resin_position)

# 그래프 생성 및 "graph.png" 파일로 저장
plt.figure(figsize=(10,6))
plt.plot(timestamps, resin_positions, marker='o', linestyle='-', color='blue')
plt.xlabel("Timestamp")
plt.ylabel("RAINFOREST_RESIN Position")
plt.title("RAINFOREST_RESIN Position Over Time")
plt.grid(True)
plt.savefig("graph.png")
plt.close()

print("Graph saved to graph.png")
