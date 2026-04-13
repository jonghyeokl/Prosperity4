import numpy as np

# CSV 파일 경로 지정
file_path = './macaron_csv.csv'

# data.shape = (n_samples, 6)
data = np.loadtxt(file_path, delimiter=',', skiprows=1)

# y: 첫 번째 열, X: 나머지 5개 열
y = data[:, 1]
X = data[:, 2:]

# 절편(intercept) 포함을 위해 X에 1열 추가
X_aug = np.hstack([np.ones((X.shape[0], 1)), X])

# 최소제곱 해 구하기
coefs, *_ = np.linalg.lstsq(X_aug, y, rcond=None)

# Python float로 변환
intercept = float(coefs[0])
slopes = tuple(float(c) for c in coefs[1:])

# 예측값 계산
y_pred = X_aug.dot(coefs)

# R² 계산
ss_res = np.sum((y - y_pred)**2)
ss_tot = np.sum((y - np.mean(y))**2)
r2 = 1 - ss_res/ss_tot

# 결과 출력
print("Intercept:", intercept)
print("Coefficients:", slopes)
print("R²:", r2)