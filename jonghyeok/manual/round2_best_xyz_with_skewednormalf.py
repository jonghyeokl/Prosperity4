import math
import matplotlib.pyplot as plt

# =========================
# skew-normal 기반 f(z) 만들기
# =========================

def phi(t: float) -> float:
    return math.exp(-0.5 * t * t) / math.sqrt(2 * math.pi)

def Phi(t: float) -> float:
    return 0.5 * (1.0 + math.erf(t / math.sqrt(2)))

def skew_normal_pdf(x: float, xi: float, omega: float, alpha: float) -> float:
    """
    xi    : 중심 위치
    omega : 폭
    alpha : 왜도. alpha > 0 이면 right-skewed
    """
    t = (x - xi) / omega
    return (2.0 / omega) * phi(t) * Phi(alpha * t)

# 파라미터
XI = 30
OMEGA = 20
ALPHA = 0

# z=0~100 에 대해 pdf 값 계산
pdf_vals = [skew_normal_pdf(z, XI, OMEGA, ALPHA) for z in range(101)]

# 누적합으로 monotone increasing f(z) 생성
cum_vals = []
running = 0.0
for v in pdf_vals:
    running += v
    cum_vals.append(running)

base = cum_vals[0]
denom = cum_vals[-1] - base

if denom <= 0:
    raise ValueError("skew-normal 누적값 정규화에 실패했습니다.")

f_table = [
    0.8 * (c - base) / denom
    for c in cum_vals
]

def f(z: int) -> float:
    if not (0 <= z <= 100):
        raise ValueError("z must be in [0, 100]")
    return f_table[z]


# =========================
# 아래는 기존 코드 그대로
# =========================
C = 14000 / math.log(101)

def S_value(x: int, y: int, z: int) -> float:
    return C * math.log(1 + x) * y * (0.1 + f(z)) - 500 * (x + y + z)

results = []

for z in range(0, 101):
    fz = f(z)

    best_x = None
    best_y = None
    best_S = -float("inf")

    remaining = 100 - z
    for x in range(0, remaining + 1):
        for y in range(0, remaining - x + 1):
            S = S_value(x, y, z)
            if S > best_S:
                best_S = S
                best_x = x
                best_y = y

    results.append({
        "z": z,
        "f_z": fz,
        "x_opt": best_x,
        "y_opt": best_y,
        "S_opt": best_S,
    })

global_best = max(results, key=lambda row: row["S_opt"])

print("=== Global optimum ===")
print(f"z*   = {global_best['z']}")
print(f"f(z*)= {global_best['f_z']:.6f}")
print(f"x*   = {global_best['x_opt']}")
print(f"y*   = {global_best['y_opt']}")
print(f"S*   = {global_best['S_opt']:.6f}")

z_vals = [row["z"] for row in results]
f_vals = [row["f_z"] for row in results]
S_vals = [row["S_opt"] for row in results]

# f(z)-f(z-1)
delta_z_vals = list(range(1, 101))
delta_f_vals = [f(z) - f(z - 1) for z in delta_z_vals]

# 그래프 1: f(z)
# 그래프 1: f(z)
target_f = 0.4

# f(z)가 target_f와 가장 가까운 z 찾기
closest_idx = min(range(len(f_vals)), key=lambda i: abs(f_vals[i] - target_f))
z_at_target = z_vals[closest_idx]
f_at_target = f_vals[closest_idx]

plt.figure(figsize=(8, 5))
plt.plot(z_vals, f_vals, marker="o", markersize=3, label="f(z)")

# f(z)=0.4 수평선
plt.axhline(target_f, linestyle="--", alpha=0.7, label=f"f(z) = {target_f}")

# 가장 가까운 점 표시
plt.scatter([z_at_target], [f_at_target], s=60, marker="x")
plt.annotate(
    f"closest point\nz={z_at_target}, f(z)={f_at_target:.6f}",
    (z_at_target, f_at_target),
    xytext=(8, 8),
    textcoords="offset points",
    fontsize=9,
)

plt.xlabel("z")
plt.ylabel("f(z)")
plt.title("f(z) built from cumulative skew-normal PDF")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# 그래프 2: f(z)-f(z-1)
plt.figure(figsize=(8, 5))
plt.plot(delta_z_vals, delta_f_vals, marker="o", markersize=3)
plt.xlabel("z")
plt.ylabel("f(z) - f(z-1)")
plt.title("Increment of f(z)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 그래프 3: S*(z)
plt.figure(figsize=(8, 5))
plt.plot(z_vals, S_vals, marker="o", markersize=3)
plt.axvline(global_best["z"], linestyle="--", alpha=0.7)
plt.xlabel("z")
plt.ylabel("S*(z)")
plt.title("Optimal S*(z) for each z")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()