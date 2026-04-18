import math
import matplotlib.pyplot as plt

# =========================
# 1. 여기만 수정해서 f(z) 구현
# =========================
def f(z: int) -> float:
    """
    f(z) 임의로 설정
    """
    if z <= 30:
        return 0.6 * z / 30
    else:
        return 0.6 + 0.2 * (z - 30) / 70


# =========================
# 2. 문제 설정
# =========================
C = 14000 / math.log(101)

def S_value(x: int, y: int, z: int) -> float:
    return C * math.log(1 + x) * y * (0.1 + f(z)) - 500 * (x + y + z)


# =========================
# 3. 각 z에 대해 최적 (x, y) 찾기
# =========================
results = []  # 각 원소: dict(z=..., x=..., y=..., S=...)

for z in range(0, 101):
    fz = f(z)
    if not (0.0 <= fz <= 0.8):
        raise ValueError(f"f({z})={fz} 가 [0, 0.8] 범위를 벗어났습니다.")

    best_x = None
    best_y = None
    best_S = -float("inf")

    remaining = 100 - z
    for x in range(0, remaining + 1):
        for y in range(0, remaining - x + 1):
            S = C * math.log(1 + x) * y * (0.1 + fz) - 500 * (x + y + z)
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


# =========================
# 4. 전체 최적 (x, y, z) 찾기
# =========================
global_best = max(results, key=lambda row: row["S_opt"])

print("=== Global optimum ===")
print(f"z*   = {global_best['z']}")
print(f"f(z*)= {global_best['f_z']:.6f}")
print(f"x*   = {global_best['x_opt']}")
print(f"y*   = {global_best['y_opt']}")
print(f"S*   = {global_best['S_opt']:.6f}")

print("\n=== Per-z optimum (first 10 rows) ===")
for row in results[:10]:
    print(
        f"z={row['z']:3d}, "
        f"f(z)={row['f_z']:.3f}, "
        f"x*={row['x_opt']:2d}, "
        f"y*={row['y_opt']:2d}, "
        f"S*(z)={row['S_opt']:.3f}"
    )


# =========================
# 5. 그래프용 데이터
# =========================
z_vals = [row["z"] for row in results]
f_vals = [row["f_z"] for row in results]
S_vals = [row["S_opt"] for row in results]

# f(z)-f(z-1)
delta_z_vals = list(range(1, 101))
delta_f_vals = [f(z) - f(z - 1) for z in delta_z_vals]


# =========================
# 6. 그래프 1: f(z)
# =========================
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
plt.title("Custom f(z)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# =========================
# 7. 그래프 2: f(z)-f(z-1)
# =========================
plt.figure(figsize=(8, 5))
plt.plot(delta_z_vals, delta_f_vals, marker="o", markersize=3)
plt.xlabel("z")
plt.ylabel("f(z) - f(z-1)")
plt.title("Increment of f(z)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# =========================
# 8. 그래프 3: S*(z)
# =========================
plt.figure(figsize=(8, 5))
plt.plot(z_vals, S_vals, marker="o", markersize=3)
plt.axvline(global_best["z"], linestyle="--", alpha=0.7)
plt.xlabel("z")
plt.ylabel("S*(z)")
plt.title("Optimal S*(z) for each z")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()