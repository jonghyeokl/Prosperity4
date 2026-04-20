import math
import matplotlib.pyplot as plt

z_f_map = {
    0: 0.0,
    1: 0.0247605011,
    2: 0.0412675018,
    3: 0.0514861213,
    4: 0.056791943,
    5: 0.0821419799,
    6: 0.0874478015,
    7: 0.0935396708,
    8: 0.0994350282,
    9: 0.1017931712,
    10: 0.1507246377,
    11: 0.1611397691,
    12: 0.1668386146,
    13: 0.1701793171,
    14: 0.1711618767,
    15: 0.1980840088,
    16: 0.2049619258,
    17: 0.2092851879,
    18: 0.2138049619,
    19: 0.2163596168,
    20: 0.2715794645,
    21: 0.2867108818,
    22: 0.295357406,
    23: 0.3018422992,
    24: 0.3061655613,
    25: 0.3399656104,
    26: 0.352345861,
    27: 0.3629575043,
    28: 0.3672807664,
    29: 0.3700319332,
    30: 0.4175878163,
    31: 0.4287889953,
    32: 0.4370424957,
    33: 0.4504053058,
    34: 0.4506018177,
    35: 0.4810611643,
    36: 0.5172193564,
    37: 0.5404077622,
    38: 0.5551461557,
    39: 0.5624170965,
    40: 0.6070253009,
    41: 0.6343404569,
    42: 0.6345369688,
    43: 0.6520265291,
    44: 0.6598870056,
    45: 0.679538197,
    46: 0.693097519,
    47: 0.7013510194,
    48: 0.7052812577,
    49: 0.7080324245,
    50: 0.724932449,
    51: 0.7371161877,
    52: 0.7481208548,
    53: 0.7559813314,
    54: 0.7603045935,
    55: 0.7667894866,
    56: 0.7709162368,
    57: 0.7742569393,
    58: 0.7775976419,
    59: 0.7783836895,
    60: 0.7842790469,
    61: 0.7844755588,
    62: 0.7850650946,
    63: 0.786440678,
    64: 0.7880127733,
    65: 0.7897813805,
    66: 0.7907639401,
    67: 0.7915499877,
    68: 0.7921395235,
    69: 0.7927290592,
    70: 0.7943011545,
    71: 0.7960697617,
    72: 0.7964627856,
    73: 0.7964627856,
    74: 0.7966592975,
    75: 0.7968558094,
    76: 0.7970523213,
    77: 0.7974453451,
    78: 0.797641857,
    79: 0.797838369,
    80: 0.7986244166,
    81: 0.7990174404,
    82: 0.7990174404,
    83: 0.7990174404,
    84: 0.7990174404,
    85: 0.7992139523,
    86: 0.7992139523,
    87: 0.7992139523,
    88: 0.7992139523,
    89: 0.7992139523,
    90: 0.7994104643,
    91: 0.7996069762,
    92: 0.7996069762,
    93: 0.7996069762,
    94: 0.7996069762,
    95: 0.7996069762,
    96: 0.7996069762,
    97: 0.7996069762,
    98: 0.7996069762,
    99: 0.7996069762,
    100: 0.8,
}

# =========================
# 1. f(z)
# =========================
def f(z: int) -> float:
    return z_f_map[z]


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