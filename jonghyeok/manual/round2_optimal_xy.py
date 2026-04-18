import math

def optimal_xy_and_S(z: int, fz: float):
    """
    입력:
        z  : Speed 투자 비율 (정수)
        fz : f(z) 값
    출력:
        최적 x, y, 그리고 그때의 S
    """
    if not (isinstance(z, int) and 0 <= z <= 100):
        raise ValueError("z는 0 이상 100 이하의 정수여야 합니다.")
    if not (0.0 <= fz <= 0.8):
        raise ValueError("f(z)는 0 이상 0.8 이하여야 합니다.")

    C = 14000 / math.log(101)

    best_x = None
    best_y = None
    best_S = -float("inf")

    max_xy_sum = 100 - z

    for x in range(max_xy_sum + 1):
        for y in range(max_xy_sum - x + 1):
            S = C * math.log(1 + x) * y * (0.1 + fz) - 500 * (x + y + z)
            if S > best_S:
                best_S = S
                best_x = x
                best_y = y

    return best_x, best_y, best_S


# 예시
z = 30
fz = 0.748

x_opt, y_opt, S_opt = optimal_xy_and_S(z, fz)
print(f"z={z}, f(z)={fz}")
print(f"optimal x={x_opt}, optimal y={y_opt}")
print(f"S={S_opt:.6f}")