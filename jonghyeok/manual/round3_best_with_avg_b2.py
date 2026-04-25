# =========================
# Parameters
# =========================
AVG_B2 = 850          # 전체 플레이어 second bid 평균 가정
BID_MIN = 670
BID_MAX = 920
BID_STEP = 1          # bid도 5 단위로 탐색. 1 단위로 보고 싶으면 1로 변경

RESERVE_PRICES = list(range(670, 921, 5))  # 670, 675, ..., 920


def penalty_factor(b2: int, avg_b2: float) -> float:
    if b2 >= avg_b2:
        return 1.0
    return ((920 - avg_b2) / (920 - b2)) ** 3


def pnl_for_counterparty(r: int, b1: int, b2: int, avg_b2: float) -> float:
    """
    reserve price r를 가진 counterparty 1명에 대한 PnL
    higher는 >= 로 해석.
    """
    # first bid 체결
    if b1 >= r:
        return 920 - b1

    # second bid 체결
    if b2 >= r:
        return (920 - b2) * penalty_factor(b2, avg_b2)

    # 미체결
    return 0.0


def expected_pnl(b1: int, b2: int, avg_b2: float) -> float:
    """
    reserve price가 균등분포일 때 counterparty 1명당 기대 PnL
    """
    return sum(
        pnl_for_counterparty(r, b1, b2, avg_b2)
        for r in RESERVE_PRICES
    ) / len(RESERVE_PRICES)


def find_best(avg_b2: float):
    best = None

    for b1 in range(BID_MIN, BID_MAX + 1, BID_STEP):
        for b2 in range(BID_MIN, BID_MAX + 1, BID_STEP):
            pnl = expected_pnl(b1, b2, avg_b2)

            if best is None or pnl > best["expected_pnl"]:
                best = {
                    "b1": b1,
                    "b2": b2,
                    "expected_pnl": pnl,
                }

    return best


if __name__ == "__main__":
    best = find_best(AVG_B2)

    print(f"AVG_B2 = {AVG_B2}")
    print(f"best b1 = {best['b1']}")
    print(f"best b2 = {best['b2']}")
    print(f"expected PnL per counterparty = {best['expected_pnl']:.6f}")