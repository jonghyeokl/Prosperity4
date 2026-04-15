from dataclasses import dataclass

BIDS = {
    30: 30000,
    29: 5000,
    28: 12000,
    27: 28000,
}

ASKS = {
    28: 40000,
    31: 20000,
    32: 20000,
    33: 30000,
}

PRICE_MIN = 27
PRICE_MAX = 33

QTY_MIN = 1
QTY_MAX_EXCLUSIVE = 50000
QTY_STEP = 1

SETTLEMENT_PRICE = 30.0
FEE = 0


@dataclass
class Result:
    side: str
    price: int
    qty: int
    clearing_price: int
    filled_qty: int
    profit: float


def clearing_price_with_user(
    bids: dict[int, int],
    asks: dict[int, int],
    user_side: str,
    user_price: int,
    user_qty: int,
) -> int:
    cand_prices = range(PRICE_MIN, PRICE_MAX + 1)

    def traded_volume_at(c: int) -> int:
        bid_vol = sum(v for p, v in bids.items() if p >= c)
        ask_vol = sum(v for p, v in asks.items() if p <= c)

        if user_side == "buy" and user_price >= c:
            bid_vol += user_qty
        elif user_side == "sell" and user_price <= c:
            ask_vol += user_qty

        return min(bid_vol, ask_vol)

    best_c = None
    best_v = -1
    for c in cand_prices:
        v = traded_volume_at(c)
        if v > best_v or (v == best_v and (best_c is None or c > best_c)):
            best_v = v
            best_c = c
    return best_c


def user_fill_qty(
    bids: dict[int, int],
    asks: dict[int, int],
    user_side: str,
    user_price: int,
    user_qty: int,
    c: int,
) -> int:
    if user_side == "buy":
        if user_price < c:
            return 0

        eligible_supply = sum(v for p, v in asks.items() if p <= c)

        # Price priority, then time priority.
        # User is last in line at their price level.
        earlier_bid = sum(v for p, v in bids.items() if p >= c and p >= user_price)

        remaining = eligible_supply - earlier_bid
        return max(0, min(user_qty, remaining))

    elif user_side == "sell":
        if user_price > c:
            return 0

        eligible_demand = sum(v for p, v in bids.items() if p >= c)

        earlier_ask = sum(v for p, v in asks.items() if p <= c and p <= user_price)

        remaining = eligible_demand - earlier_ask
        return max(0, min(user_qty, remaining))

    else:
        raise ValueError("user_side must be 'buy' or 'sell'")


def profit_of_trade(side: str, clearing_price: int, filled_qty: int) -> float:
    if side == "buy":
        # Buy in auction, then sell to guild at 20
        return filled_qty * ((SETTLEMENT_PRICE) - (clearing_price + FEE))
    elif side == "sell":
        # Sell in auction, then buy back / settle at 20
        return filled_qty * ((clearing_price) - (SETTLEMENT_PRICE + FEE))
    else:
        raise ValueError("side must be 'buy' or 'sell'")


def evaluate(side: str, price: int, qty: int) -> Result:
    c = clearing_price_with_user(BIDS, ASKS, side, price, qty)
    filled = user_fill_qty(BIDS, ASKS, side, price, qty, c)
    profit = profit_of_trade(side, c, filled)
    return Result(side, price, qty, c, filled, profit)


def brute_force():
    best_buy = None
    best_sell = None

    for price in range(PRICE_MIN, PRICE_MAX + 1):
        for qty in range(QTY_MIN, QTY_MAX_EXCLUSIVE, QTY_STEP):
            r_buy = evaluate("buy", price, qty)
            if best_buy is None or r_buy.profit > best_buy.profit:
                best_buy = r_buy

            r_sell = evaluate("sell", price, qty)
            if best_sell is None or r_sell.profit > best_sell.profit:
                best_sell = r_sell

    return best_buy, best_sell


if __name__ == "__main__":
    best_buy, best_sell = brute_force()

    print("=== BEST BUY ===")
    print(best_buy)

    print("\n=== BEST SELL ===")
    print(best_sell)

    # Example
    # print(evaluate("buy", 17, 19900))