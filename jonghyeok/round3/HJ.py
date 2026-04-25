from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Any, Dict, Optional, Tuple
import json
import jsonpickle
import math
import copy
from datamodel import (
    Listing, Observation, Order, OrderDepth, ProsperityEncoder,
    Symbol, Trade, TradingState,
)


# ============================================================
#  Logger
# ============================================================
class Logger:
    def __init__(self):
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects, sep=" ", end="\n"):
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state, orders, conversions, trader_data):
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders), conversions, "", "",
        ]))
        max_item_length = (self.max_log_length - base_length) // 3
        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders), conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))
        self.logs = ""

    def compress_state(self, state, trader_data):
        return [
            state.timestamp, trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings):
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def compress_order_depths(self, order_depths):
        return {s: [od.buy_orders, od.sell_orders] for s, od in order_depths.items()}

    def compress_trades(self, trades):
        out = []
        for arr in trades.values():
            for t in arr:
                out.append([t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp])
        return out

    def compress_observations(self, observations):
        co = {}
        for p, o in observations.conversionObservations.items():
            co[p] = [
                o.bidPrice,
                o.askPrice,
                o.transportFees,
                o.exportTariff,
                o.importTariff,
                o.sugarPrice,
                o.sunlightIndex,
            ]
        return [observations.plainValueObservations, co]

    def compress_orders(self, orders):
        out = []
        for arr in orders.values():
            for o in arr:
                out.append([o.symbol, o.price, o.quantity])
        return out

    def to_json(self, v):
        return json.dumps(v, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, v, n):
        return v if len(v) <= n else v[:n - 3] + "..."


logger = Logger()


# ============================================================
#  Black-Scholes (call, r=0)
# ============================================================
def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_call_with_greeks(S, K, T, sigma):
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return max(S - K, 0.0), (1.0 if S > K else 0.0), 0.0

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    price = S * _norm_cdf(d1) - K * _norm_cdf(d2)
    delta = _norm_cdf(d1)
    vega = S * _norm_pdf(d1) * sqrtT

    return price, delta, vega


def implied_vol(V, S, K, T, tol=1e-4, max_iter=50):
    """Bisection IV. Returns None if no solution in range."""
    if T <= 0 or V <= 0 or S <= 0 or K <= 0:
        return None

    intrinsic = max(S - K, 0.0)

    if V < intrinsic - 1e-2 or V > S + 1e-2:
        return None

    lo, hi = 1e-3, 3.0

    f_lo = bs_call_with_greeks(S, K, T, lo)[0] - V
    f_hi = bs_call_with_greeks(S, K, T, hi)[0] - V

    if f_lo * f_hi > 0:
        return None

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = bs_call_with_greeks(S, K, T, mid)[0] - V

        if abs(f_mid) < tol:
            return mid

        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid

    return 0.5 * (lo + hi)


def fit_quadratic_from_points(points: List[Tuple[float, float]]) -> Optional[Tuple[float, float, float]]:
    """OLS fit iv = a*m^2 + b*m + c. Returns None on singular."""
    n = len(points)

    if n < 4:
        return None

    Sx4 = Sx3 = Sx2 = Sx1 = 0.0
    Sy = Syx = Syx2 = 0.0

    for m, v in points:
        m2 = m * m
        Sx4 += m2 * m2
        Sx3 += m2 * m
        Sx2 += m2
        Sx1 += m
        Sy += v
        Syx += v * m
        Syx2 += v * m2

    # 3x3 normal equations
    M = [
        [Sx4, Sx3, Sx2, Syx2],
        [Sx3, Sx2, Sx1, Syx],
        [Sx2, Sx1, float(n), Sy],
    ]

    try:
        for i in range(3):
            pivot = M[i][i]

            if abs(pivot) < 1e-15:
                swap = None
                for k in range(i + 1, 3):
                    if abs(M[k][i]) > 1e-15:
                        swap = k
                        break

                if swap is None:
                    return None

                M[i], M[swap] = M[swap], M[i]
                pivot = M[i][i]

            for j in range(i + 1, 3):
                factor = M[j][i] / pivot
                for c in range(i, 4):
                    M[j][c] -= factor * M[i][c]

        x = [0.0, 0.0, 0.0]

        for i in range(2, -1, -1):
            x[i] = M[i][3]
            for j in range(i + 1, 3):
                x[i] -= M[i][j] * x[j]
            x[i] /= M[i][i]

        return x[0], x[1], x[2]

    except (ZeroDivisionError, Exception):
        return None


# ============================================================
#  Trader
# ============================================================
class Trader:

    POSITION_LIMITS = {
        "HYDROGEL_PACK": 200,
        "VELVETFRUIT_EXTRACT": 200,
        "VEV_4000": 300, "VEV_4500": 300,
        "VEV_5000": 300, "VEV_5100": 300, "VEV_5200": 300,
        "VEV_5300": 300, "VEV_5400": 300, "VEV_5500": 300,
        "VEV_6000": 300, "VEV_6500": 300,
    }

    UNDERLYING = "VELVETFRUIT_EXTRACT"

    ATM_VOUCHERS = {
        "VEV_5000": 5000,
        "VEV_5100": 5100,
        "VEV_5200": 5200,
        "VEV_5300": 5300,
    }

    # ----- Deep ITM vouchers: fair = S - K (대칭 take/make) -----
    DEEP_ITM_VOUCHERS = {
        "VEV_4000": 4000,
        "VEV_4500": 4500,
    }

    # ----- Rolling smile fit -----
    SMILE_WINDOW_PER_VOUCHER = 300   # 각 voucher 당 저장할 tick 수
    SMILE_MIN_POINTS_FOR_FIT = 600   # 총 점수 이상일 때만 rolling fit 사용


    # ----- Fallback smile (rolling warmup 중) -----
    # 5000~5300 pooled fit (실제는 rolling 로 대체되지만 초기 fallback)
    IV_A_FALLBACK = 0.13797259576052961
    IV_B_FALLBACK = 0.04060700203856375
    IV_C_FALLBACK = 0.24224047421616432

    # Valid volume
    VALID_BID_ASK_VOLUME = {
        "VELVETFRUIT_EXTRACT": 15,
        "VEV_5000": 6,
        "VEV_5100": 6,
        "VEV_5200": 6,
        "VEV_5300": 5,
        "VEV_5400": 5,
        "VEV_5500": 5,
    }

    # 실전 Round 3 시작 시 TTE = 5 days. day_num=3이면 8 - 3 = 5.
    TTE_BASE_DAYS = 5.0
    DAYS_PER_YEAR = 365.0

    THEO_DIFF_WINDOW = 20
    SCALE_WINDOW = 100
    WARMUP_TICKS = 30

    OPEN_THRESHOLD = 0.0
    CLOSE_THRESHOLD = 0.0
    ENABLE_CLOSE = True

    LOW_VEGA_CUTOFF = 0.5
    LOW_VEGA_PENALTY = 0.5

    MIN_SCALE = 0.0

    # ==========================================================
    def bid(self):
        return 0

    def get_best_bid_ask(self, order_depth: OrderDepth):
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        return best_bid, best_ask

    def get_best_valid_bid_ask(self, order_depth: OrderDepth, valid_volume: int):
        best_valid_bid = None
        best_valid_ask = None

        for bid_price, bid_vol in sorted(order_depth.buy_orders.items(), reverse=True):
            if bid_vol >= valid_volume:
                best_valid_bid = bid_price
                break

        for ask_price, ask_vol in sorted(order_depth.sell_orders.items()):
            if -ask_vol >= valid_volume:
                best_valid_ask = ask_price
                break

        best_bid, best_ask = self.get_best_bid_ask(order_depth)

        if best_valid_bid is None:
            best_valid_bid = best_bid
        if best_valid_ask is None:
            best_valid_ask = best_ask

        return best_valid_bid, best_valid_ask

    def get_valid_mid_price(self, order_depth: OrderDepth, valid_volume: int):
        best_valid_bid, best_valid_ask = self.get_best_valid_bid_ask(order_depth, valid_volume)

        if best_valid_bid is not None and best_valid_ask is not None:
            return (best_valid_bid + best_valid_ask) / 2

        return None

    def add_buy_order(self, orders: List[Order], product: str, price: int, volume: int, buy_limit: int) -> int:
        volume = min(volume, buy_limit)

        if volume > 0:
            orders.append(Order(product, int(price), int(volume)))
            buy_limit -= volume

        return buy_limit

    def add_sell_order(self, orders: List[Order], product: str, price: int, volume: int, sell_limit: int) -> int:
        volume = min(volume, sell_limit)

        if volume > 0:
            orders.append(Order(product, int(price), -int(volume)))
            sell_limit -= volume

        return sell_limit

    def get_tte_years(self, timestamp: int, day_num: int) -> float:
        progress_days = timestamp / 1_000_000.0
        remaining_days = max(8.0 - day_num - progress_days, 1e-9)
        return remaining_days / self.DAYS_PER_YEAR

    def ema(self, traderObject: dict, key: str, value: float, window: int) -> float:
        alpha = 2.0 / (window + 1.0)

        if key not in traderObject:
            traderObject[key] = value
            return value

        new_value = alpha * value + (1.0 - alpha) * traderObject[key]
        traderObject[key] = new_value

        return new_value

    # ==========================================================
    #  Deep ITM voucher trading: fair = S - K
    # ==========================================================
    def get_deep_itm_voucher_orders(self, product: str, state: TradingState) -> List[Order]:
        orders: List[Order] = []

        if self.UNDERLYING not in state.order_depths:
            return orders

        voucher_depth = state.order_depths[product]
        underlying_depth = state.order_depths[self.UNDERLYING]

        underlying_mid = self.get_valid_mid_price(
            underlying_depth,
            self.VALID_BID_ASK_VOLUME[self.UNDERLYING],
        )

        if underlying_mid is None:
            return orders

        strike = self.DEEP_ITM_VOUCHERS[product]
        fair = underlying_mid - strike

        if fair <= 0:
            return orders

        best_bid, best_ask = self.get_best_bid_ask(voucher_depth)

        if best_bid is None or best_ask is None:
            return orders

        position = state.position.get(product, 0)
        limit = self.POSITION_LIMITS[product]

        buy_limit = limit - position
        sell_limit = limit + position

        # 1. Take: 비싼 bid 에 팔기
        for bid_price, bid_vol in sorted(voucher_depth.buy_orders.items(), reverse=True):
            if bid_price > fair and sell_limit > 0:
                sell_limit = self.add_sell_order(
                    orders=orders,
                    product=product,
                    price=bid_price,
                    volume=bid_vol,
                    sell_limit=sell_limit,
                )
            else:
                break

        # 2. Take: 싼 ask 에 사기
        for ask_price, ask_vol in sorted(voucher_depth.sell_orders.items()):
            ask_qty = -ask_vol

            if ask_price < fair and buy_limit > 0:
                buy_limit = self.add_buy_order(
                    orders=orders,
                    product=product,
                    price=ask_price,
                    volume=ask_qty,
                    buy_limit=buy_limit,
                )
            else:
                break

        # 3. Make: fair 아래 지정가 bid
        buy_price = min(best_bid + 1, math.floor(fair - 0.1))

        if buy_limit > 0 and buy_price < best_ask:
            buy_limit = self.add_buy_order(
                orders=orders,
                product=product,
                price=buy_price,
                volume=buy_limit,
                buy_limit=buy_limit,
            )

        # 4. Make: fair 위 지정가 ask
        sell_price = max(best_ask - 1, math.ceil(fair + 0.1))

        if sell_limit > 0 and sell_price > best_bid:
            sell_limit = self.add_sell_order(
                orders=orders,
                product=product,
                price=sell_price,
                volume=sell_limit,
                sell_limit=sell_limit,
            )

        return orders

    # ==========================================================
    #  Rolling smile fit
    # ==========================================================
    def update_smile_history_and_fit(
        self,
        state: TradingState,
        traderObject: dict,
        day_num: int,
    ) -> Tuple[Optional[Tuple[float, float, float]], Optional[float], Optional[float]]:
        """
        Per voucher history에 (m, iv) 추가 후 전체 point로 fit.
        Returns (coeffs_or_None, S_mid, T).
        coeffs는 fit 실패/warmup 시 None.
        """
        if self.UNDERLYING not in state.order_depths:
            return None, None, None

        underlying_depth = state.order_depths[self.UNDERLYING]

        underlying_mid = self.get_valid_mid_price(
            underlying_depth,
            self.VALID_BID_ASK_VOLUME[self.UNDERLYING],
        )

        if underlying_mid is None:
            return None, None, None

        T = self.get_tte_years(state.timestamp, day_num)

        if T <= 0:
            return None, underlying_mid, T

        hist_key = "smile_hist"

        if hist_key not in traderObject:
            traderObject[hist_key] = {p: [] for p in self.ATM_VOUCHERS}

        hist = traderObject[hist_key]

        for product in self.ATM_VOUCHERS:
            if product not in hist:
                hist[product] = []

        sqrt_t = math.sqrt(T)

        for product, K in self.ATM_VOUCHERS.items():
            if product not in state.order_depths:
                continue

            option_depth = state.order_depths[product]
            option_valid_mid = self.get_valid_mid_price(
                option_depth,
                self.VALID_BID_ASK_VOLUME.get(product, 6),
            )

            if option_valid_mid is None:
                continue

            iv = implied_vol(option_valid_mid, underlying_mid, K, T)

            if iv is None:
                continue

            m = math.log(K / underlying_mid) / sqrt_t

            hist[product].append((m, iv))

            if len(hist[product]) > self.SMILE_WINDOW_PER_VOUCHER:
                hist[product] = hist[product][-self.SMILE_WINDOW_PER_VOUCHER:]

        all_points = []

        for product in self.ATM_VOUCHERS:
            all_points.extend(hist[product])

        if len(all_points) < self.SMILE_MIN_POINTS_FOR_FIT:
            return None, underlying_mid, T

        coeffs = fit_quadratic_from_points(all_points)

        return coeffs, underlying_mid, T

    def get_fair_iv(self, m: float, rolling_coeffs: Optional[Tuple[float, float, float]]) -> float:
        if rolling_coeffs is not None:
            a, b, c = rolling_coeffs
        else:
            a, b, c = self.IV_A_FALLBACK, self.IV_B_FALLBACK, self.IV_C_FALLBACK

        return a * m * m + b * m + c

    # ==========================================================
    #  ATM voucher trading
    # ==========================================================
    def get_atm_voucher_orders(
        self,
        product: str,
        state: TradingState,
        traderObject: dict,
        rolling_coeffs: Optional[Tuple[float, float, float]],
        underlying_mid: Optional[float],
        T: Optional[float],
    ) -> List[Order]:
        orders: List[Order] = []

        if underlying_mid is None or T is None:
            return orders

        if underlying_mid <= 0 or T <= 0:
            return orders

        option_depth = state.order_depths[product]

        option_best_bid, option_best_ask = self.get_best_bid_ask(option_depth)

        if option_best_bid is None or option_best_ask is None:
            return orders

        option_valid_mid = self.get_valid_mid_price(
            option_depth,
            self.VALID_BID_ASK_VOLUME.get(product, 6),
        )

        if option_valid_mid is None:
            return orders

        K = self.ATM_VOUCHERS[product]
        sqrt_t = math.sqrt(T)
        m = math.log(K / underlying_mid) / sqrt_t

        sigma = self.get_fair_iv(m, rolling_coeffs)

        if sigma <= 0 or not math.isfinite(sigma):
            return orders

        theo, _delta, vega = bs_call_with_greeks(underlying_mid, K, T, sigma)

        theo_diff = option_valid_mid - theo

        mean_key = f"{product}_mean"
        scale_key = f"{product}_scale"
        count_key = f"{product}_count"

        mean_diff = self.ema(
            traderObject=traderObject,
            key=mean_key,
            value=theo_diff,
            window=self.THEO_DIFF_WINDOW,
        )

        abs_dev = abs(theo_diff - mean_diff)

        scale = self.ema(
            traderObject=traderObject,
            key=scale_key,
            value=abs_dev,
            window=self.SCALE_WINDOW,
        )

        traderObject[count_key] = traderObject.get(count_key, 0) + 1
        count = traderObject[count_key]

        if count < self.WARMUP_TICKS:
            return orders

        if scale < self.MIN_SCALE:
            return orders

        sell_signal = option_best_bid - theo - mean_diff
        buy_signal = option_best_ask - theo - mean_diff

        low_vega_adj = self.LOW_VEGA_PENALTY if vega <= self.LOW_VEGA_CUTOFF else 0.0
        open_threshold = self.OPEN_THRESHOLD + low_vega_adj

        position = state.position.get(product, 0)
        limit = self.POSITION_LIMITS[product]

        buy_limit = limit - position
        sell_limit = limit + position

        # Open
        if sell_signal >= open_threshold and sell_limit > 0:
            sell_limit = self.add_sell_order(
                orders=orders,
                product=product,
                price=option_best_bid,
                volume=sell_limit,
                sell_limit=sell_limit,
            )

        elif buy_signal <= -open_threshold and buy_limit > 0:
            buy_limit = self.add_buy_order(
                orders=orders,
                product=product,
                price=option_best_ask,
                volume=buy_limit,
                buy_limit=buy_limit,
            )

        # Close
        if self.ENABLE_CLOSE and not orders:
            if position > 0 and sell_signal >= self.CLOSE_THRESHOLD and sell_limit > 0:
                sell_limit = self.add_sell_order(
                    orders=orders,
                    product=product,
                    price=option_best_bid,
                    volume=position,
                    sell_limit=sell_limit,
                )

            elif position < 0 and buy_signal <= -self.CLOSE_THRESHOLD and buy_limit > 0:
                buy_limit = self.add_buy_order(
                    orders=orders,
                    product=product,
                    price=option_best_ask,
                    volume=-position,
                    buy_limit=buy_limit,
                )

        return orders

    def run(self, state: TradingState, day_num: int):
        original_state = copy.deepcopy(state)

        traderObject = {}

        if state.traderData is not None and state.traderData != "":
            try:
                traderObject = jsonpickle.decode(state.traderData)
            except Exception:
                traderObject = {}

        result: dict[Symbol, List[Order]] = {}

        rolling_coeffs, underlying_mid, T = self.update_smile_history_and_fit(
            state=state,
            traderObject=traderObject,
            day_num=day_num,
        )

        for product in state.order_depths:
            orders: List[Order] = []

            if product in self.DEEP_ITM_VOUCHERS:
                orders = self.get_deep_itm_voucher_orders(product, state)

            elif product in self.ATM_VOUCHERS:
                orders = self.get_atm_voucher_orders(
                    product=product,
                    state=state,
                    traderObject=traderObject,
                    rolling_coeffs=rolling_coeffs,
                    underlying_mid=underlying_mid,
                    T=T,
                )

            elif product == "HYDROGEL_PACK":
                pass

            elif product == "VELVETFRUIT_EXTRACT":
                pass

            result[product] = orders

        traderData = jsonpickle.encode(traderObject)
        conversions = 0

        logger.flush(original_state, result, conversions, traderData)

        return result, conversions, traderData