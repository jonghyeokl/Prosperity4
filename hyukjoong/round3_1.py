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
            co[p] = [o.bidPrice, o.askPrice, o.transportFees, o.exportTariff,
                     o.importTariff, o.sugarPrice, o.sunlightIndex]
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
                # partial pivot
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
        "VEV_5000": 6, "VEV_5100": 6, "VEV_5200": 6,
        "VEV_5300": 5, "VEV_5400": 5, "VEV_5500": 5,
    }

    # 실전 Round 3 시작 시 TTE = 5 days (problem spec).
    # 히스토리 day 2 가 TTE=6d 이었으니 실전은 그 하루 뒤.
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

    def get_best_bid_ask(self, od: OrderDepth):
        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None
        return best_bid, best_ask

    def get_best_valid_bid_ask(self, od: OrderDepth, valid_volume: int):
        best_valid_bid = None
        best_valid_ask = None
        for bp, bv in sorted(od.buy_orders.items(), reverse=True):
            if bv >= valid_volume:
                best_valid_bid = bp
                break
        for ap, av in sorted(od.sell_orders.items()):
            if -av >= valid_volume:
                best_valid_ask = ap
                break
        best_bid, best_ask = self.get_best_bid_ask(od)
        if best_valid_bid is None:
            best_valid_bid = best_bid
        if best_valid_ask is None:
            best_valid_ask = best_ask
        return best_valid_bid, best_valid_ask

    def get_valid_mid(self, od: OrderDepth, valid_volume: int) -> Optional[float]:
        b, a = self.get_best_valid_bid_ask(od, valid_volume)
        if b is not None and a is not None:
            return (b + a) / 2
        return None

    def add_buy_order(self, orders, product, price, volume, buy_limit):
        volume = min(volume, buy_limit)
        if volume > 0:
            orders.append(Order(product, int(price), int(volume)))
            buy_limit -= volume
        return buy_limit

    def add_sell_order(self, orders, product, price, volume, sell_limit):
        volume = min(volume, sell_limit)
        if volume > 0:
            orders.append(Order(product, int(price), -int(volume)))
            sell_limit -= volume
        return sell_limit

    def compute_tte_years(self, timestamp: int) -> float:
        elapsed = timestamp / 1_000_000.0
        tte_days = max(self.TTE_BASE_DAYS - elapsed, 1e-9)
        return tte_days / self.DAYS_PER_YEAR

    def ema_update(self, traderObject: dict, key: str, value: float, window: int) -> float:
        alpha = 2.0 / (window + 1.0)
        if key not in traderObject:
            traderObject[key] = value
            return value
        new_val = alpha * value + (1.0 - alpha) * traderObject[key]
        traderObject[key] = new_val
        return new_val

    # ==========================================================
    #  Rolling smile fit
    # ==========================================================
    def update_smile_history_and_fit(
        self, state: TradingState, traderObject: dict,
    ) -> Tuple[Optional[Tuple[float, float, float]], float, float]:
        """
        Per voucher history 에 (m, iv) 추가 후 전체 point 로 fit.
        Returns (coefs_or_None, S_mid, T). coefs 는 fit 실패/warmup 시 None.
        """
        if self.UNDERLYING not in state.order_depths:
            return None, 0.0, 0.0

        und_od = state.order_depths[self.UNDERLYING]
        und_vol = self.VALID_BID_ASK_VOLUME[self.UNDERLYING]
        S_mid = self.get_valid_mid(und_od, und_vol)
        if S_mid is None:
            return None, 0.0, 0.0
        T = self.compute_tte_years(state.timestamp)
        if T <= 0:
            return None, S_mid, T

        # history state
        hist_key = "smile_hist"
        if hist_key not in traderObject:
            traderObject[hist_key] = {p: [] for p in self.ATM_VOUCHERS}
        hist = traderObject[hist_key]
        # ensure all keys present (for safety on trader data restore)
        for p in self.ATM_VOUCHERS:
            if p not in hist:
                hist[p] = []

        # push new (m, iv) for each voucher
        sqrtT = math.sqrt(T) if T > 0 else 1.0
        for product, K in self.ATM_VOUCHERS.items():
            if product not in state.order_depths:
                continue
            od = state.order_depths[product]
            opt_vol = self.VALID_BID_ASK_VOLUME.get(product, 6)
            mid = self.get_valid_mid(od, opt_vol)
            if mid is None:
                continue
            iv = implied_vol(mid, S_mid, K, T)
            if iv is None:
                continue
            m = math.log(K / S_mid) / sqrtT
            hist[product].append((m, iv))
            if len(hist[product]) > self.SMILE_WINDOW_PER_VOUCHER:
                # drop oldest
                hist[product] = hist[product][-self.SMILE_WINDOW_PER_VOUCHER:]

        # aggregate all points
        all_points = []
        for p in self.ATM_VOUCHERS:
            all_points.extend(hist[p])

        if len(all_points) < self.SMILE_MIN_POINTS_FOR_FIT:
            return None, S_mid, T

        coefs = fit_quadratic_from_points(all_points)
        return coefs, S_mid, T

    def fair_iv(self, m: float, rolling_coefs: Optional[Tuple[float, float, float]]) -> float:
        if rolling_coefs is not None:
            a, b, c = rolling_coefs
        else:
            a, b, c = self.IV_A_FALLBACK, self.IV_B_FALLBACK, self.IV_C_FALLBACK
        return a * m * m + b * m + c

    # ==========================================================
    def get_atm_orders(self, product: str, state: TradingState,
                       traderObject: dict,
                       rolling_coefs: Optional[Tuple[float, float, float]],
                       S_mid: float, T: float) -> List[Order]:
        orders: List[Order] = []
        option_od = state.order_depths[product]

        o_bid, o_ask = self.get_best_bid_ask(option_od)
        if o_bid is None or o_ask is None:
            return orders
        if S_mid <= 0 or T <= 0:
            return orders

        opt_vol = self.VALID_BID_ASK_VOLUME.get(product, 6)
        option_mid = self.get_valid_mid(option_od, opt_vol)
        if option_mid is None:
            return orders

        K = self.ATM_VOUCHERS[product]
        sqrtT = math.sqrt(T)
        m = math.log(K / S_mid) / sqrtT

        sigma = self.fair_iv(m, rolling_coefs)
        if sigma <= 0 or not math.isfinite(sigma):
            return orders

        theo, _delta, vega = bs_call_with_greeks(S_mid, K, T, sigma)

        theo_diff = option_mid - theo
        mean_key = f"{product}_mean"
        scale_key = f"{product}_scale"
        count_key = f"{product}_count"

        mean_diff = self.ema_update(traderObject, mean_key, theo_diff, self.THEO_DIFF_WINDOW)
        abs_dev = abs(theo_diff - mean_diff)
        scale = self.ema_update(traderObject, scale_key, abs_dev, self.SCALE_WINDOW)

        traderObject[count_key] = traderObject.get(count_key, 0) + 1
        count = traderObject[count_key]
        if count < self.WARMUP_TICKS:
            return orders
        if scale < self.MIN_SCALE:
            return orders

        sell_signal = o_bid - theo - mean_diff
        buy_signal = o_ask - theo - mean_diff

        low_vega_adj = self.LOW_VEGA_PENALTY if vega <= self.LOW_VEGA_CUTOFF else 0.0
        open_thr = self.OPEN_THRESHOLD + low_vega_adj

        position = state.position.get(product, 0)
        limit = self.POSITION_LIMITS[product]
        buy_limit = limit - position
        sell_limit = limit + position

        # Open
        if sell_signal >= open_thr and sell_limit > 0:
            sell_limit = self.add_sell_order(
                orders, product, o_bid, sell_limit, sell_limit,
            )
        elif buy_signal <= -open_thr and buy_limit > 0:
            buy_limit = self.add_buy_order(
                orders, product, o_ask, buy_limit, buy_limit,
            )

        # Close
        if self.ENABLE_CLOSE and not orders:
            if position > 0 and sell_signal >= self.CLOSE_THRESHOLD and sell_limit > 0:
                sell_limit = self.add_sell_order(
                    orders, product, o_bid, position, sell_limit,
                )
            elif position < 0 and buy_signal <= -self.CLOSE_THRESHOLD and buy_limit > 0:
                buy_limit = self.add_buy_order(
                    orders, product, o_ask, -position, buy_limit,
                )

        return orders

    def run(self, state: TradingState):
        original_state = copy.deepcopy(state)

        traderObject = {}
        if state.traderData is not None and state.traderData != "":
            try:
                traderObject = jsonpickle.decode(state.traderData)
            except Exception:
                traderObject = {}

        result: dict = {}

        # ----- 공통: rolling smile fit -----
        rolling_coefs, S_mid, T = self.update_smile_history_and_fit(
            state, traderObject,
        )

        # ----- ATM voucher 별 signal -----
        for product in self.ATM_VOUCHERS:
            if product not in state.order_depths:
                continue
            orders = self.get_atm_orders(
                product, state, traderObject,
                rolling_coefs, S_mid, T,
            )
            if orders:
                result[product] = orders

        traderData = jsonpickle.encode(traderObject)
        conversions = 0
        logger.flush(original_state, result, conversions, traderData)
        return result, conversions, traderData