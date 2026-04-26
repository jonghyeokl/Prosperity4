from datamodel import OrderDepth, TradingState, Order
from typing import List, Any, Dict, Optional, Tuple
import json
import jsonpickle
import math
import copy

from datamodel import Listing, Observation, ProsperityEncoder, Symbol, Trade


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, List[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]
        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )
        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])
        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value
        return value[: max_length - 3] + "..."


logger = Logger()


class Trader:
    POSITION_LIMITS = {
        "HYDROGEL_PACK": 200,
        "VELVETFRUIT_EXTRACT": 200,
        "VEV_4000": 300,
        "VEV_4500": 300,
        "VEV_5000": 300,
        "VEV_5100": 300,
        "VEV_5200": 300,
        "VEV_5300": 300,
        "VEV_5400": 300,
        "VEV_5500": 300,
        "VEV_6000": 300,
        "VEV_6500": 300,
    }

    UNDERLYING = "VELVETFRUIT_EXTRACT"

    VOUCHER_STRIKES = {
        "VEV_4000": 4000,
        "VEV_4500": 4500,
        "VEV_5000": 5000,
        "VEV_5100": 5100,
        "VEV_5200": 5200,
        "VEV_5300": 5300,
        "VEV_5400": 5400,
        "VEV_5500": 5500,
        "VEV_6000": 6000,
        "VEV_6500": 6500,
    }

    # 처음에는 ATM 근처만 실험
    ACTIVE_VOUCHERS = {
        "VEV_5000",
        "VEV_5100",
        "VEV_5200",
        "VEV_5300",
        "VEV_5400",
        "VEV_5500",
    }

    # 실험 파라미터
    WINDOW = 100
    MIN_HISTORY = 20
    IV_EDGE = 0.005

    # 진단용이라 작게
    TAKE_SIZE = 1
    TEST_POSITION_LIMIT = 30

    ENABLE_TRADING = True
    ENABLE_DEBUG_LOG = True
    DEBUG_INTERVAL = 1000

    # IV binary search
    MIN_IV = 0.001
    MAX_IV = 2.0
    IV_SEARCH_ITER = 35

    def run(self, state: TradingState, day_num: int = 3):
        original_state = copy.deepcopy(state)

        traderObject = {}
        if state.traderData is not None and state.traderData != "":
            try:
                traderObject = jsonpickle.decode(state.traderData)
            except Exception:
                traderObject = {}

        if "iv_history" not in traderObject:
            traderObject["iv_history"] = {}

        result = {}

        # 실제 product 목록 확인
        if self.ENABLE_DEBUG_LOG and state.timestamp == 0:
            logger.print("PRODUCTS", list(state.order_depths.keys()), "day_num", day_num, "TTE", self.get_tte(day_num))

        S = None
        if self.UNDERLYING in state.order_depths:
            S = self.get_mid_price(state.order_depths[self.UNDERLYING])

        if self.ENABLE_DEBUG_LOG and state.timestamp == 0:
            logger.print("UNDERLYING", self.UNDERLYING, "S", S)

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            if product in self.ACTIVE_VOUCHERS and S is not None:
                orders = self.trade_voucher_simple_iv(
                    product=product,
                    order_depth=order_depth,
                    state=state,
                    traderObject=traderObject,
                    S=S,
                    T=self.get_tte(day_num),
                )

            result[product] = orders

        traderData = jsonpickle.encode(traderObject)
        conversions = 0

        logger.flush(original_state, result, conversions, traderData)
        return result, conversions, traderData

    # -------------------------------------------------------
    # Basic helpers
    # -------------------------------------------------------

    def get_tte(self, day_num: int) -> float:
        # backtest: day 0/1/2 => 8/7/6
        # submission: day 3 => 5
        return max(8.0 - float(day_num), 0.05)

    def get_best_bid_ask(self, order_depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        return best_bid, best_ask

    def get_mid_price(self, order_depth: OrderDepth) -> Optional[float]:
        best_bid, best_ask = self.get_best_bid_ask(order_depth)
        if best_bid is None or best_ask is None:
            return None
        return (best_bid + best_ask) / 2.0

    def norm_cdf(self, x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def black_scholes_call(self, S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
        if S <= 0 or K <= 0:
            return 0.0

        intrinsic = max(S - K, 0.0)

        if T <= 0 or sigma <= 0:
            return intrinsic

        try:
            sqrt_T = math.sqrt(T)
            d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
            d2 = d1 - sigma * sqrt_T

            price = S * self.norm_cdf(d1) - K * math.exp(-r * T) * self.norm_cdf(d2)
            return max(price, intrinsic)
        except Exception:
            return intrinsic

    def implied_vol_call(self, target_price: float, S: float, K: float, T: float) -> Optional[float]:
        if S <= 0 or K <= 0 or T <= 0:
            return None

        intrinsic = max(S - K, 0.0)

        # 너무 낮으면 IV 역산 불가능
        if target_price < intrinsic - 1.0:
            return None

        # 거의 intrinsic이면 낮은 IV로 처리
        if abs(target_price - intrinsic) <= 1.0:
            return self.MIN_IV

        low = self.MIN_IV
        high = self.MAX_IV

        price_low = self.black_scholes_call(S, K, T, low)
        price_high = self.black_scholes_call(S, K, T, high)

        if target_price < price_low - 1.0:
            return None

        if target_price > price_high + 1.0:
            return None

        for _ in range(self.IV_SEARCH_ITER):
            mid = (low + high) / 2.0
            price_mid = self.black_scholes_call(S, K, T, mid)

            if price_mid < target_price:
                low = mid
            else:
                high = mid

        return (low + high) / 2.0

    # -------------------------------------------------------
    # IV rolling mean strategy
    # -------------------------------------------------------

    def update_iv_history(self, traderObject: Dict[str, Any], product: str, iv: float):
        hist_map = traderObject["iv_history"]

        if product not in hist_map:
            hist_map[product] = []

        hist = hist_map[product]
        hist.append(float(iv))

        max_keep = self.WINDOW + 20
        if len(hist) > max_keep:
            del hist[: len(hist) - max_keep]

        if len(hist) < self.MIN_HISTORY:
            return None, len(hist)

        window = hist[-self.WINDOW:]
        avg = sum(window) / len(window)

        return avg, len(hist)

    def trade_voucher_simple_iv(
        self,
        product: str,
        order_depth: OrderDepth,
        state: TradingState,
        traderObject: Dict[str, Any],
        S: float,
        T: float,
    ) -> List[Order]:
        orders: List[Order] = []

        best_bid, best_ask = self.get_best_bid_ask(order_depth)
        option_mid = self.get_mid_price(order_depth)

        if best_bid is None or best_ask is None or option_mid is None:
            return orders

        K = self.VOUCHER_STRIKES[product]

        current_iv = self.implied_vol_call(
            target_price=option_mid,
            S=S,
            K=K,
            T=T,
        )

        if current_iv is None:
            if self.ENABLE_DEBUG_LOG and state.timestamp % self.DEBUG_INTERVAL == 0:
                intrinsic = max(S - K, 0.0)
                logger.print(
                    "IV_NONE",
                    state.timestamp,
                    product,
                    "S", round(S, 1),
                    "K", K,
                    "T", round(T, 2),
                    "mid", round(option_mid, 1),
                    "intr", round(intrinsic, 1),
                    "bid", best_bid,
                    "ask", best_ask,
                )
            return orders

        avg_iv, hist_len = self.update_iv_history(traderObject, product, current_iv)

        if avg_iv is None:
            if self.ENABLE_DEBUG_LOG and state.timestamp % self.DEBUG_INTERVAL == 0:
                logger.print(
                    "WAIT",
                    state.timestamp,
                    product,
                    "hist", hist_len,
                    "S", round(S, 1),
                    "K", K,
                    "T", round(T, 2),
                    "iv", round(current_iv, 4),
                    "mid", round(option_mid, 1),
                )
            return orders

        iv_diff = current_iv - avg_iv

        position = state.position.get(product, 0)
        limit = min(self.POSITION_LIMITS[product], self.TEST_POSITION_LIMIT)

        buy_remaining = limit - position
        sell_remaining = limit + position

        side = "HOLD"
        qty = 0
        px = None

        # 현재 IV가 평균보다 낮다 -> 평소보다 싸다 -> best ask 매수
        if current_iv < avg_iv - self.IV_EDGE and buy_remaining > 0:
            ask_qty = -order_depth.sell_orders[best_ask]
            qty = min(self.TAKE_SIZE, ask_qty, buy_remaining)

            if qty > 0:
                side = "BUY"
                px = best_ask
                if self.ENABLE_TRADING:
                    orders.append(Order(product, best_ask, qty))

        # 현재 IV가 평균보다 높다 -> 평소보다 비싸다 -> best bid 매도
        elif current_iv > avg_iv + self.IV_EDGE and sell_remaining > 0:
            bid_qty = order_depth.buy_orders[best_bid]
            qty = min(self.TAKE_SIZE, bid_qty, sell_remaining)

            if qty > 0:
                side = "SELL"
                px = best_bid
                if self.ENABLE_TRADING:
                    orders.append(Order(product, best_bid, -qty))

        # 디버깅: 일정 간격으로 상태 로그
        if self.ENABLE_DEBUG_LOG and state.timestamp % self.DEBUG_INTERVAL == 0:
            logger.print(
                "STAT",
                state.timestamp,
                product,
                "pos", position,
                "S", round(S, 1),
                "K", K,
                "T", round(T, 2),
                "bid", best_bid,
                "ask", best_ask,
                "mid", round(option_mid, 1),
                "iv", round(current_iv, 4),
                "avg", round(avg_iv, 4),
                "diff", round(iv_diff, 4),
                "hist", hist_len,
                "side", side,
                "px", px,
                "qty", qty,
            )

        # 거래 발생 시 로그
        if self.ENABLE_DEBUG_LOG and side != "HOLD":
            logger.print(
                "SIG",
                state.timestamp,
                product,
                side,
                "pos", position,
                "S", round(S, 1),
                "K", K,
                "T", round(T, 2),
                "bid", best_bid,
                "ask", best_ask,
                "mid", round(option_mid, 1),
                "iv", round(current_iv, 4),
                "avg", round(avg_iv, 4),
                "diff", round(iv_diff, 4),
                "px", px,
                "qty", qty,
            )

        return orders