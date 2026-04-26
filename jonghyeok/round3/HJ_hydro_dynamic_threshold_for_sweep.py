import json
import copy
import jsonpickle
import math
import os
from statistics import NormalDist
from datamodel import OrderDepth, TradingState, Order, Symbol, Listing, Trade, Observation, ProsperityEncoder
from typing import List, Any


NORMAL = NormalDist()


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default


# ============================================================
#  Logger
# ============================================================
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
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
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])
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
        lo, hi = 0, min(len(value), max_length)
        out = ""

        while lo <= hi:
            mid = (lo + hi) // 2

            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."

            encoded_candidate = json.dumps(candidate)

            if len(encoded_candidate) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1

        return out


logger = Logger()


class Trader:
    PRODUCT = "VELVETFRUIT_EXTRACT"

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

    VALID_BID_ASK_VOLUME = {
        "HYDROGEL_PACK": 10,
        "VELVETFRUIT_EXTRACT": 15,
        "VEV_4000": 6,
        "VEV_4500": 6,
        "VEV_5000": 6,
        "VEV_5100": 6,
        "VEV_5200": 6,
        "VEV_5300": 5,
        "VEV_5400": 5,
        "VEV_5500": 5,
        "VEV_6000": 5,
        "VEV_6500": 5,
    }

    ENABLE_MAKE = True

    # sweep parameters
    VALID_MID_HISTORY_LENGTH = env_int("HJ_VALID_MID_HISTORY_LENGTH", 1000)
    THRESHOLD_PARAM_ALPHA = env_float("HJ_THRESHOLD_PARAM_ALPHA", 0.6)
    THRESHOLD_PARAM_BETA = env_float("HJ_THRESHOLD_PARAM_BETA", 0.25)
    MIN_STD = env_float("HJ_MIN_STD", 1e-9)

    P_EPS = 1e-6

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
            return (best_valid_bid + best_valid_ask) / 2.0

        return None

    def _threshold_from_tail_prob(self, tail_prob: float) -> float:
        # threshold = Phi^{-1}(1 - tail_prob)
        p = max(self.P_EPS, min(1.0 - self.P_EPS, tail_prob))
        return NORMAL.inv_cdf(1.0 - p)

    def get_thresholds(self, position: int, limit: int):
        if limit <= 0:
            return 0.0, 0.0

        beta = self.THRESHOLD_PARAM_BETA
        alpha = self.THRESHOLD_PARAM_ALPHA

        if position >= 0:
            ratio = max(0.0, min(1.0, position / limit))
            p_buy = beta * pow(1.0 - ratio, alpha)
            p_sell = 2.0 * beta - p_buy
        else:
            ratio = max(0.0, min(1.0, abs(position) / limit))
            p_sell = beta * pow(1.0 - ratio, alpha)
            p_buy = 2.0 * beta - p_sell

        buy_threshold = self._threshold_from_tail_prob(p_buy)
        sell_threshold = self._threshold_from_tail_prob(p_sell)
        return buy_threshold, sell_threshold

    def add_take_and_make_orders(
        self,
        *,
        product: str,
        order_depth: OrderDepth,
        state: TradingState,
        fair_value: float,
        std: float,
    ) -> List[Order]:
        orders: List[Order] = []

        best_bid, best_ask = self.get_best_bid_ask(order_depth)
        if best_bid is None or best_ask is None:
            return orders

        position = state.position.get(product, 0)
        limit = self.POSITION_LIMITS[product]
        buy_limit = limit - position
        sell_limit = limit + position

        # Take sell
        for bid_price, bid_vol in sorted(order_depth.buy_orders.items(), reverse=True):
            _buy_threshold, sell_threshold = self.get_thresholds(position, limit)
            if (bid_price - fair_value) / std <= sell_threshold or sell_limit <= 0:
                break

            sell_qty = min(bid_vol, sell_limit)
            if sell_qty > 0:
                orders.append(Order(product, int(bid_price), -int(sell_qty)))
                position -= sell_qty
                sell_limit -= sell_qty

        # Take buy
        for ask_price, ask_vol in sorted(order_depth.sell_orders.items()):
            buy_threshold, _sell_threshold = self.get_thresholds(position, limit)
            if (ask_price - fair_value) / std >= -buy_threshold or buy_limit <= 0:
                break

            buy_qty = min(-ask_vol, buy_limit)
            if buy_qty > 0:
                orders.append(Order(product, int(ask_price), int(buy_qty)))
                position += buy_qty
                buy_limit -= buy_qty

        # Make
        if self.ENABLE_MAKE:
            buy_threshold, sell_threshold = self.get_thresholds(position, limit)

            sell_price = max(best_ask - 1, math.ceil(fair_value + sell_threshold * std))
            buy_price = min(best_bid + 1, math.floor(fair_value - buy_threshold * std))

            if sell_limit > 0:
                orders.append(Order(product, int(sell_price), -int(sell_limit)))

            if buy_limit > 0:
                orders.append(Order(product, int(buy_price), int(buy_limit)))

        return orders

    def get_hydrogel_orders(self, state: TradingState, traderObject: dict) -> List[Order]:
        product = self.PRODUCT
        if product not in state.order_depths:
            return []

        depth = state.order_depths[product]
        valid_mid = self.get_valid_mid_price(depth, self.VALID_BID_ASK_VOLUME[product])
        if valid_mid is None:
            return []

        key = "hydrogel_valid_mid_history"
        history = traderObject.get(key, [])
        history.append(float(valid_mid))
        history = history[-self.VALID_MID_HISTORY_LENGTH:]
        traderObject[key] = history

        if len(history) < self.VALID_MID_HISTORY_LENGTH:
            return []

        fair_value = sum(history) / len(history)
        var = sum((x - fair_value) ** 2 for x in history) / len(history)
        std = math.sqrt(var)

        if std <= self.MIN_STD or not math.isfinite(std):
            return []

        return self.add_take_and_make_orders(
            product=product,
            order_depth=depth,
            state=state,
            fair_value=fair_value,
            std=std,
        )

    def run(self, state: TradingState, day_num: int):
        traderObject = {}
        original_state = copy.deepcopy(state)

        if state.traderData is not None and state.traderData != "":
            try:
                traderObject = jsonpickle.decode(state.traderData)
            except Exception:
                traderObject = {}

        result = {}
        for product in state.order_depths:
            if product == self.PRODUCT:
                result[product] = self.get_hydrogel_orders(state, traderObject)
            else:
                result[product] = []

        traderData = jsonpickle.encode(traderObject)
        conversions = 0

        logger.flush(original_state, result, conversions, traderData)

        return result, conversions, traderData
