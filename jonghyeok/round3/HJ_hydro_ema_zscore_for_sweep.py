from datamodel import OrderDepth, TradingState, Order
from typing import List
import jsonpickle
import math
import os


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


class Trader:
    PRODUCT = "HYDROGEL_PACK"

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

    EMA_WINDOW = env_int("HJ_EMA_WINDOW", 50)
    Z_SCORE_THRESHOLD = env_float("HJ_Z_SCORE_THRESHOLD", env_float("HJ_THRESHOLD", 1.0))
    MIN_STD = env_float("HJ_MIN_STD", 1e-9)


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

    def add_take_and_make_orders(
        self,
        *,
        product: str,
        order_depth: OrderDepth,
        state: TradingState,
        fair_value: float,
        threshold: float,
    ) -> List[Order]:
        orders: List[Order] = []

        best_bid, best_ask = self.get_best_bid_ask(order_depth)
        if best_bid is None or best_ask is None:
            return orders

        position = state.position.get(product, 0)
        limit = self.POSITION_LIMITS[product]
        buy_limit = limit - position
        sell_limit = limit + position

        # Take sell: bid - fair_value > threshold
        for bid_price, bid_vol in sorted(order_depth.buy_orders.items(), reverse=True):
            if bid_price - fair_value <= threshold or sell_limit <= 0:
                break

            sell_qty = min(bid_vol, sell_limit)
            if sell_qty > 0:
                orders.append(Order(product, int(bid_price), -int(sell_qty)))
                position -= sell_qty
                sell_limit -= sell_qty

        # Take buy: ask - fair_value < -threshold
        for ask_price, ask_vol in sorted(order_depth.sell_orders.items()):
            if ask_price - fair_value >= -threshold or buy_limit <= 0:
                break

            buy_qty = min(-ask_vol, buy_limit)
            if buy_qty > 0:
                orders.append(Order(product, int(ask_price), int(buy_qty)))
                position += buy_qty
                buy_limit -= buy_qty

        # Make
        if self.ENABLE_MAKE:
            sell_price = max(best_ask - 1, math.ceil(fair_value + threshold))
            buy_price = min(best_bid + 1, math.floor(fair_value - threshold))

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

        x = float(valid_mid)
        alpha = 2.0 / (self.EMA_WINDOW + 1.0)
        ema_key = "hydrogel_ema"
        var_key = "hydrogel_ema_var"
        count_key = "hydrogel_ema_count"

        if ema_key not in traderObject:
            traderObject[ema_key] = x
            traderObject[var_key] = 0.0
            traderObject[count_key] = 1
        else:
            prev_ema = float(traderObject[ema_key])
            new_ema = alpha * x + (1.0 - alpha) * prev_ema
            dev = x - new_ema
            prev_var = float(traderObject.get(var_key, 0.0))
            new_var = alpha * dev * dev + (1.0 - alpha) * prev_var

            traderObject[ema_key] = new_ema
            traderObject[var_key] = new_var
            traderObject[count_key] = int(traderObject.get(count_key, 0)) + 1

        if int(traderObject[count_key]) < self.EMA_WINDOW:
            return []

        fair_value = float(traderObject[ema_key])
        std = math.sqrt(max(float(traderObject.get(var_key, 0.0)), 0.0))

        if std <= self.MIN_STD or not math.isfinite(std):
            return []

        threshold = self.Z_SCORE_THRESHOLD * std
        return self.add_take_and_make_orders(
            product=product,
            order_depth=depth,
            state=state,
            fair_value=fair_value,
            threshold=threshold,
        )


    def run(self, state: TradingState, day_num: int):
        traderObject = {}

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
        return result, 0, traderData
