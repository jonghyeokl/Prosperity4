from __future__ import annotations

from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Any, Dict, Optional, Tuple
from pathlib import Path
import json
import jsonpickle
import math
import copy
import os
import sys
from datamodel import (
    Listing, Observation, Order, OrderDepth, ProsperityEncoder,
    Symbol, Trade, TradingState,
)


# ============================================================
#  Cached theo/theo_diff by SMILE_WINDOW_PER_VOUCHER
# ============================================================
THEO_DIFF_CACHE_BY_SMILE_WINDOW_PATH = Path(
    "jonghyeok/data_analysis/output/round3_theo_diff_cache_by_smile_window.json"
)


def load_precomputed_theo_features_by_window() -> dict:
    payload = json.loads(THEO_DIFF_CACHE_BY_SMILE_WINDOW_PATH.read_text())
    features_by_window = payload.get("features_by_smile_window", {})

    if not features_by_window:
        raise RuntimeError("PRECOMPUTED_THEO_FEATURES_BY_WINDOW is empty")

    return features_by_window


PRECOMPUTED_THEO_FEATURES_BY_WINDOW = load_precomputed_theo_features_by_window()


def cached_feature_key(day_num: int, timestamp: int, product: str) -> str:
    return f"{day_num}|{timestamp}|{product}"


# ============================================================
#  Fast-test parameter loading
# ============================================================
DEFAULT_SMILE_WINDOW_PER_VOUCHER = 300

DEFAULT_MEAN_REVERSION_PARAMS = {
    "VEV_5000": {"ema_window": 50, "beta": -1.003},
    "VEV_5100": {"ema_window": 50, "beta": -0.998},
    "VEV_5200": {"ema_window": 30, "beta": -0.993},
    "VEV_5300": {"ema_window": 20, "beta": -0.986},
}


def _read_arg_value(name: str) -> Optional[str]:
    """
    Non-invasive argv reader.
    The prosperity backtester usually owns argv, so the sweep script uses env vars.
    This exists only for direct local calls/tests.
    Supported forms: --name value, --name=value
    """
    prefix = f"--{name}"
    for i, arg in enumerate(sys.argv):
        if arg == prefix and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith(prefix + "="):
            return arg.split("=", 1)[1]
    return None


def _read_config_value(name: str) -> Optional[str]:
    env_name = "HJ_" + name.upper()
    if env_name in os.environ:
        return os.environ[env_name]
    return _read_arg_value(name.lower())


def _load_smile_window() -> int:
    raw = _read_config_value("SMILE_WINDOW")
    if raw is None:
        return DEFAULT_SMILE_WINDOW_PER_VOUCHER
    return int(raw)


def _load_mean_reversion_params() -> Dict[str, Dict[str, float]]:
    params: Dict[str, Dict[str, float]] = json.loads(json.dumps(DEFAULT_MEAN_REVERSION_PARAMS))

    # Full JSON override, e.g.
    # HJ_MR_PARAMS='{"VEV_5000":{"ema_window":40,"beta":-1.0}, ...}'
    raw_json = _read_config_value("MR_PARAMS")
    if raw_json:
        override = json.loads(raw_json)
        for product, cfg in override.items():
            if product not in params:
                params[product] = {}
            if "ema_window" in cfg:
                params[product]["ema_window"] = int(cfg["ema_window"])
            if "beta" in cfg:
                params[product]["beta"] = float(cfg["beta"])

    # Single product override, useful for sweep.
    # HJ_TARGET_PRODUCT=VEV_5000 HJ_EMA_WINDOW=40 HJ_BETA=-1.0
    target = _read_config_value("TARGET_PRODUCT")
    ema_window = _read_config_value("EMA_WINDOW")
    beta = _read_config_value("BETA")

    if target:
        if target not in params:
            params[target] = {}
        if ema_window is not None:
            params[target]["ema_window"] = int(ema_window)
        if beta is not None:
            params[target]["beta"] = float(beta)

    return params


# ============================================================
#  Logger
# ============================================================
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(
        self,
        state: TradingState,
        orders: dict[Symbol, list[Order]],
        conversions: int,
        trader_data: str,
    ) -> None:
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


# ============================================================
#  Trader
# ============================================================
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

    # 실제 거래 대상: VEV_5000~VEV_5300
    ATM_VOUCHERS = {
        "VEV_5000": 5000,
        "VEV_5100": 5100,
        "VEV_5200": 5200,
        "VEV_5300": 5300,
    }

    # cached path에서는 이 값으로 사용할 precomputed theo/theo_diff cache를 선택합니다.
    # sweep 시 HJ_SMILE_WINDOW env var 또는 --smile_window로 override 가능합니다.
    # cache 생성 코드는 SMILE_MIN_POINTS_FOR_FIT = SMILE_WINDOW_PER_VOUCHER * 2 로 계산합니다.
    SMILE_WINDOW_PER_VOUCHER = _load_smile_window()
    SMILE_MIN_POINTS_FOR_FIT = SMILE_WINDOW_PER_VOUCHER * 2

    DEEP_ITM_VOUCHERS = {
        "VEV_4000": 4000,
        "VEV_4500": 4500,
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

    DAYS_PER_YEAR = 365.0
    THEO_DIFF_WINDOW = 20
    WARMUP_TICKS = 30
    ENABLE_MAKE = True

    # EMA_t 기준 beta 평균회귀 파라미터.
    # sweep 시 HJ_MR_PARAMS 또는 HJ_TARGET_PRODUCT/HJ_EMA_WINDOW/HJ_BETA로 override 가능합니다.
    MEAN_REVERSION_PARAMS = _load_mean_reversion_params()

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
    #  현재 run()에서는 비활성화 상태입니다.
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

        buy_price = min(best_bid + 1, math.floor(fair - 0.1))

        if buy_limit > 0 and buy_price < best_ask:
            buy_limit = self.add_buy_order(
                orders=orders,
                product=product,
                price=buy_price,
                volume=buy_limit,
                buy_limit=buy_limit,
            )

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
    #  Velvetfruit trading: fair = VEV_4000 + 4000
    # ==========================================================
    def get_velvetfruit_against_vev4000_orders(self, state: TradingState) -> List[Order]:
        orders: List[Order] = []

        product = self.UNDERLYING
        voucher = "VEV_4000"
        strike = 4000

        if product not in state.order_depths or voucher not in state.order_depths:
            return orders

        underlying_depth = state.order_depths[product]
        voucher_depth = state.order_depths[voucher]

        voucher_mid = self.get_valid_mid_price(
            voucher_depth,
            self.VALID_BID_ASK_VOLUME.get(voucher, 6),
        )

        if voucher_mid is None:
            return orders

        fair = voucher_mid + strike

        if fair <= 0:
            return orders

        best_bid, best_ask = self.get_best_bid_ask(underlying_depth)

        if best_bid is None or best_ask is None:
            return orders

        position = state.position.get(product, 0)
        limit = self.POSITION_LIMITS[product]

        buy_limit = limit - position
        sell_limit = limit + position

        # 1. Take: Velvetfruit bid가 VEV_4000 + 4000보다 비싸면 sell
        for bid_price, bid_vol in sorted(underlying_depth.buy_orders.items(), reverse=True):
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

        # 2. Take: Velvetfruit ask가 VEV_4000 + 4000보다 싸면 buy
        for ask_price, ask_vol in sorted(underlying_depth.sell_orders.items()):
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
    #  Cached ATM voucher trading
    # ==========================================================
    def get_cached_atm_voucher_orders(
        self,
        product: str,
        state: TradingState,
        traderObject: dict,
        day_num: int,
    ) -> List[Order]:
        orders: List[Order] = []

        window_key = str(self.SMILE_WINDOW_PER_VOUCHER)
        window_features = PRECOMPUTED_THEO_FEATURES_BY_WINDOW.get(window_key)

        if window_features is None:
            raise RuntimeError(
                f"No theo/theo_diff cache for SMILE_WINDOW_PER_VOUCHER={self.SMILE_WINDOW_PER_VOUCHER}. "
                f"Available windows: {sorted(PRECOMPUTED_THEO_FEATURES_BY_WINDOW.keys(), key=int)}"
            )

        key = cached_feature_key(day_num, state.timestamp, product)
        feature = window_features.get(key)

        if feature is None:
            return orders

        if product not in state.order_depths:
            return orders

        option_depth = state.order_depths[product]
        option_best_bid, option_best_ask = self.get_best_bid_ask(option_depth)

        if option_best_bid is None or option_best_ask is None:
            return orders

        theo = float(feature["theo"])
        theo_diff = float(feature["theo_diff"])

        if not math.isfinite(theo) or not math.isfinite(theo_diff):
            return orders

        params = self.MEAN_REVERSION_PARAMS.get(
            product,
            {"ema_window": self.THEO_DIFF_WINDOW, "beta": -1.0},
        )
        ema_window = int(params["ema_window"])
        beta = float(params["beta"])

        mean_key = f"{product}_mean"
        count_key = f"{product}_count"

        mean_diff = self.ema(
            traderObject=traderObject,
            key=mean_key,
            value=theo_diff,
            window=ema_window,
        )

        residual = theo_diff - mean_diff
        expected_diff = theo_diff + beta * residual
        fair_price = theo + expected_diff

        traderObject[count_key] = traderObject.get(count_key, 0) + 1
        count = traderObject[count_key]

        if count < self.WARMUP_TICKS:
            return orders

        position = state.position.get(product, 0)
        limit = self.POSITION_LIMITS[product]

        buy_limit = limit - position
        sell_limit = limit + position

        # Take sell
        for bid_price, bid_vol in sorted(option_depth.buy_orders.items(), reverse=True):
            if bid_price < fair_price or sell_limit <= 0:
                break

            sell_qty = min(bid_vol, sell_limit)

            if sell_qty > 0:
                orders.append(Order(product, bid_price, -sell_qty))
                position -= sell_qty
                sell_limit -= sell_qty

        # Take buy
        for ask_price, ask_vol in sorted(option_depth.sell_orders.items()):
            if ask_price > fair_price or buy_limit <= 0:
                break

            buy_qty = min(-ask_vol, buy_limit)

            if buy_qty > 0:
                orders.append(Order(product, ask_price, buy_qty))
                position += buy_qty
                buy_limit -= buy_qty

        # Market make / passive unwind
        if self.ENABLE_MAKE:
            sell_price = max(option_best_ask - 1, math.ceil(fair_price))
            buy_price = min(option_best_bid + 1, math.floor(fair_price))

            if position > 0 and sell_limit > 0:
                sell_qty = min(sell_limit, position)

                if sell_qty > 0:
                    orders.append(Order(product, sell_price, -sell_qty))
                    sell_limit -= sell_qty

            elif position < 0 and buy_limit > 0:
                buy_qty = min(buy_limit, -position)

                if buy_qty > 0:
                    orders.append(Order(product, buy_price, buy_qty))
                    buy_limit -= buy_qty

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

        for product in state.order_depths:
            orders: List[Order] = []

            if product in self.DEEP_ITM_VOUCHERS:
                # Deep ITM trading disabled.
                # orders = self.get_deep_itm_voucher_orders(product, state)
                pass

            elif product in self.ATM_VOUCHERS:
                orders = self.get_cached_atm_voucher_orders(
                    product=product,
                    state=state,
                    traderObject=traderObject,
                    day_num=day_num,
                )

            elif product == "HYDROGEL_PACK":
                pass

            elif product == "VELVETFRUIT_EXTRACT":
                orders = self.get_velvetfruit_against_vev4000_orders(state)

            result[product] = orders

        traderData = jsonpickle.encode(traderObject)
        conversions = 0

        logger.flush(original_state, result, conversions, traderData)

        return result, conversions, traderData
