from typing import List, Any
import json
import jsonpickle
import math
import copy
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


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

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
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

    DEEP_ITM_VOUCHERS = {
        "VEV_4000": 4000,
        "VEV_4500": 4500,
    }

    VALID_BID_ASK_VOLUME = {
        "VELVETFRUIT_EXTRACT": 30,
        "VEV_4000": 5,
        "VEV_4500": 5,
        "VEV_5000": 13,
        "VEV_5100": 13,
        "VEV_5200": 13,
        "VEV_5300": 13,
        "VEV_5400": 13,
        "VEV_5500": 13,
        "VEV_6000": 5,
        "VEV_6500": 5,
    }

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

    def get_mid_price(self, order_depth: OrderDepth):
        best_bid, best_ask = self.get_best_bid_ask(order_depth)
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2
        return None

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

        # 1. Take: 비싼 bid에는 sell
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

        # 2. Take: 싼 ask에는 buy
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

        # 3. Make: fair 이하에서 bid
        buy_price = min(best_bid + 1, math.floor(fair - 0.1))
        if buy_limit > 0 and buy_price < best_ask:
            buy_limit = self.add_buy_order(
                orders=orders,
                product=product,
                price=buy_price,
                volume=buy_limit,
                buy_limit=buy_limit,
            )

        # 4. Make: fair 이상에서 ask
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
                orders = self.get_deep_itm_voucher_orders(product, state)

            elif product == "VELVETFRUIT_EXTRACT":
                pass

            elif product == "HYDROGEL_PACK":
                pass

            result[product] = orders

        traderData = jsonpickle.encode(traderObject)
        conversions = 0

        logger.flush(original_state, result, conversions, traderData)

        return result, conversions, traderData