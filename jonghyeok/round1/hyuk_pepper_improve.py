from datamodel import OrderDepth, UserId, TradingState, Order
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

    def run(self, state: TradingState):

        original_state = copy.deepcopy(state)

        traderData = original_state.traderData

        traderObject = {}
        if state.traderData is not None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            position = state.position.get(product, 0)
            limit = 80
            must_sell_ratio = 0.7
            # 기존 248558
            # 1 249298
            # 0.9 250734
            # 0.8 251905
            # 0.75 252003
            # 0.7 252026
            # 0.65 252026
            # 0.6 251698
            # 0.5 248660

            if product == "INTARIAN_PEPPER_ROOT":
                # initial state
                if "intarian_pepper_root_last_ask_price_history" not in traderObject:
                    traderObject["intarian_pepper_root_last_ask_price_history"] = []
                if "intarian_pepper_root_last_bid_price_history" not in traderObject:
                    traderObject["intarian_pepper_root_last_bid_price_history"] = []
                # for first 10 timestamps or position < half of limit, buy all of ask 1
                beginning_never_trade = False
                if len(traderObject["intarian_pepper_root_last_ask_price_history"]) < 2:
                    beginning_never_trade = True
                
                # get fair value
                avg_ask_price = sum(traderObject["intarian_pepper_root_last_ask_price_history"]) / len(traderObject["intarian_pepper_root_last_ask_price_history"]) if len(traderObject["intarian_pepper_root_last_ask_price_history"]) > 0 else 0
                fair_ask_value = (avg_ask_price + 0.1 * (len(traderObject["intarian_pepper_root_last_ask_price_history"]) + 1)/2 + 0.1) if avg_ask_price != 0 else 0
                avg_bid_price = sum(traderObject["intarian_pepper_root_last_bid_price_history"]) / len(traderObject["intarian_pepper_root_last_bid_price_history"]) if len(traderObject["intarian_pepper_root_last_bid_price_history"]) > 0 else 0
                fair_bid_value = (avg_bid_price + 0.1 * (len(traderObject["intarian_pepper_root_last_bid_price_history"]) + 1)/2 + 0.1) if avg_bid_price != 0 else 0
                must_sell_price = (fair_ask_value * must_sell_ratio + fair_bid_value * (1 - must_sell_ratio)) if (fair_ask_value != 0 and fair_bid_value != 0) else 0
                # calculate mid price
                best_ask_price = min(order_depth.sell_orders.keys()) if (order_depth.sell_orders and any(order_depth.sell_orders.values())) else 0
                best_bid_price = max(order_depth.buy_orders.keys()) if (order_depth.buy_orders and any(order_depth.buy_orders.values())) else 0
                # save last 199 prices
                if best_ask_price != 0:
                    if len(traderObject["intarian_pepper_root_last_ask_price_history"]) < 199:
                        traderObject["intarian_pepper_root_last_ask_price_history"].append(best_ask_price)
                    else:
                        traderObject["intarian_pepper_root_last_ask_price_history"].pop(0)
                        traderObject["intarian_pepper_root_last_ask_price_history"].append(best_ask_price)
                if best_bid_price != 0:
                    if len(traderObject["intarian_pepper_root_last_bid_price_history"]) < 199:
                        traderObject["intarian_pepper_root_last_bid_price_history"].append(best_bid_price)
                    else:
                        traderObject["intarian_pepper_root_last_bid_price_history"].pop(0)
                        traderObject["intarian_pepper_root_last_bid_price_history"].append(best_bid_price)
                # Trend: +1000/day linear. Buy and hold max long
                # 1000 / 1000000 timestamp => 0.1 / 100 timestamp
                if order_depth.sell_orders:
                    for ask_price, ask_vol in sorted(order_depth.sell_orders.items()):
                        if position < limit:
                            if not beginning_never_trade and ask_price <= fair_ask_value:
                                buy_qty = min(-ask_vol, limit - position)
                                if buy_qty > 0:
                                    orders.append(Order(product, ask_price, buy_qty))
                                    position += buy_qty
                if order_depth.buy_orders:
                    for bid_price, bid_vol in sorted(order_depth.buy_orders.items(), reverse=True):
                        if position > -limit:
                            if not beginning_never_trade and bid_price >= must_sell_price:
                                sell_qty = min(bid_vol, limit + position)
                                if sell_qty > 0:
                                    orders.append(Order(product, bid_price, -sell_qty))
                                    position -= sell_qty
                
                if position < limit and not beginning_never_trade:
                    best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
                    if best_bid is not None and best_bid + 1 <= fair_ask_value:
                        orders.append(Order(product, best_bid + 1, limit - position))
                
                if position > -limit and not beginning_never_trade:
                    best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
                    if best_ask is not None and best_ask - 1 >= must_sell_price:
                        orders.append(Order(product, best_ask - 1, -limit - position))

            elif product == "ASH_COATED_OSMIUM":
                # Mean-reversion: only cross spread at favorable prices
                # NO posted orders - only take mispriced orders

                # Buy cheap asks (<= 9998)
                if order_depth.sell_orders:
                    for ask_price, ask_vol in sorted(order_depth.sell_orders.items()):
                        if ask_price <= 9998 and position < limit:
                            buy_qty = min(-ask_vol, limit - position)
                            if buy_qty > 0:
                                orders.append(Order(product, ask_price, buy_qty))
                                position += buy_qty

                # Sell expensive bids (>= 10002)
                if order_depth.buy_orders:
                    for bid_price, bid_vol in sorted(order_depth.buy_orders.items(), reverse=True):
                        if bid_price >= 10002 and position > -limit:
                            sell_qty = min(bid_vol, limit + position)
                            if sell_qty > 0:
                                orders.append(Order(product, bid_price, -sell_qty))
                                position -= sell_qty

            result[product] = orders
        
        traderData = jsonpickle.encode(traderObject)

        conversions = 0
        
        # Logger를 사용하여 로그를 출력 (참고 문서에 따라 flush() 호출)
        logger.flush(original_state, result, conversions, traderData)
        
        return result, conversions, traderData