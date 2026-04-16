import os
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

    POSITION_LIMITS = {
        "ASH_COATED_OSMIUM": 80,
        "INTARIAN_PEPPER_ROOT": 80,
    }

    VALID_BID_ASK_VOLUME = 10

    ASH_ALPHA = [
        int(os.getenv("ASH_ALPHA_1", "-75")),
        int(os.getenv("ASH_ALPHA_2", "-50")),
        int(os.getenv("ASH_ALPHA_3", "-40")),
        int(os.getenv("ASH_ALPHA_4", "-20")),
        int(os.getenv("ASH_ALPHA_5", "-10")),
        int(os.getenv("ASH_ALPHA_6", "0")),
        int(os.getenv("ASH_ALPHA_7", "10")),
        int(os.getenv("ASH_ALPHA_8", "20")),
        int(os.getenv("ASH_ALPHA_9", "30")),
        int(os.getenv("ASH_ALPHA_10", "40")),
    ]  # alpha1 ~ alpha10

    def get_best_bid_ask(self, order_depth: OrderDepth):
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        return best_bid, best_ask

    def get_mid_price(self, order_depth: OrderDepth):
        best_bid, best_ask = self.get_best_bid_ask(order_depth)
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2
        return None
    
    def get_zscore_threshold(self, quantity: int, limit: int):
        interval = len(self.ASH_ALPHA)
        # interval 0 ~ 10 => zscore_threshold -1.5 ~ 3.5
        while interval > 0 and self.ASH_ALPHA[interval - 1] > quantity:
            interval -= 1
        return -1.5 + interval * 0.5
    
    def get_best_valid_bid_ask(self, order_depth: OrderDepth):
        best_valid_bid = None
        best_valid_ask = None
        for bid_price, bid_vol in sorted(order_depth.buy_orders.items(), reverse=True):
            if bid_vol >= self.VALID_BID_ASK_VOLUME:
                best_valid_bid = bid_price
                break
        for ask_price, ask_vol in sorted(order_depth.sell_orders.items()):
            if -ask_vol >= self.VALID_BID_ASK_VOLUME:
                best_valid_ask = ask_price
                break
        return best_valid_bid, best_valid_ask

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
            limit = self.POSITION_LIMITS[product]

            buy_limit = limit - position
            sell_limit = limit + position

            if product == "INTARIAN_PEPPER_ROOT":
                pass

            elif product == "ASH_COATED_OSMIUM":
                
                past_few_mid_history_key = "ash_coated_osmium_past_few_mid_history"

                min_history_length = 1
                max_history_length = 5
                
                z_score_threshold_for_bid = self.get_zscore_threshold(position, limit)
                z_score_threshold_for_ask = self.get_zscore_threshold(-position, limit)

                if past_few_mid_history_key not in traderObject:
                    traderObject[past_few_mid_history_key] = []
                
                past_few_mid_history = traderObject[past_few_mid_history_key]

                best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
                best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

                best_valid_bid, best_valid_ask = self.get_best_valid_bid_ask(order_depth)

                # mid_price = (best_bid + best_ask) / 2 if best_bid is not None and best_ask is not None else None

                valid_mid_price = (best_valid_bid + best_valid_ask) / 2 if best_valid_bid is not None and best_valid_ask is not None else None

                # if mid_price is None:
                #     if best_bid is not None:
                #         mid_price = best_bid + 8
                #     elif best_ask is not None:
                #         mid_price = best_ask - 8

                fair_price = None

                if len(past_few_mid_history) >= min_history_length:
                    # fair_price: 평균값
                    # fair_price = sum(past_few_mid_history) / len(past_few_mid_history)

                    # fair_price: 중앙값
                    sorted_mid_history = sorted(past_few_mid_history)
                    if len(sorted_mid_history) % 2 == 1:
                        fair_price = sorted_mid_history[len(sorted_mid_history) // 2]
                    else:
                        fair_price = (sorted_mid_history[len(sorted_mid_history) // 2 - 1] + sorted_mid_history[len(sorted_mid_history) // 2]) / 2

                if fair_price is not None:
                    # --------------------
                    # 1) TAKE
                    # fair보다 유리하면 전부 체결
                    # --------------------
                    if order_depth.sell_orders:
                        for ask_price, ask_vol in sorted(order_depth.sell_orders.items()):
                            if ask_price <= fair_price - z_score_threshold_for_bid and buy_limit > 0:
                                buy_qty = min(-ask_vol, buy_limit)
                                if buy_qty > 0:
                                    orders.append(Order(product, ask_price, buy_qty))
                                    position += buy_qty
                                    buy_limit -= buy_qty

                    if order_depth.buy_orders:
                        for bid_price, bid_vol in sorted(order_depth.buy_orders.items(), reverse=True):
                            if bid_price >= fair_price + z_score_threshold_for_ask and sell_limit > 0:
                                sell_qty = min(bid_vol, sell_limit)
                                if sell_qty > 0:
                                    orders.append(Order(product, bid_price, -sell_qty))
                                    position -= sell_qty
                                    sell_limit -= sell_qty

                    # --------------------
                    # 2) MAKE
                    # 가격: highest priority이되 fair보다 불리하지 않게
                    # 수량: 현재 포지션 반영
                    # --------------------
                    make_buy_qty = max(0, min(buy_limit, (limit - position)))
                    make_sell_qty = max(0, min(sell_limit, (limit + position)))

                    if make_buy_qty > 0:
                        if best_bid is not None:
                            buy_make_price = min(best_bid + 1, math.floor(fair_price - z_score_threshold_for_bid))
                        else:
                            buy_make_price = math.floor(fair_price - z_score_threshold_for_bid)
                        orders.append(Order(product, buy_make_price, make_buy_qty))

                    if make_sell_qty > 0:
                        if best_ask is not None:
                            sell_make_price = max(best_ask - 1, math.ceil(fair_price + z_score_threshold_for_ask))
                        else:
                            sell_make_price = math.ceil(fair_price + z_score_threshold_for_ask)
                        orders.append(Order(product, sell_make_price, -make_sell_qty))
                
                if valid_mid_price is not None:
                    if len(past_few_mid_history) < max_history_length:
                        past_few_mid_history.append(valid_mid_price)
                    else:
                        past_few_mid_history.pop(0)
                        past_few_mid_history.append(valid_mid_price)

            result[product] = orders
        
        traderData = jsonpickle.encode(traderObject)

        conversions = 0
        
        # Logger를 사용하여 로그를 출력 (참고 문서에 따라 flush() 호출)
        logger.flush(original_state, result, conversions, traderData)
        
        return result, conversions, traderData