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

    PEPPER_HISTORY_LENGTH = 99
    PEPPER_MUST_SELL_BUY_COEFF = 0.33
    PEPPER_ALPHA = 9

    ASH_EMA_WINDOW = 110
    ASH_EMA_ALPHA = 2 / (ASH_EMA_WINDOW + 1)   # 여기만 바꾸면 custom alpha로도 교체 가능
    ASH_EPSILON = 0.65                          # 백테스트 파라미터
    ASH_BOUNCE_INTERCEPT = 0.008043
    ASH_BOUNCE_SLOPE = -0.496676

    def get_best_bid_ask(self, order_depth: OrderDepth):
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        return best_bid, best_ask

    def get_mid_price(self, order_depth: OrderDepth):
        best_bid, best_ask = self.get_best_bid_ask(order_depth)
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2
        return None

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
                # initial state
                if "intarian_pepper_root_last_ask_price_history" not in traderObject:
                    traderObject["intarian_pepper_root_last_ask_price_history"] = []
                if "intarian_pepper_root_last_bid_price_history" not in traderObject:
                    traderObject["intarian_pepper_root_last_bid_price_history"] = []
                
                beginning_never_trade = False
                if len(traderObject["intarian_pepper_root_last_ask_price_history"]) < 1:
                    beginning_never_trade = True
                
                fair_value_coeff = 1 - self.PEPPER_MUST_SELL_BUY_COEFF * pow(((limit + position) / (limit * 2)) , self.PEPPER_ALPHA)
                # get fair value
                avg_ask_price = sum(traderObject["intarian_pepper_root_last_ask_price_history"]) / len(traderObject["intarian_pepper_root_last_ask_price_history"]) if len(traderObject["intarian_pepper_root_last_ask_price_history"]) > 0 else 0
                fair_ask_value = (avg_ask_price + 0.1 * (len(traderObject["intarian_pepper_root_last_ask_price_history"]) + 1)/2 + 0.1) if avg_ask_price != 0 else 0
                avg_bid_price = sum(traderObject["intarian_pepper_root_last_bid_price_history"]) / len(traderObject["intarian_pepper_root_last_bid_price_history"]) if len(traderObject["intarian_pepper_root_last_bid_price_history"]) > 0 else 0
                fair_bid_value = (avg_bid_price + 0.1 * (len(traderObject["intarian_pepper_root_last_bid_price_history"]) + 1)/2 + 0.1) if avg_bid_price != 0 else 0
                fair_price = (fair_ask_value * fair_value_coeff + fair_bid_value * (1 - fair_value_coeff)) if (fair_ask_value != 0 and fair_bid_value != 0) else 0
                # calculate mid price
                best_ask_price = min(order_depth.sell_orders.keys()) if (order_depth.sell_orders and any(order_depth.sell_orders.values())) else 0
                best_bid_price = max(order_depth.buy_orders.keys()) if (order_depth.buy_orders and any(order_depth.buy_orders.values())) else 0
                # save last self.PEPPER_HISTORY_LENGTH prices
                if best_ask_price != 0:
                    if len(traderObject["intarian_pepper_root_last_ask_price_history"]) < self.PEPPER_HISTORY_LENGTH:
                        traderObject["intarian_pepper_root_last_ask_price_history"].append(best_ask_price)
                    else:
                        traderObject["intarian_pepper_root_last_ask_price_history"].pop(0)
                        traderObject["intarian_pepper_root_last_ask_price_history"].append(best_ask_price)
                if best_bid_price != 0:
                    if len(traderObject["intarian_pepper_root_last_bid_price_history"]) < self.PEPPER_HISTORY_LENGTH:
                        traderObject["intarian_pepper_root_last_bid_price_history"].append(best_bid_price)
                    else:
                        traderObject["intarian_pepper_root_last_bid_price_history"].pop(0)
                        traderObject["intarian_pepper_root_last_bid_price_history"].append(best_bid_price)
                # Trend: +1000/day linear. Buy and hold max long
                # 1000 / 1000000 timestamp => 0.1 / 100 timestamp
                if order_depth.sell_orders:
                    for ask_price, ask_vol in sorted(order_depth.sell_orders.items()):
                        if position < limit:
                            if not beginning_never_trade and ask_price <= fair_price:
                                buy_qty = min(-ask_vol, limit - position)
                                if buy_qty > 0:
                                    orders.append(Order(product, ask_price, buy_qty))
                                    position += buy_qty
                                    buy_limit -= buy_qty
                if order_depth.buy_orders:
                    for bid_price, bid_vol in sorted(order_depth.buy_orders.items(), reverse=True):
                        if position > -limit:
                            if not beginning_never_trade and bid_price >= fair_price:
                                sell_qty = min(bid_vol, limit + position)
                                if sell_qty > 0:
                                    orders.append(Order(product, bid_price, -sell_qty))
                                    position -= sell_qty
                                    sell_limit -= sell_qty
                
                if position < limit and not beginning_never_trade:
                    best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
                    if best_bid is not None and best_bid + 1 <= fair_price:
                        orders.append(Order(product, best_bid + 1, min(buy_limit, limit - position)))
                    else:
                        orders.append(Order(product, round(fair_price - 0.5), min(buy_limit, limit - position)))
                
                if position > -limit and not beginning_never_trade:
                    best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
                    if best_ask is not None and best_ask - 1 >= fair_price:
                        orders.append(Order(product, best_ask - 1, max(-sell_limit, -limit - position)))
                    else:
                        orders.append(Order(product, round(fair_price + 0.5), max(-sell_limit, -limit - position)))

            elif product == "ASH_COATED_OSMIUM":
                ema_key = "ash_coated_osmium_ema"
                prev_mid_key = "ash_coated_osmium_prev_mid"

                if ema_key not in traderObject:
                    traderObject[ema_key] = None
                if prev_mid_key not in traderObject:
                    traderObject[prev_mid_key] = None

                best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
                best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
                mid_price = (best_bid + best_ask) / 2 if best_bid is not None and best_ask is not None else None

                prev_ema = traderObject[ema_key]
                prev_mid = traderObject[prev_mid_key]

                fair_price = None

                if mid_price is not None:
                    # EMA(80)
                    if prev_ema is None:
                        ema_value = mid_price
                    else:
                        ema_value = self.ASH_EMA_ALPHA * mid_price + (1 - self.ASH_EMA_ALPHA) * prev_ema

                    # lag-1 bounce 기반 다음 틱 mid 예측
                    if prev_mid is None:
                        bounce_mid_prediction = mid_price
                    else:
                        r_t = mid_price - prev_mid
                        bounce_mid_prediction = (
                            mid_price
                            + self.ASH_BOUNCE_INTERCEPT
                            + self.ASH_BOUNCE_SLOPE * r_t
                        )

                    # 최종 fair
                    fair_price = (
                        self.ASH_EPSILON * ema_value
                        + (1 - self.ASH_EPSILON) * bounce_mid_prediction
                    )

                    traderObject[ema_key] = ema_value
                    traderObject[prev_mid_key] = mid_price
                else:
                    # 이번 틱에 양쪽 quote가 없으면 이전 EMA를 fallback fair로 사용
                    if prev_ema is not None:
                        fair_price = prev_ema

                if fair_price is not None:
                    # --------------------
                    # 1) TAKE
                    # fair보다 유리하면 전부 체결
                    # --------------------
                    if order_depth.sell_orders:
                        for ask_price, ask_vol in sorted(order_depth.sell_orders.items()):
                            if ask_price <= fair_price and buy_limit > 0:
                                buy_qty = min(-ask_vol, buy_limit)
                                if buy_qty > 0:
                                    orders.append(Order(product, ask_price, buy_qty))
                                    position += buy_qty
                                    buy_limit -= buy_qty

                    if order_depth.buy_orders:
                        for bid_price, bid_vol in sorted(order_depth.buy_orders.items(), reverse=True):
                            if bid_price >= fair_price and sell_limit > 0:
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
                    make_buy_qty = max(0, min(buy_limit, (limit - position) // 2))
                    make_sell_qty = max(0, min(sell_limit, (limit + position) // 2))

                    if make_buy_qty > 0:
                        if best_bid is not None:
                            buy_make_price = min(best_bid + 1, math.floor(fair_price))
                        else:
                            buy_make_price = math.floor(fair_price)
                        orders.append(Order(product, buy_make_price, make_buy_qty))

                    if make_sell_qty > 0:
                        if best_ask is not None:
                            sell_make_price = max(best_ask - 1, math.ceil(fair_price))
                        else:
                            sell_make_price = math.ceil(fair_price)
                        orders.append(Order(product, sell_make_price, -make_sell_qty))

            result[product] = orders
        
        traderData = jsonpickle.encode(traderObject)

        conversions = 0
        
        # Logger를 사용하여 로그를 출력 (참고 문서에 따라 flush() 호출)
        logger.flush(original_state, result, conversions, traderData)
        
        return result, conversions, traderData