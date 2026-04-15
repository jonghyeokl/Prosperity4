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

        traderObject = {}
        if state.traderData is not None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            position = state.position.get(product, 0)
            limit = self.POSITION_LIMITS.get(product, 80)

            if product == "INTARIAN_PEPPER_ROOT":
                if "intarian_pepper_root_last_mid_price_history" not in traderObject:
                    traderObject["intarian_pepper_root_last_mid_price_history"] = []

                have_to_buy = False
                beginning = False
                if len(traderObject["intarian_pepper_root_last_mid_price_history"]) < 10:
                    beginning = True
                if beginning or position < limit / 2:
                    have_to_buy = True

                avg_price = (
                    sum(traderObject["intarian_pepper_root_last_mid_price_history"])
                    / len(traderObject["intarian_pepper_root_last_mid_price_history"])
                    if len(traderObject["intarian_pepper_root_last_mid_price_history"]) > 0
                    else 0
                )
                fair_value = avg_price + 0.1

                mid_price = self.get_mid_price(order_depth)

                if mid_price is not None:
                    if len(traderObject["intarian_pepper_root_last_mid_price_history"]) < 199:
                        traderObject["intarian_pepper_root_last_mid_price_history"].append(mid_price)
                    else:
                        traderObject["intarian_pepper_root_last_mid_price_history"].pop(0)
                        traderObject["intarian_pepper_root_last_mid_price_history"].append(mid_price)

                if order_depth.sell_orders:
                    is_best_ask_price = True
                    for ask_price, ask_vol in sorted(order_depth.sell_orders.items()):
                        if position < limit:
                            if (have_to_buy and is_best_ask_price) or ask_price <= fair_value:
                                buy_qty = min(-ask_vol, limit - position)
                                if buy_qty > 0:
                                    orders.append(Order(product, ask_price, buy_qty))
                                    position += buy_qty
                        is_best_ask_price = False

                if order_depth.buy_orders:
                    for bid_price, bid_vol in sorted(order_depth.buy_orders.items(), reverse=True):
                        if position > -limit:
                            if (not beginning) and (bid_price > fair_value):
                                sell_qty = min(bid_vol, limit + position)
                                if sell_qty > 0:
                                    orders.append(Order(product, bid_price, -sell_qty))
                                    position -= sell_qty

                if position < limit and not beginning:
                    best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
                    if best_bid is not None and best_bid + 1 >= fair_value:
                        orders.append(Order(product, best_bid + 1, limit - position))

                if position > -limit and not beginning:
                    best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
                    if best_ask is not None and best_ask - 1 < fair_value:
                        orders.append(Order(product, best_ask - 1, limit + position))

            elif product == "ASH_COATED_OSMIUM":
                # -----------------------------
                # ASH_COATED_OSMIUM strategy
                # mean-reversion + inventory-aware market making
                # -----------------------------
                if "ash_mid_history" not in traderObject:
                    traderObject["ash_mid_history"] = []

                if "ash_last_fair" not in traderObject:
                    traderObject["ash_last_fair"] = 10000.0

                best_bid, best_ask = self.get_best_bid_ask(order_depth)

                if best_bid is None or best_ask is None:
                    result[product] = []
                    continue

                best_bid_vol = order_depth.buy_orders[best_bid]
                best_ask_vol = -order_depth.sell_orders[best_ask]

                mid_price = (best_bid + best_ask) / 2

                # Microprice: slightly shifts fair toward the thinner side
                total_top_vol = best_bid_vol + best_ask_vol
                if total_top_vol > 0:
                    micro_price = (best_ask * best_bid_vol + best_bid * best_ask_vol) / total_top_vol
                else:
                    micro_price = mid_price

                # Rolling history
                hist = traderObject["ash_mid_history"]
                hist.append(mid_price)
                if len(hist) > 40:
                    hist.pop(0)

                rolling_mid = sum(hist) / len(hist)

                # Blend stable fair + short-term book signal
                raw_fair = 0.7 * rolling_mid + 0.3 * micro_price

                # Inventory skew
                # long -> lower fair so we sell more aggressively
                # short -> higher fair so we buy more aggressively
                inventory_skew = 0.08 * position
                fair_value = raw_fair - inventory_skew
                traderObject["ash_last_fair"] = fair_value

                spread = best_ask - best_bid

                # Parameters
                take_edge = 1.5
                join_edge = 1.0
                default_quote_size = 12
                max_take_size = 20

                # 1) Aggressively take clearly favorable quotes
                for ask_price, ask_vol in sorted(order_depth.sell_orders.items()):
                    if position >= limit:
                        break
                    if ask_price <= fair_value - take_edge:
                        buy_qty = min(-ask_vol, limit - position, max_take_size)
                        if buy_qty > 0:
                            orders.append(Order(product, ask_price, buy_qty))
                            position += buy_qty

                for bid_price, bid_vol in sorted(order_depth.buy_orders.items(), reverse=True):
                    if position <= -limit:
                        break
                    if bid_price >= fair_value + take_edge:
                        sell_qty = min(bid_vol, limit + position, max_take_size)
                        if sell_qty > 0:
                            orders.append(Order(product, bid_price, -sell_qty))
                            position -= sell_qty

                # 2) Passive market making
                # position-aware sizes
                buy_capacity = limit - position
                sell_capacity = limit + position

                buy_size = min(default_quote_size, max(0, buy_capacity))
                sell_size = min(default_quote_size, max(0, sell_capacity))

                # reduce same-direction quoting if inventory is already large
                if position > 50:
                    buy_size = min(buy_size, 3)
                    sell_size = min(sell_size, 20)
                elif position < -50:
                    sell_size = min(sell_size, 3)
                    buy_size = min(buy_size, 20)

                # quote placement
                if spread >= 2:
                    bid_quote = min(best_bid + 1, math.floor(fair_value - join_edge))
                    ask_quote = max(best_ask - 1, math.ceil(fair_value + join_edge))
                else:
                    bid_quote = math.floor(fair_value - 1)
                    ask_quote = math.ceil(fair_value + 1)

                # make sure not to cross unintentionally
                if bid_quote >= best_ask:
                    bid_quote = best_ask - 1
                if ask_quote <= best_bid:
                    ask_quote = best_bid + 1

                if buy_size > 0 and bid_quote < best_ask:
                    orders.append(Order(product, bid_quote, buy_size))

                if sell_size > 0 and ask_quote > best_bid:
                    orders.append(Order(product, ask_quote, -sell_size))

                logger.print(
                    f"ASH pos={position}, bb={best_bid}, ba={best_ask}, mid={mid_price:.1f}, "
                    f"micro={micro_price:.2f}, fair={fair_value:.2f}, "
                    f"bid_q={bid_quote}, ask_q={ask_quote}"
                )

            result[product] = orders

        traderData = jsonpickle.encode(traderObject)
        conversions = 0

        logger.flush(original_state, result, conversions, traderData)

        return result, conversions, traderData