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
            limit = 80

            # coefficients
            history_length = 399 # 클수록 잘 나오나 느려짐, 이정도면 충분한듯
            must_sell_buy_coeff = 0.1 # 클수록 더 쉽게 팖


            fair_value_coeff = 1 - must_sell_buy_coeff + must_sell_buy_coeff * (limit - position) / (limit * 2)
            buy_limit = limit - position
            sell_limit = limit + position

            if product == "INTARIAN_PEPPER_ROOT":
                # initial state
                if "intarian_pepper_root_last_ask_price_history" not in traderObject:
                    traderObject["intarian_pepper_root_last_ask_price_history"] = []
                if "intarian_pepper_root_last_bid_price_history" not in traderObject:
                    traderObject["intarian_pepper_root_last_bid_price_history"] = []
                
                beginning_never_trade = False
                if len(traderObject["intarian_pepper_root_last_ask_price_history"]) < 2:
                    beginning_never_trade = True
                
                # get fair value
                avg_ask_price = sum(traderObject["intarian_pepper_root_last_ask_price_history"]) / len(traderObject["intarian_pepper_root_last_ask_price_history"]) if len(traderObject["intarian_pepper_root_last_ask_price_history"]) > 0 else 0
                fair_ask_value = (avg_ask_price + 0.1 * (len(traderObject["intarian_pepper_root_last_ask_price_history"]) + 1)/2 + 0.1) if avg_ask_price != 0 else 0
                avg_bid_price = sum(traderObject["intarian_pepper_root_last_bid_price_history"]) / len(traderObject["intarian_pepper_root_last_bid_price_history"]) if len(traderObject["intarian_pepper_root_last_bid_price_history"]) > 0 else 0
                fair_bid_value = (avg_bid_price + 0.1 * (len(traderObject["intarian_pepper_root_last_bid_price_history"]) + 1)/2 + 0.1) if avg_bid_price != 0 else 0
                fair_price = (fair_ask_value * fair_value_coeff + fair_bid_value * (1 - fair_value_coeff)) if (fair_ask_value != 0 and fair_bid_value != 0) else 0
                # calculate mid price
                best_ask_price = min(order_depth.sell_orders.keys()) if (order_depth.sell_orders and any(order_depth.sell_orders.values())) else 0
                best_bid_price = max(order_depth.buy_orders.keys()) if (order_depth.buy_orders and any(order_depth.buy_orders.values())) else 0
                # save last 199 prices
                if best_ask_price != 0:
                    if len(traderObject["intarian_pepper_root_last_ask_price_history"]) < history_length:
                        traderObject["intarian_pepper_root_last_ask_price_history"].append(best_ask_price)
                    else:
                        traderObject["intarian_pepper_root_last_ask_price_history"].pop(0)
                        traderObject["intarian_pepper_root_last_ask_price_history"].append(best_ask_price)
                if best_bid_price != 0:
                    if len(traderObject["intarian_pepper_root_last_bid_price_history"]) < history_length:
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
                # =====================================================
                # ASH: z-score mean reversion + stale quote capture
                # fixed logging + actual-position-based target following
                # =====================================================
                if "ash_mid_history" not in traderObject:
                    traderObject["ash_mid_history"] = []

                best_bid, best_ask = self.get_best_bid_ask(order_depth)
                if best_bid is None or best_ask is None:
                    result[product] = []
                    continue

                best_bid_vol = order_depth.buy_orders[best_bid]
                best_ask_vol = -order_depth.sell_orders[best_ask]

                mid_price = (best_bid + best_ask) / 2
                spread = best_ask - best_bid

                # ---------- REAL position vs working position ----------
                actual_pos = state.position.get(product, 0)
                working_pos = actual_pos

                # history: recent 20 mids
                hist = traderObject["ash_mid_history"]
                hist.append(mid_price)
                if len(hist) > 20:
                    hist.pop(0)

                fair_value = sum(hist) / len(hist)

                if len(hist) >= 5:
                    mean_hist = fair_value
                    variance = sum((x - mean_hist) ** 2 for x in hist) / len(hist)
                    vol = math.sqrt(variance)
                else:
                    vol = 1.0
                vol = max(vol, 1.0)

                z_mid = (mid_price - fair_value) / vol
                z_ask = (best_ask - fair_value) / vol
                z_bid = (best_bid - fair_value) / vol

                orders = []

                # working capacity only for order submission safety
                buy_remaining = limit - working_pos
                sell_remaining = limit + working_pos

                def place_buy(price: int, qty: int, reason: str):
                    nonlocal working_pos, buy_remaining, sell_remaining, orders
                    qty = max(0, min(qty, buy_remaining))
                    if qty > 0:
                        orders.append(Order(product, price, qty))
                        working_pos += qty
                        buy_remaining = limit - working_pos
                        sell_remaining = limit + working_pos
                        logger.print(
                            f"[ASHDBG] ts={state.timestamp} action=BUY reason={reason} "
                            f"px={price} qty={qty} actual_before={actual_pos} working_after={working_pos}"
                        )
                        return qty
                    return 0

                def place_sell(price: int, qty: int, reason: str):
                    nonlocal working_pos, buy_remaining, sell_remaining, orders
                    qty = max(0, min(qty, sell_remaining))
                    if qty > 0:
                        orders.append(Order(product, price, -qty))
                        working_pos -= qty
                        buy_remaining = limit - working_pos
                        sell_remaining = limit + working_pos
                        logger.print(
                            f"[ASHDBG] ts={state.timestamp} action=SELL reason={reason} "
                            f"px={price} qty={qty} actual_before={actual_pos} working_after={working_pos}"
                        )
                        return qty
                    return 0

                # =====================================================
                # 1) stale quote capture
                # =====================================================
                stale_threshold = 1.2
                stale_take_size = 20

                if z_ask <= -stale_threshold and buy_remaining > 0:
                    place_buy(best_ask, min(best_ask_vol, stale_take_size), "STALE_ASK")

                if z_bid >= stale_threshold and sell_remaining > 0:
                    place_sell(best_bid, min(best_bid_vol, stale_take_size), "STALE_BID")

                # =====================================================
                # 2) target position from z-score
                # IMPORTANT: target is interpreted against ACTUAL position
                # =====================================================
                target_strength = 10.0
                max_target = 40

                target_position = int(round(-target_strength * z_mid))
                target_position = max(-max_target, min(max_target, target_position))

                # gap based on actual position, not working position
                inventory_gap = target_position - actual_pos

                # =====================================================
                # 3) mean-reversion taker layer
                # =====================================================
                mr_entry_threshold = 0.6
                taker_size = 10

                if inventory_gap > 0 and z_ask <= -mr_entry_threshold and buy_remaining > 0:
                    qty = min(best_ask_vol, taker_size, inventory_gap)
                    place_buy(best_ask, qty, "MR_TAKER_BUY")

                if inventory_gap < 0 and z_bid >= mr_entry_threshold and sell_remaining > 0:
                    qty = min(best_bid_vol, taker_size, -inventory_gap)
                    place_sell(best_bid, qty, "MR_TAKER_SELL")

                # recompute working-gap for order submission,
                # but keep actual_gap separately for diagnosis
                working_gap = target_position - working_pos

                # =====================================================
                # 4) target-seeking passive quoting
                # =====================================================
                min_size = 2
                max_size = 16

                # size from ACTUAL gap magnitude
                gap_mag = min(abs(inventory_gap), max_target)
                base_size = int(round(min_size + (max_size - min_size) * gap_mag / max_target))

                buy_size = 0
                sell_size = 0
                bid_quote = best_bid
                ask_quote = best_ask

                if spread >= 2:
                    # strongly directional / one-sided when far from target
                    if inventory_gap >= 15:
                        # need more long: aggressive bid, minimal ask
                        bid_quote = best_bid + 1
                        ask_quote = best_ask
                        buy_size = base_size
                        sell_size = 0

                    elif inventory_gap <= -15:
                        # need more short: aggressive ask, minimal bid
                        bid_quote = best_bid
                        ask_quote = best_ask - 1
                        buy_size = 0
                        sell_size = base_size

                    else:
                        # near target: two-sided market making, but slightly tilted
                        bid_quote = best_bid + 1
                        ask_quote = best_ask - 1

                        if inventory_gap > 0:
                            buy_size = max(min_size, base_size)
                            sell_size = min_size
                        elif inventory_gap < 0:
                            buy_size = min_size
                            sell_size = max(min_size, base_size)
                        else:
                            buy_size = min_size
                            sell_size = min_size
                else:
                    bid_quote = best_bid
                    ask_quote = best_ask

                    if inventory_gap > 0:
                        buy_size = max(min_size, base_size // 2)
                        sell_size = 0
                    elif inventory_gap < 0:
                        buy_size = 0
                        sell_size = max(min_size, base_size // 2)
                    else:
                        buy_size = min_size
                        sell_size = min_size

                # safety checks
                if bid_quote >= best_ask:
                    bid_quote = best_bid
                if ask_quote <= best_bid:
                    ask_quote = best_ask

                buy_size = min(buy_size, buy_remaining)
                sell_size = min(sell_size, sell_remaining)

                if buy_size > 0 and bid_quote < best_ask:
                    place_buy(bid_quote, buy_size, "PASSIVE_BID")

                if sell_size > 0 and ask_quote > best_bid:
                    place_sell(ask_quote, sell_size, "PASSIVE_ASK")

                logger.print(
                    f"[ASHDBG] ts={state.timestamp} "
                    f"actual_pos={actual_pos} working_pos={working_pos} target={target_position} "
                    f"actual_gap={inventory_gap} working_gap={working_gap} "
                    f"mid={mid_price:.2f} fair={fair_value:.2f} vol={vol:.2f} "
                    f"z_mid={z_mid:.2f} z_ask={z_ask:.2f} z_bid={z_bid:.2f} "
                    f"bb={best_bid} ba={best_ask} bid_q={bid_quote} ask_q={ask_quote} "
                    f"buy_sz={buy_size} sell_sz={sell_size} "
                    f"buy_rem={buy_remaining} sell_rem={sell_remaining}"
                )

            result[product] = orders

        traderData = jsonpickle.encode(traderObject)
        conversions = 0

        logger.flush(original_state, result, conversions, traderData)

        return result, conversions, traderData