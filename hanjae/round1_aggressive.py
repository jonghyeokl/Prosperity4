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
            must_sell_buy_coeff = 0.15
            must_sell_ratio = 1 - must_sell_buy_coeff + must_sell_buy_coeff * (limit - position) / (limit * 2 * 2)
            must_buy_ratio = 1 - must_sell_buy_coeff + must_sell_buy_coeff * (limit - position) / (limit * 2 * 2)
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
                must_sell_price = (fair_ask_value * must_sell_ratio + fair_bid_value * (1 - must_sell_ratio)) if (fair_ask_value != 0 and fair_bid_value != 0) else 0
                must_buy_price = (fair_ask_value * must_buy_ratio + fair_bid_value * (1 - must_buy_ratio)) if (fair_ask_value != 0 and fair_bid_value != 0) else 0
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
                            if not beginning_never_trade and ask_price <= must_buy_price:
                                buy_qty = min(-ask_vol, limit - position)
                                if buy_qty > 0:
                                    orders.append(Order(product, ask_price, buy_qty))
                                    position += buy_qty
                                    buy_limit -= buy_qty
                if order_depth.buy_orders:
                    for bid_price, bid_vol in sorted(order_depth.buy_orders.items(), reverse=True):
                        if position > -limit:
                            if not beginning_never_trade and bid_price >= must_sell_price:
                                sell_qty = min(bid_vol, limit + position)
                                if sell_qty > 0:
                                    orders.append(Order(product, bid_price, -sell_qty))
                                    position -= sell_qty
                                    sell_limit -= sell_qty
                
                if position < limit and not beginning_never_trade:
                    best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
                    if best_bid is not None and best_bid + 1 <= must_buy_price:
                        orders.append(Order(product, best_bid + 1, min(buy_limit, limit - position)))
                
                if position > -limit and not beginning_never_trade:
                    best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
                    if best_ask is not None and best_ask - 1 >= must_sell_price:
                        orders.append(Order(product, best_ask - 1, max(-sell_limit, -limit - position)))

            elif product == "ASH_COATED_OSMIUM":
                # =====================================================
                # ASH trend-follow + fair-based partial unwind
                # - follow trend lightly
                # - buy pullbacks below fair in downtrend
                # - sell rallies above fair in uptrend
                # - never flip full inventory at once
                # - strictly enforce position limits
                # =====================================================

                # ---------- state ----------
                if "ash_mid_history" not in traderObject:
                    traderObject["ash_mid_history"] = []
                if "ash_fast_ema" not in traderObject:
                    traderObject["ash_fast_ema"] = 10000.0
                if "ash_slow_ema" not in traderObject:
                    traderObject["ash_slow_ema"] = 10000.0
                if "ash_trend_inventory" not in traderObject:
                    traderObject["ash_trend_inventory"] = 0
                if "ash_last_fair" not in traderObject:
                    traderObject["ash_last_fair"] = 10000.0

                best_bid, best_ask = self.get_best_bid_ask(order_depth)
                if best_bid is None or best_ask is None:
                    result[product] = []
                    continue

                mid_price = (best_bid + best_ask) / 2
                spread = best_ask - best_bid
                best_bid_vol = order_depth.buy_orders[best_bid]
                best_ask_vol = -order_depth.sell_orders[best_ask]

                # ---------- fair value: slow center, not mean-reversion trigger ----------
                hist = traderObject["ash_mid_history"]
                hist.append(mid_price)
                if len(hist) > 40:
                    hist.pop(0)

                fair_value = sum(hist) / len(hist)
                traderObject["ash_last_fair"] = fair_value

                # ---------- trend signal ----------
                fast_alpha = 0.22
                slow_alpha = 0.06

                traderObject["ash_fast_ema"] = (1 - fast_alpha) * traderObject["ash_fast_ema"] + fast_alpha * mid_price
                traderObject["ash_slow_ema"] = (1 - slow_alpha) * traderObject["ash_slow_ema"] + slow_alpha * mid_price

                fast_ema = traderObject["ash_fast_ema"]
                slow_ema = traderObject["ash_slow_ema"]
                trend_signal = fast_ema - slow_ema

                uptrend = trend_signal > 0.8
                downtrend = trend_signal < -0.8

                # track only the inventory accumulated by the trend-follow leg
                trend_inventory = traderObject["ash_trend_inventory"]

                # ---------- helper capacities ----------
                # These MUST be updated after every order to keep within limit.
                buy_remaining = limit - position
                sell_remaining = limit + position

                def place_buy(price: int, qty: int):
                    nonlocal position, buy_remaining, sell_remaining, trend_inventory
                    qty = max(0, min(qty, buy_remaining))
                    if qty > 0:
                        orders.append(Order(product, price, qty))
                        position += qty
                        buy_remaining = limit - position
                        sell_remaining = limit + position
                        return qty
                    return 0

                def place_sell(price: int, qty: int):
                    nonlocal position, buy_remaining, sell_remaining, trend_inventory
                    qty = max(0, min(qty, sell_remaining))
                    if qty > 0:
                        orders.append(Order(product, price, -qty))
                        position -= qty
                        buy_remaining = limit - position
                        sell_remaining = limit + position
                        return qty
                    return 0

                # ---------- parameters ----------
                trend_entry_size = 8          # how much to add when following trend
                unwind_size = 6               # how much to unwind per opportunity
                passive_base_size = 8         # smaller than before to keep smoother inventory
                max_trend_inventory = 30      # cap for trend-follow inventory only
                fair_band = 1.0               # "fair 이하/이상" buffer

                # =====================================================
                # 1) Trend-follow entry
                # =====================================================
                # Uptrend: buy strength, but only build a bounded long trend inventory
                if uptrend and trend_inventory < max_trend_inventory:
                    # aggressive buy only when the market is actually pushing up
                    # and we still have room in trend inventory
                    qty = min(best_ask_vol, trend_entry_size, max_trend_inventory - trend_inventory)
                    bought = place_buy(best_ask, qty)
                    trend_inventory += bought

                # Downtrend: sell weakness, but only build a bounded short trend inventory
                elif downtrend and trend_inventory > -max_trend_inventory:
                    qty = min(best_bid_vol, trend_entry_size, max_trend_inventory + trend_inventory)
                    sold = place_sell(best_bid, qty)
                    trend_inventory -= sold

                # =====================================================
                # 2) Fair-based partial unwind
                # =====================================================
                # This is the core requested logic:
                # - In downtrend, if price falls sufficiently below fair, buy back PART of the short trend inventory
                # - In uptrend, if price rises sufficiently above fair, sell PART of the long trend inventory
                #
                # Important: only unwind PART of the trend inventory, do not flip whole position.
                #
                # Downtrend + below fair -> partial buyback of short inventory
                if downtrend and mid_price <= fair_value - fair_band and trend_inventory < 0:
                    qty = min(best_ask_vol, unwind_size, -trend_inventory)
                    bought = place_buy(best_ask, qty)
                    trend_inventory += bought

                # Uptrend + above fair -> partial sell of long inventory
                if uptrend and mid_price >= fair_value + fair_band and trend_inventory > 0:
                    qty = min(best_bid_vol, unwind_size, trend_inventory)
                    sold = place_sell(best_bid, qty)
                    trend_inventory -= sold

                # =====================================================
                # 3) Passive quoting, gently aligned with current regime
                # =====================================================
                # Keep it smooth; do not go full long/full short.
                # Quote sizes lean with trend_inventory, but remain bounded.
                buy_size = min(passive_base_size, buy_remaining)
                sell_size = min(passive_base_size, sell_remaining)

                # inventory smoothing: if already long, reduce bid size; if short, reduce ask size
                if position > 0:
                    buy_size = max(0, buy_size - position // 20)
                    sell_size = min(sell_remaining, sell_size + position // 20)
                elif position < 0:
                    sell_size = max(0, sell_size - (-position) // 20)
                    buy_size = min(buy_remaining, buy_size + (-position) // 20)

                # quote placement:
                # uptrend -> slightly more aggressive bid
                # downtrend -> slightly more aggressive ask
                if spread >= 2:
                    if uptrend:
                        bid_quote = best_bid + 1
                        ask_quote = best_ask
                    elif downtrend:
                        bid_quote = best_bid
                        ask_quote = best_ask - 1
                    else:
                        bid_quote = best_bid + 1
                        ask_quote = best_ask - 1
                else:
                    bid_quote = best_bid
                    ask_quote = best_ask

                # safety checks to avoid crossing
                if bid_quote >= best_ask:
                    bid_quote = best_bid
                if ask_quote <= best_bid:
                    ask_quote = best_ask

                # recompute with latest remaining capacities
                buy_size = min(buy_size, buy_remaining)
                sell_size = min(sell_size, sell_remaining)

                if buy_size > 0 and bid_quote < best_ask:
                    placed = place_buy(bid_quote, buy_size)
                    # passive quotes are NOT counted as trend inventory

                if sell_size > 0 and ask_quote > best_bid:
                    placed = place_sell(ask_quote, sell_size)
                    # passive quotes are NOT counted as trend inventory

                traderObject["ash_trend_inventory"] = trend_inventory

                logger.print(
                    f"[ASH TF-PARTIAL] pos={position}, trend_inv={trend_inventory}, "
                    f"bb={best_bid}, ba={best_ask}, mid={mid_price:.1f}, fair={fair_value:.2f}, "
                    f"fast={fast_ema:.2f}, slow={slow_ema:.2f}, trend={trend_signal:.2f}, "
                    f"up={uptrend}, down={downtrend}, "
                    f"buy_rem={buy_remaining}, sell_rem={sell_remaining}, "
                    f"bid_q={bid_quote}, ask_q={ask_quote}, "
                    f"buy_sz={buy_size}, sell_sz={sell_size}"
                )

            result[product] = orders

        traderData = jsonpickle.encode(traderObject)
        conversions = 0

        logger.flush(original_state, result, conversions, traderData)

        return result, conversions, traderData