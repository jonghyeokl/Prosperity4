from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import jsonpickle
import math


class Trader:

    def run(self, state: TradingState):
        result = {}

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            position = state.position.get(product, 0)
            limit = 80

            if product == "INTARIAN_PEPPER_ROOT":
                # Linear upward trend: +1 per 1000 timestamps
                # We need to estimate the current fair value
                # Use mid price as base, but bias toward BUYING (trend is up)
                
                best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
                best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else 0

                if best_bid and best_ask:
                    mid = (best_bid + best_ask) / 2
                elif best_bid:
                    mid = best_bid + 5
                elif best_ask:
                    mid = best_ask - 5
                else:
                    mid = 13000  # fallback for day 1

                fair_value = mid

                # AGGRESSIVE BUY: take everything at or below fair value + small margin
                # Since price goes up ~1 per 1000 ts, even buying at fair value is profitable
                if order_depth.sell_orders:
                    for ask_price, ask_vol in sorted(order_depth.sell_orders.items()):
                        if ask_price <= fair_value + 2 and position < limit:
                            buy_qty = min(-ask_vol, limit - position)
                            if buy_qty > 0:
                                orders.append(Order(product, ask_price, buy_qty))
                                position += buy_qty

                # Only sell if bid is significantly above fair value
                if order_depth.buy_orders:
                    for bid_price, bid_vol in sorted(order_depth.buy_orders.items(), reverse=True):
                        if bid_price > fair_value + 5 and position > -limit:
                            sell_qty = min(bid_vol, limit + position)
                            if sell_qty > 0:
                                orders.append(Order(product, bid_price, -sell_qty))
                                position -= sell_qty

                # Post buy orders aggressively to build long position
                buy_qty = limit - position
                if buy_qty > 0:
                    # Buy close to fair value since trend is up
                    buy_price = int(fair_value - 2)
                    orders.append(Order(product, buy_price, buy_qty))

                # Post sell orders far above to capture spikes
                sell_qty = limit + position
                if sell_qty > 0:
                    sell_price = int(fair_value + 8)
                    orders.append(Order(product, sell_price, -sell_qty))

            elif product == "ASH_COATED_OSMIUM":
                # Mean-reverting around 10000, spread ~16
                fair_value = 10000

                best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
                best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else 0

                # Use mid price to adjust fair value slightly
                if best_bid and best_ask:
                    mid = (best_bid + best_ask) / 2
                    # Weighted toward 10000 (mean reversion)
                    fair_value = 0.7 * 10000 + 0.3 * mid

                # Take mispriced orders aggressively
                if order_depth.sell_orders:
                    for ask_price, ask_vol in sorted(order_depth.sell_orders.items()):
                        if ask_price < fair_value - 1 and position < limit:
                            buy_qty = min(-ask_vol, limit - position)
                            if buy_qty > 0:
                                orders.append(Order(product, ask_price, buy_qty))
                                position += buy_qty

                if order_depth.buy_orders:
                    for bid_price, bid_vol in sorted(order_depth.buy_orders.items(), reverse=True):
                        if bid_price > fair_value + 1 and position > -limit:
                            sell_qty = min(bid_vol, limit + position)
                            if sell_qty > 0:
                                orders.append(Order(product, bid_price, -sell_qty))
                                position -= sell_qty

                # Market make with inventory skew
                skew = -position * 0.15
                spread = 4

                buy_price = int(fair_value - spread + skew)
                sell_price = int(fair_value + spread + skew)

                buy_qty = limit - position
                sell_qty = limit + position

                if buy_qty > 0:
                    orders.append(Order(product, buy_price, buy_qty))
                if sell_qty > 0:
                    orders.append(Order(product, sell_price, -sell_qty))

            result[product] = orders

        traderData = "ROUND1"
        conversions = 0
        return result, conversions, traderData