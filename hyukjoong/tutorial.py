from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import jsonpickle


class Trader:

    def run(self, state: TradingState):
        result = {}

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            position = state.position.get(product, 0)

            if product == "EMERALDS":
                fair_value = 10000
                spread = 2 
                limit = 80

                if order_depth.sell_orders:
                    for ask_price, ask_vol in sorted(order_depth.sell_orders.items()):
                        if ask_price < fair_value:
                            buy_qty = min(-ask_vol, limit - position)
                            if buy_qty > 0:
                                orders.append(Order(product, ask_price, buy_qty))
                                position += buy_qty

                if order_depth.buy_orders:
                    for bid_price, bid_vol in sorted(order_depth.buy_orders.items(), reverse=True):
                        if bid_price > fair_value:
                            sell_qty = min(bid_vol, limit + position)
                            if sell_qty > 0:
                                orders.append(Order(product, bid_price, -sell_qty))
                                position -= sell_qty

                skew = -position * 0.05
                buy_price = int(fair_value - spread + skew)
                sell_price = int(fair_value + spread + skew)

                buy_qty = limit - position
                sell_qty = limit + position

                if buy_qty > 0:
                    orders.append(Order(product, buy_price, buy_qty))
                if sell_qty > 0:
                    orders.append(Order(product, sell_price, -sell_qty))

            elif product == "TOMATOES":
                limit = 80

                best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
                best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else 0

                if best_bid and best_ask:
                    fair_value = (best_bid + best_ask) / 2
                else:
                    fair_value = 5006

                spread = 3

                if order_depth.sell_orders:
                    for ask_price, ask_vol in sorted(order_depth.sell_orders.items()):
                        if ask_price < fair_value - 1:
                            buy_qty = min(-ask_vol, limit - position)
                            if buy_qty > 0:
                                orders.append(Order(product, ask_price, buy_qty))
                                position += buy_qty

                if order_depth.buy_orders:
                    for bid_price, bid_vol in sorted(order_depth.buy_orders.items(), reverse=True):
                        if bid_price > fair_value + 1:
                            sell_qty = min(bid_vol, limit + position)
                            if sell_qty > 0:
                                orders.append(Order(product, bid_price, -sell_qty))
                                position -= sell_qty

                skew = -position * 0.08
                buy_price = int(fair_value - spread + skew)
                sell_price = int(fair_value + spread + skew)

                buy_qty = limit - position
                sell_qty = limit + position

                if buy_qty > 0:
                    orders.append(Order(product, buy_price, buy_qty))
                if sell_qty > 0:
                    orders.append(Order(product, sell_price, -sell_qty))

            result[product] = orders

        traderData = "TUTORIAL"
        conversions = 0
        return result, conversions, traderData