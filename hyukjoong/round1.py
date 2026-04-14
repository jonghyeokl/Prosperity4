from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List


class Trader:

    def run(self, state: TradingState):
        result = {}

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            position = state.position.get(product, 0)
            limit = 80

            if product == "INTARIAN_PEPPER_ROOT":
                # Trend: +1000/day linear. Buy and hold max long
                if order_depth.sell_orders:
                    for ask_price, ask_vol in sorted(order_depth.sell_orders.items()):
                        if position < limit:
                            buy_qty = min(-ask_vol, limit - position)
                            if buy_qty > 0:
                                orders.append(Order(product, ask_price, buy_qty))
                                position += buy_qty
                if position < limit:
                    best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
                    if best_bid:
                        orders.append(Order(product, best_bid + 1, limit - position))

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

        traderData = "ROUND1"
        conversions = 0
        return result, conversions, traderData