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

    # pair position의 의미:
    # pair_pos > 0 => first product long, second product short
    PAIR_DEFS = {
        "VEV_4000_VEV_4500": {
            "first": "VEV_4000",
            "second": "VEV_4500",
            "fair_spread": 500,      # VEV_4000 - VEV_4500
            "limit": 200,
        },
        "VEV_4000_UNDERLYING": {
            "first": "VEV_4000",
            "second": "VELVETFRUIT_EXTRACT",
            "fair_spread": -4000,    # VEV_4000 - VELVETFRUIT
            "limit": 100,
        },
        "VEV_4500_UNDERLYING": {
            "first": "VEV_4500",
            "second": "VELVETFRUIT_EXTRACT",
            "fair_spread": -4500,    # VEV_4500 - VELVETFRUIT
            "limit": 100,
        },
    }

    PAIR_PRIORITY = [
        "VEV_4000_VEV_4500",
        "VEV_4000_UNDERLYING",
        "VEV_4500_UNDERLYING",
    ]

    TAKE_THRESHOLD = 0

    VALID_BID_ASK_VOLUME = {
        "VELVETFRUIT_EXTRACT": 15,
        "VEV_4000": 6,
        "VEV_4500": 6,
        "VEV_5000": 6,
        "VEV_5100": 6,
        "VEV_5200": 6,
        "VEV_5300": 5,
        "VEV_5400": 5,
        "VEV_5500": 5,
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

    def get_valid_mid_price(self, order_depth: OrderDepth, valid_volume: int):
        best_valid_bid, best_valid_ask = self.get_best_valid_bid_ask(order_depth, valid_volume)
        if best_valid_bid is not None and best_valid_ask is not None:
            return (best_valid_bid + best_valid_ask) / 2
        return None

    def add_order(
        self,
        result: dict[Symbol, List[Order]],
        buy_remaining: dict[str, int],
        sell_remaining: dict[str, int],
        product: str,
        price: int,
        quantity: int,
    ) -> int:
        if quantity == 0:
            return 0

        if quantity > 0:
            qty = min(quantity, buy_remaining.get(product, 0))
            if qty <= 0:
                return 0
            result[product].append(Order(product, int(price), int(qty)))
            buy_remaining[product] -= qty
            return qty

        qty = min(-quantity, sell_remaining.get(product, 0))
        if qty <= 0:
            return 0
        result[product].append(Order(product, int(price), -int(qty)))
        sell_remaining[product] -= qty
        return -qty

    def get_bid_levels(self, order_depth: OrderDepth):
        return [
            [int(price), int(volume)]
            for price, volume in sorted(order_depth.buy_orders.items(), reverse=True)
            if volume > 0
        ]

    def get_ask_levels(self, order_depth: OrderDepth):
        return [
            [int(price), int(-volume)]
            for price, volume in sorted(order_depth.sell_orders.items())
            if -volume > 0
        ]

    def execute_pair_take(
        self,
        pair_key: str,
        state: TradingState,
        result: dict[Symbol, List[Order]],
        buy_remaining: dict[str, int],
        sell_remaining: dict[str, int],
        pair_positions: dict[str, int],
    ) -> None:
        pair = self.PAIR_DEFS[pair_key]

        first = pair["first"]
        second = pair["second"]
        fair_spread = pair["fair_spread"]
        pair_limit = pair["limit"]

        if first not in state.order_depths or second not in state.order_depths:
            return

        first_depth = state.order_depths[first]
        second_depth = state.order_depths[second]

        pair_pos = int(pair_positions.get(pair_key, 0))

        # ============================================================
        # Direction +:
        # buy first, sell second
        #
        # condition:
        # first_ask - second_bid < fair_spread
        #
        # pair_pos increases.
        # ============================================================
        first_asks = self.get_ask_levels(first_depth)
        second_bids = self.get_bid_levels(second_depth)

        i, j = 0, 0
        while i < len(first_asks) and j < len(second_bids):
            first_ask, first_ask_vol = first_asks[i]
            second_bid, second_bid_vol = second_bids[j]

            spread = first_ask - second_bid
            edge = fair_spread - spread

            if edge <= self.TAKE_THRESHOLD:
                break

            pair_remaining = pair_limit - pair_pos
            if pair_remaining <= 0:
                break

            qty = min(
                first_ask_vol,
                second_bid_vol,
                buy_remaining.get(first, 0),
                sell_remaining.get(second, 0),
                pair_remaining,
            )

            if qty <= 0:
                break

            filled_first = self.add_order(
                result=result,
                buy_remaining=buy_remaining,
                sell_remaining=sell_remaining,
                product=first,
                price=first_ask,
                quantity=qty,
            )
            filled_second = self.add_order(
                result=result,
                buy_remaining=buy_remaining,
                sell_remaining=sell_remaining,
                product=second,
                price=second_bid,
                quantity=-qty,
            )

            actual_qty = min(filled_first, -filled_second)

            if actual_qty <= 0:
                break

            pair_pos += actual_qty
            pair_positions[pair_key] = pair_pos

            first_asks[i][1] -= actual_qty
            second_bids[j][1] -= actual_qty

            if first_asks[i][1] <= 0:
                i += 1
            if second_bids[j][1] <= 0:
                j += 1

        # ============================================================
        # Direction -:
        # sell first, buy second
        #
        # condition:
        # first_bid - second_ask > fair_spread
        #
        # pair_pos decreases.
        # ============================================================
        first_bids = self.get_bid_levels(first_depth)
        second_asks = self.get_ask_levels(second_depth)

        i, j = 0, 0
        while i < len(first_bids) and j < len(second_asks):
            first_bid, first_bid_vol = first_bids[i]
            second_ask, second_ask_vol = second_asks[j]

            spread = first_bid - second_ask
            edge = spread - fair_spread

            if edge <= self.TAKE_THRESHOLD:
                break

            pair_remaining = pair_limit + pair_pos
            if pair_remaining <= 0:
                break

            qty = min(
                first_bid_vol,
                second_ask_vol,
                sell_remaining.get(first, 0),
                buy_remaining.get(second, 0),
                pair_remaining,
            )

            if qty <= 0:
                break

            filled_first = self.add_order(
                result=result,
                buy_remaining=buy_remaining,
                sell_remaining=sell_remaining,
                product=first,
                price=first_bid,
                quantity=-qty,
            )
            filled_second = self.add_order(
                result=result,
                buy_remaining=buy_remaining,
                sell_remaining=sell_remaining,
                product=second,
                price=second_ask,
                quantity=qty,
            )

            actual_qty = min(-filled_first, filled_second)

            if actual_qty <= 0:
                break

            pair_pos -= actual_qty
            pair_positions[pair_key] = pair_pos

            first_bids[i][1] -= actual_qty
            second_asks[j][1] -= actual_qty

            if first_bids[i][1] <= 0:
                i += 1
            if second_asks[j][1] <= 0:
                j += 1

    def run(self, state: TradingState, day_num: int):
        original_state = copy.deepcopy(state)

        traderObject = {}
        if state.traderData is not None and state.traderData != "":
            try:
                traderObject = jsonpickle.decode(state.traderData)
            except Exception:
                traderObject = {}

        result: dict[Symbol, List[Order]] = {
            product: [] for product in state.order_depths
        }

        buy_remaining = {}
        sell_remaining = {}

        for product in state.order_depths:
            position = state.position.get(product, 0)
            limit = self.POSITION_LIMITS.get(product, 0)

            buy_remaining[product] = limit - position
            sell_remaining[product] = limit + position

        pair_positions = traderObject.get("pair_positions", {})
        for pair_key in self.PAIR_DEFS:
            pair_positions[pair_key] = int(pair_positions.get(pair_key, 0))

        for pair_key in self.PAIR_PRIORITY:
            self.execute_pair_take(
                pair_key=pair_key,
                state=state,
                result=result,
                buy_remaining=buy_remaining,
                sell_remaining=sell_remaining,
                pair_positions=pair_positions,
            )

        traderObject["pair_positions"] = pair_positions

        logger.print("PAIR_POSITIONS", pair_positions)

        traderData = jsonpickle.encode(traderObject)
        conversions = 0

        logger.flush(original_state, result, conversions, traderData)

        return result, conversions, traderData