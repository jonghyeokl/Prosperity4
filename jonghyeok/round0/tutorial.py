import json
from typing import Any, List, Optional
import jsonpickle
import math
import copy
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

# ------------------------------------------------------------
# Logger 클래스 (참고 문서의 boilerplate 그대로 사용)
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Trader 알고리즘 코드 (올해 코드)
# ------------------------------------------------------------
class Product:
    EMERALDS = "EMERALDS"
    TOMATOES = "TOMATOES"

PARAMS = {
    Product.EMERALDS: {
        "fair_value": 10000,   # 안정적인 상품, 고정된 가치 (예시), 고정
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 1,
        "join_edge": 2,
        "default_edge": 4,
        "soft_position_limit": 40,  # 포지션 리밋 80의 절반 정도로 설정
    },
    Product.TOMATOES: {
        "take_width": 1,
        "clear_width": 5, # 기존 0
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.229, # 기존 -0.229
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
    },
}

class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params
        self.LIMIT = {
            Product.EMERALDS: 80,
            Product.TOMATOES: 80,
        }
    
    def take_best_orders(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (int, int):
        position_limit = self.LIMIT[product]
        
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]
            
            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(best_ask_amount, position_limit - position)
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]
        
        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            
            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(best_bid_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]
        
        return buy_order_volume, sell_order_volume

    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))
        
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))
        return buy_order_volume, sell_order_volume

    def clear_position_order(
        self,
        product: str,
        fair_value: float,
        width: int,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)
        
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        
        if position_after_take > 0:
            clear_quantity = sum(
                volume for price, volume in order_depth.buy_orders.items() if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)
        
        if position_after_take < 0:
            clear_quantity = sum(
                abs(volume) for price, volume in order_depth.sell_orders.items() if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)
        
        return buy_order_volume, sell_order_volume

    def TOMATOES_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        # TOMATOES의 공정가를 산출하는 함수 (전년도 STARFRUIT 방식을 변형)
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price]) >= self.params[Product.TOMATOES]["adverse_volume"]
            ]
            filtered_bid = [
                price for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price]) >= self.params[Product.TOMATOES]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask is None or mm_bid is None:
                if traderObject.get("TOMATOES_last_price", None) is None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["TOMATOES_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if traderObject.get("TOMATOES_last_price", None) is not None:
                last_price = traderObject["TOMATOES_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = last_returns * self.params[Product.TOMATOES]["reversion_beta"]
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["TOMATOES_last_price"] = mmmid_price
            return fair
        return None


    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0
        
        buy_order_volume, sell_order_volume = self.take_best_orders(
            product,
            fair_value,
            take_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
            prevent_adverse,
            adverse_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product,
            fair_value,
            clear_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def make_orders(
        self,
        product,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        disregard_edge: float,
        join_edge: float,
        default_edge: float,
        manage_position: bool = False,
        soft_position_limit: int = 0,
    ):
        orders: List[Order] = []
        asks_above_fair = [
            price for price in order_depth.sell_orders.keys()
            if price > fair_value + disregard_edge
        ]
        bids_below_fair = [
            price for price in order_depth.buy_orders.keys()
            if price < fair_value - disregard_edge
        ]
    
        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None
    
        ask = round(fair_value + default_edge)
        if best_ask_above_fair is not None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair
            else:
                ask = best_ask_above_fair - 1
    
        bid = round(fair_value - default_edge)
        if best_bid_below_fair is not None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1
    
        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -soft_position_limit:
                bid += 1
    
        buy_order_volume, sell_order_volume = self.market_make(
            product,
            orders,
            bid,
            ask,
            position,
            buy_order_volume,
            sell_order_volume,
        )
    
        return orders, buy_order_volume, sell_order_volume

    def run(self, state: TradingState) -> tuple[dict[Symbol, List[Order]], int, str]:
        original_state = copy.deepcopy(state)
        # traderData를 디코딩하여 상태를 복원 (없으면 새로 생성)
        traderObject = {}
        if state.traderData is not None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
    
        result = {}
    
        # EMERALDS (안정적인 상품)
        if Product.EMERALDS in self.params and Product.EMERALDS in state.order_depths:
            resin_position = state.position.get(Product.EMERALDS, 0)
            resin_take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                Product.EMERALDS,
                state.order_depths[Product.EMERALDS],
                self.params[Product.EMERALDS]["fair_value"],
                self.params[Product.EMERALDS]["take_width"],
                resin_position,
            )
            resin_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                Product.EMERALDS,
                state.order_depths[Product.EMERALDS],
                self.params[Product.EMERALDS]["fair_value"],
                self.params[Product.EMERALDS]["clear_width"],
                resin_position,
                buy_order_volume,
                sell_order_volume,
            )
            resin_make_orders, _, _ = self.make_orders(
                Product.EMERALDS,
                state.order_depths[Product.EMERALDS],
                self.params[Product.EMERALDS]["fair_value"],
                resin_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.EMERALDS]["disregard_edge"],
                self.params[Product.EMERALDS]["join_edge"],
                self.params[Product.EMERALDS]["default_edge"],
                True,
                self.params[Product.EMERALDS]["soft_position_limit"],
            )
            result[Product.EMERALDS] = resin_take_orders + resin_clear_orders + resin_make_orders
        else:
            resin_position = state.position.get(Product.EMERALDS, 0)
    
        # TOMATOES (변동성이 큰 상품)
        if Product.TOMATOES in self.params and Product.TOMATOES in state.order_depths:
            TOMATOES_position = state.position.get(Product.TOMATOES, 0)
            TOMATOES_fair_value = self.TOMATOES_fair_value(state.order_depths[Product.TOMATOES], traderObject)
            TOMATOES_take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                Product.TOMATOES,
                state.order_depths[Product.TOMATOES],
                TOMATOES_fair_value,
                self.params[Product.TOMATOES]["take_width"],
                TOMATOES_position,
                self.params[Product.TOMATOES]["prevent_adverse"],
                self.params[Product.TOMATOES]["adverse_volume"],
            )
            TOMATOES_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                Product.TOMATOES,
                state.order_depths[Product.TOMATOES],
                TOMATOES_fair_value,
                self.params[Product.TOMATOES]["clear_width"],
                TOMATOES_position,
                buy_order_volume,
                sell_order_volume,
            )
            TOMATOES_make_orders, _, _ = self.make_orders(
                Product.TOMATOES,
                state.order_depths[Product.TOMATOES],
                TOMATOES_fair_value,
                TOMATOES_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.TOMATOES]["disregard_edge"],
                self.params[Product.TOMATOES]["join_edge"],
                self.params[Product.TOMATOES]["default_edge"],
            )
            result[Product.TOMATOES] = TOMATOES_take_orders + TOMATOES_clear_orders + TOMATOES_make_orders
        else:
            TOMATOES_position = state.position.get(Product.TOMATOES, 0)
            TOMATOES_fair_value = None

    
        conversions = 0  # 필요에 따라 변환 요청도 추가 가능
        traderData = jsonpickle.encode(traderObject)
    
        # Logger를 사용하여 로그를 출력 (참고 문서에 따라 flush() 호출)
        logger.flush(original_state, result, conversions, traderData)
    
        return result, conversions, traderData

# 제출 아이디: 59f81e67-f6c6-4254-b61e-39661eac6141