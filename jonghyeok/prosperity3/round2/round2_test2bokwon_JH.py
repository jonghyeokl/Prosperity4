import json
from typing import Any, List, Dict
import jsonpickle
import math
import copy
import numpy as np
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

# ------------------------------------------------------------
# Logger 클래스 (boilerplate 그대로 사용)
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
# Trader 알고리즘 코드
# ------------------------------------------------------------
class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    SYNTHETIC = "SYNTHETIC"
    SPREAD = "SPREAD"

# 신규 BASKET_WEIGHTS: 모든 구성요소에 대해 synthetic으로 취급 (양수: 매수 측, 음수: 매도 측)
BASKET_WEIGHTS = {
    Product.PICNIC_BASKET1: 3,
    Product.PICNIC_BASKET2: -5,
    Product.CROISSANTS: 2,
    Product.JAMS: 1,
    Product.DJEMBES: -3,
}

PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 1,
        "join_edge": 2,
        "default_edge": 4,
        "soft_position_limit": 25,
    },
    Product.KELP: {
        "take_width": 1,
        "clear_width": 5,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.229,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
    },
    Product.SQUID_INK: {
        # "take_width": 2,
        # "clear_width": 0, # 기존 0
        "prevent_adverse": True,
        "adverse_volume": 15,
        "zscore_range" : 400,  # z-score 범위: 200
        "zscore_threshold": 4,
        "default_std_val":20,
        "target_position":50,
    },
    Product.SPREAD: {
        "default_spread_mean": -14.76285,
        "default_spread_std": 76.07966,
        "spread_std_window": 45,
        "zscore_threshold": 1.5,
        "target_position": 20,
    },
}

class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params
        self.LIMIT = {
            Product.RAINFOREST_RESIN: 50,
            Product.KELP: 50,
            Product.SQUID_INK: 50,
            Product.CROISSANTS: 250,
            Product.JAMS: 350,
            Product.DJEMBES: 60,
            Product.PICNIC_BASKET1: 60,
            Product.PICNIC_BASKET2: 100,
        }
    
    # 기존 take_best_orders, market_make, clear_position_order, kelp_fair_value, squid_ink_fair_value, take_orders, clear_orders, make_orders는 그대로 유지

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

    def kelp_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price]) >= self.params[Product.KELP]["adverse_volume"]
            ]
            filtered_bid = [
                price for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price]) >= self.params[Product.KELP]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask is None or mm_bid is None:
                if traderObject.get("kelp_last_price", None) is None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["kelp_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2
            if traderObject.get("kelp_last_price", None) is not None:
                last_price = traderObject["kelp_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = last_returns * self.params[Product.KELP]["reversion_beta"]
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["kelp_last_price"] = mmmid_price
            return fair
        return None

    def squid_ink_spread_order(self, order_depth: OrderDepth,ink_position:int, traderObject) -> float:
        # SQUID_INK의 공정가를 산출하는 함수 (z-score 전략 적용)
        zscore_range = self.params[Product.SQUID_INK]["zscore_range"]
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price]) >= self.params[Product.SQUID_INK]["adverse_volume"]
            ]
            filtered_bid = [
                price for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price]) >= self.params[Product.SQUID_INK]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask is None or mm_bid is None:
                if traderObject.get("squid_ink_last_price", None) is None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["squid_ink_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            # 저장: 현재 중간가격을 traderObject에 저장
            traderObject["squid_ink_last_price"] = mmmid_price
        
            # 지난 관측치(history)를 traderObject에 저장하고 갱신
            if "squid_ink_history" not in traderObject:
                traderObject["squid_ink_history"] = []
            history = traderObject["squid_ink_history"]
            history.append(mmmid_price)
            # 최근 100 타임프레임만 유지
            

            history = history[-zscore_range:]
            
            traderObject["squid_ink_history"] = history
            # zscore_range개 이상의 관측치가 있을 경우, z-score를 계산하여 공정가격 산출
            mean_val = sum(history) / len(history)
            # var_val = sum((p - mean_val) ** 2 for p in history) / len(history)
            # std_val = math.sqrt(var_val)

            std_val = np.std(history)
            
            if len(history) < zscore_range:
                std_val = self.params[Product.SQUID_INK]["default_std_val"]
        
            # if len(history) <= zscore_range:
            #     std_val = 
            if std_val == 0:
                zscore = 0
            else:
                zscore = (mmmid_price - mean_val) / std_val
            

            
            if zscore >= self.params[Product.SQUID_INK]["zscore_threshold"]:
                if ink_position != -self.params[Product.SQUID_INK]["target_position"]:
                    return self.ink_execute_spread_orders(
                        -self.params[Product.SQUID_INK]["target_position"],
                        ink_position,
                        order_depth
                    )
            if zscore <= -self.params[Product.SQUID_INK]["zscore_threshold"]:
                if ink_position != self.params[Product.SQUID_INK]["target_position"]:
                    return self.ink_execute_spread_orders(
                        self.params[Product.SQUID_INK]["target_position"],
                        ink_position,
                        order_depth
                    )

        return None

    def ink_execute_spread_orders(self,target_position, ink_position,order_depth: OrderDepth):
        if target_position ==ink_position:
            return None
        orders = []
        position_diff = target_position - ink_position

        if position_diff > 0:
            # 무지성 매수: ask에서부터 사기
            sorted_asks = sorted(order_depth.sell_orders.items())  # (price, -volume)
            for ask_price, ask_volume in sorted_asks:
                volume_to_buy = min(position_diff, -ask_volume)  # ask_volume은 음수
                if volume_to_buy <= 0:
                    continue
                orders.append(Order(Product.SQUID_INK, ask_price, volume_to_buy))
                ink_position += volume_to_buy
                position_diff = target_position - ink_position
                if position_diff <= 0:
                    break

        elif position_diff < 0:
            # 무지성 매도: bid에서부터 팔기
            sorted_bids = sorted(order_depth.buy_orders.items(), reverse=True)  # (price, volume)
            for bid_price, bid_volume in sorted_bids:
                volume_to_sell = min(-position_diff, bid_volume)
                if volume_to_sell <= 0:
                    continue
                orders.append(Order(Product.SQUID_INK, bid_price, -volume_to_sell))
                ink_position -= volume_to_sell
                position_diff = target_position - ink_position
                if position_diff >= 0:
                    break        
        return orders

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

    # --- 새로운 synthetic basket 처리: 모든 구성 상품을 BASKET_WEIGHTS로 합성 ---
    def get_synthetic_basket_order_depth(
        self, order_depths: Dict[str, OrderDepth]
    ) -> OrderDepth:
        synthetic_order_depth = OrderDepth()
        synthetic_bid = 0
        synthetic_ask = 0
        bid_volume_candidates = []
        ask_volume_candidates = []
        for prod, weight in BASKET_WEIGHTS.items():
            if prod in order_depths:
                od = order_depths[prod]
                # 양수 weight -> bid: 최고 buy, ask: 최저 sell
                # 음수 weight -> bid: 최저 sell, ask: 최고 buy
                if weight > 0:
                    prod_bid = max(od.buy_orders.keys()) if od.buy_orders else 0
                    prod_ask = min(od.sell_orders.keys()) if od.sell_orders else float("inf")
                    synthetic_bid += weight * prod_bid
                    synthetic_ask += weight * prod_ask
                    if prod_bid != 0:
                        bid_volume_candidates.append(od.buy_orders[prod_bid] // weight)
                    if prod_ask != float("inf"):
                        ask_volume_candidates.append((-od.sell_orders[prod_ask]) // weight)
                    else:  # weight < 0
                        prod_bid = max(od.buy_orders.keys()) if od.buy_orders else 0
                        prod_ask = min(od.sell_orders.keys()) if od.sell_orders else float("inf")
                        synthetic_bid += weight * prod_ask   # 음수 weight: bid는 ask 가격 사용
                        synthetic_ask += weight * prod_bid      # 음수 weight: ask는 bid 가격 사용
                        # 수정: 항상 buy side의 volume는 bid 후보로, sell side의 volume은 ask 후보로 사용
                        if prod_bid != 0:
                            bid_volume_candidates.append(od.buy_orders[prod_bid] // abs(weight))
                        if prod_ask != float("inf"):
                            ask_volume_candidates.append((-od.sell_orders[prod_ask]) // abs(weight))

            else:
                if weight > 0:
                    synthetic_bid += 0
                    synthetic_ask += weight * float("inf")
                else:
                    synthetic_bid += weight * float("inf")
                    synthetic_ask += 0
        synthetic_bid_volume = min(bid_volume_candidates) if bid_volume_candidates else 0
        synthetic_ask_volume = min(ask_volume_candidates) if ask_volume_candidates else 0
        if synthetic_bid > 0 and synthetic_bid_volume > 0:
            synthetic_order_depth.buy_orders[synthetic_bid] = synthetic_bid_volume
        if synthetic_ask < float("inf") and synthetic_ask_volume > 0:
            synthetic_order_depth.sell_orders[synthetic_ask] = -synthetic_ask_volume
        return synthetic_order_depth

    # --- synthetic 주문을 구성 주문으로 변환 ---
    def convert_synthetic_basket_orders(
        self, synthetic_orders: List[Order], order_depths: Dict[str, OrderDepth]
    ) -> Dict[str, List[Order]]:
        component_orders = {}
        for prod in BASKET_WEIGHTS.keys():
            component_orders[prod] = []
        synthetic_basket_order_depth = self.get_synthetic_basket_order_depth(order_depths)
        best_bid = max(synthetic_basket_order_depth.buy_orders.keys()) if synthetic_basket_order_depth.buy_orders else 0
        best_ask = min(synthetic_basket_order_depth.sell_orders.keys()) if synthetic_basket_order_depth.sell_orders else float("inf")
        for order in synthetic_orders:
            price = order.price
            quantity = order.quantity
            if quantity > 0 and price >= best_ask:
                for prod, weight in BASKET_WEIGHTS.items():
                    od = order_depths.get(prod, None)
                    if od is None or not od.sell_orders:
                        continue
                    prod_price = min(od.sell_orders.keys())
                    comp_order = Order(prod, prod_price, quantity * weight)
                    component_orders[prod].append(comp_order)
            elif quantity < 0 and price <= best_bid:
                for prod, weight in BASKET_WEIGHTS.items():
                    od = order_depths.get(prod, None)
                    if od is None or not od.buy_orders:
                        continue
                    prod_price = max(od.buy_orders.keys())
                    comp_order = Order(prod, prod_price, quantity * weight)
                    component_orders[prod].append(comp_order)
            else:
                continue
        return component_orders

    # --- 수정된 execute_spread_orders: synthetic basket를 기준으로 주문 실행 ---
    def execute_spread_orders(
        self,
        target_position: int,
        synthetic_position: int,
        order_depths: Dict[str, OrderDepth],
    ):
        if target_position == synthetic_position:
            return None
        target_quantity = abs(target_position - synthetic_position)
        composite_order_depth = self.get_synthetic_basket_order_depth(order_depths)
        if target_position > synthetic_position:
            # 목표가 더 롱이면, synthetic ask 가격 기준으로 매수
            ask_price = min(composite_order_depth.sell_orders.keys()) if composite_order_depth.sell_orders else float("inf")
            available_volume = abs(composite_order_depth.sell_orders[ask_price]) if ask_price != float("inf") else 0
            execute_volume = min(available_volume, target_quantity)
            synthetic_orders = [Order(Product.SYNTHETIC, ask_price, execute_volume)]
        else:
            # 목표가 더 숏이면, synthetic bid 가격 기준으로 매도
            bid_price = max(composite_order_depth.buy_orders.keys()) if composite_order_depth.buy_orders else 0
            available_volume = composite_order_depth.buy_orders[bid_price] if bid_price != 0 else 0
            execute_volume = min(available_volume, target_quantity)
            synthetic_orders = [Order(Product.SYNTHETIC, bid_price, -execute_volume)]
        aggregate_orders = self.convert_synthetic_basket_orders(synthetic_orders, order_depths)
        # synthetic 주문도 result에 포함시킴 (주문 변환 후, synthetic 주문으로 포지션 조정)
        aggregate_orders[Product.SYNTHETIC] = synthetic_orders
        return aggregate_orders

    # --- 수정된 spread_orders: synthetic basket 가격과 running_mean을 비교하여 z-score 산출 ---
    def spread_orders(
        self,
        order_depths: Dict[str, OrderDepth],
        synthetic_position: int,
        spread_data: Dict[str, Any],
    ):
        # synthetic basket의 주문 깊이와 midprice 산출
        composite_order_depth = self.get_synthetic_basket_order_depth(order_depths)
        composite_swmid = self.get_swmid(composite_order_depth)
        
        spread = composite_swmid - self.params[Product.SPREAD]["default_spread_mean"]
        spread_data["spread_history"].append(spread)
        # 유지 윈도우 길이 맞추기
        window = self.params[Product.SPREAD]["spread_std_window"]
        if(len(spread_data["spread_history"]) < window):
            return None
        elif len(spread_data["spread_history"]) > window:
            spread_data["spread_history"].pop(0)
        spread_std = np.std(spread_data["spread_history"]) if spread_data["spread_history"] else 1
        zscore = spread / spread_std
        
        if zscore >= self.params[Product.SPREAD]["zscore_threshold"]:
            if synthetic_position != -self.params[Product.SPREAD]["target_position"]:
                return self.execute_spread_orders(-self.params[Product.SPREAD]["target_position"], synthetic_position, order_depths)
        if zscore <= -self.params[Product.SPREAD]["zscore_threshold"]:
            if synthetic_position != self.params[Product.SPREAD]["target_position"]:
                return self.execute_spread_orders(self.params[Product.SPREAD]["target_position"], synthetic_position, order_depths)
        spread_data["prev_zscore"] = zscore
        return None

    def get_swmid(self, order_depth: OrderDepth) -> float:
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            best_bid_vol = abs(order_depth.buy_orders[best_bid])
            best_ask_vol = abs(order_depth.sell_orders[best_ask])
            if best_bid_vol + best_ask_vol == 0:
                return 0
            return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (best_bid_vol + best_ask_vol)
        elif order_depth.buy_orders:
            return max(order_depth.buy_orders.keys())
        elif order_depth.sell_orders:
            return min(order_depth.sell_orders.keys())
        else:
            return 0

    # --- 수정된 run: synthetic_position을 state.position[Product.SYNTHETIC] 기준으로 사용 ---
    def run(self, state: TradingState) -> tuple[dict[Symbol, List[Order]], int, str]:
        original_state = copy.deepcopy(state)
        traderObject = {}
        if state.traderData is not None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
        result = {}
        # 기존 상품들 (RAINFOREST_RESIN, KELP, SQUID_INK 등) 처리는 그대로 유지
        if Product.RAINFOREST_RESIN in self.params and Product.RAINFOREST_RESIN in state.order_depths:
            resin_position = state.position.get(Product.RAINFOREST_RESIN, 0)
            resin_take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                Product.RAINFOREST_RESIN,
                state.order_depths[Product.RAINFOREST_RESIN],
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                self.params[Product.RAINFOREST_RESIN]["take_width"],
                resin_position,
            )
            resin_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                Product.RAINFOREST_RESIN,
                state.order_depths[Product.RAINFOREST_RESIN],
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                self.params[Product.RAINFOREST_RESIN]["clear_width"],
                resin_position,
                buy_order_volume,
                sell_order_volume,
            )
            resin_make_orders, _, _ = self.make_orders(
                Product.RAINFOREST_RESIN,
                state.order_depths[Product.RAINFOREST_RESIN],
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                resin_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.RAINFOREST_RESIN]["disregard_edge"],
                self.params[Product.RAINFOREST_RESIN]["join_edge"],
                self.params[Product.RAINFOREST_RESIN]["default_edge"],
                True,
                self.params[Product.RAINFOREST_RESIN]["soft_position_limit"],
            )
            result[Product.RAINFOREST_RESIN] = resin_take_orders + resin_clear_orders + resin_make_orders
        if Product.KELP in self.params and Product.KELP in state.order_depths:
            kelp_position = state.position.get(Product.KELP, 0)
            kelp_fair_value = self.kelp_fair_value(state.order_depths[Product.KELP], traderObject)
            kelp_take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                Product.KELP,
                state.order_depths[Product.KELP],
                kelp_fair_value,
                self.params[Product.KELP]["take_width"],
                kelp_position,
                self.params[Product.KELP]["prevent_adverse"],
                self.params[Product.KELP]["adverse_volume"],
            )
            kelp_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                Product.KELP,
                state.order_depths[Product.KELP],
                kelp_fair_value,
                self.params[Product.KELP]["clear_width"],
                kelp_position,
                buy_order_volume,
                sell_order_volume,
            )
            kelp_make_orders, _, _ = self.make_orders(
                Product.KELP,
                state.order_depths[Product.KELP],
                kelp_fair_value,
                kelp_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.KELP]["disregard_edge"],
                self.params[Product.KELP]["join_edge"],
                self.params[Product.KELP]["default_edge"],
            )
            result[Product.KELP] = kelp_take_orders + kelp_clear_orders + kelp_make_orders
        if Product.SQUID_INK in self.params and Product.SQUID_INK in state.order_depths:
            squid_ink_position = state.position.get(Product.SQUID_INK, 0)
            squid_ink_orders = self.squid_ink_spread_order(
                state.order_depths[Product.SQUID_INK],
                squid_ink_position,
                traderObject
            )
            result[Product.SQUID_INK] = squid_ink_orders or []
        if Product.SPREAD not in traderObject:
            traderObject[Product.SPREAD] = {
                "spread_history": [],
                "prev_zscore": 0,
                # running_mean 저장 (synthetic basket 기준 midprice)
                "running_mean": None,
            }
        # synthetic basket 포지션: state.position의 Product.SYNTHETIC 기준 (없으면 0)
        synthetic_position = state.position.get(Product.SYNTHETIC, 0)
        spread_orders = self.spread_orders(
            state.order_depths,
            synthetic_position,
            traderObject[Product.SPREAD]
        )
        if spread_orders is not None:
            # spread 주문은 구성 주문들 + synthetic 주문으로 반환됨
            for prod, orders_list in spread_orders.items():
                result[prod] = orders_list
        conversions = 1  
        traderData = jsonpickle.encode(traderObject)
        logger.flush(original_state, result, conversions, traderData)
        return result, conversions, traderData

# 제출 아이디: 59f81e67-f6c6-4254-b61e-39661eac6141