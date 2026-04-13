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
        self.max_log_length = 4000

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: Dict[Symbol, List[Order]], conversions: int, trader_data: str) -> None:
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

    def compress_state(self, state: TradingState, trader_data: str) -> List[Any]:
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

    def compress_listings(self, listings: Dict[Symbol, Listing]) -> List[List[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(self, order_depths: Dict[Symbol, OrderDepth]) -> Dict[Symbol, List[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]
        return compressed

    def compress_trades(self, trades: Dict[Symbol, List[Trade]]) -> List[List[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [trade.symbol, trade.price, trade.quantity, trade.buyer, trade.seller, trade.timestamp]
                )
        return compressed

    def compress_observations(self, observations: Observation) -> List[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice, observation.askPrice, observation.transportFees,
                observation.exportTariff, observation.importTariff, observation.sugarPrice,
                observation.sunlightIndex,
            ]
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: Dict[Symbol, List[Order]]) -> List[List[Any]]:
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

# BASKET_WEIGHTS: 각 상품별 가중치 (양수: synthetic 매수 시 매도 side, 음수: 매도 시 매수 side)
BASKET_WEIGHTS = {
    Product.PICNIC_BASKET1: 3,
    Product.PICNIC_BASKET2: -5,
    Product.CROISSANTS: 2,
    Product.JAMS: 1,
    Product.DJEMBES: -3,
}

REMAINING_PRODUCTS = [
    Product.CROISSANTS,
    Product.JAMS,
]

MIN_VOLUME = {
    Product.PICNIC_BASKET1: 4,
    Product.PICNIC_BASKET2: 6,
    Product.CROISSANTS: 25,
    Product.JAMS: 50,
    Product.DJEMBES: 15,
}

PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 1,
        "join_edge": 2,
        "default_edge": 4,
        "soft_position_limit": 25
    },
    Product.KELP: {
        "take_width": 1,
        "clear_width": 5,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.229,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1
    },
    Product.SQUID_INK: {
        "prevent_adverse": True,
        "adverse_volume": 15,
        "zscore_range": 400,
        "zscore_threshold": 4,
        "default_std_val": 20,
        "target_position": 50
    },
    Product.SPREAD: {
        "default_spread_mean": -14.76285, # -22
        "default_spread_std": 76.07966,
        "spread_std_window": 275, # 45
        "zscore_threshold": 3.4,
        "target_position": 20
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
            Product.CROISSANTS: 210, #250
            Product.JAMS: 330, # 350
            Product.DJEMBES: 60,
            Product.PICNIC_BASKET1: 60,
            Product.PICNIC_BASKET2: 100,
            Product.SYNTHETIC: 20,
        }

    # 이하 take_best_orders, market_make, clear_position_order, kelp_fair_value,
    # squid_ink_spread_order, take_orders, clear_orders, make_orders 등은 기존 구현과 동일하게 유지합니다.
    def take_best_orders(self, product: str, fair_value: int, take_width: float,
                         orders: List[Order], order_depth: OrderDepth, position: int,
                         buy_order_volume: int, sell_order_volume: int,
                         prevent_adverse: bool = False, adverse_volume: int = 0) -> (int, int):
        position_limit = self.LIMIT[product]
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -order_depth.sell_orders[best_ask]
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

    def market_make(self, product: str, orders: List[Order], bid: int, ask: int,
                    position: int, buy_order_volume: int, sell_order_volume: int) -> (int, int):
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))
        return buy_order_volume, sell_order_volume

    def clear_position_order(self, product: str, fair_value: float, width: int, orders: List[Order],
                               order_depth: OrderDepth, position: int,
                               buy_order_volume: int, sell_order_volume: int) -> (int, int):
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if position_after_take > 0:
            clear_quantity = sum(volume for price, volume in order_depth.buy_orders.items() if price >= fair_for_ask)
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)
        if position_after_take < 0:
            clear_quantity = sum(abs(volume) for price, volume in order_depth.sell_orders.items() if price <= fair_for_bid)
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)
        return buy_order_volume, sell_order_volume

    def kelp_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if order_depth.sell_orders and order_depth.buy_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= self.params[Product.KELP]["adverse_volume"]]
            filtered_bid = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= self.params[Product.KELP]["adverse_volume"]]
            mm_ask = min(filtered_ask) if filtered_ask else None
            mm_bid = max(filtered_bid) if filtered_bid else None
            if mm_ask is None or mm_bid is None:
                mmmid_price = (best_ask + best_bid) / 2 if traderObject.get("kelp_last_price") is None else traderObject["kelp_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2
            if traderObject.get("kelp_last_price") is not None:
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
    
    def take_orders(self, product: str, order_depth: OrderDepth, fair_value: float, take_width: float,
                    position: int, prevent_adverse: bool = False, adverse_volume: int = 0) -> (List[Order], int, int):
        orders: List[Order] = []
        bo = 0
        so = 0
        bo, so = self.take_best_orders(product, fair_value, take_width, orders, order_depth, position, bo, so, prevent_adverse, adverse_volume)
        return orders, bo, so

    def clear_orders(self, product: str, order_depth: OrderDepth, fair_value: float, clear_width: int,
                     position: int, bo: int, so: int) -> (List[Order], int, int):
        orders: List[Order] = []
        bo, so = self.clear_position_order(product, fair_value, clear_width, orders, order_depth, position, bo, so)
        return orders, bo, so

    def make_orders(self, product, order_depth: OrderDepth, fair_value: float, position: int,
                    bo: int, so: int, disregard_edge: float, join_edge: float, default_edge: float,
                    manage_position: bool = False, soft_position_limit: int = 0) -> (List[Order], int, int):
        orders: List[Order] = []
        asks_above_fair = [price for price in order_depth.sell_orders.keys() if price > fair_value + disregard_edge]
        bids_below_fair = [price for price in order_depth.buy_orders.keys() if price < fair_value - disregard_edge]
        best_ask_above_fair = min(asks_above_fair) if asks_above_fair else None
        best_bid_below_fair = max(bids_below_fair) if bids_below_fair else None
        ask = round(fair_value + default_edge)
        if best_ask_above_fair is not None:
            ask = best_ask_above_fair if abs(best_ask_above_fair - fair_value) <= join_edge else best_ask_above_fair - 1
        bid = round(fair_value - default_edge)
        if best_bid_below_fair is not None:
            bid = best_bid_below_fair if abs(fair_value - best_bid_below_fair) <= join_edge else best_bid_below_fair + 1
        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -soft_position_limit:
                bid += 1
        bo, so = self.market_make(product, orders, bid, ask, position, bo, so)
        return orders, bo, so

    # --- 기존 get_synthetic_basket_order_depth ---
    def get_synthetic_basket_order_depth(self, order_depths: Dict[str, OrderDepth], spread_data) -> OrderDepth:
        synthetic_order_depth = {}
        synthetic_bid = 0
        synthetic_ask = 0
        bid_volume_candidates = []
        ask_volume_candidates = []
        for prod, weight in BASKET_WEIGHTS.items():
            if prod in order_depths:
                od = order_depths[prod]
                #prod_bid: od.buy_orders에서 value가 MIN_VOLUME 이상인 key 중 최대값
                #prod_ask: od.sell_orders에서 value가 MIN_VOLUME 이상인 key 중 최소값
                prod_bid = max([price for price in od.buy_orders.keys() if abs(od.buy_orders[price]) >= MIN_VOLUME[prod]], default=0)
                prod_ask = min([price for price in od.sell_orders.keys() if abs(od.sell_orders[price]) >= MIN_VOLUME[prod]], default=float("inf"))
                if weight > 0:
                    synthetic_bid += weight * prod_bid
                    synthetic_ask += weight * prod_ask
                    bid_volume_candidates.append((abs(od.buy_orders[prod_bid]) if prod_bid != 0 else 0) // weight)
                    if prod_ask != float("inf"):
                        ask_volume_candidates.append((abs(-od.sell_orders[prod_ask])) // weight)
                else:
                    synthetic_bid += weight * prod_ask   # 음수 weight → best_bid 사용
                    synthetic_ask += weight * prod_bid      # 음수 weight → best_ask 사용
                    ask_volume_candidates.append((abs(od.buy_orders[prod_bid]) if prod_bid != 0 else 0) // abs(weight))
                    if prod_ask != float("inf"):
                        bid_volume_candidates.append((abs(od.sell_orders[prod_ask])) // abs(weight))
            else:
                if weight > 0:
                    synthetic_bid += 0
                    synthetic_ask += weight * float("inf")
                else:
                    synthetic_bid += weight * float("inf")
                    synthetic_ask += 0
        synthetic_order_depth['bid_volume_candidates'] = bid_volume_candidates
        synthetic_order_depth['ask_volume_candidates'] = ask_volume_candidates
        synthetic_bid_volume = min(bid_volume_candidates) if bid_volume_candidates else 0
        synthetic_ask_volume = min(ask_volume_candidates) if ask_volume_candidates else 0
        if synthetic_bid_volume > 0:
            synthetic_order_depth['bid_price'] = synthetic_bid
            synthetic_order_depth['bid_volume'] = synthetic_bid_volume
        if synthetic_ask_volume > 0:
            synthetic_order_depth['ask_price'] = synthetic_ask
            synthetic_order_depth['ask_volume'] = synthetic_ask_volume
        return synthetic_order_depth


    # --- 수정된 convert_synthetic_basket_orders: 기존 함수만 수정하여 알고리즘대로 동작하도록 ---
    def convert_synthetic_basket_orders(self, synthetic_orders: List[Order], order_depths: Dict[str, OrderDepth]) -> Dict[str, List[Order]]:
        component_orders = {}
        for prod in BASKET_WEIGHTS.keys():
            component_orders[prod] = []

        # synthetic_orders[0].quantity는 target synthetic 주문량 (양수: 매수, 음수: 매도)
        synthetic_qty = synthetic_orders[0].quantity if synthetic_orders else 0

        # 각 상품 주문량 = effective_qty * weight
        for prod, weight in BASKET_WEIGHTS.items():
            od = order_depths.get(prod)
            if od is None:
                continue
            comp_qty = int(synthetic_qty * weight)
            # 주문 가격 결정: 매수 synthetic → 양수 weight는 sell side, 음수 weight는 buy side; 매도 synthetic 반대
            if synthetic_qty >= 0:
                if weight > 0:
                    if not od.sell_orders:
                        continue
                    # od에서 value가 MIN_VOLUME 이상인 key 중 최소값
                    prod_price = min([price for price in od.sell_orders.keys() if abs(od.sell_orders[price]) >= MIN_VOLUME[prod]], default=float("inf"))
                else:
                    if not od.buy_orders:
                        continue
                    # od에서 value가 MIN_VOLUME 이상인 key 중 최대값
                    prod_price = max([price for price in od.buy_orders.keys() if abs(od.buy_orders[price]) >= MIN_VOLUME[prod]], default=0)
            else:
                if weight > 0:
                    if not od.buy_orders:
                        continue
                    # od에서 value가 MIN_VOLUME 이상인 key 중 최대값
                    prod_price = max([price for price in od.buy_orders.keys() if abs(od.buy_orders[price]) >= MIN_VOLUME[prod]], default=0)
                else:
                    if not od.sell_orders:
                        continue
                    # od에서 value가 MIN_VOLUME 이상인 key 중 최소값
                    prod_price = min([price for price in od.sell_orders.keys() if abs(od.sell_orders[price]) >= MIN_VOLUME[prod]], default=float("inf"))
            if comp_qty != 0:
                comp_order = Order(prod, prod_price, comp_qty)
                component_orders[prod].append(comp_order)
        return component_orders

    def execute_spread_orders(self, target_position: int, synthetic_position: int,
                              order_depths: Dict[str, OrderDepth], spread_data) -> Dict[str, List[Order]]:
        if target_position == synthetic_position:
            return None
        target_quantity = abs(target_position - synthetic_position)
        composite_order_depth = self.get_synthetic_basket_order_depth(order_depths, spread_data)
        if target_position > synthetic_position:
            ask_price = composite_order_depth['ask_price'] if ('ask_price' in composite_order_depth) else float("inf")
            available_volume = abs(composite_order_depth['ask_volume']) if ask_price != float("inf") else 0
            execute_volume = min(available_volume, target_quantity)
            synthetic_orders = [Order(Product.SYNTHETIC, ask_price, execute_volume)]
        else:
            bid_price = composite_order_depth['bid_price'] if ('bid_price' in composite_order_depth) else 0
            available_volume = composite_order_depth['bid_volume'] if bid_price != 0 else 0
            execute_volume = min(available_volume, target_quantity)
            synthetic_orders = [Order(Product.SYNTHETIC, bid_price, -execute_volume)]
        aggregate_orders = self.convert_synthetic_basket_orders(synthetic_orders, order_depths)
        aggregate_orders[Product.SYNTHETIC] = synthetic_orders
        return aggregate_orders

    def spread_orders(self, order_depths: Dict[str, OrderDepth], synthetic_position: int,
                      spread_data: Dict[str, Any]) -> Dict[str, List[Order]]:
        composite_order_depth = self.get_synthetic_basket_order_depth(order_depths, spread_data)
        composite_swmid = self.get_swmid(composite_order_depth)
        spread = composite_swmid - self.params[Product.SPREAD]["default_spread_mean"]
        spread_data.setdefault("spread_history", []).append(spread)
        window = self.params[Product.SPREAD]["spread_std_window"]
        if len(spread_data["spread_history"]) < window:
            return None
        elif len(spread_data["spread_history"]) > window:
            spread_data["spread_history"].pop(0)
        spread_std = float(np.std(spread_data["spread_history"])) if spread_data["spread_history"] else 1
        if spread_std == 0:
            return None
        zscore = spread / spread_std
        spread_data["prev_zscore"] = zscore
        if zscore >= self.params[Product.SPREAD]["zscore_threshold"]:
            if synthetic_position != -self.params[Product.SPREAD]["target_position"]:
                return self.execute_spread_orders(-self.params[Product.SPREAD]["target_position"], synthetic_position, order_depths, spread_data)
        if zscore <= -self.params[Product.SPREAD]["zscore_threshold"]:
            if synthetic_position != self.params[Product.SPREAD]["target_position"]:
                return self.execute_spread_orders(self.params[Product.SPREAD]["target_position"], synthetic_position, order_depths, spread_data)
        return None

    def get_swmid(self, order_depth: OrderDepth) -> float:
        if 'bid_price' in order_depth and 'ask_price' in order_depth:
            best_bid = order_depth['bid_price']
            best_ask = order_depth['ask_price']
            best_bid_vol = abs(order_depth['bid_volume'])
            best_ask_vol = abs(order_depth['ask_volume'])
            if best_bid_vol + best_ask_vol == 0:
                return 0
            return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (best_bid_vol + best_ask_vol)
        elif 'bid_price' in order_depth:
            return order_depth['bid_price']
        elif 'ask_price' in order_depth:
            return order_depth['ask_price']
        else:
            return 0

    def run(self, state: TradingState) -> (Dict[Symbol, List[Order]], int, str):
        original_state = copy.deepcopy(state)
        traderObject = {}
        if state.traderData:
            traderObject = jsonpickle.decode(state.traderData)
        result = {}
        if Product.RAINFOREST_RESIN in self.params and Product.RAINFOREST_RESIN in state.order_depths:
            result[Product.RAINFOREST_RESIN] = []
        if Product.KELP in self.params and Product.KELP in state.order_depths:
            result[Product.KELP] = []
        if Product.SQUID_INK in self.params and Product.SQUID_INK in state.order_depths:
            result[Product.SQUID_INK] = []
        if Product.SPREAD not in traderObject:
            traderObject[Product.SPREAD] = {"spread_history": [], "prev_zscore": 0}
        synthetic_position = - (state.position.get(Product.DJEMBES, 0) // 3)
        spread_orders = self.spread_orders(state.order_depths, synthetic_position, traderObject[Product.SPREAD])
        if spread_orders is not None:
            for prod, orders_list in spread_orders.items():
                result[prod] = orders_list
        conversions = 1
        traderData = jsonpickle.encode(traderObject)
        logger.flush(original_state, result, conversions, traderData)
        return result, conversions, traderData

# 제출 아이디: 59f81e67-f6c6-4254-b61e-39661eac6141
