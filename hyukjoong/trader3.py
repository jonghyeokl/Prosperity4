from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Any
import json
import jsonpickle
import math
import copy
from datamodel import (
    Listing, Observation, Order, OrderDepth, ProsperityEncoder,
    Symbol, Trade, TradingState,
)


# ============================================================
#  Logger (identical to round 2)
# ============================================================
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, List[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions, "", "",
        ]))
        max_item_length = (self.max_log_length - base_length) // 3
        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders), conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp, trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings):
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def compress_order_depths(self, order_depths):
        return {s: [od.buy_orders, od.sell_orders] for s, od in order_depths.items()}

    def compress_trades(self, trades):
        out = []
        for arr in trades.values():
            for t in arr:
                out.append([t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp])
        return out

    def compress_observations(self, observations):
        co = {}
        for p, o in observations.conversionObservations.items():
            co[p] = [o.bidPrice, o.askPrice, o.transportFees, o.exportTariff,
                     o.importTariff, o.sugarPrice, o.sunlightIndex]
        return [observations.plainValueObservations, co]

    def compress_orders(self, orders):
        out = []
        for arr in orders.values():
            for o in arr:
                out.append([o.symbol, o.price, o.quantity])
        return out

    def to_json(self, v):
        return json.dumps(v, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, v, n):
        return v if len(v) <= n else v[:n - 3] + "..."


logger = Logger()


# ============================================================
#  Trader
# ============================================================
class Trader:
    # ----- Product limits -----
    POSITION_LIMITS = {
        "VELVETFRUIT_EXTRACT": 200,
        "VEV_4000": 300,
        "VEV_4500": 300,
        # Phase 2+ 에서 추가: "HYDROGEL_PACK", "VEV_5000"~"VEV_5500" 등
    }

    # ----- Deep ITM vouchers: fair = S - K - THETA -----
    # 델타가 거의 1 이라 underlying (VELVETFRUIT_EXTRACT) 로 1:1 헷지 가능
    DEEP_ITM_VOUCHERS = {
        "VEV_4000": 4000,
        "VEV_4500": 4500,
    }
    # Voucher 별 theta 보정: fair 를 S-K 보다 THETA 만큼 낮게 봄
    # 크게 잡으면 buy 는 더 까다로워지고 sell 은 쉬워짐 (포지션 덜 쌓임)
    # 작게 잡으면 buy 공격적 (포지션 더 쌓임 but 평가손 위험)
    DEEP_ITM_THETA = {
        "VEV_4000": 0,
        "VEV_4500": 0,
    }

    # TAKE threshold (비대칭 - long bias)
    # fair = S - K - THETA 에서 더 보수적으로 갈 때 사용
    DEEP_ITM_TAKE_BUY_THRESHOLD = 0    # ask <= fair 이면 buy
    DEEP_ITM_TAKE_SELL_THRESHOLD = 20  # bid >= fair + 20 이어야 sell

    # MAKE threshold
    DEEP_ITM_MAKE_BUY_THRESHOLD = 0    # buy 지정가를 fair 까지 허용
    DEEP_ITM_MAKE_SELL_THRESHOLD = 20  # sell 지정가를 fair + 20 이상

    # 헷지 자체 on/off
    #   False: 헷지 안 함. Voucher 단독 포지션만 들고 있음 (델타 노출).
    #          VELVETFRUIT 한도 전체를 나중에 MM 등에 쓸 수 있음.
    #   True : 델타 헷지 함. 아래 HEDGE_PASSIVE 가 거래 방식 결정.
    HEDGE_ENABLED = False

    # 헷지 방식:
    #   True  = passive (best_bid/ask 에 지정가 포스트, 체결 안 되면 다음 tick 재시도)
    #   False = aggressive (크로스 체결, 즉시 확실한 헷지 but spread cost)
    HEDGE_PASSIVE = True

    # voucher 델타 합을 VELVETFRUIT 한도 내로 강제 (헷지 가능 범위)
    MAX_NET_VOUCHER_POS = 200

    # ==========================================================
    #  Utility
    # ==========================================================
    def get_best_bid_ask(self, order_depth: OrderDepth):
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        return best_bid, best_ask

    def get_mid_price(self, order_depth: OrderDepth):
        bid, ask = self.get_best_bid_ask(order_depth)
        if bid is not None and ask is not None:
            return (bid + ask) / 2
        return None

    # ==========================================================
    #  Deep ITM voucher trading (single product)
    # ==========================================================
    def trade_deep_itm(self, product: str, strike: int,
                       order_depth: OrderDepth, position: int,
                       S_mid: float, net_voucher_pos: int):
        """
        Returns (orders, expected_pos_delta).
        expected_pos_delta = 확실히 체결될 TAKE 수량 (+매수, -매도). MAKE 는 제외 (불확실).
        """
        orders: List[Order] = []
        limit = self.POSITION_LIMITS[product]

        buy_cap = limit - position
        sell_cap = limit + position

        # 헷지 ON 일 때만 net voucher 제약 (VELVETFRUIT 한도 내에서 헷지 가능해야 하니까)
        if self.HEDGE_ENABLED:
            net_buy_cap = max(0, self.MAX_NET_VOUCHER_POS - net_voucher_pos)
            net_sell_cap = max(0, self.MAX_NET_VOUCHER_POS + net_voucher_pos)
            buy_cap = min(buy_cap, net_buy_cap)
            sell_cap = min(sell_cap, net_sell_cap)

        theta = self.DEEP_ITM_THETA.get(product, 0)
        fair = S_mid - strike - theta
        best_bid, best_ask = self.get_best_bid_ask(order_depth)
        tk_buy = self.DEEP_ITM_TAKE_BUY_THRESHOLD
        tk_sell = self.DEEP_ITM_TAKE_SELL_THRESHOLD
        mk_buy = self.DEEP_ITM_MAKE_BUY_THRESHOLD
        mk_sell = self.DEEP_ITM_MAKE_SELL_THRESHOLD

        take_buy_qty = 0
        take_sell_qty = 0

        # ----- TAKE: 시장가가 fair 보다 우리 쪽으로 충분히 벗어나면 즉시 체결 -----
        if order_depth.sell_orders and buy_cap > 0:
            for ask_price, ask_vol in sorted(order_depth.sell_orders.items()):
                if ask_price <= fair - tk_buy:
                    q = min(-ask_vol, buy_cap)
                    if q > 0:
                        orders.append(Order(product, ask_price, q))
                        position += q
                        buy_cap -= q
                        take_buy_qty += q
                else:
                    break

        if order_depth.buy_orders and sell_cap > 0:
            for bid_price, bid_vol in sorted(order_depth.buy_orders.items(), reverse=True):
                if bid_price >= fair + tk_sell:
                    q = min(bid_vol, sell_cap)
                    if q > 0:
                        orders.append(Order(product, bid_price, -q))
                        position -= q
                        sell_cap -= q
                        take_sell_qty += q
                else:
                    break

        # ----- MAKE: 잔여 한도를 지정가로 포스트 -----
        if buy_cap > 0:
            safe_price = math.floor(fair - mk_buy)
            if best_bid is not None:
                price = min(best_bid + 1, safe_price)
            else:
                price = safe_price
            if price > 0:
                orders.append(Order(product, price, buy_cap))

        if sell_cap > 0:
            safe_price = math.ceil(fair + mk_sell)
            if best_ask is not None:
                price = max(best_ask - 1, safe_price)
            else:
                price = safe_price
            orders.append(Order(product, price, -sell_cap))

        expected_delta = take_buy_qty - take_sell_qty
        return orders, expected_delta

    # ==========================================================
    #  Delta hedge on VELVETFRUIT_EXTRACT
    # ==========================================================
    def hedge_velvetfruit(self, order_depth: OrderDepth,
                          current_pos: int, target_pos: int):
        """
        HEDGE_PASSIVE=True:
            best_bid+1 / best_ask-1 에 지정가 포스트. 체결되면 스프레드 안 먹음.
            체결 안 되면 delta 가 그대로 남아 다음 tick 재시도.
        HEDGE_PASSIVE=False:
            크로스(aggressive)로 즉시 체결. 스프레드 비용 발생.
        """
        limit = self.POSITION_LIMITS["VELVETFRUIT_EXTRACT"]
        target_pos = max(-limit, min(limit, target_pos))
        delta = target_pos - current_pos
        orders: List[Order] = []

        if delta == 0:
            return orders

        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        if self.HEDGE_PASSIVE:
            # 지정가 포스트. 크로스 안 함. spread 안에 들어가서 앞에 서기.
            if delta > 0:  # 매수
                if best_bid is not None and best_ask is not None:
                    price = min(best_bid + 1, best_ask - 1)
                elif best_bid is not None:
                    price = best_bid + 1
                elif best_ask is not None:
                    price = best_ask - 1
                else:
                    return orders
                orders.append(Order("VELVETFRUIT_EXTRACT", price, delta))
            else:  # delta < 0, 매도
                if best_bid is not None and best_ask is not None:
                    price = max(best_ask - 1, best_bid + 1)
                elif best_ask is not None:
                    price = best_ask - 1
                elif best_bid is not None:
                    price = best_bid + 1
                else:
                    return orders
                orders.append(Order("VELVETFRUIT_EXTRACT", price, delta))
            return orders

        # ----- Aggressive 모드 (크로스 체결) -----
        if delta > 0 and order_depth.sell_orders:
            need = delta
            for ask_price, ask_vol in sorted(order_depth.sell_orders.items()):
                if need <= 0:
                    break
                q = min(-ask_vol, need)
                if q > 0:
                    orders.append(Order("VELVETFRUIT_EXTRACT", ask_price, q))
                    need -= q
        elif delta < 0 and order_depth.buy_orders:
            need = -delta
            for bid_price, bid_vol in sorted(order_depth.buy_orders.items(), reverse=True):
                if need <= 0:
                    break
                q = min(bid_vol, need)
                if q > 0:
                    orders.append(Order("VELVETFRUIT_EXTRACT", bid_price, -q))
                    need -= q

        return orders

    # ==========================================================
    #  Main
    # ==========================================================
    def run(self, state: TradingState, day_num: int = 0):
        # day_num: 로컬 백테스터가 주입하는 day 인덱스 (Prosperity 서버에는 없음, 제출 시 제거)
        original_state = copy.deepcopy(state)

        traderObject = {}
        if state.traderData is not None and state.traderData != "":
            try:
                traderObject = jsonpickle.decode(state.traderData)
            except Exception:
                traderObject = {}

        result: dict = {}

        # ----- 1) Underlying mid -----
        velvet_od = state.order_depths.get("VELVETFRUIT_EXTRACT")
        S_mid = self.get_mid_price(velvet_od) if velvet_od is not None else None

        # ----- 2) Deep ITM vouchers -----
        expected_total_take = 0
        current_net_voucher = sum(
            state.position.get(p, 0) for p in self.DEEP_ITM_VOUCHERS
        )

        if S_mid is not None:
            for product, strike in self.DEEP_ITM_VOUCHERS.items():
                if product not in state.order_depths:
                    continue
                od = state.order_depths[product]
                pos = state.position.get(product, 0)

                orders, expected_delta = self.trade_deep_itm(
                    product, strike, od, pos, S_mid,
                    net_voucher_pos=current_net_voucher + expected_total_take,
                )
                if orders:
                    result[product] = orders
                expected_total_take += expected_delta

        # ----- 3) Hedge VELVETFRUIT_EXTRACT (HEDGE_ENABLED 면만) -----
        if self.HEDGE_ENABLED and velvet_od is not None:
            projected_net_voucher = current_net_voucher + expected_total_take
            target_velvet = -projected_net_voucher
            current_velvet = state.position.get("VELVETFRUIT_EXTRACT", 0)
            hedge_orders = self.hedge_velvetfruit(
                velvet_od, current_velvet, target_velvet,
            )
            if hedge_orders:
                result["VELVETFRUIT_EXTRACT"] = hedge_orders

        traderData = jsonpickle.encode(traderObject)
        conversions = 0
        logger.flush(original_state, result, conversions, traderData)
        return result, conversions, traderData