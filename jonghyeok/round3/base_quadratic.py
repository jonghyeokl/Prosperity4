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

    TRADE_VOUCHERS = {
        "VEV_5000": 5000,
        "VEV_5100": 5100,
        "VEV_5200": 5200,
        "VEV_5300": 5300,
        "VEV_5500": 5500,
    }

    # fair_iv(m) = a*m^2 + b*m + c
    IV_COEFFS = [
        0.01920208814884807,
        0.011969232861589801,
        0.2429184874002301,
    ]

    DAYS_PER_YEAR = 365.0

    THEO_DIFF_EMA_WINDOW = 20
    SWITCH_EMA_WINDOW = 100

    # IV_SCALPING_THRESHOLDS = {
    #     "VEV_5000": 0.32,
    #     "VEV_5100": 0.28,
    #     "VEV_5200": 0.26,
    #     "VEV_5300": 0.22,
    #     "VEV_5400": 0.14,
    #     "VEV_5500": 0.13,
    #     "VEV_6000": 0.0016,
    #     "VEV_6500": 0.0019,
    # }
    IV_SCALPING_THRESHOLDS = {
        "VEV_5000": 0.0,
        "VEV_5100": 0.0,
        "VEV_5200": 0.0,
        "VEV_5300": 0.0,
        "VEV_5400": 0.0,
        "VEV_5500": 0.0,
        "VEV_6000": 0.0,
        "VEV_6500": 0.0,
    }
    OPEN_THRESHOLD = 0.0
    CLOSE_THRESHOLD = 0.0

    LOW_VEGA_THRESHOLD = 0.5
    LOW_VEGA_THRESHOLD_ADJ = 0.5

    WARMUP_COUNT = 20

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

    def get_mid_price(self, order_depth: OrderDepth):
        best_bid, best_ask = self.get_best_bid_ask(order_depth)
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2
        return None
    
    def get_valid_mid_price(self, order_depth: OrderDepth, valid_volume: int):
        best_valid_bid, best_valid_ask = self.get_best_valid_bid_ask(order_depth, valid_volume)
        if best_valid_bid is not None and best_valid_ask is not None:
            return (best_valid_bid + best_valid_ask) / 2
        return None

    def norm_cdf(self, x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def norm_pdf(self, x: float) -> float:
        return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

    def bs_call_price_delta_vega(self, S: float, K: float, T: float, sigma: float):
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            intrinsic = max(S - K, 0.0)
            delta = 1.0 if S > K else 0.0
            vega = 0.0
            return intrinsic, delta, vega

        sqrt_t = math.sqrt(T)
        d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t

        price = S * self.norm_cdf(d1) - K * self.norm_cdf(d2)
        delta = self.norm_cdf(d1)
        vega = S * self.norm_pdf(d1) * sqrt_t

        return price, delta, vega

    def get_tte_years(self, timestamp: int, day_num: int) -> float:
        progress_days = timestamp / 1_000_000.0
        remaining_days = max(8.0 - day_num - progress_days, 1e-9)
        return remaining_days / self.DAYS_PER_YEAR

    def get_fair_iv(self, S: float, K: float, T: float) -> float:
        a, b, c = self.IV_COEFFS
        m = math.log(K / S) / math.sqrt(T)
        return a * m * m + b * m + c

    def ema(self, traderObject: dict, key: str, value: float, window: int):
        alpha = 2.0 / (window + 1.0)

        if key not in traderObject:
            traderObject[key] = value
            return value

        old_value = traderObject[key]
        new_value = alpha * value + (1.0 - alpha) * old_value
        traderObject[key] = new_value
        return new_value

    def add_buy_order(self, orders: List[Order], product: str, price: int, volume: int, buy_limit: int) -> int:
        volume = min(volume, buy_limit)
        if volume > 0:
            orders.append(Order(product, int(price), int(volume)))
            buy_limit -= volume
        return buy_limit

    def add_sell_order(self, orders: List[Order], product: str, price: int, volume: int, sell_limit: int) -> int:
        volume = min(volume, sell_limit)
        if volume > 0:
            orders.append(Order(product, int(price), -int(volume)))
            sell_limit -= volume
        return sell_limit

    def get_voucher_orders(self, product: str, state: TradingState, traderObject: dict, day_num: int) -> List[Order]:
        orders: List[Order] = []

        if self.UNDERLYING not in state.order_depths:
            return orders

        option_depth = state.order_depths[product]
        underlying_depth = state.order_depths[self.UNDERLYING]

        option_best_bid, option_best_ask = self.get_best_bid_ask(option_depth)
        underlying_best_bid, underlying_best_ask = self.get_best_bid_ask(underlying_depth)

        if (
            option_best_bid is None
            or option_best_ask is None
            or underlying_best_bid is None
            or underlying_best_ask is None
        ):
            return orders
        
        valid_volume = self.VALID_BID_ASK_VOLUME[product]
        option_valid_mid = self.get_valid_mid_price(option_depth, valid_volume)
        underlying_valid_mid = self.get_valid_mid_price(underlying_depth, valid_volume)

        K = self.TRADE_VOUCHERS[product]
        T = self.get_tte_years(state.timestamp, day_num)
        fair_iv = self.get_fair_iv(underlying_valid_mid, K, T)

        if fair_iv <= 0 or not math.isfinite(fair_iv):
            return orders

        theo, delta, vega = self.bs_call_price_delta_vega(
            S=underlying_valid_mid,
            K=K,
            T=T,
            sigma=fair_iv,
        )

        theo_diff = option_valid_mid - theo

        mean_key = f"{product}_theo_diff_ema"
        switch_key = f"{product}_switch_ema"
        count_key = f"{product}_count"

        mean_diff = self.ema(
            traderObject=traderObject,
            key=mean_key,
            value=theo_diff,
            window=self.THEO_DIFF_EMA_WINDOW,
        )

        abs_dev = abs(theo_diff - mean_diff)

        switch_mean = self.ema(
            traderObject=traderObject,
            key=switch_key,
            value=abs_dev,
            window=self.SWITCH_EMA_WINDOW,
        )

        traderObject[count_key] = traderObject.get(count_key, 0) + 1
        count = traderObject[count_key]

        position = state.position.get(product, 0)
        limit = self.POSITION_LIMITS[product]

        buy_limit = limit - position
        sell_limit = limit + position

        sell_signal = option_best_bid - theo - mean_diff
        buy_signal = option_best_ask - theo - mean_diff

        low_vega_adj = self.LOW_VEGA_THRESHOLD_ADJ if vega <= self.LOW_VEGA_THRESHOLD else 0.0
        open_threshold = self.OPEN_THRESHOLD + low_vega_adj

        if count < self.WARMUP_COUNT:
            return orders

        if switch_mean >= self.IV_SCALPING_THRESHOLDS[product]:
            if sell_signal >= open_threshold:
                sell_limit = self.add_sell_order(
                    orders=orders,
                    product=product,
                    price=option_best_bid,
                    volume=sell_limit,
                    sell_limit=sell_limit,
                )

            elif buy_signal <= -open_threshold:
                buy_limit = self.add_buy_order(
                    orders=orders,
                    product=product,
                    price=option_best_ask,
                    volume=buy_limit,
                    buy_limit=buy_limit,
                )

            # elif sell_signal >= self.CLOSE_THRESHOLD and position > 0:
            #     sell_limit = self.add_sell_order(
            #         orders=orders,
            #         product=product,
            #         price=option_best_bid,
            #         volume=position,
            #         sell_limit=sell_limit,
            #     )

            # elif buy_signal <= -self.CLOSE_THRESHOLD and position < 0:
            #     buy_limit = self.add_buy_order(
            #         orders=orders,
            #         product=product,
            #         price=option_best_ask,
            #         volume=-position,
            #         buy_limit=buy_limit,
            #     )

        # else:
        #     # IV scalping 기회가 없으면 기존 포지션 정리
        #     if position > 0:
        #         sell_limit = self.add_sell_order(
        #             orders=orders,
        #             product=product,
        #             price=option_best_bid,
        #             volume=position,
        #             sell_limit=sell_limit,
        #         )

        #     elif position < 0:
        #         buy_limit = self.add_buy_order(
        #             orders=orders,
        #             product=product,
        #             price=option_best_ask,
        #             volume=-position,
        #             buy_limit=buy_limit,
        #         )

        return orders

    def run(self, state: TradingState, day_num: int):
        original_state = copy.deepcopy(state)

        traderObject = {}
        if state.traderData is not None and state.traderData != "":
            try:
                traderObject = jsonpickle.decode(state.traderData)
            except Exception:
                traderObject = {}

        result: dict[Symbol, List[Order]] = {}

        for product in state.order_depths:
            orders: List[Order] = []

            if product in self.TRADE_VOUCHERS:
                orders = self.get_voucher_orders(product, state, traderObject, day_num)

            elif product == "HYDROGEL_PACK":
                pass

            elif product == "VELVETFRUIT_EXTRACT":
                pass
            
            elif product == "VEV_4000":
                pass
            
            elif product == "VEV_4500":
                pass

            # VEV_4000, VEV_4500은 명시적으로 거래하지 않음.
            result[product] = orders

        traderData = jsonpickle.encode(traderObject)
        conversions = 0

        logger.flush(original_state, result, conversions, traderData)

        return result, conversions, traderData