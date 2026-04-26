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
        "HYDROGEL_PACK": 200,
        "VELVETFRUIT_EXTRACT": 200,
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

    def rolling_mean_std(self, values: list[float]):
        if len(values) == 0:
            return None, None
        mean_val = sum(values) / len(values)
        var = sum((x - mean_val) ** 2 for x in values) / len(values)
        return mean_val, math.sqrt(var)

    def update_ema(self, previous_ema: float | None, new_value: float, alpha: float) -> float:
        if previous_ema is None:
            return new_value
        return (1 - alpha) * previous_ema + alpha * new_value

    def run(self, state: TradingState, day_num: int):
        day_num=3
        original_state = copy.deepcopy(state)

        traderObject = {}
        if state.traderData is not None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            actual_pos = state.position.get(product, 0)

            if product not in self.POSITION_LIMITS:
                result[product] = orders
                continue

            limit = self.POSITION_LIMITS[product]
            best_bid, best_ask = self.get_best_bid_ask(order_depth)

            if best_bid is None or best_ask is None:
                result[product] = orders
                continue

            best_bid_vol = order_depth.buy_orders[best_bid]
            best_ask_vol = -order_depth.sell_orders[best_ask]
            mid_price = (best_bid + best_ask) / 2
            spread = best_ask - best_bid

            working_pos = actual_pos
            buy_remaining = limit - working_pos
            sell_remaining = limit + working_pos

            def place_buy(price: int, qty: int, reason: str):
                nonlocal working_pos, buy_remaining, sell_remaining, orders
                qty = max(0, min(qty, buy_remaining))
                if qty > 0:
                    orders.append(Order(product, price, qty))
                    working_pos += qty
                    buy_remaining = limit - working_pos
                    sell_remaining = limit + working_pos
                    logger.print(
                        f"[{product}] BUY reason={reason} px={price} qty={qty} "
                        f"actual_pos={actual_pos} working_pos={working_pos}"
                    )

            def place_sell(price: int, qty: int, reason: str):
                nonlocal working_pos, buy_remaining, sell_remaining, orders
                qty = max(0, min(qty, sell_remaining))
                if qty > 0:
                    orders.append(Order(product, price, -qty))
                    working_pos -= qty
                    buy_remaining = limit - working_pos
                    sell_remaining = limit + working_pos
                    logger.print(
                        f"[{product}] SELL reason={reason} px={price} qty={qty} "
                        f"actual_pos={actual_pos} working_pos={working_pos}"
                    )

            # =====================================================
            # HYDROGEL_PACK: rolling mean reversion
            # =====================================================
            if product == "HYDROGEL_PACK":
                if "hydrogel_ultra_history" not in traderObject:
                    traderObject["hydrogel_ultra_history"] = []
                if "hydrogel_fast_history" not in traderObject:
                    traderObject["hydrogel_fast_history"] = []
                if "hydrogel_slow_history" not in traderObject:
                    traderObject["hydrogel_slow_history"] = []

                ultra_hist = traderObject["hydrogel_ultra_history"]
                fast_hist = traderObject["hydrogel_fast_history"]
                slow_hist = traderObject["hydrogel_slow_history"]

                ultra_hist.append(mid_price)
                fast_hist.append(mid_price)
                slow_hist.append(mid_price)

                if len(ultra_hist) > 4:
                    ultra_hist.pop(0)
                if len(fast_hist) > 8:
                    fast_hist.pop(0)
                if len(slow_hist) > 30:
                    slow_hist.pop(0)

                ultra_fair, _ = self.rolling_mean_std(ultra_hist)
                fast_fair, _ = self.rolling_mean_std(fast_hist)
                slow_fair, slow_vol = self.rolling_mean_std(slow_hist)

                if ultra_fair is None:
                    ultra_fair = mid_price
                if fast_fair is None:
                    fast_fair = mid_price
                if slow_fair is None:
                    slow_fair = mid_price

                vol = max(slow_vol if slow_vol is not None else 1.0, 1.0)

                z_ultra = (mid_price - ultra_fair) / vol
                z_fast = (mid_price - fast_fair) / vol
                z_slow = (mid_price - slow_fair) / vol

                z_ask_fast = (best_ask - fast_fair) / vol
                z_bid_fast = (best_bid - fast_fair) / vol

                regime_gap = abs(fast_fair - slow_fair) / vol

                combined_signal = (
                    0.8 * z_slow
                    + 1.0 * z_fast
                    + 1.0 * z_ultra
                )

                base_target_strength = 75.0
                max_target = 200

                if regime_gap >= 1.5:
                    target_strength = 55.0
                    cap_target = 140
                elif regime_gap >= 0.8:
                    target_strength = 65.0
                    cap_target = 170
                else:
                    target_strength = base_target_strength
                    cap_target = max_target

                raw_target = int(round(-target_strength * combined_signal))
                raw_target = max(-cap_target, min(cap_target, raw_target))

                # -------------------------------------------------
                # asymmetric hysteresis
                # -------------------------------------------------
                enter_threshold = 0.75

                # short 쪽은 기존처럼 비교적 끈적하게 유지
                hold_threshold_short = 0.30
                exit_threshold_short = 0.10
                reverse_threshold_short = 0.55

                # long 쪽은 더 빨리 줄이고 더 빨리 반전 허용
                hold_threshold_long = 0.18
                exit_threshold_long = 0.18
                reverse_threshold_long = 0.35

                if regime_gap >= 1.5:
                    enter_threshold = 0.95

                    hold_threshold_short = 0.40
                    exit_threshold_short = 0.18
                    reverse_threshold_short = 0.70

                    hold_threshold_long = 0.25
                    exit_threshold_long = 0.22
                    reverse_threshold_long = 0.45

                elif regime_gap >= 0.8:
                    enter_threshold = 0.85

                    hold_threshold_short = 0.35
                    exit_threshold_short = 0.14
                    reverse_threshold_short = 0.62

                    hold_threshold_long = 0.22
                    exit_threshold_long = 0.20
                    reverse_threshold_long = 0.40

                target_position = raw_target

                # short 보유 중: 기존처럼 비교적 끈적하게 유지
                if actual_pos < 0:
                    if combined_signal >= hold_threshold_short:
                        target_position = min(target_position, actual_pos)
                    elif combined_signal >= exit_threshold_short:
                        target_position = min(target_position, int(round(actual_pos * 0.6)))
                    elif combined_signal <= -reverse_threshold_short:
                        target_position = raw_target
                    else:
                        target_position = min(target_position, 0)

                # long 보유 중: 더 빨리 줄이고 더 빨리 short 허용
                elif actual_pos > 0:
                    if combined_signal <= -hold_threshold_long:
                        target_position = max(target_position, actual_pos)
                    elif combined_signal <= -exit_threshold_long:
                        target_position = max(target_position, int(round(actual_pos * 0.35)))
                    elif combined_signal >= reverse_threshold_long:
                        target_position = raw_target
                    else:
                        target_position = min(target_position, 0)

                target_position = max(-cap_target, min(cap_target, target_position))
                inventory_gap = target_position - actual_pos

                taker_threshold = enter_threshold
                taker_size = 14

                reversal_boost = abs(z_ultra - z_fast)
                if reversal_boost >= 0.5:
                    taker_size = 18
                if regime_gap >= 1.5:
                    taker_size = max(8, taker_size - 4)

                aggressive_buy_done = False
                aggressive_sell_done = False

                if inventory_gap > 0 and z_ask_fast <= -taker_threshold:
                    qty = min(best_ask_vol, taker_size, inventory_gap)
                    if qty > 0:
                        place_buy(best_ask, qty, "FAST_REVERSAL_TAKER")
                        aggressive_buy_done = True

                if inventory_gap < 0 and z_bid_fast >= taker_threshold:
                    qty = min(best_bid_vol, taker_size, -inventory_gap)
                    if qty > 0:
                        place_sell(best_bid, qty, "FAST_REVERSAL_TAKER")
                        aggressive_sell_done = True

                working_gap = target_position - working_pos

                min_size = 3
                max_size = 14

                gap_mag = min(abs(working_gap), cap_target)
                base_size = int(round(min_size + (max_size - min_size) * gap_mag / cap_target))

                if regime_gap >= 1.5:
                    base_size = max(min_size, int(base_size * 0.7))
                elif regime_gap >= 0.8:
                    base_size = max(min_size, int(base_size * 0.85))

                buy_size = 0
                sell_size = 0
                bid_quote = best_bid
                ask_quote = best_ask

                if spread >= 2:
                    if working_gap >= 6:
                        bid_quote = best_bid + 1
                        ask_quote = best_ask
                        buy_size = base_size
                        sell_size = 0

                    elif working_gap <= -6:
                        bid_quote = best_bid
                        ask_quote = best_ask - 1
                        buy_size = 0
                        sell_size = base_size

                    else:
                        bid_quote = best_bid + 1
                        ask_quote = best_ask - 1

                        if working_gap > 0:
                            buy_size = base_size
                            sell_size = min_size
                        elif working_gap < 0:
                            buy_size = min_size
                            sell_size = base_size
                        else:
                            buy_size = min_size
                            sell_size = min_size
                else:
                    bid_quote = best_bid
                    ask_quote = best_ask

                    if working_gap > 0:
                        buy_size = max(min_size, base_size // 2)
                        sell_size = 0
                    elif working_gap < 0:
                        buy_size = 0
                        sell_size = max(min_size, base_size // 2)
                    else:
                        buy_size = min_size
                        sell_size = min_size

                if aggressive_buy_done:
                    buy_size = int(buy_size * 0.5)
                if aggressive_sell_done:
                    sell_size = int(sell_size * 0.5)

                buy_size = min(buy_size, buy_remaining)
                sell_size = min(sell_size, sell_remaining)

                if bid_quote >= best_ask:
                    bid_quote = best_bid
                if ask_quote <= best_bid:
                    ask_quote = best_ask

                if buy_size > 0 and bid_quote < best_ask:
                    place_buy(bid_quote, buy_size, "PASSIVE")
                if sell_size > 0 and ask_quote > best_bid:
                    place_sell(ask_quote, sell_size, "PASSIVE")

                logger.print(
                    f"[HYDROGEL ASYM] pos={actual_pos} raw_target={raw_target} target={target_position} "
                    f"gap={inventory_gap} working_gap={working_gap} mid={mid_price:.2f} "
                    f"u={ultra_fair:.2f} f={fast_fair:.2f} s={slow_fair:.2f} "
                    f"zu={z_ultra:.2f} zf={z_fast:.2f} zs={z_slow:.2f} "
                    f"combined={combined_signal:.2f} regime_gap={regime_gap:.2f}"
                )

            # =====================================================
            # VELVETFRUIT_EXTRACT: EMA fair mean reversion
            # =====================================================
            elif product == "VELVETFRUIT_EXTRACT":
                if "velvet_mid_history" not in traderObject:
                    traderObject["velvet_mid_history"] = []
                if "velvet_ema_fair" not in traderObject:
                    traderObject["velvet_ema_fair"] = mid_price

                hist = traderObject["velvet_mid_history"]
                hist.append(mid_price)
                if len(hist) > 30:
                    hist.pop(0)

                traderObject["velvet_ema_fair"] = self.update_ema(
                    traderObject["velvet_ema_fair"], mid_price, 0.08
                )
                fair_value = traderObject["velvet_ema_fair"]

                _, vol = self.rolling_mean_std(hist)
                vol = max(vol if vol is not None else 1.0, 1.0)

                z_mid = (mid_price - fair_value) / vol
                z_ask = (best_ask - fair_value) / vol
                z_bid = (best_bid - fair_value) / vol

                target_strength = 16.0
                max_target = 100
                target_position = int(round(-target_strength * z_mid))
                target_position = max(-max_target, min(max_target, target_position))

                inventory_gap = target_position - actual_pos

                mr_entry_threshold = 1.0
                taker_size = 20

                if inventory_gap > 0 and z_ask <= -mr_entry_threshold:
                    qty = min(best_ask_vol, taker_size, inventory_gap)
                    place_buy(best_ask, qty, "MR_TAKER")

                if inventory_gap < 0 and z_bid >= mr_entry_threshold:
                    qty = min(best_bid_vol, taker_size, -inventory_gap)
                    place_sell(best_bid, qty, "MR_TAKER")

                working_gap = target_position - working_pos
                gap_mag = min(abs(working_gap), max_target)

                min_size = 4
                max_size = 20
                base_size = int(round(min_size + (max_size - min_size) * gap_mag / max_target))

                buy_size = 0
                sell_size = 0
                bid_quote = best_bid
                ask_quote = best_ask

                if spread >= 2:
                    if working_gap >= 16:
                        bid_quote = best_bid + 1
                        ask_quote = best_ask
                        buy_size = base_size
                        sell_size = 0
                    elif working_gap <= -16:
                        bid_quote = best_bid
                        ask_quote = best_ask - 1
                        buy_size = 0
                        sell_size = base_size
                    else:
                        bid_quote = best_bid + 1
                        ask_quote = best_ask - 1
                        if working_gap > 0:
                            buy_size = base_size
                            sell_size = min_size
                        elif working_gap < 0:
                            buy_size = min_size
                            sell_size = base_size
                        else:
                            buy_size = min_size
                            sell_size = min_size
                else:
                    bid_quote = best_bid
                    ask_quote = best_ask
                    if working_gap > 0:
                        buy_size = max(min_size, base_size // 2)
                        sell_size = 0
                    elif working_gap < 0:
                        buy_size = 0
                        sell_size = max(min_size, base_size // 2)
                    else:
                        buy_size = min_size
                        sell_size = min_size

                buy_size = min(buy_size, buy_remaining)
                sell_size = min(sell_size, sell_remaining)

                if bid_quote >= best_ask:
                    bid_quote = best_bid
                if ask_quote <= best_bid:
                    ask_quote = best_ask

                if buy_size > 0 and bid_quote < best_ask:
                    place_buy(bid_quote, buy_size, "PASSIVE")
                if sell_size > 0 and ask_quote > best_bid:
                    place_sell(ask_quote, sell_size, "PASSIVE")

                logger.print(
                    f"[VELVET] pos={actual_pos} target={target_position} gap={inventory_gap} "
                    f"mid={mid_price:.2f} fair={fair_value:.2f} vol={vol:.2f} "
                    f"z_mid={z_mid:.2f} z_ask={z_ask:.2f} z_bid={z_bid:.2f}"
                )

            result[product] = orders

        traderData = jsonpickle.encode(traderObject)
        conversions = 0
        logger.flush(original_state, result, conversions, traderData)
        return result, conversions, traderData