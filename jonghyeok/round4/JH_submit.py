from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Any, Dict, Optional, Tuple
from statistics import NormalDist
import json
import jsonpickle
import math
import copy
from datamodel import (
    Listing, Observation, Order, OrderDepth, ProsperityEncoder,
    Symbol, Trade, TradingState,
)

NORMAL = NormalDist()

# 여기에 생성된 DAY_VOUCHER_HISTORY_MAP을 그대로 붙여넣으시면 됩니다.
# 구조: DAY_VOUCHER_HISTORY_MAP[day_num][voucher] = list[(m, iv)]
DAY_VOUCHER_HISTORY_MAP: Dict[int, Dict[str, List[Tuple[float, float]]]] = {4: {'VEV_5000': [(-0.4413793756, 0.2418979149),
                  (-0.4423015364, 0.2422897444),
                  (-0.4432234127, 0.2427130346),
                  (-0.4477875088, 0.244749404),
                  (-0.4418791187, 0.2482858796),
                  (-0.4396124093, 0.2525659733),
                  (-0.4400789437, 0.2351367111),
                  (-0.4373557348, 0.2453557386),
                  (-0.4369110605, 0.2460221348),
                  (-0.4392006217, 0.240911191),
                  (-0.4373886557, 0.2400989313),
                  (-0.4405895295, 0.2477067156),
                  (-0.4396891898, 0.2472948656),
                  (-0.440155821, 0.2413373413),
                  (-0.4415337997, 0.2419679866),
                  (-0.4420004194, 0.2359918718),
                  (-0.4411001946, 0.2355971823),
                  (-0.4429337502, 0.2364065819),
                  (-0.4456780034, 0.2438170214),
                  (-0.4456891282, 0.2376192513),
                  (-0.4479774899, 0.2448523664),
                  (-0.4452558908, 0.2498460474),
                  (-0.445267001, 0.2498546276),
                  (-0.4457336239, 0.2376421318),
                  (-0.4411882924, 0.2533224616),
                  (-0.4457558939, 0.2376564322),
                  (-0.444400163, 0.2432707481),
                  (-0.4439556764, 0.236867053),
                  (-0.444422451, 0.2432707481),
                  (-0.4421549208, 0.2422511334),
                  (-0.4430775383, 0.2364823737),
                  (-0.4421770565, 0.2484546237),
                  (-0.4376289401, 0.2340842056),
                  (-0.435815389, 0.2394339652),
                  (-0.4394747827, 0.2472233639),
                  (-0.4376617369, 0.2463968039),
                  (-0.4404086641, 0.2353111753),
                  (-0.4417873714, 0.2482844496),
                  (-0.4404306596, 0.241488925),
                  (-0.4367931751, 0.2398715558),
                  (-0.4395407047, 0.2410999556),
                  (-0.4377273525, 0.2402848358),
                  (-0.4395626727, 0.2472734151),
                  (-0.4400296964, 0.241323041),
                  (-0.4363915745, 0.24582622),
                  (-0.436858685, 0.2399073067),
                  (-0.4359571424, 0.2562168517),
                  (-0.434142733, 0.2439628849),
                  (-0.4332407292, 0.2321650991),
                  (-0.4327951533, 0.2494556479)],
     'VEV_5100': [(-0.252449879, 0.2313499794),
                  (-0.2533673282, 0.2319148431),
                  (-0.2542844924, 0.232488287),
                  (-0.2588438761, 0.2353140354),
                  (-0.2529307733, 0.2378637872),
                  (-0.2506593508, 0.2343816528),
                  (-0.2511211717, 0.2325740891),
                  (-0.248393249, 0.2329551933),
                  (-0.2479438605, 0.2326813416),
                  (-0.2502287073, 0.2320278158),
                  (-0.2484120263, 0.2350223083),
                  (-0.2516081849, 0.2349922776),
                  (-0.2507031297, 0.2344224088),
                  (-0.2511650449, 0.2367676656),
                  (-0.2525383074, 0.2314257712),
                  (-0.2530002104, 0.2296124873),
                  (-0.2520952686, 0.2290547738),
                  (-0.2539241069, 0.2322909422),
                  (-0.2566636425, 0.2339898233),
                  (-0.2566700491, 0.2318876724),
                  (-0.2589536924, 0.2354141378),
                  (-0.2562273746, 0.2337302721),
                  (-0.2562337657, 0.2358452933),
                  (-0.2566956692, 0.231910553),
                  (-0.2521456178, 0.2374076061),
                  (-0.2567084992, 0.2319219933),
                  (-0.2553480478, 0.2331961541),
                  (-0.2548988403, 0.2329130073),
                  (-0.2553608937, 0.2332025893),
                  (-0.2530886419, 0.2338475349),
                  (-0.2540065375, 0.2302631531),
                  (-0.2531013334, 0.2339083114),
                  (-0.2485484943, 0.233106777),
                  (-0.2467302202, 0.2340012636),
                  (-0.2503848905, 0.2322151504),
                  (-0.248567121, 0.2351696019),
                  (-0.2513093241, 0.2307043188),
                  (-0.2526833071, 0.2357187352),
                  (-0.2513218705, 0.2327599936),
                  (-0.2476796609, 0.2346147485),
                  (-0.250422465, 0.2363515255),
                  (-0.248604387, 0.2352039227),
                  (-0.250434981, 0.2363636808),
                  (-0.2508972782, 0.2304597828),
                  (-0.2472544294, 0.2323131077),
                  (-0.2477168126, 0.2346504993),
                  (-0.2468105424, 0.2340784855),
                  (-0.2449914051, 0.2329273076),
                  (-0.2440846729, 0.2323538637),
                  (-0.2436343684, 0.2340606101)],
     'VEV_5200': [(-0.0671891492, 0.2398358049),
                  (-0.0681019782, 0.2395762537),
                  (-0.0690145219, 0.2404954084),
                  (-0.0735692847, 0.2402451524),
                  (-0.0676515606, 0.2415196707),
                  (-0.0653755165, 0.2404174715),
                  (-0.0658327156, 0.239693159),
                  (-0.0631001705, 0.2404825381),
                  (-0.0626461595, 0.2400310047),
                  (-0.0649263832, 0.2399802384),
                  (-0.063105079, 0.23932135),
                  (-0.0662966139, 0.2413759522),
                  (-0.0653869347, 0.2404596575),
                  (-0.0658442255, 0.239735345),
                  (-0.0672128633, 0.2399309022),
                  (-0.0676701413, 0.2392008696),
                  (-0.0667605741, 0.2394840164),
                  (-0.0685847867, 0.2401293195),
                  (-0.0713196962, 0.2404732429),
                  (-0.0713214764, 0.240479678),
                  (-0.0736004929, 0.2403470423),
                  (-0.070869548, 0.2400388699),
                  (-0.0708713116, 0.24004459),
                  (-0.0713285873, 0.2381068931),
                  (-0.0667739077, 0.2419150753),
                  (-0.0713321606, 0.2393184899),
                  (-0.0699670804, 0.2391572535),
                  (-0.0695132437, 0.2387096527),
                  (-0.0699706675, 0.2403688504),
                  (-0.0676937858, 0.2392845266),
                  (-0.0686070512, 0.239013535),
                  (-0.0676972165, 0.2404918333),
                  (-0.0631397465, 0.2382656269),
                  (-0.0613168411, 0.239962363),
                  (-0.0649668798, 0.2389381007),
                  (-0.0631444783, 0.2406584324),
                  (-0.0658820491, 0.2386828395),
                  (-0.0672513994, 0.2388723191),
                  (-0.0658853298, 0.2398847836),
                  (-0.0622384868, 0.2385627166),
                  (-0.0649766572, 0.2401661429),
                  (-0.0631539451, 0.2383188957),
                  (-0.0649799047, 0.2401779407),
                  (-0.0654375672, 0.2382645544),
                  (-0.0617900832, 0.2405004135),
                  (-0.062247831, 0.2397864687),
                  (-0.061336925, 0.2412315187),
                  (-0.0595131515, 0.2393617485),
                  (-0.0586017829, 0.2384261483),
                  (-0.0581468415, 0.2403259493)],
     'VEV_5300': [(0.1145425893, 0.2430197771),
                  (0.1136342925, 0.2434788182),
                  (0.1127262813, 0.2426615534),
                  (0.1081760514, 0.24355604),
                  (0.1140983087, 0.2439074711),
                  (0.1163788863, 0.2434230468),
                  (0.1159262212, 0.2430204921),
                  (0.1186633004, 0.2441913329),
                  (0.1191218461, 0.2433300946),
                  (0.1168461572, 0.2425807564),
                  (0.1186719967, 0.2429368351),
                  (0.1154849974, 0.2439117612),
                  (0.1163992125, 0.2434659479),
                  (0.1159464579, 0.2443321913),
                  (0.1145823567, 0.2431102268),
                  (0.1141296156, 0.2427048119),
                  (0.1150437201, 0.2422640038),
                  (0.1132240452, 0.2431645681),
                  (0.1104936737, 0.2419536862),
                  (0.1104964318, 0.2419597639),
                  (0.1082219539, 0.2436593601),
                  (0.1109574377, 0.2423866292),
                  (0.1109602135, 0.2436543549),
                  (0.1105074775, 0.2419840745),
                  (0.115066697, 0.2448512939),
                  (0.1105129845, 0.2419962298),
                  (0.1118826054, 0.2432417899),
                  (0.1123409831, 0.2424005721),
                  (0.1118881006, 0.2432575203),
                  (0.114169524, 0.242789899),
                  (0.1132608007, 0.2432435775),
                  (0.1141751778, 0.2440722826),
                  (0.1187371905, 0.2430705433),
                  (0.120564639, 0.2421181402),
                  (0.1169191437, 0.242730195),
                  (0.1187460889, 0.2430884187),
                  (0.1160130622, 0.2419268731),
                  (0.1146482563, 0.2432475101),
                  (0.1160188707, 0.2419386709),
                  (0.1196702588, 0.2426390303),
                  (0.1169366338, 0.2427684485),
                  (0.1187638917, 0.2431248846),
                  (0.1169424783, 0.2440572672),
                  (0.1164893622, 0.2423784065),
                  (0.120141393, 0.2443597195),
                  (0.1196881923, 0.2426754962),
                  (0.1206036458, 0.2447780046),
                  (0.1224319671, 0.2425353528),
                  (0.1233478839, 0.2420516436),
                  (0.1238073738, 0.2424584885)],
     'VEV_5400': [(0.2928772732, 0.2281102359),
                  (0.2919734239, 0.2277076812),
                  (0.2910698605, 0.2296997194),
                  (0.2865240787, 0.2295681562),
                  (0.2924507845, 0.2304969637),
                  (0.2947358109, 0.2240560877),
                  (0.2942875949, 0.2238051167),
                  (0.2970291236, 0.2253509841),
                  (0.297492119, 0.2256141105),
                  (0.2952208803, 0.2243356595),
                  (0.2970511703, 0.2253681445),
                  (0.2938686218, 0.2261896994),
                  (0.294787288, 0.2240954137),
                  (0.2943389849, 0.2264585459),
                  (0.2929793355, 0.2256891873),
                  (0.2925310466, 0.2254332111),
                  (0.2934496035, 0.2233460755),
                  (0.2916343814, 0.2275375071),
                  (0.288908463, 0.2309788854),
                  (0.2889156745, 0.2309846056),
                  (0.2866456505, 0.2296653986),
                  (0.2893855885, 0.2312613173),
                  (0.2893928187, 0.2312670374),
                  (0.2889445374, 0.2310074861),
                  (0.2935082121, 0.2286214733),
                  (0.2889589551, 0.2310189264),
                  (0.2903330318, 0.2293064599),
                  (0.2907958657, 0.2295767365),
                  (0.2903474396, 0.2318311861),
                  (0.2926333198, 0.2281266813),
                  (0.2917290537, 0.2276104388),
                  (0.2926478882, 0.2281395516),
                  (0.2972143587, 0.2254925575),
                  (0.2990462653, 0.2265243275),
                  (0.2954052285, 0.2244758029),
                  (0.2972366325, 0.2255090029),
                  (0.294508065, 0.2265879641),
                  (0.2931477186, 0.2258143153),
                  (0.2945227927, 0.223983871),
                  (0.298178641, 0.2260452659),
                  (0.2954494765, 0.2245101237),
                  (0.2972811952, 0.2255426087),
                  (0.2954642429, 0.224521564),
                  (0.2950155884, 0.2242698779),
                  (0.2986720809, 0.2263305578),
                  (0.2982233424, 0.2260788717),
                  (0.2991432584, 0.2265979743),
                  (0.3009760426, 0.2276304593),
                  (0.3018964225, 0.2281495619),
                  (0.3023603759, 0.2284119732)],
     'VEV_5500': [(0.4679395757, 0.2454301004),
                  (0.4670400922, 0.2450354109),
                  (0.4661408951, 0.2446407213),
                  (0.4615994798, 0.2426486831),
                  (0.4675305524, 0.2452542062),
                  (0.469819946, 0.2462595205),
                  (0.4693760975, 0.2460650358),
                  (0.472121994, 0.247269125),
                  (0.4725893576, 0.24747505),
                  (0.4703224874, 0.2464826059),
                  (0.4721571461, 0.2472891455),
                  (0.4689789667, 0.2458962917),
                  (0.4699020024, 0.2463024216),
                  (0.4694580691, 0.2461079369),
                  (0.4681027898, 0.2455144725),
                  (0.4676588713, 0.2453214178),
                  (0.4685817991, 0.2457275476),
                  (0.466770948, 0.2449324484),
                  (0.464049401, 0.2437397995),
                  (0.4640609843, 0.2437455196),
                  (0.4617953323, 0.2427502155),
                  (0.4645396427, 0.2439571648),
                  (0.4645512457, 0.2439628849),
                  (0.4641073375, 0.2437684002),
                  (0.4686753856, 0.2457761688),
                  (0.4641305022, 0.2437827005),
                  (0.465508953, 0.2443890352),
                  (0.4659761612, 0.2445949602),
                  (0.4655321099, 0.2444004755),
                  (0.4678223651, 0.2454072199),
                  (0.4669224743, 0.2450125303),
                  (0.4678456845, 0.2454186602),
                  (0.472416531, 0.2474235687),
                  (0.474252814, 0.2482272482),
                  (0.4706161538, 0.2466370497),
                  (0.4724519348, 0.2474435892),
                  (0.4697277446, 0.2462480803),
                  (0.4683717759, 0.2456560459),
                  (0.469751228, 0.2462623806),
                  (0.4734114545, 0.2478668795),
                  (0.4706866687, 0.2466742306),
                  (0.4725227663, 0.2474807701),
                  (0.4707101933, 0.2466856709),
                  (0.4702659184, 0.2464911861),
                  (0.4739267909, 0.2480985451),
                  (0.4734824327, 0.2479040604),
                  (0.4744067292, 0.2483101902),
                  (0.4762438943, 0.2491167297),
                  (0.4771686555, 0.2495228596),
                  (0.4776369905, 0.2497273545)]}}



# ============================================================
#  Logger
# ============================================================
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
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
        lo, hi = 0, min(len(value), max_length)
        out = ""

        while lo <= hi:
            mid = (lo + hi) // 2

            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."

            encoded_candidate = json.dumps(candidate)

            if len(encoded_candidate) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1

        return out

logger = Logger()


# ============================================================
#  Black-Scholes (call, r=0)
# ============================================================
def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_call_with_greeks(S, K, T, sigma):
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return max(S - K, 0.0), (1.0 if S > K else 0.0), 0.0

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    price = S * _norm_cdf(d1) - K * _norm_cdf(d2)
    delta = _norm_cdf(d1)
    vega = S * _norm_pdf(d1) * sqrtT

    return price, delta, vega


def implied_vol(V, S, K, T, tol=1e-4, max_iter=50):
    """Bisection IV. Returns None if no solution in range."""
    if T <= 0 or V <= 0 or S <= 0 or K <= 0:
        return None

    intrinsic = max(S - K, 0.0)

    if V < intrinsic - 1e-2 or V > S + 1e-2:
        return None

    lo, hi = 1e-3, 3.0

    f_lo = bs_call_with_greeks(S, K, T, lo)[0] - V
    f_hi = bs_call_with_greeks(S, K, T, hi)[0] - V

    if f_lo * f_hi > 0:
        return None

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = bs_call_with_greeks(S, K, T, mid)[0] - V

        if abs(f_mid) < tol:
            return mid

        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid

    return 0.5 * (lo + hi)


def fit_quadratic_from_points(points: List[Tuple[float, float]]) -> Optional[Tuple[float, float, float]]:
    """OLS fit iv = a*m^2 + b*m + c. Returns None on singular."""
    n = len(points)

    if n < 4:
        return None

    Sx4 = Sx3 = Sx2 = Sx1 = 0.0
    Sy = Syx = Syx2 = 0.0

    for m, v in points:
        m2 = m * m
        Sx4 += m2 * m2
        Sx3 += m2 * m
        Sx2 += m2
        Sx1 += m
        Sy += v
        Syx += v * m
        Syx2 += v * m2

    # 3x3 normal equations
    M = [
        [Sx4, Sx3, Sx2, Syx2],
        [Sx3, Sx2, Sx1, Syx],
        [Sx2, Sx1, float(n), Sy],
    ]

    try:
        for i in range(3):
            pivot = M[i][i]

            if abs(pivot) < 1e-15:
                swap = None
                for k in range(i + 1, 3):
                    if abs(M[k][i]) > 1e-15:
                        swap = k
                        break

                if swap is None:
                    return None

                M[i], M[swap] = M[swap], M[i]
                pivot = M[i][i]

            for j in range(i + 1, 3):
                factor = M[j][i] / pivot
                for c in range(i, 4):
                    M[j][c] -= factor * M[i][c]

        x = [0.0, 0.0, 0.0]

        for i in range(2, -1, -1):
            x[i] = M[i][3]
            for j in range(i + 1, 3):
                x[i] -= M[i][j] * x[j]
            x[i] /= M[i][i]

        return x[0], x[1], x[2]

    except Exception:
        return None


# ============================================================
#  Trader
# ============================================================
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

    # 실제 거래 대상은 유지: VEV_5000~VEV_5300
    ATM_VOUCHERS = {
        "VEV_5000": 5000,
        "VEV_5100": 5100,
        "VEV_5200": 5200,
        "VEV_5300": 5300,
    }

    # IV curve fitting에는 결과가 가장 좋았던 VEV_5000~VEV_5500 사용
    IV_CURVE_FIT_VOUCHERS = {
        "VEV_5000": 5000,
        "VEV_5100": 5100,
        "VEV_5200": 5200,
        "VEV_5300": 5300,
        "VEV_5400": 5400,
        "VEV_5500": 5500,
    }

    # Must Buy 0
    MUST_BUY_0_VOUCHERS = [
        "VEV_6000",
        "VEV_6500",
    ]

    # ----- Deep ITM vouchers: fair = S - K -----
    DEEP_ITM_VOUCHERS = {
        "VEV_4000": 4000,
        "VEV_4500": 4500,
    }

    # ----- Rolling smile fit -----
    SMILE_WINDOW_PER_VOUCHER = 150
    SMILE_MIN_POINTS_FOR_FIT = 300

    # Valid volume
    VALID_BID_ASK_VOLUME = {
        "HYDROGEL_PACK": 10,
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

    DAYS_PER_YEAR = 365.0

    THEO_DIFF_WINDOW = 20
    SCALE_WINDOW = 100
    WARMUP_TICKS = 30

    ENABLE_MAKE = True

    Z_SCORE_PRODUCTS = [
        "HYDROGEL_PACK",
        "VELVETFRUIT_EXTRACT",
    ]

    Z_SCORE_PARAMS = {
        "HYDROGEL_PACK": {
            "valid_mid_history_length": 500,
            "threshold_param_beta": 0.15,
            "threshold_param_alpha": 0.025,
        },
        "VELVETFRUIT_EXTRACT": {
            "valid_mid_history_length": 400,
            "threshold_param_beta": 0.11,
            "threshold_param_alpha": 1e-6,
            "threshold": 2.1,
        },
    }

    MIN_STD = 1e-9
    P_EPS = 1e-6

    # EMA_t 기준 beta 평균회귀 파라미터
    # IV curve fitting은 VEV_5000~VEV_5500, 실제 거래는 VEV_5000~VEV_5300
    MEAN_REVERSION_PARAMS = {
        "VEV_5000": {"ema_window": 50, "beta": -1.003},
        "VEV_5100": {"ema_window": 50, "beta": -0.998},
        "VEV_5200": {"ema_window": 30, "beta": -0.993},
        "VEV_5300": {"ema_window": 20, "beta": -0.986},
    }

    # ==========================================================
    # VEV_5400 EMA z-score trading
    # 기존 Z_SCORE_PRODUCTS / Z_SCORE_PARAMS와 완전히 분리해서 사용합니다.
    # window=1000, threshold=1.25 고정.
    # ==========================================================
    EMA_ZSCORE_PRODUCTS = {
        "VEV_5400": {
            "ema_window": 1000,
            "z_score_threshold": 1.25,
            "min_std": 1e-9,
        },
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

    def get_tte_years(self, timestamp: int, day_num: int) -> float:
        progress_days = timestamp / 1_000_000.0
        remaining_days = max(8.0 - day_num - progress_days, 1e-9)
        return remaining_days / self.DAYS_PER_YEAR

    def ema(self, traderObject: dict, key: str, value: float, window: int) -> float:
        alpha = 2.0 / (window + 1.0)

        if key not in traderObject:
            traderObject[key] = value
            return value

        new_value = alpha * value + (1.0 - alpha) * traderObject[key]
        traderObject[key] = new_value

        return new_value

    def init_smile_history_for_day(self, traderObject: dict, day_num: int) -> None:
        """
        day_num별 초기 smile history를 세팅합니다.
        - DAY_VOUCHER_HISTORY_MAP에 day_num key가 없으면 빈 history로 시작합니다.
        - 있으면 해당 day의 이전 데이터 history를 초기값으로 넣습니다.
        - 길이는 상품별 SMILE_WINDOW_PER_VOUCHER개로 제한합니다.
        - day가 바뀌면 history를 새로 초기화합니다.
        - 각 product별 is_avged flag도 초기화합니다.
        """
        hist_key = "smile_hist"
        hist_day_key = "smile_hist_day"
        is_avged_key = "smile_hist_is_avged"

        if traderObject.get(hist_day_key) == day_num and hist_key in traderObject:
            return

        preload = DAY_VOUCHER_HISTORY_MAP.get(day_num, {})
        hist = {}
        is_avged = {}

        for product in self.IV_CURVE_FIT_VOUCHERS:
            raw_points = preload.get(product, [])
            clean_points = []

            for item in raw_points:
                if item is None or len(item) < 2:
                    continue

                try:
                    m = float(item[0])
                    iv = float(item[1])
                except Exception:
                    continue

                if math.isfinite(m) and math.isfinite(iv) and iv > 0:
                    clean_points.append((m, iv))

            hist[product] = clean_points[-self.SMILE_WINDOW_PER_VOUCHER:]

            # preload는 이미 연속 두 점 평균으로 만들어진 history라고 가정
            is_avged[product] = True

        traderObject[hist_key] = hist
        traderObject[is_avged_key] = is_avged
        traderObject[hist_day_key] = day_num

    # ==========================================================
    #  Deep ITM voucher trading: fair = S - K
    # ==========================================================
    def get_deep_itm_voucher_orders(self, product: str, state: TradingState) -> List[Order]:
        orders: List[Order] = []

        if self.UNDERLYING not in state.order_depths:
            return orders

        voucher_depth = state.order_depths[product]
        underlying_depth = state.order_depths[self.UNDERLYING]

        underlying_mid = self.get_valid_mid_price(
            underlying_depth,
            self.VALID_BID_ASK_VOLUME[self.UNDERLYING],
        )

        if underlying_mid is None:
            return orders

        strike = self.DEEP_ITM_VOUCHERS[product]
        fair = underlying_mid - strike

        if fair <= 0:
            return orders

        best_bid, best_ask = self.get_best_bid_ask(voucher_depth)

        if best_bid is None or best_ask is None:
            return orders

        position = state.position.get(product, 0)
        limit = self.POSITION_LIMITS[product]

        buy_limit = limit - position
        sell_limit = limit + position

        for bid_price, bid_vol in sorted(voucher_depth.buy_orders.items(), reverse=True):
            if bid_price > fair:
                sell_qty = min(bid_vol, sell_limit)
                if sell_qty > 0:
                    orders.append(Order(product, bid_price, -sell_qty))
                    position -= sell_qty
                    sell_limit -= sell_qty
            else:
                break

        for ask_price, ask_vol in sorted(voucher_depth.sell_orders.items()):
            if ask_price < fair:
                buy_qty = min(-ask_vol, buy_limit)
                if buy_qty > 0:
                    orders.append(Order(product, ask_price, buy_qty))
                    position += buy_qty
                    buy_limit -= buy_qty
            else:
                break

        sell_price = max(best_ask - 1, math.ceil(fair + 0.1)) if position < 0 else max(best_ask - 1, math.ceil(fair))
        if sell_limit > 0:
            orders.append(Order(product, sell_price, -sell_limit))

        buy_price = min(best_bid + 1, math.floor(fair - 0.1)) if position > 0 else min(best_bid + 1, math.floor(fair))
        if buy_limit > 0:
            orders.append(Order(product, buy_price, buy_limit))

        return orders

    # ==========================================================
    #  Rolling smile fit
    # ==========================================================
    def update_smile_history_and_fit(
        self,
        state: TradingState,
        traderObject: dict,
        day_num: int,
    ) -> Tuple[Optional[Tuple[float, float, float]], Optional[float], Optional[float]]:
        """
        1. day_num에 맞는 사전 history를 초기화합니다.
        2. 현재 tick의 (m, iv)를 IV_CURVE_FIT_VOUCHERS 기준으로 history에 추가합니다.
        3. 전체 point가 SMILE_MIN_POINTS_FOR_FIT 이상이면 quadratic IV curve를 fit합니다.

        fallback coefficient는 쓰지 않습니다.
        fit이 불가능하면 coeffs=None을 반환하고, 거래하지 않습니다.
        """
        if self.UNDERLYING not in state.order_depths:
            return None, None, None

        underlying_depth = state.order_depths[self.UNDERLYING]

        underlying_mid = self.get_valid_mid_price(
            underlying_depth,
            self.VALID_BID_ASK_VOLUME[self.UNDERLYING],
        )

        if underlying_mid is None:
            return None, None, None

        T = self.get_tte_years(state.timestamp, day_num)

        if T <= 0:
            return None, underlying_mid, T

        self.init_smile_history_for_day(traderObject, day_num)

        hist_key = "smile_hist"
        hist = traderObject[hist_key]

        for product in self.IV_CURVE_FIT_VOUCHERS:
            if product not in hist:
                hist[product] = []

        sqrt_t = math.sqrt(T)

        for product, K in self.IV_CURVE_FIT_VOUCHERS.items():
            if product not in state.order_depths:
                continue

            option_depth = state.order_depths[product]
            option_valid_mid = self.get_valid_mid_price(
                option_depth,
                self.VALID_BID_ASK_VOLUME.get(product, 6),
            )

            if option_valid_mid is None:
                continue

            iv = implied_vol(option_valid_mid, underlying_mid, K, T)

            if iv is None or not math.isfinite(iv):
                continue

            m = math.log(K / underlying_mid) / sqrt_t

            if not math.isfinite(m):
                continue

            is_avged_key = "smile_hist_is_avged"
            if is_avged_key not in traderObject:
                traderObject[is_avged_key] = {}

            is_avged = traderObject[is_avged_key].get(product, True)

            if not is_avged and len(hist[product]) > 0:
                last_m, last_iv = hist[product][-1]
                hist[product][-1] = (
                    0.5 * (float(last_m) + float(m)),
                    0.5 * (float(last_iv) + float(iv)),
                )
                traderObject[is_avged_key][product] = True
            else:
                hist[product].append((m, iv))
                traderObject[is_avged_key][product] = False

            if len(hist[product]) > self.SMILE_WINDOW_PER_VOUCHER:
                hist[product] = hist[product][-self.SMILE_WINDOW_PER_VOUCHER:]

        all_points = []

        for product in self.IV_CURVE_FIT_VOUCHERS:
            all_points.extend(hist.get(product, []))

        if len(all_points) < self.SMILE_MIN_POINTS_FOR_FIT:
            return None, underlying_mid, T

        coeffs = fit_quadratic_from_points(all_points)

        if coeffs is None:
            return None, underlying_mid, T

        return coeffs, underlying_mid, T

    def get_fair_iv(self, m: float, rolling_coeffs: Optional[Tuple[float, float, float]]) -> Optional[float]:
        if rolling_coeffs is None:
            return None

        a, b, c = rolling_coeffs
        fair_iv = a * m * m + b * m + c

        if fair_iv <= 0 or not math.isfinite(fair_iv):
            return None

        return fair_iv

    # ==========================================================
    #  ATM voucher trading
    # ==========================================================
    def get_atm_voucher_orders(
        self,
        product: str,
        state: TradingState,
        traderObject: dict,
        rolling_coeffs: Optional[Tuple[float, float, float]],
        underlying_mid: Optional[float],
        T: Optional[float],
    ) -> List[Order]:
        orders: List[Order] = []

        if rolling_coeffs is None:
            return orders

        if underlying_mid is None or T is None:
            return orders

        if underlying_mid <= 0 or T <= 0:
            return orders

        option_depth = state.order_depths[product]

        option_best_bid, option_best_ask = self.get_best_bid_ask(option_depth)

        if option_best_bid is None or option_best_ask is None:
            return orders

        option_valid_mid = self.get_valid_mid_price(
            option_depth,
            self.VALID_BID_ASK_VOLUME.get(product, 6),
        )

        if option_valid_mid is None:
            return orders

        K = self.ATM_VOUCHERS[product]
        sqrt_t = math.sqrt(T)
        m = math.log(K / underlying_mid) / sqrt_t

        sigma = self.get_fair_iv(m, rolling_coeffs)

        if sigma is None:
            return orders

        theo, _delta, vega = bs_call_with_greeks(underlying_mid, K, T, sigma)

        theo_diff = option_valid_mid - theo

        params = self.MEAN_REVERSION_PARAMS.get(
            product,
            {"ema_window": self.THEO_DIFF_WINDOW, "beta": -1.0},
        )
        ema_window = int(params["ema_window"])
        beta = float(params["beta"])

        mean_key = f"{product}_mean"
        count_key = f"{product}_count"

        mean_diff = self.ema(
            traderObject=traderObject,
            key=mean_key,
            value=theo_diff,
            window=ema_window,
        )

        residual = theo_diff - mean_diff
        expected_diff = theo_diff + beta * residual
        fair_price = theo + expected_diff

        traderObject[count_key] = traderObject.get(count_key, 0) + 1
        count = traderObject[count_key]

        if count < self.WARMUP_TICKS:
            return orders

        position = state.position.get(product, 0)
        limit = self.POSITION_LIMITS[product]

        buy_limit = limit - position
        sell_limit = limit + position

        # Take
        if option_depth.buy_orders:
            for bid_price, bid_vol in sorted(option_depth.buy_orders.items(), reverse=True):
                if bid_price >= fair_price:
                    sell_qty = min(bid_vol, sell_limit)
                    if sell_qty > 0:
                        orders.append(Order(product, bid_price, -sell_qty))
                        position -= sell_qty
                        sell_limit -= sell_qty
        
        if option_depth.sell_orders:
            for ask_price, ask_vol in sorted(option_depth.sell_orders.items()):
                if ask_price <= fair_price:
                    buy_qty = min(-ask_vol, buy_limit)
                    if buy_qty > 0:
                        orders.append(Order(product, ask_price, buy_qty))
                        position += buy_qty
                        buy_limit -= buy_qty

        # Market make
        if self.ENABLE_MAKE:
            sell_price = max(option_best_ask - 1, math.ceil(fair_price))
            sell_qty = min(sell_limit, position)
            if sell_qty > 0:
                orders.append(Order(product, sell_price, -sell_qty))
                sell_limit -= sell_qty
            buy_price = min(option_best_bid + 1, math.floor(fair_price))
            buy_qty = min(buy_limit, -position)
            if buy_qty > 0:
                orders.append(Order(product, buy_price, buy_qty))
                buy_limit -= buy_qty

        return orders
    
    def get_must_buy_0_voucher_orders(self, product: str, state: TradingState) -> List[Order]:
        orders: List[Order] = []

        buy_limit = self.POSITION_LIMITS[product] - state.position.get(product, 0)

        if buy_limit > 0:
            orders.append(Order(product, 0, buy_limit))

        return orders
    
    def _threshold_from_tail_prob(self, tail_prob: float) -> float:
        # threshold = Phi^{-1}(1 - tail_prob)
        p = max(self.P_EPS, min(1.0 - self.P_EPS, tail_prob))
        return NORMAL.inv_cdf(1.0 - p)

    def get_z_thresholds(self, product: str, position: int, limit: int):
        if limit <= 0:
            return 0.0, 0.0

        beta = self.Z_SCORE_PARAMS[product]["threshold_param_beta"]
        alpha = self.Z_SCORE_PARAMS[product]["threshold_param_alpha"]

        if alpha == 1e-6:
            return self.Z_SCORE_PARAMS[product]["threshold"], self.Z_SCORE_PARAMS[product]["threshold"]

        if position >= 0:
            ratio = max(0.0, min(1.0, position / limit))
            p_buy = beta * pow(1.0 - ratio, alpha)
            p_sell = 2.0 * beta - p_buy
        else:
            ratio = max(0.0, min(1.0, abs(position) / limit))
            p_sell = beta * pow(1.0 - ratio, alpha)
            p_buy = 2.0 * beta - p_sell

        buy_threshold = self._threshold_from_tail_prob(p_buy)
        sell_threshold = self._threshold_from_tail_prob(p_sell)
        return buy_threshold, sell_threshold
    
    def add_take_and_make_orders(
        self,
        *,
        product: str,
        order_depth: OrderDepth,
        state: TradingState,
        fair_value: float,
        std: float,
    ) -> List[Order]:
        orders: List[Order] = []

        best_bid, best_ask = self.get_best_bid_ask(order_depth)
        if best_bid is None or best_ask is None:
            return orders

        position = state.position.get(product, 0)
        limit = self.POSITION_LIMITS[product]
        buy_limit = limit - position
        sell_limit = limit + position

        # Take sell
        for bid_price, bid_vol in sorted(order_depth.buy_orders.items(), reverse=True):
            _buy_threshold, sell_threshold = self.get_z_thresholds(product, position, limit)
            if (bid_price - fair_value) / std <= sell_threshold or sell_limit <= 0:
                break

            sell_qty = min(bid_vol, sell_limit)
            if sell_qty > 0:
                orders.append(Order(product, int(bid_price), -int(sell_qty)))
                position -= sell_qty
                sell_limit -= sell_qty

        # Take buy
        for ask_price, ask_vol in sorted(order_depth.sell_orders.items()):
            buy_threshold, _sell_threshold = self.get_z_thresholds(product, position, limit)
            if (ask_price - fair_value) / std >= -buy_threshold or buy_limit <= 0:
                break

            buy_qty = min(-ask_vol, buy_limit)
            if buy_qty > 0:
                orders.append(Order(product, int(ask_price), int(buy_qty)))
                position += buy_qty
                buy_limit -= buy_qty

        # Make
        if self.ENABLE_MAKE:
            buy_threshold, sell_threshold = self.get_z_thresholds(product, position, limit)

            sell_price = max(best_ask - 1, math.ceil(fair_value + sell_threshold * std))
            buy_price = min(best_bid + 1, math.floor(fair_value - buy_threshold * std))

            if sell_limit > 0:
                orders.append(Order(product, int(sell_price), -int(sell_limit)))

            if buy_limit > 0:
                orders.append(Order(product, int(buy_price), int(buy_limit)))

        return orders
    

    def add_take_and_make_orders_fixed_threshold(
        self,
        *,
        product: str,
        order_depth: OrderDepth,
        state: TradingState,
        fair_value: float,
        threshold: float,
    ) -> List[Order]:
        """
        VEV_5400 EMA z-score 전용 주문 함수.

        기존 add_take_and_make_orders()는 std를 받아서 get_z_thresholds()로
        position-dependent threshold를 계산합니다. 그 함수는 Z_SCORE_PARAMS[product]를
        직접 참조하므로 VEV_5400에는 쓰지 않습니다.

        여기서는 sweep-tested EMA z-score 코드처럼 threshold = z * std를 직접 사용합니다.
        """
        orders: List[Order] = []

        best_bid, best_ask = self.get_best_bid_ask(order_depth)
        if best_bid is None or best_ask is None:
            return orders

        position = state.position.get(product, 0)
        limit = self.POSITION_LIMITS[product]
        buy_limit = limit - position
        sell_limit = limit + position

        # Take sell: bid가 fair보다 threshold 이상 비싸면 판다
        for bid_price, bid_vol in sorted(order_depth.buy_orders.items(), reverse=True):
            if bid_price - fair_value <= threshold or sell_limit <= 0:
                break

            sell_qty = min(bid_vol, sell_limit)
            if sell_qty > 0:
                orders.append(Order(product, int(bid_price), -int(sell_qty)))
                position -= sell_qty
                sell_limit -= sell_qty

        # Take buy: ask가 fair보다 threshold 이상 싸면 산다
        for ask_price, ask_vol in sorted(order_depth.sell_orders.items()):
            if ask_price - fair_value >= -threshold or buy_limit <= 0:
                break

            buy_qty = min(-ask_vol, buy_limit)
            if buy_qty > 0:
                orders.append(Order(product, int(ask_price), int(buy_qty)))
                position += buy_qty
                buy_limit -= buy_qty

        # Make: fair +/- threshold 기준으로 양쪽 주문
        if self.ENABLE_MAKE:
            sell_price = max(best_ask - 1, math.ceil(fair_value + threshold))
            buy_price = min(best_bid + 1, math.floor(fair_value - threshold))

            if sell_limit > 0:
                orders.append(Order(product, int(sell_price), -int(sell_limit)))

            if buy_limit > 0:
                orders.append(Order(product, int(buy_price), int(buy_limit)))

        return orders

    def get_ema_zscore_orders(
        self,
        product: str,
        state: TradingState,
        traderObject: dict,
    ) -> List[Order]:
        """
        VEV_5400 EMA z-score trading.

        fair_value = EMA(valid_mid)
        var = EMA((valid_mid - EMA)^2)
        threshold = z_score_threshold * sqrt(var)

        설정:
            VEV_5400: ema_window=1000, z_score_threshold=1.25
        """
        if product not in state.order_depths:
            return []

        params = self.EMA_ZSCORE_PRODUCTS.get(product)
        if params is None:
            return []

        ema_window = int(params["ema_window"])
        z_score_threshold = float(params["z_score_threshold"])
        min_std = float(params.get("min_std", 1e-9))

        depth = state.order_depths[product]
        valid_mid = self.get_valid_mid_price(depth, self.VALID_BID_ASK_VOLUME[product])
        if valid_mid is None:
            return []

        x = float(valid_mid)
        alpha = 2.0 / (ema_window + 1.0)

        ema_key = f"{product}_ema"
        var_key = f"{product}_ema_var"
        count_key = f"{product}_ema_count"

        if ema_key not in traderObject:
            traderObject[ema_key] = x
            traderObject[var_key] = 0.0
            traderObject[count_key] = 1
            return []

        prev_ema = float(traderObject[ema_key])
        new_ema = alpha * x + (1.0 - alpha) * prev_ema

        dev = x - new_ema
        prev_var = float(traderObject.get(var_key, 0.0))
        new_var = alpha * dev * dev + (1.0 - alpha) * prev_var

        traderObject[ema_key] = new_ema
        traderObject[var_key] = new_var
        traderObject[count_key] = int(traderObject.get(count_key, 0)) + 1

        if int(traderObject[count_key]) < ema_window:
            return []

        fair_value = float(traderObject[ema_key])
        std = math.sqrt(max(float(traderObject.get(var_key, 0.0)), 0.0))

        if std <= min_std or not math.isfinite(std):
            return []

        threshold = z_score_threshold * std

        return self.add_take_and_make_orders_fixed_threshold(
            product=product,
            order_depth=depth,
            state=state,
            fair_value=fair_value,
            threshold=threshold,
        )
    
    def get_vev_5500_orders(self, product: str, state: TradingState, traderObject: dict, day_num: int) -> List[Order]:
        tte = max(1, 10 - day_num * 2) - (state.timestamp / 1_000_000.0)
        fair_value = tte

        orders: List[Order] = []

        best_bid, best_ask = self.get_best_bid_ask(state.order_depths[product])

        position = state.position.get(product, 0)
        limit = self.POSITION_LIMITS[product]
        buy_limit = limit - position
        sell_limit = limit + position

        for bid_price, bid_vol in sorted(state.order_depths[product].buy_orders.items(), reverse=True):
            if bid_price > fair_value:
                sell_qty = min(bid_vol, sell_limit)
                if sell_qty > 0:
                    orders.append(Order(product, bid_price, -sell_qty))
                    position -= sell_qty
                    sell_limit -= sell_qty
            else:
                break

        for ask_price, ask_vol in sorted(state.order_depths[product].sell_orders.items()):
            if ask_price < fair_value:
                buy_qty = min(-ask_vol, buy_limit)
                if buy_qty > 0:
                    orders.append(Order(product, ask_price, buy_qty))
                    position += buy_qty
                    buy_limit -= buy_qty
            else:
                break

        sell_price = max(best_ask - 1, math.ceil(fair_value)) if position < 0 else max(best_ask - 1, math.ceil(fair_value))
        if sell_limit > 0:
            orders.append(Order(product, sell_price, -sell_limit))

        buy_price = min(best_bid + 1, math.floor(fair_value)) if position > 0 else min(best_bid + 1, math.floor(fair_value))
        if buy_limit > 0:
            orders.append(Order(product, buy_price, buy_limit))

        return orders

    def get_z_score_orders(self, product: str, state: TradingState, traderObject: dict) -> List[Order]:
        params = self.Z_SCORE_PARAMS.get(product, {
            "valid_mid_history_length": 500,
            "threshold_param_beta": 0.025,
            "threshold_param_alpha": 0.15,
        })
        valid_mid_history_length = int(params["valid_mid_history_length"])

        depth = state.order_depths[product]
        valid_mid = self.get_valid_mid_price(depth, self.VALID_BID_ASK_VOLUME[product])
        if valid_mid is None:
            return []

        key = f"{product}_valid_mid_history"
        history = traderObject.get(key, [])
        is_avged = traderObject.get(f"{product}_is_avged", True)
        if not is_avged:
            history_last = history[-1]
            history[-1] = (history_last + float(valid_mid)) / 2
            is_avged = True
        else:
            history = history[-valid_mid_history_length:]
            history.append(float(valid_mid))
            is_avged = False
        
        traderObject[key] = history
        traderObject[f"{product}_is_avged"] = is_avged

        if len(history) < valid_mid_history_length:
            return []

        fair_value = sum(history) / len(history)
        var = sum((x - fair_value) ** 2 for x in history) / len(history)
        std = math.sqrt(var)

        if std <= self.MIN_STD or not math.isfinite(std):
            return []

        return self.add_take_and_make_orders(
            product=product,
            order_depth=depth,
            state=state,
            fair_value=fair_value,
            std=std,
        )
    

    def run(self, state: TradingState):
        day_num = 4
        original_state = copy.deepcopy(state)

        traderObject = {}

        if state.traderData is not None and state.traderData != "":
            try:
                traderObject = jsonpickle.decode(state.traderData)
            except Exception:
                traderObject = {}

        result: dict[Symbol, List[Order]] = {}
        # traderObject["length"] = 0

        # IV curve fitting은 주문 루프 전에 먼저 수행합니다.
        # DAY_VOUCHER_HISTORY_MAP에 day_num key가 있으면 해당 history로 시작하고,
        # 없으면 빈 history로 시작합니다. coeffs가 None이면 ATM voucher 거래를 하지 않습니다.
        rolling_coeffs, underlying_mid, T = self.update_smile_history_and_fit(
            state=state,
            traderObject=traderObject,
            day_num=day_num,
        )

        for product in state.order_depths:
            orders: List[Order] = []

            if product in self.DEEP_ITM_VOUCHERS:
                orders = self.get_deep_itm_voucher_orders(product, state)

            elif product in self.ATM_VOUCHERS:
                orders = self.get_atm_voucher_orders(
                    product=product,
                    state=state,
                    traderObject=traderObject,
                    rolling_coeffs=rolling_coeffs,
                    underlying_mid=underlying_mid,
                    T=T,
                )

            elif product in self.EMA_ZSCORE_PRODUCTS:
                orders = self.get_ema_zscore_orders(product, state, traderObject)

            elif product in self.Z_SCORE_PRODUCTS:
                orders = self.get_z_score_orders(product, state, traderObject)

            elif product in self.MUST_BUY_0_VOUCHERS:
                orders = self.get_must_buy_0_voucher_orders(product, state)
            
            elif product == "VEV_5500":
                orders = self.get_vev_5500_orders(product, state, traderObject, day_num)

            result[product] = orders

        traderData = jsonpickle.encode(traderObject, unpicklable=False)
        # traderObject["length"] = len(traderData)
        # traderData = jsonpickle.encode(traderObject, unpicklable=False)
        conversions = 0

        logger.flush(original_state, result, conversions, traderData)

        return result, conversions, traderData
