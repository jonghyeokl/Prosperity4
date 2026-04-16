import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


TARGET_SYMBOL = "ASH_COATED_OSMIUM"

HEADERS = [
    "day",
    "timestamp",
    "product",
    "bid_price_1",
    "bid_volume_1",
    "bid_price_2",
    "bid_volume_2",
    "bid_price_3",
    "bid_volume_3",
    "ask_price_1",
    "ask_volume_1",
    "ask_price_2",
    "ask_volume_2",
    "ask_price_3",
    "ask_volume_3",
    "mid_price",
    "profit_and_loss",
]


def read_price_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, delimiter=";")
    if "product" not in df.columns:
        df.columns = HEADERS[: len(df.columns)]
    return df


def infer_base_step(ts: pd.Series) -> int:
    diffs = ts.diff().dropna()
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 100
    mode = diffs.mode()
    if len(mode) == 0:
        return int(diffs.iloc[0])
    return int(mode.iloc[0])


def load_symbol_data(symbol: str) -> pd.DataFrame:
    project_root = Path(__file__).resolve().parents[2]
    round_dir = project_root / "data_capsule" / "round1"

    csv_paths = sorted(round_dir.glob("prices_round_1_day_*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"가격 파일을 찾지 못했습니다: {round_dir}")

    frames = []
    for path in csv_paths:
        df = read_price_file(path)
        df = df[df["product"] == symbol].copy()
        frames.append(df)

    raw = pd.concat(frames, ignore_index=True)

    for col in ["day", "timestamp", "bid_price_1", "ask_price_1", "mid_price"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    raw = raw.sort_values(["day", "timestamp"]).reset_index(drop=True)

    valid_quote = (
        raw["bid_price_1"].notna()
        & raw["ask_price_1"].notna()
        & (raw["bid_price_1"] > 0)
        & (raw["ask_price_1"] > 0)
        & (raw["ask_price_1"] >= raw["bid_price_1"])
    )

    raw["quote_mid"] = np.where(
        valid_quote,
        (raw["bid_price_1"] + raw["ask_price_1"]) / 2.0,
        np.nan,
    )

    raw["analysis_mid"] = raw["quote_mid"]

    base_steps = {}
    for day, g in raw.groupby("day", sort=False):
        base_steps[day] = infer_base_step(g["timestamp"])
    raw["base_step"] = raw["day"].map(base_steps)

    df = raw[raw["analysis_mid"].notna()].copy()
    df = df.sort_values(["day", "timestamp"]).reset_index(drop=True)
    return df


def make_lag1_pairs(df: pd.DataFrame) -> pd.DataFrame:
    pairs = []

    for _, g in df.groupby("day", sort=False):
        g = g.sort_values("timestamp").copy()

        p = g["analysis_mid"]
        ts = g["timestamp"]
        step = int(g["base_step"].iloc[0])

        r_t = p.diff()
        gap1 = ts.diff()
        r_t = r_t.where(gap1 == step)

        sub = pd.DataFrame({
            "timestamp": ts,
            "r_t": r_t,
        }).dropna()

        sub["r_t1"] = sub["r_t"].shift(-1)
        future_gap = sub["timestamp"].shift(-1) - sub["timestamp"]

        sub = sub[(sub["r_t1"].notna()) & (future_gap == step)].copy()
        pairs.append(sub[["r_t", "r_t1"]])

    out = pd.concat(pairs, ignore_index=True)
    return out


def fit_line(x: np.ndarray, y: np.ndarray):
    b, a = np.polyfit(x, y, 1)
    corr = np.corrcoef(x, y)[0, 1]

    n = len(x)
    if n >= 3 and abs(corr) < 1:
        t_stat = corr * math.sqrt((n - 2) / (1 - corr**2))
    else:
        t_stat = np.nan

    return a, b, corr, t_stat


def gaussian_kernel_1d(size: int = 9, sigma: float = 1.5) -> np.ndarray:
    if size % 2 == 0:
        raise ValueError("size must be odd")
    half = size // 2
    x = np.arange(-half, half + 1)
    k = np.exp(-(x**2) / (2 * sigma**2))
    k = k / k.sum()
    return k


def smooth_2d_histogram(H: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    # axis=0 smoothing
    temp = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=0, arr=H)
    # axis=1 smoothing
    out = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=1, arr=temp)
    return out


def plot_continuous_density(x: np.ndarray, y: np.ndarray, title: str) -> None:
    # 전체 데이터 사용
    bins = 180

    x_low, x_high = np.percentile(x, [0.5, 99.5])
    y_low, y_high = np.percentile(y, [0.5, 99.5])

    # 바깥 극단값도 조금 보이게 여유 추가
    x_pad = 0.08 * (x_high - x_low)
    y_pad = 0.08 * (y_high - y_low)

    x_range = (x_low - x_pad, x_high + x_pad)
    y_range = (y_low - y_pad, y_high + y_pad)

    H, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[x_range, y_range])

    kernel = gaussian_kernel_1d(size=11, sigma=2.0)
    H_smooth = smooth_2d_histogram(H, kernel)

    positive = H_smooth[H_smooth > 0]
    vmin = max(np.percentile(positive, 5), 1e-3)
    vmax = np.percentile(positive, 99.5)

    plt.figure(figsize=(8, 8))
    im = plt.imshow(
        H_smooth.T,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect="auto",
        interpolation="bilinear",
        norm=LogNorm(vmin=vmin, vmax=vmax),
    )
    plt.colorbar(im, label="smoothed density")
    plt.axhline(0, linewidth=1)
    plt.axvline(0, linewidth=1)
    plt.xlabel("r_t")
    plt.ylabel("r_(t+1) + 0.5 r_t")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def main():
    df = load_symbol_data(TARGET_SYMBOL)
    pairs = make_lag1_pairs(df)

    x = pairs["r_t"].to_numpy()
    y = pairs["r_t1"].to_numpy()

    a, b, corr, t_stat = fit_line(x, y)

    pairs["adjusted"] = pairs["r_t1"] + 0.5 * pairs["r_t"]

    print(f"n = {len(pairs)}")
    print(f"corr(r_t, r_t+1) = {corr:.6f}")
    print(f"t_stat = {t_stat:.6f}")
    print(f"fitted line: r_(t+1) = {a:.6f} + ({b:.6f}) * r_t")
    print()
    print("adjusted = r_(t+1) + 0.5 * r_t")
    print(f"mean(adjusted) = {pairs['adjusted'].mean():.6f}")
    print(f"std(adjusted) = {pairs['adjusted'].std(ddof=0):.6f}")

    # 1) 연속적 밀도 그래프
    plot_continuous_density(
        pairs["r_t"].to_numpy(),
        pairs["adjusted"].to_numpy(),
        f"{TARGET_SYMBOL}: smoothed density of (r_t, r_(t+1) + 0.5 r_t)"
    )

    # 2) 분포 그래프
    plt.figure(figsize=(8, 6))
    plt.hist(pairs["adjusted"], bins=60)
    plt.axvline(pairs["adjusted"].mean(), linewidth=1)
    plt.xlabel("r_(t+1) + 0.5 r_t")
    plt.ylabel("count")
    plt.title(f"{TARGET_SYMBOL}: distribution of r_(t+1) + 0.5 r_t")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()