from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# 바꿔서 사용
TARGET_Z = 1

# 필요하면 경로 수정
CSV_PATH = Path("./jonghyeok/manual/manual_round2_outputs/manual_round2_localmax_candidates.csv")

df = pd.read_csv(CSV_PATH)
sub = df[df["z"] == TARGET_Z].copy()

if sub.empty:
    print(f"z={TARGET_Z} 에 해당하는 후보가 없습니다.")
else:
    # float 오차 방지
    sub["f_z"] = sub["f_z"].round(3)
    sub["f_prime"] = sub["f_prime"].round(3)

    # 각 f(z)별 가능한 f'(z) 범위
    bounds = (
        sub.groupby("f_z", as_index=False)
        .agg(
            f_prime_min=("f_prime", "min"),
            f_prime_max=("f_prime", "max"),
            count=("f_prime", "size"),
            S_min=("S_z", "min"),
            S_max=("S_z", "max"),
        )
        .sort_values("f_z")
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    # 되는 점들만 실제 좌표에 작게 표시
    sc = ax.scatter(
        sub["f_z"],
        sub["f_prime"],
        c=sub["S_z"],
        s=10,
        cmap="viridis",
        marker="s",
        linewidths=0,
    )
    plt.colorbar(sc, ax=ax, label="S_z")

    # 테두리 표시: 위/아래 경계 + 좌우 닫기
    ax.plot(bounds["f_z"], bounds["f_prime_min"], lw=1.8, color="black")
    ax.plot(bounds["f_z"], bounds["f_prime_max"], lw=1.8, color="black")

    if len(bounds) >= 1:
        ax.plot(
            [bounds.iloc[0]["f_z"], bounds.iloc[0]["f_z"]],
            [bounds.iloc[0]["f_prime_min"], bounds.iloc[0]["f_prime_max"]],
            lw=1.8,
            color="black",
        )
        ax.plot(
            [bounds.iloc[-1]["f_z"], bounds.iloc[-1]["f_z"]],
            [bounds.iloc[-1]["f_prime_min"], bounds.iloc[-1]["f_prime_max"]],
            lw=1.8,
            color="black",
        )

    # hover용 annotation
    annot = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(12, 12),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.9),
        fontsize=9,
    )
    annot.set_visible(False)

    # hover 시 현재 f(z)에 대한 가능한 f'(z) 범위 강조
    hover_line, = ax.plot([], [], color="red", lw=2, alpha=0.9)
    hover_dot, = ax.plot([], [], "ro", ms=4)

    # 빠른 조회용 dict
    bounds_map = {
        round(row.f_z, 3): row
        for row in bounds.itertuples(index=False)
    }

    def on_move(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            annot.set_visible(False)
            hover_line.set_data([], [])
            hover_dot.set_data([], [])
            fig.canvas.draw_idle()
            return

        x = round(event.xdata, 3)

        if x not in bounds_map:
            annot.set_visible(False)
            hover_line.set_data([], [])
            hover_dot.set_data([], [])
            fig.canvas.draw_idle()
            return

        row = bounds_map[x]
        y_min = row.f_prime_min
        y_max = row.f_prime_max

        # 세로 범위선 + 현재 마우스 y를 범위 안으로 clip한 점
        y_hover = min(max(event.ydata, y_min), y_max)
        hover_line.set_data([x, x], [y_min, y_max])
        hover_dot.set_data([x], [y_hover])

        annot.xy = (x, y_hover)
        annot.set_text(
            f"f(z) = {x:.3f}\n"
            f"f'(z) range = [{y_min:.3f}, {y_max:.3f}]\n"
            f"count = {int(row.count)}"
        )
        annot.set_visible(True)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)

    ax.set_xlabel("f(z)")
    ax.set_ylabel("f'(z)")
    ax.set_title(f"Valid (f(z), f'(z)) pairs for z = {TARGET_Z}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()