from pathlib import Path
import csv

INPUT_CSV = Path("./jonghyeok/manual/round2_result_withoutus.csv")
OUTPUT_TXT = Path("./jonghyeok/manual/round2_result_f.txt")


def load_speed_counts(csv_path: Path) -> dict[int, int]:
    counts = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            z = int(row[0])
            cnt = int(row[1])
            counts[z] = cnt
    return counts


def build_f_map(counts: dict[int, int]) -> dict[int, float]:
    counts_full = {z: counts.get(z, 0) for z in range(101)}
    M = sum(counts_full.values())
    if M == 0:
        raise ValueError("팀 수가 0입니다.")

    positive_z = [z for z, cnt in counts_full.items() if cnt > 0]
    min_existing_z = min(positive_z)

    # greater_count[z] = 기존 팀들 중 z보다 크게 투자한 팀 수
    greater_count = {}
    running = 0
    for z in range(100, -1, -1):
        greater_count[z] = running
        running += counts_full[z]

    # 기존 분포에서의 최하위 rank
    existing_lowest_rank = 1 + greater_count[min_existing_z]

    f_map = {}

    for z in range(101):
        # 내가 z에 새로 투자했다고 가정했을 때 내 rank
        my_rank = 1 + greater_count[z]

        # 내가 추가되면 lowest rank가 어떻게 바뀌는지 반영
        if z < min_existing_z:
            lowest_rank = M + 1
        elif z == min_existing_z:
            lowest_rank = existing_lowest_rank
        else:
            lowest_rank = existing_lowest_rank + 1

        # all-tie 예외
        if lowest_rank == 1:
            speed_multiplier = 0.9
        else:
            speed_multiplier = 0.9 - 0.8 * (my_rank - 1) / (lowest_rank - 1)

        f_val = speed_multiplier - 0.1

        if abs(f_val) < 1e-12:
            f_val = 0.0
        if abs(f_val - 0.8) < 1e-12:
            f_val = 0.8

        f_map[z] = f_val

    return f_map


def format_float(x: float) -> str:
    s = f"{x:.10f}".rstrip("0").rstrip(".")
    if "." not in s:
        s += ".0"
    return s


def save_f_map_as_python_dict(f_map: dict[int, float], output_path: Path) -> None:
    lines = ["{"]
    for z in range(101):
        lines.append(f"    {z}: {format_float(f_map[z])},")
    lines.append("}")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    counts = load_speed_counts(INPUT_CSV)
    f_map = build_f_map(counts)
    save_f_map_as_python_dict(f_map, OUTPUT_TXT)
    print(f"saved to {OUTPUT_TXT}")


if __name__ == "__main__":
    main()