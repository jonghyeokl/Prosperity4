from pathlib import Path
import csv

INPUT_CSV = Path("./jonghyeok/manual/round2_result.csv")
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
    """
    counts[z] = speed에 z 투자한 팀 수
    반환값:
        f_map[z] = 해당 z의 f(z)
    """
    total_teams = sum(counts.values())
    if total_teams <= 0:
        raise ValueError("팀 수가 0입니다.")

    counts_full = {z: counts.get(z, 0) for z in range(101)}

    # greater_count[z] = z보다 큰 speed 투자한 팀 수
    greater_count = {}
    running = 0
    for z in range(100, -1, -1):
        greater_count[z] = running
        running += counts_full[z]

    f_map = {}

    for z in range(101):
        rank = 1 + greater_count[z]

        if total_teams == 1:
            speed_multiplier = 0.9
        else:
            speed_multiplier = 0.9 - 0.8 * (rank - 1) / (total_teams - 1)

        f_map[z] = speed_multiplier - 0.1

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