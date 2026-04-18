import csv
import math
from pathlib import Path

OUTPUT_DIR = Path("./jonghyeok/manual/manual_round2_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_BASE = 200_000 / math.log(101)
SCORE_BASE = 14_000 / math.log(101)  # = research coefficient * scale coefficient (7/100)

FZ_MIN = 0.0
FZ_MAX = 0.8
FZ_STEP = 0.001
FP_MIN = 0.0
FP_MAX = 0.05
FP_STEP = 0.001

Z_MIN = 1
Z_MAX = 99


def score(x: int, y: int, z: int, fz: float) -> float:
    """
    S(x, y, z) = (14000 / ln(101)) * ln(1+x) * y * (0.1 + f(z)) - 500(x+y+z)
    """
    return SCORE_BASE * math.log(1 + x) * y * (0.1 + fz) - 500 * (x + y + z)


def precompute_best_xy_by_z():
    """
    For each fixed z, precompute the best (x, y) when the remaining budget B=100-z is used optimally.

    Because the objective is linear in y for fixed x,
    the optimum y is either 0 or (100-z-x).

    If we use the full remaining budget, y = B - x and the objective becomes:
        SCORE_BASE * (0.1 + fz) * (B - x) * ln(1+x) - 50000

    So for each z we only need to maximize (B - x) * ln(1+x) over integer x.
    """
    best_by_z = {}

    for z in range(101):
        B = 100 - z
        best_x = 0
        best_shape_value = -1.0

        for x in range(B + 1):
            y = B - x
            shape_value = y * math.log(1 + x)
            if shape_value > best_shape_value + 1e-15:
                best_shape_value = shape_value
                best_x = x

        best_y = B - best_x
        best_by_z[z] = {
            "x_full": best_x,
            "y_full": best_y,
            "shape_value": best_shape_value,
        }

    return best_by_z


def build_optimal_table(best_by_z):
    """
    Precompute, for every z and every discrete f(z) value, the optimal
    (x, y, S*) under the user's model.

    We compare two possibilities:
    1) use none of the remaining budget for x,y  -> (x,y)=(0,0), score=-500z
    2) use the full remaining budget optimally   -> precomputed best x,y
    """
    table = {}
    n_fz = int(round((FZ_MAX - FZ_MIN) / FZ_STEP)) + 1

    for z in range(101):
        row = []
        idle_score = -500 * z
        x_full = best_by_z[z]["x_full"]
        y_full = best_by_z[z]["y_full"]
        shape_value = best_by_z[z]["shape_value"]

        for fz_idx in range(n_fz):
            fz = round(FZ_MIN + FZ_STEP * fz_idx, 3)
            speed_multiplier = 0.1 + fz
            full_score = SCORE_BASE * speed_multiplier * shape_value - 50_000

            if full_score >= idle_score:
                row.append({
                    "x": x_full,
                    "y": y_full,
                    "score": full_score,
                })
            else:
                row.append({
                    "x": 0,
                    "y": 0,
                    "score": idle_score,
                })

        table[z] = row

    return table


def scan_candidates(table):
    """
    Find all (z, f(z), f'(z)) such that
        S*(z) >= max(S*(z-1), S*(z+1))
    under the local assumption
        f'(z) = f(z) - f(z-1) = f(z+1) - f(z)
    with 0 <= f'(z) <= 0.05.

    We also enforce local feasibility:
        0 <= f(z)-f'(z)
        f(z)+f'(z) <= 0.8
    so that the neighboring f-values stay in [0, 0.8] and monotonicity is preserved locally.
    """
    n_fz = int(round((FZ_MAX - FZ_MIN) / FZ_STEP)) + 1
    n_fp = int(round((FP_MAX - FP_MIN) / FP_STEP)) + 1

    candidates = []

    for z in range(Z_MIN, Z_MAX + 1):
        cur_row = table[z]
        prev_row = table[z - 1]
        next_row = table[z + 1]

        for fz_idx in range(n_fz):
            fz = round(FZ_MIN + FZ_STEP * fz_idx, 3)

            # Local feasibility for neighbors: f(z-1)=f(z)-f'(z), f(z+1)=f(z)+f'(z)
            max_fp_idx = min(n_fp - 1, fz_idx, (n_fz - 1) - fz_idx)

            cur_opt = cur_row[fz_idx]
            cur_score = cur_opt["score"]

            for fp_idx in range(max_fp_idx + 1):
                fp = round(FP_MIN + FP_STEP * fp_idx, 3)

                prev_fz_idx = fz_idx - fp_idx
                next_fz_idx = fz_idx + fp_idx

                prev_opt = prev_row[prev_fz_idx]
                next_opt = next_row[next_fz_idx]

                prev_score = prev_opt["score"]
                next_score = next_opt["score"]

                if cur_score >= prev_score and cur_score >= next_score:
                    candidates.append({
                        "z": z,
                        "f_z": fz,
                        "f_prime": fp,
                        "f_z_minus_1": round(fz - fp, 3),
                        "f_z_plus_1": round(fz + fp, 3),
                        "speed_z_minus_1": round(0.1 + fz - fp, 3),
                        "speed_z": round(0.1 + fz, 3),
                        "speed_z_plus_1": round(0.1 + fz + fp, 3),
                        "x_z_minus_1": prev_opt["x"],
                        "y_z_minus_1": prev_opt["y"],
                        "S_z_minus_1": prev_score,
                        "x_z": cur_opt["x"],
                        "y_z": cur_opt["y"],
                        "S_z": cur_score,
                        "x_z_plus_1": next_opt["x"],
                        "y_z_plus_1": next_opt["y"],
                        "S_z_plus_1": next_score,
                        "margin_vs_prev": cur_score - prev_score,
                        "margin_vs_next": cur_score - next_score,
                        "margin_vs_best_neighbor": cur_score - max(prev_score, next_score),
                    })

    return candidates


def write_csv(path: Path, rows):
    if not rows:
        raise ValueError("No rows to write.")

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, best_by_z, candidates):
    best_candidate = max(candidates, key=lambda r: (r["S_z"], r["margin_vs_best_neighbor"], -r["z"]))

    lines = []
    lines.append("Round 2 manual challenge candidate scan summary\n")
    lines.append("=" * 60 + "\n")
    lines.append(f"Total candidate rows: {len(candidates):,}\n")
    lines.append(f"Best candidate by S_z: z={best_candidate['z']}, f(z)={best_candidate['f_z']:.3f}, f'(z)={best_candidate['f_prime']:.3f}\n")
    lines.append(f"Optimal (x, y) at z: ({best_candidate['x_z']}, {best_candidate['y_z']})\n")
    lines.append(f"S(z-1)={best_candidate['S_z_minus_1']:.6f}, S(z)={best_candidate['S_z']:.6f}, S(z+1)={best_candidate['S_z_plus_1']:.6f}\n")
    lines.append(f"Local-max margin vs best neighbor: {best_candidate['margin_vs_best_neighbor']:.6f}\n")
    lines.append("\nBest full-budget (x, y) by z (ignoring the speed level threshold):\n")

    for z in range(0, 101):
        x_full = best_by_z[z]["x_full"]
        y_full = best_by_z[z]["y_full"]
        lines.append(f"z={z:3d}: x={x_full:2d}, y={y_full:2d}\n")

    path.write_text("".join(lines), encoding="utf-8")


def main():
    best_by_z = precompute_best_xy_by_z()
    table = build_optimal_table(best_by_z)
    candidates = scan_candidates(table)

    if not candidates:
        raise RuntimeError("No candidates found under the current grid and assumptions.")

    all_csv = OUTPUT_DIR / "manual_round2_localmax_candidates.csv"
    summary_txt = OUTPUT_DIR / "manual_round2_summary.txt"

    write_csv(all_csv, candidates)

    print(f"Saved: {all_csv}")
    print(f"Saved: {summary_txt}")
    print(f"Candidate count: {len(candidates):,}")


if __name__ == "__main__":
    main()
