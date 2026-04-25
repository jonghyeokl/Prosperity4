"""
Runs all phases in order. Set DATA_DIR env var if CSVs are not in the working dir.
"""
import subprocess, sys, os

scripts = [
    "p1_raw_prices.py",
    "p23_iv_smile_fit.py",
    "p4_coef_timeseries.py",
]
for s in scripts:
    print(f"\n{'='*70}\n  Running {s}\n{'='*70}")
    r = subprocess.run([sys.executable, s], env=os.environ.copy())
    if r.returncode != 0:
        print(f"[FAIL] {s} exited with {r.returncode}"); sys.exit(1)
print("\nAll phases complete.")