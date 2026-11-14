"""Run a small synthetic smoke test.

This script is intentionally NOT the DrugComb v1.5 manuscript workflow.
It creates synthetic inputs, preprocesses them and runs one short training fold.
It does not print or hard-code performance metrics.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    print("\n$", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="PATH-SYN synthetic smoke test (not paper reproduction)")
    parser.add_argument("--out", default="runs/synthetic-smoke-test")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.out)
    data = Path("data/synthetic-smoke-test")
    cache = data / "cache"

    run([sys.executable, "scripts/make_synthetic_data.py", "--out", str(data), "--seed", str(args.seed)])
    run([
        sys.executable, "-m", "pathsyn.preprocess",
        "--depmap-dir", str(data / "depmap"),
        "--msigdb-dir", str(data / "msigdb"),
        "--synergy", str(data / "synergy.csv"),
        "--drug-table", str(data / "drugs.csv"),
        "--out", str(cache),
    ])
    run([
        sys.executable, "-m", "pathsyn.train",
        "--cache", str(cache),
        "--table", str(data / "synergy.csv"),
        "--split", "random",
        "--epochs", str(args.epochs),
        "--folds", "2",
        "--max-folds", "1",
        "--bs", "32",
        "--seed", str(args.seed),
        "--out", str(out),
    ])
    print(f"\nSynthetic smoke test completed. Outputs: {out}")
    print("These outputs are synthetic diagnostics, not manuscript results.")


if __name__ == "__main__":
    main()
