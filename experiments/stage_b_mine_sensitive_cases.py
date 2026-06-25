from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tbgmp.sensitive_cases import select_sensitive_cases
from tbgmp.utils import parse_bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select FP16-found/aggressive-not-found cases. Execution status is "
            "never used as a proxy for retrieval success."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "demo" / "demo_cases.csv",
        help="Wide CSV containing fp16_found and aggressive_found.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "audit" / "demo_sensitive_cases.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = pd.read_csv(args.input)
    for column in ("fp16_found", "aggressive_found"):
        if column not in cases:
            raise SystemExit(f"Stage B input is missing column: {column}")
        cases[column] = cases[column].map(parse_bool)
    sensitive = select_sensitive_cases(cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sensitive.to_csv(args.output, index=False)
    print(f"Stage B sensitive-case mining: {len(sensitive)} row(s) -> {args.output}")


if __name__ == "__main__":
    main()
