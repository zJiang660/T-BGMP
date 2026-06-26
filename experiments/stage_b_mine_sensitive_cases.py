from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml


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
        "--config",
        type=Path,
        default=ROOT / "configs" / "default_experiment.yaml",
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
    parser.add_argument("--model-key")
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--backend")
    parser.add_argument("--turboquant-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.output_dir:
        args.output = args.output_dir / "sensitive_cases.csv"
    if args.dry_run:
        print(f"Stage B: mine sensitive cases -> {args.output}")
        print("No model execution performed in dry-run mode.")
        return
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
