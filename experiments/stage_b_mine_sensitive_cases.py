from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tbgmp.sensitive_cases import select_sensitive_cases
from tbgmp.utils import parse_bool


def main() -> None:
    cases = pd.read_csv(ROOT / "data" / "demo" / "demo_cases.csv")
    for column in ["fp16_found", "aggressive_found"]:
        cases[column] = cases[column].map(parse_bool)
    sensitive = select_sensitive_cases(cases)
    output_dir = ROOT / "results" / "audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "demo_sensitive_cases.csv"
    sensitive.to_csv(output, index=False)
    print(f"Stage B sensitive-case mining: {len(sensitive)} row(s) -> {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
