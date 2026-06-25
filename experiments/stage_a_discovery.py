from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tbgmp.retrieval_eval import found_answer


def main() -> None:
    cases = pd.read_csv(ROOT / "data" / "demo" / "demo_cases.csv")
    checks = {
        "fp16_found": [
            found_answer(response, answer)
            for response, answer in zip(cases["response_fp16"], cases["answer"])
        ],
        "aggressive_found": [
            found_answer(response, answer)
            for response, answer in zip(cases["response_aggressive"], cases["answer"])
        ],
    }
    for column, computed in checks.items():
        declared = cases[column].astype(bool).tolist()
        if declared != computed:
            raise AssertionError(f"Discovery field mismatch for {column}")
    print(f"Stage A discovery: validated {len(cases)} demo rows")


if __name__ == "__main__":
    main()
