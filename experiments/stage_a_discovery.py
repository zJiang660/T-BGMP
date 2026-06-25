from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tbgmp.retrieval_eval import found_answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate model-free discovery responses and write a compact Stage A "
            "CSV. Real generation is handled by run_full_pipeline.py with an "
            "external backend."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "demo" / "demo_cases.csv",
        help="CSV containing answer, response_fp16, and response_aggressive.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "audit" / "demo_stage_a_discovery.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = pd.read_csv(args.input)
    required = {"case_id", "answer", "response_fp16", "response_aggressive"}
    missing = sorted(required - set(cases.columns))
    if missing:
        raise SystemExit(f"Stage A input is missing columns: {missing}")

    rows: list[dict] = []
    for case in cases.to_dict("records"):
        common = {
            key: case.get(key, "")
            for key in ("case_id", "domain", "context_length", "depth", "seed", "answer")
        }
        for policy, response_column in (
            ("fp16", "response_fp16"),
            ("aggressive_uniform", "response_aggressive"),
        ):
            rows.append(
                {
                    **common,
                    "policy": policy,
                    "found": found_answer(case[response_column], case["answer"]),
                    "status": "success",
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Stage A discovery: {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
