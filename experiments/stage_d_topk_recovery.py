from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tbgmp.policy import make_topk_layers
from tbgmp.sensitive_cases import select_sensitive_cases
from tbgmp.utils import parse_bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a model-free Top-k recovery table from declared demo outcomes. "
            "Real Top1--Top12 generation requires run_full_pipeline.py and a backend."
        )
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=ROOT / "data" / "demo" / "demo_cases.csv",
    )
    parser.add_argument(
        "--risk-ranking",
        type=Path,
        default=ROOT / "results" / "audit" / "demo_key_risk_ranking.csv",
    )
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "audit" / "demo_topk_recovery.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.risk_ranking.exists():
        raise SystemExit("Run stage_c_profile_key_risk.py first or pass --risk-ranking")
    ranked = pd.read_csv(args.risk_ranking).sort_values("rank")
    layers = make_topk_layers(ranked["layer"].astype(int).tolist(), args.k)

    cases = pd.read_csv(args.cases)
    for column in ("fp16_found", "aggressive_found", "topk_found"):
        cases[column] = cases[column].map(parse_bool)
    sensitive = select_sensitive_cases(cases)
    rows = sensitive.copy()
    rows["policy"] = f"tbgmp_top{args.k}"
    rows["topk_k"] = args.k
    rows["protected_layers"] = str(layers)
    rows["found"] = rows["topk_found"]
    rows["status"] = "success"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.output, index=False)
    print(f"Stage D Top-k recovery: {int(rows['found'].sum())}/{len(rows)} -> {args.output}")


if __name__ == "__main__":
    main()
