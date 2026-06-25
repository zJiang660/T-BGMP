from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tbgmp.controls import bottomk_layers, sample_random_layers
from tbgmp.sensitive_cases import select_sensitive_cases
from tbgmp.utils import parse_bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create same-budget Random-k and Bottom-k model-free control rows."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "default_experiment.yaml",
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "audit" / "demo_random_bottom_controls.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not args.risk_ranking.exists():
        raise SystemExit("Run stage_c_profile_key_risk.py first or pass --risk-ranking")
    ranked = pd.read_csv(args.risk_ranking).sort_values("rank")["layer"].astype(int).tolist()
    selections = {
        f"random{args.k}_seed{args.seed}": sample_random_layers(ranked, args.k, args.seed),
        f"bottom{args.k}": bottomk_layers(ranked, args.k),
    }
    cases = pd.read_csv(args.cases)
    for column in ("fp16_found", "aggressive_found", "random_found", "bottom_found"):
        cases[column] = cases[column].map(parse_bool)
    sensitive = select_sensitive_cases(cases)

    rows: list[dict] = []
    for case in sensitive.to_dict("records"):
        for policy, layers in selections.items():
            is_random = policy.startswith("random")
            rows.append(
                {
                    "case_id": case["case_id"],
                    "policy": policy,
                    "policy_type": "random_k" if is_random else "bottom_k",
                    "protected_layers": str(layers),
                    "matched_topk_k": args.k,
                    "seed": args.seed if is_random else "",
                    "found": case["random_found"] if is_random else case["bottom_found"],
                    "status": "success",
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Stage E matched controls: {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
