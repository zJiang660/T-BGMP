from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tbgmp.policy import make_topk_layers, make_topk_policy
from tbgmp.sensitive_cases import select_sensitive_cases
from tbgmp.utils import parse_bool


def main() -> None:
    ranking_path = ROOT / "results" / "audit" / "demo_key_risk_ranking.csv"
    if not ranking_path.exists():
        raise FileNotFoundError("Run stage_c_profile_key_risk.py first")
    ranked = pd.read_csv(ranking_path)
    layers = make_topk_layers(ranked["layer"].astype(int).tolist(), k=2)
    policy = make_topk_policy(2, 2, 6, layers)

    cases = pd.read_csv(ROOT / "data" / "demo" / "demo_cases.csv")
    for column in ["fp16_found", "aggressive_found", "topk_found"]:
        cases[column] = cases[column].map(parse_bool)
    sensitive = select_sensitive_cases(cases)
    recovered = int(sensitive["topk_found"].sum())
    print(f"Stage D Top-k recovery policy: {policy}")
    print(f"Stage D sensitive-case recovery: {recovered}/{len(sensitive)}")


if __name__ == "__main__":
    main()
