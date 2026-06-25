from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tbgmp.controls import bottomk_layers, sample_random_layers
from tbgmp.sensitive_cases import select_sensitive_cases
from tbgmp.utils import parse_bool


def main() -> None:
    ranking_path = ROOT / "results" / "audit" / "demo_key_risk_ranking.csv"
    if not ranking_path.exists():
        raise FileNotFoundError("Run stage_c_profile_key_risk.py first")
    ranked = pd.read_csv(ranking_path)["layer"].astype(int).tolist()
    random_layers = sample_random_layers(ranked, k=2, seed=0)
    bottom_layers = bottomk_layers(ranked, k=2)

    cases = pd.read_csv(ROOT / "data" / "demo" / "demo_cases.csv")
    for column in [
        "fp16_found",
        "aggressive_found",
        "random_found",
        "bottom_found",
    ]:
        cases[column] = cases[column].map(parse_bool)
    sensitive = select_sensitive_cases(cases)
    random_found = int(sensitive["random_found"].sum())
    bottom_found = int(sensitive["bottom_found"].sum())
    print(f"Stage E Random-k layers (seed 0): {random_layers}; found rows={random_found}")
    print(f"Stage E Bottom-k layers: {bottom_layers}; found rows={bottom_found}")


if __name__ == "__main__":
    main()
