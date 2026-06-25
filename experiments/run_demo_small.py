from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tbgmp.metrics import recovery_rate
from tbgmp.policy import make_topk_layers, make_topk_policy
from tbgmp.risk_score import compute_risk_scores
from tbgmp.sensitive_cases import select_sensitive_cases


def main() -> None:
    cases = pd.DataFrame(
        [
            {"case_id": "math-1", "fp16_found": True, "aggressive_found": False},
            {"case_id": "code-1", "fp16_found": True, "aggressive_found": True},
            {"case_id": "science-1", "fp16_found": False, "aggressive_found": False},
        ]
    )
    sensitive = select_sensitive_cases(cases)

    layer_stats = pd.DataFrame(
        {
            "layer": [0, 1, 2, 3],
            "mse_p95": [0.12, 0.80, 0.30, 0.55],
            "ip_p95": [0.25, 1.60, 0.40, 0.90],
            "effective_dim": [48.0, 11.0, 37.0, 21.0],
        }
    )
    ranked = compute_risk_scores(layer_stats)
    protected = make_topk_layers(ranked["layer"].astype(int).tolist(), k=2)
    policy = make_topk_policy(2, 2, 6, protected)

    paper = pd.read_csv(ROOT / "results" / "paper_tables" / "table_main_evidence.csv")
    restored = paper["recovered"].str.split("/").str[0].astype(int).sum()
    total = paper["sensitive"].astype(int).sum()

    print("Toy sensitive case IDs:", sensitive["case_id"].tolist())
    print("Toy ranked layers:", ranked["layer"].astype(int).tolist())
    print("Toy Top-2 policy:", policy)
    print(f"Paper main-set recovery: {restored}/{total} ({recovery_rate(restored, total):.1%})")


if __name__ == "__main__":
    main()
