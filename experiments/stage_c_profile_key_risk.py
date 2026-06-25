from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tbgmp.risk_score import compute_risk_scores


def main() -> None:
    stats = pd.DataFrame(
        {
            "layer": [0, 1, 2, 3],
            "mse_p95": [0.12, 0.80, 0.30, 0.55],
            "ip_p95": [0.25, 1.60, 0.40, 0.90],
            "effective_dim": [48.0, 11.0, 37.0, 21.0],
        }
    )
    ranked = compute_risk_scores(stats)
    output_dir = ROOT / "results" / "audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "demo_key_risk_ranking.csv"
    ranked.to_csv(output, index=False)
    print(f"Stage C key-risk profiling: ranking {ranked['layer'].astype(int).tolist()}")


if __name__ == "__main__":
    main()
