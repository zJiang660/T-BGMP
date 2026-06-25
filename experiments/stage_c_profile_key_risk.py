from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tbgmp.risk_score import compute_risk_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rank layers from precomputed key-distortion statistics. This script "
            "does not extract KV tensors or run a language model."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Optional CSV with layer, mse_p95, ip_p95, and effective_dim.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "audit" / "demo_key_risk_ranking.csv",
    )
    return parser.parse_args()


def demo_stats() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "layer": [0, 1, 2, 3],
            "mse_p95": [0.12, 0.80, 0.30, 0.55],
            "ip_p95": [0.25, 1.60, 0.40, 0.90],
            "effective_dim": [48.0, 11.0, 37.0, 21.0],
        }
    )


def main() -> None:
    args = parse_args()
    stats = pd.read_csv(args.input) if args.input else demo_stats()
    ranked = compute_risk_scores(stats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(args.output, index=False)
    print(f"Stage C key-risk profiling: {len(ranked)} layers -> {args.output}")


if __name__ == "__main__":
    main()
