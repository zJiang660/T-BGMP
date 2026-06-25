from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tbgmp.metrics import kv_saving_advantage
from tbgmp.utils import parse_bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize KV-saving advantage for restored Top-k cases."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "results" / "audit" / "demo_topk_recovery.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "audit" / "demo_efficiency_summary.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = pd.read_csv(args.input)
    required = {"found", "tbgmp_saving", "uniform_safe_saving"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise SystemExit(f"Stage F input is missing columns: {missing}")
    rows["found"] = rows["found"].map(parse_bool)
    recovered = rows.loc[rows["found"]].copy()
    recovered["saving_advantage"] = recovered.apply(
        lambda row: kv_saving_advantage(
            float(row["tbgmp_saving"]),
            float(row["uniform_safe_saving"]),
        ),
        axis=1,
    )
    summary = pd.DataFrame(
        [
            {
                "restored_cases": len(recovered),
                "mean_tbgmp_kv_saving": recovered["tbgmp_saving"].mean(),
                "mean_uniform_safe_kv_saving": recovered["uniform_safe_saving"].mean(),
                "mean_saving_advantage": recovered["saving_advantage"].mean(),
            }
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(f"Stage F efficiency summary -> {args.output}")


if __name__ == "__main__":
    main()
