from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tbgmp.metrics import kv_saving_advantage
from tbgmp.utils import parse_bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize KV-saving advantage for restored Top-k cases."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "default_experiment.yaml",
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
    parser.add_argument("--model-key")
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--backend")
    parser.add_argument("--turboquant-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.output_dir:
        args.output = args.output_dir / "efficiency_summary.csv"
    if args.dry_run:
        print(f"Stage F: efficiency analysis -> {args.output}")
        print("No model execution performed in dry-run mode.")
        return
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
