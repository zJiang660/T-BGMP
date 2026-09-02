from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tbgmp.profiling import profile_model_key_risk
from tbgmp.risk_score import compute_risk_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rank layers from either precomputed key-distortion statistics or "
            "a real model key-cache profiling pass."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "default_experiment.yaml",
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
    parser.add_argument("--model-key")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--model-id")
    parser.add_argument("--model-root", type=Path)
    parser.add_argument(
        "--model-registry",
        type=Path,
        default=ROOT / "configs" / "model_registry.yaml",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--backend")
    parser.add_argument("--turboquant-root", type=Path)
    parser.add_argument("--context-file", type=Path)
    parser.add_argument("--context-length", type=int)
    parser.add_argument("--bits", type=int)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--ip-pairs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument(
        "--demo-stats",
        action="store_true",
        help="Explicitly rank the four-layer model-free demonstration statistics.",
    )
    parser.add_argument("--dry-run", action="store_true")
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
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.output_dir:
        args.output = args.output_dir / "risk_ranking.csv"
    if args.dry_run:
        print(f"Stage C: profile key risk -> {args.output}")
        print("No model execution performed in dry-run mode.")
        return
    if args.model_path is None and args.model_key and args.model_root:
        registry = yaml.safe_load(args.model_registry.read_text(encoding="utf-8"))
        models = registry.get("models", {})
        if args.model_key not in models:
            raise SystemExit(f"Unknown --model-key: {args.model_key}")
        entry = models[args.model_key]
        args.model_path = args.model_root / entry["local_dir"]
        args.model_id = args.model_id or entry.get("display_name") or args.model_key
    profiling = config.get("profiling", {})
    if args.context_file is None and args.model_path:
        domain = str(profiling.get("domain", "literature"))
        source = config.get("case_generation", {}).get("domain_sources", {}).get(domain)
        if source:
            args.context_file = Path(source)
            if not args.context_file.is_absolute():
                args.context_file = ROOT / args.context_file
    if args.turboquant_root is None:
        configured_root = os.environ.get("TURBOQUANT_ROOT")
        if configured_root:
            args.turboquant_root = Path(configured_root)
    if args.input:
        ranked = compute_risk_scores(pd.read_csv(args.input))
    elif args.model_path and args.context_file and args.turboquant_root:
        ranked = profile_model_key_risk(
            model_path=args.model_path,
            model_id=args.model_id or args.model_key or args.model_path.name,
            context_file=args.context_file,
            turboquant_root=args.turboquant_root,
            context_length=args.context_length
            or int(profiling.get("context_length", 4096)),
            bits=args.bits or int(profiling.get("quant_bits", 2)),
            max_rows=args.max_rows or int(profiling.get("max_rows", 20000)),
            ip_pairs=args.ip_pairs or int(profiling.get("ip_pairs", 8192)),
            seed=args.seed if args.seed is not None else int(profiling.get("seed", 42)),
            device=args.device or str(profiling.get("device", "cuda")),
            load_in_4bit=args.load_in_4bit
            or bool(profiling.get("load_in_4bit", False)),
        )
    elif args.model_path or args.context_file or args.turboquant_root:
        raise SystemExit(
            "Real profiling requires --model-path, --context-file, and "
            "--turboquant-root together."
        )
    elif args.demo_stats:
        ranked = compute_risk_scores(demo_stats())
    else:
        raise SystemExit(
            "Stage C requires --input, a complete real-profiling model setup, "
            "or the explicit --demo-stats flag."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(args.output, index=False)
    print(f"Stage C key-risk profiling: {len(ranked)} layers -> {args.output}")


if __name__ == "__main__":
    main()
