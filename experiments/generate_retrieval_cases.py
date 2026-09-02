from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tbgmp.case_generation import generate_case_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the domain/context/depth/seed retrieval grid with a "
            "model tokenizer and deterministic hidden answers."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "default_experiment.yaml",
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Case generation requires transformers for exact tokenizer budgets."
        ) from exc

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    experiment = config.get("experiment", {})
    generation = config.get("case_generation", {})
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    cases = generate_case_grid(
        tokenizer,
        experiment,
        generation,
        base_dir=args.config.resolve().parent.parent,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases.to_csv(args.output, index=False)
    print(f"Generated {len(cases)} retrieval cases -> {args.output}")


if __name__ == "__main__":
    main()
