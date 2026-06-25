from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tbgmp.kv_cache_wrapper import DryRunBackend, load_backend
from tbgmp.controls import bottomk_layers, sample_random_layers
from tbgmp.quantization import QuantizationConfig
from tbgmp.retrieval_eval import found_answer


def load_cases(path: Path) -> pd.DataFrame:
    cases = pd.read_csv(path)
    required = {"case_id", "context", "question", "answer"}
    missing = sorted(required - set(cases.columns))
    if missing:
        raise ValueError(f"case file missing columns: {missing}")
    return cases


def policy_config(name: str, policies: dict) -> QuantizationConfig | None:
    raw = policies[name]
    if raw.get("type") == "fp16":
        return None
    return QuantizationConfig(
        key_bits=int(raw["key_bits"]),
        value_bits=int(raw["value_bits"]),
        residual_window=int(raw.get("residual_window", 128)),
    )


def execute(
    *,
    backend,
    case: pd.Series,
    model_path: str,
    model_id: str,
    policy_name: str,
    policy_type: str,
    quantization: QuantizationConfig | None,
    max_new_tokens: int,
    stage: str,
) -> dict:
    prompt = f"{case['context']}\n\n{case['question']}"
    result = backend.generate(
        model_path=model_path,
        prompt=prompt,
        answer=str(case["answer"]),
        policy_name=policy_name,
        quantization=quantization,
        max_new_tokens=max_new_tokens,
    )
    if result.status == "success":
        result.found = found_answer(result.response, str(case["answer"]))
    return {
        "model": model_id,
        "case_id": case["case_id"],
        "domain": case.get("domain", ""),
        "context_length": case.get("context_length", ""),
        "depth": case.get("depth", ""),
        "seed": case.get("seed", ""),
        "answer": case["answer"],
        "stage": stage,
        "policy": policy_name,
        "policy_type": policy_type,
        "key_bits": 16 if quantization is None else quantization.key_bits,
        "value_bits": 16 if quantization is None else quantization.value_bits,
        "protected_key_bits": (
            "" if quantization is None else quantization.protected_key_bits or ""
        ),
        "protected_layers": (
            "" if quantization is None else list(quantization.protected_layers)
        ),
        "residual_window": (
            0 if quantization is None else quantization.residual_window
        ),
        **result.to_dict(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "External-backend experiment runner. This repository supplies the "
            "pipeline contract; users supply model weights and a compatible backend."
        )
    )
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument(
        "--policies",
        type=Path,
        default=ROOT / "configs" / "policies.yaml",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", help="External backend as module.path:factory")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument(
        "--risk-ranking",
        type=Path,
        help="CSV with layer and rank columns; required for Top-k stages.",
    )
    parser.add_argument("--maximum-topk", type=int, default=12)
    parser.add_argument("--random-seeds", default="0,1,2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dry_run and not args.backend:
        raise SystemExit("Provide --backend module:factory or use --dry-run")
    backend = DryRunBackend() if args.dry_run else load_backend(args.backend)
    cases = load_cases(args.cases)
    policies = yaml.safe_load(args.policies.read_text(encoding="utf-8"))
    discovery_names = [
        "fp16",
        "uniform_k2_v2_rw128",
        "uniform_k4_v2_rw128",
        "uniform_k6_v2_rw128",
        "uniform_k6_v4_rw128",
    ]

    rows: list[dict] = []
    discovery_by_case: dict[str, dict[str, dict]] = {}
    for _, case in cases.iterrows():
        case_results: dict[str, dict] = {}
        for name in discovery_names:
            row = execute(
                backend=backend,
                case=case,
                model_path=args.model_path,
                model_id=args.model_id,
                policy_name=name,
                policy_type="fp16" if name == "fp16" else "uniform",
                quantization=policy_config(name, policies),
                max_new_tokens=args.max_new_tokens,
                stage="stage_a_discovery",
            )
            rows.append(row)
            case_results[name] = row
        discovery_by_case[str(case["case_id"])] = case_results

    sensitive_cases = []
    for _, case in cases.iterrows():
        case_results = discovery_by_case[str(case["case_id"])]
        fp16 = case_results["fp16"]
        if fp16["status"] != "success" or not fp16["found"]:
            continue
        if not case_results["uniform_k4_v2_rw128"]["found"]:
            sensitive_cases.append((case, "uniform_k4_v2_rw128"))
        elif not case_results["uniform_k2_v2_rw128"]["found"]:
            sensitive_cases.append((case, "uniform_k2_v2_rw128"))

    if sensitive_cases and not args.risk_ranking:
        raise SystemExit("--risk-ranking is required when sensitive cases are found")

    if sensitive_cases:
        ranking = pd.read_csv(args.risk_ranking).sort_values("rank")
        ranked_layers = ranking["layer"].astype(int).tolist()
        tbgmp = policies["tbgmp_topk"]
        random_seeds = [int(value) for value in args.random_seeds.split(",")]
        maximum_topk = min(args.maximum_topk, len(ranked_layers))

        for case, aggressive_name in sensitive_cases:
            first_success_k = None
            for k in range(1, maximum_topk + 1):
                layers = tuple(ranked_layers[:k])
                config = QuantizationConfig(
                    key_bits=int(tbgmp["default_key_bits"]),
                    value_bits=int(tbgmp["default_value_bits"]),
                    protected_key_bits=int(tbgmp["protected_key_bits"]),
                    protected_layers=layers,
                    residual_window=int(tbgmp.get("residual_window", 128)),
                )
                row = execute(
                    backend=backend,
                    case=case,
                    model_path=args.model_path,
                    model_id=args.model_id,
                    policy_name=f"tbgmp_top{k}",
                    policy_type="tbgmp_topk",
                    quantization=config,
                    max_new_tokens=args.max_new_tokens,
                    stage="stage_d_topk_recovery",
                )
                row["aggressive_policy"] = aggressive_name
                row["topk_k"] = k
                rows.append(row)
                if first_success_k is None and row["found"]:
                    first_success_k = k

            if first_success_k is None:
                continue
            k = first_success_k
            for seed in random_seeds:
                random_layers = tuple(sample_random_layers(ranked_layers, k, seed))
                config = QuantizationConfig(
                    key_bits=int(tbgmp["default_key_bits"]),
                    value_bits=int(tbgmp["default_value_bits"]),
                    protected_key_bits=int(tbgmp["protected_key_bits"]),
                    protected_layers=random_layers,
                    residual_window=int(tbgmp.get("residual_window", 128)),
                )
                row = execute(
                    backend=backend,
                    case=case,
                    model_path=args.model_path,
                    model_id=args.model_id,
                    policy_name=f"random{k}_seed{seed}",
                    policy_type="random_k",
                    quantization=config,
                    max_new_tokens=args.max_new_tokens,
                    stage="stage_e_controls",
                )
                row["control_seed"] = seed
                row["matched_topk_k"] = k
                rows.append(row)

            bottom_layers = tuple(bottomk_layers(ranked_layers, k))
            config = QuantizationConfig(
                key_bits=int(tbgmp["default_key_bits"]),
                value_bits=int(tbgmp["default_value_bits"]),
                protected_key_bits=int(tbgmp["protected_key_bits"]),
                protected_layers=bottom_layers,
                residual_window=int(tbgmp.get("residual_window", 128)),
            )
            row = execute(
                backend=backend,
                case=case,
                model_path=args.model_path,
                model_id=args.model_id,
                policy_name=f"bottom{k}",
                policy_type="bottom_k",
                quantization=config,
                max_new_tokens=args.max_new_tokens,
                stage="stage_e_controls",
            )
            row["matched_topk_k"] = k
            rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    metadata = {
        "model_id": args.model_id,
        "backend": "dry_run" if args.dry_run else args.backend,
        "cases": len(cases),
        "policies": discovery_names,
        "sensitive_cases": len(sensitive_cases),
        "maximum_topk": args.maximum_topk,
        "model_weights_in_repository": False,
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
