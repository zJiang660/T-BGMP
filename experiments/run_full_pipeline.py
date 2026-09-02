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
from tbgmp.backends.turboquant_backend import TurboQuantBackend
from tbgmp.prompting import render_retrieval_prompt
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


def protected_config(
    aggressive: QuantizationConfig,
    tbgmp_policy: dict,
    layers: tuple[int, ...],
) -> QuantizationConfig:
    """Protect ranked keys while preserving the case's aggressive baseline."""
    config = QuantizationConfig(
        key_bits=aggressive.key_bits,
        value_bits=aggressive.value_bits,
        protected_key_bits=int(tbgmp_policy["protected_key_bits"]),
        protected_layers=layers,
        residual_window=aggressive.residual_window,
    )
    config.validate()
    return config


def execute(
    *,
    backend,
    case: pd.Series,
    prompt: str,
    model_path: str,
    model_id: str,
    policy_name: str,
    policy_type: str,
    quantization: QuantizationConfig | None,
    max_new_tokens: int,
    seed: int,
    stage: str,
) -> dict:
    try:
        result = backend.generate(
            model_path=model_path,
            prompt=prompt,
            answer=str(case["answer"]),
            policy_name=policy_name,
            quantization=quantization,
            max_new_tokens=max_new_tokens,
            seed=seed,
            add_special_tokens=False,
        )
    except (NotImplementedError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from None
    if result.status == "success":
        result.found = found_answer(result.response, str(case["answer"]))
    if hasattr(result, "to_dict"):
        result_fields = result.to_dict()
    else:
        metadata = dict(getattr(result, "metadata", {}))
        result_fields = {
            "response": result.response,
            "found": result.found,
            "status": result.status,
            "error": metadata.get("error", ""),
            "runtime_s": metadata.get("runtime_s"),
            "tok_per_s": metadata.get("tok_per_s"),
            "peak_gpu_gb": metadata.get("peak_gpu_gb"),
            "kv_saving": metadata.get("kv_saving"),
            "actual_context_tokens": metadata.get("actual_context_tokens", ""),
        }
    result_fields.setdefault("actual_context_tokens", "")
    return {
        "model": model_id,
        "case_id": case["case_id"],
        "domain": case.get("domain", ""),
        "context_length": case.get("context_length", ""),
        "document_tokens": case.get(
            "document_tokens", case.get("actual_context_tokens", "")
        ),
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
        **result_fields,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "External-backend experiment runner. This repository supplies the "
            "pipeline contract; users supply model weights and a compatible backend."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "default_experiment.yaml",
    )
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--model-path")
    parser.add_argument("--model-id")
    parser.add_argument("--model-key")
    parser.add_argument("--model-root", type=Path)
    parser.add_argument(
        "--model-registry",
        type=Path,
        default=ROOT / "configs" / "model_registry.yaml",
    )
    parser.add_argument(
        "--policies",
        type=Path,
        default=ROOT / "configs" / "policies.yaml",
    )
    parser.add_argument("--prompt-template", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--backend",
        help="'turboquant' or an external backend as module.path:factory",
    )
    parser.add_argument("--turboquant-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--risk-ranking",
        type=Path,
        help="CSV with layer and rank columns; required for Top-k stages.",
    )
    parser.add_argument("--maximum-topk", type=int)
    parser.add_argument("--random-seeds")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> None:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    experiment = config.get("experiment", {})
    args.max_new_tokens = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else int(experiment.get("max_new_tokens", 32))
    )
    args.seed = (
        args.seed if args.seed is not None else int(experiment.get("default_seed", 0))
    )
    args.maximum_topk = (
        args.maximum_topk
        if args.maximum_topk is not None
        else int(experiment.get("maximum_topk", 12))
    )
    if args.random_seeds is None:
        values = experiment.get("random_seeds", [0, 1, 2])
        args.random_seeds = ",".join(str(value) for value in values)
    if args.prompt_template is None:
        configured_prompt = config.get("inputs", {}).get(
            "prompt_template", "configs/prompt_template.yaml"
        )
        args.prompt_template = Path(configured_prompt)
        if not args.prompt_template.is_absolute():
            args.prompt_template = ROOT / args.prompt_template
    if args.cases is None:
        configured_cases = config.get("inputs", {}).get("cases")
        if configured_cases and not str(configured_cases).startswith("/path/to/"):
            args.cases = Path(configured_cases)
    if args.cases is None and args.dry_run:
        args.cases = ROOT / "data" / "demo" / "full_runner_cases.csv"
    if args.cases is None:
        raise SystemExit("Provide --cases with a prompt-case CSV.")

    if args.model_key:
        registry = yaml.safe_load(args.model_registry.read_text(encoding="utf-8"))
        models = registry.get("models", {})
        if args.model_key not in models:
            raise SystemExit(f"Unknown --model-key: {args.model_key}")
        entry = models[args.model_key]
        args.model_id = args.model_id or entry.get("display_name") or args.model_key
        if args.model_path is None:
            if args.model_root is None:
                raise SystemExit(
                    "Provide --model-root when resolving a model from --model-key."
                )
            args.model_path = str(args.model_root / entry["local_dir"])

    if not args.model_path or not args.model_id:
        raise SystemExit(
            "Provide --model-path and --model-id, or use --model-key with "
            "--model-root."
        )
    if args.output is None:
        if args.output_dir is None:
            raise SystemExit("Provide --output or --output-dir.")
        args.output = args.output_dir / "full_pipeline_results.csv"


def main() -> None:
    args = parse_args()
    resolve_paths(args)
    if not args.dry_run and not args.backend:
        raise SystemExit(
            "Full model execution requires the external TurboQuant backend. "
            "See docs/backend_integration.md."
        )
    if args.dry_run:
        print("Stage A: discovery")
        print("Stage B: mine sensitive cases")
        print("Stage C: profile key risk")
        print("Stage D: Top-k recovery")
        print("Stage E: Random/Bottom controls")
        print("Stage F: efficiency analysis")
        print(f"Backend: {args.backend or 'dry_run'}")
        print(f"Model key: {args.model_key or ''}")
        print("No model execution performed in dry-run mode.")
        backend = DryRunBackend()
    elif args.backend == "turboquant":
        try:
            backend = TurboQuantBackend(
                turboquant_root=(
                    str(args.turboquant_root) if args.turboquant_root else None
                )
            )
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from None
    else:
        backend = load_backend(args.backend)
    cases = load_cases(args.cases)
    policies = yaml.safe_load(args.policies.read_text(encoding="utf-8"))
    prompt_config = yaml.safe_load(args.prompt_template.read_text(encoding="utf-8"))
    tokenizer = None
    if not args.dry_run:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise SystemExit(
                "Full execution requires transformers for prompt rendering."
            ) from exc
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
    prompts = {
        str(case["case_id"]): render_retrieval_prompt(
            str(case["context"]),
            str(case["question"]),
            prompt_config=prompt_config,
            tokenizer=tokenizer,
        )
        for _, case in cases.iterrows()
    }
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
                prompt=prompts[str(case["case_id"])],
                model_path=args.model_path,
                model_id=args.model_id,
                policy_name=name,
                policy_type="fp16" if name == "fp16" else "uniform",
                quantization=policy_config(name, policies),
                max_new_tokens=args.max_new_tokens,
                seed=args.seed,
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
            aggressive_config = policy_config(aggressive_name, policies)
            if aggressive_config is None:
                raise RuntimeError("Sensitive-case aggressive policy cannot be FP16.")
            first_success_k = None
            for k in range(1, maximum_topk + 1):
                layers = tuple(ranked_layers[:k])
                config = protected_config(aggressive_config, tbgmp, layers)
                row = execute(
                    backend=backend,
                    case=case,
                    prompt=prompts[str(case["case_id"])],
                    model_path=args.model_path,
                    model_id=args.model_id,
                    policy_name=f"tbgmp_top{k}",
                    policy_type="tbgmp_topk",
                    quantization=config,
                    max_new_tokens=args.max_new_tokens,
                    seed=args.seed,
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
                config = protected_config(aggressive_config, tbgmp, random_layers)
                row = execute(
                    backend=backend,
                    case=case,
                    prompt=prompts[str(case["case_id"])],
                    model_path=args.model_path,
                    model_id=args.model_id,
                    policy_name=f"random{k}_seed{seed}",
                    policy_type="random_k",
                    quantization=config,
                    max_new_tokens=args.max_new_tokens,
                    seed=seed,
                    stage="stage_e_controls",
                )
                row["control_seed"] = seed
                row["matched_topk_k"] = k
                rows.append(row)

            bottom_layers = tuple(bottomk_layers(ranked_layers, k))
            config = protected_config(aggressive_config, tbgmp, bottom_layers)
            row = execute(
                backend=backend,
                case=case,
                prompt=prompts[str(case["case_id"])],
                model_path=args.model_path,
                model_id=args.model_id,
                policy_name=f"bottom{k}",
                policy_type="bottom_k",
                quantization=config,
                max_new_tokens=args.max_new_tokens,
                seed=args.seed,
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
        "seed": args.seed,
        "model_weights_in_repository": False,
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
