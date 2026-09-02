from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tbgmp.kv_cache_wrapper import DryRunBackend, load_backend
from tbgmp.case_generation import generate_case_grid
from tbgmp.controls import bottomk_layers, sample_random_layers
from tbgmp.backends.turboquant_backend import TurboQuantBackend
from tbgmp.experiment_config import configured_path, validate_experiment_config
from tbgmp.metrics import kv_saving_advantage
from tbgmp.prompting import render_retrieval_prompt
from tbgmp.quantization import QuantizationConfig, estimate_nominal_kv_saving
from tbgmp.retrieval_eval import found_answer
from tbgmp.result_store import IncrementalResultStore
from tbgmp.utils import parse_bool


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
                "actual_context_tokens": metadata.get(
                    "actual_context_tokens", ""
                ),
            }
        status = str(result_fields.get("status", ""))
        error = str(result_fields.get("error", "") or "")
        result_fields["oom"] = status.lower() == "oom" or "out of memory" in error.lower()
        result_fields["completed"] = status == "dry_run" or (
            status == "success" and not error
        )
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}".replace("\n", " ").strip()
        is_oom = "out of memory" in message.lower() or "cuda oom" in message.lower()
        result_fields = {
            "response": "",
            "found": False,
            "status": "oom" if is_oom else "error",
            "error": message,
            "runtime_s": None,
            "tok_per_s": None,
            "peak_gpu_gb": None,
            "kv_saving": None,
            "actual_context_tokens": "",
            "oom": is_oom,
            "completed": False,
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
    )
    parser.add_argument(
        "--policies",
        type=Path,
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
    parser.add_argument("--checkpoint-every", type=int)
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[dict, dict, dict]:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    experiment = config.get("experiment", {})
    inputs = config.get("inputs", {})
    outputs = config.get("outputs", {})
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
    args.checkpoint_every = (
        args.checkpoint_every
        if args.checkpoint_every is not None
        else int(outputs.get("checkpoint_every", 25))
    )
    if args.checkpoint_every <= 0:
        raise SystemExit("--checkpoint-every must be positive")
    if args.model_registry is None:
        args.model_registry = configured_path(
            ROOT, inputs.get("model_registry", "configs/model_registry.yaml")
        ) or ROOT / "configs" / "model_registry.yaml"
    if args.policies is None:
        args.policies = configured_path(
            ROOT, inputs.get("policies", "configs/policies.yaml")
        ) or ROOT / "configs" / "policies.yaml"
    if args.prompt_template is None:
        configured_prompt = inputs.get(
            "prompt_template", "configs/prompt_template.yaml"
        )
        args.prompt_template = configured_path(ROOT, configured_prompt)
    if args.prompt_template is None:
        raise SystemExit("Configure inputs.prompt_template or pass --prompt-template")
    if args.cases is None:
        args.cases = configured_path(ROOT, inputs.get("cases"))
    if args.cases is None and args.dry_run:
        args.cases = ROOT / "data" / "demo" / "full_runner_cases.csv"
    if args.risk_ranking is None:
        args.risk_ranking = configured_path(ROOT, inputs.get("risk_ranking"))

    backend_config_path = configured_path(
        ROOT, config.get("backend", {}).get("config")
    )
    if backend_config_path and backend_config_path.is_file():
        backend_config = yaml.safe_load(
            backend_config_path.read_text(encoding="utf-8")
        ).get("backend", {})
        args.backend = args.backend or backend_config.get("name")
        configured_tq = configured_path(ROOT, backend_config.get("turboquant_root"))
        args.turboquant_root = args.turboquant_root or configured_tq

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
        if args.output_dir is not None:
            args.output = args.output_dir / "full_pipeline_results.csv"
        else:
            args.output = configured_path(ROOT, outputs.get("results"))
    if args.output is None:
        raise SystemExit("Provide --output/--output-dir or configure outputs.results")

    policies = yaml.safe_load(args.policies.read_text(encoding="utf-8"))
    try:
        protocol = validate_experiment_config(config, policies)
    except ValueError as exc:
        raise SystemExit(f"Invalid experiment config: {exc}") from None
    return config, policies, protocol


def row_bool(row: dict, field: str) -> bool:
    value = row.get(field, False)
    if isinstance(value, bool):
        return value
    try:
        return parse_bool(value)
    except ValueError:
        return False


def load_tokenizer(backend, model_path: str):
    if hasattr(backend, "get_tokenizer"):
        return backend.get_tokenizer(model_path)
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Full execution requires a backend-owned tokenizer or transformers."
        ) from exc
    return AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )


def write_dataframe_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def write_json_atomic(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def prepare_run_manifest(
    *,
    output: Path,
    model_id: str,
    cases: pd.DataFrame,
    experiment: dict,
    selection: dict,
    policies: dict,
    prompt_config: dict,
    risk_ranking: Path | None,
    runtime: dict,
) -> tuple[Path, str]:
    case_hashes = pd.util.hash_pandas_object(
        cases.astype(str), index=True
    ).values.tobytes()
    ranking_hash = ""
    if risk_ranking and risk_ranking.is_file():
        ranking_hash = hashlib.sha256(risk_ranking.read_bytes()).hexdigest()
    signature_payload = {
        "model_id": model_id,
        "cases_sha256": hashlib.sha256(case_hashes).hexdigest(),
        "experiment": experiment,
        "selection": selection,
        "policies": policies,
        "prompt": prompt_config,
        "runtime": runtime,
    }
    canonical = json.dumps(
        signature_payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    signature = hashlib.sha256(canonical).hexdigest()
    path = output.with_suffix(".run.json")
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("signature") != signature:
            raise SystemExit(
                "Existing output was created with a different experiment "
                "signature; use a new output path instead of mixing runs"
            )
        existing_ranking = str(existing.get("risk_ranking_sha256", ""))
        if existing_ranking and ranking_hash and existing_ranking != ranking_hash:
            raise SystemExit(
                "Existing output uses a different risk ranking; use a new "
                "output path instead of mixing rankings"
            )
        if not existing_ranking and ranking_hash:
            downstream_exists = False
            if output.is_file() and output.stat().st_size:
                existing_rows = pd.read_csv(output, usecols=["stage"])
                downstream_exists = any(
                    stage != "stage_a_discovery"
                    for stage in existing_rows["stage"].astype(str)
                )
            if downstream_exists:
                raise SystemExit(
                    "Cannot attach a risk ranking after downstream stages exist"
                )
            write_json_atomic(
                {
                    **existing,
                    "risk_ranking_sha256": ranking_hash,
                },
                path,
            )
    else:
        journal = output.with_suffix(".jsonl")
        if output.exists() or journal.exists():
            raise SystemExit(
                "Existing resumable output has no run signature; use a new "
                "output path to avoid mixing protocols"
            )
        write_json_atomic(
            {
                "signature": signature,
                "risk_ranking_sha256": ranking_hash,
                **signature_payload,
            },
            path,
        )
    return path, signature


def run_combination(
    store: IncrementalResultStore,
    *,
    extra: dict | None = None,
    **execute_kwargs,
) -> dict:
    identity = {
        "model": execute_kwargs["model_id"],
        "case_id": execute_kwargs["case"]["case_id"],
        "stage": execute_kwargs["stage"],
        "policy": execute_kwargs["policy_name"],
    }
    if store.completed(identity):
        existing = store.get(identity)
        if existing is None:
            raise RuntimeError("completed result disappeared from result store")
        return dict(existing)
    row = execute(**execute_kwargs)
    if extra:
        row.update(extra)
    store.append(row)
    return row


def parse_protected_layers(value) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(int(item) for item in value)
    if value in ("", None):
        return ()
    parsed = ast.literal_eval(str(value))
    return tuple(int(item) for item in parsed)


def quantization_from_row(row: dict) -> QuantizationConfig:
    protected_bits = row.get("protected_key_bits", "")
    return QuantizationConfig(
        key_bits=int(row["key_bits"]),
        value_bits=int(row["value_bits"]),
        residual_window=int(row.get("residual_window", 128)),
        protected_key_bits=(
            None if protected_bits in ("", None) else int(protected_bits)
        ),
        protected_layers=parse_protected_layers(row.get("protected_layers", "")),
    )


def saving_for_row(row: dict, config: QuantizationConfig, num_layers: int) -> float:
    value = row.get("kv_saving", "")
    try:
        if value not in ("", None) and not pd.isna(value):
            return float(value)
    except (TypeError, ValueError):
        pass
    return estimate_nominal_kv_saving(config, num_layers)


def add_stage_f_rows(
    store: IncrementalResultStore,
    *,
    model_id: str,
    recovered: list[tuple[pd.Series, dict]],
    discovery_by_case: dict[str, dict[str, dict]],
    safe_policies: list[str],
    policies: dict,
    num_layers: int,
) -> None:
    for case, topk_row in recovered:
        top_config = quantization_from_row(topk_row)
        top_saving = saving_for_row(topk_row, top_config, num_layers)
        case_results = discovery_by_case[str(case["case_id"])]
        for safe_name in safe_policies:
            safe_row = case_results[safe_name]
            safe_config = policy_config(safe_name, policies)
            if safe_config is None:
                raise ValueError("Stage F safe policy cannot be FP16")
            safe_saving = saving_for_row(safe_row, safe_config, num_layers)
            comparable = (
                str(safe_row.get("status")) == "success"
                and row_bool(safe_row, "found")
            )
            row = {
                "model": model_id,
                "case_id": case["case_id"],
                "domain": case.get("domain", ""),
                "context_length": case.get("context_length", ""),
                "depth": case.get("depth", ""),
                "seed": case.get("seed", ""),
                "stage": "stage_f_efficiency",
                "policy": f"{topk_row['policy']}_vs_{safe_name}",
                "policy_type": "efficiency_pair",
                "tbgmp_policy": topk_row["policy"],
                "safe_policy": safe_name,
                "found": True,
                "safe_found": row_bool(safe_row, "found"),
                "comparison_valid": comparable,
                "tbgmp_saving": top_saving,
                "uniform_safe_saving": safe_saving,
                "saving_advantage": kv_saving_advantage(
                    top_saving, safe_saving
                ),
                "saving_basis": "backend_reported_or_nominal_bit_budget",
                "status": "success" if comparable else "excluded",
                "error": "" if comparable else "safe baseline did not retrieve",
                "oom": False,
                "completed": True,
            }
            identity = {
                field: row[field]
                for field in ("model", "case_id", "stage", "policy")
            }
            if not store.completed(identity):
                store.append(row)


def write_stage_f_outputs(store: IncrementalResultStore, output: Path) -> None:
    rows = pd.DataFrame(store.rows(stage="stage_f_efficiency"))
    if rows.empty:
        rows = pd.DataFrame(
            columns=[
                "model",
                "case_id",
                "tbgmp_policy",
                "safe_policy",
                "comparison_valid",
                "tbgmp_saving",
                "uniform_safe_saving",
                "saving_advantage",
                "saving_basis",
            ]
        )
    detail_path = output.with_name(f"{output.stem}_stage_f.csv")
    summary_path = output.with_name(f"{output.stem}_stage_f_summary.csv")
    write_dataframe_atomic(rows, detail_path)
    valid = rows.loc[rows.get("comparison_valid", pd.Series(dtype=bool)) == True]
    summary = pd.DataFrame(
        [
            {
                "valid_pairs": len(valid),
                "restored_cases": valid["case_id"].nunique() if len(valid) else 0,
                "mean_tbgmp_kv_saving": (
                    valid["tbgmp_saving"].mean() if len(valid) else None
                ),
                "mean_uniform_safe_kv_saving": (
                    valid["uniform_safe_saving"].mean() if len(valid) else None
                ),
                "mean_saving_advantage": (
                    valid["saving_advantage"].mean() if len(valid) else None
                ),
            }
        ]
    )
    write_dataframe_atomic(summary, summary_path)


def main() -> None:
    args = parse_args()
    config, policies, protocol = resolve_paths(args)
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

    prompt_config = yaml.safe_load(args.prompt_template.read_text(encoding="utf-8"))
    tokenizer = None
    if args.cases is not None:
        cases = load_cases(args.cases)
    else:
        if args.dry_run:
            raise RuntimeError("dry-run case fallback should have been resolved")
        tokenizer = load_tokenizer(backend, args.model_path)
        cases = generate_case_grid(
            tokenizer,
            config["experiment"],
            config.get("case_generation", {}),
            base_dir=ROOT,
        )
        generated_path = configured_path(
            ROOT, config.get("outputs", {}).get("generated_cases")
        ) or args.output.with_name(f"{args.output.stem}_cases.csv")
        write_dataframe_atomic(cases, generated_path)
        args.cases = generated_path

    discovery_names = protocol["discovery_policies"]
    run_manifest, run_signature = prepare_run_manifest(
        output=args.output,
        model_id=args.model_id,
        cases=cases,
        experiment=config["experiment"],
        selection=config.get("selection", {}),
        policies=policies,
        prompt_config=prompt_config,
        risk_ranking=args.risk_ranking,
        runtime={
            "backend": "dry_run" if args.dry_run else args.backend,
            "model_path": str(args.model_path),
            "maximum_topk": args.maximum_topk,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "random_seeds": args.random_seeds,
        },
    )
    if tokenizer is None and not args.dry_run:
        tokenizer = load_tokenizer(backend, args.model_path)
    prompts = {
        str(case["case_id"]): render_retrieval_prompt(
            str(case["context"]),
            str(case["question"]),
            prompt_config=prompt_config,
            tokenizer=tokenizer,
        )
        for _, case in cases.iterrows()
    }
    store = IncrementalResultStore(
        args.output,
        checkpoint_every=args.checkpoint_every,
    )
    discovery_by_case: dict[str, dict[str, dict]] = {}
    sensitive_cases: list[tuple[pd.Series, str]] = []
    recovered: list[tuple[pd.Series, dict]] = []
    ranked_layers: list[int] = []
    try:
        for _, case in cases.iterrows():
            case_results: dict[str, dict] = {}
            for name in discovery_names:
                row = run_combination(
                    store,
                    backend=backend,
                    case=case,
                    prompt=prompts[str(case["case_id"])],
                    model_path=args.model_path,
                    model_id=args.model_id,
                    policy_name=name,
                    policy_type=(
                        "fp16" if policies[name].get("type") == "fp16" else "uniform"
                    ),
                    quantization=policy_config(name, policies),
                    max_new_tokens=args.max_new_tokens,
                    seed=args.seed,
                    stage="stage_a_discovery",
                )
                case_results[name] = row
            discovery_by_case[str(case["case_id"])] = case_results

        selection = config.get("selection", {})
        required_aggressive_found = bool(
            selection.get("require_aggressive_found", False)
        )
        for _, case in cases.iterrows():
            case_results = discovery_by_case[str(case["case_id"])]
            fp16 = case_results["fp16"]
            if selection.get("exclude_execution_errors", True) and str(
                fp16.get("status")
            ) != "success":
                continue
            if selection.get("require_fp16_found", True) and not row_bool(
                fp16, "found"
            ):
                continue
            for aggressive_name in protocol["aggressive_policies"]:
                aggressive_row = case_results[aggressive_name]
                if str(aggressive_row.get("status")) != "success":
                    continue
                if row_bool(aggressive_row, "found") == required_aggressive_found:
                    sensitive_cases.append((case, aggressive_name))
                    break

        if sensitive_cases and not args.risk_ranking:
            raise SystemExit(
                "--risk-ranking is required when sensitive cases are found; "
                "completed discovery rows have been checkpointed"
            )

        if sensitive_cases:
            ranking = pd.read_csv(args.risk_ranking).sort_values("rank")
            if not {"layer", "rank"}.issubset(ranking.columns):
                raise SystemExit("risk ranking must contain layer and rank columns")
            if ranking["layer"].duplicated().any():
                raise SystemExit("risk ranking contains duplicate layers")
            ranked_layers = ranking["layer"].astype(int).tolist()
            tbgmp = policies[protocol["tbgmp_policy"]]
            random_seeds = [int(value) for value in args.random_seeds.split(",")]
            maximum_topk = min(args.maximum_topk, len(ranked_layers))

            for case, aggressive_name in sensitive_cases:
                aggressive_config = policy_config(aggressive_name, policies)
                if aggressive_config is None:
                    raise RuntimeError(
                        "Sensitive-case aggressive policy cannot be FP16."
                    )
                first_success: dict | None = None
                for k in range(1, maximum_topk + 1):
                    layers = tuple(ranked_layers[:k])
                    top_config = protected_config(aggressive_config, tbgmp, layers)
                    row = run_combination(
                        store,
                        extra={"aggressive_policy": aggressive_name, "topk_k": k},
                        backend=backend,
                        case=case,
                        prompt=prompts[str(case["case_id"])],
                        model_path=args.model_path,
                        model_id=args.model_id,
                        policy_name=f"tbgmp_top{k}",
                        policy_type="tbgmp_topk",
                        quantization=top_config,
                        max_new_tokens=args.max_new_tokens,
                        seed=args.seed,
                        stage="stage_d_topk_recovery",
                    )
                    if (
                        first_success is None
                        and str(row.get("status")) == "success"
                        and row_bool(row, "found")
                    ):
                        first_success = row

                if first_success is None:
                    continue
                recovered.append((case, first_success))
                k = int(first_success["topk_k"])
                for control_seed in random_seeds:
                    random_layers = tuple(
                        sample_random_layers(ranked_layers, k, control_seed)
                    )
                    control_config = protected_config(
                        aggressive_config, tbgmp, random_layers
                    )
                    run_combination(
                        store,
                        extra={
                            "control_seed": control_seed,
                            "matched_topk_k": k,
                            "aggressive_policy": aggressive_name,
                        },
                        backend=backend,
                        case=case,
                        prompt=prompts[str(case["case_id"])],
                        model_path=args.model_path,
                        model_id=args.model_id,
                        policy_name=f"random{k}_seed{control_seed}",
                        policy_type="random_k",
                        quantization=control_config,
                        max_new_tokens=args.max_new_tokens,
                        seed=control_seed,
                        stage="stage_e_controls",
                    )

                bottom_layers = tuple(bottomk_layers(ranked_layers, k))
                bottom_config = protected_config(
                    aggressive_config, tbgmp, bottom_layers
                )
                run_combination(
                    store,
                    extra={
                        "matched_topk_k": k,
                        "aggressive_policy": aggressive_name,
                    },
                    backend=backend,
                    case=case,
                    prompt=prompts[str(case["case_id"])],
                    model_path=args.model_path,
                    model_id=args.model_id,
                    policy_name=f"bottom{k}",
                    policy_type="bottom_k",
                    quantization=bottom_config,
                    max_new_tokens=args.max_new_tokens,
                    seed=args.seed,
                    stage="stage_e_controls",
                )

            add_stage_f_rows(
                store,
                model_id=args.model_id,
                recovered=recovered,
                discovery_by_case=discovery_by_case,
                safe_policies=protocol["safe_policies"],
                policies=policies,
                num_layers=len(ranked_layers),
            )
    finally:
        store.close()

    write_stage_f_outputs(store, args.output)
    incomplete_rows = [
        row for row in store.rows() if not row_bool(row, "completed")
    ]
    oom_rows = [row for row in incomplete_rows if row_bool(row, "oom")]
    metadata = {
        "model_id": args.model_id,
        "backend": "dry_run" if args.dry_run else args.backend,
        "cases": len(cases),
        "policies": discovery_names,
        "sensitive_cases": len(sensitive_cases),
        "recovered_cases": len(recovered),
        "maximum_topk": args.maximum_topk,
        "seed": args.seed,
        "checkpoint_every": args.checkpoint_every,
        "resume_skipped": store.skipped,
        "new_attempts": store.appended,
        "incomplete_rows": len(incomplete_rows),
        "oom_rows": len(oom_rows),
        "journal": str(store.journal_path),
        "run_manifest": str(run_manifest),
        "run_signature": run_signature,
        "stage_f_detail": str(
            args.output.with_name(f"{args.output.stem}_stage_f.csv")
        ),
        "model_weights_in_repository": False,
    }
    write_json_atomic(
        metadata,
        args.output.with_suffix(".metadata.json"),
    )
    print(
        f"Results: {len(store.rows())} current rows -> {args.output}; "
        f"new attempts={store.appended}, resumed/skipped={store.skipped}"
    )
    if incomplete_rows:
        raise SystemExit(
            f"{len(incomplete_rows)} combinations remain incomplete "
            f"({len(oom_rows)} OOM); rerun the same command to resume"
        )


if __name__ == "__main__":
    main()
