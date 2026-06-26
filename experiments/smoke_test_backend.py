from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tbgmp.backends.base import GenerationRequest
from tbgmp.backends.turboquant_backend import TurboQuantBackend
from tbgmp.kv_cache_wrapper import load_backend
from tbgmp.retrieval_eval import found_answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one minimal backend generation smoke test. This script never "
            "fabricates responses: if the backend is not bound to real "
            "generation, it exits before writing raw output."
        )
    )
    parser.add_argument("--backend", default="turboquant")
    parser.add_argument("--turboquant-root", type=Path)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument(
        "--model-registry",
        type=Path,
        default=ROOT / "configs" / "model_registry.yaml",
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--answer", required=True)
    parser.add_argument("--policy", default="fp16")
    parser.add_argument("--protected-layer-ids", default="")
    parser.add_argument("--default-key-bits", type=int, default=4)
    parser.add_argument("--default-value-bits", type=int, default=2)
    parser.add_argument("--protected-key-bits", type=int, default=6)
    parser.add_argument("--residual-window", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def resolve_model_path(args: argparse.Namespace) -> tuple[str, str]:
    registry = yaml.safe_load(args.model_registry.read_text(encoding="utf-8"))
    models = registry.get("models", {})
    if args.model_key not in models:
        raise SystemExit(f"Unknown --model-key: {args.model_key}")
    entry = models[args.model_key]
    return entry.get("display_name", args.model_key), str(
        args.model_root / entry["local_dir"]
    )


def parse_layer_ids(text: str) -> list[int]:
    if not text:
        return []
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def make_policy(args: argparse.Namespace) -> dict:
    name = args.policy
    if name == "fp16":
        return {
            "name": "fp16",
            "key_bits": 16,
            "value_bits": 16,
            "residual_window": 0,
        }
    if name == "uniform_k4_v2_rw128":
        return {
            "name": name,
            "key_bits": 4,
            "value_bits": 2,
            "residual_window": 128,
            "protected_layer_ids": [],
            "protected_key_bits": "",
        }
    if name == "tbgmp_topk":
        protected_layer_ids = parse_layer_ids(args.protected_layer_ids)
        if not protected_layer_ids:
            raise SystemExit("--protected-layer-ids is required for tbgmp_topk")
        return {
            "name": name,
            "key_bits": args.default_key_bits,
            "value_bits": args.default_value_bits,
            "default_key_bits": args.default_key_bits,
            "default_value_bits": args.default_value_bits,
            "protected_layer_ids": protected_layer_ids,
            "protected_key_bits": args.protected_key_bits,
            "residual_window": args.residual_window,
        }
    raise SystemExit(
        "Unsupported --policy. Use fp16, uniform_k4_v2_rw128, or tbgmp_topk."
    )


def main() -> None:
    args = parse_args()
    model_id, model_path = resolve_model_path(args)
    if args.backend == "turboquant":
        backend = TurboQuantBackend(
            turboquant_root=str(args.turboquant_root) if args.turboquant_root else None
        )
    else:
        backend = load_backend(args.backend)

    if hasattr(backend, "check_available"):
        status = backend.check_available()
        print(json.dumps(status, indent=2, sort_keys=True))
        if not status.get("ready_for_tbgmp_generation", True):
            raise SystemExit(
                "Backend smoke test failed before generation. See "
                "docs/backend_integration.md and docs/turboquant_api_findings.md."
            )

    request = GenerationRequest(
        model_path=model_path,
        prompt=args.prompt,
        answer=args.answer,
        policy=make_policy(args),
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    try:
        result = backend.generate(request)
    except NotImplementedError as exc:
        raise SystemExit(str(exc)) from None

    response = str(result.response)
    row = {
        "model": model_id,
        "case_id": "backend_smoke_0",
        "domain": "smoke",
        "context_length": len(args.prompt.split()),
        "depth": 0,
        "seed": args.seed,
        "policy": args.policy,
        "policy_type": "fp16" if args.policy == "fp16" else "quantized",
        "key_bits": request.policy["key_bits"],
        "value_bits": request.policy["value_bits"],
        "protected_key_bits": request.policy.get("protected_key_bits", ""),
        "protected_layers": request.policy.get("protected_layer_ids", []),
        "residual_window": request.policy["residual_window"],
        "answer": args.answer,
        "response": response,
        "found": found_answer(response, args.answer),
        "status": result.status,
        **result.metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote backend smoke JSONL -> {args.output}")


if __name__ == "__main__":
    main()
