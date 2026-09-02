#!/usr/bin/env python3
"""Resume-safe TurboQuant evaluation for frozen-ranking RULER task transfer."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import os
import random
import re
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("TBGMP_RULER_ROOT", Path.cwd())).expanduser().resolve()
MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", ROOT / "models")).expanduser().resolve()
CONFIG_PATH = Path(os.environ.get("TBGMP_RULER_CONFIG", REPO_ROOT / "configs" / "extensions" / "ruler.json"))
POLICY_PATH = Path(os.environ.get("TBGMP_RULER_POLICY", REPO_ROOT / "configs" / "extensions" / "ruler_policy.json"))
MANIFEST_PATH = Path(os.environ.get("TBGMP_RULER_CASES", ROOT / "manifests" / "RULER_FORMAL_CASE_MANIFEST.json"))
CONDITIONAL_DIR = ROOT / "conditionals"
OUTPUTS = ROOT / "outputs"
TASKS = ("niah_multikey_1", "vt", "fwe")
CONTROL_BUDGETS = (1, 4, 8, 12)
RANDOM_SEEDS = (0, 1, 2)
MODEL_SPECS = {
    "qwen3_4b": {
        "backend_model_id": "qwen3_4b",
        "model_name": "Qwen3-4B-Instruct-2507",
        "model_path": MODEL_ROOT / "Qwen3-4B-Instruct-2507",
        "backend": Path(os.environ.get(
            "TBGMP_QWEN3_BACKEND",
            ROOT / "runtime" / "backends" / "run_tbgmp_single_model.py",
        )),
    },
    "qwen25_3b": {
        "backend_model_id": "qwen25_3b_instruct",
        "model_name": "Qwen2.5-3B-Instruct",
        "model_path": MODEL_ROOT / "Qwen2.5-3B-Instruct",
        "backend": Path(os.environ.get(
            "TBGMP_QWEN25_BACKEND",
            ROOT / "runtime" / "backends" / "run_tbgmp_single_model.py",
        )),
    },
}
FIELDS = [
    "model", "model_name", "task", "category", "context_length", "sample_id",
    "ruler_sample_id", "generation_seed", "policy", "policy_group", "k", "random_seed",
    "protected_layers", "default_key_bits", "default_value_bits", "protected_key_bits",
    "residual_window", "prediction", "references", "official_soft_score", "full_credit_success",
    "valid_execution", "actual_prompt_tokens", "generated_tokens", "prompt_sha256", "runtime_s",
    "peak_gpu_gb", "kv_saving_percent", "oom", "error", "completed", "started_at",
    "finished_at", "slurm_job_id", "slurm_array_task_id", "notes",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, path)


def write_csv_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def import_backend(path: Path, model_id: str):
    spec = importlib.util.spec_from_file_location(f"ruler_backend_{model_id}", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_chat_prompt(tokenizer, question: str) -> str:
    messages = [{"role": "user", "content": question}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def official_ruler_score(prediction: str, references: list[str]) -> float:
    """Exact per-sample string_match_all semantics from NVIDIA RULER/NeMo-Skills."""
    if not references:
        raise ValueError("RULER sample has no references")
    return sum(1.0 if str(ref).lower() in prediction.lower() else 0.0 for ref in references) / len(references)


def postprocess_official(prediction: str) -> str:
    prediction = prediction.strip()
    return re.compile(r"[\x00-\x1f]").sub("\n", prediction).strip()


def make_policy(policy_cfg: dict, model_id: str, group: str, k: int | None = None, seed: int | None = None) -> dict:
    cfg = policy_cfg["models"][model_id]
    base_k = int(cfg["aggressive_key_bits"])
    if group == "fp16" or group == "official_reference_fp16":
        return {
            "policy_name": "FP16 baseline" if group == "fp16" else "Official RULER HF FP16",
            "policy_group": group,
            "fp16": True,
            "default_key_bits": "", "default_value_bits": "", "protected_key_bits": "",
            "protected_key_layers": [], "layer_key_bits": {}, "layer_value_bits": {},
            "residual_window": "",
        }
    if group == "aggressive":
        layers = []
        name = f"Uniform K{base_k}/V2 rw128"
    elif group == "top":
        layers = [int(layer) for layer in cfg["ranking"][: int(k)]]
        name = f"T-BGMP Top{k}"
    elif group == "bottom":
        layers = [int(layer) for layer in cfg["bottom_layers"][str(k)]]
        name = f"Bottom{k}"
    elif group == "random":
        layers = [int(layer) for layer in cfg["random_layers"][str(seed)][str(k)]]
        name = f"Random{k} seed{seed}"
    else:
        raise ValueError(group)
    return {
        "policy_name": name,
        "policy_group": group,
        "fp16": False,
        "default_key_bits": base_k,
        "default_value_bits": 2,
        "protected_key_bits": 6 if layers else "",
        "protected_value_bits": 2,
        "protected_key_layers": layers,
        "protected_value_layers": [],
        "layer_key_bits": {int(layer): 6 for layer in layers},
        "layer_value_bits": {},
        "residual_window": 128,
    }


def checkpoint_path(stage: str, model_id: str, sample_id: str, policy_name: str) -> Path:
    key = hashlib.sha256(f"{model_id}|{sample_id}|{policy_name}".encode("utf-8")).hexdigest()
    return OUTPUTS / stage / "checkpoints" / model_id / f"{key}.json"


def base_row(sample: dict, policy: dict, k=None, seed=None) -> dict:
    return {
        "model": sample["model"],
        "model_name": sample["model_name"],
        "task": sample["task"],
        "category": sample["category"],
        "context_length": sample["context_length"],
        "sample_id": sample["sample_id"],
        "ruler_sample_id": sample["ruler_sample_id"],
        "generation_seed": sample["generation_seed"],
        "policy": policy["policy_name"],
        "policy_group": policy["policy_group"],
        "k": "" if k is None else k,
        "random_seed": "" if seed is None else seed,
        "protected_layers": policy.get("protected_key_layers", []),
        "default_key_bits": policy.get("default_key_bits", ""),
        "default_value_bits": policy.get("default_value_bits", ""),
        "protected_key_bits": policy.get("protected_key_bits", ""),
        "residual_window": policy.get("residual_window", ""),
        "prediction": "",
        "references": sample["reference_outputs"],
        "official_soft_score": "",
        "full_credit_success": False,
        "valid_execution": False,
        "actual_prompt_tokens": "",
        "generated_tokens": "",
        "prompt_sha256": "",
        "runtime_s": "",
        "peak_gpu_gb": "",
        "kv_saving_percent": "",
        "oom": False,
        "error": "",
        "completed": False,
        "started_at": now_iso(),
        "finished_at": "",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        "notes": "Frozen original T-BGMP ranking; no RULER profiling or tuning.",
    }


def run_one(hpc, model, tokenizer, dims, sample: dict, policy: dict, official_reference_path=False, k=None, seed=None) -> dict:
    row = base_row(sample, policy, k=k, seed=seed)
    try:
        question = sample["input"] + sample.get("answer_prefix", "")
        prompt = safe_chat_prompt(tokenizer, question)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
        prompt_tokens = int(input_ids.shape[1])
        if prompt_tokens > int(sample["context_length"]):
            raise RuntimeError(f"Prompt has {prompt_tokens} tokens, exceeds target {sample['context_length']}")
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": int(sample["max_new_tokens"]),
            "do_sample": False,
            "use_cache": True,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if not official_reference_path:
            kwargs["past_key_values"] = hpc.cache_from_policy(policy, dims)
        start = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**kwargs)
        torch.cuda.synchronize()
        runtime = time.perf_counter() - start
        new_tokens = generated[0][prompt_tokens:]
        prediction = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        score = official_ruler_score(prediction, [str(ref) for ref in sample["reference_outputs"]])
        fp16_bytes, compressed_bytes, _ = hpc.estimate_kv_bytes(prompt_tokens, dims, policy)
        row.update({
            "prediction": postprocess_official(prediction),
            "official_soft_score": score,
            "full_credit_success": bool(abs(score - 1.0) < 1e-12),
            "valid_execution": True,
            "actual_prompt_tokens": prompt_tokens,
            "generated_tokens": int(new_tokens.numel()),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "runtime_s": round(runtime, 4),
            "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 4),
            "kv_saving_percent": round((1.0 - compressed_bytes / fp16_bytes) * 100.0, 4) if fp16_bytes else 0.0,
        })
    except Exception as exc:
        message = f"{type(exc).__name__}: {str(exc).replace(chr(10), ' ').strip()}"
        row["error"] = message
        row["oom"] = "out of memory" in message.lower() or ("cuda" in message.lower() and "memory" in message.lower())
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    finally:
        row["completed"] = True
        row["finished_at"] = now_iso()
    return row


def samples_for_args(args, manifest: list[dict]) -> list[dict]:
    rows = [row for row in manifest if row["model"] == args.model]
    if args.stage == "smoke":
        selected = []
        for task in TASKS:
            matches = [row for row in rows if row["task"] == task and int(row["context_length"]) == 4096]
            selected.append(sorted(matches, key=lambda row: int(row["ruler_sample_id"]))[0])
        return selected
    if args.stage == "screening":
        return [row for row in rows if row["task"] == args.task and int(row["context_length"]) == args.length]
    conditional_path = CONDITIONAL_DIR / ("Qwen3_RULER_CONDITIONAL.json" if args.model == "qwen3_4b" else "Qwen25_RULER_CONDITIONAL.json")
    if not conditional_path.exists():
        raise RuntimeError(f"Conditional manifest not found: {conditional_path}")
    conditional = load_json(conditional_path)
    rows = conditional.get("cases", [])
    if not rows:
        marker = OUTPUTS / args.stage / args.model / "NO_CONDITIONAL_CASES"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("NO_CONDITIONAL_CASES\n", encoding="utf-8")
        return []
    return [row for index, row in enumerate(rows) if index % args.total_chunks == args.chunk_index]


def policies_for_stage(stage: str, policy_cfg: dict, model_id: str):
    if stage == "smoke":
        return [
            (make_policy(policy_cfg, model_id, "official_reference_fp16"), None, None, True),
            (make_policy(policy_cfg, model_id, "fp16"), None, None, False),
            (make_policy(policy_cfg, model_id, "aggressive"), None, None, False),
            (make_policy(policy_cfg, model_id, "top", 1), 1, None, False),
        ]
    if stage == "screening":
        return [
            (make_policy(policy_cfg, model_id, "fp16"), None, None, False),
            (make_policy(policy_cfg, model_id, "aggressive"), None, None, False),
        ]
    if stage == "top":
        return [(make_policy(policy_cfg, model_id, "top", k), k, None, False) for k in range(1, 13)]
    if stage == "bottom":
        return [(make_policy(policy_cfg, model_id, "bottom", k), k, None, False) for k in CONTROL_BUDGETS]
    if stage == "random":
        return [
            (make_policy(policy_cfg, model_id, "random", k, seed), k, seed, False)
            for k in CONTROL_BUDGETS for seed in RANDOM_SEEDS
        ]
    raise ValueError(stage)


def verify_no_duplicates(rows: list[dict]) -> None:
    keys = [(row["model"], row["sample_id"], row["policy"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate model/sample/policy rows detected in shard output")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["smoke", "screening", "top", "bottom", "random"], required=True)
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--length", type=int, choices=[4096, 8192])
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--total-chunks", type=int, default=1)
    args = parser.parse_args()
    if args.stage == "screening" and (args.task is None or args.length is None):
        parser.error("screening requires --task and --length")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run inference stages through SLURM")
    config = load_json(CONFIG_PATH)
    if config["scientific_isolation"]["ruler_specific_profiling"]:
        raise RuntimeError("Formal config unexpectedly enables RULER-specific profiling")
    policy_cfg = load_json(POLICY_PATH)
    manifest = load_json(MANIFEST_PATH)
    samples = samples_for_args(args, manifest)
    spec = MODEL_SPECS[args.model]
    hpc = import_backend(spec["backend"], args.model)
    backend_id = spec["backend_model_id"]
    if backend_id not in hpc.MODEL_REGISTRY:
        hpc.MODEL_REGISTRY[backend_id] = {
            "model_name": spec["model_name"], "default_path": str(spec["model_path"]),
            "preferred_dtype": "bfloat16", "prompt_mode": "chat_template",
            "decoding_mode": "deterministic", "max_new_tokens": 128,
        }
    tokenizer, model = hpc.load_model(str(spec["model_path"]), backend_id)
    dims = hpc.get_dims(model)
    if int(dims[0]) != 36:
        raise RuntimeError(f"Expected 36 layers for {args.model}, got {dims[0]}")
    results = []
    plans = policies_for_stage(args.stage, policy_cfg, args.model)
    for sample in samples:
        for policy, k, seed, official_reference_path in plans:
            checkpoint = checkpoint_path(args.stage, args.model, sample["sample_id"], policy["policy_name"])
            if checkpoint.exists():
                row = load_json(checkpoint)
                if not row.get("completed"):
                    raise RuntimeError(f"Incomplete checkpoint exists: {checkpoint}")
                results.append(row)
                continue
            print(f"[{args.stage}] {sample['sample_id']} {policy['policy_name']}", flush=True)
            row = run_one(hpc, model, tokenizer, dims, sample, policy, official_reference_path, k=k, seed=seed)
            write_json_atomic(checkpoint, row)
            results.append(row)
    verify_no_duplicates(results)
    shard_name = (
        f"{args.model}_{args.task}_{args.length}.csv" if args.stage == "screening"
        else f"{args.model}_chunk{args.chunk_index:02d}_of_{args.total_chunks:02d}.csv"
    )
    if args.stage == "smoke":
        shard_name = f"{args.model}_smoke.csv"
    shard = OUTPUTS / args.stage / "shards" / shard_name
    write_csv_atomic(shard, sorted(results, key=lambda row: (row["sample_id"], row["policy"])))
    if args.stage == "smoke":
        by_sample = {}
        for row in results:
            by_sample.setdefault(row["sample_id"], {})[row["policy_group"]] = row
        alignment = []
        for sample_id, group in sorted(by_sample.items()):
            official = group["official_reference_fp16"]
            wrapper = group["fp16"]
            alignment.append({
                "sample_id": sample_id,
                "prompt_hash_match": official["prompt_sha256"] == wrapper["prompt_sha256"],
                "score_match": official["official_soft_score"] == wrapper["official_soft_score"],
                "full_credit_match": official["full_credit_success"] == wrapper["full_credit_success"],
                "both_valid": official["valid_execution"] and wrapper["valid_execution"],
            })
        passed = len(alignment) == 3 and all(
            row["prompt_hash_match"] and row["score_match"] and row["full_credit_match"] and row["both_valid"]
            for row in alignment
        ) and all(row["valid_execution"] for row in results)
        write_json_atomic(OUTPUTS / "smoke" / f"{args.model}_alignment.json", {
            "model": args.model, "pass": passed, "alignment": alignment,
            "official_reference_source": str(ROOT / "RULER_generator" / "scripts" / "pred" / "model_wrappers.py"),
            "official_scorer_source": str(ROOT / "Skills" / "nemo_skills" / "evaluation" / "evaluator" / "ruler.py"),
        })
        if not passed:
            raise RuntimeError(f"Smoke alignment failed for {args.model}")
    print(json.dumps({"stage": args.stage, "model": args.model, "rows": len(results), "shard": str(shard)}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        crash = ROOT / "logs" / f"crash_{os.environ.get('SLURM_JOB_ID', 'local')}_{os.environ.get('SLURM_ARRAY_TASK_ID', 'na')}.log"
        crash.parent.mkdir(parents=True, exist_ok=True)
        crash.write_text(traceback.format_exc(), encoding="utf-8")
        raise
