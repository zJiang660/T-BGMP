#!/usr/bin/env python3
"""Resume-safe operational evaluation for a frozen model-level protection policy."""

from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import hashlib
import importlib.util
import json
import os
import traceback
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("TBGMP_FROZEN_ROOT", Path.cwd())).expanduser().resolve()
CONFIG_PATH = Path(
    os.environ.get(
        "TBGMP_FROZEN_CONFIG",
        REPO_ROOT / "configs" / "extensions" / "frozen_top3.json",
    )
)
CASE_MANIFEST_PATH = Path(
    os.environ.get(
        "TBGMP_FROZEN_CASES",
        REPO_ROOT / "data" / "extension_cases" / "frozen_top3_cases.json",
    )
)
TASK_MANIFEST_PATH = Path(
    os.environ.get(
        "TBGMP_FROZEN_TASKS",
        REPO_ROOT / "configs" / "extensions" / "frozen_top3_tasks.csv",
    )
)
EXTRA_FIELDS = ["case_id", "gpu_name", "quantizer_active", "slurm_job_id", "slurm_array_task_id"]


def load_json(path: Path):
    value = json.loads(path.read_text(encoding="utf-8"))

    def expand(item):
        if isinstance(item, str):
            return os.path.expandvars(item)
        if isinstance(item, list):
            return [expand(part) for part in item]
        if isinstance(item, dict):
            return {key: expand(part) for key, part in item.items()}
        return item

    return expand(value)


def write_json_atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, path)


def write_csv_atomic(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def import_backend(path: Path):
    spec = importlib.util.spec_from_file_location("frozen_topk_backend", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_policy(model_id: str, model_cfg: dict, policy_name: str, random_seed: str = "") -> dict:
    if policy_name == "frozen_top":
        layers = model_cfg["frozen_top_layers"]
        label = f"Frozen Top{model_cfg['k_star']}"
        kind = "frozen_top"
    elif policy_name == "frozen_bottom":
        layers = model_cfg["frozen_bottom_layers"]
        label = f"Frozen Bottom{model_cfg['k_star']}"
        kind = "frozen_bottom"
    elif policy_name == "frozen_random":
        layers = model_cfg["frozen_random_layers"][str(random_seed)]
        label = f"Frozen Random{model_cfg['k_star']} seed{random_seed}"
        kind = f"frozen_random_seed{random_seed}"
    else:
        raise ValueError(policy_name)
    layers = [int(layer) for layer in layers]
    if len(layers) != int(model_cfg["k_star"]):
        raise RuntimeError(f"Protection budget mismatch for {model_id}/{label}")
    return {
        "policy_name": label,
        "policy_type": kind,
        "fp16": False,
        "default_key_bits": int(model_cfg["aggressive_key_bits"]),
        "default_value_bits": int(model_cfg["aggressive_value_bits"]),
        "protected_key_bits": int(model_cfg["protected_key_bits"]),
        "protected_value_bits": int(model_cfg["protected_value_bits"]),
        "protected_key_layers": layers,
        "protected_value_layers": [],
        "layer_key_bits": {int(layer): int(model_cfg["protected_key_bits"]) for layer in layers},
        "layer_value_bits": {},
        "residual_window": int(model_cfg["residual_window"]),
        "notes": "Frozen model-level policy selected from calibration only; one inference per unseen request.",
    }


def gpu_slug(gpu_name: str) -> str:
    return "4090" if "4090" in gpu_name.upper() else "a800" if "A800" in gpu_name.upper() else "unknown"


def checkpoint_path(model_id: str, policy: dict, case_id: str, debug: bool, gpu_name: str) -> Path:
    namespace = "debug" if debug else "outputs"
    key = hashlib.sha256(f"{model_id}|{policy['policy_name']}|{case_id}".encode("utf-8")).hexdigest()
    if debug:
        return ROOT / namespace / gpu_slug(gpu_name) / model_id / "checkpoints" / f"{key}.json"
    return ROOT / namespace / model_id / "checkpoints" / f"{key}.json"


def run_policy(hpc, model, tokenizer, dims, model_id: str, model_cfg: dict, cases: list[dict], policy: dict, debug: bool) -> list[dict]:
    rows = []
    gpu_name = torch.cuda.get_device_name(0)
    for case in cases:
        checkpoint = checkpoint_path(model_id, policy, case["case_id"], debug, gpu_name)
        if checkpoint.exists():
            row = load_json(checkpoint)
            if not row.get("completed"):
                raise RuntimeError(f"Incomplete checkpoint exists: {checkpoint}")
            rows.append(row)
            continue
        print(f"[{model_id}] {policy['policy_name']} {case['case_id']}", flush=True)
        row = hpc.run_generation(
            model, tokenizer, dims,
            model_cfg["backend_model_id"], model_cfg["model_name"],
            case["domain"], int(case["context_length"]), int(case["needle_depth"]),
            int(case["seed"]), case["needle_string"], policy,
            "frozen_operational_debug" if debug else "frozen_operational_formal",
            model_cfg["case_type"],
        )
        row.update({
            "case_id": case["case_id"],
            "gpu_name": gpu_name,
            "quantizer_active": True,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        })
        write_json_atomic(checkpoint, row)
        rows.append(row)
    return rows


def read_task(task_id: int) -> dict:
    with TASK_MANIFEST_PATH.open(newline="", encoding="utf-8") as handle:
        tasks = {int(row["task_id"]): row for row in csv.DictReader(handle)}
    if task_id not in tasks:
        raise KeyError(f"Unknown formal task id {task_id}")
    return tasks[task_id]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--dry-run-model", choices=("qwen3_4b", "qwen25_3b"))
    args = parser.parse_args()
    if (args.task_id is None) == (args.dry_run_model is None):
        parser.error("Specify exactly one of --task-id or --dry-run-model")

    config = load_json(CONFIG_PATH)
    if config["scientific_isolation"]["evaluation_used_to_select_k"]:
        raise RuntimeError("Frozen config violates evaluation isolation")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    allowed_gpu_tokens = [str(token).upper() for token in config["hardware"].get("allowed_gpu_tokens", ["A800"])]
    expected_gpu_token = os.environ.get("EXPECTED_GPU_TOKEN", "").upper()
    if not any(token in gpu_name.upper() for token in allowed_gpu_tokens):
        raise RuntimeError(f"Allowed GPU {allowed_gpu_tokens} required, got {gpu_name or 'NO CUDA GPU'}")
    if expected_gpu_token and expected_gpu_token not in gpu_name.upper():
        raise RuntimeError(f"Expected GPU token {expected_gpu_token}, got {gpu_name}")

    debug = args.dry_run_model is not None
    if debug:
        model_id = args.dry_run_model
        task = None
    else:
        task = read_task(args.task_id)
        model_id = task["model_id"]
        lock_path = ROOT / "outputs" / "task_locks" / f"task_{args.task_id}.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        task_lock = lock_path.open("w", encoding="utf-8")
        fcntl.flock(task_lock.fileno(), fcntl.LOCK_EX)
        task_lock.write(f"job={os.environ.get('SLURM_JOB_ID', '')} gpu={gpu_name}\n")
        task_lock.flush()
    model_cfg = config["models"][model_id]
    hpc = import_backend(Path(config["backend_path"]))
    backend_id = model_cfg["backend_model_id"]
    hpc.MODEL_REGISTRY.setdefault(backend_id, {})
    hpc.MODEL_REGISTRY[backend_id].update({
        "model_name": model_cfg["model_name"],
        "default_path": model_cfg["model_path"],
        "preferred_dtype": "bfloat16",
        "prompt_mode": "chat_template",
        "decoding_mode": "deterministic",
        "max_new_tokens": 24,
    })
    context_root = Path(config["domain_context_root"])
    hpc.DOMAIN_FILES = {
        domain: str(context_root / f"{domain}.txt")
        for domain in ("math", "literature", "science", "code")
    }
    tokenizer, model = hpc.load_model(model_cfg["model_path"], backend_id)
    dims = hpc.get_dims(model)
    if int(dims[0]) != int(model_cfg["num_hidden_layers"]):
        raise RuntimeError(f"Layer count mismatch: expected {model_cfg['num_hidden_layers']}, got {dims[0]}")
    cases = [row for row in load_json(CASE_MANIFEST_PATH) if row["model_id"] == model_id]
    if len(cases) != 36 or {int(row["seed"]) for row in cases} != {int(model_cfg["evaluation_seed"])}:
        raise RuntimeError(f"Invalid evaluation case manifest for {model_id}")
    if debug:
        cases = cases[:1]
        plans = [
            make_policy(model_id, model_cfg, "frozen_top"),
            make_policy(model_id, model_cfg, "frozen_bottom"),
            make_policy(model_id, model_cfg, "frozen_random", "0"),
        ]
    else:
        plans = [make_policy(model_id, model_cfg, task["policy"], task["random_seed"])]

    all_rows = []
    for policy in plans:
        cache = hpc.cache_from_policy(policy, dims)
        if cache is None or cache.__class__.__name__ != "TBGMPAdaptiveCache":
            raise RuntimeError("TurboQuant adaptive cache was not activated")
        all_rows.extend(run_policy(hpc, model, tokenizer, dims, model_id, model_cfg, cases, policy, debug))

    fields = list(dict.fromkeys(list(hpc.FIELDS) + EXTRA_FIELDS))
    if debug:
        slug = gpu_slug(gpu_name)
        shard = ROOT / "debug" / slug / model_id / "dry_run.csv"
        write_csv_atomic(shard, all_rows, fields)
        passed = (
            len(all_rows) == 3
            and all(row.get("completed") for row in all_rows)
            and all(not row.get("error") for row in all_rows)
            and all(slug.upper() in str(row.get("gpu_name", "")).upper() for row in all_rows)
            and all(row.get("quantizer_active") for row in all_rows)
            and all(int(row.get("default_value_bits")) == int(model_cfg["aggressive_value_bits"]) for row in all_rows)
            and all(len(json.loads(row["protected_key_layers"])) == int(model_cfg["k_star"]) for row in all_rows)
        )
        alignment = {
            "model_id": model_id,
            "pass": passed,
            "job_id": os.environ.get("SLURM_JOB_ID", ""),
            "gpu": gpu_name,
            "rows": len(all_rows),
            "expected_k_star": model_cfg["k_star"],
            "policies": [{"name": row["policy_name"], "layers": json.loads(row["protected_key_layers"]), "error": row["error"]} for row in all_rows],
            "turboquant_active": all(row.get("quantizer_active") for row in all_rows),
            "exact_match_check_executed": all(isinstance(row.get("found"), bool) for row in all_rows),
        }
        write_json_atomic(ROOT / "debug" / f"{model_id}_{slug}_alignment.json", alignment)
        if not passed:
            raise RuntimeError(f"Dry-run failed for {model_id}")
    else:
        policy_slug = plans[0]["policy_type"]
        shard = ROOT / "outputs" / model_id / f"{policy_slug}.csv"
        write_csv_atomic(shard, sorted(all_rows, key=lambda row: row["case_id"]), fields)
        if len(all_rows) != 36:
            raise RuntimeError(f"Expected 36 rows, got {len(all_rows)}")
    print(json.dumps({"status": "PASS", "model": model_id, "debug": debug, "rows": len(all_rows), "output": str(shard), "gpu": gpu_name}, indent=2))
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        crash = ROOT / "logs" / f"crash_{os.environ.get('SLURM_JOB_ID', 'local')}_{os.environ.get('SLURM_ARRAY_TASK_ID', 'na')}.log"
        crash.parent.mkdir(parents=True, exist_ok=True)
        crash.write_text(traceback.format_exc(), encoding="utf-8")
        raise
