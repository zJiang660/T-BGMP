from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping


MODEL_IDENTITY_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "model.safetensors.index.json",
)
PATH_ARGUMENTS = {
    "--cases": "<CASE_FILE>",
    "--config": "<CONFIG_FILE>",
    "--model-path": "<MODEL_PATH>",
    "--model-root": "<MODEL_ROOT>",
    "--model-registry": "<MODEL_REGISTRY>",
    "--output": "<OUTPUT_PATH>",
    "--output-dir": "<OUTPUT_DIR>",
    "--policies": "<POLICY_FILE>",
    "--prompt-template": "<PROMPT_TEMPLATE>",
    "--risk-ranking": "<RISK_RANKING>",
    "--turboquant-root": "<TURBOQUANT_ROOT>",
}
PACKAGE_NAMES = (
    "accelerate",
    "numpy",
    "pandas",
    "PyYAML",
    "safetensors",
    "torch",
    "transformers",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_fingerprint(label: str, path: Path | None) -> dict | None:
    if path is None or not path.is_file():
        return None
    return {
        "label": label,
        "filename": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def fingerprint_inputs(paths: Mapping[str, Path | None]) -> list[dict]:
    fingerprints = []
    for label, path in sorted(paths.items()):
        fingerprint = file_fingerprint(label, path)
        if fingerprint is not None:
            fingerprints.append(fingerprint)
    return fingerprints


def repository_state(root: Path) -> dict:
    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            text=True,
            capture_output=True,
            check=True,
        )
        return completed.stdout.strip()

    try:
        status = git("status", "--porcelain", "--untracked-files=no")
        return {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "tracked_files_dirty": bool(status),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "branch": "unknown", "tracked_files_dirty": None}


def runtime_environment() -> dict:
    packages = {}
    for name in PACKAGE_NAMES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    cuda = {"available": False, "runtime": None, "cudnn": None}
    gpu = {"count": 0, "names": []}
    try:
        import torch

        cuda = {
            "available": bool(torch.cuda.is_available()),
            "runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        }
        if cuda["available"]:
            gpu = {
                "count": torch.cuda.device_count(),
                "names": [
                    torch.cuda.get_device_name(index)
                    for index in range(torch.cuda.device_count())
                ],
            }
    except (ImportError, RuntimeError):
        pass

    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.system(),
        "platform_release": platform.release(),
        "packages": packages,
        "cuda": cuda,
        "gpu": gpu,
    }


def scheduler_context(environ: Mapping[str, str] | None = None) -> dict:
    source = os.environ if environ is None else environ
    names = {
        "job_id": "SLURM_JOB_ID",
        "array_job_id": "SLURM_ARRAY_JOB_ID",
        "array_task_id": "SLURM_ARRAY_TASK_ID",
        "partition": "SLURM_JOB_PARTITION",
    }
    return {
        label: source[variable]
        for label, variable in names.items()
        if source.get(variable)
    }


def model_identity(model_id: str, model_path: Path) -> dict:
    identity = {
        "model_id": model_id,
        "directory_name": model_path.name,
        "exists": model_path.is_dir(),
        "snapshot_revision": None,
        "identity_files": [],
    }
    if not model_path.is_dir():
        return identity

    for filename in MODEL_IDENTITY_FILES:
        fingerprint = file_fingerprint(filename, model_path / filename)
        if fingerprint is not None:
            identity["identity_files"].append(fingerprint)

    config_path = model_path / "config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            revision = config.get("_commit_hash")
            if isinstance(revision, str) and revision:
                identity["snapshot_revision"] = revision
        except (OSError, json.JSONDecodeError):
            pass
    if identity["snapshot_revision"] is None:
        for part in reversed(model_path.parts):
            if re.fullmatch(r"[0-9a-fA-F]{40,64}", part):
                identity["snapshot_revision"] = part
                break
    return identity


def sanitized_argv(argv: Iterable[str] | None = None) -> list[str]:
    values = list(sys.argv if argv is None else argv)
    if values:
        values[0] = Path(values[0]).name
    sanitized: list[str] = []
    redact_next: str | None = None
    for value in values:
        if redact_next is not None:
            sanitized.append(redact_next)
            redact_next = None
            continue
        if value in PATH_ARGUMENTS:
            sanitized.append(value)
            redact_next = PATH_ARGUMENTS[value]
            continue
        matched = False
        for option, replacement in PATH_ARGUMENTS.items():
            prefix = f"{option}="
            if value.startswith(prefix):
                sanitized.append(f"{option}={replacement}")
                matched = True
                break
        if not matched:
            sanitized.append(value)
    return sanitized


def new_invocation(argv: Iterable[str] | None = None) -> dict:
    return {
        "invocation_id": uuid.uuid4().hex,
        "started_at_utc": utc_now(),
        "finished_at_utc": None,
        "status": "running",
        "argv": sanitized_argv(argv),
        "scheduler": scheduler_context(),
    }


def summarize_numeric(values: Iterable[object]) -> dict:
    parsed: list[int] = []
    for value in values:
        if value is None or value == "":
            continue
        try:
            parsed.append(int(float(value)))
        except (TypeError, ValueError):
            continue
    if not parsed:
        return {"count": 0, "min": None, "max": None, "unique": []}
    return {
        "count": len(parsed),
        "min": min(parsed),
        "max": max(parsed),
        "unique": sorted(set(parsed)),
    }
