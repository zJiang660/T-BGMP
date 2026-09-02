#!/usr/bin/env python3
"""Validate a GPU environment, TurboQuant checkout, and model fingerprints."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _result(check: str, expected: str, actual: str, passed: bool) -> dict[str, Any]:
    return {
        "check": check,
        "expected": expected,
        "actual": actual,
        "passed": passed,
    }


def check_python(expected: str) -> dict[str, Any]:
    actual = ".".join(str(part) for part in sys.version_info[:3])
    return _result("python", expected, actual, actual == expected)


def check_packages(packages: dict[str, str]) -> list[dict[str, Any]]:
    checks = []
    for name, expected in packages.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = "MISSING"
        checks.append(_result(f"package:{name}", expected, actual, actual == expected))
    return checks


def check_cuda(expected: str) -> dict[str, Any]:
    if expected == "site-provided":
        return _result("cuda_runtime", expected, "site-provided", True)
    try:
        import torch

        actual = str(torch.version.cuda)
    except (ImportError, AttributeError):
        actual = "MISSING"
    return _result("cuda_runtime", expected, actual, actual == expected)


def check_turboquant(root: Path, expected_commit: str) -> dict[str, Any]:
    try:
        actual = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        actual = "UNAVAILABLE"
    return _result("turboquant_commit", expected_commit, actual, actual == expected_commit)


def check_model(model_root: Path, key: str, spec: dict[str, Any]) -> list[dict[str, Any]]:
    model_dir = model_root / spec["local_dir"]
    checks = []
    for field, filename in (
        ("config_sha256", "config.json"),
        ("weight_index_sha256", "model.safetensors.index.json"),
    ):
        expected = spec.get(field)
        if not expected:
            continue
        path = model_dir / filename
        actual = _sha256(path) if path.is_file() else "MISSING"
        checks.append(_result(f"model:{key}:{filename}", expected, actual, actual == expected))

    weight_file = spec.get("weight_file")
    if weight_file:
        exists = (model_dir / weight_file).is_file()
        checks.append(_result(f"model:{key}:weight_file", weight_file, weight_file if exists else "MISSING", exists))
    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=Path("configs/runtime_lock.yaml"))
    parser.add_argument("--profile", required=True)
    parser.add_argument("--turboquant-root", type=Path)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--model-key", action="append", default=[])
    parser.add_argument("--skip-python", action="store_true")
    parser.add_argument("--skip-packages", action="store_true")
    parser.add_argument("--skip-cuda", action="store_true")
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lock = yaml.safe_load(args.lock.read_text(encoding="utf-8"))
    if args.profile not in lock["profiles"]:
        raise SystemExit(f"unknown profile: {args.profile}")
    profile = lock["profiles"][args.profile]
    checks: list[dict[str, Any]] = []

    if not args.skip_python:
        checks.append(check_python(str(profile["python"])))
    if not args.skip_packages:
        checks.extend(check_packages(profile["packages"]))
    if not args.skip_cuda:
        checks.append(check_cuda(str(profile["cuda_runtime"])))

    if args.turboquant_root:
        expected = lock["external_runtime"]["turboquant"]["upstream_commit"]
        checks.append(check_turboquant(args.turboquant_root, expected))
    if args.model_key and not args.model_root:
        raise SystemExit("--model-root is required when --model-key is used")
    for key in args.model_key:
        if key not in lock["models"]:
            raise SystemExit(f"unknown model key: {key}")
        checks.extend(check_model(args.model_root, key, lock["models"][key]))

    report = {"profile": args.profile, "passed": all(item["passed"] for item in checks), "checks": checks}
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

