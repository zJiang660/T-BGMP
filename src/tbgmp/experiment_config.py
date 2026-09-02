from __future__ import annotations

from pathlib import Path
from typing import Any


DEFAULT_DISCOVERY_POLICIES = [
    "fp16",
    "uniform_k2_v2_rw128",
    "uniform_k4_v2_rw128",
    "uniform_k6_v2_rw128",
    "uniform_k6_v4_rw128",
]


def configured_path(root: Path, value: Any) -> Path | None:
    if value in (None, "") or str(value).startswith("/path/to/"):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def validate_experiment_config(config: dict, policies: dict) -> dict:
    experiment = config.get("experiment", {})
    if not isinstance(experiment, dict):
        raise ValueError("experiment config must be a mapping")
    positive_lists = ("context_lengths",)
    for name in positive_lists:
        values = experiment.get(name, [])
        if not values or any(int(value) <= 0 for value in values):
            raise ValueError(f"experiment.{name} must contain positive values")
    depths = experiment.get("needle_depths", [])
    if not depths or any(not 0 <= int(value) <= 100 for value in depths):
        raise ValueError("experiment.needle_depths must be within [0, 100]")
    if not experiment.get("domains"):
        raise ValueError("experiment.domains must not be empty")
    if not experiment.get("seeds"):
        raise ValueError("experiment.seeds must not be empty")
    if int(experiment.get("maximum_topk", 0)) <= 0:
        raise ValueError("experiment.maximum_topk must be positive")

    discovery = list(
        experiment.get("discovery_policies", DEFAULT_DISCOVERY_POLICIES)
    )
    aggressive = list(
        experiment.get(
            "aggressive_policies",
            ["uniform_k4_v2_rw128", "uniform_k2_v2_rw128"],
        )
    )
    safe = list(
        experiment.get(
            "safe_policies", ["uniform_k6_v2_rw128", "uniform_k6_v4_rw128"]
        )
    )
    tbgmp = str(experiment.get("tbgmp_policy", "tbgmp_topk"))
    referenced = discovery + aggressive + safe + [tbgmp]
    missing = sorted(set(referenced) - set(policies))
    if missing:
        raise ValueError(f"configured policies are missing from policy file: {missing}")
    if "fp16" not in discovery:
        raise ValueError("experiment.discovery_policies must include fp16")
    if any(name not in discovery for name in aggressive + safe):
        raise ValueError("aggressive_policies and safe_policies must be discovery policies")
    return {
        "discovery_policies": discovery,
        "aggressive_policies": aggressive,
        "safe_policies": safe,
        "tbgmp_policy": tbgmp,
    }
