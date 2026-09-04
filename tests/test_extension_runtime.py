from __future__ import annotations

import ast
from pathlib import Path

from tbgmp.extension_runtime import (
    control_policy,
    estimate_kv_bytes,
    tbgmp_policy,
)


ROOT = Path(__file__).resolve().parents[1]


def test_extension_policy_preserves_aggressive_case_bits() -> None:
    ranking = [5, 2, 8, 1]
    k2 = tbgmp_policy("qwen", 2, ranking, "K2-sensitive")
    k4 = tbgmp_policy("qwen", 2, ranking, "K4-sensitive")
    assert k2["default_key_bits"] == 2
    assert k4["default_key_bits"] == 4
    assert k2["protected_key_layers"] == [5, 2]
    assert k4["protected_key_layers"] == [5, 2]


def test_same_budget_controls_are_deterministic() -> None:
    ranking = list(range(12))
    first = control_policy("qwen", "random", 4, ranking, "K2-sensitive", seed=7)
    second = control_policy("qwen", "random", 4, ranking, "K2-sensitive", seed=7)
    bottom = control_policy("qwen", "bottom", 4, ranking, "K2-sensitive")
    assert first["protected_key_layers"] == second["protected_key_layers"]
    assert bottom["protected_key_layers"] == [8, 9, 10, 11]


def test_kv_estimate_rewards_selective_protection() -> None:
    dims = (12, 12, 4, 64, 768)
    uniform = tbgmp_policy("qwen", 0, list(range(12)), "K2-sensitive")
    protected = tbgmp_policy("qwen", 2, list(range(12)), "K2-sensitive")
    fp16_uniform, compressed_uniform, _ = estimate_kv_bytes(4096, dims, uniform)
    fp16_protected, compressed_protected, _ = estimate_kv_bytes(4096, dims, protected)
    assert fp16_uniform == fp16_protected
    assert compressed_uniform < compressed_protected < fp16_protected


def test_extension_runners_do_not_import_external_helper_files() -> None:
    runners = [
        ROOT / "experiments" / "extensions" / "run_domain_heldout.py",
        ROOT / "experiments" / "extensions" / "run_frozen_top3.py",
        ROOT / "experiments" / "extensions" / "run_ruler_transfer.py",
    ]
    for runner in runners:
        source = runner.read_text(encoding="utf-8")
        ast.parse(source)
        assert "run_tbgmp_single_model_hpc" not in source
        assert "runtime/backends/run_tbgmp_single_model" not in source
        assert "TBGMP_BACKEND_SCRIPT" not in source
