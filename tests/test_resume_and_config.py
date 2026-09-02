from __future__ import annotations

from pathlib import Path

from tbgmp.backends.turboquant_backend import TurboQuantBackend
from tbgmp.experiment_config import validate_experiment_config
from tbgmp.result_store import IncrementalResultStore


def identity(policy: str) -> dict:
    return {
        "model": "model",
        "case_id": "case",
        "stage": "stage_a_discovery",
        "policy": policy,
    }


def test_incremental_store_resumes_only_completed_rows(tmp_path: Path) -> None:
    output = tmp_path / "results.csv"
    store = IncrementalResultStore(output, checkpoint_every=1)
    store.append(
        {
            **identity("fp16"),
            "status": "success",
            "completed": True,
            "oom": False,
        }
    )
    store.append(
        {
            **identity("uniform_k2_v2_rw128"),
            "status": "oom",
            "completed": False,
            "oom": True,
            "error": "CUDA out of memory",
        }
    )
    store.close()

    resumed = IncrementalResultStore(output, checkpoint_every=1)
    assert resumed.completed(identity("fp16")) is True
    assert resumed.completed(identity("uniform_k2_v2_rw128")) is False
    assert resumed.get(identity("uniform_k2_v2_rw128"))["oom"] is True


class CachedRuntimeBackend(TurboQuantBackend):
    def __init__(self):
        super().__init__(turboquant_root=None)

    def _raise_if_unavailable(self):
        return None

    def _load_runtime(self, model_path: str):
        self.model_load_count += 1
        return {"tokenizer": object()}


def test_backend_runtime_is_loaded_once_per_model_path() -> None:
    backend = CachedRuntimeBackend()
    first = backend.get_tokenizer("model-a")
    second = backend.get_tokenizer("model-a")
    assert first is second
    assert backend.model_load_count == 1


def test_configured_policy_sets_are_validated() -> None:
    policies = {
        "fp16": {"type": "fp16"},
        "aggressive": {"key_bits": 2, "value_bits": 2},
        "safe": {"key_bits": 6, "value_bits": 2},
        "topk": {"protected_key_bits": 6},
    }
    config = {
        "experiment": {
            "domains": ["math"],
            "context_lengths": [4096],
            "needle_depths": [50],
            "seeds": [0],
            "maximum_topk": 12,
            "discovery_policies": ["fp16", "aggressive", "safe"],
            "aggressive_policies": ["aggressive"],
            "safe_policies": ["safe"],
            "tbgmp_policy": "topk",
        }
    }
    resolved = validate_experiment_config(config, policies)
    assert resolved["aggressive_policies"] == ["aggressive"]
    assert resolved["safe_policies"] == ["safe"]
