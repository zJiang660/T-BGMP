from pathlib import Path
import subprocess

from scripts.manage_turboquant_patch import inspect_patch_state
from tbgmp.turboquant_patch_validation import validate_runtime_contract


class FakeCompressor:
    def __init__(
        self,
        head_dim,
        key_bits=4,
        value_bits=2,
        residual_window=0,
        layer_idx=0,
        n_layers=4,
        protected_layers=0,
        protected_layer_ids=None,
        protected_key_bits=8,
        **kwargs,
    ):
        del head_dim, residual_window, n_layers, protected_layers, kwargs
        selected = layer_idx in set(protected_layer_ids or [])
        self.key_bits = protected_key_bits if selected else key_bits
        self.value_bits = value_bits


class FakeCache:
    def __init__(
        self,
        key_bits=4,
        value_bits=2,
        residual_window=0,
        protected_layers=0,
        protected_layer_ids=None,
        protected_key_bits=8,
        n_layers=4,
    ):
        self.kwargs = {
            "key_bits": key_bits,
            "value_bits": value_bits,
            "residual_window": residual_window,
            "protected_layers": protected_layers,
            "protected_layer_ids": protected_layer_ids,
            "protected_key_bits": protected_key_bits,
            "n_layers": n_layers,
        }

    def _get_compressor(self, layer_idx, head_dim, device):
        return FakeCompressor(
            head_dim=head_dim,
            layer_idx=layer_idx,
            device=device,
            **self.kwargs,
        )


class UnpatchedCache:
    def __init__(self, key_bits=4, value_bits=2):
        del key_bits, value_bits


class BothKvProtectedCompressor(FakeCompressor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.key_bits == kwargs.get("protected_key_bits", 8):
            self.value_bits = self.key_bits


class BothKvProtectedCache(FakeCache):
    def _get_compressor(self, layer_idx, head_dim, device):
        return BothKvProtectedCompressor(
            head_dim=head_dim,
            layer_idx=layer_idx,
            device=device,
            **self.kwargs,
        )


def test_runtime_contract_checks_key_only_behavior():
    status = validate_runtime_contract(FakeCompressor, FakeCache)
    assert status["passed"] is True
    assert status["selected_layer_bits"] == {"key_bits": 8, "value_bits": 2}
    assert status["unselected_layer_bits"] == {"key_bits": 4, "value_bits": 2}


def test_runtime_contract_rejects_missing_cache_api():
    status = validate_runtime_contract(FakeCompressor, UnpatchedCache)
    assert status["passed"] is False
    assert status["signature_ok"] is False


def test_runtime_contract_rejects_value_precision_change():
    status = validate_runtime_contract(FakeCompressor, BothKvProtectedCache)
    assert status["signature_ok"] is True
    assert status["behavior_ok"] is False
    assert status["selected_layer_bits"] == {"key_bits": 8, "value_bits": 8}


def test_patch_state_is_idempotent(tmp_path: Path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    target = tmp_path / "example.txt"
    target.write_text("before\n", encoding="utf-8")
    patch = tmp_path / "change.patch"
    patch.write_text(
        "diff --git a/example.txt b/example.txt\n"
        "--- a/example.txt\n"
        "+++ b/example.txt\n"
        "@@ -1 +1 @@\n"
        "-before\n"
        "+after\n",
        encoding="utf-8",
    )

    assert inspect_patch_state(tmp_path, patch)["state"] == "ready_to_apply"
    subprocess.run(["git", "-C", str(tmp_path), "apply", str(patch)], check=True)
    assert inspect_patch_state(tmp_path, patch)["state"] == "already_applied"
