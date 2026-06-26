from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any

from .base import GenerationRequest, GenerationResult


INTEGRATION_MESSAGE = (
    "Full model execution requires the external TurboQuant backend. "
    "See docs/backend_integration.md."
)


class TurboQuantBackend:
    """Validated integration boundary for the external TurboQuant runtime.

    The upstream runtime currently exposes TurboQuantV3 and example cache
    wrappers, but its protected-layer option protects first/last layer counts.
    T-BGMP requires arbitrary ranked key-layer IDs and key-only protection.
    This adapter therefore refuses to approximate the method silently.
    """

    request_api = True

    def __init__(
        self,
        turboquant_root: str | None = None,
        device: str = "cuda",
        **_: Any,
    ):
        configured = turboquant_root or os.environ.get("TURBOQUANT_ROOT")
        self.root = Path(configured).expanduser().resolve() if configured else None
        self.device = device
        self.runtime_module = None
        self.compressor_class = None
        self._import_error = ""
        self._compressors_module = None
        self._compressors_file = (
            self.root / "turboquant" / "compressors_v3.py" if self.root else None
        )
        self._generation_file = (
            self.root / "turboquant" / "generation_test.py" if self.root else None
        )

        if self.root and self.root.is_dir():
            root_text = str(self.root)
            if root_text not in sys.path:
                sys.path.insert(0, root_text)
            try:
                self.runtime_module = importlib.import_module("turboquant")
                self._compressors_module = importlib.import_module(
                    "turboquant.compressors_v3"
                )
                if hasattr(self._compressors_module, "TurboQuantV3"):
                    self.compressor_class = self._compressors_module.TurboQuantV3
            except Exception as exc:  # pragma: no cover - message asserted indirectly
                self._import_error = repr(exc)

    def check_available(self) -> dict[str, Any]:
        root_ok = self.root is not None and self.root.is_dir()
        import_ok = self.runtime_module is not None and not self._import_error
        compressor_ok = self.compressor_class is not None
        compressors_file_ok = (
            self._compressors_file is not None and self._compressors_file.is_file()
        )
        generation_file_ok = (
            self._generation_file is not None and self._generation_file.is_file()
        )
        source_text = ""
        if compressors_file_ok:
            source_text += self._compressors_file.read_text(
                encoding="utf-8", errors="replace"
            )
        if generation_file_ok:
            source_text += self._generation_file.read_text(
                encoding="utf-8", errors="replace"
            )
        arbitrary_patch_detected = (
            "protected_layer_ids" in source_text
            and "protected_key_bits" in source_text
        )
        return {
            "backend": "turboquant",
            "root_configured": self.root is not None,
            "root_exists": root_ok,
            "compressors_v3_file": compressors_file_ok,
            "generation_test_file": generation_file_ok,
            "import_ok": import_ok,
            "compressor_v3_found": compressor_ok,
            "protected_layers_semantics": "prefix_suffix_count",
            "arbitrary_patch_detected": arbitrary_patch_detected,
            "arbitrary_protected_key_layer_ids": arbitrary_patch_detected,
            "key_only_protection": arbitrary_patch_detected,
            "residual_window": "supported by upstream V3 cache",
            "ready_for_tbgmp_generation": False,
            "error": self._import_error,
            "message": (
                "The external TurboQuant runtime can be inspected. Exact "
                "generation still requires a callable backend binding for "
                "arbitrary risk-ranked protected key-layer IDs."
            ),
        }

    def _raise_if_unavailable(self) -> None:
        status = self.check_available()
        if not status["root_configured"]:
            raise RuntimeError(f"{INTEGRATION_MESSAGE} Set TURBOQUANT_ROOT first.")
        if not status["root_exists"]:
            raise RuntimeError(
                f"{INTEGRATION_MESSAGE} The configured TurboQuant directory "
                "does not exist."
            )
        if not status["import_ok"]:
            raise RuntimeError(
                f"{INTEGRATION_MESSAGE} The external package could not be imported."
            )
        if not status["compressor_v3_found"]:
            raise RuntimeError(
                f"{INTEGRATION_MESSAGE} TurboQuantV3 was not found in the "
                "configured runtime."
            )

    def generate(
        self,
        request: GenerationRequest | None = None,
        **legacy_kwargs: Any,
    ) -> GenerationResult:
        if request is None:
            quantization = legacy_kwargs.get("quantization")
            policy = {
                "name": legacy_kwargs.get("policy_name", ""),
                "key_bits": getattr(quantization, "key_bits", 16),
                "value_bits": getattr(quantization, "value_bits", 16),
                "protected_key_bits": getattr(
                    quantization, "protected_key_bits", None
                ),
                "protected_layers": list(
                    getattr(quantization, "protected_layers", ())
                ),
                "residual_window": getattr(quantization, "residual_window", 0),
            }
            request = GenerationRequest(
                model_path=str(legacy_kwargs.get("model_path", "")),
                prompt=str(legacy_kwargs.get("prompt", "")),
                answer=str(legacy_kwargs.get("answer", "")),
                policy=policy,
                max_new_tokens=int(legacy_kwargs.get("max_new_tokens", 32)),
                seed=int(legacy_kwargs.get("seed", 0)),
            )

        self._raise_if_unavailable()
        raise NotImplementedError(
            "Full TurboQuant generation is unavailable because the public "
            "TurboQuant runtime does not directly expose arbitrary risk-ranked "
            "protected key-layer IDs. Apply the patch described in "
            "docs/turboquant_patch_guide.md or provide a compatible backend."
        )


def create_backend(**kwargs: Any) -> TurboQuantBackend:
    return TurboQuantBackend(**kwargs)
