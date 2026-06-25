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
        if not configured:
            raise RuntimeError(f"{INTEGRATION_MESSAGE} Set TURBOQUANT_ROOT first.")

        self.root = Path(configured).expanduser().resolve()
        self.device = device
        if not self.root.is_dir():
            raise RuntimeError(
                f"{INTEGRATION_MESSAGE} The configured TurboQuant directory "
                "does not exist."
            )

        root_text = str(self.root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        try:
            module = importlib.import_module("turboquant")
            compressors = importlib.import_module("turboquant.compressors_v3")
        except Exception as exc:
            raise RuntimeError(
                f"{INTEGRATION_MESSAGE} The external package could not be imported."
            ) from exc

        if not hasattr(compressors, "TurboQuantV3"):
            raise RuntimeError(
                f"{INTEGRATION_MESSAGE} TurboQuantV3 was not found in the "
                "configured runtime."
            )
        self.runtime_module = module
        self.compressor_class = compressors.TurboQuantV3

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

        raise NotImplementedError(
            f"{INTEGRATION_MESSAGE} The upstream runtime must be adapted to "
            "apply arbitrary protected key-layer IDs without changing value "
            "precision. No inference result was generated."
        )


def create_backend(**kwargs: Any) -> TurboQuantBackend:
    return TurboQuantBackend(**kwargs)
