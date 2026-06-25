from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from .quantization import QuantizationConfig


@dataclass
class GenerationResult:
    response: str
    found: bool
    status: str = "success"
    error: str = ""
    runtime_s: float | None = None
    tok_per_s: float | None = None
    peak_gpu_gb: float | None = None
    kv_saving: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class KVCacheBackend(Protocol):
    def generate(
        self,
        *,
        model_path: str,
        prompt: str,
        answer: str,
        policy_name: str,
        quantization: QuantizationConfig | None,
        max_new_tokens: int,
        seed: int = 0,
    ) -> GenerationResult:
        """Run one generation under a cache policy."""


def load_backend(spec: str, **kwargs) -> KVCacheBackend:
    """Load an external backend from `module.path:factory_name`."""
    if ":" not in spec:
        raise ValueError("backend spec must use module.path:factory_name")
    module_name, factory_name = spec.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name)
    backend = factory(**kwargs)
    if not hasattr(backend, "generate"):
        raise TypeError("backend factory must return an object with generate()")
    return backend


class DryRunBackend:
    """Schema-testing backend that never performs model inference."""

    def generate(self, **kwargs) -> GenerationResult:
        return GenerationResult(
            response="",
            found=False,
            status="dry_run",
            error="No model execution was performed.",
        )


# A production backend must be supplied by the user because TurboQuant forks,
# transformers cache APIs, model remote code, and GPU kernels vary by setup.
# The adapter boundary keeps the public experiment pipeline explicit without
# pretending that this repository ships those external components.
