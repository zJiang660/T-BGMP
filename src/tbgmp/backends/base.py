from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class GenerationRequest:
    model_path: str
    prompt: str
    answer: str
    policy: dict
    max_new_tokens: int = 32
    seed: int = 0


@dataclass
class GenerationResult:
    response: str
    found: bool
    status: str
    metadata: dict[str, Any] = field(default_factory=dict)


class KVCacheBackend(Protocol):
    def generate(self, request: GenerationRequest) -> GenerationResult:
        ...
