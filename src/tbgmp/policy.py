from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


POLICIES = {
    "fp16": {"type": "fp16"},
    "uniform_k2_v2_rw128": {
        "key_bits": 2,
        "value_bits": 2,
        "residual_window": 128,
    },
    "uniform_k4_v2_rw128": {
        "key_bits": 4,
        "value_bits": 2,
        "residual_window": 128,
    },
    "uniform_k6_v2_rw128": {
        "key_bits": 6,
        "value_bits": 2,
        "residual_window": 128,
    },
    "uniform_k6_v4_rw128": {
        "key_bits": 6,
        "value_bits": 4,
        "residual_window": 128,
    },
    "tbgmp_topk": {"type": "risk_ranked_key_protection"},
    "random_k": {"type": "same_budget_random_key_protection"},
    "bottom_k": {"type": "same_budget_bottom_key_protection"},
}


@dataclass(frozen=True)
class TopKPolicy:
    default_key_bits: int
    default_value_bits: int
    protected_key_bits: int
    protected_layers: tuple[int, ...]
    residual_window: int = 128


def make_topk_policy(
    default_key_bits: int,
    default_value_bits: int,
    protected_key_bits: int,
    protected_layers: Iterable[int],
    residual_window: int = 128,
) -> TopKPolicy:
    return TopKPolicy(
        default_key_bits=default_key_bits,
        default_value_bits=default_value_bits,
        protected_key_bits=protected_key_bits,
        protected_layers=tuple(int(x) for x in protected_layers),
        residual_window=residual_window,
    )


def make_bottomk_layers(ranked_layers: list[int], k: int) -> list[int]:
    if k < 0:
        raise ValueError("k must be non-negative")
    return list(reversed(ranked_layers))[:k]


def make_topk_layers(ranked_layers: list[int], k: int) -> list[int]:
    if k < 0:
        raise ValueError("k must be non-negative")
    return ranked_layers[:k]
