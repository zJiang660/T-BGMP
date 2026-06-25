from __future__ import annotations

import random
from collections.abc import Iterable


def sample_random_layers(layers: Iterable[int], k: int, seed: int = 0) -> list[int]:
    eligible = sorted({int(layer) for layer in layers})
    if k < 0 or k > len(eligible):
        raise ValueError("k must satisfy 0 <= k <= number of eligible layers")
    return sorted(random.Random(seed).sample(eligible, k))


def bottomk_layers(ranked_layers: Iterable[int], k: int) -> list[int]:
    ranked = [int(layer) for layer in ranked_layers]
    if k < 0 or k > len(ranked):
        raise ValueError("k must satisfy 0 <= k <= number of ranked layers")
    return list(reversed(ranked))[:k]
