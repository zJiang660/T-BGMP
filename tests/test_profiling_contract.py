from __future__ import annotations

from tbgmp.profiling import get_cache_layer


class LegacyCache:
    def to_legacy_cache(self):
        return [("key-0", "value-0"), ("key-1", "value-1")]


def test_cache_layer_supports_transformers_legacy_conversion() -> None:
    key, value = get_cache_layer(LegacyCache(), 1)
    assert key == "key-1"
    assert value == "value-1"
