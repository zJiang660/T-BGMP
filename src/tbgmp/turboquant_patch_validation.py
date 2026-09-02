from __future__ import annotations

import inspect
from typing import Any


REQUIRED_COMPRESSOR_PARAMETERS = {"protected_layer_ids", "protected_key_bits"}
REQUIRED_CACHE_PARAMETERS = {"protected_layer_ids", "protected_key_bits"}


def _signature_parameters(callable_obj: Any) -> set[str]:
    return set(inspect.signature(callable_obj).parameters)


def validate_runtime_contract(compressor_class: Any, cache_class: Any) -> dict[str, Any]:
    """Verify the patched public API and key-only layer behavior.

    The behavioral check follows the same cache construction path used by the
    backend. It proves that an explicitly protected layer receives the higher
    key precision while its value precision and an unprotected layer remain at
    the aggressive defaults.
    """

    compressor_parameters = _signature_parameters(compressor_class.__init__)
    cache_parameters = _signature_parameters(cache_class.__init__)
    missing_compressor = sorted(REQUIRED_COMPRESSOR_PARAMETERS - compressor_parameters)
    missing_cache = sorted(REQUIRED_CACHE_PARAMETERS - cache_parameters)
    signature_ok = not missing_compressor and not missing_cache

    behavior_ok = False
    behavior_error = ""
    selected_bits: dict[str, int] = {}
    unselected_bits: dict[str, int] = {}
    if signature_ok:
        try:
            cache = cache_class(
                key_bits=4,
                value_bits=2,
                residual_window=0,
                protected_layers=0,
                protected_layer_ids=[2],
                protected_key_bits=8,
                n_layers=4,
            )
            selected = cache._get_compressor(2, 8, "cpu")
            unselected = cache._get_compressor(1, 8, "cpu")
            selected_bits = {
                "key_bits": int(selected.key_bits),
                "value_bits": int(selected.value_bits),
            }
            unselected_bits = {
                "key_bits": int(unselected.key_bits),
                "value_bits": int(unselected.value_bits),
            }
            behavior_ok = selected_bits == {"key_bits": 8, "value_bits": 2} and (
                unselected_bits == {"key_bits": 4, "value_bits": 2}
            )
        except Exception as exc:  # pragma: no cover - depends on external runtime
            behavior_error = repr(exc)

    return {
        "signature_ok": signature_ok,
        "behavior_ok": behavior_ok,
        "passed": signature_ok and behavior_ok,
        "missing_compressor_parameters": missing_compressor,
        "missing_cache_parameters": missing_cache,
        "selected_layer_bits": selected_bits,
        "unselected_layer_bits": unselected_bits,
        "behavior_error": behavior_error,
    }

