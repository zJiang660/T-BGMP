from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuantizationConfig:
    key_bits: int
    value_bits: int
    residual_window: int = 128
    protected_key_bits: int | None = None
    protected_layers: tuple[int, ...] = ()

    def validate(self) -> None:
        for name, value in [
            ("key_bits", self.key_bits),
            ("value_bits", self.value_bits),
        ]:
            if value <= 0 or value > 16:
                raise ValueError(f"{name} must be in [1, 16]")
        if self.protected_key_bits is not None and not 1 <= self.protected_key_bits <= 16:
            raise ValueError("protected_key_bits must be in [1, 16]")
        if self.residual_window < 0:
            raise ValueError("residual_window must be non-negative")


def estimate_nominal_kv_saving(
    config: QuantizationConfig,
    num_layers: int,
) -> float:
    """Estimate bit-budget saving before residual-window and metadata overhead."""
    config.validate()
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    protected = len(set(config.protected_layers))
    if protected > num_layers:
        raise ValueError("protected layer count exceeds num_layers")
    protected_bits = config.protected_key_bits or config.key_bits
    key_total = (num_layers - protected) * config.key_bits + protected * protected_bits
    value_total = num_layers * config.value_bits
    fp16_total = num_layers * 32
    return 100.0 * (1.0 - (key_total + value_total) / fp16_total)


# This module describes precision policies and nominal memory accounting. It
# does not implement a quantizer kernel. A real backend must apply these values
# to its own KV-cache implementation.
