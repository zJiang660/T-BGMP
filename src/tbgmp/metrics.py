from __future__ import annotations

from math import sqrt


def safe_divide(num: float, den: float, default: float = 0.0) -> float:
    return default if den == 0 else num / den


def recovery_rate(restored: int, sensitive: int) -> float:
    return safe_divide(restored, sensitive)


def kv_saving_advantage(tbgmp_saving: float, uniform_saving: float) -> float:
    return tbgmp_saving - uniform_saving


def fisher_exact_safe(table):
    try:
        from scipy.stats import fisher_exact

        return fisher_exact(table)
    except Exception:
        return None


def wilson_interval_safe(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
) -> tuple[float, float] | None:
    if total <= 0:
        return None
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return center - half, center + half
