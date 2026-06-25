from __future__ import annotations


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
