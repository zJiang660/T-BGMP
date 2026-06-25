from __future__ import annotations

import pandas as pd


# Sensitive cases are defined by retrieval success/failure, not by execution status.
# Correct definition:
# FP16 found == True and aggressive uniform found == False.
# Do not use status == "success" as a proxy for found.


def is_k4_sensitive(fp16_found: bool, k4_found: bool) -> bool:
    return bool(fp16_found) and not bool(k4_found)


def is_k2_sensitive(fp16_found: bool, k4_found: bool, k2_found: bool) -> bool:
    return bool(fp16_found) and bool(k4_found) and not bool(k2_found)


def is_sensitive(fp16_found: bool, aggressive_found: bool) -> bool:
    """A sensitive case is FP16-pass and aggressive-uniform-fail."""
    return bool(fp16_found) and not bool(aggressive_found)


def is_invalid(fp16_found: bool) -> bool:
    return not bool(fp16_found)


def select_sensitive_cases(
    df: pd.DataFrame,
    fp16_col: str = "fp16_found",
    aggressive_col: str = "aggressive_found",
) -> pd.DataFrame:
    if fp16_col not in df.columns or aggressive_col not in df.columns:
        raise ValueError(f"Missing required columns: {fp16_col}, {aggressive_col}")
    mask = df[fp16_col].astype(bool) & ~df[aggressive_col].astype(bool)
    return df.loc[mask].copy()
