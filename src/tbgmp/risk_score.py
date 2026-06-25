from __future__ import annotations

import numpy as np
import pandas as pd


def minmax_norm(values, eps: float = 1e-12):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.isnan(arr).all():
        raise ValueError("values must contain at least one finite value")
    lo = np.nanmin(arr)
    hi = np.nanmax(arr)
    return (arr - lo) / (hi - lo + eps)


def compute_risk_scores(
    df: pd.DataFrame,
    mse_col: str = "mse_p95",
    ip_col: str = "ip_p95",
    effdim_col: str | None = "effective_dim",
) -> pd.DataFrame:
    out = df.copy()
    if mse_col not in out.columns or ip_col not in out.columns:
        raise ValueError(f"Missing required columns: {mse_col}, {ip_col}")

    risk = minmax_norm(out[mse_col]) + minmax_norm(np.log1p(out[ip_col]))
    if effdim_col and effdim_col in out.columns:
        inv_eff = 1.0 / (out[effdim_col].astype(float) + 1e-12)
        risk = risk + minmax_norm(inv_eff)

    out["risk_score"] = risk
    return out.sort_values("risk_score", ascending=False).reset_index(drop=True)
