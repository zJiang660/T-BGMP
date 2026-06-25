from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tbgmp.metrics import kv_saving_advantage
from tbgmp.sensitive_cases import select_sensitive_cases
from tbgmp.utils import parse_bool


def main() -> None:
    cases = pd.read_csv(ROOT / "data" / "demo" / "demo_cases.csv")
    for column in ["fp16_found", "aggressive_found", "topk_found"]:
        cases[column] = cases[column].map(parse_bool)
    sensitive = select_sensitive_cases(cases)
    recovered = sensitive[sensitive["topk_found"]].copy()
    recovered["saving_advantage"] = recovered.apply(
        lambda row: kv_saving_advantage(
            float(row["tbgmp_saving"]),
            float(row["uniform_safe_saving"]),
        ),
        axis=1,
    )
    print(
        "Stage F efficiency: mean saving advantage "
        f"{recovered['saving_advantage'].mean():.2f} percentage points"
    )


if __name__ == "__main__":
    main()
