from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "results" / "paper_tables"

REQUIRED_COLUMNS = {
    "table_main_evidence.csv": {
        "model",
        "sensitive",
        "recovered",
        "top_rate",
        "random_rate",
        "bottom_rate",
    },
    "table_first_success_k.csv": {
        "model",
        "median",
        "mean",
        "max",
        "top1",
        "top4",
        "top8",
        "top12",
    },
    "table_control_statistics.csv": {
        "model",
        "top",
        "random",
        "bottom",
        "p_random",
        "p_bottom",
    },
    "table_qwen25_scale.csv": {
        "model",
        "scale",
        "sensitive_cases",
        "recovery",
        "controls",
        "saving_advantage",
    },
    "table_supporting_models.csv": {
        "model",
        "sensitive",
        "recovered",
        "overall_recovery",
        "usage",
    },
    "table_gemma2_boundary.csv": {"policy", "found"},
    "table_boundary_models.csv": {"model", "status", "reason", "classification"},
}


def main() -> None:
    failures: list[str] = []
    for filename, required in REQUIRED_COLUMNS.items():
        path = DATA_DIR / filename
        if not path.exists():
            failures.append(f"missing file: {path.relative_to(ROOT)}")
            continue
        columns = set(pd.read_csv(path).columns)
        missing = sorted(required - columns)
        if missing:
            failures.append(f"{filename}: missing columns {missing}")
        else:
            print(f"PASS {filename}: {len(columns)} columns")

    for path in sorted((ROOT / "data" / "schema").glob("*.json")):
        json.loads(path.read_text(encoding="utf-8"))
        print(f"PASS JSON schema parse: {path.relative_to(ROOT)}")

    for path in sorted((ROOT / "configs").glob("*.yaml")):
        yaml.safe_load(path.read_text(encoding="utf-8"))
        print(f"PASS YAML parse: {path.relative_to(ROOT)}")

    if failures:
        raise SystemExit("\n".join(failures))
    print("CSV/config schema validation: PASS")


if __name__ == "__main__":
    main()
