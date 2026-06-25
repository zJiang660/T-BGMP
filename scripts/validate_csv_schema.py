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

CASE_LEVEL_REQUIRED = {
    "sensitive_cases.csv": {
        "model",
        "case_id",
        "fp16_found",
        "aggressive_found",
    },
    "topk_recovery.csv": {
        "model",
        "case_id",
        "policy",
        "found",
        "topk_k",
        "protected_layers",
    },
    "random_bottom_controls.csv": {
        "model",
        "case_id",
        "policy",
        "found",
        "protected_layers",
    },
    "risk_ranking.csv": {"rank"},
    "efficiency_summary.csv": {
        "model",
        "sensitive_cases",
        "restored_cases",
    },
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

    case_dirs = [
        ROOT / "results" / "main_evidence" / "qwen3_4b",
        ROOT / "results" / "main_evidence" / "qwen25_3b",
        ROOT / "results" / "main_evidence" / "qwen25_14b",
        ROOT / "results" / "main_evidence" / "llama32_3b",
        ROOT / "results" / "supporting" / "gemma2_9b",
    ]
    for directory in case_dirs:
        for filename, required in CASE_LEVEL_REQUIRED.items():
            path = directory / filename
            if not path.exists():
                failures.append(f"missing case-level file: {path.relative_to(ROOT)}")
                continue
            columns = set(pd.read_csv(path).columns)
            missing = sorted(required - columns)
            if missing:
                failures.append(
                    f"{path.relative_to(ROOT)}: missing columns {missing}"
                )
        provenance = directory / "source_provenance.json"
        if not provenance.exists():
            failures.append(f"missing provenance: {provenance.relative_to(ROOT)}")
        else:
            json.loads(provenance.read_text(encoding="utf-8"))
        print(f"PASS case-level bundle: {directory.relative_to(ROOT)}")

    if failures:
        raise SystemExit("\n".join(failures))
    print("CSV/config schema validation: PASS")


if __name__ == "__main__":
    main()
