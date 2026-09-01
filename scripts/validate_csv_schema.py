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
        "n",
        "top_percent",
        "random_percent",
        "bottom_percent",
        "top_minus_random_pp",
        "top_minus_random_ci95",
        "top_minus_bottom_pp",
        "top_minus_bottom_ci95",
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
    "table_risk_ablation.csv": {
        "model", "score", "top1_cumulative", "k100", "mean_first_success_k",
        "cumulative_auc_percent",
    },
    "table_weight_sensitivity.csv": {
        "model", "alpha", "beta", "gamma", "spearman_rho", "top3_overlap",
        "mean_first_success_k",
    },
    "table_domain_heldout.csv": {
        "model", "profiling_domains", "evaluation_domains", "topk", "randomk", "bottomk",
    },
    "table_frozen_top3.csv": {
        "model", "fp16", "aggressive", "uniform", "frozen_top3", "random3",
        "bottom3", "recovery", "kv_saving_percent",
    },
    "table_ruler_transfer.csv": {
        "model", "n", "top1_percent", "top4_percent", "top8_percent",
        "top12_percent", "cumulative_top12", "random12_percent",
        "bottom12_percent", "kv_saving_percent",
    },
    "figure_2_domain_recovery.csv": {
        "model", "domain", "k1", "k2", "k3", "k4", "k5", "k6", "k7",
        "k8", "k9", "k10", "k11", "k12",
    },
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

QWEN25_RISK_ABLATION_REQUIRED = {
    "risk_ablation_first_success_by_case.csv": {
        "domain", "context_length", "needle_depth", "seed", "risk_score",
        "first_success_k", "recovered_within_top12",
    },
    "risk_ablation_summary.csv": {
        "model", "risk_score", "n_sensitive", "top1_cumulative_count",
        "k100", "mean_first_success_k", "cumulative_auc_percent",
    },
    "risk_ablation_topk_curve.csv": {
        "model", "risk_score", "k", "cumulative_count", "cumulative_percent",
        "exact_at_k_count", "exact_at_k_percent",
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

    raw_schema = json.loads(
        (ROOT / "data" / "schema" / "raw_output_schema.json").read_text(
            encoding="utf-8"
        )
    )
    raw_required = set(raw_schema["required"])
    raw_demo = ROOT / "data" / "demo" / "demo_raw_outputs.jsonl"
    for line_number, line in enumerate(
        raw_demo.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        missing = sorted(raw_required - set(record))
        if missing:
            failures.append(
                f"{raw_demo.relative_to(ROOT)} line {line_number}: "
                f"missing fields {missing}"
            )
    print(f"PASS raw JSONL fixture: {raw_demo.relative_to(ROOT)}")

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

    qwen25_dir = ROOT / "results" / "main_evidence" / "qwen25_3b"
    for filename, required in QWEN25_RISK_ABLATION_REQUIRED.items():
        path = qwen25_dir / filename
        if not path.exists():
            failures.append(f"missing Qwen2.5 risk-ablation file: {path.relative_to(ROOT)}")
            continue
        columns = set(pd.read_csv(path).columns)
        missing = sorted(required - columns)
        if missing:
            failures.append(f"{path.relative_to(ROOT)}: missing columns {missing}")
    print("PASS Qwen2.5 camera-ready risk-ablation bundle")

    if failures:
        raise SystemExit("\n".join(failures))
    print("CSV/config schema validation: PASS")


if __name__ == "__main__":
    main()
