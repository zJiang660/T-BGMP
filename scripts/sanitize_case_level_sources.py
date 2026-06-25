from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RESULT_COLUMNS = [
    "model",
    "case_id",
    "domain",
    "context_length",
    "actual_context_length",
    "depth",
    "seed",
    "answer",
    "policy",
    "policy_type",
    "key_bits",
    "value_bits",
    "protected_key_bits",
    "protected_layers",
    "residual_window",
    "found",
    "status",
    "response_excerpt",
    "topk_k",
    "kv_saving",
    "runtime_s",
    "tok_per_s",
    "peak_gpu_gb",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def excerpt(value, limit: int = 80) -> str:
    if pd.isna(value):
        return ""
    text = " ".join(str(value).split())
    text = re.sub(r"[A-Za-z]:\\[^\s]+|/gpfs/[^\s]+", "[REDACTED_PATH]", text)
    return text[:limit]


def case_id(row: pd.Series) -> str:
    if "case_id" in row and pd.notna(row["case_id"]):
        return str(row["case_id"])
    depth = row.get("needle_depth", row.get("depth", ""))
    return (
        f"{row.get('domain', 'unknown')}_ctx{int(row.get('context_length', 0))}"
        f"_d{int(depth)}_s{int(row.get('seed', 0))}"
    )


def topk_k(row: pd.Series) -> int | None:
    for column in ["topk_k", "policy_type", "policy_name"]:
        value = str(row.get(column, ""))
        match = re.search(r"top(\d+)", value, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def clean_result_rows(df: pd.DataFrame, model: str) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "model": model,
                "case_id": case_id(row),
                "domain": row.get("domain", ""),
                "context_length": row.get(
                    "requested_context_length",
                    row.get("context_length", ""),
                ),
                "actual_context_length": row.get(
                    "actual_context_length",
                    row.get("actual_prompt_tokens", ""),
                ),
                "depth": row.get("needle_depth", row.get("depth", "")),
                "seed": row.get("seed", ""),
                "answer": row.get("needle_string", row.get("answer_string", "")),
                "policy": row.get("policy_name", ""),
                "policy_type": row.get("policy_type", ""),
                "key_bits": row.get("default_key_bits", ""),
                "value_bits": row.get("default_value_bits", ""),
                "protected_key_bits": row.get("protected_key_bits", ""),
                "protected_layers": row.get(
                    "protected_key_layers",
                    row.get("topk_layers", row.get("control_layers", "")),
                ),
                "residual_window": row.get("residual_window", ""),
                "found": bool(row.get("found", False)),
                "status": row.get("status", "completed"),
                "response_excerpt": excerpt(row.get("response", "")),
                "topk_k": topk_k(row),
                "kv_saving": row.get("kv_saving_percent", ""),
                "runtime_s": row.get("runtime_s", ""),
                "tok_per_s": row.get("tok_per_s", row.get("tok_s", "")),
                "peak_gpu_gb": row.get("peak_gpu_gb", row.get("peak_gb", "")),
            }
        )
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def write_provenance(output_dir: Path, sources: list[Path], notes: str) -> None:
    payload = {
        "source_files": [
            {
                "filename": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sources
        ],
        "notes": notes,
    }
    (output_dir / "source_provenance.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def write_sensitive(
    classification: pd.DataFrame,
    model: str,
    output_dir: Path,
    case_type_column: str,
) -> pd.DataFrame:
    sensitive = classification[
        classification[case_type_column].astype(str).str.contains("sensitive", case=False)
    ].copy()
    rows = []
    for _, row in sensitive.iterrows():
        ctype = str(row[case_type_column])
        aggressive = (
            "Uniform K4/V2 rw128"
            if "K4" in ctype.upper()
            else "Uniform K2/V2 rw128"
        )
        aggressive_found = (
            bool(row.get("uniform_k4v2_found", False))
            if "K4" in ctype.upper()
            else bool(row.get("uniform_k2v2_found", False))
        )
        rows.append(
            {
                "model": model,
                "case_id": case_id(row),
                "domain": row.get("domain", ""),
                "context_length": row.get("context_length", ""),
                "depth": row.get("needle_depth", row.get("depth", "")),
                "seed": row.get("seed", ""),
                "answer": row.get("needle_string", row.get("answer_string", "")),
                "case_type": ctype,
                "aggressive_policy": aggressive,
                "fp16_found": bool(row.get("fp16_found", True)),
                "aggressive_found": aggressive_found,
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "sensitive_cases.csv", index=False)
    return result


def write_first_success_and_efficiency(
    topk: pd.DataFrame,
    output_dir: Path,
) -> None:
    successful = topk[topk["found"]].dropna(subset=["topk_k"]).copy()
    successful["topk_k"] = successful["topk_k"].astype(int)
    first = (
        successful.sort_values(["case_id", "topk_k"])
        .groupby("case_id", as_index=False)
        .first()
    )
    first[
        ["model", "case_id", "topk_k", "protected_layers", "kv_saving"]
    ].to_csv(output_dir / "first_success_cases.csv", index=False)
    summary = pd.DataFrame(
        [
            {
                "model": topk["model"].iloc[0] if len(topk) else "",
                "sensitive_cases": topk["case_id"].nunique(),
                "restored_cases": first["case_id"].nunique(),
                "mean_first_success_k": first["topk_k"].mean(),
                "max_first_success_k": first["topk_k"].max(),
                "mean_first_success_kv_saving": pd.to_numeric(
                    first["kv_saving"], errors="coerce"
                ).mean(),
            }
        ]
    )
    summary.to_csv(output_dir / "efficiency_summary.csv", index=False)


def sanitize_sip_model(
    source_dir: Path,
    risk_path: Path,
    output_dir: Path,
    model: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sensitive_path = source_dir / "sensitive_cases.csv"
    recovery_path = source_dir / "recovery_topk_results.csv"
    controls_path = source_dir / "same_budget_controls.csv"
    classification = pd.read_csv(sensitive_path)

    discovery_rows = []
    for _, row in classification.iterrows():
        base = {
            "model": model,
            "case_id": case_id(row),
            "domain": row["domain"],
            "context_length": row["context_length"],
            "depth": row["needle_depth"],
            "seed": row["seed"],
            "answer": row["needle_string"],
            "status": "completed" if bool(row.get("completed", True)) else "incomplete",
        }
        for policy, key_bits, value_bits, found in [
            ("FP16", 16, 16, row["fp16_found"]),
            ("Uniform K2/V2 rw128", 2, 2, row["uniform_k2v2_found"]),
            ("Uniform K4/V2 rw128", 4, 2, row["uniform_k4v2_found"]),
        ]:
            discovery_rows.append(
                {
                    **base,
                    "policy": policy,
                    "key_bits": key_bits,
                    "value_bits": value_bits,
                    "residual_window": 0 if key_bits == 16 else 128,
                    "found": bool(found),
                }
            )
    pd.DataFrame(discovery_rows).to_csv(
        output_dir / "stage_a_discovery.csv",
        index=False,
    )
    sensitive = write_sensitive(
        classification,
        model,
        output_dir,
        "case_type",
    )
    sensitive_ids = set(sensitive["case_id"])

    recovery_raw = pd.read_csv(recovery_path)
    recovery = clean_result_rows(recovery_raw, model)
    recovery = recovery[recovery["case_id"].isin(sensitive_ids)]
    recovery.to_csv(output_dir / "topk_recovery.csv", index=False)

    controls_raw = pd.read_csv(controls_path)
    controls = clean_result_rows(controls_raw, model)
    controls = controls[controls["case_id"].isin(sensitive_ids)]
    controls.to_csv(output_dir / "random_bottom_controls.csv", index=False)

    risk = pd.read_csv(risk_path).copy()
    risk = risk.drop(
        columns=[
            column
            for column in [
                "model_path",
                "source_file",
                "started_at",
                "finished_at",
                "notes",
            ]
            if column in risk.columns
        ]
    )
    if "risk_rank" in risk.columns and "rank" not in risk.columns:
        risk = risk.rename(columns={"risk_rank": "rank"})
    if "rank" not in risk.columns:
        risk = risk.sort_values("risk_score", ascending=False).reset_index(drop=True)
        risk["rank"] = np.arange(1, len(risk) + 1)
    risk.to_csv(output_dir / "risk_ranking.csv", index=False)
    write_first_success_and_efficiency(recovery, output_dir)
    write_provenance(
        output_dir,
        [sensitive_path, recovery_path, controls_path, risk_path],
        "Sanitized from case-level SIP experiment exports; paths, timestamps, full responses, and logs are excluded.",
    )


def inferred_ranking(recovery: pd.DataFrame, model: str) -> pd.DataFrame:
    row = recovery.sort_values("topk_k").dropna(subset=["protected_layers"]).iloc[-1]
    layers = [int(value) for value in re.findall(r"\d+", str(row["protected_layers"]))]
    return pd.DataFrame(
        {
            "model": model,
            "layer": layers,
            "rank": np.arange(1, len(layers) + 1),
            "risk_score": np.nan,
            "ranking_source": "policy_order_only",
        }
    )


def sanitize_xec_qwen25(
    source_dir: Path,
    output_dir: Path,
    model: str,
    risk_path: Path | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    full_path = source_dir / "tbgmp_full_single_model_results.csv"
    class_path = source_dir / "tbgmp_case_classification.csv"
    full = pd.read_csv(full_path)
    classification = pd.read_csv(class_path)

    discovery = clean_result_rows(full[full["stage"] == "discovery"], model)
    discovery.to_csv(output_dir / "stage_a_discovery.csv", index=False)

    sensitive = write_sensitive(
        classification,
        model,
        output_dir,
        "case_type",
    )
    sensitive_ids = set(sensitive["case_id"])

    topk = clean_result_rows(full[full["stage"] == "tbgmp_recovery"], model)
    topk = topk[topk["case_id"].isin(sensitive_ids)]
    topk.to_csv(output_dir / "topk_recovery.csv", index=False)
    controls = clean_result_rows(
        full[full["stage"] == "same_budget_control"],
        model,
    )
    controls = controls[controls["case_id"].isin(sensitive_ids)]
    controls.to_csv(output_dir / "random_bottom_controls.csv", index=False)

    sources = [full_path, class_path]
    if risk_path is not None:
        risk = pd.read_csv(risk_path).drop(
            columns=[
                column
                for column in ["source_file", "model_path"]
                if column in pd.read_csv(risk_path, nrows=1).columns
            ]
        )
        if "risk_rank" in risk.columns and "rank" not in risk.columns:
            risk = risk.rename(columns={"risk_rank": "rank"})
        if "rank" not in risk.columns:
            risk = risk.sort_values("risk_score", ascending=False).reset_index(drop=True)
            risk["rank"] = np.arange(1, len(risk) + 1)
        risk["ranking_source"] = "profiled_risk"
        sources.append(risk_path)
    else:
        risk = inferred_ranking(topk, model)
    risk.to_csv(output_dir / "risk_ranking.csv", index=False)
    write_first_success_and_efficiency(topk, output_dir)
    write_provenance(
        output_dir,
        sources,
        "Sanitized from case-level XEC experiment exports. Qwen2.5-14B ranking records policy order because a separate risk-score file was not available.",
    )


def sanitize_gemma(source_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "a": source_dir / "stage_a" / "stage_a_discovery_valid_falcon_gemma_xec.csv",
        "b": source_dir / "stage_b" / "table_gemma2_xec_sensitive_cases.csv",
        "c": source_dir / "stage_c_risk" / "gemma2_9b_key_risk_ranked_xec.csv",
        "d": source_dir / "stage_d_topk" / "stage_d_topk_recovery_falcon_gemma_xec.csv",
        "e": source_dir / "stage_e_controls" / "stage_e_controls_falcon_gemma_xec.csv",
    }
    stage_a_raw = pd.read_csv(paths["a"])
    stage_a = clean_result_rows(
        stage_a_raw[stage_a_raw["model_id"] == "gemma2_9b"],
        "Gemma2-9B",
    )
    stage_a.to_csv(output_dir / "stage_a_discovery.csv", index=False)

    classification = pd.read_csv(paths["b"]).rename(
        columns={
            "classification": "case_type",
            "answer_string": "needle_string",
            "depth": "needle_depth",
        }
    )
    classification["uniform_k2v2_found"] = classification["k2_found"]
    classification["uniform_k4v2_found"] = classification["k4_found"]
    sensitive = write_sensitive(
        classification,
        "Gemma2-9B",
        output_dir,
        "case_type",
    )
    sensitive_ids = set(sensitive["case_id"])

    stage_d_raw = pd.read_csv(paths["d"])
    topk = clean_result_rows(
        stage_d_raw[stage_d_raw["model_id"] == "gemma2_9b"],
        "Gemma2-9B",
    )
    topk = topk[topk["case_id"].isin(sensitive_ids)]
    topk.to_csv(output_dir / "topk_recovery.csv", index=False)

    controls_raw = pd.read_csv(paths["e"])
    controls = clean_result_rows(
        controls_raw[controls_raw["model_id"] == "gemma2_9b"],
        "Gemma2-9B",
    )
    controls = controls[controls["case_id"].isin(sensitive_ids)]
    controls.to_csv(output_dir / "random_bottom_controls.csv", index=False)

    risk = pd.read_csv(paths["c"]).drop(
        columns=[
            column
            for column in ["profile_token_limit"]
            if column in pd.read_csv(paths["c"], nrows=1).columns
        ]
    )
    risk.to_csv(output_dir / "risk_ranking.csv", index=False)
    write_first_success_and_efficiency(topk, output_dir)
    write_provenance(
        output_dir,
        list(paths.values()),
        "Sanitized from the Gemma2 XEC Stage A--E evidence package; host, partition, device, timestamps, errors, and full responses are excluded.",
    )


def write_model_audit(output_dir: Path) -> None:
    sensitive = pd.read_csv(output_dir / "sensitive_cases.csv")
    topk = pd.read_csv(output_dir / "topk_recovery.csv")
    restored = topk[topk["found"].astype(bool)]["case_id"].nunique()
    text = (
        f"# Case-Level Audit\n\n"
        f"- Sensitive cases: {len(sensitive)}\n"
        f"- Cases restored by at least one Top-k policy: {restored}/{len(sensitive)}\n"
        f"- Top-k rows: {len(topk)}\n"
        f"- Same-budget control rows: "
        f"{len(pd.read_csv(output_dir / 'random_bottom_controls.csv'))}\n"
        f"- Risk rows: {len(pd.read_csv(output_dir / 'risk_ranking.csv'))}\n\n"
        "Sensitive cases are selected from retrieval fields, not execution status.\n"
    )
    (output_dir / "audit_summary.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sanitize local case-level experiment exports into the repository schema."
    )
    parser.add_argument("--qwen3-dir", type=Path, required=True)
    parser.add_argument("--qwen3-risk", type=Path, required=True)
    parser.add_argument("--llama-dir", type=Path, required=True)
    parser.add_argument("--llama-risk", type=Path, required=True)
    parser.add_argument("--qwen25-3b-dir", type=Path, required=True)
    parser.add_argument("--qwen25-3b-risk", type=Path, required=True)
    parser.add_argument("--qwen25-14b-dir", type=Path, required=True)
    parser.add_argument("--gemma-package", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = {
        "qwen3_4b": ROOT / "results" / "main_evidence" / "qwen3_4b",
        "qwen25_3b": ROOT / "results" / "main_evidence" / "qwen25_3b",
        "qwen25_14b": ROOT / "results" / "main_evidence" / "qwen25_14b",
        "llama32_3b": ROOT / "results" / "main_evidence" / "llama32_3b",
        "gemma2_9b": ROOT / "results" / "supporting" / "gemma2_9b",
    }
    sanitize_sip_model(
        args.qwen3_dir,
        args.qwen3_risk,
        targets["qwen3_4b"],
        "Qwen3-4B",
    )
    sanitize_sip_model(
        args.llama_dir,
        args.llama_risk,
        targets["llama32_3b"],
        "Llama3.2-3B",
    )
    sanitize_xec_qwen25(
        args.qwen25_3b_dir,
        targets["qwen25_3b"],
        "Qwen2.5-3B",
        args.qwen25_3b_risk,
    )
    sanitize_xec_qwen25(
        args.qwen25_14b_dir,
        targets["qwen25_14b"],
        "Qwen2.5-14B",
        None,
    )
    sanitize_gemma(args.gemma_package, targets["gemma2_9b"])
    for target in targets.values():
        write_model_audit(target)
    print("Sanitized case-level release generated for four main models and Gemma2-9B.")


if __name__ == "__main__":
    main()
