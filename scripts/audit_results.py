from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "results" / "paper_tables"
AUDIT_DIR = ROOT / "results" / "audit"

MAIN_MODELS = {"Qwen3-4B", "Qwen2.5-3B", "Qwen2.5-14B", "Llama3.2-3B"}


def parse_fraction(value: str) -> tuple[int, int]:
    numerator, denominator = str(value).split("/", maxsplit=1)
    return int(numerator), int(denominator)


def normalized_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path).fillna("").astype(str)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    checks: dict[str, bool] = {}

    main_df = pd.read_csv(PAPER_DIR / "table_main_evidence.csv")
    restored_pairs = main_df["recovered"].map(parse_fraction)
    restored = sum(pair[0] for pair in restored_pairs)
    denominators = sum(pair[1] for pair in restored_pairs)
    sensitive = int(main_df["sensitive"].sum())
    checks["main_model_set_exact"] = set(main_df["model"]) == MAIN_MODELS
    checks["main_sensitive_total_183"] = sensitive == 183
    checks["main_restored_183_of_183"] = restored == denominators == 183

    supporting = pd.read_csv(PAPER_DIR / "table_supporting_models.csv")
    checks["supporting_not_main"] = not bool(set(supporting["model"]) & MAIN_MODELS)
    gemma_support = supporting[supporting["model"] == "Gemma2-9B"]
    checks["gemma_boundary_supporting"] = (
        len(gemma_support) == 1
        and "Boundary-supporting" in str(gemma_support.iloc[0]["usage"])
        and "value bottleneck" in str(gemma_support.iloc[0]["usage"]).lower()
    )

    gemma = pd.read_csv(PAPER_DIR / "table_gemma2_boundary.csv").set_index("policy")
    checks["gemma_key_only_7_of_72"] = gemma.loc["Key-only Top1--Top12", "found"] == "7/72 unique"
    checks["gemma_k6_v2_7_of_72"] = gemma.loc["Uniform K6/V2", "found"] == "7/72"
    checks["gemma_k6_v4_72_of_72"] = gemma.loc["Uniform K6/V4", "found"] == "72/72"

    boundary = pd.read_csv(PAPER_DIR / "table_boundary_models.csv")
    checks["excluded_classified_not_retrieval_failure"] = all(
        "limited" in value.lower() or "invalid" in value.lower()
        for value in boundary["classification"].astype(str)
    )

    demo = pd.read_csv(ROOT / "data" / "demo" / "demo_expected_outputs.csv")
    checks["demo_has_one_sensitive_case"] = (
        demo["selected_sensitive"].astype(bool).sum() == 1
        and demo.loc[demo["selected_sensitive"].astype(bool), "case_id"].iloc[0]
        == "math_sensitive"
    )
    checks["demo_topk_beats_controls"] = (
        demo["topk_recovered"].astype(bool).sum() == 1
        and demo["random_recovered"].astype(bool).sum() == 0
        and demo["bottom_recovered"].astype(bool).sum() == 0
    )

    copies = {
        "table_main_evidence.csv": ROOT / "results" / "main_evidence",
        "table_first_success_k.csv": ROOT / "results" / "main_evidence",
        "table_qwen25_scale.csv": ROOT / "results" / "main_evidence",
        "table_control_statistics.csv": ROOT / "results" / "supporting",
        "table_supporting_models.csv": ROOT / "results" / "supporting",
        "table_gemma2_boundary.csv": ROOT / "results" / "boundary_excluded",
        "table_boundary_models.csv": ROOT / "results" / "boundary_excluded",
    }
    for filename, directory in copies.items():
        canonical = normalized_csv(PAPER_DIR / filename)
        grouped = normalized_csv(directory / filename)
        checks[f"grouped_copy_matches_{Path(filename).stem}"] = canonical.equals(grouped)

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    tracked_results = sorted(
        path
        for path in (ROOT / "results").glob("**/*.csv")
        if AUDIT_DIR not in path.parents
    )
    hash_rows = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in tracked_results
    ]
    pd.DataFrame(hash_rows).to_csv(AUDIT_DIR / "artifact_hashes.csv", index=False)

    checks = {name: bool(value) for name, value in checks.items()}
    summary = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "main_sensitive": sensitive,
        "main_restored": restored,
        "main_denominator": denominators,
        "checks": checks,
    }
    (AUDIT_DIR / "audit_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    print(f"Main conditional aggregate: {restored}/{denominators}")
    print(f"Audit artifacts: {AUDIT_DIR.relative_to(ROOT)}")
    if not all(checks.values()):
        raise SystemExit("Result audit failed")
    print("Result audit: PASS")


if __name__ == "__main__":
    main()
