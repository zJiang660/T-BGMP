#!/usr/bin/env python3
"""Build frozen RULER conditional sets from completed screening checkpoints."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(os.environ.get("TBGMP_RULER_ROOT", Path.cwd())).expanduser().resolve()
MANIFEST = ROOT / "manifests" / "RULER_FORMAL_CASE_MANIFEST.json"
CHECKPOINTS = ROOT / "outputs" / "screening" / "checkpoints"
OUT = ROOT / "conditionals"
MODELS = {
    "qwen3_4b": "Qwen3_RULER_CONDITIONAL.json",
    "qwen25_3b": "Qwen25_RULER_CONDITIONAL.json",
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, path)


def main() -> None:
    manifest = load_json(MANIFEST)
    all_summary = {}
    for model_id, filename in MODELS.items():
        model_manifest = [row for row in manifest if row["model"] == model_id]
        if len(model_manifest) != 300:
            raise RuntimeError(f"{model_id} manifest has {len(model_manifest)} rows, expected 300")
        rows = [load_json(path) for path in sorted((CHECKPOINTS / model_id).glob("*.json"))]
        keys = [(row["sample_id"], row["policy_group"]) for row in rows]
        if len(keys) != len(set(keys)):
            raise RuntimeError(f"Duplicate screening checkpoints for {model_id}")
        expected = {(row["sample_id"], group) for row in model_manifest for group in ("fp16", "aggressive")}
        found = set(keys)
        missing = sorted(expected - found)
        extra = sorted(found - expected)
        if missing or extra:
            raise RuntimeError(
                f"Incomplete screening for {model_id}: rows={len(rows)} missing={len(missing)} extra={len(extra)}"
            )
        by_sample = defaultdict(dict)
        for row in rows:
            by_sample[row["sample_id"]][row["policy_group"]] = row
        manifest_by_id = {row["sample_id"]: row for row in model_manifest}
        conditional = []
        invalid = []
        for sample_id in sorted(manifest_by_id):
            fp16 = by_sample[sample_id]["fp16"]
            aggressive = by_sample[sample_id]["aggressive"]
            if not fp16["valid_execution"] or not aggressive["valid_execution"]:
                invalid.append({
                    "sample_id": sample_id,
                    "fp16_error": fp16.get("error", ""),
                    "aggressive_error": aggressive.get("error", ""),
                    "fp16_oom": fp16.get("oom", False),
                    "aggressive_oom": aggressive.get("oom", False),
                })
                continue
            if fp16["full_credit_success"] and not aggressive["full_credit_success"]:
                case = dict(manifest_by_id[sample_id])
                case["screening"] = {
                    "fp16_soft_score": fp16["official_soft_score"],
                    "aggressive_soft_score": aggressive["official_soft_score"],
                    "fp16_checkpoint_job": fp16.get("slurm_job_id", ""),
                    "aggressive_checkpoint_job": aggressive.get("slurm_job_id", ""),
                }
                conditional.append(case)
        strata = Counter((row["task"], int(row["context_length"])) for row in conditional)
        payload = {
            "model": model_id,
            "definition": "FP16 full-credit AND aggressive not full-credit AND both valid",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "total_screened_cases": len(model_manifest),
            "conditional_case_count": len(conditional),
            "insufficient_failure_population": len(conditional) < 5,
            "invalid_execution_count": len(invalid),
            "strata": [
                {"task": task, "context_length": length, "conditional_cases": strata[(task, length)]}
                for task in ("niah_multikey_1", "vt", "fwe") for length in (4096, 8192)
            ],
            "cases": conditional,
            "invalid_executions": invalid,
        }
        path = OUT / filename
        write_json_atomic(path, payload)
        if not conditional:
            marker = OUT / f"{model_id}_NO_CONDITIONAL_CASES"
            marker.write_text("NO_CONDITIONAL_CASES\n", encoding="utf-8")
        all_summary[model_id] = {
            "conditional_cases": len(conditional),
            "invalid_executions": len(invalid),
            "insufficient_failure_population": len(conditional) < 5,
            "path": str(path),
        }
    write_json_atomic(OUT / "RULER_CONDITIONAL_BUILD_SUMMARY.json", all_summary)
    print(json.dumps(all_summary, indent=2))


if __name__ == "__main__":
    main()

