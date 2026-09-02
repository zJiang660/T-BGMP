from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tbgmp.provenance import (
    fingerprint_inputs,
    model_identity,
    sanitized_argv,
    scheduler_context,
    summarize_numeric,
)


def test_input_and_model_fingerprints_are_content_addressed(tmp_path: Path) -> None:
    case_file = tmp_path / "cases.csv"
    case_file.write_text("case_id,answer\na,x\n", encoding="utf-8")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    config = b'{"model_type":"test","_commit_hash":"abc123"}'
    (model_dir / "config.json").write_bytes(config)
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer": "model-00001.safetensors"}}),
        encoding="utf-8",
    )

    fingerprints = fingerprint_inputs({"case_file": case_file})
    assert fingerprints == [
        {
            "label": "case_file",
            "filename": "cases.csv",
            "bytes": case_file.stat().st_size,
            "sha256": hashlib.sha256(case_file.read_bytes()).hexdigest(),
        }
    ]
    identity = model_identity("test-model", model_dir)
    assert identity["snapshot_revision"] == "abc123"
    assert identity["identity_files"][0]["sha256"] == hashlib.sha256(
        config
    ).hexdigest()
    assert str(tmp_path) not in json.dumps(identity)


def test_command_and_scheduler_provenance_are_sanitized() -> None:
    argv = sanitized_argv(
        [
            "run.py",
            "--model-path",
            "/private/models/qwen",
            "--output=/private/results/run.csv",
            "--seed",
            "7",
        ]
    )
    assert argv == [
        "run.py",
        "--model-path",
        "<MODEL_PATH>",
        "--output=<OUTPUT_PATH>",
        "--seed",
        "7",
    ]
    scheduler = scheduler_context(
        {
            "SLURM_JOB_ID": "123",
            "SLURM_JOB_PARTITION": "gpu",
            "SLURM_JOB_NODELIST": "private-node",
        }
    )
    assert scheduler == {"job_id": "123", "partition": "gpu"}


def test_numeric_summary_ignores_missing_values() -> None:
    assert summarize_numeric(["4096", 8192, "", None, "invalid"]) == {
        "count": 2,
        "min": 4096,
        "max": 8192,
        "unique": [4096, 8192],
    }
