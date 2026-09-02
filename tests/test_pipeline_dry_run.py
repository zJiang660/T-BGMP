from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def run_command(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )


def test_run_full_pipeline_dry_run(tmp_path) -> None:
    result = run_command(
        "experiments/run_full_pipeline.py",
        "--dry-run",
        "--backend",
        "turboquant",
        "--model-key",
        "qwen25_3b",
        "--model-root",
        "/path/to/models",
        "--turboquant-root",
        "/path/to/turboquant-pytorch",
        "--output-dir",
        str(tmp_path),
    )
    assert "Stage A: discovery" in result.stdout
    assert "No model execution performed in dry-run mode." in result.stdout
    assert (tmp_path / "full_pipeline_results.csv").exists()


def test_stage_a_help() -> None:
    result = run_command("experiments/stage_a_discovery.py", "--help")
    assert "--backend" in result.stdout
    assert "--dry-run" in result.stdout


def test_stage_c_demo_must_be_explicit() -> None:
    result = subprocess.run(
        [sys.executable, "experiments/stage_c_profile_key_risk.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "explicit --demo-stats flag" in result.stderr


def test_smoke_help() -> None:
    result = run_command("experiments/smoke_test_backend.py", "--help")
    assert "--model-key" in result.stdout
    assert "--backend" in result.stdout


def test_full_pipeline_resumes_and_materializes_stage_f(tmp_path) -> None:
    output = tmp_path / "full.csv"
    command = (
        "experiments/run_full_pipeline.py",
        "--backend",
        "tests.fake_backend:create_backend",
        "--model-path",
        "fake-model",
        "--model-id",
        "fake-model",
        "--cases",
        "data/demo/full_runner_cases.csv",
        "--risk-ranking",
        "results/audit/demo_key_risk_ranking.csv",
        "--output",
        str(output),
        "--maximum-topk",
        "4",
        "--checkpoint-every",
        "2",
    )
    first = run_command(*command)
    journal = output.with_suffix(".jsonl")
    first_attempts = len(journal.read_text(encoding="utf-8").splitlines())
    second = run_command(*command)
    second_attempts = len(journal.read_text(encoding="utf-8").splitlines())

    assert "new attempts=15" in first.stdout
    assert "new attempts=0" in second.stdout
    assert first_attempts == second_attempts == 15
    rows = pd.read_csv(output)
    assert len(rows) == 15
    stage_f = pd.read_csv(tmp_path / "full_stage_f.csv")
    assert len(stage_f) == 2
    assert stage_f["comparison_valid"].all()
    summary = pd.read_csv(tmp_path / "full_stage_f_summary.csv")
    assert int(summary.iloc[0]["valid_pairs"]) == 2
    manifest_path = output.with_suffix(".run.json")
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "2.0"
    assert manifest["latest_status"] == "complete"
    assert len(manifest["invocations"]) == 2
    assert all(
        invocation["status"] == "complete"
        for invocation in manifest["invocations"]
    )
    assert manifest["repository"]["commit"]
    assert manifest["input_artifacts"]
    assert manifest["model_identity"]["model_id"] == "fake-model"
    serialized = json.dumps(manifest)
    assert str(tmp_path) not in serialized
    assert "<OUTPUT_PATH>" in serialized

    changed = subprocess.run(
        [sys.executable, *command, "--maximum-topk", "3"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert changed.returncode != 0
    assert "different experiment signature" in changed.stderr
