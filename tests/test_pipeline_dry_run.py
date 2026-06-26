from __future__ import annotations

import subprocess
import sys
from pathlib import Path


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


def test_smoke_help() -> None:
    result = run_command("experiments/smoke_test_backend.py", "--help")
    assert "--model-key" in result.stdout
    assert "--backend" in result.stdout
