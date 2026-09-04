from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments" / "extensions" / "run_domain_heldout.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("domain_heldout_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_heldout_profile_uses_normalized_full_score() -> None:
    runner = load_runner()
    rows = runner.finalize_profile_rows(
        [
            {"layer": 0, "mse_p95": 10.0, "ip_p95": 1.0, "effective_dim": 10.0},
            {"layer": 1, "mse_p95": 1.0, "ip_p95": 100.0, "effective_dim": 5.0},
            {"layer": 2, "mse_p95": 2.0, "ip_p95": 2.0, "effective_dim": 1.0},
        ]
    )

    assert [row["rank"] for row in rows] == [1, 2, 3]
    assert {row["score_protocol"] for row in rows} == {
        runner.PROFILE_SCORE_PROTOCOL
    }
    assert {row["layer"] for row in rows} == {0, 1, 2}
    assert rows[0]["layer"] == 2


def test_heldout_runner_has_no_legacy_proxy_profiler() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "uniform_quantize_dequantize" not in source
    assert "mse + math.log1p" not in source
    assert "layer_distortion_metrics(" in source
