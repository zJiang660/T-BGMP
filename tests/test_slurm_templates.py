from __future__ import annotations

from pathlib import Path

from scripts.validate_slurm_templates import ROOT, validate_template


def templates() -> list[Path]:
    return sorted(
        path
        for path in (ROOT / "slurm").rglob("*")
        if path.suffix in {".sbatch", ".slurm"}
    )


def test_all_slurm_references_exist() -> None:
    failures = {
        str(path.relative_to(ROOT)): validate_template(path)
        for path in templates()
        if validate_template(path)
    }
    assert not failures


def test_formal_templates_use_the_canonical_resumable_runner() -> None:
    formal = [
        ROOT / "slurm" / site / "submit_full_pipeline_a800_template.sbatch"
        for site in ("xec", "sip")
    ]
    for path in formal:
        text = path.read_text(encoding="utf-8")
        assert "experiments/run_full_pipeline.py" in text
        assert '"${CASES_FILE}"' in text
        assert '"${RISK_RANKING}"' in text
        assert '"${OUTPUT_FILE}"' in text
        assert "logs/" not in text
