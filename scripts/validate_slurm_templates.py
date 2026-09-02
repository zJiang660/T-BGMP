from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SLURM_ROOT = ROOT / "slurm"
REPO_REFERENCE = re.compile(
    r"(?<![/A-Za-z0-9_.-])((?:experiments|scripts|configs)/[A-Za-z0-9_./-]+)"
)


def validate_template(path: Path) -> list[str]:
    errors: list[str] = []
    text = path.read_text(encoding="utf-8")
    if not text.startswith("#!/usr/bin/env bash"):
        errors.append("missing portable bash shebang")
    if "set -euo pipefail" not in text:
        errors.append("missing strict shell mode")
    if "#SBATCH --gres=gpu:1" not in text:
        errors.append("missing single-GPU request")
    for reference in REPO_REFERENCE.findall(text):
        candidate = ROOT / reference
        if not candidate.exists():
            errors.append(f"missing repository target: {reference}")
    return errors


def main() -> None:
    templates = sorted(
        path
        for path in SLURM_ROOT.rglob("*")
        if path.suffix in {".sbatch", ".slurm"}
    )
    if not templates:
        raise SystemExit("No SLURM templates found")
    failures = []
    for path in templates:
        errors = validate_template(path)
        label = path.relative_to(ROOT)
        if errors:
            failures.extend(f"{label}: {error}" for error in errors)
            print(f"FAIL {label}")
        else:
            print(f"PASS {label}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"SLURM template validation: PASS ({len(templates)} files)")


if __name__ == "__main__":
    main()
