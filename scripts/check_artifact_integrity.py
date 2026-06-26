from __future__ import annotations

import csv
import importlib.util
import re
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_SIZE = 25 * 1024 * 1024
WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".ckpt"}
REQUIRED_FILES = [
    "README.md",
    "ARTIFACT_EVALUATION.md",
    "CLAIMS_TO_ARTIFACTS.md",
    "REPRODUCE.md",
    "PAPER_RESULTS_MANIFEST.yaml",
    "VERSION_LOCK.md",
    "CITATION.cff",
    "NOTICE.md",
    "LICENSE",
    "docs/data_provenance.md",
    "docs/model_setup.md",
    "docs/backend_integration.md",
    "docs/smoke_test.md",
    "results/paper_tables/table_main_evidence.csv",
    "results/paper_tables/table_control_statistics.csv",
    "results/paper_tables/table_gemma2_boundary.csv",
    "examples/smoke_test/SMOKE_TEST_SUMMARY.md",
    "examples/smoke_test/smoke_raw_example.sanitized.jsonl",
    "examples/smoke_test/smoke_case_level_example.csv",
    "scripts/audit_results.py",
]
SENSITIVE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"/gpfs/work",
        r"C:\\Users",
        r"[A-Za-z]:\\\\",
        r"zhenghaojiang23",
        r"ziqizhao23",
        r"Halli",
        r"XJTLU",
        r"Xi'an",
        r"Liverpool",
        r"OneDrive",
        r"gmail",
        r"outlook",
        r"ghp_",
        r"private_key",
        r"password\s*=",
        r"(^|[^A-Za-z_])token\s*=",
        r"hf_[A-Za-z0-9]",
    ]
]
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".venv"}
SENSITIVE_SCAN_ALLOWLIST = {
    Path(".gitignore"),
    Path("scripts/check_artifact_integrity.py"),
}


def iter_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file():
            files.append(path)
    return files


def fail(message: str) -> None:
    print(f"FAIL {message}")
    raise SystemExit(1)


def check_required_files() -> None:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).is_file()]
    if missing:
        fail(f"missing required files: {missing}")


def check_audit_import() -> None:
    path = ROOT / "scripts" / "audit_results.py"
    spec = importlib.util.spec_from_file_location("audit_results_check", path)
    if spec is None or spec.loader is None:
        fail("audit_results.py cannot be imported")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def check_manifest() -> None:
    manifest = yaml.safe_load((ROOT / "PAPER_RESULTS_MANIFEST.yaml").read_text())
    aggregate = manifest["main_evidence"]["aggregate"]
    if aggregate["sensitive_cases"] != 183 or aggregate["restored"] != 183:
        fail("manifest aggregate is not 183/183")


def check_main_table() -> None:
    path = ROOT / "results" / "paper_tables" / "table_main_evidence.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    restored = 0
    sensitive = 0
    for row in rows:
        sensitive += int(row["sensitive"])
        left, right = row["recovered"].split("/")
        restored += int(left)
        if int(left) != int(right):
            fail(f"model is not fully restored: {row['model']}")
    if sensitive != 183 or restored != 183:
        fail(f"main aggregate is {restored}/{sensitive}, expected 183/183")


def check_smoke_examples() -> None:
    smoke_dir = ROOT / "examples" / "smoke_test"
    if not smoke_dir.is_dir():
        fail("examples/smoke_test is missing")
    raw_text = (smoke_dir / "smoke_raw_example.sanitized.jsonl").read_text(
        encoding="utf-8"
    )
    if "response_excerpt" not in raw_text:
        fail("smoke raw example does not use response_excerpt")
    if '"response":' in raw_text:
        fail("smoke raw example contains full response field")


def check_mimo_absent(files: list[Path]) -> None:
    for path in files:
        rel = path.relative_to(ROOT)
        if rel in SENSITIVE_SCAN_ALLOWLIST:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"mimo|mimo7b", text, re.IGNORECASE):
            fail(f"MiMo appears in public artifact file: {rel}")


def check_file_safety(files: list[Path]) -> None:
    for path in files:
        rel = path.relative_to(ROOT)
        if path.suffix.lower() in WEIGHT_SUFFIXES:
            fail(f"model/checkpoint-like file found: {rel}")
        if path.stat().st_size > MAX_FILE_SIZE:
            fail(f"file over 25 MB found: {rel}")
        if rel in SENSITIVE_SCAN_ALLOWLIST:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in SENSITIVE_PATTERNS:
            if pattern.search(text):
                fail(f"sensitive pattern found in {rel}: {pattern.pattern}")


def main() -> None:
    files = iter_files()
    check_required_files()
    check_audit_import()
    check_manifest()
    check_main_table()
    check_smoke_examples()
    check_mimo_absent(files)
    check_file_safety(files)
    print("ARTIFACT INTEGRITY: PASS")


if __name__ == "__main__":
    main()
