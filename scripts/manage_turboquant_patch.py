#!/usr/bin/env python3
"""Check or apply the pinned T-BGMP TurboQuant patch safely."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tbgmp.turboquant_patch_validation import validate_runtime_contract


def _run_git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_patch_state(root: Path, patch: Path) -> dict[str, Any]:
    forward = _run_git(root, "apply", "--check", str(patch))
    reverse = _run_git(root, "apply", "--reverse", "--check", str(patch))
    if forward.returncode == 0:
        state = "ready_to_apply"
    elif reverse.returncode == 0:
        state = "already_applied"
    else:
        state = "incompatible"
    return {
        "state": state,
        "forward_error": forward.stderr.strip(),
        "reverse_error": reverse.stderr.strip(),
    }


def validate_imported_runtime(root: Path) -> dict[str, Any]:
    root_text = str(root.resolve())
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    compressors = importlib.import_module("turboquant.compressors_v3")
    generation = importlib.import_module("turboquant.generation_test")
    return validate_runtime_contract(compressors.TurboQuantV3, generation.V3Cache)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turboquant-root", type=Path, required=True)
    parser.add_argument(
        "--patch",
        type=Path,
        default=REPO_ROOT / "patches" / "turboquant_arbitrary_protected_layers.patch",
    )
    parser.add_argument("--lock", type=Path, default=REPO_ROOT / "configs" / "runtime_lock.yaml")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--skip-runtime-validation", action="store_true")
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.turboquant_root.expanduser().resolve()
    patch = args.patch.expanduser().resolve()
    lock = yaml.safe_load(args.lock.read_text(encoding="utf-8"))
    spec = lock["external_runtime"]["turboquant"]

    head_result = _run_git(root, "rev-parse", "HEAD")
    actual_commit = head_result.stdout.strip() if head_result.returncode == 0 else "UNAVAILABLE"
    expected_commit = str(spec["upstream_commit"])
    actual_patch_hash = _sha256(patch) if patch.is_file() else "MISSING"
    expected_patch_hash = str(spec["patch_sha256"])
    tracked_status = _run_git(root, "status", "--porcelain", "--untracked-files=no")
    tracked_clean = tracked_status.returncode == 0 and not tracked_status.stdout.strip()

    report: dict[str, Any] = {
        "upstream_commit": actual_commit,
        "expected_upstream_commit": expected_commit,
        "commit_ok": actual_commit == expected_commit,
        "patch_sha256": actual_patch_hash,
        "expected_patch_sha256": expected_patch_hash,
        "patch_hash_ok": actual_patch_hash == expected_patch_hash,
        "tracked_worktree_clean": tracked_clean,
    }
    state = inspect_patch_state(root, patch) if patch.is_file() else {"state": "missing_patch"}
    report.update(state)

    prerequisites_ok = report["commit_ok"] and report["patch_hash_ok"]
    if args.apply:
        if not prerequisites_ok:
            report["apply"] = "refused"
        elif state["state"] == "already_applied":
            report["apply"] = "already_applied"
        elif state["state"] != "ready_to_apply" or not tracked_clean:
            report["apply"] = "refused"
        else:
            applied = _run_git(root, "apply", str(patch))
            report["apply"] = "success" if applied.returncode == 0 else "failed"
            report["apply_error"] = applied.stderr.strip()
            report.update(inspect_patch_state(root, patch))

    patched = report.get("state") == "already_applied"
    if patched and not args.skip_runtime_validation:
        try:
            report["runtime_contract"] = validate_imported_runtime(root)
        except Exception as exc:  # pragma: no cover - external dependency failure
            report["runtime_contract"] = {"passed": False, "error": repr(exc)}
    elif patched:
        report["runtime_contract"] = {"passed": None, "skipped": True}

    runtime_ok = report.get("runtime_contract", {}).get("passed")
    report["passed"] = prerequisites_ok and patched and (
        runtime_ok is True or args.skip_runtime_validation
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

