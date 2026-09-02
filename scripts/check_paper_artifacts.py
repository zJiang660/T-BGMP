from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "PAPER_RESULTS_MANIFEST.yaml"
REQUIRED_PAPER_ITEMS = {"table_1", "table_2", "figure_2", *(f"table_{i}" for i in range(3, 8))}
REQUIRED_TEXT_ITEMS = {
    "fixed_layer_check",
    "cross_seed_heldout",
    "additional_models",
    "gemma2_incomplete_key_only_recovery",
}
PATH_SUFFIXES = {".csv", ".json", ".yaml", ".yml", ".md", ".py"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS {message}")


def walk_values(value: Any, key: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for child_key, child in value.items():
            yield from walk_values(child, str(child_key))
    elif isinstance(value, list):
        for child in value:
            yield from walk_values(child, key)
    else:
        yield key, value


def referenced_paths(manifest: dict[str, Any]) -> set[Path]:
    paths: set[Path] = set()
    for key, value in walk_values(manifest):
        if not isinstance(value, str) or key.endswith("command"):
            continue
        candidate = Path(value)
        if candidate.suffix.lower() in PATH_SUFFIXES and not value.startswith(("http://", "https://")):
            paths.add(candidate)
    return paths


def referenced_scripts(manifest: dict[str, Any]) -> set[Path]:
    scripts: set[Path] = set()
    for key, value in walk_values(manifest):
        if not key.endswith("command") or not isinstance(value, str):
            continue
        tokens = shlex.split(value)
        for token in tokens:
            if token.endswith(".py"):
                scripts.add(Path(token))
    return scripts


def check_manifest(manifest_path: Path = MANIFEST_PATH) -> None:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    paper_items = manifest.get("paper_items", {})
    text_items = manifest.get("paper_text_items", {})

    require(set(paper_items) == REQUIRED_PAPER_ITEMS, "Table 1--7 and Figure 2 mappings are complete")
    require(REQUIRED_TEXT_ITEMS <= set(text_items), "all final-paper text result mappings are present")

    missing_paths = sorted(str(path) for path in referenced_paths(manifest) if not (ROOT / path).is_file())
    require(not missing_paths, f"all manifest file paths exist (missing={missing_paths})")
    missing_scripts = sorted(str(path) for path in referenced_scripts(manifest) if not (ROOT / path).is_file())
    require(not missing_scripts, f"all manifest analysis commands exist (missing={missing_scripts})")

    authoritative = [Path(value).as_posix() for value in paper_items.values()]
    require(len(authoritative) == len(set(authoritative)), "paper table/figure authoritative paths are unique")

    additional = text_items["additional_models"]
    supporting = pd.read_csv(ROOT / additional["summary"])
    require(
        set(additional["included_rows"]) <= set(supporting["model"]),
        "all final-paper additional-model rows exist",
    )

    manifest_text = manifest_path.read_text(encoding="utf-8").lower()
    require("value-bottleneck" not in manifest_text, "obsolete value-bottleneck claim is absent")
    require("kivi" not in manifest_text and "kvquant" not in manifest_text, "removed quantizer experiments are absent from final-paper manifest")
    print("Paper artifact consistency: PASS")


if __name__ == "__main__":
    check_manifest()

