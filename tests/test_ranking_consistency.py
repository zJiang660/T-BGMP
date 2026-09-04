from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "qwen3_4b": "qwen3_4b",
    "qwen25_3b": "qwen25_3b",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ranking_from_csv(model_dir: str) -> tuple[Path, list[int]]:
    path = ROOT / "results" / "main_evidence" / model_dir / "risk_ranking.csv"
    rows = pd.read_csv(path).sort_values("rank")
    assert len(rows) == len(set(rows["layer"]))
    assert list(rows["rank"].astype(int)) == list(range(1, len(rows) + 1))
    return path, list(rows["layer"].astype(int))


def longest_policy_prefix(model_dir: str) -> list[int]:
    rows = pd.read_csv(
        ROOT / "results" / "main_evidence" / model_dir / "topk_recovery.csv"
    )
    prefixes = [
        [int(layer) for layer in ast.literal_eval(value)]
        for value in rows["protected_layers"].dropna().astype(str)
        if value.strip() not in {"", "[]"}
    ]
    return max(prefixes, key=len)


def test_main_recovery_uses_authoritative_ranking_prefix() -> None:
    for model_dir in MODELS.values():
        _, ranking = ranking_from_csv(model_dir)
        prefix = longest_policy_prefix(model_dir)
        assert prefix == ranking[: len(prefix)]


def test_frozen_and_ruler_configs_pin_authoritative_rankings() -> None:
    frozen = json.loads(
        (ROOT / "configs" / "extensions" / "frozen_top3.json").read_text()
    )
    ruler = json.loads(
        (ROOT / "configs" / "extensions" / "ruler.json").read_text()
    )
    for config_key, model_dir in MODELS.items():
        path, ranking = ranking_from_csv(model_dir)
        digest = sha256(path)

        frozen_model = frozen["models"][config_key]
        assert frozen_model["ranking"] == ranking
        assert frozen_model["ranking_source_sha256"] == digest
        assert frozen_model["frozen_top_layers"] == ranking[: frozen_model["k_star"]]

        ruler_model = ruler["models"][config_key]["ranking"]
        assert ruler_model["ranking"] == ranking
        assert ruler_model["top12"] == ranking[:12]
        assert ruler_model["source_sha256"] == digest
