from __future__ import annotations

import csv
import json

from scripts.convert_raw_outputs_to_case_csv import convert


def test_converter_recomputes_found_and_drops_full_response(tmp_path) -> None:
    raw_path = tmp_path / "raw.jsonl"
    response = "The hidden answer is ABC123."
    raw_path.write_text(
        json.dumps(
            {
                "model": "demo",
                "domain": "smoke",
                "context_length": 32,
                "depth": 0,
                "seed": 0,
                "policy": "fp16",
                "key_bits": 16,
                "value_bits": 16,
                "residual_window": 0,
                "answer": "ABC123",
                "response": response,
                "status": "success",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rows = convert(raw_path, excerpt_chars=12)
    assert rows[0]["found"] is True
    assert rows[0]["response_excerpt"] == response[:12]
    assert "response" not in rows[0]


def test_converter_output_shape_via_csv(tmp_path) -> None:
    raw_path = tmp_path / "raw.jsonl"
    output_path = tmp_path / "case.csv"
    raw_path.write_text(
        json.dumps(
            {
                "model": "demo",
                "domain": "smoke",
                "context_length": 32,
                "depth": 0,
                "seed": 0,
                "policy": "fp16",
                "key_bits": 16,
                "value_bits": 16,
                "residual_window": 0,
                "answer": "ABC123",
                "response": "ABC123",
                "status": "success",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rows = convert(raw_path, excerpt_chars=200)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert output_path.read_text(encoding="utf-8").startswith("model,case_id")
