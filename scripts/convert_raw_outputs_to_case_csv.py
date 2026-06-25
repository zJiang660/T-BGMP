from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_FIELDS = {
    "model",
    "domain",
    "context_length",
    "depth",
    "seed",
    "policy",
    "key_bits",
    "value_bits",
    "residual_window",
    "answer",
    "response",
    "status",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert model-run JSONL into a compact case-level CSV. Full "
            "responses and runtime metadata are not copied."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-excerpt-chars", type=int, default=200)
    return parser.parse_args()


def response_excerpt(response: str, limit: int) -> str:
    compact = re.sub(r"\s+", " ", str(response)).strip()
    return compact[:limit]


def found_answer(response: str, answer: str) -> bool:
    normalized = str(answer).strip()
    return bool(normalized) and normalized in str(response)


def make_case_id(row: dict[str, Any], line_number: int) -> str:
    if row.get("case_id"):
        return str(row["case_id"])
    return "_".join(
        [
            str(row["model"]),
            str(row["domain"]),
            f"ctx{row['context_length']}",
            f"depth{row['depth']}",
            f"seed{row['seed']}",
            f"line{line_number}",
        ]
    )


def convert(path: Path, excerpt_chars: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            missing = sorted(REQUIRED_FIELDS - set(raw))
            if missing:
                raise ValueError(
                    f"Line {line_number} is missing required fields: {missing}"
                )
            response = str(raw["response"])
            answer = str(raw["answer"])
            rows.append(
                {
                    "model": raw["model"],
                    "case_id": make_case_id(raw, line_number),
                    "domain": raw["domain"],
                    "context_length": raw["context_length"],
                    "actual_context_length": raw.get(
                        "actual_context_length", raw["context_length"]
                    ),
                    "depth": raw["depth"],
                    "seed": raw["seed"],
                    "answer": answer,
                    "policy": raw["policy"],
                    "policy_type": raw.get("policy_type", ""),
                    "key_bits": raw["key_bits"],
                    "value_bits": raw["value_bits"],
                    "protected_key_bits": raw.get("protected_key_bits", ""),
                    "protected_layers": raw.get("protected_layers", []),
                    "residual_window": raw["residual_window"],
                    "found": found_answer(response, answer),
                    "status": raw["status"],
                    "response_excerpt": response_excerpt(response, excerpt_chars),
                    "runtime_s": raw.get("runtime_s", ""),
                    "tok_per_s": raw.get("tok_per_s", ""),
                    "peak_gpu_gb": raw.get("peak_gpu_gb", ""),
                    "kv_saving": raw.get("kv_saving", ""),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    if args.max_excerpt_chars < 0:
        raise SystemExit("--max-excerpt-chars must be non-negative")
    rows = convert(args.input, args.max_excerpt_chars)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Converted {len(rows)} JSONL row(s) -> {args.output}")


if __name__ == "__main__":
    main()
