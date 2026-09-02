from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd


def stable_answer(domain: str, context_length: int, depth: int, seed: int) -> str:
    payload = f"{domain}|{context_length}|{depth}|{seed}".encode("utf-8")
    suffix = hashlib.sha256(payload).hexdigest()[:8].upper()
    return f"TBGMP-{suffix}"


def repeat_to_length(tokens: list[int], length: int) -> list[int]:
    if length <= 0:
        return []
    if not tokens:
        raise ValueError("domain source must produce at least one token")
    repeats = (length + len(tokens) - 1) // len(tokens)
    return (tokens * repeats)[:length]


def build_needle_context(
    tokenizer,
    source_text: str,
    *,
    context_length: int,
    depth: int,
    answer: str,
    needle_template: str = "The hidden answer is {answer}.",
    reserve_tokens: int = 64,
) -> tuple[str, int]:
    if not 0 <= depth <= 100:
        raise ValueError("needle depth must be between 0 and 100")
    source_tokens = tokenizer.encode(source_text, add_special_tokens=False)
    needle = needle_template.format(answer=answer)
    target_document_tokens = int(context_length) - int(reserve_tokens)
    if target_document_tokens <= 0:
        raise ValueError("reserve_tokens must be smaller than context_length")

    def render(filler_tokens: list[int]) -> tuple[str, int]:
        before_length = int(len(filler_tokens) * (float(depth) / 100.0))
        before = tokenizer.decode(
            filler_tokens[:before_length], skip_special_tokens=True
        )
        after = tokenizer.decode(
            filler_tokens[before_length:], skip_special_tokens=True
        )
        text = (
            f"{before}\n\n--- Internal Memo ---\n{needle}\n"
            f"--- End Memo ---\n\n{after}"
        )
        token_count = len(tokenizer.encode(text, add_special_tokens=False))
        return text, token_count

    _, fixed_tokens = render([])
    if fixed_tokens > target_document_tokens:
        raise ValueError(
            "needle and separators exceed the document token budget; "
            "increase context_length or reduce reserve_tokens"
        )
    filler = repeat_to_length(source_tokens, target_document_tokens - fixed_tokens)
    context, actual_tokens = render(filler)
    while actual_tokens > target_document_tokens and filler:
        overflow = max(1, actual_tokens - target_document_tokens)
        filler = filler[:-overflow]
        context, actual_tokens = render(filler)
    return context, actual_tokens


def generate_case_grid(
    tokenizer,
    experiment: dict[str, Any],
    generation: dict[str, Any],
    *,
    base_dir: Path,
) -> pd.DataFrame:
    sources = generation.get("domain_sources", {})
    domains = list(experiment.get("domains", []))
    missing_sources = [domain for domain in domains if domain not in sources]
    if missing_sources:
        raise ValueError(f"missing domain sources: {missing_sources}")

    question = str(
        generation.get(
            "question", "What is the hidden answer? Reply with the exact string."
        )
    )
    needle_template = str(
        generation.get("needle_template", "The hidden answer is {answer}.")
    )
    reserve_tokens = int(generation.get("reserve_tokens", 64))
    rows: list[dict[str, Any]] = []
    for domain in domains:
        source_path = Path(str(sources[domain]))
        if not source_path.is_absolute():
            source_path = base_dir / source_path
        source_text = source_path.read_text(encoding="utf-8", errors="ignore")
        for context_length in experiment.get("context_lengths", []):
            for depth in experiment.get("needle_depths", []):
                for seed in experiment.get("seeds", []):
                    answer = stable_answer(domain, context_length, depth, seed)
                    context, document_tokens = build_needle_context(
                        tokenizer,
                        source_text,
                        context_length=int(context_length),
                        depth=int(depth),
                        answer=answer,
                        needle_template=needle_template,
                        reserve_tokens=reserve_tokens,
                    )
                    rows.append(
                        {
                            "case_id": (
                                f"{domain}_ctx{context_length}_d{depth}_s{seed}"
                            ),
                            "domain": domain,
                            "context_length": int(context_length),
                            "document_tokens": document_tokens,
                            "depth": int(depth),
                            "seed": int(seed),
                            "context": context,
                            "question": question,
                            "answer": answer,
                        }
                    )
    return pd.DataFrame(rows)
