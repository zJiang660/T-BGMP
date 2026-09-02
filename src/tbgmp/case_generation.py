from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any

import pandas as pd


def stable_answer(domain: str, context_length: int, depth: int, seed: int) -> str:
    payload = f"{domain}|{context_length}|{depth}|{seed}".encode("utf-8")
    suffix = hashlib.sha256(payload).hexdigest()[:8].upper()
    return f"TBGMP-{suffix}"


def formal_answer(seed: int, answers_by_seed: dict[Any, Any]) -> str:
    for key in (seed, str(seed)):
        if key in answers_by_seed:
            return str(answers_by_seed[key])
    raise ValueError(f"formal protocol has no hidden answer for seed {seed}")


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


def build_formal_needle_context(
    tokenizer,
    source_text: str,
    *,
    context_length: int,
    depth: int,
    seed: int,
    answer: str,
    reserve_tokens: int = 96,
) -> tuple[str, int]:
    """Reproduce the context construction used by the original HPC runner."""
    if not 0 <= depth <= 100:
        raise ValueError("needle depth must be between 0 and 100")
    cleaned = "\n".join(
        line.strip()
        for line in source_text.replace("\r", "\n").splitlines()
        if line.strip()
    )
    source_tokens = tokenizer.encode(cleaned, add_special_tokens=False)
    if not source_tokens:
        raise ValueError("domain source must produce at least one token")
    if len(source_tokens) < context_length:
        source_tokens *= context_length // len(source_tokens) + 2

    needle = f"The secret project code name is {answer}."
    needle_tokens = tokenizer.encode(needle, add_special_tokens=False)
    rng = random.Random(
        1000003 + int(seed) * 1009 + int(context_length) * 37 + int(depth) * 17
    )
    offset = rng.randint(0, min(97, max(0, len(source_tokens) - 1)))
    source_tokens = source_tokens[offset:] + source_tokens[:offset]
    budget = max(1, int(context_length) - len(needle_tokens) - int(reserve_tokens))
    position = max(0.0, min(1.0, float(depth) / 100.0))
    before_length = max(0, min(budget, int(budget * position) + rng.randint(-8, 8)))
    after_length = budget - before_length
    before = tokenizer.decode(source_tokens[:before_length], skip_special_tokens=True)
    after = tokenizer.decode(
        source_tokens[before_length : before_length + after_length],
        skip_special_tokens=True,
    )
    context = (
        f"{before}\n\n--- Internal Memo ---\n{needle}\n"
        f"--- End Memo ---\n\n{after}"
    )
    actual_tokens = len(tokenizer.encode(context, add_special_tokens=False))
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
    protocol = str(generation.get("protocol", "deterministic_demo_v1"))
    answers_by_seed = generation.get("answers_by_seed", {})
    rows: list[dict[str, Any]] = []
    for domain in domains:
        source_path = Path(str(sources[domain]))
        if not source_path.is_absolute():
            source_path = base_dir / source_path
        source_text = source_path.read_text(encoding="utf-8", errors="ignore")
        for context_length in experiment.get("context_lengths", []):
            for depth in experiment.get("needle_depths", []):
                for seed in experiment.get("seeds", []):
                    if protocol == "formal_hpc_v1":
                        answer = formal_answer(int(seed), answers_by_seed)
                        context, document_tokens = build_formal_needle_context(
                            tokenizer,
                            source_text,
                            context_length=int(context_length),
                            depth=int(depth),
                            seed=int(seed),
                            answer=answer,
                            reserve_tokens=reserve_tokens,
                        )
                    elif protocol == "deterministic_demo_v1":
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
                    else:
                        raise ValueError(f"unsupported case-generation protocol: {protocol}")
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
                            "generation_protocol": protocol,
                        }
                    )
    return pd.DataFrame(rows)
