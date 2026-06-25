from __future__ import annotations


def found_answer(response: str, answer: str) -> bool:
    """Return whether a non-empty normalized answer occurs in the response."""
    normalized_answer = str(answer).strip()
    if not normalized_answer:
        return False
    return normalized_answer in str(response)


# Full model evaluation requires an external model runtime and a compatible
# quantized KV-cache implementation. This repository validates the analysis
# pipeline without downloading model weights.
