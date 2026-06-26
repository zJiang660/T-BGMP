from __future__ import annotations

from tbgmp.backends.base import GenerationRequest, GenerationResult
from tbgmp.backends.turboquant_backend import TurboQuantBackend


def test_backend_dataclasses_importable() -> None:
    request = GenerationRequest(
        model_path="/path/to/models/example",
        prompt="question",
        answer="answer",
        policy={"name": "fp16"},
    )
    result = GenerationResult(response="answer", found=True, status="success")
    assert request.answer == "answer"
    assert result.found is True


def test_turboquant_check_available_is_informative_without_root() -> None:
    backend = TurboQuantBackend(turboquant_root=None)
    status = backend.check_available()
    assert status["backend"] == "turboquant"
    assert status["ready_for_tbgmp_generation"] is False
    assert status["arbitrary_protected_key_layer_ids"] is False
    assert "message" in status
