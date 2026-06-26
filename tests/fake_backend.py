from __future__ import annotations

from tbgmp.kv_cache_wrapper import GenerationResult


class FakeBackend:
    """Deterministic backend used to exercise the full pipeline without a GPU."""

    def generate(
        self,
        *,
        model_path,
        prompt,
        answer,
        policy_name,
        quantization,
        max_new_tokens,
        seed=0,
    ) -> GenerationResult:
        if policy_name == "fp16":
            response = answer
        elif policy_name.startswith("tbgmp_top") and len(
            quantization.protected_layers
        ) >= 2:
            response = answer
        else:
            response = "not found"
        return GenerationResult(
            response=response,
            found=False,
            status="success",
            runtime_s=0.01,
            tok_per_s=100.0,
            peak_gpu_gb=0.0,
            kv_saving=80.0,
        )


def create_backend():
    return FakeBackend()
