from __future__ import annotations

from tbgmp.kv_cache_wrapper import GenerationResult


class FakeTokenizer:
    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    ):
        return "\n".join(message["content"] for message in messages)


class FakeBackend:
    """Deterministic backend used to exercise the full pipeline without a GPU."""

    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.tokenizer_requests = 0
        self.generate_requests = 0

    def get_tokenizer(self, model_path):
        self.tokenizer_requests += 1
        return self.tokenizer

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
        add_special_tokens=True,
    ) -> GenerationResult:
        self.generate_requests += 1
        if policy_name == "fp16" or policy_name.startswith("uniform_k6_"):
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
