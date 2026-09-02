from __future__ import annotations

from pathlib import Path

from experiments.run_full_pipeline import execute, protected_config
from tbgmp.backends.base import GenerationResult
from tbgmp.case_generation import generate_case_grid, stable_answer
from tbgmp.prompting import render_retrieval_prompt
from tbgmp.quantization import QuantizationConfig


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]

    def decode(self, tokens, skip_special_tokens=True):
        return "".join(chr(token) for token in tokens)


class ChatTokenizer(CharacterTokenizer):
    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    ):
        assert tokenize is False
        assert add_generation_prompt is True
        assert enable_thinking is False
        return "CHAT:" + "|".join(message["content"] for message in messages)


class CapturingBackend:
    def __init__(self):
        self.add_special_tokens = None

    def generate(self, **kwargs):
        self.add_special_tokens = kwargs["add_special_tokens"]
        return GenerationResult(
            response=kwargs["answer"],
            found=True,
            status="success",
            metadata={"actual_context_tokens": 240},
        )


class OOMBackend:
    def generate(self, **kwargs):
        raise RuntimeError("CUDA out of memory while allocating cache")


def test_protected_policy_inherits_k2_aggressive_bits() -> None:
    aggressive = QuantizationConfig(key_bits=2, value_bits=2, residual_window=128)
    config = protected_config(
        aggressive,
        {"protected_key_bits": 6},
        (25, 2, 18),
    )
    assert config.key_bits == 2
    assert config.value_bits == 2
    assert config.protected_key_bits == 6
    assert config.protected_layers == (25, 2, 18)


def test_protected_policy_inherits_k4_aggressive_bits() -> None:
    aggressive = QuantizationConfig(key_bits=4, value_bits=2, residual_window=64)
    config = protected_config(aggressive, {"protected_key_bits": 6}, (0,))
    assert config.key_bits == 4
    assert config.value_bits == 2
    assert config.residual_window == 64


def test_retrieval_prompt_uses_model_chat_template() -> None:
    prompt = render_retrieval_prompt(
        "context text",
        "question text",
        prompt_config={
            "system": "system text",
            "user_template": "CTX={context}; Q={question}",
            "use_chat_template": True,
        },
        tokenizer=ChatTokenizer(),
    )
    assert prompt == "CHAT:system text|CTX=context text; Q=question text"


def test_runner_marks_chat_prompt_as_already_rendered() -> None:
    backend = CapturingBackend()
    row = execute(
        backend=backend,
        case={
            "case_id": "case-1",
            "answer": "TBGMP-ANSWER",
            "document_tokens": 200,
        },
        prompt="already-rendered-chat-prompt",
        model_path="/models/example",
        model_id="example",
        policy_name="fp16",
        policy_type="fp16",
        quantization=None,
        max_new_tokens=16,
        seed=0,
        stage="stage_a_discovery",
    )
    assert backend.add_special_tokens is False
    assert row["document_tokens"] == 200
    assert row["actual_context_tokens"] == 240


def test_runner_records_oom_without_treating_it_as_retrieval_failure() -> None:
    row = execute(
        backend=OOMBackend(),
        case={"case_id": "case-oom", "answer": "answer"},
        prompt="prompt",
        model_path="/models/example",
        model_id="example",
        policy_name="uniform_k2_v2_rw128",
        policy_type="uniform",
        quantization=QuantizationConfig(key_bits=2, value_bits=2),
        max_new_tokens=16,
        seed=0,
        stage="stage_a_discovery",
    )
    assert row["status"] == "oom"
    assert row["oom"] is True
    assert row["completed"] is False
    assert row["found"] is False


def test_case_grid_is_config_driven_and_deterministic(tmp_path: Path) -> None:
    sources = {}
    for domain in ("math", "literature"):
        path = tmp_path / f"{domain}.txt"
        path.write_text(f"Public {domain} context. ", encoding="utf-8")
        sources[domain] = str(path)
    experiment = {
        "domains": ["math", "literature"],
        "context_lengths": [256],
        "needle_depths": [10, 90],
        "seeds": [0, 1],
    }
    generation = {
        "domain_sources": sources,
        "needle_template": "The hidden answer is {answer}.",
        "question": "Return the hidden answer.",
        "reserve_tokens": 32,
    }
    first = generate_case_grid(
        CharacterTokenizer(), experiment, generation, base_dir=tmp_path
    )
    second = generate_case_grid(
        CharacterTokenizer(), experiment, generation, base_dir=tmp_path
    )
    assert len(first) == 8
    assert first.equals(second)
    assert first["case_id"].is_unique
    assert first.iloc[0]["answer"] == stable_answer("math", 256, 10, 0)
    assert first["context"].str.contains("--- Internal Memo ---").all()
    assert (first["document_tokens"] <= 224).all()
    assert "actual_context_tokens" not in first.columns


def test_formal_case_grid_matches_recorded_hpc_protocol(tmp_path: Path) -> None:
    source = tmp_path / "literature.txt"
    source.write_text("0123456789" * 100, encoding="utf-8")
    experiment = {
        "domains": ["literature"],
        "context_lengths": [256],
        "needle_depths": [50],
        "seeds": [0, 1],
    }
    generation = {
        "protocol": "formal_hpc_v1",
        "domain_sources": {"literature": str(source)},
        "answers_by_seed": {0: "AURORA-7749", 1: "NEBULA-3186"},
        "reserve_tokens": 96,
        "question": "What is the secret project code name?",
    }
    first = generate_case_grid(
        CharacterTokenizer(), experiment, generation, base_dir=tmp_path
    )
    second = generate_case_grid(
        CharacterTokenizer(), experiment, generation, base_dir=tmp_path
    )
    assert first.equals(second)
    assert first["answer"].tolist() == ["AURORA-7749", "NEBULA-3186"]
    assert (first["generation_protocol"] == "formal_hpc_v1").all()
    assert first["context"].str.contains(
        "The secret project code name is ", regex=False
    ).all()
    assert first.iloc[0]["context"] != first.iloc[1]["context"]
