from __future__ import annotations

from typing import Any


DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant. Answer concisely."
DEFAULT_USER_TEMPLATE = "Read this document:\n\n{context}\n\n{question}"


def build_messages(
    context: str,
    question: str,
    prompt_config: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    config = prompt_config or {}
    system = str(config.get("system", DEFAULT_SYSTEM_PROMPT))
    template = str(config.get("user_template", DEFAULT_USER_TEMPLATE))
    user = template.format(context=context, question=question)
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return messages


def render_messages_text(messages: list[dict[str, str]]) -> str:
    parts = [message["content"] for message in messages if message.get("content")]
    return "\n\n".join(parts)


def apply_chat_template(tokenizer, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except (AttributeError, ValueError):
        return render_messages_text(messages) + "\nAnswer:"


def render_retrieval_prompt(
    context: str,
    question: str,
    *,
    prompt_config: dict[str, Any] | None = None,
    tokenizer=None,
) -> str:
    messages = build_messages(context, question, prompt_config)
    use_chat_template = bool((prompt_config or {}).get("use_chat_template", True))
    if tokenizer is not None and use_chat_template:
        return apply_chat_template(tokenizer, messages)
    return render_messages_text(messages)
