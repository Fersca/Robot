from __future__ import annotations


def build_chat_prompt(history: list[str], voice_config: dict) -> tuple[str, int]:
    system_prompt = str(voice_config.get("system_prompt", "")).strip()
    max_new_tokens = int(voice_config.get("max_new_tokens", 300))
    max_words_estimate = max(1, int(max_new_tokens * 0.65))
    system_limit_note = f"Anexo: No respondas con mas de {max_words_estimate} palabras."
    if system_prompt:
        effective_system_prompt = f"{system_prompt}\n{system_limit_note}"
        prompt = f"System: {effective_system_prompt}\n" + "\n".join(history) + "\nAssistant:"
    else:
        prompt = f"System: {system_limit_note}\n" + "\n".join(history) + "\nAssistant:"
    return prompt, max_new_tokens


def append_user_message(history: list[str], user_text: str) -> None:
    history.append(f"User: {user_text}")


def append_assistant_message(history: list[str], answer: str) -> None:
    history.append(f"Assistant: {answer}")
