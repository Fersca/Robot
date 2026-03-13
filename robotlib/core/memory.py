from __future__ import annotations


def build_effective_system_prompt(system_prompt: str, max_new_tokens: int) -> str:
    max_words_estimate = max(1, int(max_new_tokens * 0.65))
    system_limit_note = f"Anexo: No respondas con mas de {max_words_estimate} palabras."
    system_prompt = str(system_prompt or "").strip()
    return f"{system_prompt}\n{system_limit_note}" if system_prompt else system_limit_note


def render_llm_context(history: list[str], system_prompt: str, max_new_tokens: int) -> str:
    memory = [str(item).strip() for item in list(history or []) if str(item).strip()]
    effective_system_prompt = build_effective_system_prompt(system_prompt, max_new_tokens)
    lines = ["", "Current LLM context:", "", "System:", "-----", effective_system_prompt, "-----", ""]
    lines.append(f"Memory entries: {len(memory)}")
    if not memory:
        lines.append("(history is empty)")
        lines.append("")
        return "\n".join(lines)
    for idx, item in enumerate(memory, 1):
        lines.append(f"{idx:>3}. {item}")
    lines.append("")
    return "\n".join(lines)

