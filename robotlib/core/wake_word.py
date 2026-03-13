from __future__ import annotations

import re
import unicodedata


def is_valid_auto_listen_transcript(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(text or "").strip().lower())
    words = [w for w in re.split(r"\s+", normalized) if w]
    if not words:
        return False
    if len(words) >= 4:
        return True
    return "hola" in words


def spoken_phrase_to_words(text: str) -> list[str]:
    raw_words = re.split(r"[\s,.;:!?¡¿()\[\]{}\"“”'`´\-_/\\]+", str(text or "").strip().lower())
    words: list[str] = []
    for raw in raw_words:
        cleaned = "".join(ch for ch in raw if ch.isalnum())
        if not cleaned:
            continue
        normalized = unicodedata.normalize("NFKD", cleaned)
        without_accents = "".join(ch for ch in normalized if not unicodedata.combining(ch))
        if without_accents:
            words.append(without_accents)
    return words


def transcript_matches_phrase(text: str, phrase: str) -> bool:
    text_words = spoken_phrase_to_words(text)
    phrase_words = spoken_phrase_to_words(phrase)
    if not text_words or not phrase_words:
        return False
    return text_words == phrase_words


def should_run_auto_listen_worker(config: dict) -> bool:
    return bool(config.get("auto_listen_enabled", False) or config.get("wake_word_enabled", False))


def apply_wake_word_transcript(
    text: str,
    config: dict,
) -> tuple[bool, str | None]:
    if not bool(config.get("wake_word_enabled", False)):
        return False, None
    wake_phrase = str(config.get("wake_word_phrase", "hola robot")).strip()
    stop_phrase = str(config.get("wake_word_stop_phrase", "adios robot")).strip()
    auto_active = bool(config.get("auto_listen_enabled", False))
    if not auto_active:
        if transcript_matches_phrase(text, wake_phrase):
            config["auto_listen_enabled"] = True
            return True, str(config.get("wake_word_on_response", "Te escucho.")).strip() or None
        return True, None
    if transcript_matches_phrase(text, stop_phrase):
        config["auto_listen_enabled"] = False
        return True, str(config.get("wake_word_off_response", "Modo escucha desactivado.")).strip() or None
    return False, None

