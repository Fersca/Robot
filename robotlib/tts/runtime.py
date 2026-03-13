from __future__ import annotations

import queue


def vision_event_tts_worker(camera_runtime: dict, *, speak_text_backend, print_error_red, is_audio_playback_active, is_tts_active) -> None:
    stop_event = camera_runtime.get("stop_event")
    event_queue = camera_runtime.get("vision_event_queue")
    if stop_event is None or not isinstance(event_queue, queue.Queue):
        return
    while not stop_event.is_set() or not event_queue.empty():
        try:
            message = event_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        if not message:
            continue
        config = camera_runtime.get("voice_config")
        tts_runtime = camera_runtime.get("tts_runtime")
        speaker = camera_runtime.get("speaker")
        if not isinstance(config, dict) or not bool(config.get("audio_enabled", True)):
            continue
        if not isinstance(tts_runtime, dict):
            continue
        if is_audio_playback_active() or is_tts_active():
            continue
        try:
            speak_text_backend(
                speaker,
                str(message),
                config,
                tts_runtime,
                allow_interrupt=True,
            )
        except Exception as exc:
            print_error_red(f"ERROR: Vision event TTS failed: {exc}")


def emit_vision_event_message(camera_runtime: dict, message: str, *, is_audio_playback_active, is_tts_active) -> None:
    if not message:
        return
    if is_audio_playback_active() or is_tts_active():
        return
    event_queue = camera_runtime.get("vision_event_queue")
    if not isinstance(event_queue, queue.Queue):
        return
    try:
        event_queue.put_nowait(str(message))
    except queue.Full:
        pass


def interrupt_audio_and_speak(
    camera_runtime: dict,
    message: str,
    *,
    request_audio_cancel,
    is_audio_playback_active,
    is_tts_active,
    clear_audio_cancel,
    sleep_fn,
    monotonic_fn,
    speak_text_backend,
    print_error_red,
) -> None:
    if not message:
        return
    config = camera_runtime.get("voice_config")
    tts_runtime = camera_runtime.get("tts_runtime")
    speaker = camera_runtime.get("speaker")
    if not isinstance(config, dict) or not bool(config.get("audio_enabled", True)):
        return
    if not isinstance(tts_runtime, dict):
        return
    request_audio_cancel()
    deadline = monotonic_fn() + 1.0
    while monotonic_fn() < deadline and (is_audio_playback_active() or is_tts_active()):
        sleep_fn(0.02)
    clear_audio_cancel()
    try:
        speak_text_backend(
            speaker,
            str(message),
            config,
            tts_runtime,
            allow_interrupt=True,
        )
    except Exception as exc:
        print_error_red(f"ERROR: Vision interrupt TTS failed: {exc}")
