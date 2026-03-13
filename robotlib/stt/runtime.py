from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import gc
import os
import time


@dataclass(slots=True)
class STTRuntimeDeps:
    ensure_dependency: object
    ov_genai: object
    load_whisper_ov_models: object
    mark_runtime_chip_compat: object
    is_downloaded: object
    download_whisper_ov_model: object
    save_robot_config: object
    whisper_ov_device_options: tuple[str, ...]
    print_error_red: object
    default_whisper_model: str
    whisper_models: tuple[str, ...]
    whisper_local_model_path: object
    resolve_ov_whisper_language: object
    resolve_audio_input_device: object
    record_until_space: object
    whisper_language_options: tuple[str, ...]


def ensure_stt_runtime(stt_runtime: dict, config: dict, deps: STTRuntimeDeps) -> bool:
    compat = stt_runtime.get("compat")

    def reset_loaded_models() -> None:
        stt_runtime["model"] = None
        stt_runtime["model_name"] = None
        stt_runtime["ov_pipeline"] = None
        stt_runtime["ov_model_id"] = None
        stt_runtime["ov_model_dir"] = None
        stt_runtime["backend"] = None

    if stt_runtime.get("numpy") is None:
        stt_runtime["numpy"] = deps.ensure_dependency("numpy", "numpy", "NumPy")
        if stt_runtime["numpy"] is None:
            return False
    if stt_runtime.get("sounddevice") is None:
        stt_runtime["sounddevice"] = deps.ensure_dependency("sounddevice", "sounddevice", "SoundDevice")
        if stt_runtime["sounddevice"] is None:
            return False

    use_ov = bool(config.get("whisper_openvino", False))
    if use_ov:
        requested_device = str(config.get("whisper_openvino_device", "AUTO")).strip().upper()
        selected_id_for_key = str(config.get("whisper_openvino_model_id", "")).strip() or "unknown"
        compat_key = f"stt:whisper_ov:{selected_id_for_key}"
        if deps.ov_genai is None:
            deps.print_error_red(
                "ERROR: whisper_openvino is enabled but openvino_genai is not installed "
                "in this environment."
            )
            deps.mark_runtime_chip_compat(compat, compat_key, requested_device, False)
            return False
        ov_models = deps.load_whisper_ov_models()
        if not ov_models:
            deps.print_error_red("ERROR: Whisper OpenVINO is enabled but no OV models are configured.")
            deps.mark_runtime_chip_compat(compat, compat_key, requested_device, False)
            return False

        selected_id = str(config.get("whisper_openvino_model_id", "")).strip()
        selected = next((m for m in ov_models if m["id"] == selected_id), None)
        if selected is None:
            selected = ov_models[0]
            config["whisper_openvino_model_id"] = selected["id"]
            deps.save_robot_config(config)
        compat_key = f"stt:whisper_ov:{selected['id']}"

        ov_device = str(config.get("whisper_openvino_device", "AUTO")).strip().upper()
        if ov_device not in deps.whisper_ov_device_options:
            ov_device = "AUTO"
            config["whisper_openvino_device"] = ov_device
            deps.save_robot_config(config)

        key = ("ov", selected["id"], ov_device)
        if stt_runtime.get("active_key") != key or stt_runtime.get("ov_pipeline") is None:
            if stt_runtime.get("active_key") is not None:
                print("Releasing previous STT model...")
                reset_loaded_models()
                stt_runtime["active_key"] = None
                gc.collect()
            try:
                if not deps.is_downloaded(selected["local"]):
                    deps.download_whisper_ov_model(selected)
                print(f"Loading Whisper OpenVINO model '{selected['display']}' on {ov_device}...")
                stt_runtime["ov_pipeline"] = deps.ov_genai.WhisperPipeline(str(selected["local"]), ov_device)
                stt_runtime["ov_model_id"] = selected["id"]
                stt_runtime["ov_model_dir"] = str(selected["local"])
                stt_runtime["whisper"] = None
                stt_runtime["backend"] = "ov"
                stt_runtime["active_key"] = key
                deps.mark_runtime_chip_compat(compat, compat_key, ov_device, True)
                print(f"✅ Whisper OV model active: {selected['id']} ({ov_device})\n")
            except Exception as exc:
                deps.print_error_red(f"ERROR: Failed to initialize Whisper OpenVINO: {exc}")
                deps.mark_runtime_chip_compat(compat, compat_key, ov_device, False)
                reset_loaded_models()
                stt_runtime["active_key"] = None
                return False
        else:
            stt_runtime["backend"] = "ov"
        return True

    if stt_runtime.get("whisper") is None:
        stt_runtime["whisper"] = deps.ensure_dependency("whisper", "openai-whisper", "Whisper")
        if stt_runtime["whisper"] is None:
            return False

    model_name = str(config.get("whisper_model", deps.default_whisper_model))
    compat_key = f"stt:whisper:{model_name}"
    if model_name not in deps.whisper_models:
        model_name = deps.default_whisper_model
        config["whisper_model"] = model_name
        deps.save_robot_config(config)
        compat_key = f"stt:whisper:{model_name}"

    key = ("whisper", model_name)
    if stt_runtime.get("active_key") != key or stt_runtime.get("model") is None:
        if stt_runtime.get("active_key") is not None:
            print("Releasing previous STT model...")
            reset_loaded_models()
            stt_runtime["active_key"] = None
            gc.collect()

        local_model = deps.whisper_local_model_path(stt_runtime["whisper"], model_name)
        if local_model and os.path.exists(local_model):
            print(f"Whisper model '{model_name}' already downloaded.")
        else:
            print(f"Whisper model '{model_name}' is not downloaded. Downloading...")
        print(f"Loading Whisper model '{model_name}'...")
        try:
            stt_runtime["model"] = stt_runtime["whisper"].load_model(model_name)
            stt_runtime["model_name"] = model_name
            stt_runtime["backend"] = "whisper"
            stt_runtime["active_key"] = key
            deps.mark_runtime_chip_compat(compat, compat_key, "CPU", True)
            print(f"✅ Whisper model active: {model_name}\n")
        except Exception as exc:
            print(f"\n⚠️ Failed to load Whisper model '{model_name}': {exc}\n")
            deps.mark_runtime_chip_compat(compat, compat_key, "CPU", False)
            reset_loaded_models()
            stt_runtime["active_key"] = None
            return False
    return True


def transcribe_from_mic(stt_runtime: dict, config: dict, deps: STTRuntimeDeps, transcribe_audio_buffer_fn) -> tuple[str, float]:
    if not ensure_stt_runtime(stt_runtime, config, deps):
        return "", 0.0
    np_mod = stt_runtime["numpy"]
    device = deps.resolve_audio_input_device(config, stt_runtime["sounddevice"])
    audio, speech_end_ts = deps.record_until_space(stt_runtime["sounddevice"], np_mod, device=device)
    if getattr(audio, "size", 0) == 0:
        print("\n⚠️ No audio captured.\n")
        return "", 0.0
    return transcribe_audio_buffer_fn(stt_runtime, config, audio, speech_end_ts)


def transcribe_audio_buffer(stt_runtime: dict, config: dict, audio, deps: STTRuntimeDeps, speech_end_ts: float | None = None) -> tuple[str, float]:
    if not ensure_stt_runtime(stt_runtime, config, deps):
        return "", 0.0
    if getattr(audio, "size", 0) == 0:
        return "", 0.0
    speech_end_ts = speech_end_ts if speech_end_ts is not None else time.perf_counter()
    is_ov = stt_runtime.get("backend") == "ov"
    whisper_language = str(config.get("whisper_language", "es")).strip().lower()
    if whisper_language not in deps.whisper_language_options:
        whisper_language = "es"
    print("📝 Transcribing with Whisper OpenVINO..." if is_ov else "📝 Transcribing...")
    try:
        if is_ov:
            ov_kwargs = {"task": "transcribe"}
            if whisper_language != "auto":
                model_dir = Path(str(stt_runtime.get("ov_model_dir", "")).strip()) if stt_runtime.get("ov_model_dir") else None
                resolved_lang = whisper_language
                warning = None
                if model_dir is not None and model_dir.exists():
                    resolved_lang, warning = deps.resolve_ov_whisper_language(model_dir, whisper_language)
                if warning:
                    deps.print_error_red(f"ERROR: {warning}")
                if resolved_lang:
                    ov_kwargs["language"] = resolved_lang
            result = stt_runtime["ov_pipeline"].generate(audio.tolist(), **ov_kwargs)
            text = ""
            if hasattr(result, "text"):
                text = str(getattr(result, "text", "")).strip()
            elif hasattr(result, "texts"):
                texts = getattr(result, "texts", [])
                text = str(texts[0]).strip() if texts else ""
            if not text:
                text = str(result).strip()
        else:
            whisper_kwargs = {"fp16": False, "task": "transcribe"}
            if whisper_language != "auto":
                whisper_kwargs["language"] = whisper_language
            result = stt_runtime["model"].transcribe(audio, **whisper_kwargs)
            text = str(result.get("text", "")).strip()
    except Exception as exc:
        deps.print_error_red(f"ERROR: STT transcription failed: {exc}")
        return "", time.perf_counter() - speech_end_ts
    speech_end_to_text_s = time.perf_counter() - speech_end_ts
    if not text:
        print("\n⚠️ No speech detected.\n")
        return "", speech_end_to_text_s
    print(f"You said: {text}\n")
    return text, speech_end_to_text_s
