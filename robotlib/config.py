from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RobotConfigEnv:
    robot_config_file: Path
    platform_name: str
    default_device: str
    default_performance_hint: str
    default_whisper_model: str
    default_tts_backend: str
    whisper_ov_device_options: list[str]
    whisper_language_options: list[str]
    openvino_tts_device_options: list[str]
    kokoro_device_options: list[str]
    babelvox_device_options: list[str]
    babelvox_precision_options: list[str]


def build_default_robot_config(env: RobotConfigEnv) -> dict:
    return {
        "voice_index": 0,
        "rate": -2,
        "volume": 100,
        "silence": 600,
        "whisper_model": env.default_whisper_model,
        "whisper_language": "es",
        "repeat": False,
        "current_model_repo": "",
        "llm_backend": "local",
        "llm_device": env.default_device,
        "llm_performance_hint": env.default_performance_hint,
        "external_llm_base_url": "http://localhost:1234",
        "external_llm_model": "",
        "external_llm_api_key": "",
        "audio_enabled": True,
        "audio_input_device": "",
        "audio_monitor_enabled": False,
        "visual_effects_enabled": True,
        "panel_backend": "opencv",
        "camera_enabled": False,
        "camera_device_index": 0,
        "vision_enabled": False,
        "vision_model_id": "",
        "vision_model_path": "",
        "vision_labels_path": "",
        "vision_device": "AUTO",
        "vision_threshold": 0.4,
        "vision_log_enabled": False,
        "vision_log_interval_s": 1.0,
        "vision_event_processing_enabled": True,
        "auto_listen_enabled": False,
        "wake_word_enabled": False,
        "wake_word_phrase": "hola robot",
        "wake_word_stop_phrase": "adios robot",
        "wake_word_on_response": "Te escucho.",
        "wake_word_off_response": "Modo escucha desactivado.",
        "auto_listen_aggressiveness": 3,
        "auto_listen_threshold": 0.50,
        "auto_listen_frame_ms": 32,
        "auto_listen_preroll_ms": 350,
        "auto_listen_min_speech_ms": 1400,
        "auto_listen_silence_ms": 1600,
        "auto_listen_max_segment_s": 60.0,
        "auto_listen_resume_delay_ms": 1500,
        "auto_listen_min_segment_ms": 1400,
        "auto_listen_min_voiced_ratio": 0.60,
        "tts_streaming_enabled": False,
        "tts_stream_min_words": 12,
        "tts_stream_cut_on_punctuation": False,
        "system_prompt": "",
        "max_new_tokens": 300,
        "warmup_tts": True,
        "whisper_openvino": False,
        "whisper_openvino_device": "AUTO",
        "whisper_openvino_model_id": "",
        "tts_backend": env.default_tts_backend,
        "openvino_tts_device": "AUTO",
        "openvino_tts_model_id": "",
        "openvino_tts_timeout_s": 25,
        "openvino_tts_isolated_gpu": True,
        "openvino_tts_speed": 1.0,
        "openvino_tts_gain": 1.0,
        "kokoro_model_id": "kokoro-tts-intel",
        "kokoro_device": "GPU",
        "kokoro_voice": "af_sarah",
        "babelvox_model_id": "babelvox-openvino-int8",
        "babelvox_device": "CPU",
        "babelvox_precision": "int8",
        "babelvox_language": "es",
        "tada_model_id": "HumeAI/tada-1b",
        "tada_codec_id": "HumeAI/tada-codec",
        "tada_device": "cpu",
        "tada_language": "en",
        "tada_reference_audio_path": "",
        "tada_reference_text": "",
        "tada_sample_rate": 24000,
        "espeak_voice": "es",
        "espeak_rate": 145,
        "espeak_pitch": 45,
        "espeak_amplitude": 120,
    }


def load_robot_config(
    env: RobotConfigEnv,
    normalize_tts_backend_for_platform,
) -> tuple[dict, str | None]:
    cfg = build_default_robot_config(env)
    warning = None
    if env.robot_config_file.exists():
        try:
            data = json.loads(env.robot_config_file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                if "repeat" not in data and "repetir" in data:
                    data["repeat"] = data.get("repetir", False)
                data.pop("repetir", None)
                cfg.update(data)
        except Exception as exc:
            warning = f"Could not read {env.robot_config_file}: {exc}"
    cfg["whisper_openvino"] = bool(cfg.get("whisper_openvino", False))
    llm_backend = str(cfg.get("llm_backend", "local")).strip().lower()
    cfg["llm_backend"] = llm_backend if llm_backend in {"local", "external"} else "local"
    cfg["external_llm_base_url"] = str(cfg.get("external_llm_base_url", "http://localhost:1234")).strip() or "http://localhost:1234"
    cfg["external_llm_model"] = str(cfg.get("external_llm_model", "")).strip()
    cfg["external_llm_api_key"] = str(cfg.get("external_llm_api_key", "")).strip()
    cfg["audio_input_device"] = str(cfg.get("audio_input_device", "")).strip()
    cfg["audio_monitor_enabled"] = bool(cfg.get("audio_monitor_enabled", False))
    cfg["visual_effects_enabled"] = bool(cfg.get("visual_effects_enabled", True))
    panel_backend = str(cfg.get("panel_backend", "opencv")).strip().lower()
    cfg["panel_backend"] = panel_backend if panel_backend in {"opencv", "qt"} else "opencv"
    cfg["camera_enabled"] = bool(cfg.get("camera_enabled", False))
    try:
        cfg["camera_device_index"] = max(0, int(cfg.get("camera_device_index", 0)))
    except Exception:
        cfg["camera_device_index"] = 0
    cfg["vision_enabled"] = bool(cfg.get("vision_enabled", False))
    cfg["vision_model_id"] = str(cfg.get("vision_model_id", "")).strip()
    cfg["vision_model_path"] = str(cfg.get("vision_model_path", "")).strip()
    cfg["vision_labels_path"] = str(cfg.get("vision_labels_path", "")).strip()
    vision_device = str(cfg.get("vision_device", "AUTO")).strip().upper() or "AUTO"
    cfg["vision_device"] = vision_device if vision_device in {"CPU", "GPU", "NPU", "AUTO"} else "AUTO"
    try:
        cfg["vision_threshold"] = float(cfg.get("vision_threshold", 0.4))
    except Exception:
        cfg["vision_threshold"] = 0.4
    cfg["vision_threshold"] = min(1.0, max(0.0, cfg["vision_threshold"]))
    cfg["vision_log_enabled"] = bool(cfg.get("vision_log_enabled", False))
    try:
        cfg["vision_log_interval_s"] = max(0.1, float(cfg.get("vision_log_interval_s", 1.0)))
    except Exception:
        cfg["vision_log_interval_s"] = 1.0
    cfg["vision_event_processing_enabled"] = bool(cfg.get("vision_event_processing_enabled", True))
    cfg["auto_listen_enabled"] = bool(cfg.get("auto_listen_enabled", False))
    cfg["wake_word_enabled"] = bool(cfg.get("wake_word_enabled", False))
    cfg["wake_word_phrase"] = str(cfg.get("wake_word_phrase", "hola robot")).strip() or "hola robot"
    cfg["wake_word_stop_phrase"] = str(cfg.get("wake_word_stop_phrase", "adios robot")).strip() or "adios robot"
    cfg["wake_word_on_response"] = str(cfg.get("wake_word_on_response", "Te escucho.")).strip() or "Te escucho."
    cfg["wake_word_off_response"] = str(cfg.get("wake_word_off_response", "Modo escucha desactivado.")).strip() or "Modo escucha desactivado."
    try:
        cfg["auto_listen_aggressiveness"] = min(3, max(0, int(cfg.get("auto_listen_aggressiveness", 3))))
    except Exception:
        cfg["auto_listen_aggressiveness"] = 3
    try:
        cfg["auto_listen_threshold"] = float(cfg.get("auto_listen_threshold", 0.50))
    except Exception:
        cfg["auto_listen_threshold"] = 0.50
    cfg["auto_listen_threshold"] = min(0.99, max(0.01, cfg["auto_listen_threshold"]))
    try:
        frame_ms = int(cfg.get("auto_listen_frame_ms", 32))
    except Exception:
        frame_ms = 32
    cfg["auto_listen_frame_ms"] = frame_ms if frame_ms in {32, 64, 96} else 32
    for key, default, minimum, caster in (
        ("auto_listen_preroll_ms", 300, 0, int),
        ("auto_listen_min_speech_ms", 1400, 30, int),
        ("auto_listen_silence_ms", 900, 100, int),
        ("auto_listen_resume_delay_ms", 1500, 0, int),
        ("auto_listen_min_segment_ms", 1400, 100, int),
    ):
        try:
            cfg[key] = max(minimum, caster(cfg.get(key, default)))
        except Exception:
            cfg[key] = default
    try:
        cfg["auto_listen_max_segment_s"] = max(1.0, float(cfg.get("auto_listen_max_segment_s", 15.0)))
    except Exception:
        cfg["auto_listen_max_segment_s"] = 15.0
    try:
        cfg["auto_listen_min_voiced_ratio"] = min(1.0, max(0.0, float(cfg.get("auto_listen_min_voiced_ratio", 0.60))))
    except Exception:
        cfg["auto_listen_min_voiced_ratio"] = 0.60
    cfg["tts_streaming_enabled"] = bool(cfg.get("tts_streaming_enabled", False))
    try:
        cfg["tts_stream_min_words"] = max(1, int(cfg.get("tts_stream_min_words", 12)))
    except Exception:
        cfg["tts_stream_min_words"] = 12
    cfg["tts_stream_cut_on_punctuation"] = bool(cfg.get("tts_stream_cut_on_punctuation", False))
    whisper_language = str(cfg.get("whisper_language", "es")).strip().lower() or "es"
    cfg["whisper_language"] = whisper_language
    whisper_ov_device = str(cfg.get("whisper_openvino_device", "AUTO")).strip().upper()
    cfg["whisper_openvino_device"] = whisper_ov_device if whisper_ov_device in env.whisper_ov_device_options else "AUTO"
    cfg["whisper_openvino_model_id"] = str(cfg.get("whisper_openvino_model_id", "")).strip()
    cfg["tts_backend"] = normalize_tts_backend_for_platform(cfg.get("tts_backend", env.default_tts_backend), env.platform_name)
    openvino_tts_device = str(cfg.get("openvino_tts_device", "AUTO")).strip().upper()
    cfg["openvino_tts_device"] = openvino_tts_device if openvino_tts_device in env.openvino_tts_device_options else "AUTO"
    cfg["openvino_tts_model_id"] = str(cfg.get("openvino_tts_model_id", "")).strip()
    try:
        cfg["openvino_tts_timeout_s"] = max(3, int(cfg.get("openvino_tts_timeout_s", 25)))
    except Exception:
        cfg["openvino_tts_timeout_s"] = 25
    cfg["openvino_tts_isolated_gpu"] = bool(cfg.get("openvino_tts_isolated_gpu", True))
    try:
        cfg["openvino_tts_speed"] = min(2.0, max(0.5, float(cfg.get("openvino_tts_speed", 1.0))))
    except Exception:
        cfg["openvino_tts_speed"] = 1.0
    try:
        cfg["openvino_tts_gain"] = min(3.0, max(0.1, float(cfg.get("openvino_tts_gain", 1.0))))
    except Exception:
        cfg["openvino_tts_gain"] = 1.0
    cfg["kokoro_model_id"] = str(cfg.get("kokoro_model_id", "kokoro-tts-intel")).strip()
    kokoro_device = str(cfg.get("kokoro_device", "GPU")).strip().upper()
    cfg["kokoro_device"] = kokoro_device if kokoro_device in env.kokoro_device_options else "GPU"
    cfg["kokoro_voice"] = str(cfg.get("kokoro_voice", "af_sarah")).strip() or "af_sarah"
    cfg["babelvox_model_id"] = str(cfg.get("babelvox_model_id", "babelvox-openvino-int8")).strip()
    babelvox_device = str(cfg.get("babelvox_device", "CPU")).strip().upper()
    cfg["babelvox_device"] = babelvox_device if babelvox_device in env.babelvox_device_options else "CPU"
    babelvox_precision = str(cfg.get("babelvox_precision", "int8")).strip().lower()
    cfg["babelvox_precision"] = babelvox_precision if babelvox_precision in env.babelvox_precision_options else "int8"
    babelvox_language = str(cfg.get("babelvox_language", "es")).strip().lower()
    cfg["babelvox_language"] = babelvox_language if babelvox_language in env.whisper_language_options else "es"
    cfg["tada_model_id"] = str(cfg.get("tada_model_id", "HumeAI/tada-1b")).strip() or "HumeAI/tada-1b"
    cfg["tada_codec_id"] = str(cfg.get("tada_codec_id", "HumeAI/tada-codec")).strip() or "HumeAI/tada-codec"
    cfg["tada_device"] = str(cfg.get("tada_device", "cpu")).strip() or "cpu"
    cfg["tada_language"] = str(cfg.get("tada_language", "en")).strip().lower() or "en"
    cfg["tada_reference_audio_path"] = str(cfg.get("tada_reference_audio_path", "")).strip()
    cfg["tada_reference_text"] = str(cfg.get("tada_reference_text", "")).strip()
    try:
        cfg["tada_sample_rate"] = max(8000, int(cfg.get("tada_sample_rate", 24000)))
    except Exception:
        cfg["tada_sample_rate"] = 24000
    cfg["espeak_voice"] = str(cfg.get("espeak_voice", "es")).strip() or "es"
    try:
        cfg["espeak_rate"] = min(450, max(80, int(cfg.get("espeak_rate", 145))))
    except Exception:
        cfg["espeak_rate"] = 145
    try:
        cfg["espeak_pitch"] = min(99, max(0, int(cfg.get("espeak_pitch", 45))))
    except Exception:
        cfg["espeak_pitch"] = 45
    try:
        cfg["espeak_amplitude"] = min(200, max(0, int(cfg.get("espeak_amplitude", 120))))
    except Exception:
        cfg["espeak_amplitude"] = 120
    return cfg, warning


def save_robot_config(config: dict, robot_config_file: Path) -> tuple[bool, str]:
    try:
        robot_config_file.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        return True, f"Config saved to {robot_config_file}"
    except Exception as exc:
        return False, f"Could not save config: {exc}"
