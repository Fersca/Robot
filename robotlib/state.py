from __future__ import annotations


def build_stt_runtime(compat: dict) -> dict:
    return {
        "numpy": None,
        "sounddevice": None,
        "whisper": None,
        "model": None,
        "model_name": None,
        "ov_pipeline": None,
        "ov_model_id": None,
        "backend": None,
        "active_key": None,
        "compat": compat,
    }


def build_tts_runtime(compat: dict) -> dict:
    return {
        "pipeline": None,
        "model_id": None,
        "backend": None,
        "active_key": None,
        "ov_model_dir": None,
        "ov_device": None,
        "numpy": None,
        "sounddevice": None,
        "kokoro_engine": None,
        "babelvox_engine": None,
        "tada_model": None,
        "tada_encoder": None,
        "torch": None,
        "torchaudio": None,
        "compat": compat,
    }


def build_server_state(config: dict) -> dict:
    return {"pipe": None, "current": None, "config": config}


def build_llm_state(pipe, current, history: list[str], server_state: dict, compat: dict) -> dict:
    return {
        "pipe": pipe,
        "current": current,
        "history": history,
        "server_state": server_state,
        "compat": compat,
    }


def build_camera_runtime(voice_config: dict, tts_runtime: dict, speaker=None) -> dict:
    return {
        "cv2": None,
        "thread": None,
        "vision_event_thread": None,
        "vision_event_queue": None,
        "stop_event": None,
        "panel_enabled": False,
        "panel_backend": None,
        "active_device_index": None,
        "qt_panel_process": None,
        "qt_panel_state_queue": None,
        "qt_panel_action_queue": None,
        "qt_panel_stop_event": None,
        "speaker": speaker,
        "voice_config": voice_config,
        "tts_runtime": tts_runtime,
        "vision_log_enabled": bool(voice_config.get("vision_log_enabled", False)),
        "vision_log_interval_s": float(voice_config.get("vision_log_interval_s", 1.0)),
        "vision_log_last_ts": 0.0,
        "vision_event_processing_enabled": bool(voice_config.get("vision_event_processing_enabled", True)),
        "vision_event_last_ts": 0.0,
        "vision_last_detection_count": 0,
        "suppress_next_join_after_interrupt": False,
        "robot_face_gesture": "",
        "robot_face_gesture_until": 0.0,
    }


def build_auto_listen_runtime(
    voice_config: dict,
    stt_runtime: dict,
    tts_runtime: dict,
    speaker,
    llm_state: dict,
    stats: dict,
) -> dict:
    return {
        "thread": None,
        "stop_event": None,
        "audio_monitor_thread": None,
        "audio_monitor_stop_event": None,
        "voice_config": voice_config,
        "stt_runtime": stt_runtime,
        "tts_runtime": tts_runtime,
        "speaker": speaker,
        "llm_state": llm_state,
        "stats": stats,
        "silero_vad": None,
        "silero_model": None,
        "silero_vad_iterator_cls": None,
        "cv2": None,
        "last_is_speech": False,
        "last_speech_probability": 0.0,
        "last_speech_probability_threshold": 0.50,
        "last_speech_started": False,
        "last_speech_frames": 0,
        "last_start_event": 0.0,
        "last_end_event": 0.0,
        "last_display_segment_frames": 1,
        "last_recording": False,
        "vad_log_last_ts": 0.0,
    }

