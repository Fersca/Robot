import builtins
import importlib
import json
import os
import queue
import sys
import types
import uuid
from pathlib import Path


def load_robot_module(monkeypatch):
    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.snapshot_download = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
    monkeypatch.delitem(sys.modules, "robot", raising=False)
    spec = importlib.util.spec_from_file_location("robot", Path(__file__).resolve().parent.parent / "robot.py")
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules["robot"] = module
    spec.loader.exec_module(module)
    return module


def make_local_tmp_dir() -> Path:
    base = Path(".tmp_tests")
    base.mkdir(exist_ok=True)
    path = base / f"pytest_{uuid.uuid4().hex}"
    path.mkdir()
    return path


def test_default_tts_backend_varies_by_platform(monkeypatch):
    robot = load_robot_module(monkeypatch)
    assert robot.default_tts_backend_for_platform("windows") == "windows"
    assert robot.default_tts_backend_for_platform("linux") == "espeakng"


def test_default_robot_config_contains_camera_defaults(monkeypatch):
    robot = load_robot_module(monkeypatch)
    cfg = robot.default_robot_config()
    assert cfg["camera_enabled"] is False
    assert cfg["camera_device_index"] == 0
    assert cfg["vision_enabled"] is False
    assert cfg["vision_device"] == "AUTO"
    assert cfg["vision_model_path"] == ""
    assert cfg["vision_log_enabled"] is False
    assert cfg["vision_log_interval_s"] == 1.0
    assert cfg["vision_event_processing_enabled"] is True
    assert cfg["auto_listen_enabled"] is False
    assert cfg["audio_input_device"] == ""
    assert cfg["audio_monitor_enabled"] is False
    assert cfg["visual_effects_enabled"] is True
    assert cfg["auto_listen_aggressiveness"] == 3
    assert cfg["auto_listen_threshold"] == 0.50
    assert cfg["auto_listen_frame_ms"] == 32
    assert cfg["auto_listen_preroll_ms"] == 350
    assert cfg["auto_listen_min_speech_ms"] == 1400
    assert cfg["auto_listen_silence_ms"] == 1600
    assert cfg["auto_listen_max_segment_s"] == 60.0
    assert cfg["auto_listen_min_segment_ms"] == 1400
    assert cfg["auto_listen_min_voiced_ratio"] == 0.60
    assert cfg["auto_listen_resume_delay_ms"] == 1500


def test_load_robot_config_accepts_npu_for_vision_device(monkeypatch):
    robot = load_robot_module(monkeypatch)
    tmp_dir = make_local_tmp_dir()
    try:
        cfg_path = tmp_dir / "robot_config.json"
        cfg_path.write_text(json.dumps({"vision_device": "npu"}), encoding="utf-8")
        monkeypatch.setattr(robot, "ROBOT_CONFIG_FILE", cfg_path)

        cfg = robot.load_robot_config()

        assert cfg["vision_device"] == "NPU"
    finally:
        if cfg_path.exists():
            cfg_path.unlink()
        os.rmdir(tmp_dir)


def test_normalize_windows_tts_to_espeak_on_linux(monkeypatch):
    robot = load_robot_module(monkeypatch)
    assert robot.normalize_tts_backend_for_platform("windows", "linux") == "espeakng"
    assert robot.normalize_tts_backend_for_platform("espeakng", "linux") == "espeakng"
    assert robot.normalize_tts_backend_for_platform("windows", "windows") == "windows"


def test_load_robot_config_normalizes_windows_backend_on_linux(monkeypatch):
    robot = load_robot_module(monkeypatch)
    tmp_dir = make_local_tmp_dir()
    try:
        cfg_path = tmp_dir / "robot_config.json"
        cfg_path.write_text(json.dumps({"tts_backend": "windows", "repetir": True}), encoding="utf-8")
        monkeypatch.setattr(robot, "ROBOT_CONFIG_FILE", cfg_path)
        monkeypatch.setattr(robot, "PLATFORM_NAME", "linux")
        monkeypatch.setattr(robot, "DEFAULT_TTS_BACKEND", "espeakng")

        cfg = robot.load_robot_config()

        assert cfg["tts_backend"] == "espeakng"
        assert cfg["repeat"] is True
        assert "repetir" not in cfg
    finally:
        if cfg_path.exists():
            cfg_path.unlink()
        os.rmdir(tmp_dir)


def test_load_robot_config_keeps_windows_backend_on_windows(monkeypatch):
    robot = load_robot_module(monkeypatch)
    tmp_dir = make_local_tmp_dir()
    try:
        cfg_path = tmp_dir / "robot_config.json"
        cfg_path.write_text(json.dumps({"tts_backend": "windows"}), encoding="utf-8")
        monkeypatch.setattr(robot, "ROBOT_CONFIG_FILE", cfg_path)
        monkeypatch.setattr(robot, "PLATFORM_NAME", "windows")
        monkeypatch.setattr(robot, "DEFAULT_TTS_BACKEND", "windows")

        cfg = robot.load_robot_config()

        assert cfg["tts_backend"] == "windows"
    finally:
        if cfg_path.exists():
            cfg_path.unlink()
        os.rmdir(tmp_dir)


def test_initialize_native_voice_engine_skips_non_windows(monkeypatch):
    robot = load_robot_module(monkeypatch)
    monkeypatch.setattr(robot, "PLATFORM_NAME", "linux")

    speaker, voices, error = robot.initialize_native_voice_engine({})

    assert speaker is None
    assert voices is None
    assert "unavailable" in error.lower()


def test_initialize_native_voice_engine_windows_path(monkeypatch):
    robot = load_robot_module(monkeypatch)
    monkeypatch.setattr(robot, "PLATFORM_NAME", "windows")

    class FakeVoice:
        def GetDescription(self):
            return "Voice A"

    class FakeVoices:
        Count = 1

        def Item(self, index):
            return FakeVoice()

    class FakeSpeaker:
        def __init__(self):
            self.Rate = None
            self.Volume = None
            self.Voice = None

        def GetVoices(self):
            return FakeVoices()

    class FakeClientModule:
        @staticmethod
        def Dispatch(name):
            assert name == "SAPI.SpVoice"
            return FakeSpeaker()

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "win32com.client":
            return FakeClientModule()
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    speaker, voices, error = robot.initialize_native_voice_engine({"voice_index": 0, "rate": 1, "volume": 55})

    assert error is None
    assert speaker is not None
    assert voices is not None
    assert speaker.Rate == 1
    assert speaker.Volume == 55


def test_choose_voice_interactive_is_blocked_on_linux(monkeypatch, capsys):
    robot = load_robot_module(monkeypatch)
    monkeypatch.setattr(robot, "PLATFORM_NAME", "linux")

    robot.choose_voice_interactive(None, None, {})

    out = capsys.readouterr().out
    assert "only available on windows" in out.lower()


def test_speak_text_backend_uses_espeak_on_linux_when_windows_requested(monkeypatch):
    robot = load_robot_module(monkeypatch)
    monkeypatch.setattr(robot, "PLATFORM_NAME", "linux")

    called = {}

    def fake_espeak(text, config, allow_interrupt=False):
        called["text"] = text
        called["backend"] = config["tts_backend"]
        return False, 0.123

    monkeypatch.setattr(robot, "speak_espeak_ng", fake_espeak)

    interrupted, latency = robot.speak_text_backend(
        speaker=None,
        text="hello",
        config={"tts_backend": "windows"},
        tts_runtime={},
        allow_interrupt=False,
    )

    assert interrupted is False
    assert latency == 0.123
    assert called["text"] == "hello"


def test_resolve_llm_device_falls_back_to_cpu_on_linux_gpu_probe_failure(monkeypatch):
    robot = load_robot_module(monkeypatch)
    monkeypatch.setattr(robot, "IS_LINUX", True)

    def fake_probe(model_path, device, performance_hint, timeout_s=45):
        if device == "GPU":
            return False, "segfault in plugin"
        return True, ""

    monkeypatch.setattr(robot, "probe_llm_pipeline_load", fake_probe)

    device, reason = robot.resolve_llm_device_for_load(Path("model"), "GPU", allow_linux_fallback=True)

    assert device == "CPU"
    assert "segfault" in reason


def test_start_camera_preview_selects_device_and_updates_config(monkeypatch):
    robot = load_robot_module(monkeypatch)

    started = {}

    class FakeThread:
        def __init__(self, target=None, args=(), daemon=None):
            self.target = target
            self.args = args
            self.daemon = daemon
            self._alive = False

        def start(self):
            started["called"] = True
            self._alive = True

        def is_alive(self):
            return self._alive

        def join(self, timeout=None):
            self._alive = False

    monkeypatch.setattr(robot, "list_camera_devices", lambda runtime: [(2, "Camera 2")])
    monkeypatch.setattr(robot, "open_camera_capture", lambda cv2_mod, device_index: None)
    monkeypatch.setattr(robot, "save_robot_config", lambda config: started.setdefault("saved", dict(config)))
    monkeypatch.setattr(robot.threading, "Thread", FakeThread)
    monkeypatch.setattr(builtins, "input", lambda prompt="": "1")

    camera_runtime = {"cv2": object(), "thread": None, "stop_event": None, "active_device_index": None}
    config = {"camera_enabled": False, "camera_device_index": 0, "vision_enabled": False}

    ok = robot.start_camera_preview(camera_runtime, config)

    assert ok is True
    assert config["camera_enabled"] is True
    assert config["camera_device_index"] == 2
    assert camera_runtime["active_device_index"] == 2
    assert started["called"] is True


def test_start_camera_preview_reuses_saved_device_without_prompt(monkeypatch):
    robot = load_robot_module(monkeypatch)
    started = {}

    class FakeCap:
        def isOpened(self):
            return True

        def read(self):
            return True, object()

        def release(self):
            return None

    class FakeThread:
        def __init__(self, target=None, args=(), daemon=None):
            self._alive = False

        def start(self):
            started["called"] = True
            self._alive = True

        def is_alive(self):
            return self._alive

        def join(self, timeout=None):
            self._alive = False

    monkeypatch.setattr(robot, "ensure_camera_runtime", lambda runtime: True)
    monkeypatch.setattr(robot, "open_camera_capture", lambda cv2_mod, device_index: FakeCap())
    monkeypatch.setattr(robot, "list_camera_devices", lambda runtime: (_ for _ in ()).throw(AssertionError("should not scan devices")))
    monkeypatch.setattr(robot, "save_robot_config", lambda config: started.setdefault("saved", dict(config)))
    monkeypatch.setattr(robot.threading, "Thread", FakeThread)
    monkeypatch.setattr(builtins, "input", lambda prompt="": (_ for _ in ()).throw(AssertionError("should not prompt")))

    camera_runtime = {"cv2": object(), "thread": None, "stop_event": None, "active_device_index": None}
    config = {"camera_enabled": False, "camera_device_index": 3, "vision_enabled": False}

    ok = robot.start_camera_preview(camera_runtime, config)

    assert ok is True
    assert config["camera_device_index"] == 3
    assert camera_runtime["active_device_index"] == 3
    assert started["called"] is True


def test_parse_detection_results_handles_ssd_layout(monkeypatch):
    robot = load_robot_module(monkeypatch)
    raw = [[[[0, 1, 0.9, 0.1, 0.2, 0.7, 0.8]]]]

    detections = robot.parse_detection_results(raw, (100, 200, 3), 0.5, ["bg", "person"])

    assert len(detections) == 1
    assert detections[0]["label"] == "person"
    assert detections[0]["box"] == (20, 20, 140, 80)


def test_format_vision_debug_output_includes_detections(monkeypatch):
    robot = load_robot_module(monkeypatch)
    text = robot.format_vision_debug_output([[1, 2, 3]], [{"label": "face", "score": 0.9, "box": (1, 2, 3, 4)}])
    payload = json.loads(text)
    assert payload["shape"] == [1, 3]
    assert payload["detections"][0]["label"] == "face"


def test_format_vad_debug_output_serializes_payload(monkeypatch):
    robot = load_robot_module(monkeypatch)
    text = robot.format_vad_debug_output({"speech": True, "rms": 0.12})
    payload = json.loads(text)
    assert payload["speech"] is True
    assert payload["rms"] == 0.12


def test_build_audio_monitor_frame_has_expected_shape(monkeypatch):
    robot = load_robot_module(monkeypatch)
    np_mod = importlib.import_module("numpy")
    frame = robot.build_audio_monitor_frame(np_mod, 0.5, 0.25, True)
    assert frame.shape == (250, 420, 3)


def test_list_audio_input_devices_filters_generic_and_duplicates(monkeypatch):
    robot = load_robot_module(monkeypatch)

    class FakeSoundDevice:
        @staticmethod
        def query_hostapis():
            return [
                {"name": "MME"},
                {"name": "Windows WASAPI"},
            ]

        @staticmethod
        def query_devices():
            return [
                {"name": "Asignador de sonido Microsoft - Input", "max_input_channels": 2, "hostapi": 0},
                {"name": "Microphone (C922 Pro Stream Webcam)", "max_input_channels": 2, "hostapi": 0},
                {"name": "Microphone (C922 Pro Stream Webcam)", "max_input_channels": 2, "hostapi": 1},
                {"name": "Headset Microphone (EMBERTON III)", "max_input_channels": 1, "hostapi": 0},
            ]

    devices = robot.list_audio_input_devices(FakeSoundDevice())

    assert devices == [
        (3, "Headset Microphone (EMBERTON III)", 1),
        (2, "Microphone (C922 Pro Stream Webcam)", 2),
    ]


def test_should_emit_vision_log_respects_interval(monkeypatch):
    robot = load_robot_module(monkeypatch)
    runtime = {"vision_log_enabled": True, "vision_log_interval_s": 1.0, "vision_log_last_ts": 0.0}
    assert robot.should_emit_vision_log(runtime, now_ts=1.2) is True
    assert robot.should_emit_vision_log(runtime, now_ts=1.8) is False
    assert robot.should_emit_vision_log(runtime, now_ts=2.3) is True


def test_should_process_vision_events_respects_flag_and_interval(monkeypatch):
    robot = load_robot_module(monkeypatch)
    runtime = {"vision_event_processing_enabled": True, "vision_log_interval_s": 1.0, "vision_event_last_ts": 0.0}
    assert robot.should_process_vision_events(runtime, now_ts=1.2) is True
    assert robot.should_process_vision_events(runtime, now_ts=1.8) is False
    runtime["vision_event_processing_enabled"] = False
    assert robot.should_process_vision_events(runtime, now_ts=3.0) is False


def test_handle_vision_tick_tracks_face_count(monkeypatch):
    robot = load_robot_module(monkeypatch)
    runtime = {"vision_last_detection_count": 0}
    monkeypatch.setattr(robot, "choose_vision_event_response", lambda category, count: f"{category}:{count}")

    assert robot.handle_vision_tick(runtime, [{"label": "face"}]) == "first_person_joined:1"
    assert runtime["vision_last_detection_count"] == 1
    assert robot.handle_vision_tick(runtime, [{"label": "face"}]) is None
    assert robot.handle_vision_tick(runtime, [{"label": "face"}, {"label": "face"}]) == "more_people_joined:2"
    assert robot.handle_vision_tick(runtime, [{"label": "face"}]) == "fewer_people_left:1"
    assert robot.handle_vision_tick(runtime, []) == "alone_again:0"


def test_handle_vision_tick_interrupts_audio_when_alone(monkeypatch):
    robot = load_robot_module(monkeypatch)
    runtime = {"vision_last_detection_count": 1}
    monkeypatch.setattr(robot, "is_audio_playback_active", lambda: True)
    monkeypatch.setattr(robot, "is_tts_active", lambda: False)

    assert robot.handle_vision_tick(runtime, []) == "__INTERRUPT_AUDIO__:me cayo"
    assert runtime["suppress_next_join_after_interrupt"] is True


def test_handle_vision_tick_suppresses_next_join_after_interrupt(monkeypatch):
    robot = load_robot_module(monkeypatch)
    runtime = {
        "vision_last_detection_count": 0,
        "suppress_next_join_after_interrupt": True,
    }
    monkeypatch.setattr(robot, "choose_vision_event_response", lambda category, count: f"{category}:{count}")

    assert robot.handle_vision_tick(runtime, [{"label": "face"}]) is None
    assert runtime["suppress_next_join_after_interrupt"] is False


def test_choose_vision_event_response_formats_count(monkeypatch):
    robot = load_robot_module(monkeypatch)
    monkeypatch.setattr(
        robot,
        "load_vision_event_responses",
        lambda: {"more_people_joined": ["Ahora hay: {count}"]},
    )
    monkeypatch.setattr(robot.random, "choice", lambda items: items[0])
    assert robot.choose_vision_event_response("more_people_joined", 3) == "Ahora hay: 3"


def test_tts_activity_scope_sets_and_clears_flag(monkeypatch):
    robot = load_robot_module(monkeypatch)
    assert robot.is_tts_active() is False
    with robot.tts_activity_scope():
        assert robot.is_tts_active() is True
    assert robot.is_tts_active() is False


def test_is_tts_blocking_auto_listen_honors_cooldown(monkeypatch):
    robot = load_robot_module(monkeypatch)
    monkeypatch.setattr(robot.time, "monotonic", lambda: 10.0)
    robot.LAST_TTS_ACTIVITY_END_TS = 9.2
    assert robot.is_tts_blocking_auto_listen({"auto_listen_resume_delay_ms": 1500}) is True
    robot.LAST_TTS_ACTIVITY_END_TS = 8.0
    assert robot.is_tts_blocking_auto_listen({"auto_listen_resume_delay_ms": 1500}) is False


def test_finalize_vad_segment_returns_audio_for_basic_segment(monkeypatch):
    robot = load_robot_module(monkeypatch)
    np_mod = importlib.import_module("numpy")
    frame = (np_mod.zeros(480, dtype=np_mod.int16)).tobytes()
    state = {
        "frame_ms": 30,
        "min_segment_ms": 1400,
        "speech_frames": [frame] * 47,
    }
    audio, reason = robot._finalize_vad_segment(np_mod, state)
    assert audio is not None
    assert reason == ""


def test_is_valid_auto_listen_transcript_filters_short_noise(monkeypatch):
    robot = load_robot_module(monkeypatch)
    assert robot.is_valid_auto_listen_transcript("hola como estas hoy") is True
    assert robot.is_valid_auto_listen_transcript("hola") is True
    assert robot.is_valid_auto_listen_transcript("hola amigo") is True
    assert robot.is_valid_auto_listen_transcript("buenas hola") is True
    assert robot.is_valid_auto_listen_transcript("hola como va") is True
    assert robot.is_valid_auto_listen_transcript("como te va") is False
    assert robot.is_valid_auto_listen_transcript("ok") is False
    assert robot.is_valid_auto_listen_transcript("a b") is False
    assert robot.is_valid_auto_listen_transcript("y") is False
    assert robot.is_valid_auto_listen_transcript("sh") is False


def test_emit_vision_event_message_enqueues_for_worker(monkeypatch):
    robot = load_robot_module(monkeypatch)
    runtime = {
        "vision_event_queue": queue.Queue(maxsize=4),
    }

    robot.emit_vision_event_message(runtime, "Hola, como andas?")

    assert runtime["vision_event_queue"].get_nowait() == "Hola, como andas?"


def test_load_vision_models_reads_catalog(monkeypatch):
    robot = load_robot_module(monkeypatch)
    models = robot.load_vision_models()
    assert models
    assert models[0]["id"] == "face-detection-retail-0004"
    assert models[0]["xml_url"].endswith(".xml")


def test_is_valid_openvino_xml_file_rejects_html(monkeypatch):
    robot = load_robot_module(monkeypatch)
    path = Path(".tmp_tests") / "bad_vision_model.xml"
    path.parent.mkdir(exist_ok=True)
    path.write_text("<!DOCTYPE html><html></html>", encoding="utf-8")
    try:
        assert robot.is_valid_openvino_xml_file(path) is False
    finally:
        if path.exists():
            path.unlink()


def test_vision_select_updates_config(monkeypatch):
    robot = load_robot_module(monkeypatch)
    selected = {
        "id": "face-detection-retail-0004",
        "display": "Face Detection Retail 0004",
        "xml_path": Path("vision_models/face.xml"),
        "local": Path("vision_models"),
        "labels": ["background", "face"],
    }

    saved = {}
    monkeypatch.setattr(robot, "load_vision_models", lambda: [selected])
    monkeypatch.setattr(robot, "choose_vision_model_interactive", lambda models, selected_id=None, allow_download=True: selected)
    monkeypatch.setattr(robot, "choose_vision_device_interactive", lambda selected_device=None: "NPU")
    monkeypatch.setattr(robot, "save_robot_config", lambda config: saved.setdefault("config", dict(config)))

    camera_runtime = {"vision_compiled_model": object(), "vision_active_key": ("x", "CPU"), "vision_labels": []}
    voice_config = {"vision_model_id": "", "vision_model_path": "", "vision_labels_path": ""}

    models = robot.load_vision_models()
    chosen = robot.choose_vision_model_interactive(models, selected_id="", allow_download=True)
    chosen_device = robot.choose_vision_device_interactive("AUTO")
    voice_config["vision_model_id"] = chosen["id"]
    voice_config["vision_model_path"] = str(chosen["xml_path"])
    voice_config["vision_device"] = chosen_device
    labels_path = chosen["local"] / "labels.txt"
    voice_config["vision_labels_path"] = str(labels_path) if labels_path.exists() else ""
    robot.save_robot_config(voice_config)
    camera_runtime["vision_compiled_model"] = None
    camera_runtime["vision_active_key"] = None
    camera_runtime["vision_labels"] = list(chosen.get("labels", []))

    assert saved["config"]["vision_model_id"] == "face-detection-retail-0004"
    assert saved["config"]["vision_model_path"].endswith("face.xml")
    assert saved["config"]["vision_device"] == "NPU"
    assert camera_runtime["vision_labels"] == ["background", "face"]
