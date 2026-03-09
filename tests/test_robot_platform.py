import builtins
import importlib
import json
import os
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
