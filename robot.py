from __future__ import annotations

import contextlib
import gc
import _thread
import importlib
import json
import math
import multiprocessing as mp
import os
import platform
import queue
import re
import random
import select
import shutil
import subprocess
import sys
import threading
import time
import unicodedata
import uuid
import wave
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

IS_WINDOWS = platform.system().lower() == "windows"
IS_LINUX = platform.system().lower() == "linux"
PLATFORM_NAME = "windows" if IS_WINDOWS else "linux" if IS_LINUX else platform.system().lower()
TTS_ACTIVITY_LOCK = threading.Lock()
TTS_ACTIVITY_COUNT = 0
TTS_PLAYING_EVENT = threading.Event()
PLAYBACK_ACTIVITY_LOCK = threading.Lock()
PLAYBACK_ACTIVITY_COUNT = 0
AUDIO_PLAYBACK_EVENT = threading.Event()
LAST_TTS_ACTIVITY_END_TS = 0.0
APP_EXIT_REQUESTED = threading.Event()
AUDIO_CANCEL_EVENT = threading.Event()
WINDOWS_TTS_LOCK = threading.RLock()
SAPI_SVSFLAGSASYNC = 1
SAPI_SVSFPURGEBEFORESPEAK = 2
SAPI_SVSFISXML = 8
WINDOWS_TTS_WARMUP_SILENCE_MS = 120

try:
    import msvcrt  # type: ignore
except Exception:
    msvcrt = None

try:
    import openvino_genai as ov_genai
except Exception:
    ov_genai = None
from huggingface_hub import snapshot_download


def detect_runtime_platform() -> str:
    return PLATFORM_NAME


def default_tts_backend_for_platform(platform_name: str) -> str:
    return "windows" if platform_name == "windows" else "espeakng"


def normalize_tts_backend_for_platform(tts_backend: str, platform_name: str) -> str:
    backend = str(tts_backend or "").strip().lower()
    if backend not in TTS_BACKEND_OPTIONS:
        return default_tts_backend_for_platform(platform_name)
    if backend == "windows" and platform_name != "windows":
        return "espeakng"
    return backend


def platform_supports_native_voices(platform_name: str) -> bool:
    return platform_name == "windows"


class KeyboardAdapter:
    @contextlib.contextmanager
    def capture(self):
        yield self

    def clear_buffer(self) -> None:
        return None

    def read_key_nonblocking(self) -> str | None:
        return None


class WindowsKeyboardAdapter(KeyboardAdapter):
    def clear_buffer(self) -> None:
        if msvcrt is None:
            return
        while msvcrt.kbhit():
            msvcrt.getwch()

    def read_key_nonblocking(self) -> str | None:
        if msvcrt is None or not msvcrt.kbhit():
            return None
        return msvcrt.getwch()


class PosixKeyboardAdapter(KeyboardAdapter):
    def __init__(self):
        self._fd: int | None = None
        self._old_termios = None
        self._old_flags: int | None = None
        self._active = False

    @contextlib.contextmanager
    def capture(self):
        if not sys.stdin or not hasattr(sys.stdin, "isatty") or not sys.stdin.isatty():
            yield self
            return
        import fcntl
        import termios
        import tty

        fd = sys.stdin.fileno()
        old_termios = termios.tcgetattr(fd)
        old_flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        try:
            tty.setcbreak(fd)
            fcntl.fcntl(fd, fcntl.F_SETFL, old_flags | os.O_NONBLOCK)
            self._fd = fd
            self._old_termios = old_termios
            self._old_flags = old_flags
            self._active = True
            yield self
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_termios)
            except Exception:
                pass
            try:
                fcntl.fcntl(fd, fcntl.F_SETFL, old_flags)
            except Exception:
                pass
            self._fd = None
            self._old_termios = None
            self._old_flags = None
            self._active = False

    def clear_buffer(self) -> None:
        while self.read_key_nonblocking() is not None:
            pass

    def read_key_nonblocking(self) -> str | None:
        if not self._active or self._fd is None:
            return None
        try:
            readable, _, _ = select.select([sys.stdin], [], [], 0)
        except Exception:
            return None
        if not readable:
            return None
        try:
            data = os.read(self._fd, 1)
        except BlockingIOError:
            return None
        except Exception:
            return None
        if not data:
            return None
        return data.decode("utf-8", errors="ignore")


def create_keyboard_adapter(platform_name: str) -> KeyboardAdapter:
    if platform_name == "windows":
        return WindowsKeyboardAdapter()
    return PosixKeyboardAdapter()

# =========================
# Config
# =========================
DEFAULT_DEVICE = "NPU"
DEFAULT_PERFORMANCE_HINT = "LATENCY"

DEVICE_OPTIONS = [
    "CPU",
    "GPU",
    "NPU",
    "AUTO",
    "AUTO:NPU,GPU,CPU",
    "MULTI:NPU,GPU",
    "HETERO:NPU,GPU,CPU",
]

PERFORMANCE_HINT_OPTIONS = [
    "LATENCY",
    "THROUGHPUT",
    "CUMULATIVE_THROUGHPUT",
    "UNDEFINED",
]

BENCHMARK_DEVICES = ["CPU", "GPU", "NPU"]
STATS_MODE_NORMAL = "normal"
STATS_MODE_BENCHMARK = "benchmark"

ACTIVE_DEVICE = DEFAULT_DEVICE
ACTIVE_PERFORMANCE_HINT = DEFAULT_PERFORMANCE_HINT

CACHE_DIR = Path.home() / "ov_models"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

STATS_FILE = CACHE_DIR / "stats.json"
AUTH_FILE = CACHE_DIR / "hf_auth.json"  # {"hf_token": "hf_..."}
BENCHMARK_PROMPTS_FILE = CACHE_DIR / "benchmark_prompts.json"
DEVICE_COMPAT_FILE = CACHE_DIR / "device_compat.json"
MODELS_FILE = CACHE_DIR / "models.json"
ROBOT_CONFIG_FILE = Path(__file__).resolve().parent / "robot_config.json"
WHISPER_OV_MODELS_FILE = CACHE_DIR / "whisper_models.json"
OV_TTS_MODELS_FILE = CACHE_DIR / "openvino_tts_models.json"
KOKORO_MODELS_FILE = CACHE_DIR / "kokoro_models.json"
BABELVOX_MODELS_FILE = CACHE_DIR / "babelvox_models.json"
VISION_MODELS_FILE = Path(__file__).resolve().parent / "vision_models.json"
VISION_EVENT_RESPONSES_FILE = Path(__file__).resolve().parent / "vision_event_responses.json"

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
DEFAULT_WHISPER_MODEL = "base"
WHISPER_OV_DEVICE_OPTIONS = ["AUTO", "CPU", "GPU"]
WHISPER_LANGUAGE_OPTIONS = ["auto", "es", "en", "pt", "fr", "it", "de"]
PARLER_LANGUAGE_OPTIONS = ["auto", "es", "en", "pt", "fr", "it", "de"]
PARLER_STYLE_OPTIONS = ["neutral", "conversational", "calm", "energetic", "formal"]
PARLER_LANGUAGE_HINT = {
    "auto": "Use the language that best matches the input text.",
    "es": "Speak in Spanish with native pronunciation.",
    "en": "Speak in English with native pronunciation.",
    "pt": "Speak in Portuguese with native pronunciation.",
    "fr": "Speak in French with native pronunciation.",
    "it": "Speak in Italian with native pronunciation.",
    "de": "Speak in German with native pronunciation.",
}
PARLER_STYLE_HINT = {
    "neutral": "Use a neutral, clear, natural speaking style.",
    "conversational": "Use a conversational and friendly speaking style.",
    "calm": "Use a calm, soft, relaxed speaking style.",
    "energetic": "Use an energetic and expressive speaking style.",
    "formal": "Use a formal and professional speaking style.",
}
WHISPER_LANGUAGE_CODE_TO_NAME = {
    "es": "spanish",
    "en": "english",
    "pt": "portuguese",
    "fr": "french",
    "it": "italian",
    "de": "german",
}
WHISPER_EXPECTED_SIZE_BYTES = {
    "tiny": 75 * 1024 * 1024,
    "base": 142 * 1024 * 1024,
    "small": 466 * 1024 * 1024,
    "medium": 1500 * 1024 * 1024,
    "large": 2900 * 1024 * 1024,
    "large-v2": 2900 * 1024 * 1024,
    "large-v3": 2900 * 1024 * 1024,
}
WHISPER_OV_EXPECTED_SIZE_BYTES = {
    "openvino/whisper-tiny-fp16-ov": 151 * 1024 * 1024,
    "openvino/whisper-tiny-int8-ov": 82 * 1024 * 1024,
    "openvino/whisper-base-fp16-ov": 289 * 1024 * 1024,
    "openvino/whisper-base-int8-ov": 156 * 1024 * 1024,
    "openvino/whisper-base-int4-ov": 86 * 1024 * 1024,
    "openvino/whisper-small-fp16-ov": 945 * 1024 * 1024,
    "openvino/whisper-small-int8-ov": 489 * 1024 * 1024,
    "openvino/whisper-medium-int8-ov": 1536 * 1024 * 1024,
    "openvino/whisper-large-v3-fp16-ov": 3100 * 1024 * 1024,
    "openvino/whisper-large-v3-int8-ov": 1600 * 1024 * 1024,
    "openvino/whisper-large-v3-int4-ov": 950 * 1024 * 1024,
    "openvino/distil-whisper-large-v2-int8-ov": 820 * 1024 * 1024,
    "openvino/distil-whisper-large-v3-fp16-ov": 1550 * 1024 * 1024,
    "openvino/distil-whisper-large-v3-int4-ov": 500 * 1024 * 1024,
}

DEFAULT_WHISPER_OV_MODELS_DATA = [
    {
        "id": "whisper-tiny-fp16-ov",
        "display": "Whisper Tiny FP16 (OV)",
        "repo_url": "OpenVINO/whisper-tiny-fp16-ov",
        "local_dir": "whisper-tiny-fp16-ov",
    },
    {
        "id": "whisper-tiny-int8-ov",
        "display": "Whisper Tiny INT8 (OV)",
        "repo_url": "OpenVINO/whisper-tiny-int8-ov",
        "local_dir": "whisper-tiny-int8-ov",
    },
    {
        "id": "whisper-base-fp16-ov",
        "display": "Whisper Base FP16 (OV)",
        "repo_url": "OpenVINO/whisper-base-fp16-ov",
        "local_dir": "whisper-base-fp16-ov",
    },
    {
        "id": "whisper-base-int8-ov",
        "display": "Whisper Base INT8 (OV)",
        "repo_url": "OpenVINO/whisper-base-int8-ov",
        "local_dir": "whisper-base-int8-ov",
    },
    {
        "id": "whisper-base-int4-ov",
        "display": "Whisper Base INT4 (OV)",
        "repo_url": "OpenVINO/whisper-base-int4-ov",
        "local_dir": "whisper-base-int4-ov",
    },
    {
        "id": "whisper-small-fp16-ov",
        "display": "Whisper Small FP16 (OV)",
        "repo_url": "OpenVINO/whisper-small-fp16-ov",
        "local_dir": "whisper-small-fp16-ov",
    },
    {
        "id": "whisper-small-int8-ov",
        "display": "Whisper Small INT8 (OV)",
        "repo_url": "OpenVINO/whisper-small-int8-ov",
        "local_dir": "whisper-small-int8-ov",
    },
    {
        "id": "whisper-medium-int8-ov",
        "display": "Whisper Medium INT8 (OV)",
        "repo_url": "OpenVINO/whisper-medium-int8-ov",
        "local_dir": "whisper-medium-int8-ov",
    },
    {
        "id": "whisper-large-v3-fp16-ov",
        "display": "Whisper Large v3 FP16 (OV)",
        "repo_url": "OpenVINO/whisper-large-v3-fp16-ov",
        "local_dir": "whisper-large-v3-fp16-ov",
    },
    {
        "id": "whisper-large-v3-int8-ov",
        "display": "Whisper Large v3 INT8 (OV)",
        "repo_url": "OpenVINO/whisper-large-v3-int8-ov",
        "local_dir": "whisper-large-v3-int8-ov",
    },
    {
        "id": "whisper-large-v3-int4-ov",
        "display": "Whisper Large v3 INT4 (OV)",
        "repo_url": "OpenVINO/whisper-large-v3-int4-ov",
        "local_dir": "whisper-large-v3-int4-ov",
    },
    {
        "id": "distil-whisper-large-v2-int8-ov",
        "display": "Distil-Whisper Large v2 INT8 (OV)",
        "repo_url": "OpenVINO/distil-whisper-large-v2-int8-ov",
        "local_dir": "distil-whisper-large-v2-int8-ov",
    },
    {
        "id": "distil-whisper-large-v3-fp16-ov",
        "display": "Distil-Whisper Large v3 FP16 (OV)",
        "repo_url": "OpenVINO/distil-whisper-large-v3-fp16-ov",
        "local_dir": "distil-whisper-large-v3-fp16-ov",
    },
    {
        "id": "distil-whisper-large-v3-int4-ov",
        "display": "Distil-Whisper Large v3 INT4 (OV)",
        "repo_url": "OpenVINO/distil-whisper-large-v3-int4-ov",
        "local_dir": "distil-whisper-large-v3-int4-ov",
    },
]

TTS_BACKEND_OPTIONS = ["windows", "openvino", "kokoro", "babelvox", "espeakng", "tada"]
DEFAULT_TTS_BACKEND = default_tts_backend_for_platform(PLATFORM_NAME)
KEYBOARD = create_keyboard_adapter(PLATFORM_NAME)
PARLER_OV_DEVICE_OPTIONS = ["AUTO", "CPU", "GPU", "NPU"]
OV_TTS_EXPECTED_SIZE_BYTES = {
    "llmware/speech-t5-tts-ov": 430 * 1024 * 1024,
    "suno/bark-small": 2300 * 1024 * 1024,
}
DEFAULT_OV_TTS_MODELS_DATA = [
    {
        "id": "speech-t5-tts-ov",
        "display": "SpeechT5 TTS (OpenVINO)",
        "repo_url": "llmware/speech-t5-tts-ov",
        "local_dir": "speech-t5-tts-ov",
    },
    {
        "id": "bark-small-ov",
        "display": "Bark Small (OpenVINO, Experimental)",
        "repo_url": "suno/bark-small",
        "local_dir": "bark-small-ov",
    },
]
KOKORO_DEVICE_OPTIONS = ["CPU", "GPU", "NPU"]
BABELVOX_DEVICE_OPTIONS = ["CPU", "NPU"]
BABELVOX_PRECISION_OPTIONS = ["fp16", "int8", "int4", "fp32"]
BABELVOX_LANGUAGE_MAP = {
    "auto": "English",
    "en": "English",
    "es": "Spanish",
    "pt": "Portuguese",
    "fr": "French",
    "it": "Italian",
    "de": "German",
}
DEFAULT_KOKORO_MODELS_DATA = [
    {
        "id": "kokoro-tts-intel",
        "display": "Kokoro TTS Intel",
        "repo_url": "magicunicorn/kokoro-tts-intel",
        "local_dir": "kokoro-tts-intel",
        "expected_size_bytes": 340 * 1024 * 1024,
    }
]
DEFAULT_BABELVOX_MODELS_DATA = [
    {
        "id": "babelvox-openvino-int8",
        "display": "BabelVox OpenVINO INT8",
        "repo_url": "djwarf/babelvox-openvino-int8",
        "local_dir": "babelvox-openvino-int8",
        "expected_size_bytes": 2800 * 1024 * 1024,
    }
]

# =========================
# Model list
# =========================
DEFAULT_MODELS_DATA = [
    {
        "display": "Phi-4 mini instruct INT4",
        "params": "?3.8B",
        "repo": "FluidInference/phi-4-mini-instruct-int4-ov-npu",
        "local_dir": "phi-4-mini-instruct-int4-ov-npu",
    },
    {
        "display": "Qwen2.5 1.5B instruct INT4",
        "params": "1.5B",
        "repo": "OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov",
        "local_dir": "qwen2.5-1.5b-instruct-int4-ov",
    },
    {
        "display": "TinyLlama 1.1B Chat INT4",
        "params": "1.1B",
        "repo": "OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov",
        "local_dir": "tinyllama-1.1b-chat-int4-ov",
    },
    {
        "display": "Phi-3 Medium 4k Instruct INT4",
        "params": "3.8B",
        "repo": "OpenVINO/Phi-3-medium-4k-instruct-int4-ov",
        "local_dir": "phi-3-medium-4k-instruct-int4-ov",
    },
    {
        "display": "Qwen2.5 7B INT4 NPU OV",
        "params": "7B",
        "repo": "FluidInference/qwen2.5-7b-int4-npu-ov",
        "local_dir": "qwen2.5-7b-int4-npu-ov",
    },
    {
        "display": "Llama 3.1 8B Instruct INT4 NPU OV",
        "params": "8B",
        "repo": "llmware/llama-3.1-8b-instruct-npu-ov",
        "local_dir": "llama-3.1-8b-instruct-npu-ov",
    },
]

MODELS = []

# =========================
# HF token handling
# =========================
def ensure_auth_file() -> None:
    if not AUTH_FILE.exists():
        AUTH_FILE.write_text(json.dumps({"hf_token": ""}, indent=2), encoding="utf-8")


def load_hf_token() -> str | None:
    """
    Look for a token in:
      1) env HF_TOKEN
      2) ov_models/hf_auth.json  ({"hf_token": "hf_..."})
    If the file exists but is empty, ask for one and save it.
    """
    ensure_auth_file()

    env_token = os.environ.get("HF_TOKEN", "").strip()
    if env_token:
        return env_token

    token = ""
    try:
        data = json.loads(AUTH_FILE.read_text(encoding="utf-8") or "{}")
        token = (data.get("hf_token") or "").strip()
    except Exception:
        token = ""

    if not token:
        print("\nâš ï¸ HF token not configured (optional).")
        print("Paste your token (starts with 'hf_') or press Enter to continue without one.\n")
        token = input("HF token: ").strip()
        AUTH_FILE.write_text(json.dumps({"hf_token": token}, indent=2), encoding="utf-8")

    if token:
        os.environ["HF_TOKEN"] = token
        return token

    return None


# =========================
# Disk size helpers
# =========================
def dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def human_bytes(num: int) -> str:
    if num <= 0:
        return "—"
    units = ["B", "KB", "MB", "GB", "TB"]
    n = float(num)
    i = 0
    while n >= 1024 and i < len(units) - 1:
        n /= 1024.0
        i += 1
    if i == 0:
        return f"{int(n)} {units[i]}"
    return f"{n:.2f} {units[i]}"


# =========================
# Models: download/select/delete
# =========================
def is_downloaded(model_dir: Path) -> bool:
    if not model_dir.exists():
        return False
    has_xml = any(model_dir.rglob("*.xml"))
    has_bin = any(model_dir.rglob("*.bin"))
    return has_xml and has_bin


def is_repo_downloaded(model_dir: Path) -> bool:
    if not model_dir.exists():
        return False
    try:
        return any(model_dir.iterdir())
    except OSError:
        return False


def model_menu_label(m: dict) -> str:
    if m.get("kind") == "external":
        model_name = str(m.get("repo") or m.get("display") or "external").strip()
        base_url = str(m.get("base_url", "")).strip() or "(unknown url)"
        return f"{model_name} [external @ {base_url}]"
    if is_downloaded(m["local"]):
        size = human_bytes(dir_size_bytes(m["local"]))
    else:
        size = "—"
    return f"{m['display']} ({m['params']}, {size})"


def slug_from_repo(repo: str) -> str:
    name = repo.strip().split("/")[-1].strip().lower()
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in name)
    while "--" in safe:
        safe = safe.replace("--", "-")
    return safe.strip("-") or "model"


def normalize_hf_repo_id(value: str) -> str:
    text = str(value or "").strip()
    if "huggingface.co/" in text:
        text = text.split("huggingface.co/", 1)[1]
    text = text.strip().strip("/")
    if "/tree/" in text:
        text = text.split("/tree/", 1)[0]
    if "/resolve/" in text:
        text = text.split("/resolve/", 1)[0]
    return text


def model_to_storage_entry(model: dict) -> dict:
    return {
        "display": model["display"],
        "params": model["params"],
        "repo": model["repo"],
        "local_dir": model["local"].name,
    }


def parse_model_entry(entry: dict) -> dict | None:
    if not isinstance(entry, dict):
        return None
    display = str(entry.get("display", "")).strip()
    params = str(entry.get("params", "")).strip()
    repo = str(entry.get("repo", "")).strip()
    local_dir = str(entry.get("local_dir", "")).strip()
    if not display or not params or not repo:
        return None
    if not local_dir:
        local_dir = slug_from_repo(repo)
    return {
        "display": display,
        "params": params,
        "repo": repo,
        "local": CACHE_DIR / local_dir,
    }


def save_models(models: list[dict]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = [model_to_storage_entry(m) for m in models]
    MODELS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_models() -> list[dict]:
    if MODELS_FILE.exists():
        try:
            raw = json.loads(MODELS_FILE.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                parsed = [parse_model_entry(x) for x in raw]
                models = [x for x in parsed if x is not None]
                if models:
                    return models
        except Exception:
            pass

    defaults = []
    for entry in DEFAULT_MODELS_DATA:
        parsed = parse_model_entry(entry)
        if parsed is not None:
            defaults.append(parsed)
    save_models(defaults)
    return defaults


def whisper_ov_to_storage_entry(model: dict) -> dict:
    payload = {
        "id": model["id"],
        "display": model["display"],
        "repo_url": model["repo_url"],
        "local_dir": model["local"].name,
    }
    expected_bytes = int(model.get("expected_size_bytes", 0) or 0)
    if expected_bytes > 0:
        payload["expected_size_bytes"] = expected_bytes
    return payload


def parse_whisper_ov_entry(entry: dict) -> dict | None:
    if not isinstance(entry, dict):
        return None
    model_id = str(entry.get("id", "")).strip()
    display = str(entry.get("display", "")).strip()
    repo_url = str(entry.get("repo_url", "")).strip()
    local_dir = str(entry.get("local_dir", "")).strip()
    if not model_id or not display or not repo_url:
        return None
    if not local_dir:
        local_dir = slug_from_repo(repo_url)
    expected_size_bytes = 0
    if isinstance(entry.get("expected_size_bytes"), int):
        expected_size_bytes = int(entry.get("expected_size_bytes") or 0)
    elif isinstance(entry.get("expected_size_mb"), (int, float)):
        expected_size_bytes = int(float(entry.get("expected_size_mb")) * 1024 * 1024)
    if expected_size_bytes <= 0:
        expected_size_bytes = WHISPER_OV_EXPECTED_SIZE_BYTES.get(normalize_hf_repo_id(repo_url).lower(), 0)
    return {
        "id": model_id,
        "display": display,
        "repo_url": repo_url,
        "local": CACHE_DIR / local_dir,
        "expected_size_bytes": expected_size_bytes,
    }


def save_whisper_ov_models(models: list[dict]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = [whisper_ov_to_storage_entry(m) for m in models]
    WHISPER_OV_MODELS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_whisper_ov_models() -> list[dict]:
    if WHISPER_OV_MODELS_FILE.exists():
        try:
            raw = json.loads(WHISPER_OV_MODELS_FILE.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                parsed = [parse_whisper_ov_entry(x) for x in raw]
                return [x for x in parsed if x is not None]
        except Exception:
            pass
    defaults = []
    for entry in DEFAULT_WHISPER_OV_MODELS_DATA:
        parsed = parse_whisper_ov_entry(entry)
        if parsed is not None:
            defaults.append(parsed)
    save_whisper_ov_models(defaults)
    return defaults


def add_whisper_ov_model_interactive(models: list[dict]) -> dict | None:
    print("\nAdd Whisper OpenVINO model\n")
    model_id = input("Model id (unique): ").strip()
    if not model_id:
        print("\nCancelled: model id is required.\n")
        return None
    if any(m["id"] == model_id for m in models):
        print("\n⚠️ A model with that id already exists.\n")
        return None

    display = input("Display name: ").strip()
    if not display:
        print("\nCancelled: display name is required.\n")
        return None

    repo_url = input("HF repo URL/id for OV Whisper: ").strip()
    if not repo_url:
        print("\nCancelled: repo URL/id is required.\n")
        return None

    default_local = slug_from_repo(repo_url)
    local_dir = input(f"Local folder [{default_local}]: ").strip() or default_local
    model = {
        "id": model_id,
        "display": display,
        "repo_url": repo_url,
        "local": CACHE_DIR / local_dir,
    }
    models.append(model)
    save_whisper_ov_models(models)
    print(f"\n✅ Whisper OV model added: {display} ({model_id})\n")
    return model


def download_whisper_ov_model(model: dict) -> None:
    load_hf_token()
    model["local"].parent.mkdir(parents=True, exist_ok=True)
    repo_id = normalize_hf_repo_id(model["repo_url"])
    print(f"\n📥 Downloading Whisper OV model: {model['repo_url']}")
    print(f"   -> destination: {model['local']}\n")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(model["local"]),
        local_dir_use_symlinks=False,
        resume_download=True,
        token=os.environ.get("HF_TOKEN") or None,
    )
    print("\n✅ Whisper OV download complete.\n")


def download_repo_model(model: dict, label: str) -> None:
    load_hf_token()
    model["local"].parent.mkdir(parents=True, exist_ok=True)
    repo_id = normalize_hf_repo_id(model["repo_url"])
    print(f"\n📥 Downloading {label}: {model['repo_url']}")
    print(f"   -> destination: {model['local']}\n")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(model["local"]),
        local_dir_use_symlinks=False,
        resume_download=True,
        token=os.environ.get("HF_TOKEN") or None,
    )
    print(f"\n✅ {label} download complete.\n")


def whisper_ov_status_line(model: dict, selected_id: str | None = None) -> str:
    downloaded = is_downloaded(model["local"])
    icon = "✅" if downloaded else "⬇️"
    if downloaded:
        size = human_bytes(dir_size_bytes(model["local"]))
    else:
        expected_size = int(model.get("expected_size_bytes", 0) or 0)
        size = f"~{human_bytes(expected_size)}" if expected_size > 0 else "—"
    selected = " (selected)" if selected_id and model["id"] == selected_id else ""
    return f"{icon} {model['display']} [{model['id']}] ({size}){selected}"


def list_whisper_ov_models(models: list[dict], selected_id: str | None = None) -> None:
    print("\nWhisper OpenVINO models:\n")
    if not models:
        print("(none configured)")
        print(f"Edit {WHISPER_OV_MODELS_FILE} or use /whisper_add\n")
        return
    for i, model in enumerate(models, 1):
        print(f"  {i}) {whisper_ov_status_line(model, selected_id)}")
    print("")


def choose_whisper_ov_model_interactive(models: list[dict], selected_id: str | None = None, allow_download: bool = True) -> dict | None:
    list_whisper_ov_models(models, selected_id)
    if not models:
        return None
    print("  0) Cancel\n")
    while True:
        choice = input("Option: ").strip()
        if choice == "0":
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            model = models[int(choice) - 1]
            if allow_download and not is_downloaded(model["local"]):
                download_whisper_ov_model(model)
            return model
        print("Invalid option.")


def ov_tts_to_storage_entry(model: dict) -> dict:
    payload = {
        "id": model["id"],
        "display": model["display"],
        "repo_url": model["repo_url"],
        "local_dir": model["local"].name,
    }
    expected_bytes = int(model.get("expected_size_bytes", 0) or 0)
    if expected_bytes > 0:
        payload["expected_size_bytes"] = expected_bytes
    return payload


def parse_ov_tts_entry(entry: dict) -> dict | None:
    if not isinstance(entry, dict):
        return None
    model_id = str(entry.get("id", "")).strip()
    display = str(entry.get("display", "")).strip()
    repo_url = str(entry.get("repo_url", "")).strip()
    local_dir = str(entry.get("local_dir", "")).strip()
    if not model_id or not display or not repo_url:
        return None
    if not local_dir:
        local_dir = slug_from_repo(repo_url)
    expected_size_bytes = 0
    if isinstance(entry.get("expected_size_bytes"), int):
        expected_size_bytes = int(entry.get("expected_size_bytes") or 0)
    elif isinstance(entry.get("expected_size_mb"), (int, float)):
        expected_size_bytes = int(float(entry.get("expected_size_mb")) * 1024 * 1024)
    if expected_size_bytes <= 0:
        expected_size_bytes = OV_TTS_EXPECTED_SIZE_BYTES.get(normalize_hf_repo_id(repo_url).lower(), 0)
    return {
        "id": model_id,
        "display": display,
        "repo_url": repo_url,
        "local": CACHE_DIR / local_dir,
        "expected_size_bytes": expected_size_bytes,
    }


def save_ov_tts_models(models: list[dict]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = [ov_tts_to_storage_entry(m) for m in models]
    OV_TTS_MODELS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_ov_tts_models() -> list[dict]:
    defaults = []
    for entry in DEFAULT_OV_TTS_MODELS_DATA:
        parsed = parse_ov_tts_entry(entry)
        if parsed is not None:
            defaults.append(parsed)

    if OV_TTS_MODELS_FILE.exists():
        try:
            raw = json.loads(OV_TTS_MODELS_FILE.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                parsed = [parse_ov_tts_entry(x) for x in raw]
                models = [x for x in parsed if x is not None]
                if models:
                    # Migration for previous invalid default Bark repo id.
                    changed = False
                    for model in models:
                        if model.get("id") == "bark-small-ov":
                            repo_norm = normalize_hf_repo_id(str(model.get("repo_url", ""))).lower()
                            if repo_norm == "openvino/bark-small":
                                model["repo_url"] = "suno/bark-small"
                                changed = True
                    seen_ids = {m["id"] for m in models}
                    for default_model in defaults:
                        if default_model["id"] not in seen_ids:
                            models.append(default_model)
                            changed = True
                    if changed:
                        save_ov_tts_models(models)
                    return models
        except Exception:
            pass
    save_ov_tts_models(defaults)
    return defaults


def add_ov_tts_model_interactive(models: list[dict]) -> dict | None:
    print("\nAdd OpenVINO TTS model\n")
    model_id = input("Model id (unique): ").strip()
    if not model_id:
        print("\nCancelled: model id is required.\n")
        return None
    if any(m["id"] == model_id for m in models):
        print("\n⚠️ A model with that id already exists.\n")
        return None
    display = input("Display name: ").strip()
    if not display:
        print("\nCancelled: display name is required.\n")
        return None
    repo_url = input("HF repo URL/id for OpenVINO TTS model: ").strip()
    if not repo_url:
        print("\nCancelled: repo URL/id is required.\n")
        return None
    default_local = slug_from_repo(repo_url)
    local_dir = input(f"Local folder [{default_local}]: ").strip() or default_local
    model = {
        "id": model_id,
        "display": display,
        "repo_url": repo_url,
        "local": CACHE_DIR / local_dir,
        "expected_size_bytes": OV_TTS_EXPECTED_SIZE_BYTES.get(normalize_hf_repo_id(repo_url).lower(), 0),
    }
    models.append(model)
    save_ov_tts_models(models)
    print(f"\n✅ OpenVINO TTS model added: {display} ({model_id})\n")
    return model


def download_ov_tts_model(model: dict) -> None:
    load_hf_token()
    model["local"].parent.mkdir(parents=True, exist_ok=True)
    repo_id = normalize_hf_repo_id(model["repo_url"])
    print(f"\n📥 Downloading OpenVINO TTS model: {model['repo_url']}")
    print(f"   -> destination: {model['local']}\n")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(model["local"]),
        local_dir_use_symlinks=False,
        resume_download=True,
        token=os.environ.get("HF_TOKEN") or None,
    )
    print("\n✅ OpenVINO TTS model download complete.\n")


def ov_tts_status_line(model: dict, selected_id: str | None = None) -> str:
    downloaded = is_downloaded(model["local"])
    icon = "✅" if downloaded else "⬇️"
    if downloaded:
        size = human_bytes(dir_size_bytes(model["local"]))
    else:
        expected_size = int(model.get("expected_size_bytes", 0) or 0)
        size = f"~{human_bytes(expected_size)}" if expected_size > 0 else "—"
    selected = " (selected)" if selected_id and model["id"] == selected_id else ""
    return f"{icon} {model['display']} [{model['id']}] ({size}){selected}"


def list_ov_tts_models(models: list[dict], selected_id: str | None = None) -> None:
    print("\nOpenVINO TTS models (verified):\n")
    if not models:
        print("(none configured)")
        print(f"Edit {OV_TTS_MODELS_FILE} or use /openvino_tts_add\n")
        return
    for i, model in enumerate(models, 1):
        print(f"  {i}) {ov_tts_status_line(model, selected_id)}")
    print("")


def choose_ov_tts_model_interactive(models: list[dict], selected_id: str | None = None, allow_download: bool = True) -> dict | None:
    list_ov_tts_models(models, selected_id)
    if not models:
        return None
    print("  0) Cancel\n")
    while True:
        choice = input("Option: ").strip()
        if choice == "0":
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            model = models[int(choice) - 1]
            if allow_download and not is_downloaded(model["local"]):
                download_ov_tts_model(model)
            return model
        print("Invalid option.")


def _parse_tts_repo_entry(entry: dict, expected_map: dict[str, int]) -> dict | None:
    if not isinstance(entry, dict):
        return None
    model_id = str(entry.get("id", "")).strip()
    display = str(entry.get("display", "")).strip()
    repo_url = str(entry.get("repo_url", "")).strip()
    local_dir = str(entry.get("local_dir", "")).strip()
    if not model_id or not display or not repo_url:
        return None
    if not local_dir:
        local_dir = slug_from_repo(repo_url)
    expected_size_bytes = 0
    if isinstance(entry.get("expected_size_bytes"), int):
        expected_size_bytes = int(entry.get("expected_size_bytes") or 0)
    if expected_size_bytes <= 0:
        expected_size_bytes = int(expected_map.get(normalize_hf_repo_id(repo_url).lower(), 0))
    return {
        "id": model_id,
        "display": display,
        "repo_url": repo_url,
        "local": CACHE_DIR / local_dir,
        "expected_size_bytes": expected_size_bytes,
    }


def _load_tts_repo_models(file_path: Path, defaults: list[dict], expected_map: dict[str, int]) -> list[dict]:
    if file_path.exists():
        try:
            raw = json.loads(file_path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                parsed = [_parse_tts_repo_entry(x, expected_map) for x in raw]
                models = [x for x in parsed if x is not None]
                if models:
                    return models
        except Exception:
            pass
    parsed_defaults = [_parse_tts_repo_entry(x, expected_map) for x in defaults]
    models = [x for x in parsed_defaults if x is not None]
    file_path.write_text(
        json.dumps(
            [
                {
                    "id": m["id"],
                    "display": m["display"],
                    "repo_url": m["repo_url"],
                    "local_dir": m["local"].name,
                    "expected_size_bytes": int(m.get("expected_size_bytes", 0) or 0),
                }
                for m in models
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    return models


def load_kokoro_models() -> list[dict]:
    expected = {"magicunicorn/kokoro-tts-intel": 340 * 1024 * 1024}
    return _load_tts_repo_models(KOKORO_MODELS_FILE, DEFAULT_KOKORO_MODELS_DATA, expected)


def load_babelvox_models() -> list[dict]:
    expected = {"djwarf/babelvox-openvino-int8": 2800 * 1024 * 1024}
    return _load_tts_repo_models(BABELVOX_MODELS_FILE, DEFAULT_BABELVOX_MODELS_DATA, expected)


def _status_line_repo_model(model: dict, selected_id: str | None = None) -> str:
    downloaded = is_repo_downloaded(model["local"])
    icon = "✅" if downloaded else "⬇️"
    size = human_bytes(dir_size_bytes(model["local"])) if downloaded else f"~{human_bytes(int(model.get('expected_size_bytes', 0) or 0))}"
    selected = " (selected)" if selected_id and model["id"] == selected_id else ""
    return f"{icon} {model['display']} [{model['id']}] ({size}){selected}"


def _list_repo_models(title: str, models: list[dict], selected_id: str | None = None) -> None:
    print(f"\n{title}\n")
    for i, model in enumerate(models, 1):
        print(f"  {i}) {_status_line_repo_model(model, selected_id)}")
    print("")


def _choose_repo_model_interactive(title: str, models: list[dict], selected_id: str | None = None) -> dict | None:
    _list_repo_models(title, models, selected_id)
    if not models:
        return None
    print("  0) Cancel\n")
    while True:
        choice = input("Option: ").strip()
        if choice == "0":
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            return models[int(choice) - 1]
        print("Invalid option.")


def has_openvino_tts_artifacts(model_dir: Path) -> bool:
    if not model_dir.exists() or not model_dir.is_dir():
        return False
    xml_files = list(model_dir.rglob("*.xml"))
    bin_files = list(model_dir.rglob("*.bin"))
    return bool(xml_files and bin_files)


def apply_openvino_tts_postprocess(audio, np_mod, config: dict):
    speed = float(config.get("openvino_tts_speed", 1.0) or 1.0)
    gain = float(config.get("openvino_tts_gain", 1.0) or 1.0)

    if speed < 0.5:
        speed = 0.5
    if speed > 2.0:
        speed = 2.0
    if gain < 0.1:
        gain = 0.1
    if gain > 3.0:
        gain = 3.0

    out = np_mod.array(audio, dtype=np_mod.float32).reshape(-1)
    if out.size == 0:
        return out

    if abs(speed - 1.0) > 1e-4:
        new_len = max(1, int(out.size / speed))
        x_old = np_mod.arange(out.size, dtype=np_mod.float32)
        x_new = np_mod.linspace(0, out.size - 1, new_len, dtype=np_mod.float32)
        out = np_mod.interp(x_new, x_old, out).astype(np_mod.float32)

    if abs(gain - 1.0) > 1e-4:
        out = np_mod.clip(out * gain, -1.0, 1.0).astype(np_mod.float32)

    return out


def find_espeak_executable() -> str | None:
    for candidate in ("espeak-ng", "espeak"):
        found = shutil.which(candidate)
        if found:
            return found
    return None


def print_espeak_install_suggestion() -> None:
    if IS_WINDOWS:
        print("Install suggestion (Windows): winget install eSpeakNG.eSpeakNG\n")
    elif IS_LINUX:
        print("Install suggestion (Linux): install the 'espeak-ng' package with your system package manager.\n")
    else:
        print("Install suggestion: install 'espeak-ng' and ensure it is available on PATH.\n")


def initialize_native_voice_engine(config: dict):
    if not platform_supports_native_voices(PLATFORM_NAME):
        return None, None, "Native Windows voices are unavailable on this platform."
    try:
        speaker = __import__("win32com.client", fromlist=["Dispatch"]).Dispatch("SAPI.SpVoice")
        voices = speaker.GetVoices()
        if voices.Count == 0:
            raise RuntimeError("No SAPI voices available")
        apply_voice_config(speaker, voices, config)
        return speaker, voices, None
    except Exception as exc:
        return None, None, str(exc)


def ensure_camera_runtime(camera_runtime: dict) -> bool:
    if camera_runtime.get("cv2") is None:
        camera_runtime["cv2"] = ensure_dependency("cv2", "opencv-python", "OpenCV")
        if camera_runtime["cv2"] is None:
            return False
    return True


def load_vision_labels(labels_path: str) -> list[str]:
    path = Path(str(labels_path or "").strip())
    if not path.exists():
        return []
    try:
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception:
        return []


def vision_model_to_runtime_entry(entry: dict) -> dict | None:
    if not isinstance(entry, dict):
        return None
    model_id = str(entry.get("id", "")).strip()
    display = str(entry.get("display", "")).strip()
    xml_url = str(entry.get("xml_url", "")).strip()
    bin_url = str(entry.get("bin_url", "")).strip()
    local_dir = str(entry.get("local_dir", "")).strip()
    if not all([model_id, display, xml_url, bin_url, local_dir]):
        return None
    model_dir = CACHE_DIR / "vision_models" / local_dir
    return {
        "id": model_id,
        "display": display,
        "xml_url": xml_url,
        "bin_url": bin_url,
        "local_dir": local_dir,
        "local": model_dir,
        "xml_path": model_dir / f"{model_id}.xml",
        "bin_path": model_dir / f"{model_id}.bin",
        "labels": list(entry.get("labels", [])) if isinstance(entry.get("labels", []), list) else [],
        "description": str(entry.get("description", "")).strip(),
    }


def load_vision_models() -> list[dict]:
    if not VISION_MODELS_FILE.exists():
        return []
    try:
        raw = json.loads(VISION_MODELS_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        print_error_red(f"ERROR: Could not read vision model catalog: {exc}")
        return []
    if not isinstance(raw, list):
        return []
    models: list[dict] = []
    for entry in raw:
        parsed = vision_model_to_runtime_entry(entry)
        if parsed is not None:
            models.append(parsed)
    return models


def is_valid_openvino_xml_file(path: Path) -> bool:
    if not path.exists() or path.suffix.lower() != ".xml":
        return False
    try:
        head = path.read_text(encoding="utf-8", errors="ignore")[:256].lstrip()
    except Exception:
        return False
    if "<!DOCTYPE html" in head or "<html" in head.lower():
        return False
    return "<net" in head or head.startswith("<?xml")


def is_vision_model_downloaded(model: dict) -> bool:
    return is_valid_openvino_xml_file(model["xml_path"]) and model["bin_path"].exists()


def download_vision_model(model: dict) -> bool:
    model["local"].mkdir(parents=True, exist_ok=True)
    print(f"\n📥 Downloading vision model: {model['display']}")
    try:
        urllib.request.urlretrieve(model["xml_url"], str(model["xml_path"]))
        urllib.request.urlretrieve(model["bin_url"], str(model["bin_path"]))
        if not is_valid_openvino_xml_file(model["xml_path"]):
            raise RuntimeError("downloaded XML is not a valid OpenVINO IR file")
        labels = list(model.get("labels", []))
        if labels:
            (model["local"] / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
        print("✅ Vision model download complete.\n")
        return True
    except Exception as exc:
        print_error_red(f"ERROR: Failed to download vision model: {exc}")
        return False


def vision_model_status_line(model: dict, selected_id: str | None = None) -> str:
    icon = "✅" if is_vision_model_downloaded(model) else "⬇️"
    selected = " (current)" if selected_id and model["id"] == selected_id else ""
    return f"{icon} {model['display']} [{model['id']}]{selected}"


def list_vision_models(models: list[dict], selected_id: str | None = None) -> None:
    print("\nVision models:\n")
    if not models:
        print("No vision models are configured.\n")
        return
    for idx, model in enumerate(models, 1):
        print(f"  {idx}) {vision_model_status_line(model, selected_id)}")
        if model.get("description"):
            print(f"      {model['description']}")
    print("")


def choose_vision_model_interactive(models: list[dict], selected_id: str | None = None, allow_download: bool = True) -> dict | None:
    if not models:
        print("\nNo vision models available.\n")
        return None
    list_vision_models(models, selected_id)
    while True:
        value = input("Choose vision model number (or 'cancel'): ").strip().lower()
        if value == "cancel":
            return None
        if not value.isdigit() or not (1 <= int(value) <= len(models)):
            print("Invalid option.")
            continue
        selected = models[int(value) - 1]
        if not is_vision_model_downloaded(selected):
            if not allow_download:
                print("That model is not downloaded yet.")
                continue
            if not download_vision_model(selected):
                return None
        return selected


def choose_vision_device_interactive(selected_device: str | None = None) -> str | None:
    options = ["CPU", "GPU", "NPU", "AUTO"]
    current = str(selected_device or "AUTO").strip().upper() or "AUTO"
    print("\nVision devices:\n")
    for idx, device in enumerate(options, 1):
        marker = " (current)" if device == current else ""
        print(f"  {idx}) {device}{marker}")
    print("")
    while True:
        value = input("Choose vision device number (or 'cancel'): ").strip().lower()
        if value == "cancel":
            return None
        if not value.isdigit() or not (1 <= int(value) <= len(options)):
            print("Invalid option.")
            continue
        return options[int(value) - 1]


def ensure_vision_runtime(camera_runtime: dict, config: dict) -> bool:
    if not ensure_camera_runtime(camera_runtime):
        return False
    model_path_value = str(config.get("vision_model_path", "")).strip()
    if not model_path_value:
        print_error_red("ERROR: vision_model_path is empty. Set it before enabling vision.")
        return False
    model_path = Path(model_path_value)
    if not model_path.exists():
        print_error_red(f"ERROR: Vision model file does not exist: {model_path}")
        return False

    active_key = (
        str(model_path.resolve()),
        str(config.get("vision_device", "AUTO")).strip().upper(),
    )
    if camera_runtime.get("vision_active_key") == active_key and camera_runtime.get("vision_compiled_model") is not None:
        return True

    try:
        ov_mod = importlib.import_module("openvino")
    except Exception:
        ov_mod = ensure_dependency("openvino", "openvino", "OpenVINO")
        if ov_mod is None:
            return False

    try:
        core = ov_mod.Core()
        model = core.read_model(str(model_path))
        compiled = core.compile_model(model, active_key[1])
    except Exception as exc:
        print_error_red(f"ERROR: Failed to initialize vision model: {exc}")
        return False

    input_tensor = compiled.input(0)
    shape = list(input_tensor.shape)
    if len(shape) != 4:
        print_error_red(f"ERROR: Unsupported vision model input shape: {shape}")
        return False

    camera_runtime["vision_core"] = core
    camera_runtime["vision_compiled_model"] = compiled
    camera_runtime["vision_input_shape"] = shape
    camera_runtime["vision_active_key"] = active_key
    labels = load_vision_labels(str(config.get("vision_labels_path", "")).strip())
    if not labels:
        selected_id = str(config.get("vision_model_id", "")).strip()
        if selected_id:
            selected = next((m for m in load_vision_models() if m["id"] == selected_id), None)
            if selected is not None:
                labels = list(selected.get("labels", []))
    camera_runtime["vision_labels"] = labels
    model_id = str(config.get("vision_model_id", "")).strip() or model_path.stem
    print(f"\n✅ Vision model loaded on {active_key[1]}: {model_id}\n")
    return True


def preprocess_vision_frame(frame, cv2_mod, np_mod, input_shape: list[int]):
    _, _, input_h, input_w = input_shape
    resized = cv2_mod.resize(frame, (int(input_w), int(input_h)))
    blob = resized.transpose(2, 0, 1)[None, ...].astype(np_mod.float32)
    return blob


def parse_detection_results(raw_output, frame_shape, threshold: float, labels: list[str]) -> list[dict]:
    np_mod = importlib.import_module("numpy")
    arr = np_mod.array(raw_output)
    detections: list[dict] = []
    frame_h, frame_w = frame_shape[:2]

    if arr.ndim == 4 and arr.shape[-1] >= 7:
        rows = arr.reshape(-1, arr.shape[-1])
        for row in rows:
            confidence = float(row[2])
            if confidence < threshold:
                continue
            class_id = int(row[1])
            x1 = max(0, min(frame_w - 1, int(float(row[3]) * frame_w)))
            y1 = max(0, min(frame_h - 1, int(float(row[4]) * frame_h)))
            x2 = max(0, min(frame_w - 1, int(float(row[5]) * frame_w)))
            y2 = max(0, min(frame_h - 1, int(float(row[6]) * frame_h)))
            label = labels[class_id] if 0 <= class_id < len(labels) else f"class_{class_id}"
            detections.append({"label": label, "score": confidence, "box": (x1, y1, x2, y2)})
        return detections

    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 2 and arr.shape[-1] >= 6:
        rows = arr
        for row in rows:
            confidence = float(row[4])
            if confidence < threshold:
                continue
            class_id = int(row[5])
            coords = [float(v) for v in row[:4]]
            normalized = max(coords) <= 1.5
            if normalized:
                x1 = max(0, min(frame_w - 1, int(coords[0] * frame_w)))
                y1 = max(0, min(frame_h - 1, int(coords[1] * frame_h)))
                x2 = max(0, min(frame_w - 1, int(coords[2] * frame_w)))
                y2 = max(0, min(frame_h - 1, int(coords[3] * frame_h)))
            else:
                x1 = max(0, min(frame_w - 1, int(coords[0])))
                y1 = max(0, min(frame_h - 1, int(coords[1])))
                x2 = max(0, min(frame_w - 1, int(coords[2])))
                y2 = max(0, min(frame_h - 1, int(coords[3])))
            label = labels[class_id] if 0 <= class_id < len(labels) else f"class_{class_id}"
            detections.append({"label": label, "score": confidence, "box": (x1, y1, x2, y2)})
        return detections

    return []


def annotate_frame_with_detections(frame, detections: list[dict], cv2_mod):
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = det["label"]
        score = det["score"]
        cv2_mod.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2_mod.putText(
            frame,
            f"{label} {score:.2f}",
            (x1, max(20, y1 - 8)),
            cv2_mod.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2_mod.LINE_AA,
        )
    return frame


def format_vision_debug_output(raw_output, detections: list[dict]) -> str:
    np_mod = importlib.import_module("numpy")
    arr = np_mod.array(raw_output)
    payload = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "detection_count": len(detections),
        "detections": detections,
    }
    text = json.dumps(payload, ensure_ascii=False)
    if len(text) > 2000:
        text = text[:2000] + "...(truncated)"
    return text


def format_vad_debug_output(payload: dict) -> str:
    text = json.dumps(payload, ensure_ascii=False)
    if len(text) > 1200:
        text = text[:1200] + "...(truncated)"
    return text


def show_llm_context(history: list[str], config: dict) -> None:
    memory = [str(item).strip() for item in list(history or []) if str(item).strip()]
    system_prompt = str(config.get("system_prompt", "")).strip()
    max_new_tokens = int(config.get("max_new_tokens", 300))
    max_words_estimate = max(1, int(max_new_tokens * 0.65))
    system_limit_note = f"Anexo: No respondas con mas de {max_words_estimate} palabras."
    effective_system_prompt = (
        f"{system_prompt}\n{system_limit_note}" if system_prompt else system_limit_note
    )

    print("\nCurrent LLM context:\n")
    print("System:")
    print("-----")
    print(effective_system_prompt)
    print("-----\n")
    print(f"Memory entries: {len(memory)}")
    if not memory:
        print("(history is empty)\n")
        return
    for idx, item in enumerate(memory, 1):
        print(f"{idx:>3}. {item}")
    print("")


def should_emit_vision_log(camera_runtime: dict, now_ts: float | None = None) -> bool:
    if not bool(camera_runtime.get("vision_log_enabled", False)):
        return False
    if now_ts is None:
        now_ts = time.monotonic()
    last_ts = float(camera_runtime.get("vision_log_last_ts", 0.0))
    interval_s = max(0.1, float(camera_runtime.get("vision_log_interval_s", 1.0)))
    if (now_ts - last_ts) < interval_s:
        return False
    camera_runtime["vision_log_last_ts"] = now_ts
    return True


def should_emit_auto_listen_log(auto_listen_runtime: dict, now_ts: float | None = None) -> bool:
    config = auto_listen_runtime.get("voice_config")
    if not isinstance(config, dict) or not bool(config.get("vision_log_enabled", False)):
        return False
    if now_ts is None:
        now_ts = time.monotonic()
    last_ts = float(auto_listen_runtime.get("vad_log_last_ts", 0.0))
    interval_s = max(0.1, float(config.get("vision_log_interval_s", 1.0)))
    if (now_ts - last_ts) < interval_s:
        return False
    auto_listen_runtime["vad_log_last_ts"] = now_ts
    return True


def should_process_vision_events(camera_runtime: dict, now_ts: float | None = None) -> bool:
    if not bool(camera_runtime.get("vision_event_processing_enabled", True)):
        return False
    if now_ts is None:
        now_ts = time.monotonic()
    last_ts = float(camera_runtime.get("vision_event_last_ts", 0.0))
    interval_s = max(0.1, float(camera_runtime.get("vision_log_interval_s", 1.0)))
    if (now_ts - last_ts) < interval_s:
        return False
    camera_runtime["vision_event_last_ts"] = now_ts
    return True


def load_vision_event_responses() -> dict[str, list[str]]:
    defaults = {
        "first_person_joined": ["Hola, como andas?"],
        "more_people_joined": ["Se sumo una persona, ahora hay: {count}"],
        "fewer_people_left": ["Se fue una persona."],
        "alone_again": ["Me dejaron solo."],
    }
    if not VISION_EVENT_RESPONSES_FILE.exists():
        return defaults
    try:
        raw = json.loads(VISION_EVENT_RESPONSES_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        print_error_red(f"ERROR: Could not read vision event responses: {exc}")
        return defaults
    if not isinstance(raw, dict):
        return defaults
    merged: dict[str, list[str]] = {}
    for key, fallback in defaults.items():
        values = raw.get(key, fallback)
        if isinstance(values, list):
            cleaned = [str(item).strip() for item in values if str(item).strip()]
            merged[key] = cleaned or fallback
        else:
            merged[key] = fallback
    return merged


def choose_vision_event_response(category: str, count: int) -> str | None:
    responses = load_vision_event_responses()
    options = list(responses.get(category, []))
    if not options:
        return None
    template = random.choice(options)
    try:
        return str(template).format(count=count)
    except Exception:
        return str(template)


def set_robot_face_gesture(camera_runtime: dict, gesture: str, duration_s: float = 1.2) -> None:
    camera_runtime["robot_face_gesture"] = str(gesture or "").strip()
    camera_runtime["robot_face_gesture_until"] = time.monotonic() + max(0.1, float(duration_s))


def handle_vision_tick(camera_runtime: dict, detections: list[dict]) -> str | None:
    current_count = len(detections)
    previous_count = int(camera_runtime.get("vision_last_detection_count", 0))
    camera_runtime["vision_last_detection_count"] = current_count

    if current_count == previous_count:
        return None
    if previous_count == 0 and current_count > 0:
        set_robot_face_gesture(camera_runtime, "join")
        if bool(camera_runtime.get("suppress_next_join_after_interrupt", False)):
            camera_runtime["suppress_next_join_after_interrupt"] = False
            return None
        return choose_vision_event_response("first_person_joined", current_count)
    if current_count > previous_count:
        set_robot_face_gesture(camera_runtime, "join")
        return choose_vision_event_response("more_people_joined", current_count)
    if current_count == 0:
        set_robot_face_gesture(camera_runtime, "leave")
        if is_audio_playback_active() or is_tts_active():
            camera_runtime["suppress_next_join_after_interrupt"] = True
            return "__INTERRUPT_AUDIO__:me cayo"
        return choose_vision_event_response("alone_again", current_count)
    set_robot_face_gesture(camera_runtime, "leave")
    return choose_vision_event_response("fewer_people_left", current_count)


def _vision_event_tts_worker(camera_runtime: dict) -> None:
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


def emit_vision_event_message(camera_runtime: dict, message: str) -> None:
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


def interrupt_audio_and_speak(camera_runtime: dict, message: str) -> None:
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
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and (is_audio_playback_active() or is_tts_active()):
        time.sleep(0.02)
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


def open_camera_capture(cv2_mod, device_index: int):
    backends: list[int | None] = [None]
    if PLATFORM_NAME == "windows":
        for attr_name in ("CAP_DSHOW", "CAP_MSMF"):
            backend = getattr(cv2_mod, attr_name, None)
            if backend is not None:
                backends.insert(0, backend)
    elif PLATFORM_NAME == "linux":
        backend = getattr(cv2_mod, "CAP_V4L2", None)
        if backend is not None:
            backends.insert(0, backend)

    last_cap = None
    for backend in backends:
        cap = cv2_mod.VideoCapture(device_index) if backend is None else cv2_mod.VideoCapture(device_index, backend)
        last_cap = cap
        try:
            if cap is not None and cap.isOpened():
                return cap
        except Exception:
            pass
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
    return last_cap


def list_camera_devices(camera_runtime: dict, max_devices: int = 10) -> list[tuple[int, str]]:
    if not ensure_camera_runtime(camera_runtime):
        return []
    cv2_mod = camera_runtime["cv2"]
    found: list[tuple[int, str]] = []
    for idx in range(max_devices):
        cap = open_camera_capture(cv2_mod, idx)
        try:
            if cap is None or not cap.isOpened():
                continue
            ok, _frame = cap.read()
            if ok:
                found.append((idx, f"Camera {idx}"))
        finally:
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass
    return found


def stop_camera_preview(camera_runtime: dict) -> None:
    stop_qt_panel(camera_runtime)
    stop_event = camera_runtime.get("stop_event")
    thread = camera_runtime.get("thread")
    event_thread = camera_runtime.get("vision_event_thread")
    if stop_event is not None:
        stop_event.set()
    if thread is not None and thread.is_alive():
        thread.join(timeout=2)
    if event_thread is not None and event_thread.is_alive():
        event_thread.join(timeout=2)
    camera_runtime["thread"] = None
    camera_runtime["vision_event_thread"] = None
    camera_runtime["vision_event_queue"] = None
    camera_runtime["stop_event"] = None
    camera_runtime["active_device_index"] = None
    camera_runtime["camera_enabled"] = False
    camera_runtime["panel_enabled"] = False
    camera_runtime["panel_button_rects"] = {}
    camera_runtime["panel_action_queue"] = None


def stop_qt_panel(camera_runtime: dict) -> None:
    stop_event = camera_runtime.get("qt_panel_stop_event")
    if stop_event is not None:
        with contextlib.suppress(Exception):
            stop_event.set()
    proc = camera_runtime.get("qt_panel_process")
    if proc is not None and proc.is_alive():
        proc.join(timeout=1.5)
        if proc.is_alive():
            with contextlib.suppress(Exception):
                proc.terminate()
            proc.join(timeout=1.0)
    camera_runtime["qt_panel_process"] = None
    camera_runtime["qt_panel_state_queue"] = None
    camera_runtime["qt_panel_action_queue"] = None
    camera_runtime["qt_panel_stop_event"] = None
    camera_runtime["panel_backend"] = None


def draw_robot_face(frame, np_mod, cv2_mod, rect: tuple[int, int, int, int], *, speaking: bool, listening: bool, gesture: str, visual_effects_enabled: bool) -> None:
    x, y, w, h = rect
    face_bg = (32, 38, 46)
    accent = (80, 190, 255)
    accent_dim = (70, 110, 140)
    metal = (180, 190, 205)
    dark = (18, 24, 28)
    frame[y:y + h, x:x + w] = face_bg

    t = time.monotonic()
    ear_w = 28
    ear_h = 110
    ear_base_y = y + 96
    ear_offset = 0
    if visual_effects_enabled and listening:
        ear_offset = int(10 * (0.5 + 0.5 * math.sin(t * 14.0)))
    left_ear = ((x + 16, ear_base_y - ear_offset), (x + 16 + ear_w, ear_base_y + ear_h - ear_offset))
    right_ear = ((x + w - 16 - ear_w, ear_base_y - ear_offset), (x + w - 16, ear_base_y + ear_h - ear_offset))
    cv2_mod.rectangle(frame, left_ear[0], left_ear[1], accent if listening else accent_dim, -1)
    cv2_mod.rectangle(frame, right_ear[0], right_ear[1], accent if listening else accent_dim, -1)

    head_x1 = x + 48
    head_y1 = y + 52
    head_x2 = x + w - 48
    head_y2 = y + h - 44
    cv2_mod.rectangle(frame, (head_x1, head_y1), (head_x2, head_y2), metal, -1)
    cv2_mod.rectangle(frame, (head_x1 + 10, head_y1 + 10), (head_x2 - 10, head_y2 - 10), (95, 102, 112), -1)

    eye_y = y + 150
    eye_w = 52
    eye_h = 28
    blink = visual_effects_enabled and (math.sin(t * 1.4) > 0.985)
    if gesture == "join":
        eye_h = 34
    elif gesture == "leave":
        eye_h = 18
    if blink:
        eye_h = 6
    left_eye_center = (x + 112, eye_y)
    right_eye_center = (x + w - 112, eye_y)
    cv2_mod.ellipse(frame, left_eye_center, (eye_w, eye_h), 0, 0, 360, dark, -1)
    cv2_mod.ellipse(frame, right_eye_center, (eye_w, eye_h), 0, 0, 360, dark, -1)
    if not blink:
        pupil_shift = 0
        if gesture == "join":
            pupil_shift = -4
        elif gesture == "leave":
            pupil_shift = 4
        cv2_mod.circle(frame, (left_eye_center[0] + pupil_shift, left_eye_center[1]), 8, accent, -1)
        cv2_mod.circle(frame, (right_eye_center[0] + pupil_shift, right_eye_center[1]), 8, accent, -1)

    mouth_x1 = x + 96
    mouth_x2 = x + w - 96
    mouth_cy = y + 286
    mouth_open = 10
    mouth_curve = 0
    if visual_effects_enabled and speaking:
        phase_a = 0.5 + 0.5 * math.sin(t * 13.0)
        phase_b = 0.5 + 0.5 * math.sin(t * 21.0 + 0.9)
        mouth_open = 6 + int(10 * phase_a + 8 * phase_b)
        mouth_curve = int(6 * math.sin(t * 9.0 + 0.4))
    elif gesture == "join":
        mouth_open = 18
        mouth_curve = -6
    elif gesture == "leave":
        mouth_open = 4
        mouth_curve = 6
    mouth_color = accent if speaking else accent_dim
    outer_pts = np_mod.array(
        [
            [mouth_x1, mouth_cy - mouth_open + mouth_curve],
            [mouth_x2, mouth_cy - mouth_open - mouth_curve],
            [mouth_x2, mouth_cy + mouth_open - mouth_curve],
            [mouth_x1, mouth_cy + mouth_open + mouth_curve],
        ],
        dtype=np_mod.int32,
    )
    inner_pts = np_mod.array(
        [
            [mouth_x1 + 8, mouth_cy - max(2, mouth_open - 6) + mouth_curve // 2],
            [mouth_x2 - 8, mouth_cy - max(2, mouth_open - 6) - mouth_curve // 2],
            [mouth_x2 - 8, mouth_cy + max(2, mouth_open - 6) - mouth_curve // 2],
            [mouth_x1 + 8, mouth_cy + max(2, mouth_open - 6) + mouth_curve // 2],
        ],
        dtype=np_mod.int32,
    )
    cv2_mod.fillConvexPoly(frame, outer_pts, dark)
    cv2_mod.fillConvexPoly(frame, inner_pts, mouth_color)

    badge_color = accent if visual_effects_enabled else accent_dim
    cv2_mod.circle(frame, (x + (w // 2), y + h - 86), 18, badge_color, -1)
    cv2_mod.putText(frame, "R", (x + (w // 2) - 8, y + h - 79), cv2_mod.FONT_HERSHEY_SIMPLEX, 0.7, dark, 2, cv2_mod.LINE_AA)


def build_panel_frame(np_mod, cv2_mod, camera_frame, camera_enabled: bool, camera_device_index: int | None, metrics: list[dict], button_specs: list[dict], show_audio_monitor: bool, face_state: dict):
    panel_w = 1440
    panel_h = 760
    frame = np_mod.zeros((panel_h, panel_w, 3), dtype=np_mod.uint8)
    frame[:, :] = (20, 20, 20)

    face_x = 24
    camera_y = 72
    face_w = 280
    camera_x = face_x + face_w + 24
    camera_w = 700
    camera_h = 616
    sidebar_x = camera_x + camera_w + 28
    sidebar_y = 72
    sidebar_w = panel_w - sidebar_x - 24

    frame[camera_y - 18:camera_y + camera_h + 18, face_x - 18:face_x + face_w + 18] = (32, 32, 32)
    frame[camera_y - 18:camera_y + camera_h + 18, camera_x - 18:camera_x + camera_w + 18] = (32, 32, 32)
    frame[sidebar_y - 18:panel_h - 24, sidebar_x - 18:panel_w - 24] = (28, 28, 28)

    cv2_mod.putText(frame, "Robot Control Panel", (24, 36), cv2_mod.FONT_HERSHEY_SIMPLEX, 1.0, (235, 235, 235), 2, cv2_mod.LINE_AA)
    cv2_mod.putText(frame, "Avatar", (face_x, 62), cv2_mod.FONT_HERSHEY_SIMPLEX, 0.7, (210, 210, 210), 1, cv2_mod.LINE_AA)
    cv2_mod.putText(frame, "Camera", (camera_x, 62), cv2_mod.FONT_HERSHEY_SIMPLEX, 0.7, (210, 210, 210), 1, cv2_mod.LINE_AA)
    cv2_mod.putText(frame, "Controls", (sidebar_x, 62), cv2_mod.FONT_HERSHEY_SIMPLEX, 0.7, (210, 210, 210), 1, cv2_mod.LINE_AA)
    draw_robot_face(
        frame,
        np_mod,
        cv2_mod,
        (face_x, camera_y, face_w, camera_h),
        speaking=bool(face_state.get("speaking", False)),
        listening=bool(face_state.get("listening", False)),
        gesture=str(face_state.get("gesture", "")),
        visual_effects_enabled=bool(face_state.get("visual_effects_enabled", True)),
    )

    if camera_frame is not None:
        resized = cv2_mod.resize(camera_frame, (camera_w, camera_h))
        frame[camera_y:camera_y + camera_h, camera_x:camera_x + camera_w] = resized
    else:
        frame[camera_y:camera_y + camera_h, camera_x:camera_x + camera_w] = (35, 35, 35)
        status = "Camera Off" if not camera_enabled else "Waiting for camera..."
        device_text = f"device={camera_device_index}" if camera_device_index is not None else "device=none"
        cv2_mod.putText(frame, status, (camera_x + 165, camera_y + 290), cv2_mod.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2, cv2_mod.LINE_AA)
        cv2_mod.putText(frame, device_text, (camera_x + 215, camera_y + 330), cv2_mod.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1, cv2_mod.LINE_AA)

    button_h = 34
    button_pitch = 40
    button_rects: dict[str, tuple[int, int, int, int]] = {}
    for idx, spec in enumerate(button_specs):
        x1 = sidebar_x
        y1 = sidebar_y + idx * button_pitch
        x2 = x1 + sidebar_w
        y2 = y1 + button_h
        active = bool(spec.get("active", False))
        color = (50, 145, 70) if active else (70, 70, 70)
        frame[y1:y2, x1:x2] = color
        cv2_mod.rectangle(frame, (x1, y1), (x2, y2), (110, 110, 110), 1)
        label = str(spec.get("label", "Toggle"))
        state_text = "ON" if active else "OFF"
        cv2_mod.putText(frame, f"{label}", (x1 + 12, y1 + 23), cv2_mod.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 1, cv2_mod.LINE_AA)
        cv2_mod.putText(frame, state_text, (x2 - 56, y1 + 23), cv2_mod.FONT_HERSHEY_SIMPLEX, 0.54, (240, 240, 240), 1, cv2_mod.LINE_AA)
        button_rects[str(spec.get("action", label))] = (x1, y1, x2, y2)

    metrics_y = sidebar_y + len(button_specs) * button_pitch + 12
    if show_audio_monitor:
        metrics_frame = build_audio_monitor_frame(np_mod, 0.0, 1.0, False, width=sidebar_w, height=250, metrics=metrics)
        frame[metrics_y:metrics_y + 250, sidebar_x:sidebar_x + sidebar_w] = metrics_frame
        cv2_mod.putText(frame, "Audio / VAD", (sidebar_x, metrics_y - 10), cv2_mod.FONT_HERSHEY_SIMPLEX, 0.65, (210, 210, 210), 1, cv2_mod.LINE_AA)
        for idx, metric in enumerate(metrics):
            label_y = metrics_y + 40 + idx * 42
            cv2_mod.putText(
                frame,
                str(metric.get("label", "")),
                (sidebar_x + 8, label_y),
                cv2_mod.FONT_HERSHEY_SIMPLEX,
                0.42,
                (230, 230, 230),
                1,
                cv2_mod.LINE_AA,
            )
    else:
        frame[metrics_y:metrics_y + 250, sidebar_x:sidebar_x + sidebar_w] = (35, 35, 35)
        cv2_mod.putText(frame, "Audio / VAD", (sidebar_x, metrics_y - 10), cv2_mod.FONT_HERSHEY_SIMPLEX, 0.65, (210, 210, 210), 1, cv2_mod.LINE_AA)
        cv2_mod.putText(frame, "Audio monitor is off", (sidebar_x + 70, metrics_y + 120), cv2_mod.FONT_HERSHEY_SIMPLEX, 0.8, (160, 160, 160), 2, cv2_mod.LINE_AA)
    return frame, button_rects


def _put_latest_panel_state(state_queue, payload: dict) -> None:
    if state_queue is None:
        return
    try:
        while True:
            state_queue.get_nowait()
    except Exception:
        pass
    with contextlib.suppress(Exception):
        state_queue.put_nowait(payload)


def _qt_panel_process_main(state_queue, action_queue, stop_event) -> None:
    from PySide6 import QtCore, QtGui, QtWidgets

    class RobotFaceWidget(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.setMinimumSize(240, 420)
            self._speaking = False
            self._listening = False
            self._gesture = ""
            self._visual_effects_enabled = True

        def set_state(self, speaking: bool, listening: bool, gesture: str, visual_effects_enabled: bool) -> None:
            self._speaking = bool(speaking)
            self._listening = bool(listening)
            self._gesture = str(gesture or "")
            self._visual_effects_enabled = bool(visual_effects_enabled)
            self.update()

        def paintEvent(self, _event):
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.Antialiasing)
            rect = self.rect()
            painter.fillRect(rect, QtGui.QColor(22, 24, 28))

            face_bg = QtGui.QColor(32, 38, 46)
            accent = QtGui.QColor(80, 190, 255)
            accent_dim = QtGui.QColor(70, 110, 140)
            metal = QtGui.QColor(180, 190, 205)
            dark = QtGui.QColor(18, 24, 28)

            t = time.monotonic()
            x = 16
            y = 16
            w = rect.width() - 32
            h = rect.height() - 32
            painter.fillRect(QtCore.QRect(x, y, w, h), face_bg)

            ear_offset = 0
            if self._visual_effects_enabled and self._listening:
                ear_offset = int(8 * (0.5 + 0.5 * math.sin(t * 14.0)))
            ear_color = accent if self._listening else accent_dim
            painter.fillRect(QtCore.QRect(x + 8, y + 90 - ear_offset, 22, 90), ear_color)
            painter.fillRect(QtCore.QRect(x + w - 30, y + 90 - ear_offset, 22, 90), ear_color)

            head_rect = QtCore.QRect(x + 36, y + 40, w - 72, h - 84)
            painter.fillRect(head_rect, metal)
            painter.fillRect(head_rect.adjusted(8, 8, -8, -8), QtGui.QColor(95, 102, 112))

            blink = self._visual_effects_enabled and (math.sin(t * 1.4) > 0.985)
            eye_h = 18
            if self._gesture == "join":
                eye_h = 22
            elif self._gesture == "leave":
                eye_h = 10
            if blink:
                eye_h = 4
            painter.setBrush(dark)
            painter.setPen(QtCore.Qt.NoPen)
            left_eye = QtCore.QRect(x + 74, y + 118, 64, eye_h)
            right_eye = QtCore.QRect(x + w - 138, y + 118, 64, eye_h)
            painter.drawRoundedRect(left_eye, 12, 12)
            painter.drawRoundedRect(right_eye, 12, 12)
            if not blink:
                painter.setBrush(accent)
                painter.drawEllipse(QtCore.QPoint(x + 106, y + 127), 6, 6)
                painter.drawEllipse(QtCore.QPoint(x + w - 106, y + 127), 6, 6)

            mouth_open = 8
            mouth_curve = 0
            if self._visual_effects_enabled and self._speaking:
                phase_a = 0.5 + 0.5 * math.sin(t * 13.0)
                phase_b = 0.5 + 0.5 * math.sin(t * 21.0 + 0.9)
                mouth_open = 6 + int(8 * phase_a + 6 * phase_b)
                mouth_curve = int(5 * math.sin(t * 9.0 + 0.4))
            elif self._gesture == "join":
                mouth_open = 16
                mouth_curve = -5
            elif self._gesture == "leave":
                mouth_open = 4
                mouth_curve = 5
            painter.setBrush(dark)
            mouth_rect = QtCore.QRect(x + 82, y + 232 - mouth_open, w - 164, mouth_open * 2)
            painter.drawRoundedRect(mouth_rect, 10, 10)
            painter.setBrush(accent if self._speaking else accent_dim)
            inner = mouth_rect.adjusted(6, 4 + mouth_curve, -6, -4 - mouth_curve)
            painter.drawRoundedRect(inner, 8, 8)

            painter.setBrush(accent if self._visual_effects_enabled else accent_dim)
            painter.drawEllipse(QtCore.QPoint(x + w // 2, y + h - 46), 15, 15)
            painter.setPen(dark)
            font = painter.font()
            font.setBold(True)
            font.setPointSize(14)
            painter.setFont(font)
            painter.drawText(QtCore.QRect(x + w // 2 - 12, y + h - 59, 24, 24), QtCore.Qt.AlignCenter, "R")

    class RobotQtPanel(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Robot Control Panel (Qt)")
            self.resize(1380, 760)
            self._state = {}

            root = QtWidgets.QHBoxLayout(self)
            root.setContentsMargins(18, 18, 18, 18)
            root.setSpacing(18)

            left = QtWidgets.QVBoxLayout()
            left.setSpacing(12)
            root.addLayout(left, 3)

            self.face = RobotFaceWidget()
            left.addWidget(self.face, 2)

            self.camera_label = QtWidgets.QLabel("Camera Off")
            self.camera_label.setAlignment(QtCore.Qt.AlignCenter)
            self.camera_label.setMinimumSize(700, 420)
            self.camera_label.setStyleSheet("background:#232323; border:1px solid #444; color:#bdbdbd;")
            left.addWidget(self.camera_label, 3)

            right = QtWidgets.QVBoxLayout()
            right.setSpacing(10)
            root.addLayout(right, 2)

            title = QtWidgets.QLabel("Controls")
            title.setStyleSheet("color:#f0f0f0; font-size:18px; font-weight:600;")
            right.addWidget(title)

            self.buttons = {}
            for label, action in [
                ("Camera", "toggle_camera"),
                ("Vision", "toggle_vision"),
                ("Auto Listen", "toggle_auto_listen"),
                ("Audio", "toggle_audio"),
                ("Vision Events", "toggle_vision_events"),
                ("Log", "toggle_log"),
                ("Audio Monitor", "toggle_audio_monitor"),
                ("Visual Effects", "toggle_visual_effects"),
            ]:
                btn = QtWidgets.QPushButton(label)
                btn.setCheckable(True)
                btn.clicked.connect(lambda _checked=False, action=action: self._emit_action(action))
                btn.setStyleSheet("QPushButton{padding:10px; text-align:left;} QPushButton:checked{background:#3a7f4a; color:white;}")
                right.addWidget(btn)
                self.buttons[action] = btn

            exit_btn = QtWidgets.QPushButton("Exit")
            exit_btn.clicked.connect(lambda: self._emit_action("exit_program"))
            exit_btn.setStyleSheet("QPushButton{padding:10px; background:#7a3434; color:white;}")
            right.addWidget(exit_btn)

            metrics_title = QtWidgets.QLabel("Audio / VAD")
            metrics_title.setStyleSheet("color:#f0f0f0; font-size:16px; font-weight:600; margin-top:8px;")
            right.addWidget(metrics_title)

            self.metric_rows = []
            for _idx in range(6):
                row = QtWidgets.QWidget()
                layout = QtWidgets.QHBoxLayout(row)
                layout.setContentsMargins(0, 0, 0, 0)
                layout.setSpacing(8)
                label = QtWidgets.QLabel("")
                label.setMinimumWidth(120)
                label.setStyleSheet("color:#d8d8d8;")
                bar = QtWidgets.QProgressBar()
                bar.setRange(0, 100)
                bar.setTextVisible(False)
                bar.setStyleSheet("QProgressBar{background:#333; border:1px solid #555; height:14px;} QProgressBar::chunk{background:#00aaff;}")
                lamp = QtWidgets.QLabel("●")
                lamp.setStyleSheet("color:#4a4a4a; font-size:16px;")
                layout.addWidget(label)
                layout.addWidget(bar, 1)
                layout.addWidget(lamp)
                right.addWidget(row)
                self.metric_rows.append((label, bar, lamp, row))

            right.addStretch(1)

            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self._poll_state)
            self.timer.start(50)

        def closeEvent(self, event):
            self._emit_action("close_panel")
            super().closeEvent(event)

        def _emit_action(self, action: str) -> None:
            with contextlib.suppress(Exception):
                action_queue.put_nowait(str(action))

        def _poll_state(self) -> None:
            if stop_event.is_set():
                self.close()
                return
            latest = None
            while True:
                try:
                    latest = state_queue.get_nowait()
                except Exception:
                    break
            if latest is None:
                return
            self._state = latest
            self._apply_state(latest)

        def _apply_state(self, state: dict) -> None:
            face_state = state.get("face_state", {}) if isinstance(state, dict) else {}
            self.face.set_state(
                bool(face_state.get("speaking", False)),
                bool(face_state.get("listening", False)),
                str(face_state.get("gesture", "")),
                bool(face_state.get("visual_effects_enabled", True)),
            )
            frame_bytes = state.get("camera_jpeg")
            if frame_bytes:
                image = QtGui.QImage.fromData(frame_bytes, "JPG")
                if not image.isNull():
                    pix = QtGui.QPixmap.fromImage(image)
                    self.camera_label.setPixmap(pix.scaled(self.camera_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
                    self.camera_label.setText("")
            else:
                self.camera_label.setPixmap(QtGui.QPixmap())
                placeholder = state.get("camera_placeholder", "Camera Off")
                self.camera_label.setText(str(placeholder))

            buttons = {str(item.get("action")): bool(item.get("active", False)) for item in state.get("button_specs", [])}
            for action, btn in self.buttons.items():
                btn.blockSignals(True)
                btn.setChecked(buttons.get(action, False))
                btn.blockSignals(False)

            metrics = state.get("metrics", [])
            show_audio_monitor = bool(state.get("show_audio_monitor", False))
            for idx, (label, bar, lamp, row) in enumerate(self.metric_rows):
                if idx < len(metrics) and show_audio_monitor:
                    metric = metrics[idx]
                    label.setText(str(metric.get("label", "")))
                    bar.setValue(int(max(0.0, min(1.0, float(metric.get("value", 0.0)))) * 100.0))
                    lamp.setStyleSheet("color:#26c96f; font-size:16px;" if bool(metric.get("active", False)) else "color:#4a4a4a; font-size:16px;")
                    row.show()
                else:
                    row.hide()

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setStyle("Fusion")
    panel = RobotQtPanel()
    panel.show()
    app.exec()


def ensure_camera_worker(camera_runtime: dict, config: dict) -> bool:
    if not ensure_camera_runtime(camera_runtime):
        return False
    camera_runtime["voice_config"] = config
    if camera_runtime.get("thread") is not None and camera_runtime["thread"].is_alive():
        return True
    stop_event = threading.Event()
    camera_runtime["stop_event"] = stop_event
    camera_runtime["vision_event_queue"] = queue.Queue(maxsize=32)
    camera_runtime["panel_action_queue"] = queue.Queue(maxsize=32)
    event_thread = threading.Thread(target=_vision_event_tts_worker, args=(camera_runtime,), daemon=True)
    camera_runtime["vision_event_thread"] = event_thread
    event_thread.start()
    thread = threading.Thread(target=_camera_preview_worker, args=(camera_runtime, None), daemon=True)
    camera_runtime["thread"] = thread
    thread.start()
    return True


def start_qt_panel(camera_runtime: dict, config: dict) -> bool:
    pyside_mod = ensure_dependency("PySide6", "PySide6", "PySide6")
    if pyside_mod is None:
        return False
    if not ensure_camera_worker(camera_runtime, config):
        return False
    existing = camera_runtime.get("qt_panel_process")
    if existing is not None and existing.is_alive():
        camera_runtime["panel_backend"] = "qt"
        camera_runtime["panel_enabled"] = False
        return True
    ctx = mp.get_context("spawn")
    state_queue = ctx.Queue(maxsize=1)
    action_queue = ctx.Queue(maxsize=32)
    stop_event = ctx.Event()
    proc = ctx.Process(
        target=_qt_panel_process_main,
        args=(state_queue, action_queue, stop_event),
        daemon=True,
    )
    proc.start()
    camera_runtime["qt_panel_process"] = proc
    camera_runtime["qt_panel_state_queue"] = state_queue
    camera_runtime["qt_panel_action_queue"] = action_queue
    camera_runtime["qt_panel_stop_event"] = stop_event
    camera_runtime["panel_backend"] = "qt"
    camera_runtime["panel_enabled"] = False
    return True


def start_camera_panel(camera_runtime: dict, config: dict, backend: str = "opencv") -> bool:
    selected_backend = str(backend or "opencv").strip().lower()
    if selected_backend not in {"opencv", "qt"}:
        selected_backend = "opencv"
    config["panel_backend"] = selected_backend
    save_robot_config(config)
    if selected_backend == "qt":
        return start_qt_panel(camera_runtime, config)
    stop_qt_panel(camera_runtime)
    camera_runtime["panel_backend"] = "opencv"
    camera_runtime["panel_enabled"] = True
    return ensure_camera_worker(camera_runtime, config)


def set_camera_enabled(camera_runtime: dict, config: dict, enabled: bool) -> bool:
    camera_runtime["voice_config"] = config
    if not enabled:
        config["camera_enabled"] = False
        camera_runtime["camera_enabled"] = False
        camera_runtime["active_device_index"] = None
        save_robot_config(config)
        print("\n✅ camera = off\n")
        return True

    if not ensure_camera_runtime(camera_runtime):
        return False
    saved_device_index = int(config.get("camera_device_index", 0))
    cv2_mod = camera_runtime.get("cv2")
    device_index = None
    if cv2_mod is not None:
        saved_cap = open_camera_capture(cv2_mod, saved_device_index)
        try:
            if saved_cap is not None and saved_cap.isOpened():
                ok, _frame = saved_cap.read()
                if ok:
                    device_index = saved_device_index
        finally:
            with contextlib.suppress(Exception):
                if saved_cap is not None:
                    saved_cap.release()
    if device_index is None:
        devices = list_camera_devices(camera_runtime)
        if not devices:
            print_error_red("ERROR: No camera devices were detected.")
            return False
        print("\nAvailable cameras:\n")
        for number, (listed_device_index, label) in enumerate(devices, 1):
            marker = " (current)" if int(config.get("camera_device_index", 0)) == listed_device_index else ""
            print(f"  {number}) {label} [index={listed_device_index}]{marker}")
        while True:
            value = input("\nChoose camera number (or 'cancel'): ").strip().lower()
            if value == "cancel":
                print("\nCancelled.\n")
                return False
            if not value.isdigit() or not (1 <= int(value) <= len(devices)):
                print("Invalid option.")
                continue
            device_index = devices[int(value) - 1][0]
            break
    config["camera_enabled"] = True
    config["camera_device_index"] = int(device_index)
    camera_runtime["camera_enabled"] = True
    camera_runtime["active_device_index"] = int(device_index)
    save_robot_config(config)
    print(f"\n✅ camera = on | device {device_index}\n")
    return True


def _camera_preview_worker(camera_runtime: dict, _device_index_unused) -> None:
    cv2_mod = camera_runtime.get("cv2")
    stop_event = camera_runtime.get("stop_event")
    if cv2_mod is None or stop_event is None:
        return
    window_name = "Robot Control Panel"
    action_queue = camera_runtime.get("panel_action_queue")
    np_mod = importlib.import_module("numpy")
    cap = None
    opened_device_index = None
    window_open = False

    def mouse_callback(event, x, y, _flags, _param):
        if event != cv2_mod.EVENT_LBUTTONUP:
            return
        rects = camera_runtime.get("panel_button_rects") or {}
        if not isinstance(action_queue, queue.Queue):
            return
        for action, rect in rects.items():
            x1, y1, x2, y2 = rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                with contextlib.suppress(queue.Full):
                    action_queue.put_nowait(str(action))
                break

    try:
        while not stop_event.is_set():
            loop_started_ts = time.monotonic()
            config = camera_runtime.get("voice_config", {})
            panel_backend = str(camera_runtime.get("panel_backend") or config.get("panel_backend", "opencv")).strip().lower()
            qt_proc = camera_runtime.get("qt_panel_process")
            if panel_backend == "qt" and qt_proc is not None and not qt_proc.is_alive():
                stop_qt_panel(camera_runtime)
                panel_backend = ""
            panel_enabled = bool(camera_runtime.get("panel_enabled", False)) and panel_backend == "opencv"
            qt_panel_active = panel_backend == "qt" and qt_proc is not None and qt_proc.is_alive()
            if panel_enabled and not window_open:
                cv2_mod.namedWindow(window_name)
                cv2_mod.setMouseCallback(window_name, mouse_callback)
                window_open = True
            elif (not panel_enabled or panel_backend != "opencv") and window_open:
                with contextlib.suppress(Exception):
                    cv2_mod.destroyWindow(window_name)
                camera_runtime["panel_button_rects"] = {}
                window_open = False
            desired_camera_enabled = bool(config.get("camera_enabled", False))
            desired_device_index = int(config.get("camera_device_index", 0)) if desired_camera_enabled else None
            if desired_camera_enabled:
                if cap is None or opened_device_index != desired_device_index:
                    if cap is not None:
                        with contextlib.suppress(Exception):
                            cap.release()
                    cap = open_camera_capture(cv2_mod, int(desired_device_index))
                    opened_device_index = desired_device_index
                    if cap is None or not cap.isOpened():
                        print_error_red(f"ERROR: Could not open camera device {desired_device_index}.")
                        cap = None
                        config["camera_enabled"] = False
                        camera_runtime["camera_enabled"] = False
                        save_robot_config(config)
                camera_runtime["active_device_index"] = desired_device_index
            else:
                if cap is not None:
                    with contextlib.suppress(Exception):
                        cap.release()
                cap = None
                opened_device_index = None
                camera_runtime["active_device_index"] = None

            frame = None
            if cap is not None:
                ok, frame = cap.read()
                if not ok:
                    print_error_red(f"ERROR: Failed to read from camera device {opened_device_index}.")
                    with contextlib.suppress(Exception):
                        cap.release()
                    cap = None
                    opened_device_index = None
                    frame = None
                    config["camera_enabled"] = False
                    camera_runtime["camera_enabled"] = False
                    save_robot_config(config)

            if frame is not None and bool(camera_runtime.get("vision_enabled", False)):
                try:
                    compiled = camera_runtime.get("vision_compiled_model")
                    input_shape = camera_runtime.get("vision_input_shape")
                    if compiled is not None and input_shape is not None:
                        blob = preprocess_vision_frame(frame, cv2_mod, np_mod, input_shape)
                        result = compiled([blob])
                        raw_output = next(iter(result.values()))
                        threshold = float(camera_runtime.get("vision_threshold", 0.4))
                        labels = list(camera_runtime.get("vision_labels", []))
                        detections = parse_detection_results(raw_output, frame.shape, threshold, labels)
                        now_ts = time.monotonic()
                        if should_process_vision_events(camera_runtime, now_ts=now_ts):
                            event_message = handle_vision_tick(camera_runtime, detections)
                            if event_message:
                                if str(event_message).startswith("__INTERRUPT_AUDIO__:"):
                                    interrupt_audio_and_speak(
                                        camera_runtime,
                                        str(event_message).split(":", 1)[1],
                                    )
                                else:
                                    emit_vision_event_message(camera_runtime, event_message)
                        if should_emit_vision_log(camera_runtime, now_ts=now_ts):
                            debug_text = format_vision_debug_output(raw_output, detections)
                            print(f"\n[vision-log] {debug_text}\n")
                        if panel_enabled or qt_panel_active:
                            frame = annotate_frame_with_detections(frame, detections, cv2_mod)
                except Exception as exc:
                    print_error_red(f"ERROR: Vision inference failed: {exc}")
                    camera_runtime["vision_enabled"] = False

            speech = bool(camera_runtime.get("auto_listen_runtime", {}).get("last_is_speech", False))
            speech_prob = float(camera_runtime.get("auto_listen_runtime", {}).get("last_speech_probability", 0.0))
            speech_prob_threshold = float(camera_runtime.get("auto_listen_runtime", {}).get("last_speech_probability_threshold", 0.5))
            speech_started = bool(camera_runtime.get("auto_listen_runtime", {}).get("last_speech_started", False))
            speech_frames = int(camera_runtime.get("auto_listen_runtime", {}).get("last_speech_frames", 0))
            start_event = float(camera_runtime.get("auto_listen_runtime", {}).get("last_start_event", 0.0))
            end_event = float(camera_runtime.get("auto_listen_runtime", {}).get("last_end_event", 0.0))
            recording = bool(camera_runtime.get("auto_listen_runtime", {}).get("last_recording", False))
            display_segment_frames = int(camera_runtime.get("auto_listen_runtime", {}).get("last_display_segment_frames", 1))
            metrics = [
                {"label": "Speech Prob", "value": min(1.0, max(0.0, speech_prob)), "threshold": min(1.0, max(0.0, speech_prob_threshold)), "active": speech},
                {"label": "Start Event", "value": 1.0 if start_event > 0.0 else 0.0, "threshold": 1.0, "active": start_event > 0.0},
                {"label": "End Event", "value": 1.0 if end_event > 0.0 else 0.0, "threshold": 1.0, "active": end_event > 0.0},
                {"label": "Segment Open", "value": 1.0 if speech_started else 0.0, "threshold": 1.0, "active": speech_started},
                {"label": "Segment Length", "value": min(1.0, speech_frames / max(1, display_segment_frames)), "threshold": 1.0, "active": speech_started},
                {"label": "Recording State", "value": 1.0 if recording else 0.0, "threshold": 1.0, "active": recording},
            ]
            gesture = ""
            gesture_until = float(camera_runtime.get("robot_face_gesture_until", 0.0))
            if time.monotonic() < gesture_until:
                gesture = str(camera_runtime.get("robot_face_gesture", ""))
            face_state = {
                "speaking": is_audio_playback_active(),
                "listening": speech or speech_started,
                "gesture": gesture,
                "visual_effects_enabled": bool(config.get("visual_effects_enabled", True)),
            }
            button_specs = [
                {"label": "Camera", "action": "toggle_camera", "active": bool(config.get("camera_enabled", False))},
                {"label": "Vision", "action": "toggle_vision", "active": bool(config.get("vision_enabled", False))},
                {"label": "Auto Listen", "action": "toggle_auto_listen", "active": bool(config.get("auto_listen_enabled", False))},
                {"label": "Audio", "action": "toggle_audio", "active": bool(config.get("audio_enabled", True))},
                {"label": "Vision Events", "action": "toggle_vision_events", "active": bool(config.get("vision_event_processing_enabled", True))},
                {"label": "Log", "action": "toggle_log", "active": bool(config.get("vision_log_enabled", False))},
                {"label": "Audio Monitor", "action": "toggle_audio_monitor", "active": bool(config.get("audio_monitor_enabled", False))},
                {"label": "Visual Effects", "action": "toggle_visual_effects", "active": bool(config.get("visual_effects_enabled", True))},
                {"label": "Exit", "action": "exit_program", "active": False},
            ]
            if qt_panel_active:
                camera_jpeg = None
                camera_placeholder = "Camera Off" if not bool(config.get("camera_enabled", False)) else "Waiting for camera..."
                if frame is not None:
                    try:
                        preview = cv2_mod.resize(frame, (700, 616))
                        ok, encoded = cv2_mod.imencode(".jpg", preview, [int(cv2_mod.IMWRITE_JPEG_QUALITY), 80])
                        if ok:
                            camera_jpeg = encoded.tobytes()
                    except Exception:
                        camera_jpeg = None
                _put_latest_panel_state(
                    camera_runtime.get("qt_panel_state_queue"),
                    {
                        "camera_jpeg": camera_jpeg,
                        "camera_placeholder": camera_placeholder,
                        "metrics": metrics,
                        "button_specs": button_specs,
                        "show_audio_monitor": bool(config.get("audio_monitor_enabled", False)),
                        "face_state": face_state,
                    },
                )
                qt_action_queue = camera_runtime.get("qt_panel_action_queue")
                if qt_action_queue is not None:
                    while True:
                        try:
                            action = qt_action_queue.get_nowait()
                        except Exception:
                            break
                        handle_panel_action(str(action), camera_runtime)
            if panel_enabled and window_open:
                panel_frame, rects = build_panel_frame(
                    np_mod,
                    cv2_mod,
                    frame,
                    bool(config.get("camera_enabled", False)),
                    opened_device_index,
                    metrics,
                    button_specs,
                    bool(config.get("audio_monitor_enabled", False)),
                    face_state,
                )
                camera_runtime["panel_button_rects"] = rects
                cv2_mod.imshow(window_name, panel_frame)
                while isinstance(action_queue, queue.Queue):
                    try:
                        action = action_queue.get_nowait()
                    except queue.Empty:
                        break
                    handle_panel_action(str(action), camera_runtime)
                key = cv2_mod.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    camera_runtime["panel_enabled"] = False
                    continue
            panel_is_active = any(
                [
                    panel_enabled or qt_panel_active,
                    cap is not None,
                    bool(config.get("audio_monitor_enabled", False)),
                    bool(config.get("vision_enabled", False)),
                    bool(speech),
                    bool(speech_started),
                    bool(start_event > 0.0),
                    bool(end_event > 0.0),
                    bool(recording),
                    bool(face_state.get("speaking", False)),
                    bool(gesture),
                ]
            )
            target_frame_s = 0.05 if panel_is_active else 0.25
            remaining_s = target_frame_s - (time.monotonic() - loop_started_ts)
            if remaining_s > 0:
                time.sleep(remaining_s)
    finally:
        if cap is not None:
            with contextlib.suppress(Exception):
                cap.release()
        with contextlib.suppress(Exception):
            if window_open:
                cv2_mod.destroyWindow(window_name)


def start_camera_preview(camera_runtime: dict, config: dict) -> bool:
    if not ensure_camera_worker(camera_runtime, config):
        return False
    camera_runtime["vision_enabled"] = bool(config.get("vision_enabled", False))
    camera_runtime["vision_threshold"] = float(config.get("vision_threshold", 0.4))
    camera_runtime["vision_log_enabled"] = bool(config.get("vision_log_enabled", False))
    camera_runtime["vision_log_interval_s"] = float(config.get("vision_log_interval_s", 1.0))
    camera_runtime["vision_log_last_ts"] = 0.0
    camera_runtime["vision_event_processing_enabled"] = bool(config.get("vision_event_processing_enabled", True))
    camera_runtime["vision_event_last_ts"] = 0.0
    camera_runtime["vision_last_detection_count"] = 0
    if camera_runtime["vision_enabled"]:
        if not ensure_vision_runtime(camera_runtime, config):
            return False
    ok = set_camera_enabled(camera_runtime, config, True)
    if ok and camera_runtime.get("thread") is not None and camera_runtime["thread"].is_alive():
        print("Press ESC or q in the panel window to close it.\n")
    return ok


def handle_panel_action(action: str, camera_runtime: dict) -> None:
    config = camera_runtime.get("voice_config")
    if not isinstance(config, dict):
        return
    auto_listen_runtime = camera_runtime.get("auto_listen_runtime")
    if action == "toggle_camera":
        set_camera_enabled(camera_runtime, config, not bool(config.get("camera_enabled", False)))
        return
    if action == "toggle_vision":
        enabled = not bool(config.get("vision_enabled", False))
        if enabled and not ensure_vision_runtime(camera_runtime, config):
            return
        config["vision_enabled"] = enabled
        camera_runtime["vision_enabled"] = enabled
        save_robot_config(config)
        print(f"\n✅ vision = {'on' if enabled else 'off'}\n")
        return
    if action == "toggle_auto_listen" and isinstance(auto_listen_runtime, dict):
        if bool(config.get("auto_listen_enabled", False)):
            config["auto_listen_enabled"] = False
            save_robot_config(config)
            refresh_auto_listen_worker(auto_listen_runtime, config)
            print("\n✅ auto_listen = off\n")
        else:
            config["auto_listen_enabled"] = True
            if refresh_auto_listen_worker(auto_listen_runtime, config):
                save_robot_config(config)
                print("\n✅ auto_listen = on\n")
        return
    if action == "toggle_audio":
        config["audio_enabled"] = not bool(config.get("audio_enabled", True))
        save_robot_config(config)
        print(f"\n✅ audio = {'on' if bool(config.get('audio_enabled', True)) else 'off'}\n")
        return
    if action == "toggle_vision_events":
        config["vision_event_processing_enabled"] = not bool(config.get("vision_event_processing_enabled", True))
        camera_runtime["vision_event_processing_enabled"] = bool(config["vision_event_processing_enabled"])
        camera_runtime["vision_last_detection_count"] = 0
        save_robot_config(config)
        print(f"\n✅ vision_events = {'on' if bool(config['vision_event_processing_enabled']) else 'off'}\n")
        return
    if action == "toggle_log":
        config["vision_log_enabled"] = not bool(config.get("vision_log_enabled", False))
        camera_runtime["vision_log_enabled"] = bool(config["vision_log_enabled"])
        camera_runtime["vision_log_last_ts"] = 0.0
        if isinstance(auto_listen_runtime, dict):
            auto_listen_runtime["vad_log_last_ts"] = 0.0
        save_robot_config(config)
        print(f"\n✅ vision log = {'on' if bool(config['vision_log_enabled']) else 'off'}\n")
        return
    if action == "toggle_audio_monitor":
        config["audio_monitor_enabled"] = not bool(config.get("audio_monitor_enabled", False))
        save_robot_config(config)
        print(f"\n✅ audio_monitor = {'on' if bool(config['audio_monitor_enabled']) else 'off'}\n")
        return
    if action == "toggle_visual_effects":
        config["visual_effects_enabled"] = not bool(config.get("visual_effects_enabled", True))
        save_robot_config(config)
        print(f"\n✅ visual_effects = {'on' if bool(config['visual_effects_enabled']) else 'off'}\n")
        return
    if action == "close_panel":
        camera_runtime["panel_enabled"] = False
        if str(camera_runtime.get("panel_backend") or "").lower() == "qt":
            stop_qt_panel(camera_runtime)
        print("\n✅ panel = off\n")
        return
    if action == "exit_program":
        APP_EXIT_REQUESTED.set()
        if isinstance(auto_listen_runtime, dict):
            config["auto_listen_enabled"] = False
            stop_auto_listen(auto_listen_runtime)
        config["camera_enabled"] = False
        camera_runtime["camera_enabled"] = False
        save_robot_config(config)
        stop_event = camera_runtime.get("stop_event")
        if stop_event is not None:
            stop_event.set()
        stop_qt_panel(camera_runtime)
        print("\nExiting from panel...\n")
        with contextlib.suppress(Exception):
            _thread.interrupt_main()
        return


def speak_espeak_ng(text: str, config: dict, allow_interrupt: bool = False) -> tuple[bool, float]:
    exe = find_espeak_executable()
    if not exe:
        print_error_red("ERROR: eSpeak NG executable not found (espeak-ng/espeak).")
        print_espeak_install_suggestion()
        return False, 0.0

    voice = str(config.get("espeak_voice", "es")).strip() or "es"
    rate = int(config.get("espeak_rate", 145))
    pitch = int(config.get("espeak_pitch", 45))
    amplitude = int(config.get("espeak_amplitude", 120))
    cmd = [
        exe,
        "-v",
        voice,
        "-s",
        str(rate),
        "-p",
        str(pitch),
        "-a",
        str(amplitude),
        text,
    ]
    t_start = time.perf_counter()
    proc = subprocess.Popen(cmd)
    calc_latency_s = time.perf_counter() - t_start

    with audio_playback_scope():
        if not allow_interrupt:
            proc.wait()
            return False, calc_latency_s

        with KEYBOARD.capture():
            while proc.poll() is None:
                if consume_esc_pressed() or is_audio_cancel_requested():
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                    return True, calc_latency_s
                time.sleep(0.02)
    return False, calc_latency_s


def list_espeak_voices() -> None:
    exe = find_espeak_executable()
    if not exe:
        print_error_red("ERROR: eSpeak NG executable not found (espeak-ng/espeak).")
        print_espeak_install_suggestion()
        return
    print("\neSpeak voices:\n")
    subprocess.run([exe, "--voices"], check=False)
    print("")


def add_model_interactive(models: list[dict]) -> dict | None:
    print("\nAdd model\n")
    display = input("Display name: ").strip()
    if not display:
        print("\nCancelled: display name is required.\n")
        return None
    params = input("Params (e.g. 7B): ").strip()
    if not params:
        print("\nCancelled: params are required.\n")
        return None
    repo = input("HF repo (owner/name): ").strip()
    if not repo:
        print("\nCancelled: repo is required.\n")
        return None

    if any(m["repo"] == repo for m in models):
        print("\n⚠️ A model with that repo already exists.\n")
        return None

    default_local = slug_from_repo(repo)
    local_dir = input(f"Local folder [{default_local}]: ").strip() or default_local
    new_model = {
        "display": display,
        "params": params,
        "repo": repo,
        "local": CACHE_DIR / local_dir,
    }
    models.append(new_model)
    save_models(models)
    print(f"\n✅ Model added as #{len(models)}: {model_menu_label(new_model)}\n")
    return new_model


def load_device_compat() -> dict:
    try:
        if DEVICE_COMPAT_FILE.exists():
            data = json.loads(DEVICE_COMPAT_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def save_device_compat(compat: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    DEVICE_COMPAT_FILE.write_text(json.dumps(compat, indent=2), encoding="utf-8")


def mark_model_device_compat(compat: dict, model_repo: str, device: str, is_ok: bool) -> None:
    model_entry = compat.setdefault(model_repo, {})
    model_entry[device.upper()] = bool(is_ok)


def normalize_chip_name(device: str) -> str | None:
    text = str(device or "").strip().upper()
    if text in {"CPU", "GPU", "NPU"}:
        return text
    if "NPU" in text:
        return "NPU"
    if "GPU" in text:
        return "GPU"
    if "CPU" in text:
        return "CPU"
    return None


def mark_runtime_chip_compat(compat: dict | None, key: str, device: str, is_ok: bool) -> None:
    if not compat:
        return
    chip = normalize_chip_name(device)
    if chip is None:
        return
    mark_model_device_compat(compat, key, chip, is_ok)
    save_device_compat(compat)


def model_device_badges(compat: dict, model_repo: str) -> str:
    model_entry = compat.get(model_repo, {})
    badges = []
    for device in BENCHMARK_DEVICES:
        value = model_entry.get(device)
        mark = "✅" if value is True else ("❌" if value is False else "❔")
        badges.append(f"{device}:{mark}")
    return " ".join(badges)


def chip_marks_for_key(compat: dict, key: str) -> tuple[str, str, str]:
    entry = compat.get(key, {})
    def mark(dev: str) -> str:
        value = entry.get(dev)
        return "✅" if value is True else ("❌" if value is False else "❔")
    return mark("CPU"), mark("GPU"), mark("NPU")


def download_model(repo_id: str, local_dir: Path) -> None:
    load_hf_token()
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n📥 Downloading model: {repo_id}")
    print(f"   -> destination: {local_dir}\n")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        token=os.environ.get("HF_TOKEN") or None,
    )
    print("\n✅ Download complete.\n")


def choose_model_interactive(allow_download: bool, title: str, compat: dict | None = None) -> dict | None:
    print(f"\n{title}\n")
    compat = compat or {}
    for i, m in enumerate(MODELS, 1):
        status = "✅" if is_downloaded(m["local"]) else "⬇️"
        badges = model_device_badges(compat, m["repo"])
        print(f"  {i}) {status} {model_menu_label(m)} | {badges}")
    print("  0) Cancel\n")

    while True:
        choice = input("Option: ").strip()
        if choice == "0":
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(MODELS):
            m = MODELS[int(choice) - 1]
            if allow_download and not is_downloaded(m["local"]):
                download_model(m["repo"], m["local"])
            return m
        print("Invalid option.")


def delete_model_files(m: dict) -> bool:
    path = m["local"]
    if not path.exists():
        print("\nℹ️ That model is not on disk.\n")
        return False

    try:
        resolved = path.resolve()
        cache = CACHE_DIR.resolve()
        if cache not in resolved.parents and resolved != cache:
            print("\n❌ Security check failed: path is outside ov_models, skipping delete.\n")
            return False
    except Exception:
        print("\nâŒ Could not resolve paths for safe deletion.\n")
        return False

    size_before = human_bytes(dir_size_bytes(path))
    print(f"\n🗑️ Deleting: {model_menu_label(m)}")
    print(f"   Path: {path}")
    print(f"   Size: {size_before}\n")

    try:
        shutil.rmtree(path)
        print("✅ Delete complete.\n")
        return True
    except Exception as e:
        print(f"\nâŒ Delete error: {e}\n")
        return False


# =========================
# Stats persistence
# =========================
def load_stats() -> dict:
    try:
        if STATS_FILE.exists():
            raw = json.loads(STATS_FILE.read_text(encoding="utf-8"))
            return normalize_stats_schema(raw)
    except Exception:
        pass
    return normalize_stats_schema({"models": {}})


def save_stats(stats: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    STATS_FILE.write_text(json.dumps(stats, indent=2), encoding="utf-8")


def normalize_stats_schema(stats: dict) -> dict:
    models = stats.setdefault("models", {})
    for entry in models.values():
        if not isinstance(entry, dict):
            continue

        modes = entry.get("modes")
        if isinstance(modes, dict):
            for mode_name in (STATS_MODE_NORMAL, STATS_MODE_BENCHMARK):
                mode_entry = modes.setdefault(mode_name, {})
                mode_entry.setdefault("devices", {})
            continue

        normal_devices = {}
        legacy_devices = entry.get("devices", {})
        if isinstance(legacy_devices, dict):
            normal_devices = legacy_devices
        else:
            runs = int(entry.get("runs", 0) or 0)
            ttft_s = list(entry.get("ttft_s", []))
            tps = list(entry.get("tps", []))
            if runs or ttft_s or tps:
                normal_devices["UNKNOWN"] = {"runs": runs, "ttft_s": ttft_s, "tps": tps}

        entry["modes"] = {
            STATS_MODE_NORMAL: {"devices": normal_devices},
            STATS_MODE_BENCHMARK: {"devices": {}},
        }
    return stats


def get_mode_devices(entry: dict, mode: str, create: bool = False) -> dict:
    modes = entry.get("modes")
    if not isinstance(modes, dict):
        if not create:
            return {}
        entry["modes"] = {
            STATS_MODE_NORMAL: {"devices": {}},
            STATS_MODE_BENCHMARK: {"devices": {}},
        }
        modes = entry["modes"]

    mode_entry = modes.get(mode)
    if not isinstance(mode_entry, dict):
        if not create:
            return {}
        mode_entry = {"devices": {}}
        modes[mode] = mode_entry

    devices = mode_entry.get("devices")
    if not isinstance(devices, dict):
        if not create:
            return {}
        mode_entry["devices"] = {}
        devices = mode_entry["devices"]
    return devices


def record_stats(
    stats: dict,
    model_key: str,
    model_name: str,
    device: str,
    ttft_s: float,
    tps: float,
    mode: str = STATS_MODE_NORMAL,
) -> None:
    models = stats.setdefault("models", {})
    entry = models.setdefault(
        model_key,
        {
            "name": model_name,
            "modes": {
                STATS_MODE_NORMAL: {"devices": {}},
                STATS_MODE_BENCHMARK: {"devices": {}},
            },
        },
    )
    entry["name"] = model_name
    devices = get_mode_devices(entry, mode, create=True)
    device_entry = devices.setdefault(device, {"runs": 0, "ttft_s": [], "tps": []})
    device_entry["runs"] += 1
    device_entry["ttft_s"].append(ttft_s)
    device_entry["tps"].append(tps)


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def build_stats_rows(stats: dict, mode: str) -> list[dict]:
    models = stats.get("models", {})
    model_index_by_repo = {m["repo"]: i for i, m in enumerate(MODELS, 1)}
    rows = []
    for k, v in models.items():
        model_name = v.get("name", k)
        model_index = model_index_by_repo.get(k)
        model_label = f"{model_index}) {model_name}" if model_index is not None else f"?) {model_name}"
        devices = get_mode_devices(v, mode, create=False)
        for device, d in devices.items():
            ttfts = d.get("ttft_s", [])
            tpss = d.get("tps", [])
            rows.append(
                {
                    "model": model_label,
                    "device": device,
                    "runs": d.get("runs", 0),
                    "ttft_avg": mean(ttfts),
                    "tps_avg": mean(tpss),
                    "ttft_last": ttfts[-1] if ttfts else 0.0,
                    "tps_last": tpss[-1] if tpss else 0.0,
                }
            )
    rows.sort(key=lambda r: (r["model"].lower(), -r["tps_avg"], r["device"].lower()))
    return rows


def print_stats_mode_table(rows: list[dict], title: str) -> None:
    print(f"\n{title}")
    if not rows:
        print("(no stats)")
        return

    def f(x):
        return f"{x:0.3f}"

    headers = ["Model", "Device", "n", "TTFT avg(s)", "TPS avg", "TTFT last(s)", "TPS last"]
    colw = [56, 18, 4, 12, 10, 13, 9]

    def cut(s, w):
        s = str(s)
        return s if len(s) <= w else s[: w - 1] + "…"

    line = (
        f"{cut(headers[0], colw[0]):<{colw[0]}} "
        f"{cut(headers[1], colw[1]):<{colw[1]}} "
        f"{headers[2]:>{colw[2]}} "
        f"{headers[3]:>{colw[3]}} "
        f"{headers[4]:>{colw[4]}} "
        f"{headers[5]:>{colw[5]}} "
        f"{headers[6]:>{colw[6]}}"
    )
    sep = "-" * len(line)

    print("\n" + line)
    print(sep)
    last_model = None
    for r in rows:
        model_cell = r["model"] if r["model"] != last_model else ""
        last_model = r["model"]
        print(
            f"{cut(model_cell, colw[0]):<{colw[0]}} "
            f"{cut(r['device'], colw[1]):<{colw[1]}} "
            f"{r['runs']:>{colw[2]}} "
            f"{f(r['ttft_avg']):>{colw[3]}} "
            f"{f(r['tps_avg']):>{colw[4]}} "
            f"{f(r['ttft_last']):>{colw[5]}} "
            f"{f(r['tps_last']):>{colw[6]}}"
        )
    print("")


def print_stats_table(stats: dict) -> None:
    normal_rows = build_stats_rows(stats, STATS_MODE_NORMAL)
    benchmark_rows = build_stats_rows(stats, STATS_MODE_BENCHMARK)
    if not normal_rows and not benchmark_rows:
        print("\n(no stats yet)\n")
        return
    print_stats_mode_table(normal_rows, "Normal stats")
    print_stats_mode_table(benchmark_rows, "Benchmark stats")


def clear_stats(stats: dict, model_number: int | None = None, device: str | None = None) -> None:
    models = stats.setdefault("models", {})

    if model_number is None:
        if not models:
            print("\n(no stats to clear)\n")
            return
        stats["models"] = {}
        save_stats(stats)
        print("\n✅ All stats were cleared.\n")
        return

    if model_number < 1 or model_number > len(MODELS):
        print(f"\n⚠️ Invalid model number. Choose 1..{len(MODELS)}.\n")
        return

    model = MODELS[model_number - 1]
    model_key = model["repo"]
    model_label = model_menu_label(model)
    entry = models.get(model_key)
    if entry is None:
        print(f"\n(no stats found for model {model_number}: {model_label})\n")
        return

    if device is None:
        del models[model_key]
        save_stats(stats)
        print(f"\n✅ Cleared all stats for model {model_number}: {model_label}\n")
        return

    normalized_device = device.strip().upper()
    if not normalized_device:
        print("\n⚠️ Device cannot be empty.\n")
        return

    deleted_any = False
    available_devices = set()
    for mode in (STATS_MODE_NORMAL, STATS_MODE_BENCHMARK):
        mode_devices = get_mode_devices(entry, mode, create=False)
        for existing_device in list(mode_devices.keys()):
            available_devices.add(str(existing_device))
            if str(existing_device).upper() == normalized_device:
                del mode_devices[existing_device]
                deleted_any = True

    if not deleted_any:
        available = ", ".join(sorted(available_devices)) or "(none)"
        print(f"\n⚠️ Device not found for that model. Available: {available}\n")
        return

    normal_left = get_mode_devices(entry, STATS_MODE_NORMAL, create=False)
    benchmark_left = get_mode_devices(entry, STATS_MODE_BENCHMARK, create=False)
    if not normal_left and not benchmark_left:
        del models[model_key]
    save_stats(stats)
    print(f"\n✅ Cleared stats for model {model_number} on device {normalized_device}.\n")


# =========================
# Pipeline
# =========================
def should_probe_llm_device(device: str) -> bool:
    dev = str(device or "").upper()
    return IS_LINUX and "GPU" in dev


def probe_llm_pipeline_load(model_path: Path, device: str, performance_hint: str, timeout_s: int = 45) -> tuple[bool, str]:
    cmd = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "import openvino_genai as ov; "
            "ov.LLMPipeline(sys.argv[1], sys.argv[2], PERFORMANCE_HINT=sys.argv[3]); "
            "print('OK', flush=True)"
        ),
        str(model_path),
        str(device),
        str(performance_hint),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"probe timed out after {timeout_s}s"

    if result.returncode == 0:
        return True, ""

    detail = (result.stderr or result.stdout or "").strip()
    if not detail:
        detail = f"probe exited with code {result.returncode}"
    return False, detail


def resolve_llm_device_for_load(model_path: Path, requested_device: str, allow_linux_fallback: bool = True) -> tuple[str, str | None]:
    device = str(requested_device or "").upper()
    if not should_probe_llm_device(device):
        return device, None

    ok, detail = probe_llm_pipeline_load(model_path, device, ACTIVE_PERFORMANCE_HINT)
    if ok:
        return device, None
    if not allow_linux_fallback:
        raise RuntimeError(f"LLM probe failed on {device}: {detail}")

    print_error_red(f"ERROR: LLM probe failed on {device}: {detail}")
    print("Falling back to CPU for this session.\n")

    cpu_ok, cpu_detail = probe_llm_pipeline_load(model_path, "CPU", ACTIVE_PERFORMANCE_HINT)
    if not cpu_ok:
        raise RuntimeError(f"LLM probe also failed on CPU: {cpu_detail}")
    return "CPU", detail


def load_pipeline(selected_model: dict, allow_linux_fallback: bool = True) -> ov_genai.LLMPipeline:
    global ACTIVE_DEVICE
    model_path = selected_model["local"]
    requested_device = ACTIVE_DEVICE
    actual_device, fallback_reason = resolve_llm_device_for_load(
        model_path,
        requested_device,
        allow_linux_fallback=allow_linux_fallback,
    )
    ACTIVE_DEVICE = actual_device
    if fallback_reason and actual_device != requested_device:
        print_chip_fallback_warning(
            "LLM",
            requested_chip=requested_device,
            actual_chip=actual_device,
            reason=fallback_reason,
        )
    print(
        f"\n✅ Model loaded on {ACTIVE_DEVICE}: {model_menu_label(selected_model)} "
        f"(PERFORMANCE_HINT={ACTIVE_PERFORMANCE_HINT})"
    )
    return ov_genai.LLMPipeline(
        str(model_path),
        ACTIVE_DEVICE,
        PERFORMANCE_HINT=ACTIVE_PERFORMANCE_HINT,
    )


def normalize_openai_base_url(base_url: str) -> str:
    base = str(base_url or "").strip().rstrip("/")
    if not base:
        return "http://localhost:1234/v1"
    if base.endswith("/v1"):
        return base
    if base.endswith("/v1/chat/completions"):
        return base[: -len("/chat/completions")]
    return base + "/v1"


class OpenAICompatPipeline:
    def __init__(self, base_url: str, model: str, api_key: str = ""):
        self.base_url = normalize_openai_base_url(base_url)
        self.model = str(model or "").strip()
        self.api_key = str(api_key or "").strip()
        if not self.model:
            raise RuntimeError("external_llm_model is empty.")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 300,
        temperature: float = 0.7,
        top_p: float = 0.9,
        streamer=None,
    ):
        url = self.base_url + "/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": str(prompt)}],
            "max_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "stream": True,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                while True:
                    raw_line = resp.readline()
                    if not raw_line:
                        break
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = obj.get("choices", [])
                    if not choices:
                        continue
                    choice = choices[0] or {}
                    delta = choice.get("delta") or {}
                    text = delta.get("content")
                    if text is None:
                        message = choice.get("message") or {}
                        text = message.get("content")
                    if text and streamer is not None:
                        streamer(str(text))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"HTTP {exc.code}: {body or exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Connection failed: {exc.reason}") from exc


def build_external_model_descriptor(config: dict) -> dict:
    base_url = normalize_openai_base_url(str(config.get("external_llm_base_url", "http://localhost:1234")).strip())
    model_name = str(config.get("external_llm_model", "")).strip() or "external-model"
    return {
        "kind": "external",
        "repo": model_name,
        "display": model_name,
        "params": "API",
        "base_url": base_url,
    }


def load_external_pipeline(config: dict) -> OpenAICompatPipeline:
    base_url = str(config.get("external_llm_base_url", "http://localhost:1234")).strip()
    model_name = str(config.get("external_llm_model", "")).strip()
    api_key = str(config.get("external_llm_api_key", "")).strip()
    pipeline = OpenAICompatPipeline(base_url=base_url, model=model_name, api_key=api_key)
    print(f"\n✅ External OpenAI-compatible model active: {model_name} @ {pipeline.base_url}")
    return pipeline


def activate_llm_from_config(
    config: dict,
    compat: dict | None = None,
) -> tuple[object | None, dict | None, list[str]]:
    llm_backend = str(config.get("llm_backend", "local")).strip().lower()
    if llm_backend == "external":
        current = build_external_model_descriptor(config)
        pipe = load_external_pipeline(config)
        return pipe, current, []

    saved_model_repo = str(config.get("current_model_repo", "")).strip()
    if not saved_model_repo:
        return None, None, []
    model_from_cfg = next((m for m in MODELS if m["repo"] == saved_model_repo), None)
    if model_from_cfg is None:
        print(f"⚠️ Saved model repo not found in model list: {saved_model_repo}")
        return None, None, []
    if not is_downloaded(model_from_cfg["local"]):
        print(f"⚠️ Saved model is not downloaded yet: {model_menu_label(model_from_cfg)}")
        return None, None, []
    current = model_from_cfg
    pipe = load_pipeline(current)
    config["llm_device"] = ACTIVE_DEVICE
    save_robot_config(config)
    if compat is not None:
        mark_model_device_compat(compat, current["repo"], ACTIVE_DEVICE, True)
        save_device_compat(compat)
    return pipe, current, []


def release_llm_pipe(pipe) -> None:
    if pipe is None:
        return
    try:
        if hasattr(pipe, "close") and callable(pipe.close):
            pipe.close()
    except Exception:
        pass
    try:
        del pipe
    except Exception:
        pass
    gc.collect()


def choose_from_options(title: str, options: list[str], current_value: str) -> str:
    print(f"\n{title}")
    for i, option in enumerate(options, 1):
        marker = "(current)" if option == current_value else ""
        print(f"  {i}) {option} {marker}".rstrip())

    while True:
        choice = input("Option: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print("Invalid option.")


def configure_runtime() -> None:
    global ACTIVE_DEVICE, ACTIVE_PERFORMANCE_HINT

    print("\n⚙️ Runtime configuration")
    ACTIVE_DEVICE = choose_from_options("Select DEVICE:", DEVICE_OPTIONS, ACTIVE_DEVICE)
    ACTIVE_PERFORMANCE_HINT = choose_from_options(
        "Select PERFORMANCE_HINT:",
        PERFORMANCE_HINT_OPTIONS,
        ACTIVE_PERFORMANCE_HINT,
    )

    print(
        f"\n✅ Runtime updated: DEVICE={ACTIVE_DEVICE}, "
        f"PERFORMANCE_HINT={ACTIVE_PERFORMANCE_HINT}\n"
    )


# =========================
# UI / Commands
# =========================
HELP_TEXT = """\
Commands:
  /help                     Show this help
  /context                  Show the current LLM conversation memory/context
  /voices                   List native voices and select one (Windows only)
  /config                   Configure voice + Whisper STT settings
  /llm_backend <name>       Set LLM backend: local | external
  /max_tokens <n>           Set max new tokens for chat responses
  /audio <on|off>           Enable or disable speaker output
  /audio_inputs             List available microphone/input devices
  /audio_input_select       Select the microphone/input device for STT and auto-listen
  /audio_monitor <on|off>   Show or hide a live audio monitor window for auto-listen input
  /panel [opencv|qt]        Open the control panel window with the selected backend
  /camera <on|off>          Enable or disable camera preview
  /vision <on|off>          Enable or disable OpenVINO object detection on camera frames
  /log <on|off|seconds>     Print throttled raw vision output to the console for debugging
  /vision_events <on|off>   Enable or disable throttled vision event processing and TTS reactions
  /auto_listen <on|off>     Enable or disable continuous microphone VAD + automatic STT
  /wake_word_enabled <t/f>  Enable or disable wake word mode
  /wake_word_phrase <text>  Set the phrase that activates auto-listen
  /wake_word_stop_phrase    Set the phrase that deactivates auto-listen
  /wake_word_on_response    Set the TTS phrase spoken when wake mode activates
  /wake_word_off_response   Set the TTS phrase spoken when wake mode deactivates
  /vad_preroll <ms>         Set pre-roll audio padding before detected speech
  /vad_silence <ms>         Set silence time before auto-listen closes a phrase
  /vad_max_segment <sec>    Set the maximum duration of a captured auto-listen phrase
  /vision_models            List configured vision models
  /vision_select            Select and download a vision model
  /vision_model             Set the OpenVINO detection model XML path
  /vision_labels            Set the optional labels file path
  /vision_device <name>     Set vision device: CPU | GPU | NPU | AUTO
  /repeat <true|false>      If true, repeats input directly with TTS (no LLM)
  /listen                   Continuous listen mode (SPACE start/stop each turn)
  /tts_backend <name>       Set TTS backend: windows | openvino | kokoro | babelvox | espeakng | tada
  /openvino_tts_models      List verified OpenVINO TTS models
  /openvino_tts_add         Add an OpenVINO TTS model entry
  /openvino_tts_select      Select OpenVINO TTS model id to use
  /kokoro_models            List Kokoro model catalog
  /kokoro_select            Select Kokoro model id
  /babelvox_models          List BabelVox model catalog
  /babelvox_select          Select BabelVox model id
  /tada_reference_audio     Set Hume TADA reference audio path
  /tada_reference_text      Set Hume TADA reference transcript
  /tada_reference_record    Record reference audio, transcribe it, and save both for TADA
  /tada_model               Set Hume TADA HF model id
  /tada_codec               Set Hume TADA codec/encoder HF id
  /tada_device              Set Hume TADA device string (cpu, cuda, ...)
  /tada_language            Set Hume TADA encoder language code
  /espeak_voices            List eSpeak NG installed voices
  /whisper_models           List Whisper OpenVINO catalog models and local status
  /whisper_add              Add a Whisper OpenVINO model entry to ov_models/whisper_models.json
  /whisper_select           Select Whisper OpenVINO model id to use
  /models                   Select and load a model (download if missing)
  /add_model                Add a new model to ov_models/models.json
  /delete                   Delete model files from disk (keeps it in the list)
  /stats                    Show separate tables: normal stats and benchmark stats
  /all_models               Show all downloaded models with CPU/GPU/NPU compatibility marks
  /clear_stats              Clear all stats
  /clear_stats <n>          Clear all stats for model number <n>
  /clear_stats <n> <device> Clear stats only for model <n> and device <device>
  /current_model            Show the currently loaded model and runtime config
  /benchmark                Ask: all models or only missing models (always CPU/GPU/NPU)
  /benchmark <number>       Run benchmark on CPU/GPU/NPU only for model number <number>
  /start_server             Start OpenAI-compatible chat server on port 1311
  /exit                     Exit

Usage:
  - First run '/models' to load one.
  - Then type any regular prompt to chat.
"""


def is_command(s: str) -> bool:
    s = s.strip()
    return s.startswith("/")


def normalize_command(s: str) -> str:
    s = s.strip()
    if s.startswith("/"):
        s = s[1:]
    return s



def load_saved_benchmark_prompts() -> list[str]:
    try:
        if BENCHMARK_PROMPTS_FILE.exists():
            data = json.loads(BENCHMARK_PROMPTS_FILE.read_text(encoding="utf-8"))
            prompts = data.get("prompts", []) if isinstance(data, dict) else []
            return [str(p) for p in prompts if str(p).strip()]
    except Exception:
        pass
    return []


def save_benchmark_prompts(prompts: list[str]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"prompts": prompts}
    BENCHMARK_PROMPTS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def prompt_yes_no(message: str, default_yes: bool = True) -> bool:
    suffix = "[Y/n]" if default_yes else "[y/N]"
    while True:
        answer = input(f"{message} {suffix}: ").strip().lower()
        if not answer:
            return default_yes
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please answer with 'y' or 'n'.")


def print_error_red(message: str) -> None:
    print(f"\x1b[31m{message}\x1b[0m")


def print_chip_fallback_warning(context: str, requested_chip: str, actual_chip: str, reason: str = "") -> None:
    details = f" | reason: {reason}" if reason else ""
    print_error_red(
        f"WARNING: {context} requested chip '{requested_chip}' but is using '{actual_chip}' (fallback){details}"
    )


def _openvino_tts_worker(model_dir: str, device: str, text: str, result_queue) -> None:
    try:
        import openvino_genai as _ov_genai

        pipeline = _ov_genai.Text2SpeechPipeline(model_dir, device)
        result = pipeline.generate(text)
        speeches = getattr(result, "speeches", None)
        if not speeches:
            result_queue.put({"ok": False, "error": "openvino TTS returned no audio."})
            return
        audio_obj = speeches[0]
        if hasattr(audio_obj, "data"):
            audio_list = list(audio_obj.data)
        else:
            audio_list = list(audio_obj)
        result_queue.put({"ok": True, "audio": audio_list})
    except Exception as exc:
        result_queue.put({"ok": False, "error": str(exc)})


def ensure_kokoro_model_files(model_dir: Path) -> tuple[Path, Path]:
    model_dir.mkdir(parents=True, exist_ok=True)
    onnx_files = sorted(model_dir.rglob("kokoro-v1.0*.onnx"))
    voices_files = sorted(model_dir.rglob("voices-v1.0*.bin"))
    if onnx_files and voices_files:
        return onnx_files[0], voices_files[0]

    onnx_path = model_dir / "kokoro-v1.0.onnx"
    voices_path = model_dir / "voices-v1.0.bin"
    if not onnx_path.exists():
        print("Downloading Kokoro ONNX model file...")
        urllib.request.urlretrieve(
            "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
            str(onnx_path),
        )
    if not voices_path.exists():
        print("Downloading Kokoro voices file...")
        urllib.request.urlretrieve(
            "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
            str(voices_path),
        )
    return onnx_path, voices_path


def extract_tada_audio_output(output, torch_mod, np_mod):
    sample_rate = None
    audio_obj = output
    if isinstance(output, dict):
        for key in ("audio", "waveform", "wav", "samples"):
            if key in output and output[key] is not None:
                audio_obj = output[key]
                break
        if output.get("sample_rate") is not None:
            try:
                sample_rate = int(output["sample_rate"])
            except Exception:
                sample_rate = None
    else:
        for attr_name in ("audio", "waveform", "wav", "samples"):
            if hasattr(output, attr_name):
                try:
                    candidate = getattr(output, attr_name)
                    if candidate is not None:
                        audio_obj = candidate
                        break
                except Exception:
                    pass
        if hasattr(output, "sample_rate"):
            try:
                sample_rate = int(getattr(output, "sample_rate"))
            except Exception:
                sample_rate = None
    if torch_mod is not None and hasattr(torch_mod, "Tensor") and isinstance(audio_obj, torch_mod.Tensor):
        audio = audio_obj.detach().cpu().float().numpy()
    else:
        audio = np_mod.array(audio_obj, dtype=np_mod.float32)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.reshape(-1)
    return np_mod.array(audio, dtype=np_mod.float32).reshape(-1), sample_rate


def ensure_tada_huggingface_compat() -> None:
    try:
        import inspect
        from huggingface_hub import utils as hf_utils
    except Exception:
        return

    def _wrap_if_needed(attr_name: str) -> None:
        fn = getattr(hf_utils, attr_name, None)
        if fn is None:
            return
        try:
            params = inspect.signature(fn).parameters
        except Exception:
            return
        if "reason" in params:
            return

        def compat_wrapper(*args, **kwargs):
            kwargs.pop("reason", None)
            return fn(*args, **kwargs)

        setattr(hf_utils, attr_name, compat_wrapper)

    _wrap_if_needed("disable_progress_bars")
    _wrap_if_needed("enable_progress_bars")


def collect_benchmark_prompts(count: int = 5) -> list[str]:
    saved_prompts = load_saved_benchmark_prompts()
    if len(saved_prompts) >= count:
        print("\nSaved benchmark prompts were found:\n")
        for i, prompt in enumerate(saved_prompts[:count], 1):
            print(f"  {i}) {prompt}")
        if prompt_yes_no("Do you want to reuse the saved prompts?", default_yes=True):
            print("")
            return saved_prompts[:count]

    print(f"\nEnter {count} prompts for the benchmark:\n")
    prompts: list[str] = []
    for i in range(1, count + 1):
        prompt = input(f"Prompt {i}: ").strip()
        prompts.append(prompt)

    save_benchmark_prompts(prompts)
    print(f"\n✅ Saved {count} benchmark prompts to {BENCHMARK_PROMPTS_FILE}.\n")
    return prompts

def benchmark_models(
    stats: dict,
    prompts: list[str],
    model_number: int | None = None,
    only_missing_models: bool = False,
    compat: dict | None = None,
) -> None:
    global ACTIVE_DEVICE

    if not prompts:
        print("\nNo prompts were provided.\n")
        return

    model_candidates = list(enumerate(MODELS, 1))
    if model_number is not None:
        model_candidates = [x for x in model_candidates if x[0] == model_number]
        if not model_candidates:
            print("\nInvalid model number for benchmark.\n")
            return

    downloaded = [(idx, m) for idx, m in model_candidates if is_downloaded(m["local"])]
    if model_number is None and only_missing_models:
        filtered = []
        for idx, model in downloaded:
            entry = stats.get("models", {}).get(model["repo"], {})
            benchmark_devices = get_mode_devices(entry, STATS_MODE_BENCHMARK, create=False)
            if not benchmark_devices:
                filtered.append((idx, model))
        downloaded = filtered

    if not downloaded:
        print("\nNo downloaded models match this benchmark selection.\n")
        return

    mode_text = "missing models only" if only_missing_models else "all models"
    print(
        f"\nStarting benchmark on {len(downloaded)} model(s) "
        f"across devices ({mode_text}): {', '.join(BENCHMARK_DEVICES)}.\n"
    )

    previous_device = ACTIVE_DEVICE
    try:
        for idx, model in downloaded:
            for device in BENCHMARK_DEVICES:
                print(f"\n===== Benchmark model {idx} on {device}: {model_menu_label(model)} =====")
                ACTIVE_DEVICE = device
                try:
                    pipe = load_pipeline(model, allow_linux_fallback=False)
                    if compat is not None:
                        mark_model_device_compat(compat, model["repo"], device, True)
                        save_device_compat(compat)
                except Exception as exc:
                    if compat is not None:
                        mark_model_device_compat(compat, model["repo"], device, False)
                        save_device_compat(compat)
                    print(f"Failed to load model on {device}: {exc}")
                    continue

                for prompt_idx, prompt in enumerate(prompts, 1):
                    print(f"\n[{idx}.{device}.{prompt_idx}] Prompt: {prompt}")
                    t_start = time.perf_counter()
                    first_token_time = None
                    token_events = 0

                    def streamer(chunk: str):
                        nonlocal first_token_time, token_events
                        now = time.perf_counter()
                        if first_token_time is None:
                            first_token_time = now
                        token_events += 1
                        print(chunk, end="", flush=True)

                    print("🤖 > ", end="", flush=True)
                    pipe.generate(
                        prompt,
                        max_new_tokens=300,
                        temperature=0.7,
                        top_p=0.9,
                        streamer=streamer,
                    )
                    t_end = time.perf_counter()
                    print("\n")

                    if first_token_time is None:
                        ttft = t_end - t_start
                        tps = 0.0
                    else:
                        ttft = first_token_time - t_start
                        decode_time = max(1e-9, t_end - first_token_time)
                        tps = token_events / decode_time

                    stats_name = model_menu_label(model)
                    record_stats(
                        stats,
                        model["repo"],
                        stats_name,
                        device,
                        ttft,
                        tps,
                        mode=STATS_MODE_BENCHMARK,
                    )
                    save_stats(stats)
                    print(f"TTFT: {ttft:0.3f}s | TPS~ {tps:0.2f} | events: {token_events}")

                del pipe
                gc.collect()
    finally:
        ACTIVE_DEVICE = previous_device


def build_chat_prompt(messages: list[dict]) -> str:
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role", "user")).strip().lower()
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        if role == "assistant":
            lines.append(f"Assistant: {content}")
        elif role == "system":
            lines.append(f"System: {content}")
        else:
            lines.append(f"User: {content}")
    return "\n".join(lines) + "\nAssistant:"


def create_openai_chat_response(model_name: str, content: str) -> dict:
    now = int(time.time())
    completion_tokens = max(1, len(content.split())) if content else 0
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": now,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": completion_tokens,
            "total_tokens": completion_tokens,
        },
    }


def start_openai_compatible_server(state: dict) -> ThreadingHTTPServer:
    class OpenAICompatHandler(BaseHTTPRequestHandler):
        def _send_json(self, status_code: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self._send_json(404, {"error": {"message": "Not found"}})
                return

            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            try:
                data = json.loads(raw_body.decode("utf-8") or "{}")
            except Exception:
                self._send_json(400, {"error": {"message": "Invalid JSON body"}})
                return

            pipe = state.get("pipe")
            current = state.get("current")
            if pipe is None or current is None:
                self._send_json(400, {"error": {"message": "No model loaded. Use '/models' first."}})
                return

            messages = data.get("messages")
            if not isinstance(messages, list) or not messages:
                self._send_json(400, {"error": {"message": "'messages' must be a non-empty list"}})
                return

            prompt = build_chat_prompt(messages)
            chunks: list[str] = []

            def streamer(chunk: str):
                chunks.append(chunk)

            try:
                cfg = state.get("config") or {}
                default_max_tokens = int(cfg.get("max_new_tokens", 300))
                pipe.generate(
                    prompt,
                    max_new_tokens=int(data.get("max_tokens", default_max_tokens)),
                    temperature=float(data.get("temperature", 0.7)),
                    top_p=float(data.get("top_p", 0.9)),
                    streamer=streamer,
                )
            except Exception as exc:
                self._send_json(500, {"error": {"message": f"Generation failed: {exc}"}})
                return

            text = "".join(chunks)
            response = create_openai_chat_response(current["repo"], text)
            self._send_json(200, response)

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("0.0.0.0", 1311), OpenAICompatHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


# =========================
# Voice + STT
# =========================
def default_robot_config() -> dict:
    return {
        "voice_index": 0,
        "rate": -2,
        "volume": 100,
        "silence": 600,
        "whisper_model": DEFAULT_WHISPER_MODEL,
        "whisper_language": "es",
        "repeat": False,
        "current_model_repo": "",
        "llm_backend": "local",
        "llm_device": DEFAULT_DEVICE,
        "llm_performance_hint": DEFAULT_PERFORMANCE_HINT,
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
        "tts_backend": DEFAULT_TTS_BACKEND,
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


def load_robot_config() -> dict:
    cfg = default_robot_config()
    if not ROBOT_CONFIG_FILE.exists():
        return cfg
    try:
        data = json.loads(ROBOT_CONFIG_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            if "repeat" not in data and "repetir" in data:
                data["repeat"] = data.get("repetir", False)
            data.pop("repetir", None)
            cfg.update(data)
    except Exception as exc:
        print(f"\n⚠️ Could not read {ROBOT_CONFIG_FILE}: {exc}\n")
    cfg["whisper_openvino"] = bool(cfg.get("whisper_openvino", False))
    llm_backend = str(cfg.get("llm_backend", "local")).strip().lower()
    if llm_backend not in {"local", "external"}:
        llm_backend = "local"
    cfg["llm_backend"] = llm_backend
    cfg["external_llm_base_url"] = str(cfg.get("external_llm_base_url", "http://localhost:1234")).strip() or "http://localhost:1234"
    cfg["external_llm_model"] = str(cfg.get("external_llm_model", "")).strip()
    cfg["external_llm_api_key"] = str(cfg.get("external_llm_api_key", "")).strip()
    cfg["audio_input_device"] = str(cfg.get("audio_input_device", "")).strip()
    cfg["audio_monitor_enabled"] = bool(cfg.get("audio_monitor_enabled", False))
    cfg["visual_effects_enabled"] = bool(cfg.get("visual_effects_enabled", True))
    panel_backend = str(cfg.get("panel_backend", "opencv")).strip().lower()
    if panel_backend not in {"opencv", "qt"}:
        panel_backend = "opencv"
    cfg["panel_backend"] = panel_backend
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
    if vision_device not in {"CPU", "GPU", "NPU", "AUTO"}:
        vision_device = "AUTO"
    cfg["vision_device"] = vision_device
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
    try:
        cfg["auto_listen_preroll_ms"] = max(0, int(cfg.get("auto_listen_preroll_ms", 300)))
    except Exception:
        cfg["auto_listen_preroll_ms"] = 300
    try:
        cfg["auto_listen_min_speech_ms"] = max(30, int(cfg.get("auto_listen_min_speech_ms", 1400)))
    except Exception:
        cfg["auto_listen_min_speech_ms"] = 1400
    try:
        cfg["auto_listen_silence_ms"] = max(100, int(cfg.get("auto_listen_silence_ms", 900)))
    except Exception:
        cfg["auto_listen_silence_ms"] = 900
    try:
        cfg["auto_listen_max_segment_s"] = max(1.0, float(cfg.get("auto_listen_max_segment_s", 15.0)))
    except Exception:
        cfg["auto_listen_max_segment_s"] = 15.0
    try:
        cfg["auto_listen_resume_delay_ms"] = max(0, int(cfg.get("auto_listen_resume_delay_ms", 1500)))
    except Exception:
        cfg["auto_listen_resume_delay_ms"] = 1500
    try:
        cfg["auto_listen_min_segment_ms"] = max(100, int(cfg.get("auto_listen_min_segment_ms", 1400)))
    except Exception:
        cfg["auto_listen_min_segment_ms"] = 1400
    try:
        cfg["auto_listen_min_voiced_ratio"] = min(1.0, max(0.0, float(cfg.get("auto_listen_min_voiced_ratio", 0.60))))
    except Exception:
        cfg["auto_listen_min_voiced_ratio"] = 0.60
    cfg["tts_streaming_enabled"] = bool(cfg.get("tts_streaming_enabled", False))
    try:
        cfg["tts_stream_min_words"] = int(cfg.get("tts_stream_min_words", 12))
    except Exception:
        cfg["tts_stream_min_words"] = 12
    cfg["tts_stream_min_words"] = max(1, cfg["tts_stream_min_words"])
    cfg["tts_stream_cut_on_punctuation"] = bool(cfg.get("tts_stream_cut_on_punctuation", False))
    whisper_language = str(cfg.get("whisper_language", "es")).strip().lower()
    if not whisper_language:
        whisper_language = "es"
    cfg["whisper_language"] = whisper_language
    whisper_ov_device = str(cfg.get("whisper_openvino_device", "AUTO")).strip().upper()
    if whisper_ov_device not in WHISPER_OV_DEVICE_OPTIONS:
        whisper_ov_device = "AUTO"
    cfg["whisper_openvino_device"] = whisper_ov_device
    cfg["whisper_openvino_model_id"] = str(cfg.get("whisper_openvino_model_id", "")).strip()
    tts_backend = normalize_tts_backend_for_platform(
        cfg.get("tts_backend", DEFAULT_TTS_BACKEND),
        PLATFORM_NAME,
    )
    cfg["tts_backend"] = tts_backend
    openvino_tts_device = str(cfg.get("openvino_tts_device", "AUTO")).strip().upper()
    if openvino_tts_device not in PARLER_OV_DEVICE_OPTIONS:
        openvino_tts_device = "AUTO"
    cfg["openvino_tts_device"] = openvino_tts_device
    cfg["openvino_tts_model_id"] = str(cfg.get("openvino_tts_model_id", "")).strip()
    try:
        cfg["openvino_tts_timeout_s"] = max(3, int(cfg.get("openvino_tts_timeout_s", 25)))
    except Exception:
        cfg["openvino_tts_timeout_s"] = 25
    cfg["openvino_tts_isolated_gpu"] = bool(cfg.get("openvino_tts_isolated_gpu", True))
    try:
        cfg["openvino_tts_speed"] = float(cfg.get("openvino_tts_speed", 1.0))
    except Exception:
        cfg["openvino_tts_speed"] = 1.0
    cfg["openvino_tts_speed"] = min(2.0, max(0.5, cfg["openvino_tts_speed"]))
    try:
        cfg["openvino_tts_gain"] = float(cfg.get("openvino_tts_gain", 1.0))
    except Exception:
        cfg["openvino_tts_gain"] = 1.0
    cfg["openvino_tts_gain"] = min(3.0, max(0.1, cfg["openvino_tts_gain"]))
    cfg["kokoro_model_id"] = str(cfg.get("kokoro_model_id", "kokoro-tts-intel")).strip()
    kokoro_device = str(cfg.get("kokoro_device", "GPU")).strip().upper()
    if kokoro_device not in KOKORO_DEVICE_OPTIONS:
        kokoro_device = "GPU"
    cfg["kokoro_device"] = kokoro_device
    cfg["kokoro_voice"] = str(cfg.get("kokoro_voice", "af_sarah")).strip() or "af_sarah"
    cfg["babelvox_model_id"] = str(cfg.get("babelvox_model_id", "babelvox-openvino-int8")).strip()
    babelvox_device = str(cfg.get("babelvox_device", "CPU")).strip().upper()
    if babelvox_device not in BABELVOX_DEVICE_OPTIONS:
        babelvox_device = "CPU"
    cfg["babelvox_device"] = babelvox_device
    babelvox_precision = str(cfg.get("babelvox_precision", "int8")).strip().lower()
    if babelvox_precision not in BABELVOX_PRECISION_OPTIONS:
        babelvox_precision = "int8"
    cfg["babelvox_precision"] = babelvox_precision
    babelvox_language = str(cfg.get("babelvox_language", "es")).strip().lower()
    if babelvox_language not in WHISPER_LANGUAGE_OPTIONS:
        babelvox_language = "es"
    cfg["babelvox_language"] = babelvox_language
    cfg["tada_model_id"] = str(cfg.get("tada_model_id", "HumeAI/tada-1b")).strip() or "HumeAI/tada-1b"
    cfg["tada_codec_id"] = str(cfg.get("tada_codec_id", "HumeAI/tada-codec")).strip() or "HumeAI/tada-codec"
    cfg["tada_device"] = str(cfg.get("tada_device", "cpu")).strip() or "cpu"
    tada_language = str(cfg.get("tada_language", "en")).strip().lower()
    cfg["tada_language"] = tada_language or "en"
    cfg["tada_reference_audio_path"] = str(cfg.get("tada_reference_audio_path", "")).strip()
    cfg["tada_reference_text"] = str(cfg.get("tada_reference_text", "")).strip()
    try:
        cfg["tada_sample_rate"] = max(8000, int(cfg.get("tada_sample_rate", 24000)))
    except Exception:
        cfg["tada_sample_rate"] = 24000
    cfg["espeak_voice"] = str(cfg.get("espeak_voice", "es")).strip() or "es"
    try:
        cfg["espeak_rate"] = int(cfg.get("espeak_rate", 145))
    except Exception:
        cfg["espeak_rate"] = 145
    cfg["espeak_rate"] = min(450, max(80, cfg["espeak_rate"]))
    try:
        cfg["espeak_pitch"] = int(cfg.get("espeak_pitch", 45))
    except Exception:
        cfg["espeak_pitch"] = 45
    cfg["espeak_pitch"] = min(99, max(0, cfg["espeak_pitch"]))
    try:
        cfg["espeak_amplitude"] = int(cfg.get("espeak_amplitude", 120))
    except Exception:
        cfg["espeak_amplitude"] = 120
    cfg["espeak_amplitude"] = min(200, max(0, cfg["espeak_amplitude"]))
    return cfg


def save_robot_config(config: dict) -> None:
    try:
        ROBOT_CONFIG_FILE.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"✅ Config saved to {ROBOT_CONFIG_FILE}\n")
    except Exception as exc:
        print(f"\n⚠️ Could not save config: {exc}\n")


def apply_runtime_from_config(config: dict) -> None:
    global ACTIVE_DEVICE, ACTIVE_PERFORMANCE_HINT
    device = str(config.get("llm_device", DEFAULT_DEVICE)).strip().upper()
    hint = str(config.get("llm_performance_hint", DEFAULT_PERFORMANCE_HINT)).strip().upper()

    if device not in DEVICE_OPTIONS:
        device = DEFAULT_DEVICE
    if hint not in PERFORMANCE_HINT_OPTIONS:
        hint = DEFAULT_PERFORMANCE_HINT

    ACTIVE_DEVICE = device
    ACTIVE_PERFORMANCE_HINT = hint
    config["llm_device"] = device
    config["llm_performance_hint"] = hint


def list_voices(voices) -> None:
    for i in range(voices.Count):
        print(f"{i}: {voices.Item(i).GetDescription()}")


def apply_voice_config(speaker, voices, config: dict) -> None:
    idx = int(config.get("voice_index", 0))
    if not (0 <= idx < voices.Count):
        idx = 0
        config["voice_index"] = 0
    speaker.Voice = voices.Item(idx)
    speaker.Rate = int(config.get("rate", -2))
    speaker.Volume = int(config.get("volume", 100))
    config["_tts_warmup_done"] = False


def _coinit_windows_com() -> object | None:
    if not IS_WINDOWS:
        return None
    try:
        pythoncom_mod = importlib.import_module("pythoncom")
    except Exception:
        return None
    try:
        pythoncom_mod.CoInitialize()
    except Exception:
        return None
    return pythoncom_mod


def warmup_windows_tts_if_needed(speaker, config: dict) -> None:
    if speaker is None:
        return
    if not bool(config.get("warmup_tts", True)):
        return
    if bool(config.get("_tts_warmup_done", False)):
        return
    warmup_xml = f'<speak><silence msec="{WINDOWS_TTS_WARMUP_SILENCE_MS}"/></speak>'
    try:
        speaker.Speak(warmup_xml, SAPI_SVSFISXML)
    except Exception:
        pass
    finally:
        config["_tts_warmup_done"] = True


def speak_text(speaker, text: str, config: dict, allow_interrupt: bool = False) -> tuple[bool, float]:
    safe_text = xml_escape(text or "")
    base_silence = max(0, int(config.get("silence", 0)))
    xml = f'<speak><silence msec="{base_silence}"/>{safe_text}</speak>'
    t_start = time.perf_counter()
    pythoncom_mod = _coinit_windows_com()
    try:
        with WINDOWS_TTS_LOCK:
            warmup_windows_tts_if_needed(speaker, config)
            with audio_playback_scope():
                speaker.Speak(xml, SAPI_SVSFLAGSASYNC | SAPI_SVSFISXML)
                calc_latency_s = time.perf_counter() - t_start

                if not allow_interrupt:
                    while not speaker.WaitUntilDone(50):
                        pass
                    return False, calc_latency_s

                with KEYBOARD.capture():
                    while not speaker.WaitUntilDone(50):
                        if consume_esc_pressed() or is_audio_cancel_requested():
                            speaker.Speak("", SAPI_SVSFPURGEBEFORESPEAK)
                            return True, calc_latency_s
    finally:
        if pythoncom_mod is not None:
            try:
                pythoncom_mod.CoUninitialize()
            except Exception:
                pass
    return False, calc_latency_s


def ensure_tts_runtime(tts_runtime: dict, config: dict) -> bool:
    compat = tts_runtime.get("compat")
    def release_tts_models() -> None:
        tts_runtime["pipeline"] = None
        tts_runtime["model_id"] = None
        tts_runtime["ov_model_dir"] = None
        tts_runtime["ov_device"] = None
        tts_runtime["torch"] = None
        tts_runtime["torchaudio"] = None
        tts_runtime["kokoro_engine"] = None
        tts_runtime["babelvox_engine"] = None
        tts_runtime["tada_model"] = None
        tts_runtime["tada_encoder"] = None
        tts_runtime["backend"] = None
        tts_runtime["active_key"] = None
        gc.collect()

    backend = normalize_tts_backend_for_platform(
        config.get("tts_backend", DEFAULT_TTS_BACKEND),
        PLATFORM_NAME,
    )
    if backend == "windows":
        if tts_runtime.get("active_key") is not None:
            print("Releasing previous TTS model...")
            release_tts_models()
        tts_runtime["backend"] = "windows"
        return True
    if backend == "espeakng":
        if tts_runtime.get("active_key") is not None:
            print("Releasing previous TTS model...")
            release_tts_models()
        exe = find_espeak_executable()
        if not exe:
            print_error_red("ERROR: eSpeak NG executable not found (espeak-ng/espeak).")
            print_espeak_install_suggestion()
            return False
        tts_runtime["backend"] = "espeakng"
        tts_runtime["active_key"] = ("espeakng", exe)
        print(f"✅ eSpeak NG backend active: {exe}\n")
        return True

    if backend == "openvino":
        requested_device = str(config.get("openvino_tts_device", "AUTO")).strip().upper()
        selected_id_for_key = str(config.get("openvino_tts_model_id", "")).strip() or "unknown"
        compat_key = f"tts:openvino:{selected_id_for_key}"
        if ov_genai is None:
            print_error_red(
                "ERROR: openvino_genai is not installed in this environment. "
                "Use a venv with OpenVINO GenAI or switch TTS backend."
            )
            mark_runtime_chip_compat(compat, compat_key, requested_device, False)
            return False
        models = load_ov_tts_models()
        if not models:
            print_error_red("ERROR: openvino backend is enabled but no verified OpenVINO TTS models are configured.")
            mark_runtime_chip_compat(compat, compat_key, requested_device, False)
            return False

        selected_id = str(config.get("openvino_tts_model_id", "")).strip()
        selected = next((m for m in models if m["id"] == selected_id), None)
        if selected is None:
            selected = models[0]
            config["openvino_tts_model_id"] = selected["id"]
            save_robot_config(config)
        compat_key = f"tts:openvino:{selected['id']}"

        device = str(config.get("openvino_tts_device", "AUTO")).strip().upper()
        if device not in PARLER_OV_DEVICE_OPTIONS:
            device = "AUTO"
            config["openvino_tts_device"] = device
            save_robot_config(config)

        key = ("openvino", selected["id"], device)
        if tts_runtime.get("active_key") == key and tts_runtime.get("pipeline") is not None:
            tts_runtime["backend"] = "openvino"
            return True
        if tts_runtime.get("active_key") is not None and tts_runtime.get("active_key") != key:
            print("Releasing previous TTS model...")
            release_tts_models()

        is_bark = "bark" in str(selected.get("id", "")).lower() or "bark" in str(selected.get("repo_url", "")).lower()
        if is_bark and not has_openvino_tts_artifacts(selected["local"]):
            print_error_red(
                "ERROR: Bark selected, but no OpenVINO IR artifacts were found in local model directory."
            )
            print_error_red(
                f"ERROR: Expected .xml/.bin files under: {selected['local']}"
            )
            print_error_red(
                "ERROR: Bark from HF (suno/bark-small) must be converted/exported to OpenVINO first."
            )
            mark_runtime_chip_compat(compat, compat_key, device, False)
            return False

        try:
            if not is_downloaded(selected["local"]):
                download_ov_tts_model(selected)
        except Exception as exc:
            print_error_red(f"ERROR: Failed to download OpenVINO TTS model: {exc}")
            mark_runtime_chip_compat(compat, compat_key, device, False)
            return False
        if is_bark and not has_openvino_tts_artifacts(selected["local"]):
            print_error_red(
                "ERROR: Downloaded Bark assets are not OpenVINO IR (.xml/.bin)."
            )
            print_error_red(
                "ERROR: Convert Bark to OpenVINO first, then point this model entry to that local folder."
            )
            mark_runtime_chip_compat(compat, compat_key, device, False)
            return False

        try:
            print(f"Loading OpenVINO TTS model '{selected['display']}' on {device}...")
            tts_runtime["pipeline"] = ov_genai.Text2SpeechPipeline(str(selected["local"]), device)
            tts_runtime["model_id"] = selected["id"]
            tts_runtime["ov_model_dir"] = str(selected["local"])
            tts_runtime["ov_device"] = device
            tts_runtime["backend"] = "openvino"
            tts_runtime["active_key"] = key
            if tts_runtime.get("numpy") is None:
                tts_runtime["numpy"] = ensure_dependency("numpy", "numpy", "NumPy")
                if tts_runtime["numpy"] is None:
                    mark_runtime_chip_compat(compat, compat_key, device, False)
                    return False
            if tts_runtime.get("sounddevice") is None:
                tts_runtime["sounddevice"] = ensure_dependency("sounddevice", "sounddevice", "SoundDevice")
                if tts_runtime["sounddevice"] is None:
                    mark_runtime_chip_compat(compat, compat_key, device, False)
                    return False
            mark_runtime_chip_compat(compat, compat_key, device, True)
            print(f"✅ OpenVINO TTS model active: {selected['id']} ({device})\n")
            return True
        except Exception as exc:
            print_error_red(f"ERROR: Failed to initialize OpenVINO TTS backend: {exc}")
            if "bark" in str(selected.get("id", "")).lower() or "bark" in str(selected.get("repo_url", "")).lower():
                print_error_red(
                    "ERROR: Bark OpenVINO is experimental in this app. "
                    "If this fails, use a Bark model exported specifically for OpenVINO TTS runtime."
                )
            mark_runtime_chip_compat(compat, compat_key, device, False)
            release_tts_models()
            return False

    if backend == "kokoro":
        models = load_kokoro_models()
        selected_id = str(config.get("kokoro_model_id", "kokoro-tts-intel")).strip()
        compat_key = f"tts:kokoro:{selected_id or 'unknown'}"
        selected = next((m for m in models if m["id"] == selected_id), None)
        if selected is None and models:
            selected = models[0]
            config["kokoro_model_id"] = selected["id"]
            save_robot_config(config)
        if selected is None:
            print_error_red("ERROR: No Kokoro models configured.")
            mark_runtime_chip_compat(compat, compat_key, config.get("kokoro_device", "GPU"), False)
            return False
        compat_key = f"tts:kokoro:{selected['id']}"

        device = str(config.get("kokoro_device", "GPU")).strip().upper()
        if device not in KOKORO_DEVICE_OPTIONS:
            device = "GPU"
        key = ("kokoro", selected["id"], device)
        if tts_runtime.get("active_key") == key and tts_runtime.get("kokoro_engine") is not None:
            tts_runtime["backend"] = "kokoro"
            return True
        if tts_runtime.get("active_key") is not None and tts_runtime.get("active_key") != key:
            print("Releasing previous TTS model...")
            release_tts_models()

        if not is_repo_downloaded(selected["local"]):
            try:
                download_repo_model(selected, "Kokoro model")
            except Exception as exc:
                print_error_red(f"ERROR: Failed to download Kokoro model: {exc}")
                mark_runtime_chip_compat(compat, compat_key, device, False)
                return False

        kokoro_mod = ensure_dependency_no_deps("kokoro_onnx", "kokoro-onnx", "Kokoro-ONNX")
        if kokoro_mod is None:
            mark_runtime_chip_compat(compat, compat_key, device, False)
            return False
        np_mod = ensure_dependency("numpy", "numpy", "NumPy")
        if np_mod is None:
            mark_runtime_chip_compat(compat, compat_key, device, False)
            return False
        sd_mod = ensure_dependency("sounddevice", "sounddevice", "SoundDevice")
        if sd_mod is None:
            mark_runtime_chip_compat(compat, compat_key, device, False)
            return False
        device_hint = {"CPU": "cpu", "GPU": "igpu", "NPU": "npu"}.get(device, "cpu")
        try:
            engine = None
            ctor = getattr(kokoro_mod, "Kokoro", None)
            if ctor is None:
                raise RuntimeError("Kokoro class not found in kokoro_onnx package.")

            model_path, voices_path = ensure_kokoro_model_files(selected["local"])
            # Current kokoro-onnx API does not accept providers in constructor.
            engine = ctor(str(model_path), str(voices_path))
            if device in {"GPU", "NPU"}:
                rt_mod = ensure_dependency(
                    "onnxruntime",
                    "onnxruntime-openvino",
                    "ONNX Runtime OpenVINO",
                )
                if rt_mod is None:
                    mark_runtime_chip_compat(compat, compat_key, device, False)
                    return False
                provider_chain = [[("OpenVINOExecutionProvider", {"device_type": device})], ["CPUExecutionProvider"]]
                last_exc = None
                switched = False
                selected_provider_desc = "CPUExecutionProvider"
                for providers in provider_chain:
                    try:
                        engine.sess = rt_mod.InferenceSession(str(model_path), providers=providers)
                        switched = True
                        selected_provider_desc = str(providers[0])
                        break
                    except Exception as exc:
                        last_exc = exc
                if not switched:
                    raise RuntimeError(
                        f"Could not configure Kokoro ONNX providers for device {device}: {last_exc}"
                    )
                if selected_provider_desc == "CPUExecutionProvider":
                    mark_runtime_chip_compat(compat, compat_key, device, False)
                    mark_runtime_chip_compat(compat, compat_key, "CPU", True)
                    print_chip_fallback_warning(
                        "Kokoro",
                        requested_chip=device,
                        actual_chip="CPU",
                        reason="OpenVINOExecutionProvider is unavailable or failed to initialize",
                    )
            tts_runtime["kokoro_engine"] = engine
            tts_runtime["kokoro_lang"] = "en-us"
            tts_runtime["numpy"] = np_mod
            tts_runtime["sounddevice"] = sd_mod
            tts_runtime["model_id"] = selected["id"]
            tts_runtime["backend"] = "kokoro"
            tts_runtime["active_key"] = key
            active_providers = []
            try:
                active_providers = list(engine.sess.get_providers())
            except Exception:
                active_providers = []
            providers_text = ", ".join(active_providers) if active_providers else "unknown"
            if device in {"GPU", "NPU"} and "OpenVINOExecutionProvider" not in active_providers:
                actual_chip = "CPU" if "CPUExecutionProvider" in active_providers else providers_text
                mark_runtime_chip_compat(compat, compat_key, device, False)
                if "CPUExecutionProvider" in active_providers:
                    mark_runtime_chip_compat(compat, compat_key, "CPU", True)
                print_chip_fallback_warning(
                    "Kokoro",
                    requested_chip=device,
                    actual_chip=actual_chip,
                    reason="OpenVINOExecutionProvider not active in ONNX Runtime session",
                )
            else:
                mark_runtime_chip_compat(compat, compat_key, device, True)
            print(f"✅ Kokoro model active: {selected['id']} ({device})\n")
            print(f"ℹ️ Kokoro runtime providers: {providers_text}\n")
            return True
        except Exception as exc:
            print_error_red(f"ERROR: Failed to initialize Kokoro backend: {exc}")
            mark_runtime_chip_compat(compat, compat_key, device, False)
            release_tts_models()
            return False

    if backend == "babelvox":
        models = load_babelvox_models()
        selected_id = str(config.get("babelvox_model_id", "babelvox-openvino-int8")).strip()
        compat_key = f"tts:babelvox:{selected_id or 'unknown'}"
        selected = next((m for m in models if m["id"] == selected_id), None)
        if selected is None and models:
            selected = models[0]
            config["babelvox_model_id"] = selected["id"]
            save_robot_config(config)
        if selected is None:
            print_error_red("ERROR: No BabelVox models configured.")
            mark_runtime_chip_compat(compat, compat_key, config.get("babelvox_device", "CPU"), False)
            return False
        compat_key = f"tts:babelvox:{selected['id']}"

        device = str(config.get("babelvox_device", "CPU")).strip().upper()
        if device not in BABELVOX_DEVICE_OPTIONS:
            device = "CPU"
        precision = str(config.get("babelvox_precision", "int8")).strip().lower()
        if precision not in BABELVOX_PRECISION_OPTIONS:
            precision = "int8"
        key = ("babelvox", selected["id"], device, precision)
        if tts_runtime.get("active_key") == key and tts_runtime.get("babelvox_engine") is not None:
            tts_runtime["backend"] = "babelvox"
            return True
        if tts_runtime.get("active_key") is not None and tts_runtime.get("active_key") != key:
            print("Releasing previous TTS model...")
            release_tts_models()

        # Keep model download/cache in the same location used by the catalog entry
        # so "downloaded" status and runtime cache point to the same folder.
        if not is_repo_downloaded(selected["local"]):
            try:
                download_repo_model(selected, "BabelVox model")
            except Exception as exc:
                print_error_red(f"ERROR: Failed to download BabelVox model: {exc}")
                mark_runtime_chip_compat(compat, compat_key, device, False)
                return False

        babelvox_mod = ensure_dependency("babelvox", "babelvox", "BabelVox")
        if babelvox_mod is None:
            mark_runtime_chip_compat(compat, compat_key, device, False)
            return False
        np_mod = ensure_dependency("numpy", "numpy", "NumPy")
        if np_mod is None:
            mark_runtime_chip_compat(compat, compat_key, device, False)
            return False
        sd_mod = ensure_dependency("sounddevice", "sounddevice", "SoundDevice")
        if sd_mod is None:
            mark_runtime_chip_compat(compat, compat_key, device, False)
            return False
        try:
            engine = babelvox_mod.BabelVox(
                device=device,
                precision=precision,
                cache_dir=str(selected["local"]),
            )
            actual_device = None
            for attr_name in ("device", "current_device", "runtime_device"):
                if hasattr(engine, attr_name):
                    try:
                        actual_device = str(getattr(engine, attr_name)).upper()
                        break
                    except Exception:
                        pass
            tts_runtime["babelvox_engine"] = engine
            tts_runtime["numpy"] = np_mod
            tts_runtime["sounddevice"] = sd_mod
            tts_runtime["model_id"] = selected["id"]
            tts_runtime["backend"] = "babelvox"
            tts_runtime["active_key"] = key
            if actual_device and actual_device != device:
                mark_runtime_chip_compat(compat, compat_key, device, False)
                mark_runtime_chip_compat(compat, compat_key, actual_device, True)
                print_chip_fallback_warning(
                    "BabelVox",
                    requested_chip=device,
                    actual_chip=actual_device,
                    reason="backend reported a different runtime device",
                )
            else:
                mark_runtime_chip_compat(compat, compat_key, device, True)
            print(f"✅ BabelVox model active: {selected['id']} ({device}, {precision})\n")
            print(f"ℹ️ BabelVox runtime chip: {device} | precision: {precision}\n")
            return True
        except Exception as exc:
            print_error_red(f"ERROR: Failed to initialize BabelVox backend: {exc}")
            mark_runtime_chip_compat(compat, compat_key, device, False)
            release_tts_models()
            return False

    if backend == "tada":
        ensure_tada_huggingface_compat()
        hf_token = load_hf_token()
        model_id = str(config.get("tada_model_id", "HumeAI/tada-1b")).strip() or "HumeAI/tada-1b"
        codec_id = str(config.get("tada_codec_id", "HumeAI/tada-codec")).strip() or "HumeAI/tada-codec"
        device = str(config.get("tada_device", "cpu")).strip() or "cpu"
        language = str(config.get("tada_language", "en")).strip().lower() or "en"
        reference_audio = Path(str(config.get("tada_reference_audio_path", "")).strip())
        reference_text = str(config.get("tada_reference_text", "")).strip()
        if not reference_audio.exists():
            print_error_red("ERROR: TADA requires an existing reference audio file.")
            print_error_red("ERROR: Set it with /tada_reference_audio <path> or in /config.")
            return False
        if not reference_text:
            print_error_red("ERROR: TADA requires the transcript of the reference audio.")
            print_error_red("ERROR: Set it with /tada_reference_text <text> or in /config.")
            return False
        key = ("tada", model_id, codec_id, device, language, str(reference_audio))
        if (
            tts_runtime.get("active_key") == key
            and tts_runtime.get("tada_model") is not None
            and tts_runtime.get("tada_encoder") is not None
        ):
            tts_runtime["backend"] = "tada"
            return True
        if tts_runtime.get("active_key") is not None and tts_runtime.get("active_key") != key:
            print("Releasing previous TTS model...")
            release_tts_models()

        torch_mod = ensure_dependency("torch", "torch", "PyTorch")
        if torch_mod is None:
            return False
        torchaudio_mod = ensure_dependency("torchaudio", "torchaudio", "TorchAudio")
        if torchaudio_mod is None:
            return False
        tada_root = ensure_dependency("tada", "git+https://github.com/HumeAI/tada.git", "HumeAI TADA")
        if tada_root is None:
            return False
        np_mod = ensure_dependency("numpy", "numpy", "NumPy")
        if np_mod is None:
            return False
        sd_mod = ensure_dependency("sounddevice", "sounddevice", "SoundDevice")
        if sd_mod is None:
            return False
        try:
            encoder_mod = importlib.import_module("tada.modules.encoder")
            model_mod = importlib.import_module("tada.modules.tada")
            encoder_cls = getattr(encoder_mod, "Encoder")
            tada_cls = getattr(model_mod, "TadaForCausalLM")
            print(f"Loading Hume TADA model '{model_id}' on {device}...")
            try:
                encoder = encoder_cls.from_pretrained(codec_id, language=language, token=hf_token)
            except TypeError:
                try:
                    encoder = encoder_cls.from_pretrained(codec_id, language=language, use_auth_token=hf_token)
                except TypeError:
                    encoder = encoder_cls.from_pretrained(codec_id)
            try:
                model = tada_cls.from_pretrained(model_id, token=hf_token)
            except TypeError:
                try:
                    model = tada_cls.from_pretrained(model_id, use_auth_token=hf_token)
                except TypeError:
                    model = tada_cls.from_pretrained(model_id)
            if hasattr(encoder, "to"):
                encoder = encoder.to(device)
            if hasattr(model, "to"):
                model = model.to(device)
            if hasattr(model, "eval"):
                model.eval()
            tts_runtime["tada_model"] = model
            tts_runtime["tada_encoder"] = encoder
            tts_runtime["torch"] = torch_mod
            tts_runtime["torchaudio"] = torchaudio_mod
            tts_runtime["numpy"] = np_mod
            tts_runtime["sounddevice"] = sd_mod
            tts_runtime["model_id"] = model_id
            tts_runtime["backend"] = "tada"
            tts_runtime["active_key"] = key
            print(f"✅ Hume TADA backend active: {model_id} ({device})\n")
            return True
        except Exception as exc:
            print_error_red(f"ERROR: Failed to initialize Hume TADA backend: {exc}")
            http_proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
            https_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
            if http_proxy or https_proxy:
                print_error_red(
                    f"ERROR: Proxy env detected. HTTP_PROXY={http_proxy!r} HTTPS_PROXY={https_proxy!r}"
                )
                print_error_red("ERROR: If those proxies are invalid, TADA model download from Hugging Face will fail.")
            if hf_token:
                print_error_red("ERROR: HF token is present, so if this still says 'gated repo', your account likely does not have access approved for that model.")
            print_error_red("ERROR: TADA needs a valid reference audio and transcript, and the Hugging Face assets may require access.")
            release_tts_models()
            return False

    print_error_red(f"ERROR: Unsupported TTS backend: {backend}")
    return False


def speak_text_backend(
    speaker,
    text: str,
    config: dict,
    tts_runtime: dict,
    allow_interrupt: bool = False,
) -> tuple[bool, float]:
    with tts_activity_scope():
        backend = normalize_tts_backend_for_platform(
            config.get("tts_backend", DEFAULT_TTS_BACKEND),
            PLATFORM_NAME,
        )
        if backend == "windows":
            if speaker is None:
                print_error_red("ERROR: Windows TTS backend selected but SAPI voice engine is unavailable.")
                return False, 0.0
            return speak_text(speaker, text, config, allow_interrupt=allow_interrupt)
        if backend == "espeakng":
            return speak_espeak_ng(text, config, allow_interrupt=allow_interrupt)
        if not ensure_tts_runtime(tts_runtime, config):
            return False, 0.0
        try:
            t_start = time.perf_counter()
            np_mod = tts_runtime["numpy"]
            sd_mod = tts_runtime["sounddevice"]
            if backend == "openvino":
                timeout_s = int(config.get("openvino_tts_timeout_s", 25))
                device = str(config.get("openvino_tts_device", "AUTO")).strip().upper()
                isolated_gpu = bool(config.get("openvino_tts_isolated_gpu", True))
                use_isolated = isolated_gpu and device == "GPU"
                if use_isolated:
                    model_dir = str(tts_runtime.get("ov_model_dir") or "")
                    if not model_dir:
                        raise RuntimeError("OpenVINO TTS model directory is unavailable.")
                    ctx = mp.get_context("spawn")
                    result_queue = ctx.Queue(maxsize=1)
                    proc = ctx.Process(
                        target=_openvino_tts_worker,
                        args=(model_dir, device, text, result_queue),
                        daemon=True,
                    )
                    proc.start()
                    proc.join(timeout=timeout_s)
                    if proc.is_alive():
                        proc.terminate()
                        proc.join(timeout=2)
                        print_error_red(
                            f"ERROR: OpenVINO TTS generation timed out after {timeout_s}s on device {device}."
                        )
                        print_error_red("ERROR: Worker process was terminated to avoid CPU lock.")
                        print_error_red("ERROR: Suggestion: switch OpenVINO TTS device to CPU.")
                        return False, time.perf_counter() - t_start
                    payload = result_queue.get_nowait() if not result_queue.empty() else None
                    if not payload:
                        raise RuntimeError("OpenVINO TTS worker returned no data.")
                    if not payload.get("ok"):
                        raise RuntimeError(str(payload.get("error", "unknown OpenVINO TTS worker error")))
                    calc_latency_s = time.perf_counter() - t_start
                    audio = np_mod.array(payload.get("audio", []), dtype=np_mod.float32).reshape(-1)
                else:
                    result_box = {"result": None, "error": None}

                    def _ov_generate():
                        try:
                            result_box["result"] = tts_runtime["pipeline"].generate(text)
                        except Exception as exc:
                            result_box["error"] = exc

                    worker = threading.Thread(target=_ov_generate, daemon=True)
                    worker.start()
                    worker.join(timeout=timeout_s)
                    if worker.is_alive():
                        print_error_red(
                            f"ERROR: OpenVINO TTS generation timed out after {timeout_s}s on "
                            f"device {config.get('openvino_tts_device', 'AUTO')}."
                        )
                        print_error_red("ERROR: Suggestion: switch OpenVINO TTS device to CPU.")
                        tts_runtime["pipeline"] = None
                        tts_runtime["active_key"] = None
                        tts_runtime["backend"] = None
                        return False, time.perf_counter() - t_start
                    if result_box["error"] is not None:
                        raise result_box["error"]
                    result = result_box["result"]
                    calc_latency_s = time.perf_counter() - t_start
                    speeches = getattr(result, "speeches", None)
                    if not speeches:
                        print_error_red("ERROR: openvino TTS returned no audio.")
                        return False, calc_latency_s
                    audio_obj = speeches[0]
                    if hasattr(audio_obj, "data"):
                        audio = np_mod.array(audio_obj.data, dtype=np_mod.float32).reshape(-1)
                    else:
                        audio = np_mod.array(audio_obj, dtype=np_mod.float32).reshape(-1)
                sample_rate = 24000
                audio = apply_openvino_tts_postprocess(audio, np_mod, config)
            elif backend == "kokoro":
                engine = tts_runtime.get("kokoro_engine")
                if engine is None:
                    raise RuntimeError("Kokoro engine not initialized.")
                voice = str(config.get("kokoro_voice", "af_sarah")).strip() or "af_sarah"
                lang_code = str(config.get("babelvox_language", "en")).strip().lower()
                kokoro_lang = {
                    "es": "es",
                    "en": "en-us",
                    "pt": "pt-br",
                    "fr": "fr-fr",
                    "it": "it",
                    "de": "de",
                }.get(lang_code, "en-us")
                output = engine.create(text, voice=voice, speed=1.0, lang=kokoro_lang)
                calc_latency_s = time.perf_counter() - t_start
                sample_rate = 24000
                if isinstance(output, tuple) and len(output) >= 2:
                    audio_raw = output[0]
                    try:
                        sample_rate = int(output[1])
                    except Exception:
                        pass
                elif isinstance(output, dict):
                    audio_raw = output.get("audio") or output.get("wav") or output.get("samples")
                    if output.get("sample_rate") is not None:
                        sample_rate = int(output["sample_rate"])
                else:
                    audio_raw = output
                audio = np_mod.array(audio_raw, dtype=np_mod.float32).reshape(-1)
            elif backend == "babelvox":
                engine = tts_runtime.get("babelvox_engine")
                if engine is None:
                    raise RuntimeError("BabelVox engine not initialized.")
                lang_code = str(config.get("babelvox_language", "es")).strip().lower()
                language = BABELVOX_LANGUAGE_MAP.get(lang_code, "English")
                wav, sample_rate = engine.generate(text, language=language)
                calc_latency_s = time.perf_counter() - t_start
                audio = np_mod.array(wav, dtype=np_mod.float32).reshape(-1)
            elif backend == "tada":
                model = tts_runtime.get("tada_model")
                encoder = tts_runtime.get("tada_encoder")
                torch_mod = tts_runtime.get("torch")
                torchaudio_mod = tts_runtime.get("torchaudio")
                if model is None or encoder is None or torch_mod is None or torchaudio_mod is None:
                    raise RuntimeError("TADA runtime not initialized.")
                reference_path = Path(str(config.get("tada_reference_audio_path", "")).strip())
                reference_text = str(config.get("tada_reference_text", "")).strip()
                if not reference_path.exists():
                    raise RuntimeError("TADA reference audio file does not exist.")
                if not reference_text:
                    raise RuntimeError("TADA reference transcript is empty.")
                waveform, ref_sample_rate = torchaudio_mod.load(str(reference_path))
                if getattr(waveform, "dim", lambda: 0)() == 1:
                    waveform = waveform.unsqueeze(0)
                if hasattr(waveform, "to"):
                    waveform = waveform.to(str(config.get("tada_device", "cpu")).strip() or "cpu")
                prompt_kwargs = {"sample_rate": int(ref_sample_rate)}
                if reference_text:
                    prompt_kwargs["text"] = [reference_text]
                with torch_mod.no_grad():
                    try:
                        prompt = encoder(waveform, **prompt_kwargs)
                    except TypeError:
                        prompt_kwargs["text"] = reference_text
                        prompt = encoder(waveform, **prompt_kwargs)
                    try:
                        output = model.generate(prompt=prompt, text=text)
                    except TypeError:
                        output = model.generate(prompt=prompt, text=[text])
                calc_latency_s = time.perf_counter() - t_start
                audio, detected_sample_rate = extract_tada_audio_output(output, torch_mod, np_mod)
                sample_rate = int(detected_sample_rate or config.get("tada_sample_rate", 24000))
            else:
                print_error_red(f"ERROR: Unsupported TTS backend: {backend}")
                return False, 0.0
            if getattr(audio, "size", 0) == 0:
                print_error_red(f"ERROR: {backend} returned an empty audio buffer.")
                return False, calc_latency_s
            with audio_playback_scope():
                sd_mod.play(audio, samplerate=sample_rate)
                if not allow_interrupt:
                    sd_mod.wait()
                    return False, calc_latency_s
                while True:
                    if consume_esc_pressed() or is_audio_cancel_requested():
                        sd_mod.stop()
                        return True, calc_latency_s
                    stream = sd_mod.get_stream()
                    if stream is None or not stream.active:
                        break
                    time.sleep(0.02)
                return False, calc_latency_s
        except Exception as exc:
            print_error_red(f"ERROR: {backend} synthesis failed: {exc}")
            return False, 0.0


def choose_voice_interactive(speaker, voices, config: dict) -> None:
    if not platform_supports_native_voices(PLATFORM_NAME):
        print("\n⚠️ Native voice selection is only available on Windows.")
        print("   On Linux, use '/config' with the 'espeakng' backend and '/espeak_voices'.\n")
        return
    print("\nAvailable voices:\n")
    list_voices(voices)
    while True:
        value = input("Choose voice number (or 'cancel'): ").strip().lower()
        if value == "cancel":
            print("")
            return
        if value.isdigit() and 0 <= int(value) < voices.Count:
            idx = int(value)
            speaker.Voice = voices.Item(idx)
            config["voice_index"] = idx
            save_robot_config(config)
            print(f"\n✅ Voice changed to: {voices.Item(idx).GetDescription()}\n")
            return
        print("Invalid voice number.")


def ask_install_dependency(name: str) -> bool:
    while True:
        answer = input(f"Missing dependency '{name}'. Install now? [Y/n]: ").strip().lower()
        if answer in {"", "y", "yes", "s", "si"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please answer y or n.")


def ensure_dependency(module_name: str, pip_name: str, display_name: str):
    try:
        return importlib.import_module(module_name)
    except ImportError:
        if not ask_install_dependency(display_name):
            print(f"\n⚠️ Cancelled. {display_name} is required.\n")
            return None
        print(f"Installing dependency: {display_name}")
        result = subprocess.run([os.sys.executable, "-m", "pip", "install", pip_name], check=False)
        if result.returncode != 0:
            print(f"\n⚠️ Failed to install {pip_name}.\n")
            return None
        try:
            return importlib.import_module(module_name)
        except ImportError:
            print(f"\n⚠️ {display_name} is still unavailable after install.\n")
            return None


def ensure_dependency_no_deps(module_name: str, pip_name: str, display_name: str):
    try:
        return importlib.import_module(module_name)
    except ImportError:
        if not ask_install_dependency(display_name):
            print(f"\n⚠️ Cancelled. {display_name} is required.\n")
            return None
        print(f"Installing dependency: {display_name}")
        result = subprocess.run(
            [os.sys.executable, "-m", "pip", "install", "--no-deps", pip_name],
            check=False,
        )
        if result.returncode != 0:
            print(f"\n⚠️ Failed to install {pip_name}.\n")
            return None
        try:
            return importlib.import_module(module_name)
        except ImportError:
            print(f"\n⚠️ {display_name} is still unavailable after install.\n")
            return None


def clear_keyboard_buffer() -> None:
    KEYBOARD.clear_buffer()


def set_tts_activity(active: bool) -> None:
    global TTS_ACTIVITY_COUNT, LAST_TTS_ACTIVITY_END_TS
    with TTS_ACTIVITY_LOCK:
        if active:
            TTS_ACTIVITY_COUNT += 1
        else:
            TTS_ACTIVITY_COUNT = max(0, TTS_ACTIVITY_COUNT - 1)
        if TTS_ACTIVITY_COUNT > 0:
            TTS_PLAYING_EVENT.set()
        else:
            TTS_PLAYING_EVENT.clear()
            LAST_TTS_ACTIVITY_END_TS = time.monotonic()


def is_tts_active() -> bool:
    return TTS_PLAYING_EVENT.is_set()


def set_audio_playback_activity(active: bool) -> None:
    global PLAYBACK_ACTIVITY_COUNT
    with PLAYBACK_ACTIVITY_LOCK:
        if active:
            PLAYBACK_ACTIVITY_COUNT += 1
        else:
            PLAYBACK_ACTIVITY_COUNT = max(0, PLAYBACK_ACTIVITY_COUNT - 1)
        if PLAYBACK_ACTIVITY_COUNT > 0:
            AUDIO_PLAYBACK_EVENT.set()
        else:
            AUDIO_PLAYBACK_EVENT.clear()


@contextlib.contextmanager
def audio_playback_scope():
    set_audio_playback_activity(True)
    try:
        yield
    finally:
        set_audio_playback_activity(False)


def is_audio_playback_active() -> bool:
    return AUDIO_PLAYBACK_EVENT.is_set()


def request_audio_cancel() -> None:
    AUDIO_CANCEL_EVENT.set()


def clear_audio_cancel() -> None:
    AUDIO_CANCEL_EVENT.clear()


def is_audio_cancel_requested() -> bool:
    return AUDIO_CANCEL_EVENT.is_set()


def is_tts_blocking_auto_listen(config: dict | None = None) -> bool:
    if is_tts_active():
        return True
    cfg = config if isinstance(config, dict) else {}
    cooldown_ms = max(0, int(cfg.get("auto_listen_resume_delay_ms", 1500)))
    if cooldown_ms <= 0:
        return False
    return (time.monotonic() - LAST_TTS_ACTIVITY_END_TS) < (cooldown_ms / 1000.0)


@contextlib.contextmanager
def tts_activity_scope():
    set_tts_activity(True)
    try:
        yield
    finally:
        set_tts_activity(False)


def consume_esc_pressed() -> bool:
    esc = False
    while True:
        key = KEYBOARD.read_key_nonblocking()
        if key is None:
            break
        if key == "\x1b":
            esc = True
    return esc


def record_until_space(sd_mod, np_mod, sample_rate: int = 16000, channels: int = 1, blocksize: int = 1024, device=None):
    frames = []
    stop_ts = time.perf_counter()

    def callback(indata, _frames, _time, status):
        if status:
            print(f"[audio] warning: {status}")
        frames.append(indata.copy())

    print("🎙️ Listening... press SPACE to stop.")
    with KEYBOARD.capture():
        clear_keyboard_buffer()
        with sd_mod.InputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype="float32",
            blocksize=blocksize,
            device=device,
            callback=callback,
        ):
            while True:
                key = KEYBOARD.read_key_nonblocking()
                if key == " ":
                    stop_ts = time.perf_counter()
                    break
                time.sleep(0.01)

    if not frames:
        return np_mod.array([], dtype=np_mod.float32), stop_ts
    return np_mod.concatenate(frames, axis=0).reshape(-1), stop_ts


def whisper_local_model_path(whisper_mod, model_name: str) -> str | None:
    models = getattr(whisper_mod, "_MODELS", {})
    if model_name not in models:
        return None
    return os.path.join(os.path.expanduser("~"), ".cache", "whisper", os.path.basename(models[model_name]))


def whisper_model_size_info(whisper_mod, model_name: str) -> tuple[bool, str]:
    local_model = whisper_local_model_path(whisper_mod, model_name)
    if local_model and os.path.exists(local_model):
        try:
            size = os.path.getsize(local_model)
            return True, human_bytes(size)
        except OSError:
            return True, "unknown"
    expected = WHISPER_EXPECTED_SIZE_BYTES.get(model_name)
    if expected is not None:
        return False, f"~{human_bytes(expected)}"
    return False, "unknown"


def resolve_audio_input_device(config: dict, sd_mod):
    raw = str(config.get("audio_input_device", "")).strip()
    if not raw:
        return None
    try:
        if raw.lstrip("-").isdigit():
            return int(raw)
    except Exception:
        pass
    try:
        devices = sd_mod.query_devices()
    except Exception:
        return raw
    for idx, dev in enumerate(devices):
        try:
            if int(dev.get("max_input_channels", 0)) <= 0:
                continue
            if str(dev.get("name", "")).strip() == raw:
                return idx
        except Exception:
            continue
    return raw


def _normalized_audio_device_name(name: str) -> str:
    text = " ".join(str(name or "").replace("\r", " ").replace("\n", " ").split())
    return text.strip()


def list_audio_input_devices(sd_mod) -> list[tuple[int, str, int]]:
    devices = []
    hostapi_names: dict[int, str] = {}
    with contextlib.suppress(Exception):
        for idx, hostapi in enumerate(sd_mod.query_hostapis()):
            hostapi_names[idx] = str(hostapi.get("name", "")).strip()
    preferred_hostapis = ["Windows WASAPI", "WASAPI", "Windows DirectSound", "DirectSound", "MME"]
    generic_names = {
        "microsoft sound mapper - input",
        "asignador de sonido microsoft - input",
        "primary sound capture driver",
        "controlador primario de captura de sonido",
    }
    by_name: dict[str, tuple[int, str, int, int]] = {}
    for idx, dev in enumerate(sd_mod.query_devices()):
        try:
            max_in = int(dev.get("max_input_channels", 0))
        except Exception:
            max_in = 0
        if max_in <= 0:
            continue
        raw_name = _normalized_audio_device_name(dev.get("name", f"Input {idx}")) or f"Input {idx}"
        if raw_name.lower() in generic_names:
            continue
        hostapi_index = int(dev.get("hostapi", -1))
        hostapi_name = hostapi_names.get(hostapi_index, "")
        priority = preferred_hostapis.index(hostapi_name) if hostapi_name in preferred_hostapis else len(preferred_hostapis)
        key = raw_name.lower()
        current = by_name.get(key)
        candidate = (priority, raw_name, max_in, idx)
        if current is None or candidate[0] < current[0]:
            by_name[key] = candidate
    for _priority, name, max_in, idx in sorted(by_name.values(), key=lambda item: item[1].lower()):
        devices.append((idx, name, max_in))
    return devices


def build_audio_monitor_frame(
    np_mod,
    level: float,
    threshold: float,
    speech_detected: bool,
    width: int = 420,
    height: int = 250,
    metrics: list[dict] | None = None,
):
    frame = np_mod.zeros((height, width, 3), dtype=np_mod.uint8)
    frame[:, :] = (18, 18, 18)
    bar_x = 24
    bar_w = width - 48
    bar_h = 18
    if metrics is None:
        metrics = [{"label": "Speech Prob", "value": level, "threshold": threshold, "active": speech_detected}]
    for idx, metric in enumerate(metrics):
        value = max(0.0, min(1.0, float(metric.get("value", 0.0))))
        metric_threshold = max(0.0, min(1.0, float(metric.get("threshold", 1.0))))
        active = bool(metric.get("active", False))
        bar_y = 44 + idx * 42
        fill_w = int(bar_w * value)
        threshold_x = bar_x + int(bar_w * metric_threshold)
        color = (0, 200, 0) if active else (0, 170, 255)
        frame[bar_y:bar_y + bar_h, bar_x:bar_x + bar_w] = (45, 45, 45)
        if fill_w > 0:
            frame[bar_y:bar_y + bar_h, bar_x:bar_x + fill_w] = color
        frame[bar_y - 6:bar_y + bar_h + 6, max(bar_x, threshold_x - 1):min(width, threshold_x + 1)] = (0, 0, 255)
        indicator_cx = width - 16
        indicator_cy = bar_y + (bar_h // 2)
        radius = 6
        yy, xx = np_mod.ogrid[:height, :width]
        mask = (xx - indicator_cx) ** 2 + (yy - indicator_cy) ** 2 <= radius ** 2
        frame[mask] = (0, 190, 0) if active else (70, 70, 70)
    return frame


def choose_audio_input_device_interactive(sd_mod, current_value: str = "") -> tuple[str, str] | None:
    devices = list_audio_input_devices(sd_mod)
    if not devices:
        print("\nNo audio input devices were detected.\n")
        return None
    print("\nAudio input devices:\n")
    current = str(current_value or "").strip()
    for number, (idx, name, channels) in enumerate(devices, 1):
        marker = ""
        if current and (current == str(idx) or current == name):
            marker = " (current)"
        print(f"  {number}) {name} [index={idx}, channels={channels}]{marker}")
    print("")
    while True:
        value = input("Choose input device number (or 'cancel'): ").strip().lower()
        if value == "cancel":
            return None
        if not value.isdigit() or not (1 <= int(value) <= len(devices)):
            print("Invalid option.")
            continue
        idx, name, _channels = devices[int(value) - 1]
        return str(idx), name


def is_whisper_classic_downloaded(model_name: str) -> bool:
    try:
        whisper_mod = importlib.import_module("whisper")
        local_model = whisper_local_model_path(whisper_mod, model_name)
        return bool(local_model and os.path.exists(local_model))
    except Exception:
        return False


def resolve_ov_whisper_language(model_dir: Path, requested_lang: str) -> tuple[str | None, str | None]:
    req = str(requested_lang or "").strip().lower()
    if not req or req == "auto":
        return None, None

    gen_cfg = model_dir / "generation_config.json"
    try:
        data = json.loads(gen_cfg.read_text(encoding="utf-8"))
    except Exception:
        return req, None

    lang_to_id = data.get("lang_to_id")
    if not isinstance(lang_to_id, dict) or not lang_to_id:
        return req, None

    keys = {str(k): k for k in lang_to_id.keys()}
    name = WHISPER_LANGUAGE_CODE_TO_NAME.get(req, req)
    candidates = [
        req,
        req.lower(),
        req.upper(),
        f"<|{req}|>",
        f"<|{req.lower()}|>",
        name,
        name.lower(),
        name.upper(),
        f"<|{name}|>",
        f"<|{name.lower()}|>",
    ]
    for cand in candidates:
        if cand in keys:
            return keys[cand], None

    available = ", ".join(list(keys.keys())[:8])
    if len(keys) > 8:
        available += ", ..."
    warning = (
        f"Whisper OV language '{req}' not found in generation_config.json lang_to_id. "
        f"Using auto. Available keys: {available}"
    )
    return None, warning


def ensure_stt_runtime(stt_runtime: dict, config: dict) -> bool:
    compat = stt_runtime.get("compat")
    def reset_loaded_models() -> None:
        stt_runtime["model"] = None
        stt_runtime["model_name"] = None
        stt_runtime["ov_pipeline"] = None
        stt_runtime["ov_model_id"] = None
        stt_runtime["ov_model_dir"] = None
        stt_runtime["backend"] = None

    if stt_runtime.get("numpy") is None:
        stt_runtime["numpy"] = ensure_dependency("numpy", "numpy", "NumPy")
        if stt_runtime["numpy"] is None:
            return False
    if stt_runtime.get("sounddevice") is None:
        stt_runtime["sounddevice"] = ensure_dependency("sounddevice", "sounddevice", "SoundDevice")
        if stt_runtime["sounddevice"] is None:
            return False

    use_ov = bool(config.get("whisper_openvino", False))
    if use_ov:
        requested_device = str(config.get("whisper_openvino_device", "AUTO")).strip().upper()
        selected_id_for_key = str(config.get("whisper_openvino_model_id", "")).strip() or "unknown"
        compat_key = f"stt:whisper_ov:{selected_id_for_key}"
        if ov_genai is None:
            print_error_red(
                "ERROR: whisper_openvino is enabled but openvino_genai is not installed "
                "in this environment."
            )
            mark_runtime_chip_compat(compat, compat_key, requested_device, False)
            return False
        ov_models = load_whisper_ov_models()
        if not ov_models:
            print_error_red("ERROR: Whisper OpenVINO is enabled but no OV models are configured.")
            mark_runtime_chip_compat(compat, compat_key, requested_device, False)
            return False

        selected_id = str(config.get("whisper_openvino_model_id", "")).strip()
        selected = next((m for m in ov_models if m["id"] == selected_id), None)
        if selected is None:
            selected = ov_models[0]
            config["whisper_openvino_model_id"] = selected["id"]
            save_robot_config(config)
        compat_key = f"stt:whisper_ov:{selected['id']}"

        ov_device = str(config.get("whisper_openvino_device", "AUTO")).strip().upper()
        if ov_device not in WHISPER_OV_DEVICE_OPTIONS:
            ov_device = "AUTO"
            config["whisper_openvino_device"] = ov_device
            save_robot_config(config)

        key = ("ov", selected["id"], ov_device)
        if stt_runtime.get("active_key") != key or stt_runtime.get("ov_pipeline") is None:
            if stt_runtime.get("active_key") is not None:
                print("Releasing previous STT model...")
                reset_loaded_models()
                stt_runtime["active_key"] = None
                gc.collect()
            try:
                if not is_downloaded(selected["local"]):
                    download_whisper_ov_model(selected)
                print(f"Loading Whisper OpenVINO model '{selected['display']}' on {ov_device}...")
                stt_runtime["ov_pipeline"] = ov_genai.WhisperPipeline(str(selected["local"]), ov_device)
                stt_runtime["ov_model_id"] = selected["id"]
                stt_runtime["ov_model_dir"] = str(selected["local"])
                stt_runtime["whisper"] = None
                stt_runtime["backend"] = "ov"
                stt_runtime["active_key"] = key
                mark_runtime_chip_compat(compat, compat_key, ov_device, True)
                print(f"✅ Whisper OV model active: {selected['id']} ({ov_device})\n")
            except Exception as exc:
                print_error_red(f"ERROR: Failed to initialize Whisper OpenVINO: {exc}")
                mark_runtime_chip_compat(compat, compat_key, ov_device, False)
                reset_loaded_models()
                stt_runtime["active_key"] = None
                return False
        else:
            stt_runtime["backend"] = "ov"
        return True

    if stt_runtime.get("whisper") is None:
        stt_runtime["whisper"] = ensure_dependency("whisper", "openai-whisper", "Whisper")
        if stt_runtime["whisper"] is None:
            return False

    model_name = str(config.get("whisper_model", DEFAULT_WHISPER_MODEL))
    compat_key = f"stt:whisper:{model_name}"
    if model_name not in WHISPER_MODELS:
        model_name = DEFAULT_WHISPER_MODEL
        config["whisper_model"] = model_name
        save_robot_config(config)
        compat_key = f"stt:whisper:{model_name}"

    key = ("whisper", model_name)
    if stt_runtime.get("active_key") != key or stt_runtime.get("model") is None:
        if stt_runtime.get("active_key") is not None:
            print("Releasing previous STT model...")
            reset_loaded_models()
            stt_runtime["active_key"] = None
            gc.collect()

        local_model = whisper_local_model_path(stt_runtime["whisper"], model_name)
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
            mark_runtime_chip_compat(compat, compat_key, "CPU", True)
            print(f"✅ Whisper model active: {model_name}\n")
        except Exception as exc:
            print(f"\n⚠️ Failed to load Whisper model '{model_name}': {exc}\n")
            mark_runtime_chip_compat(compat, compat_key, "CPU", False)
            reset_loaded_models()
            stt_runtime["active_key"] = None
            return False
    return True


def transcribe_from_mic(stt_runtime: dict, config: dict) -> tuple[str, float]:
    if not ensure_stt_runtime(stt_runtime, config):
        return "", 0.0
    np_mod = stt_runtime["numpy"]
    device = resolve_audio_input_device(config, stt_runtime["sounddevice"])
    audio, speech_end_ts = record_until_space(stt_runtime["sounddevice"], np_mod, device=device)
    if getattr(audio, "size", 0) == 0:
        print("\n⚠️ No audio captured.\n")
        return "", 0.0
    return transcribe_audio_buffer(stt_runtime, config, audio, speech_end_ts)


def transcribe_audio_buffer(stt_runtime: dict, config: dict, audio, speech_end_ts: float | None = None) -> tuple[str, float]:
    if not ensure_stt_runtime(stt_runtime, config):
        return "", 0.0
    if getattr(audio, "size", 0) == 0:
        return "", 0.0
    speech_end_ts = speech_end_ts if speech_end_ts is not None else time.perf_counter()
    is_ov = stt_runtime.get("backend") == "ov"
    whisper_language = str(config.get("whisper_language", "es")).strip().lower()
    if whisper_language not in WHISPER_LANGUAGE_OPTIONS:
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
                    resolved_lang, warning = resolve_ov_whisper_language(model_dir, whisper_language)
                if warning:
                    print_error_red(f"ERROR: {warning}")
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
        print_error_red(f"ERROR: STT transcription failed: {exc}")
        return "", time.perf_counter() - speech_end_ts
    speech_end_to_text_s = time.perf_counter() - speech_end_ts
    if not text:
        print("\n⚠️ No speech detected.\n")
        return "", speech_end_to_text_s
    print(f"You said: {text}\n")
    return text, speech_end_to_text_s


def save_float_audio_to_wav(path: Path, audio, sample_rate: int, np_mod) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np_mod.array(audio, dtype=np_mod.float32).reshape(-1)
    if getattr(data, "size", 0) == 0:
        raise ValueError("Cannot save empty audio.")
    clipped = np_mod.clip(data, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np_mod.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(pcm16.tobytes())


def capture_tada_reference_from_mic(stt_runtime: dict, config: dict, sample_rate: int = 16000) -> tuple[Path | None, str]:
    if not ensure_stt_runtime(stt_runtime, config):
        return None, ""
    np_mod = stt_runtime["numpy"]
    sd_mod = stt_runtime["sounddevice"]
    device = resolve_audio_input_device(config, sd_mod)
    print("\nTADA reference capture")
    print("Speak with your normal voice and press SPACE to stop.\n")
    audio, speech_end_ts = record_until_space(sd_mod, np_mod, sample_rate=sample_rate, device=device)
    if getattr(audio, "size", 0) == 0:
        print("\n⚠️ No audio captured.\n")
        return None, ""
    text, _stt_time = transcribe_audio_buffer(stt_runtime, config, audio, speech_end_ts=speech_end_ts)
    if not text:
        print("\n⚠️ Could not derive a reference transcript from the recording.\n")
        return None, ""
    target_path = CACHE_DIR / "tada_reference.wav"
    save_float_audio_to_wav(target_path, audio, sample_rate, np_mod)
    config["tada_reference_audio_path"] = str(target_path)
    config["tada_reference_text"] = text
    save_robot_config(config)
    print(f"✅ TADA reference audio saved: {target_path}")
    print("✅ TADA reference transcript updated.\n")
    return target_path, text


def process_auto_listen_text(
    text: str,
    llm_state: dict,
    stats: dict,
    speaker,
    voice_config: dict,
    stt_runtime: dict,
    tts_runtime: dict,
) -> None:
    user_text = str(text or "").strip()
    if not user_text:
        return
    print(f"[auto-listen] {user_text}\n")
    if bool(voice_config.get("repeat", False)):
        if bool(voice_config.get("audio_enabled", True)):
            speak_text_backend(
                speaker,
                user_text,
                voice_config,
                tts_runtime,
                allow_interrupt=True,
            )
        return
    run_chat_turn(
        user_text,
        llm_state.get("pipe"),
        llm_state.get("current"),
        llm_state.get("history", []),
        stats,
        speaker,
        voice_config,
        tts_runtime,
    )


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


def speak_wake_word_response(
    message: str,
    speaker,
    config: dict,
    tts_runtime: dict,
) -> None:
    if not message:
        return
    if not bool(config.get("audio_enabled", True)):
        return
    try:
        speak_text_backend(
            speaker,
            str(message),
            config,
            tts_runtime,
            allow_interrupt=True,
        )
    except Exception as exc:
        print_error_red(f"ERROR: Wake word TTS failed: {exc}")


def handle_wake_word_transcript(
    text: str,
    auto_listen_runtime: dict,
    speaker,
    config: dict,
    tts_runtime: dict,
) -> bool:
    if not bool(config.get("wake_word_enabled", False)):
        return False
    wake_phrase = str(config.get("wake_word_phrase", "hola robot")).strip()
    stop_phrase = str(config.get("wake_word_stop_phrase", "adios robot")).strip()
    auto_active = bool(config.get("auto_listen_enabled", False))
    if not auto_active:
        if transcript_matches_phrase(text, wake_phrase):
            config["auto_listen_enabled"] = True
            save_robot_config(config)
            print(f"[wake-word] activated by: {text}\n")
            speak_wake_word_response(str(config.get("wake_word_on_response", "Te escucho.")).strip(), speaker, config, tts_runtime)
            return True
        return True
    if transcript_matches_phrase(text, stop_phrase):
        config["auto_listen_enabled"] = False
        save_robot_config(config)
        print(f"[wake-word] deactivated by: {text}\n")
        speak_wake_word_response(
            str(config.get("wake_word_off_response", "Modo escucha desactivado.")).strip(),
            speaker,
            config,
            tts_runtime,
        )
        return True
    return False


def _reset_vad_segment_state(state: dict) -> None:
    state["pre_roll"] = []
    state["speech_frames"] = []
    state["speech_started"] = False
    state["trigger_window"] = []
    state["endpoint_window"] = []


def ms_to_frame_count(duration_ms: int, frame_ms: int, minimum: int = 1) -> int:
    if frame_ms <= 0:
        return max(minimum, 1)
    return max(minimum, int((max(0, int(duration_ms)) + frame_ms - 1) // frame_ms))


def _finalize_vad_segment(np_mod, state: dict):
    speech_frames = list(state.get("speech_frames", []))
    if not speech_frames:
        _reset_vad_segment_state(state)
        return None, "empty"
    frame_ms = int(state.get("frame_ms", 30))
    min_segment_ms = int(state.get("min_segment_ms", 1400))
    if len(speech_frames) * frame_ms < min_segment_ms:
        _reset_vad_segment_state(state)
        return None, f"segment_too_short:{len(speech_frames) * frame_ms}ms"
    audio_i16 = np_mod.frombuffer(b"".join(speech_frames), dtype=np_mod.int16)
    audio = audio_i16.astype(np_mod.float32) / 32768.0
    _reset_vad_segment_state(state)
    return audio.reshape(-1), ""


def prepare_auto_listen_runtime(auto_listen_runtime: dict, config: dict) -> bool:
    stt_runtime = auto_listen_runtime["stt_runtime"]
    np_mod = stt_runtime.get("numpy") or ensure_dependency("numpy", "numpy", "NumPy")
    if np_mod is None:
        return False
    stt_runtime["numpy"] = np_mod

    sd_mod = stt_runtime.get("sounddevice") or ensure_dependency("sounddevice", "sounddevice", "SoundDevice")
    if sd_mod is None:
        return False
    stt_runtime["sounddevice"] = sd_mod

    silero_mod = auto_listen_runtime.get("silero_vad") or ensure_dependency("silero_vad", "silero-vad", "Silero VAD")
    if silero_mod is None:
        return False
    auto_listen_runtime["silero_vad"] = silero_mod

    torch_mod = stt_runtime.get("torch") or ensure_dependency("torch", "torch", "PyTorch")
    if torch_mod is None:
        return False
    stt_runtime["torch"] = torch_mod

    model = auto_listen_runtime.get("silero_model")
    if model is None:
        load_model = getattr(silero_mod, "load_silero_vad", None)
        if load_model is None:
            print_error_red("ERROR: Silero VAD package does not expose load_silero_vad().")
            return False
        try:
            model = load_model()
        except TypeError:
            model = load_model(onnx=True)
        except Exception as exc:
            print_error_red(f"ERROR: Failed to initialize Silero VAD: {exc}")
            return False
        auto_listen_runtime["silero_model"] = model

    vad_iterator_cls = auto_listen_runtime.get("silero_vad_iterator_cls")
    if vad_iterator_cls is None:
        vad_iterator_cls = getattr(silero_mod, "VADIterator", None)
        if vad_iterator_cls is None:
            print_error_red("ERROR: Silero VAD package does not expose VADIterator.")
            return False
        auto_listen_runtime["silero_vad_iterator_cls"] = vad_iterator_cls

    with contextlib.suppress(Exception):
        model.reset_states()

    try:
        input_device = None
        with contextlib.suppress(Exception):
            input_device = sd_mod.query_devices(kind="input")
        selected_device = resolve_audio_input_device(config, sd_mod)
        sd_mod.check_input_settings(device=selected_device, samplerate=16000, channels=1, dtype="int16")
        if isinstance(input_device, dict):
            name = str(input_device.get("name", "(default input)"))
            if selected_device is None:
                print(f"\n🎙️ Auto-listen input device: {name}\n")
        if selected_device is not None:
            selected_name = str(config.get("audio_input_device", "")).strip() or str(selected_device)
            print(f"\n🎙️ Auto-listen input device: {selected_name}\n")
        return True
    except Exception as exc:
        print_error_red(f"ERROR: Auto-listen microphone is unavailable: {exc}")
        if IS_WINDOWS:
            print("Open Windows Settings > Privacy & security > Microphone and allow desktop apps to access the microphone.\n")
        else:
            print("Check the default input device and OS microphone permissions.\n")
        return False


def stop_audio_monitor(auto_listen_runtime: dict) -> None:
    camera_runtime = auto_listen_runtime.get("camera_runtime")
    if isinstance(camera_runtime, dict):
        return
    stop_event = auto_listen_runtime.get("audio_monitor_stop_event")
    thread = auto_listen_runtime.get("audio_monitor_thread")
    if stop_event is not None:
        stop_event.set()
    if thread is not None and thread.is_alive():
        thread.join(timeout=2)
    auto_listen_runtime["audio_monitor_thread"] = None
    auto_listen_runtime["audio_monitor_stop_event"] = None


def _audio_monitor_worker(auto_listen_runtime: dict) -> None:
    try:
        cv2_mod = auto_listen_runtime.get("cv2")
        if cv2_mod is None:
            cv2_mod = ensure_dependency("cv2", "opencv-python", "OpenCV")
            if cv2_mod is None:
                return
            auto_listen_runtime["cv2"] = cv2_mod
        np_mod = auto_listen_runtime["stt_runtime"].get("numpy") or ensure_dependency("numpy", "numpy", "NumPy")
        if np_mod is None:
            return
        stop_event = auto_listen_runtime.get("audio_monitor_stop_event")
        if stop_event is None:
            return
        window_name = "Robot Audio Monitor"
        while not stop_event.is_set():
            speech = bool(auto_listen_runtime.get("last_is_speech", False))
            speech_prob = float(auto_listen_runtime.get("last_speech_probability", 0.0))
            speech_prob_threshold = float(auto_listen_runtime.get("last_speech_probability_threshold", 0.5))
            speech_started = bool(auto_listen_runtime.get("last_speech_started", False))
            speech_frames = int(auto_listen_runtime.get("last_speech_frames", 0))
            recording = bool(auto_listen_runtime.get("last_recording", False))
            start_event = float(auto_listen_runtime.get("last_start_event", 0.0))
            end_event = float(auto_listen_runtime.get("last_end_event", 0.0))
            metrics = [
                {
                    "label": "Speech Prob",
                    "value": min(1.0, max(0.0, speech_prob)),
                    "threshold": min(1.0, max(0.0, speech_prob_threshold)),
                    "active": speech,
                },
                {
                    "label": "Start Event",
                    "value": 1.0 if start_event > 0.0 else 0.0,
                    "threshold": 1.0,
                    "active": start_event > 0.0,
                },
                {
                    "label": "End Event",
                    "value": 1.0 if end_event > 0.0 else 0.0,
                    "threshold": 1.0,
                    "active": end_event > 0.0,
                },
                {
                    "label": "Segment Open",
                    "value": 1.0 if speech_started else 0.0,
                    "threshold": 1.0,
                    "active": speech_started,
                },
                {
                    "label": "Segment Length",
                    "value": min(1.0, speech_frames / max(1, int(auto_listen_runtime.get("last_display_segment_frames", 1)))),
                    "threshold": 1.0,
                    "active": speech_started,
                },
                {
                    "label": "Recording State",
                    "value": 1.0 if recording else 0.0,
                    "threshold": 1.0,
                    "active": recording,
                },
            ]
            frame = build_audio_monitor_frame(np_mod, 0.0, 1.0, speech, metrics=metrics)
            cv2_mod.putText(frame, "Silero VAD Monitor", (24, 30), cv2_mod.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2_mod.LINE_AA)
            for idx, metric in enumerate(metrics):
                text_y = 74 + idx * 42
                state_text = "PASS" if metric["active"] else "WAIT"
                value_text = f"{metric['value']:.2f}/{metric['threshold']:.2f}"
                cv2_mod.putText(frame, f"{metric['label']}: {value_text} [{state_text}]", (24, text_y), cv2_mod.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2_mod.LINE_AA)
            cv2_mod.putText(frame, f"speech>{speech_prob_threshold:.2f}", (24, 236), cv2_mod.FONT_HERSHEY_SIMPLEX, 0.5, (170, 170, 170), 1, cv2_mod.LINE_AA)
            cv2_mod.imshow(window_name, frame)
            key = cv2_mod.waitKey(50) & 0xFF
            if key in (27, ord("q")):
                stop_event.set()
                break
        with contextlib.suppress(Exception):
            cv2_mod.destroyWindow(window_name)
    except Exception as exc:
        print_error_red(f"ERROR: Audio monitor failed: {exc}")


def start_audio_monitor(auto_listen_runtime: dict) -> bool:
    camera_runtime = auto_listen_runtime.get("camera_runtime")
    if isinstance(camera_runtime, dict):
        return True
    stop_audio_monitor(auto_listen_runtime)
    stop_event = threading.Event()
    auto_listen_runtime["audio_monitor_stop_event"] = stop_event
    thread = threading.Thread(target=_audio_monitor_worker, args=(auto_listen_runtime,), daemon=True)
    auto_listen_runtime["audio_monitor_thread"] = thread
    thread.start()
    return True


def _auto_listen_worker(auto_listen_runtime: dict) -> None:
    stop_event = auto_listen_runtime.get("stop_event")
    if stop_event is None:
        return
    config = auto_listen_runtime["voice_config"]
    stt_runtime = auto_listen_runtime["stt_runtime"]
    tts_runtime = auto_listen_runtime["tts_runtime"]
    llm_state = auto_listen_runtime["llm_state"]
    stats = auto_listen_runtime["stats"]
    speaker = auto_listen_runtime.get("speaker")

    np_mod = stt_runtime.get("numpy")
    if np_mod is None:
        print_error_red("ERROR: Auto-listen NumPy runtime is not initialized.")
        return
    stt_runtime["numpy"] = np_mod
    sd_mod = stt_runtime.get("sounddevice")
    if sd_mod is None:
        print_error_red("ERROR: Auto-listen sounddevice runtime is not initialized.")
        return
    stt_runtime["sounddevice"] = sd_mod
    silero_model = auto_listen_runtime.get("silero_model")
    if silero_model is None:
        print_error_red("ERROR: Auto-listen Silero VAD runtime is not initialized.")
        return
    auto_listen_runtime["silero_model"] = silero_model
    silero_mod = auto_listen_runtime.get("silero_vad")
    if silero_mod is None:
        print_error_red("ERROR: Auto-listen Silero VAD package is not initialized.")
        return
    vad_iterator_cls = auto_listen_runtime.get("silero_vad_iterator_cls") or getattr(silero_mod, "VADIterator", None)
    if vad_iterator_cls is None:
        print_error_red("ERROR: Auto-listen Silero VAD iterator is not available.")
        return
    auto_listen_runtime["silero_vad_iterator_cls"] = vad_iterator_cls
    torch_mod = stt_runtime.get("torch")
    if torch_mod is None:
        print_error_red("ERROR: Auto-listen PyTorch runtime is not initialized.")
        return

    sample_rate = 16000
    frame_ms = int(config.get("auto_listen_frame_ms", 32))
    if frame_ms not in {32, 64, 96}:
        frame_ms = 32
    frame_samples = {32: 512, 64: 1024, 96: 1536}.get(frame_ms, 512)
    max_segment_frames = max(1, int(float(config.get("auto_listen_max_segment_s", 15.0)) * 1000 / frame_ms))
    speech_threshold = float(config.get("auto_listen_threshold", 0.50))
    audio_queue: queue.Queue = queue.Queue(maxsize=256)
    state = {
        "frame_ms": frame_ms,
        "min_segment_ms": 0,
    }
    _reset_vad_segment_state(state)
    auto_listen_runtime["last_speech_probability_threshold"] = speech_threshold
    auto_listen_runtime["last_display_segment_frames"] = max_segment_frames
    auto_listen_runtime["last_start_event"] = 0.0
    auto_listen_runtime["last_end_event"] = 0.0
    auto_listen_runtime["last_speech_probability"] = 0.0
    vad_iterator = vad_iterator_cls(
        silero_model,
        threshold=speech_threshold,
        sampling_rate=sample_rate,
        min_silence_duration_ms=int(config.get("auto_listen_silence_ms", 1600)),
        speech_pad_ms=int(config.get("auto_listen_preroll_ms", 350)),
    )
    with contextlib.suppress(Exception):
        silero_model.reset_states()
    with contextlib.suppress(Exception):
        vad_iterator.reset_states()

    def callback(indata, _frames, _time, status):
        if status:
            return
        try:
            audio_queue.put_nowait(indata.copy().tobytes())
        except queue.Full:
            pass

    print("\n✅ Auto-listen Silero VAD active.\n")
    try:
        selected_device = resolve_audio_input_device(config, sd_mod)
        with sd_mod.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            blocksize=frame_samples,
            device=selected_device,
            callback=callback,
        ):
            while not stop_event.is_set():
                if not should_run_auto_listen_worker(config):
                    time.sleep(0.05)
                    continue
                if is_tts_blocking_auto_listen(config):
                    with contextlib.suppress(queue.Empty):
                        while True:
                            audio_queue.get_nowait()
                    _reset_vad_segment_state(state)
                    auto_listen_runtime["last_recording"] = False
                    auto_listen_runtime["last_start_event"] = 0.0
                    auto_listen_runtime["last_end_event"] = 0.0
                    auto_listen_runtime["last_speech_probability"] = 0.0
                    with contextlib.suppress(Exception):
                        silero_model.reset_states()
                    with contextlib.suppress(Exception):
                        vad_iterator.reset_states()
                    time.sleep(0.05)
                    continue
                try:
                    frame_bytes = audio_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                auto_listen_runtime["last_start_event"] = 0.0
                auto_listen_runtime["last_end_event"] = 0.0
                try:
                    audio_i16 = np_mod.frombuffer(frame_bytes, dtype=np_mod.int16)
                    audio = audio_i16.astype(np_mod.float32) / 32768.0
                    audio_tensor = torch_mod.from_numpy(audio.copy())
                    speech_prob = float(silero_model(audio_tensor, sample_rate).item())
                    try:
                        speech_dict = vad_iterator(audio_tensor, return_seconds=False)
                    except TypeError:
                        speech_dict = vad_iterator(audio_tensor)
                except Exception:
                    speech_prob = 0.0
                    speech_dict = {}
                is_speech = speech_prob >= speech_threshold
                auto_listen_runtime["last_is_speech"] = bool(is_speech)
                auto_listen_runtime["last_speech_probability"] = float(speech_prob)
                auto_listen_runtime["last_speech_started"] = bool(state.get("speech_started", False))
                auto_listen_runtime["last_speech_frames"] = len(state.get("speech_frames", []))
                auto_listen_runtime["last_recording"] = bool(state.get("speech_started", False))
                now_ts = time.monotonic()
                if isinstance(speech_dict, dict) and "start" in speech_dict:
                    state["speech_started"] = True
                    state["speech_frames"] = []
                    auto_listen_runtime["last_start_event"] = 1.0
                if state["speech_started"]:
                    state["speech_frames"].append(frame_bytes)
                if isinstance(speech_dict, dict) and "end" in speech_dict:
                    auto_listen_runtime["last_end_event"] = 1.0
                if should_emit_auto_listen_log(auto_listen_runtime, now_ts):
                    debug_text = format_vad_debug_output(
                        {
                            "speech": is_speech,
                            "speech_prob": round(speech_prob, 3),
                            "speech_threshold": speech_threshold,
                            "speech_started": bool(state.get("speech_started", False)),
                            "start_event": bool(auto_listen_runtime.get("last_start_event", 0.0)),
                            "end_event": bool(auto_listen_runtime.get("last_end_event", 0.0)),
                            "speech_frames": len(state.get("speech_frames", [])),
                            "queue_size": int(audio_queue.qsize()),
                        }
                    )
                    print(f"\n[vad-log] {debug_text}\n")

                auto_listen_runtime["last_speech_started"] = bool(state.get("speech_started", False))
                auto_listen_runtime["last_speech_frames"] = len(state.get("speech_frames", []))
                auto_listen_runtime["last_recording"] = bool(state.get("speech_started", False))

                if not state["speech_started"]:
                    continue

                if len(state["speech_frames"]) >= max_segment_frames:
                    audio, discard_reason = _finalize_vad_segment(np_mod, state)
                    with contextlib.suppress(Exception):
                        silero_model.reset_states()
                    with contextlib.suppress(Exception):
                        vad_iterator.reset_states()
                elif bool(auto_listen_runtime.get("last_end_event", 0.0)):
                    audio, discard_reason = _finalize_vad_segment(np_mod, state)
                    with contextlib.suppress(Exception):
                        silero_model.reset_states()
                    with contextlib.suppress(Exception):
                        vad_iterator.reset_states()
                else:
                    audio, discard_reason = None, ""
                if discard_reason and bool(config.get("vision_log_enabled", False)):
                    print(f"\n[vad-log] segment discarded: {discard_reason}\n")
                if audio is None:
                    continue
                auto_listen_runtime["last_recording"] = False
                auto_listen_runtime["last_speech_started"] = False
                auto_listen_runtime["last_speech_frames"] = 0
                if bool(config.get("vision_log_enabled", False)):
                    print(f"\n[vad-log] segment accepted: {len(audio) / sample_rate:0.2f}s\n")
                text, _stt_time = transcribe_audio_buffer(stt_runtime, config, audio, speech_end_ts=time.perf_counter())
                if text and handle_wake_word_transcript(text, auto_listen_runtime, speaker, config, tts_runtime):
                    continue
                if text and not is_valid_auto_listen_transcript(text):
                    if bool(config.get("vision_log_enabled", False)):
                        print(f"\n[vad-log] transcript ignored: {text!r}\n")
                    continue
                if text:
                    process_auto_listen_text(text, llm_state, stats, speaker, config, stt_runtime, tts_runtime)
    except Exception as exc:
        print_error_red(f"ERROR: Auto-listen stopped: {exc}")
        if IS_WINDOWS:
            print("Check Windows microphone privacy settings and the default recording device.\n")


def stop_auto_listen(auto_listen_runtime: dict) -> None:
    stop_event = auto_listen_runtime.get("stop_event")
    thread = auto_listen_runtime.get("thread")
    if stop_event is not None:
        stop_event.set()
    if thread is not None and thread.is_alive():
        thread.join(timeout=2)
    stop_audio_monitor(auto_listen_runtime)
    auto_listen_runtime["thread"] = None
    auto_listen_runtime["stop_event"] = None


def start_auto_listen(auto_listen_runtime: dict, config: dict, activate_session: bool = True) -> bool:
    stop_auto_listen(auto_listen_runtime)
    if not prepare_auto_listen_runtime(auto_listen_runtime, config):
        return False
    config["auto_listen_enabled"] = bool(activate_session)
    auto_listen_runtime["voice_config"] = config
    if bool(config.get("audio_monitor_enabled", False)):
        start_audio_monitor(auto_listen_runtime)
    stop_event = threading.Event()
    auto_listen_runtime["stop_event"] = stop_event
    thread = threading.Thread(target=_auto_listen_worker, args=(auto_listen_runtime,), daemon=True)
    auto_listen_runtime["thread"] = thread
    thread.start()
    return True


def refresh_auto_listen_worker(auto_listen_runtime: dict, config: dict) -> bool:
    if should_run_auto_listen_worker(config):
        thread = auto_listen_runtime.get("thread")
        if thread is not None and thread.is_alive():
            return True
        return start_auto_listen(
            auto_listen_runtime,
            config,
            activate_session=bool(config.get("auto_listen_enabled", False)),
        )
    stop_auto_listen(auto_listen_runtime)
    return True


def split_tts_segment(buffer: str, min_words: int, cut_on_punctuation: bool) -> tuple[str, str]:
    text = str(buffer or "")
    if not text.strip():
        return "", ""
    words = list(re.finditer(r"\S+", text))
    min_words = max(1, int(min_words))
    cut_idx_words = words[min_words - 1].end() if len(words) >= min_words else None
    cut_idx_punct = None
    if cut_on_punctuation:
        punct_match = re.search(r"[.!?,;:](?:\s|$)", text)
        if punct_match is not None:
            cut_idx_punct = punct_match.end()

    if cut_on_punctuation:
        if cut_idx_words is None and cut_idx_punct is None:
            return "", text
        if cut_idx_words is None:
            cut_idx = cut_idx_punct
        elif cut_idx_punct is None:
            cut_idx = cut_idx_words
        else:
            cut_idx = min(cut_idx_words, cut_idx_punct)
    else:
        if cut_idx_words is None:
            return "", text
        cut_idx = cut_idx_words

    if cut_idx is None:
        return "", text
    segment = text[:cut_idx].strip()
    rest = text[cut_idx:].lstrip()
    return segment, rest


def wait_for_listen_action() -> str:
    """
    Returns:
      - "start" when SPACE is pressed
      - "exit" when ESC is pressed
    """
    print("Press SPACE to start recording, or ESC to exit listen mode.")
    with KEYBOARD.capture():
        clear_keyboard_buffer()
        while True:
            ch = KEYBOARD.read_key_nonblocking()
            if ch is None:
                time.sleep(0.01)
                continue
            if ch == " ":
                return "start"
            if ch == "\x1b":
                return "exit"


def run_listen_mode(
    pipe,
    current,
    history: list[str],
    stats: dict,
    speaker,
    voice_config: dict,
    stt_runtime: dict,
    tts_runtime: dict,
) -> None:
    print("\n🎙️ Listen mode active.")
    while True:
        action = wait_for_listen_action()
        if action == "exit":
            print("\n⌨️ Back to text mode.\n")
            return

        user_input, stt_time = transcribe_from_mic(stt_runtime, voice_config)
        if not user_input:
            continue
        cycle_timings: dict = {"speech_end_to_text_s": stt_time}

        if bool(voice_config.get("repeat", False)):
            if bool(voice_config.get("audio_enabled", True)):
                try:
                    interrupted, calc_latency_s = speak_text_backend(
                        speaker,
                        user_input,
                        voice_config,
                        tts_runtime,
                        allow_interrupt=True,
                    )
                    cycle_timings["text_to_audio_s"] = calc_latency_s
                    if interrupted:
                        print("⏹️ Audio interrupted by ESC.")
                except Exception as exc:
                    print(f"\n⚠️ TTS failed: {exc}\n")
            else:
                cycle_timings["text_to_audio_s"] = None
            print("⏱️ Cycle timings:")
            if cycle_timings.get("text_to_audio_s", "missing") is None:
                print("  text -> audio: skipped (audio off)")
            elif "text_to_audio_s" in cycle_timings:
                print(f"  text -> audio: {cycle_timings['text_to_audio_s']:0.3f}s")
            print(f"  speech_end -> text: {cycle_timings['speech_end_to_text_s']:0.3f}s")
            continue

        ok = run_chat_turn(
            user_input,
            pipe,
            current,
            history,
            stats,
            speaker,
            voice_config,
            tts_runtime,
            timings=cycle_timings,
            show_roundtrip_lines=False,
        )
        if ok:
            print("⏱️ Cycle timings:")
            if "text_to_llm_response_s" in cycle_timings:
                print(f"  text -> llm response: {cycle_timings['text_to_llm_response_s']:0.3f}s")
            if cycle_timings.get("text_to_audio_s", "missing") is None:
                print("  text -> audio: skipped (audio off)")
            elif "text_to_audio_s" in cycle_timings:
                print(f"  text -> audio: {cycle_timings['text_to_audio_s']:0.3f}s")
            print(f"  speech_end -> text: {cycle_timings['speech_end_to_text_s']:0.3f}s")


def show_voice_config(config: dict, speaker) -> None:
    print("\nCurrent voice config:")
    if speaker is not None:
        print(f"  voice   : {speaker.Voice.GetDescription()}")
    else:
        print("  voice   : (windows engine unavailable)")
    print(f"  rate    : {config.get('rate', -2)}")
    print(f"  volume  : {config.get('volume', 100)}")
    print(f"  silence : {config.get('silence', 600)}")
    print(f"  tts_backend: {config.get('tts_backend', DEFAULT_TTS_BACKEND)}")
    print(f"  openvino_tts_device: {config.get('openvino_tts_device', 'AUTO')}")
    print(f"  openvino_tts_model_id: {config.get('openvino_tts_model_id', '') or '(none)'}")
    print(f"  openvino_tts_timeout_s: {int(config.get('openvino_tts_timeout_s', 25))}")
    print(f"  openvino_tts_isolated_gpu: {bool(config.get('openvino_tts_isolated_gpu', True))}")
    print(f"  openvino_tts_speed: {float(config.get('openvino_tts_speed', 1.0)):.2f}")
    print(f"  openvino_tts_gain: {float(config.get('openvino_tts_gain', 1.0)):.2f}")
    print(f"  kokoro_model_id: {config.get('kokoro_model_id', 'kokoro-tts-intel')}")
    print(f"  kokoro_device: {config.get('kokoro_device', 'GPU')}")
    print(f"  kokoro_voice: {config.get('kokoro_voice', 'af_sarah')}")
    print(f"  babelvox_model_id: {config.get('babelvox_model_id', 'babelvox-openvino-int8')}")
    print(f"  babelvox_device: {config.get('babelvox_device', 'CPU')}")
    print(f"  babelvox_precision: {config.get('babelvox_precision', 'int8')}")
    print(f"  babelvox_language: {config.get('babelvox_language', 'es')}")
    print(f"  tada_model_id: {config.get('tada_model_id', 'HumeAI/tada-1b')}")
    print(f"  tada_codec_id: {config.get('tada_codec_id', 'HumeAI/tada-codec')}")
    print(f"  tada_device: {config.get('tada_device', 'cpu')}")
    print(f"  tada_language: {config.get('tada_language', 'en')}")
    print(f"  tada_reference_audio_path: {config.get('tada_reference_audio_path', '') or '(empty)'}")
    print(f"  tada_reference_text: {'(set)' if str(config.get('tada_reference_text', '')).strip() else '(empty)'}")
    print(f"  tada_sample_rate: {int(config.get('tada_sample_rate', 24000))}")
    print(f"  espeak_voice: {config.get('espeak_voice', 'es')}")
    print(f"  espeak_rate: {int(config.get('espeak_rate', 145))}")
    print(f"  espeak_pitch: {int(config.get('espeak_pitch', 45))}")
    print(f"  espeak_amplitude: {int(config.get('espeak_amplitude', 120))}")
    print(f"  whisper : {config.get('whisper_model', DEFAULT_WHISPER_MODEL)}")
    print(f"  whisper_language: {config.get('whisper_language', 'es')}")
    print(f"  whisper_openvino: {bool(config.get('whisper_openvino', False))}")
    print(f"  whisper_ov_device: {config.get('whisper_openvino_device', 'AUTO')}")
    print(f"  whisper_ov_model_id: {config.get('whisper_openvino_model_id', '') or '(none)'}")
    print(f"  repeat : {bool(config.get('repeat', False))}")
    print(f"  llm_backend: {config.get('llm_backend', 'local')}")
    print(f"  external_llm_base_url: {config.get('external_llm_base_url', 'http://localhost:1234')}")
    print(f"  external_llm_model: {config.get('external_llm_model', '') or '(empty)'}")
    print(f"  external_llm_api_key: {'(set)' if str(config.get('external_llm_api_key', '')).strip() else '(empty)'}")
    print(f"  audio   : {'on' if bool(config.get('audio_enabled', True)) else 'off'}")
    print(f"  audio_input_device: {config.get('audio_input_device', '') or '(default)'}")
    print(f"  audio_monitor_enabled: {bool(config.get('audio_monitor_enabled', False))}")
    print(f"  visual_effects_enabled: {bool(config.get('visual_effects_enabled', True))}")
    print(f"  panel_backend: {config.get('panel_backend', 'opencv')}")
    print(f"  camera_enabled: {bool(config.get('camera_enabled', False))}")
    print(f"  camera_device_index: {int(config.get('camera_device_index', 0))}")
    print(f"  vision_enabled: {bool(config.get('vision_enabled', False))}")
    print(f"  vision_model_id: {config.get('vision_model_id', '') or '(empty)'}")
    print(f"  vision_model_path: {config.get('vision_model_path', '') or '(empty)'}")
    print(f"  vision_labels_path: {config.get('vision_labels_path', '') or '(empty)'}")
    print(f"  vision_device: {config.get('vision_device', 'AUTO')}")
    print(f"  vision_threshold: {float(config.get('vision_threshold', 0.4)):.2f}")
    print(f"  vision_log_enabled: {bool(config.get('vision_log_enabled', False))}")
    print(f"  vision_log_interval_s: {float(config.get('vision_log_interval_s', 1.0)):.1f}")
    print(f"  vision_event_processing_enabled: {bool(config.get('vision_event_processing_enabled', True))}")
    print(f"  auto_listen_enabled: {bool(config.get('auto_listen_enabled', False))}")
    print(f"  wake_word_enabled: {bool(config.get('wake_word_enabled', False))}")
    print(f"  wake_word_phrase: {config.get('wake_word_phrase', 'hola robot')}")
    print(f"  wake_word_stop_phrase: {config.get('wake_word_stop_phrase', 'adios robot')}")
    print(f"  wake_word_on_response: {config.get('wake_word_on_response', 'Te escucho.')}")
    print(f"  wake_word_off_response: {config.get('wake_word_off_response', 'Modo escucha desactivado.')}")
    print(f"  auto_listen_threshold: {float(config.get('auto_listen_threshold', 0.50)):.2f}")
    print(f"  auto_listen_frame_ms: {int(config.get('auto_listen_frame_ms', 32))}")
    print(f"  auto_listen_resume_delay_ms: {int(config.get('auto_listen_resume_delay_ms', 1500))}")
    print(f"  tts_streaming_enabled: {bool(config.get('tts_streaming_enabled', False))}")
    print(f"  tts_stream_min_words: {int(config.get('tts_stream_min_words', 12))}")
    print(f"  tts_stream_cut_on_punctuation: {bool(config.get('tts_stream_cut_on_punctuation', False))}")
    print(f"  warmup_tts: {bool(config.get('warmup_tts', True))}")
    print(f"  max_tokens: {int(config.get('max_new_tokens', 300))}")
    print("  system prompt:")
    current_system = str(config.get("system_prompt", ""))
    if current_system:
        print("  ---")
        print(current_system)
        print("  ---")
    else:
        print("  (empty)")
    print("")


def configure_voice_and_stt(
    config: dict,
    speaker,
    voices,
    stt_runtime: dict,
    tts_runtime: dict,
    llm_state: dict | None = None,
) -> None:
    print("\nConfig options:")
    print(
        "  rate, volume, silence, tts_backend, "
        "openvino_tts_device, openvino_tts_model, openvino_tts_models, openvino_tts_timeout, openvino_tts_isolated_gpu, "
        "openvino_tts_speed, openvino_tts_gain, "
        "kokoro_model, kokoro_models, kokoro_device, kokoro_voice, "
        "babelvox_model, babelvox_models, babelvox_device, babelvox_precision, babelvox_language, "
        "tada_model, tada_codec, tada_device, tada_language, tada_reference_audio, tada_reference_text, tada_reference_record, "
        "espeak_voices, espeak_voice, espeak_rate, espeak_pitch, espeak_amplitude, "
        "whisper, whisper_language, whisper_backend, whisper_ov_device, whisper_ov_model, whisper_ov_models, "
        "llm_backend, external_llm_base_url, external_llm_model, external_llm_api_key, "
        "audio_inputs, audio_input, audio_monitor, repeat, log, log_interval, tts_streaming, tts_stream_min_words, tts_stream_punctuation, warmup_tts, max_tokens, system, show, exit"
    )
    show_voice_config(config, speaker)

    def reload_llm_from_config() -> None:
        if llm_state is None:
            return
        active_pipe = llm_state.get("pipe")
        release_llm_pipe(active_pipe)
        try:
            new_pipe, new_current, new_history = activate_llm_from_config(
                config,
                compat=llm_state.get("compat"),
            )
            llm_state["pipe"] = new_pipe
            llm_state["current"] = new_current
            llm_state["history"] = new_history
            server_state = llm_state.get("server_state")
            if isinstance(server_state, dict):
                server_state["pipe"] = new_pipe
                server_state["current"] = new_current
            if new_current is not None:
                print(f"LLM reloaded: {model_menu_label(new_current)}")
            else:
                print("LLM backend updated, but no model is active.")
        except Exception as exc:
            llm_state["pipe"] = None
            llm_state["current"] = None
            llm_state["history"] = []
            server_state = llm_state.get("server_state")
            if isinstance(server_state, dict):
                server_state["pipe"] = None
                server_state["current"] = None
            print_error_red(f"ERROR: Failed to reload LLM backend: {exc}")

    while True:
        key = input("Config key: ").strip().lower()
        if key == "exit":
            save_robot_config(config)
            return
        if key == "show":
            show_voice_config(config, speaker)
            continue
        if key == "tts_backend":
            value = input("Set TTS backend (windows|openvino|kokoro|babelvox|espeakng|tada): ").strip().lower()
            if value not in TTS_BACKEND_OPTIONS:
                print(f"Invalid value. Use one of: {', '.join(TTS_BACKEND_OPTIONS)}")
                continue
            config["tts_backend"] = value
            save_robot_config(config)
            tts_runtime["active_key"] = None
            if value in {"openvino", "kokoro", "babelvox", "espeakng", "tada"}:
                ensure_tts_runtime(tts_runtime, config)
            print(f"tts_backend updated to {value}")
            continue
        if key == "openvino_tts_models":
            models = load_ov_tts_models()
            list_ov_tts_models(models, str(config.get("openvino_tts_model_id", "")).strip())
            continue
        if key == "openvino_tts_model":
            models = load_ov_tts_models()
            selected_id = str(config.get("openvino_tts_model_id", "")).strip()
            selected = choose_ov_tts_model_interactive(models, selected_id=selected_id, allow_download=False)
            if selected is None:
                print("Cancelled.")
                continue
            config["openvino_tts_model_id"] = selected["id"]
            save_robot_config(config)
            tts_runtime["active_key"] = None
            if str(config.get("tts_backend", DEFAULT_TTS_BACKEND)).lower() == "openvino":
                ensure_tts_runtime(tts_runtime, config)
            print(f"openvino_tts_model updated to {selected['id']}")
            continue
        if key == "openvino_tts_device":
            print("\nOpenVINO TTS devices:\n")
            current_device = str(config.get("openvino_tts_device", "AUTO")).upper()
            for i, option in enumerate(PARLER_OV_DEVICE_OPTIONS):
                marker = " (current)" if option == current_device else ""
                print(f"  {i}) {option}{marker}")
            value = input("\nChoose device number or 'cancel': ").strip().lower()
            if value == "cancel":
                continue
            if not value.isdigit() or not (0 <= int(value) < len(PARLER_OV_DEVICE_OPTIONS)):
                print("Invalid option.")
                continue
            config["openvino_tts_device"] = PARLER_OV_DEVICE_OPTIONS[int(value)]
            save_robot_config(config)
            tts_runtime["active_key"] = None
            if str(config.get("tts_backend", DEFAULT_TTS_BACKEND)).lower() == "openvino":
                ensure_tts_runtime(tts_runtime, config)
            print(f"openvino_tts_device updated to {config['openvino_tts_device']}")
            continue
        if key == "openvino_tts_timeout":
            value = input("Set OpenVINO TTS timeout in seconds (3..180): ").strip()
            try:
                ivalue = int(value)
            except ValueError:
                print("Please enter an integer.")
                continue
            if not (3 <= ivalue <= 180):
                print("openvino_tts_timeout must be between 3 and 180 seconds.")
                continue
            config["openvino_tts_timeout_s"] = ivalue
            save_robot_config(config)
            print(f"openvino_tts_timeout updated to {ivalue}s")
            continue
        if key == "openvino_tts_isolated_gpu":
            value = input("Set openvino_tts_isolated_gpu (true/false): ").strip().lower()
            if value in {"true", "1", "on", "si", "sí", "yes", "y"}:
                config["openvino_tts_isolated_gpu"] = True
                save_robot_config(config)
                print("openvino_tts_isolated_gpu updated to True")
                continue
            if value in {"false", "0", "off", "no", "n"}:
                config["openvino_tts_isolated_gpu"] = False
                save_robot_config(config)
                print("openvino_tts_isolated_gpu updated to False")
                continue
            print("Invalid value. Use true or false.")
            continue
        if key == "openvino_tts_speed":
            value = input("Set openvino_tts_speed (0.5..2.0, 1.0 default): ").strip()
            try:
                fvalue = float(value)
            except ValueError:
                print("Please enter a number.")
                continue
            if not (0.5 <= fvalue <= 2.0):
                print("openvino_tts_speed must be between 0.5 and 2.0.")
                continue
            config["openvino_tts_speed"] = fvalue
            save_robot_config(config)
            print(f"openvino_tts_speed updated to {fvalue:.2f}")
            continue
        if key == "openvino_tts_gain":
            value = input("Set openvino_tts_gain (0.1..3.0, 1.0 default): ").strip()
            try:
                fvalue = float(value)
            except ValueError:
                print("Please enter a number.")
                continue
            if not (0.1 <= fvalue <= 3.0):
                print("openvino_tts_gain must be between 0.1 and 3.0.")
                continue
            config["openvino_tts_gain"] = fvalue
            save_robot_config(config)
            print(f"openvino_tts_gain updated to {fvalue:.2f}")
            continue
        if key == "kokoro_models":
            models = load_kokoro_models()
            _list_repo_models("Kokoro models:", models, str(config.get("kokoro_model_id", "")).strip())
            continue
        if key == "kokoro_model":
            models = load_kokoro_models()
            selected = _choose_repo_model_interactive("Kokoro models:", models, str(config.get("kokoro_model_id", "")).strip())
            if selected is None:
                continue
            config["kokoro_model_id"] = selected["id"]
            save_robot_config(config)
            tts_runtime["active_key"] = None
            if str(config.get("tts_backend", DEFAULT_TTS_BACKEND)).lower() == "kokoro":
                ensure_tts_runtime(tts_runtime, config)
            print(f"kokoro_model updated to {selected['id']}")
            continue
        if key == "kokoro_device":
            print("\nKokoro devices:\n")
            for i, option in enumerate(KOKORO_DEVICE_OPTIONS):
                marker = " (current)" if option == str(config.get("kokoro_device", "GPU")).upper() else ""
                print(f"  {i}) {option}{marker}")
            value = input("\nChoose device number or 'cancel': ").strip().lower()
            if value == "cancel":
                continue
            if not value.isdigit() or not (0 <= int(value) < len(KOKORO_DEVICE_OPTIONS)):
                print("Invalid option.")
                continue
            config["kokoro_device"] = KOKORO_DEVICE_OPTIONS[int(value)]
            save_robot_config(config)
            tts_runtime["active_key"] = None
            if str(config.get("tts_backend", DEFAULT_TTS_BACKEND)).lower() == "kokoro":
                ensure_tts_runtime(tts_runtime, config)
            print(f"kokoro_device updated to {config['kokoro_device']}")
            continue
        if key == "kokoro_voice":
            value = input("Set Kokoro voice (e.g. af_sarah): ").strip()
            if not value:
                print("Voice cannot be empty.")
                continue
            config["kokoro_voice"] = value
            save_robot_config(config)
            print(f"kokoro_voice updated to {value}")
            continue
        if key == "babelvox_models":
            models = load_babelvox_models()
            _list_repo_models("BabelVox models:", models, str(config.get("babelvox_model_id", "")).strip())
            continue
        if key == "babelvox_model":
            models = load_babelvox_models()
            selected = _choose_repo_model_interactive("BabelVox models:", models, str(config.get("babelvox_model_id", "")).strip())
            if selected is None:
                continue
            config["babelvox_model_id"] = selected["id"]
            save_robot_config(config)
            tts_runtime["active_key"] = None
            if str(config.get("tts_backend", DEFAULT_TTS_BACKEND)).lower() == "babelvox":
                ensure_tts_runtime(tts_runtime, config)
            print(f"babelvox_model updated to {selected['id']}")
            continue
        if key == "babelvox_device":
            print("\nBabelVox devices:\n")
            for i, option in enumerate(BABELVOX_DEVICE_OPTIONS):
                marker = " (current)" if option == str(config.get("babelvox_device", "CPU")).upper() else ""
                print(f"  {i}) {option}{marker}")
            value = input("\nChoose device number or 'cancel': ").strip().lower()
            if value == "cancel":
                continue
            if not value.isdigit() or not (0 <= int(value) < len(BABELVOX_DEVICE_OPTIONS)):
                print("Invalid option.")
                continue
            config["babelvox_device"] = BABELVOX_DEVICE_OPTIONS[int(value)]
            save_robot_config(config)
            tts_runtime["active_key"] = None
            if str(config.get("tts_backend", DEFAULT_TTS_BACKEND)).lower() == "babelvox":
                ensure_tts_runtime(tts_runtime, config)
            print(f"babelvox_device updated to {config['babelvox_device']}")
            continue
        if key == "babelvox_precision":
            print("\nBabelVox precision options:\n")
            for i, option in enumerate(BABELVOX_PRECISION_OPTIONS):
                marker = " (current)" if option == str(config.get("babelvox_precision", "int8")).lower() else ""
                print(f"  {i}) {option}{marker}")
            value = input("\nChoose precision number or 'cancel': ").strip().lower()
            if value == "cancel":
                continue
            if not value.isdigit() or not (0 <= int(value) < len(BABELVOX_PRECISION_OPTIONS)):
                print("Invalid option.")
                continue
            config["babelvox_precision"] = BABELVOX_PRECISION_OPTIONS[int(value)]
            save_robot_config(config)
            tts_runtime["active_key"] = None
            if str(config.get("tts_backend", DEFAULT_TTS_BACKEND)).lower() == "babelvox":
                ensure_tts_runtime(tts_runtime, config)
            print(f"babelvox_precision updated to {config['babelvox_precision']}")
            continue
        if key == "babelvox_language":
            print("\nBabelVox language options:\n")
            langs = ["auto", "es", "en", "pt", "fr", "it", "de"]
            current_lang = str(config.get("babelvox_language", "es")).lower()
            for i, option in enumerate(langs):
                marker = " (current)" if option == current_lang else ""
                print(f"  {i}) {option}{marker}")
            value = input("\nChoose language number or code: ").strip().lower()
            if value == "cancel":
                continue
            if value.isdigit() and 0 <= int(value) < len(langs):
                lang = langs[int(value)]
            else:
                lang = value
            if lang not in langs:
                print(f"Invalid language. Use one of: {', '.join(langs)}")
                continue
            config["babelvox_language"] = lang
            save_robot_config(config)
            print(f"babelvox_language updated to {lang}")
            continue
        if key == "tada_model":
            value = input("Set Hume TADA model id (e.g. HumeAI/tada-1b): ").strip()
            if not value:
                print("tada_model cannot be empty.")
                continue
            config["tada_model_id"] = value
            save_robot_config(config)
            tts_runtime["active_key"] = None
            if str(config.get("tts_backend", DEFAULT_TTS_BACKEND)).lower() == "tada":
                ensure_tts_runtime(tts_runtime, config)
            print(f"tada_model updated to {value}")
            continue
        if key == "tada_codec":
            value = input("Set Hume TADA codec id (e.g. HumeAI/tada-codec): ").strip()
            if not value:
                print("tada_codec cannot be empty.")
                continue
            config["tada_codec_id"] = value
            save_robot_config(config)
            tts_runtime["active_key"] = None
            if str(config.get("tts_backend", DEFAULT_TTS_BACKEND)).lower() == "tada":
                ensure_tts_runtime(tts_runtime, config)
            print(f"tada_codec updated to {value}")
            continue
        if key == "tada_device":
            value = input("Set Hume TADA device string (cpu, cuda, ...): ").strip()
            if not value:
                print("tada_device cannot be empty.")
                continue
            config["tada_device"] = value
            save_robot_config(config)
            tts_runtime["active_key"] = None
            if str(config.get("tts_backend", DEFAULT_TTS_BACKEND)).lower() == "tada":
                ensure_tts_runtime(tts_runtime, config)
            print(f"tada_device updated to {value}")
            continue
        if key == "tada_language":
            value = input("Set Hume TADA encoder language code (e.g. en, es): ").strip().lower()
            if not value:
                print("tada_language cannot be empty.")
                continue
            config["tada_language"] = value
            save_robot_config(config)
            tts_runtime["active_key"] = None
            if str(config.get("tts_backend", DEFAULT_TTS_BACKEND)).lower() == "tada":
                ensure_tts_runtime(tts_runtime, config)
            print(f"tada_language updated to {value}")
            continue
        if key == "tada_reference_audio":
            value = input("Set Hume TADA reference audio path: ").strip()
            if not value:
                print("tada_reference_audio cannot be empty.")
                continue
            config["tada_reference_audio_path"] = value
            save_robot_config(config)
            tts_runtime["active_key"] = None
            print(f"tada_reference_audio updated to {value}")
            continue
        if key == "tada_reference_text":
            value = input("Set Hume TADA reference transcript: ").strip()
            if not value:
                print("tada_reference_text cannot be empty.")
                continue
            config["tada_reference_text"] = value
            save_robot_config(config)
            tts_runtime["active_key"] = None
            print("tada_reference_text updated.")
            continue
        if key == "tada_reference_record":
            capture_tada_reference_from_mic(stt_runtime, config)
            tts_runtime["active_key"] = None
            continue
        if key == "espeak_voices":
            list_espeak_voices()
            continue
        if key == "espeak_voice":
            value = input("Set eSpeak voice (e.g. es, es-la, en-us, mb-es1): ").strip()
            if not value:
                print("espeak_voice cannot be empty.")
                continue
            config["espeak_voice"] = value
            save_robot_config(config)
            print(f"espeak_voice updated to {value}")
            continue
        if key == "espeak_rate":
            value = input("Set eSpeak rate (80..450, default 145): ").strip()
            try:
                ivalue = int(value)
            except ValueError:
                print("Please enter an integer.")
                continue
            if not (80 <= ivalue <= 450):
                print("espeak_rate must be between 80 and 450.")
                continue
            config["espeak_rate"] = ivalue
            save_robot_config(config)
            print(f"espeak_rate updated to {ivalue}")
            continue
        if key == "espeak_pitch":
            value = input("Set eSpeak pitch (0..99, default 45): ").strip()
            try:
                ivalue = int(value)
            except ValueError:
                print("Please enter an integer.")
                continue
            if not (0 <= ivalue <= 99):
                print("espeak_pitch must be between 0 and 99.")
                continue
            config["espeak_pitch"] = ivalue
            save_robot_config(config)
            print(f"espeak_pitch updated to {ivalue}")
            continue
        if key == "espeak_amplitude":
            value = input("Set eSpeak amplitude (0..200, default 120): ").strip()
            try:
                ivalue = int(value)
            except ValueError:
                print("Please enter an integer.")
                continue
            if not (0 <= ivalue <= 200):
                print("espeak_amplitude must be between 0 and 200.")
                continue
            config["espeak_amplitude"] = ivalue
            save_robot_config(config)
            print(f"espeak_amplitude updated to {ivalue}")
            continue
        if key == "whisper":
            print("\nWhisper models:\n")
            whisper_mod = stt_runtime.get("whisper")
            if whisper_mod is None:
                whisper_mod = ensure_dependency("whisper", "openai-whisper", "Whisper")
                if whisper_mod is not None:
                    stt_runtime["whisper"] = whisper_mod
            for i, model in enumerate(WHISPER_MODELS):
                downloaded, size_text = (
                    whisper_model_size_info(whisper_mod, model) if whisper_mod is not None else (False, "unknown")
                )
                icon = "✅" if downloaded else "⬇️"
                marker = " (current)" if config.get("whisper_model", DEFAULT_WHISPER_MODEL) == model else ""
                print(f"  {i}) {icon} {model} [{size_text}]{marker}")
            value = input("\nChoose model number or 'cancel': ").strip().lower()
            if value == "cancel":
                continue
            if not value.isdigit() or not (0 <= int(value) < len(WHISPER_MODELS)):
                print("Invalid model option.")
                continue
            new_model = WHISPER_MODELS[int(value)]
            previous = config.get("whisper_model", DEFAULT_WHISPER_MODEL)
            config["whisper_model"] = new_model
            print(f"Whisper model changed to: {new_model}")
            if not ensure_stt_runtime(stt_runtime, config):
                config["whisper_model"] = previous
                print(f"Restored previous model: {previous}")
            else:
                save_robot_config(config)
            continue
        if key == "whisper_language":
            print("\nWhisper language options:\n")
            current_lang = str(config.get("whisper_language", "es")).strip().lower()
            for i, option in enumerate(WHISPER_LANGUAGE_OPTIONS):
                marker = " (current)" if option == current_lang else ""
                print(f"  {i}) {option}{marker}")
            value = input("\nChoose language number or type code (e.g. es, en, auto): ").strip().lower()
            if value == "cancel":
                continue
            if value.isdigit() and 0 <= int(value) < len(WHISPER_LANGUAGE_OPTIONS):
                lang = WHISPER_LANGUAGE_OPTIONS[int(value)]
            else:
                lang = value
            if lang not in WHISPER_LANGUAGE_OPTIONS:
                print(f"Invalid language. Use one of: {', '.join(WHISPER_LANGUAGE_OPTIONS)}")
                continue
            config["whisper_language"] = lang
            save_robot_config(config)
            print(f"whisper_language updated to: {lang}")
            continue
        if key == "whisper_backend":
            value = input("Set Whisper backend (openvino|whisper): ").strip().lower()
            if value in {"openvino", "ov"}:
                config["whisper_openvino"] = True
                save_robot_config(config)
                stt_runtime["active_key"] = None
                print("Whisper backend updated to OpenVINO.")
                ensure_stt_runtime(stt_runtime, config)
                continue
            if value in {"whisper", "default", "cpu"}:
                config["whisper_openvino"] = False
                save_robot_config(config)
                stt_runtime["active_key"] = None
                print("Whisper backend updated to openai-whisper.")
                ensure_stt_runtime(stt_runtime, config)
                continue
            print("Invalid value. Use openvino or whisper.")
            continue
        if key == "whisper_ov_models":
            models = load_whisper_ov_models()
            list_whisper_ov_models(models, str(config.get("whisper_openvino_model_id", "")).strip())
            continue
        if key == "whisper_ov_model":
            models = load_whisper_ov_models()
            selected_id = str(config.get("whisper_openvino_model_id", "")).strip()
            selected = choose_whisper_ov_model_interactive(models, selected_id=selected_id, allow_download=False)
            if selected is None:
                print("Cancelled.")
                continue
            config["whisper_openvino_model_id"] = selected["id"]
            save_robot_config(config)
            stt_runtime["active_key"] = None
            print(f"Whisper OV model selected: {selected['id']}")
            if bool(config.get("whisper_openvino", False)):
                ensure_stt_runtime(stt_runtime, config)
            continue
        if key == "whisper_ov_device":
            print("\nWhisper OpenVINO devices:\n")
            for i, option in enumerate(WHISPER_OV_DEVICE_OPTIONS):
                marker = " (current)" if option == str(config.get("whisper_openvino_device", "AUTO")).upper() else ""
                print(f"  {i}) {option}{marker}")
            value = input("\nChoose device number or 'cancel': ").strip().lower()
            if value == "cancel":
                continue
            if not value.isdigit() or not (0 <= int(value) < len(WHISPER_OV_DEVICE_OPTIONS)):
                print("Invalid option.")
                continue
            config["whisper_openvino_device"] = WHISPER_OV_DEVICE_OPTIONS[int(value)]
            save_robot_config(config)
            stt_runtime["active_key"] = None
            print(f"Whisper OpenVINO device updated to: {config['whisper_openvino_device']}")
            if bool(config.get("whisper_openvino", False)):
                ensure_stt_runtime(stt_runtime, config)
            continue
        if key == "llm_backend":
            value = input("Set LLM backend (local|external): ").strip().lower()
            if value not in {"local", "external"}:
                print("Invalid value. Use local or external.")
                continue
            config["llm_backend"] = value
            save_robot_config(config)
            print(f"llm_backend updated to {value}")
            reload_llm_from_config()
            continue
        if key == "external_llm_base_url":
            value = input("Set external LLM base URL (e.g. http://localhost:1234): ").strip()
            if not value:
                print("external_llm_base_url cannot be empty.")
                continue
            config["external_llm_base_url"] = value
            save_robot_config(config)
            print(f"external_llm_base_url updated to {value}")
            if str(config.get("llm_backend", "local")).lower() == "external":
                reload_llm_from_config()
            continue
        if key == "audio_inputs":
            sd_mod = stt_runtime.get("sounddevice") or ensure_dependency("sounddevice", "sounddevice", "SoundDevice")
            if sd_mod is None:
                continue
            stt_runtime["sounddevice"] = sd_mod
            devices = list_audio_input_devices(sd_mod)
            if not devices:
                print("No audio input devices were detected.")
                continue
            print("")
            for idx, name, channels in devices:
                marker = ""
                current_audio = str(config.get("audio_input_device", "")).strip()
                if current_audio and (current_audio == str(idx) or current_audio == name):
                    marker = " (current)"
                print(f"  - {name} [index={idx}, channels={channels}]{marker}")
            print("")
            continue
        if key == "audio_input":
            sd_mod = stt_runtime.get("sounddevice") or ensure_dependency("sounddevice", "sounddevice", "SoundDevice")
            if sd_mod is None:
                continue
            stt_runtime["sounddevice"] = sd_mod
            selected = choose_audio_input_device_interactive(sd_mod, str(config.get("audio_input_device", "")).strip())
            if selected is None:
                print("Cancelled.")
                continue
            config["audio_input_device"] = selected[0]
            save_robot_config(config)
            print(f"audio_input_device updated to: {selected[1]} [index={selected[0]}]")
            continue
        if key == "audio_monitor":
            value = input("Set audio_monitor (true/false): ").strip().lower()
            if value in {"true", "1", "on", "si", "sí", "yes", "y"}:
                config["audio_monitor_enabled"] = True
                save_robot_config(config)
                print("audio_monitor_enabled updated to True")
                continue
            if value in {"false", "0", "off", "no", "n"}:
                config["audio_monitor_enabled"] = False
                save_robot_config(config)
                print("audio_monitor_enabled updated to False")
                continue
            print("Invalid value. Use true or false.")
            continue
        if key == "external_llm_model":
            value = input("Set external LLM model name: ").strip()
            if not value:
                print("external_llm_model cannot be empty.")
                continue
            config["external_llm_model"] = value
            save_robot_config(config)
            print(f"external_llm_model updated to {value}")
            if str(config.get("llm_backend", "local")).lower() == "external":
                reload_llm_from_config()
            continue
        if key == "external_llm_api_key":
            value = input("Set external LLM API key (empty to clear): ").strip()
            config["external_llm_api_key"] = value
            save_robot_config(config)
            print("external_llm_api_key updated.")
            if str(config.get("llm_backend", "local")).lower() == "external":
                reload_llm_from_config()
            continue
        if key == "repeat":
            value = input("Set repeat (true/false): ").strip().lower()
            if value in {"true", "1", "on", "si", "sí", "yes", "y"}:
                config["repeat"] = True
                save_robot_config(config)
                print("repeat updated to True")
                continue
            if value in {"false", "0", "off", "no", "n"}:
                config["repeat"] = False
                save_robot_config(config)
                print("repeat updated to False")
                continue
            print("Invalid value. Use true or false.")
            continue
        if key == "tts_streaming":
            value = input("Set tts_streaming (true/false): ").strip().lower()
            if value in {"true", "1", "on", "si", "sí", "yes", "y"}:
                config["tts_streaming_enabled"] = True
                save_robot_config(config)
                print("tts_streaming_enabled updated to True")
                continue
            if value in {"false", "0", "off", "no", "n"}:
                config["tts_streaming_enabled"] = False
                save_robot_config(config)
                print("tts_streaming_enabled updated to False")
                continue
            print("Invalid value. Use true or false.")
            continue
        if key == "tts_stream_min_words":
            value = input("Set tts_stream_min_words (>=1): ").strip()
            try:
                ivalue = int(value)
            except ValueError:
                print("Please enter an integer.")
                continue
            if ivalue < 1:
                print("tts_stream_min_words must be >= 1.")
                continue
            config["tts_stream_min_words"] = ivalue
            save_robot_config(config)
            print(f"tts_stream_min_words updated to {ivalue}")
            continue
        if key == "tts_stream_punctuation":
            value = input("Set tts_stream_punctuation (true/false): ").strip().lower()
            if value in {"true", "1", "on", "si", "sí", "yes", "y"}:
                config["tts_stream_cut_on_punctuation"] = True
                save_robot_config(config)
                print("tts_stream_cut_on_punctuation updated to True")
                continue
            if value in {"false", "0", "off", "no", "n"}:
                config["tts_stream_cut_on_punctuation"] = False
                save_robot_config(config)
                print("tts_stream_cut_on_punctuation updated to False")
                continue
            print("Invalid value. Use true or false.")
            continue
        if key == "warmup_tts":
            value = input("Set warmup_tts (true/false): ").strip().lower()
            if value in {"true", "1", "on", "si", "sí", "yes", "y"}:
                config["warmup_tts"] = True
                config["_tts_warmup_done"] = False
                save_robot_config(config)
                print("warmup_tts updated to True")
                continue
            if value in {"false", "0", "off", "no", "n"}:
                config["warmup_tts"] = False
                config["_tts_warmup_done"] = False
                save_robot_config(config)
                print("warmup_tts updated to False")
                continue
            print("Invalid value. Use true or false.")
            continue
        if key == "system":
            print("\nCurrent system prompt:")
            current_system = str(config.get("system_prompt", ""))
            if current_system:
                print("---")
                print(current_system)
                print("---")
            else:
                print("(empty)")
            print("\nEnter new system prompt. Finish with a line containing only /end.")
            print("If you want to clear it, type only /end on the first line.\n")

            lines: list[str] = []
            while True:
                line = input()
                if line.strip() == "/end":
                    break
                lines.append(line)
            config["system_prompt"] = "\n".join(lines).strip()
            save_robot_config(config)
            print("system prompt updated.\n")
            continue
        if key == "max_tokens":
            value = input("Set max_tokens (1..8192): ").strip()
            try:
                ivalue = int(value)
            except ValueError:
                print("Please enter an integer.")
                continue
            if not (1 <= ivalue <= 8192):
                print("max_tokens must be between 1 and 8192.")
                continue
            config["max_new_tokens"] = ivalue
            save_robot_config(config)
            print(f"max_tokens updated to {ivalue}")
            continue
        if key not in {"rate", "volume", "silence"}:
            print("Invalid config key.")
            continue
        value = input(f"New value for {key}: ").strip()
        try:
            ivalue = int(value)
        except ValueError:
            print("Please enter an integer.")
            continue
        if key == "volume" and not (0 <= ivalue <= 100):
            print("Volume must be between 0 and 100.")
            continue
        if key == "silence" and ivalue < 0:
            print("Silence cannot be negative.")
            continue
        config[key] = ivalue
        if speaker is not None and voices is not None:
            apply_voice_config(speaker, voices, config)
        elif key in {"rate", "volume"}:
            print("⚠️ Native voice engine unavailable. Value saved, but not applied live.")
        print(f"{key} updated to {ivalue}")


def run_chat_turn(
    user_text: str,
    pipe,
    current,
    history: list[str],
    stats: dict,
    speaker,
    voice_config: dict,
    tts_runtime: dict,
    timings: dict | None = None,
    show_roundtrip_lines: bool = True,
) -> bool:
    if pipe is None or current is None:
        print("⚠️ No model loaded. Use '/models' to load one (or '/help').\n")
        return False

    history.append(f"User: {user_text}")
    system_prompt = str(voice_config.get("system_prompt", "")).strip()
    max_new_tokens = int(voice_config.get("max_new_tokens", 300))
    # Approximation for Spanish: 1 token ~= 0.65 words
    max_words_estimate = max(1, int(max_new_tokens * 0.65))
    system_limit_note = (
        f"Anexo: No respondas con mas de {max_words_estimate} palabras."
    )
    if system_prompt:
        effective_system_prompt = f"{system_prompt}\n{system_limit_note}"
        prompt = f"System: {effective_system_prompt}\n" + "\n".join(history) + "\nAssistant:"
    else:
        prompt = f"System: {system_limit_note}\n" + "\n".join(history) + "\nAssistant:"

    t_start = time.perf_counter()
    first_token_time = None
    token_events = 0
    chunks: list[str] = []
    generation_interrupted = False
    esc_stop_event = threading.Event()
    stream_tts_enabled = bool(voice_config.get("tts_streaming_enabled", False))
    stream_min_words = max(1, int(voice_config.get("tts_stream_min_words", 12)))
    stream_cut_on_punctuation = bool(voice_config.get("tts_stream_cut_on_punctuation", False))
    audio_enabled = bool(voice_config.get("audio_enabled", True))
    response_audio_activity_started = False
    if audio_enabled:
        set_tts_activity(True)
        response_audio_activity_started = True
    try:
        stream_audio_active = stream_tts_enabled and audio_enabled
        pending_tts_text = ""
        tts_audio_queue: queue.Queue | None = queue.Queue() if stream_audio_active else None
        tts_audio_thread: threading.Thread | None = None
        tts_audio_stop_event = threading.Event()
        tts_calc_latencies_s: list[float] = []

        def esc_monitor():
            with KEYBOARD.capture():
                while not esc_stop_event.is_set():
                    if is_audio_cancel_requested() or KEYBOARD.read_key_nonblocking() == "\x1b":
                        esc_stop_event.set()
                        tts_audio_stop_event.set()
                        return
                    time.sleep(0.01)

        def tts_worker():
            if tts_audio_queue is None:
                return
            while not tts_audio_stop_event.is_set():
                try:
                    item = tts_audio_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                if item is None:
                    return
                text_piece = str(item).strip()
                if not text_piece:
                    continue
                try:
                    interrupted, calc_latency_s = speak_text_backend(
                        speaker,
                        text_piece,
                        voice_config,
                        tts_runtime,
                        allow_interrupt=True,
                    )
                    tts_calc_latencies_s.append(float(calc_latency_s))
                    if interrupted:
                        tts_audio_stop_event.set()
                        esc_stop_event.set()
                except Exception as exc:
                    print(f"\n⚠️ TTS failed: {exc}\n")
                    tts_audio_stop_event.set()
                    esc_stop_event.set()
                    return

        if stream_audio_active:
            tts_audio_thread = threading.Thread(target=tts_worker, daemon=True)
            tts_audio_thread.start()

        def streamer(chunk: str):
            nonlocal first_token_time, token_events, pending_tts_text
            if esc_stop_event.is_set():
                raise RuntimeError("__ESC_ABORT__")
            now = time.perf_counter()
            if first_token_time is None:
                first_token_time = now
            token_events += 1
            chunks.append(chunk)
            print(chunk, end="", flush=True)
            if stream_audio_active and tts_audio_queue is not None:
                pending_tts_text += chunk
                while True:
                    segment, rest = split_tts_segment(
                        pending_tts_text,
                        min_words=stream_min_words,
                        cut_on_punctuation=stream_cut_on_punctuation,
                    )
                    if not segment:
                        break
                    pending_tts_text = rest
                    if not tts_audio_stop_event.is_set() and not is_audio_cancel_requested():
                        tts_audio_queue.put(segment)

        print("🤖 > ", end="", flush=True)
        esc_thread = threading.Thread(target=esc_monitor, daemon=True)
        esc_thread.start()
        try:
            pipe.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                streamer=streamer,
            )
        except RuntimeError as exc:
            if "__ESC_ABORT__" not in str(exc):
                raise
            generation_interrupted = True
        finally:
            esc_stop_event.set()
            esc_thread.join(timeout=0.1)
        if stream_audio_active and tts_audio_queue is not None:
            if not generation_interrupted and not is_audio_cancel_requested():
                remaining = pending_tts_text.strip()
                if remaining:
                    tts_audio_queue.put(remaining)
            tts_audio_queue.put(None)
            if tts_audio_thread is not None:
                if generation_interrupted:
                    tts_audio_thread.join(timeout=0.2)
                else:
                    tts_audio_thread.join()
        t_end = time.perf_counter()
        print("\n")

        if generation_interrupted:
            print("⏹️ Generation interrupted by ESC.")
            if timings is not None:
                timings["text_to_llm_response_s"] = t_end - t_start
                timings["text_to_audio_s"] = None
            return False

        if first_token_time is None:
            ttft = t_end - t_start
            tps = 0.0
        else:
            ttft = first_token_time - t_start
            decode_time = max(1e-9, t_end - first_token_time)
            tps = token_events / decode_time

        stats_name = model_menu_label(current)
        record_stats(
            stats,
            current["repo"],
            stats_name,
            ACTIVE_DEVICE,
            ttft,
            tps,
            mode=STATS_MODE_NORMAL,
        )
        save_stats(stats)
        print(f"📈 TTFT: {ttft:0.3f}s | TPS≈ {tps:0.2f} | events: {token_events}")
        llm_roundtrip_s = t_end - t_start
        if timings is not None:
            timings["text_to_llm_response_s"] = llm_roundtrip_s
        if show_roundtrip_lines:
            print(f"⏱️ text -> llm response: {llm_roundtrip_s:0.3f}s")

        answer = "".join(chunks).strip()
        if answer and audio_enabled and stream_audio_active:
            audio_s = sum(tts_calc_latencies_s) if tts_calc_latencies_s else 0.0
            if timings is not None:
                timings["text_to_audio_s"] = audio_s
            if show_roundtrip_lines:
                print(f"⏱️ text -> audio: {audio_s:0.3f}s")
        elif answer and audio_enabled:
            try:
                interrupted, calc_latency_s = speak_text_backend(
                    speaker,
                    answer,
                    voice_config,
                    tts_runtime,
                    allow_interrupt=True,
                )
                audio_s = calc_latency_s
                if timings is not None:
                    timings["text_to_audio_s"] = audio_s
                if show_roundtrip_lines:
                    print(f"⏱️ text -> audio: {audio_s:0.3f}s")
                if interrupted:
                    print("⏹️ Audio interrupted by ESC.")
            except Exception as exc:
                print(f"\n⚠️ TTS failed: {exc}\n")
        elif answer:
            if timings is not None:
                timings["text_to_audio_s"] = None
            if show_roundtrip_lines:
                print("⏱️ text -> audio: skipped (audio off)")

        history.append(f"Assistant: {answer}")
        return True
    finally:
        if response_audio_activity_started:
            set_tts_activity(False)


def print_startup_summary(current, voice_config: dict) -> None:
    llm_name = "(none)"
    if current is not None:
        llm_name = current.get("display") or current.get("repo", "(unknown)")
    else:
        repo = str(voice_config.get("current_model_repo", "")).strip()
        if repo:
            llm_name = f"(pending load) {repo}"

    rows = [
        ("LLM", llm_name),
        ("Saved LLM repo", str(voice_config.get("current_model_repo", "")) or "(empty)"),
        ("LLM backend", str(voice_config.get("llm_backend", "local"))),
        ("External LLM URL", str(voice_config.get("external_llm_base_url", "http://localhost:1234"))),
        ("External LLM model", str(voice_config.get("external_llm_model", "")) or "(empty)"),
        ("Device", ACTIVE_DEVICE),
        ("Perf Hint", ACTIVE_PERFORMANCE_HINT),
        ("Voice index", str(int(voice_config.get("voice_index", 0)))),
        ("Whisper", str(voice_config.get("whisper_model", DEFAULT_WHISPER_MODEL))),
        ("Whisper language", str(voice_config.get("whisper_language", "es"))),
        ("Whisper backend", "openvino" if bool(voice_config.get("whisper_openvino", False)) else "whisper"),
        ("Whisper OV device", str(voice_config.get("whisper_openvino_device", "AUTO"))),
        ("Whisper OV model", str(voice_config.get("whisper_openvino_model_id", "")) or "(none)"),
        ("TTS backend", str(voice_config.get("tts_backend", DEFAULT_TTS_BACKEND))),
        ("OpenVINO TTS device", str(voice_config.get("openvino_tts_device", "AUTO"))),
        ("OpenVINO TTS model", str(voice_config.get("openvino_tts_model_id", "")) or "(none)"),
        ("OpenVINO TTS timeout", f"{int(voice_config.get('openvino_tts_timeout_s', 25))}s"),
        ("OpenVINO TTS isolated GPU", str(bool(voice_config.get("openvino_tts_isolated_gpu", True)))),
        ("OpenVINO TTS speed", f"{float(voice_config.get('openvino_tts_speed', 1.0)):.2f}"),
        ("OpenVINO TTS gain", f"{float(voice_config.get('openvino_tts_gain', 1.0)):.2f}"),
        ("Kokoro model", str(voice_config.get("kokoro_model_id", "kokoro-tts-intel"))),
        ("Kokoro device", str(voice_config.get("kokoro_device", "GPU"))),
        ("Kokoro voice", str(voice_config.get("kokoro_voice", "af_sarah"))),
        ("BabelVox model", str(voice_config.get("babelvox_model_id", "babelvox-openvino-int8"))),
        ("BabelVox device", str(voice_config.get("babelvox_device", "CPU"))),
        ("BabelVox precision", str(voice_config.get("babelvox_precision", "int8"))),
        ("BabelVox language", str(voice_config.get("babelvox_language", "es"))),
        ("TADA model", str(voice_config.get("tada_model_id", "HumeAI/tada-1b"))),
        ("TADA codec", str(voice_config.get("tada_codec_id", "HumeAI/tada-codec"))),
        ("TADA device", str(voice_config.get("tada_device", "cpu"))),
        ("TADA language", str(voice_config.get("tada_language", "en"))),
        ("TADA ref audio", str(voice_config.get("tada_reference_audio_path", "")) or "(none)"),
        ("eSpeak voice", str(voice_config.get("espeak_voice", "es"))),
        ("eSpeak rate", str(int(voice_config.get("espeak_rate", 145)))),
        ("eSpeak pitch", str(int(voice_config.get("espeak_pitch", 45)))),
        ("eSpeak amplitude", str(int(voice_config.get("espeak_amplitude", 120)))),
        ("Max tokens", str(int(voice_config.get("max_new_tokens", 300)))),
        ("Rate", str(voice_config.get("rate", -2))),
        ("Volume", str(voice_config.get("volume", 100))),
        ("Silence", str(voice_config.get("silence", 600))),
        ("Repeat Mode", str(bool(voice_config.get("repeat", False)))),
        ("Audio", "on" if bool(voice_config.get("audio_enabled", True)) else "off"),
        ("Audio Input", str(voice_config.get("audio_input_device", "")) or "(default)"),
        ("Audio Monitor", str(bool(voice_config.get("audio_monitor_enabled", False)))),
        ("Visual Effects", str(bool(voice_config.get("visual_effects_enabled", True)))),
        ("Panel backend", str(voice_config.get("panel_backend", "opencv"))),
        ("Camera", "on" if bool(voice_config.get("camera_enabled", False)) else "off"),
        ("Camera Device", str(int(voice_config.get("camera_device_index", 0)))),
        ("Vision", "on" if bool(voice_config.get("vision_enabled", False)) else "off"),
        ("Vision Model ID", str(voice_config.get("vision_model_id", "")) or "(empty)"),
        ("Vision Device", str(voice_config.get("vision_device", "AUTO"))),
        ("Vision Threshold", f"{float(voice_config.get('vision_threshold', 0.4)):.2f}"),
        ("Vision Model", str(voice_config.get("vision_model_path", "")) or "(empty)"),
        ("Vision Log", str(bool(voice_config.get("vision_log_enabled", False)))),
        ("Vision Log Interval", f"{float(voice_config.get('vision_log_interval_s', 1.0)):.1f}s"),
        ("Vision Events", str(bool(voice_config.get("vision_event_processing_enabled", True)))),
        ("Auto Listen", str(bool(voice_config.get("auto_listen_enabled", False)))),
        ("Wake Word", str(bool(voice_config.get("wake_word_enabled", False)))),
        ("VAD Threshold", f"{float(voice_config.get('auto_listen_threshold', 0.50)):.2f}"),
        ("VAD Frame", f"{int(voice_config.get('auto_listen_frame_ms', 32))}ms"),
        ("VAD Resume Delay", f"{int(voice_config.get('auto_listen_resume_delay_ms', 1500))}ms"),
        ("TTS streaming", str(bool(voice_config.get("tts_streaming_enabled", False)))),
        ("TTS stream min words", str(int(voice_config.get("tts_stream_min_words", 12)))),
        ("TTS stream punctuation", str(bool(voice_config.get("tts_stream_cut_on_punctuation", False)))),
        ("Warmup TTS", str(bool(voice_config.get("warmup_tts", True)))),
        ("System prompt", str(voice_config.get("system_prompt", "")).replace("\n", " ") or "(empty)"),
    ]
    k_w = max(len(k) for k, _ in rows)
    v_w = min(72, max(len(v) for _, v in rows))
    line = f"+-{'-' * k_w}-+-{'-' * v_w}-+"
    print("\n" + line)
    for key, value in rows:
        shown = value if len(value) <= v_w else value[: v_w - 1] + "…"
        print(f"| {key:<{k_w}} | {shown:<{v_w}} |")
    print(line + "\n")


def print_startup_logo() -> None:
    colors = [
        "\033[91m",  # red
        "\033[92m",  # green
        "\033[93m",  # yellow
        "\033[94m",  # blue
        "\033[96m",  # cyan
    ]
    reset = "\033[0m"
    logo = [
        "      [::]      ",
        "    .-====-.    ",
        "   /  .--.  \\   ",
        "  |  | oo |  |  ",
        "  |  | __ |  |  ",
        "   \\  ----  /   ",
        "    '-.__.-'    ",
        "    _|    |_    ",
        "   /_|_/\\_|_\\   ",
    ]
    color = random.choice(colors)
    print("")
    for line in logo:
        print(f"{color}{line}{reset}")
    print("")


def print_all_models_summary(compat: dict) -> None:
    rows: list[tuple[str, str, str]] = []

    for m in MODELS:
        if is_downloaded(m["local"]):
            rows.append(("LLM", m["display"], m["repo"]))

    for m in load_whisper_ov_models():
        if is_downloaded(m["local"]):
            rows.append(("STT Whisper OV", m["display"], f"stt:whisper_ov:{m['id']}"))

    for model_name in WHISPER_MODELS:
        if is_whisper_classic_downloaded(model_name):
            rows.append(("STT Whisper", model_name, f"stt:whisper:{model_name}"))

    for m in load_ov_tts_models():
        if is_downloaded(m["local"]):
            rows.append(("TTS OpenVINO", m["display"], f"tts:openvino:{m['id']}"))

    for m in load_kokoro_models():
        if is_repo_downloaded(m["local"]):
            rows.append(("TTS Kokoro", m["display"], f"tts:kokoro:{m['id']}"))

    for m in load_babelvox_models():
        if is_repo_downloaded(m["local"]):
            rows.append(("TTS BabelVox", m["display"], f"tts:babelvox:{m['id']}"))

    if not rows:
        print("\nNo downloaded models found.\n")
        return

    rows.sort(key=lambda x: (x[0], x[1].lower()))
    family_w = max(6, max(len(r[0]) for r in rows))
    model_w = max(10, min(64, max(len(r[1]) for r in rows)))
    line = f"+-{'-' * family_w}-+-{'-' * model_w}-+-----+-----+-----+"
    print("\nDownloaded Models By Chip:\n")
    print(line)
    print(f"| {'Family':<{family_w}} | {'Model':<{model_w}} | CPU | GPU | NPU |")
    print(line)
    for family, name, key in rows:
        cpu, gpu, npu = chip_marks_for_key(compat, key)
        shown = name if len(name) <= model_w else name[: model_w - 1] + "…"
        print(f"| {family:<{family_w}} | {shown:<{model_w}} | {cpu:^3} | {gpu:^3} | {npu:^3} |")
    print(line + "\n")


# =========================
# Main loop
# =========================
def main() -> None:
    global MODELS

    ensure_auth_file()
    MODELS = load_models()
    stats = load_stats()
    compat = load_device_compat()

    pipe = None
    current = None
    history = []
    server = None
    server_state = {"pipe": None, "current": None, "config": None}
    voice_config = load_robot_config()
    apply_runtime_from_config(voice_config)
    server_state["config"] = voice_config
    stt_runtime = {
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
    tts_runtime = {
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
    speaker = None
    voices = None
    camera_runtime = {
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
    llm_state = {
        "pipe": pipe,
        "current": current,
        "history": history,
        "server_state": server_state,
        "compat": compat,
    }
    auto_listen_runtime = {
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
    camera_runtime["auto_listen_runtime"] = auto_listen_runtime
    auto_listen_runtime["camera_runtime"] = camera_runtime

    speaker, voices, voice_error = initialize_native_voice_engine(voice_config)
    camera_runtime["speaker"] = speaker
    auto_listen_runtime["speaker"] = speaker
    if speaker is not None and voices is not None:
        save_robot_config(voice_config)
    else:
        print(f"\n⚠️ Voice engine unavailable: {voice_error}")
        if IS_WINDOWS:
            print("   The chat will continue (Windows TTS features may be unavailable).\n")
        else:
            print("   The chat will continue. Use eSpeak NG or another non-Windows TTS backend.\n")

    try:
        print("Loading Whisper STT backend...")
        if ensure_stt_runtime(stt_runtime, voice_config):
            backend_name = "openvino" if bool(voice_config.get("whisper_openvino", False)) else "whisper"
            print(f"✅ Whisper STT ready: {backend_name}\n")
        else:
            print("\n⚠️ Whisper STT could not be preloaded. It will be retried on first use.\n")
    except Exception as exc:
        print(f"\n⚠️ Whisper STT preload failed: {exc}")
        print("   It will be retried on first use.\n")

    try:
        pipe, current, history = activate_llm_from_config(voice_config, compat=compat)
        server_state["pipe"] = pipe
        server_state["current"] = current
        llm_state["pipe"] = pipe
        llm_state["current"] = current
        llm_state["history"] = history
        if current is not None:
            print(f"✅ Auto-loaded LLM on startup: {model_menu_label(current)}")
    except Exception as exc:
        pipe = None
        current = None
        history = []
        server_state["pipe"] = None
        server_state["current"] = None
        llm_state["pipe"] = None
        llm_state["current"] = None
        llm_state["history"] = history
        print(f"⚠️ Could not auto-load configured LLM on startup: {exc}")

    voice_config["camera_enabled"] = False
    voice_config["auto_listen_enabled"] = False
    camera_runtime["camera_enabled"] = False
    save_robot_config(voice_config)
    print_startup_logo()
    print_startup_summary(current, voice_config)
    print("Ready. Type '/help' to list commands.\n")
    APP_EXIT_REQUESTED.clear()

    while True:
        if APP_EXIT_REQUESTED.is_set():
            break
        try:
            user_input = input("🧑 > ").strip()
        except (EOFError, KeyboardInterrupt):
            if APP_EXIT_REQUESTED.is_set():
                print("")
                break
            print("")
            continue

        if not user_input:
            continue

        if is_command(user_input):
            cmd = normalize_command(user_input)

            if cmd == "help":
                print("\n" + HELP_TEXT)
                continue

            if cmd == "context":
                show_llm_context(history, voice_config)
                continue

            if cmd.startswith("panel"):
                parts = cmd.split(maxsplit=1)
                selected_backend = str(voice_config.get("panel_backend", "opencv")).strip().lower()
                if len(parts) == 2:
                    selected_backend = parts[1].strip().lower()
                if selected_backend not in {"opencv", "qt"}:
                    print("\n⚠️ Use /panel opencv|qt\n")
                    continue
                if start_camera_panel(camera_runtime, voice_config, backend=selected_backend):
                    print(f"\n✅ panel = on | backend = {selected_backend}\n")
                    if selected_backend == "opencv":
                        print("Use ESC or q in the panel window to close it.\n")
                    else:
                        print("Close the Qt window to hide the panel.\n")
                continue

            if cmd == "voices":
                if speaker is None or voices is None:
                    print("\n⚠️ Voice engine is unavailable.\n")
                    continue
                choose_voice_interactive(speaker, voices, voice_config)
                continue

            if cmd == "config":
                configure_voice_and_stt(
                    voice_config,
                    speaker,
                    voices,
                    stt_runtime,
                    tts_runtime,
                    llm_state=llm_state,
                )
                pipe = llm_state["pipe"]
                current = llm_state["current"]
                history = llm_state["history"]
                continue

            if cmd.startswith("llm_backend"):
                parts = cmd.split(maxsplit=1)
                if len(parts) != 2:
                    current_backend = str(voice_config.get("llm_backend", "local"))
                    print(f"\nLLM backend is currently: {current_backend}")
                    print("Usage: /llm_backend local|external\n")
                    continue
                value = parts[1].strip().lower()
                if value not in {"local", "external"}:
                    print("\n⚠️ Invalid backend. Use local or external.\n")
                    continue
                voice_config["llm_backend"] = value
                save_robot_config(voice_config)
                release_llm_pipe(pipe)
                try:
                    pipe, current, history = activate_llm_from_config(voice_config, compat=compat)
                    server_state["pipe"] = pipe
                    server_state["current"] = current
                    llm_state["pipe"] = pipe
                    llm_state["current"] = current
                    llm_state["history"] = history
                    if current is not None:
                        print(f"\n✅ llm_backend = {value} | loaded: {model_menu_label(current)}\n")
                    else:
                        print(f"\n✅ llm_backend = {value}\n")
                except Exception as exc:
                    pipe = None
                    current = None
                    history = []
                    server_state["pipe"] = None
                    server_state["current"] = None
                    llm_state["pipe"] = None
                    llm_state["current"] = None
                    llm_state["history"] = history
                    print(f"\n❌ Failed to activate LLM backend '{value}': {exc}\n")
                continue

            if cmd.startswith("tts_backend"):
                parts = cmd.split(maxsplit=1)
                if len(parts) != 2:
                    current_tts = str(voice_config.get("tts_backend", DEFAULT_TTS_BACKEND))
                    print(f"\nTTS backend is currently: {current_tts}")
                    print("Usage: /tts_backend windows|openvino|kokoro|babelvox|espeakng|tada\n")
                    continue
                value = parts[1].strip().lower()
                if value not in TTS_BACKEND_OPTIONS:
                    print(f"\n⚠️ Invalid backend. Use one of: {', '.join(TTS_BACKEND_OPTIONS)}\n")
                    continue
                voice_config["tts_backend"] = value
                save_robot_config(voice_config)
                tts_runtime["active_key"] = None
                if value in {"openvino", "kokoro", "babelvox", "espeakng", "tada"}:
                    ensure_tts_runtime(tts_runtime, voice_config)
                print(f"\n✅ tts_backend = {value}\n")
                continue

            if cmd.startswith("tada_reference_audio"):
                parts = cmd.split(maxsplit=1)
                if len(parts) != 2:
                    current_value = str(voice_config.get("tada_reference_audio_path", "")).strip()
                    print(f"\nTADA reference audio is currently: {current_value or '(empty)'}")
                    print("Usage: /tada_reference_audio <path>\n")
                    continue
                voice_config["tada_reference_audio_path"] = parts[1].strip()
                save_robot_config(voice_config)
                tts_runtime["active_key"] = None
                print(f"\n✅ tada_reference_audio_path = {voice_config['tada_reference_audio_path']}\n")
                continue

            if cmd.startswith("tada_reference_text"):
                parts = cmd.split(maxsplit=1)
                if len(parts) != 2:
                    current_value = str(voice_config.get("tada_reference_text", "")).strip()
                    print(f"\nTADA reference text is currently: {current_value or '(empty)'}")
                    print("Usage: /tada_reference_text <text>\n")
                    continue
                voice_config["tada_reference_text"] = parts[1].strip()
                save_robot_config(voice_config)
                tts_runtime["active_key"] = None
                print("\n✅ tada_reference_text updated\n")
                continue

            if cmd == "tada_reference_record":
                capture_tada_reference_from_mic(stt_runtime, voice_config)
                tts_runtime["active_key"] = None
                continue

            if cmd.startswith("tada_model"):
                parts = cmd.split(maxsplit=1)
                if len(parts) != 2:
                    current_value = str(voice_config.get("tada_model_id", "HumeAI/tada-1b")).strip()
                    print(f"\nTADA model is currently: {current_value}")
                    print("Usage: /tada_model <hf-model-id>\n")
                    continue
                voice_config["tada_model_id"] = parts[1].strip()
                save_robot_config(voice_config)
                tts_runtime["active_key"] = None
                if str(voice_config.get("tts_backend", DEFAULT_TTS_BACKEND)).lower() == "tada":
                    ensure_tts_runtime(tts_runtime, voice_config)
                print(f"\n✅ tada_model_id = {voice_config['tada_model_id']}\n")
                continue

            if cmd.startswith("tada_codec"):
                parts = cmd.split(maxsplit=1)
                if len(parts) != 2:
                    current_value = str(voice_config.get("tada_codec_id", "HumeAI/tada-codec")).strip()
                    print(f"\nTADA codec is currently: {current_value}")
                    print("Usage: /tada_codec <hf-codec-id>\n")
                    continue
                voice_config["tada_codec_id"] = parts[1].strip()
                save_robot_config(voice_config)
                tts_runtime["active_key"] = None
                if str(voice_config.get("tts_backend", DEFAULT_TTS_BACKEND)).lower() == "tada":
                    ensure_tts_runtime(tts_runtime, voice_config)
                print(f"\n✅ tada_codec_id = {voice_config['tada_codec_id']}\n")
                continue

            if cmd.startswith("tada_device"):
                parts = cmd.split(maxsplit=1)
                if len(parts) != 2:
                    current_value = str(voice_config.get("tada_device", "cpu")).strip()
                    print(f"\nTADA device is currently: {current_value}")
                    print("Usage: /tada_device <device>\n")
                    continue
                voice_config["tada_device"] = parts[1].strip()
                save_robot_config(voice_config)
                tts_runtime["active_key"] = None
                if str(voice_config.get("tts_backend", DEFAULT_TTS_BACKEND)).lower() == "tada":
                    ensure_tts_runtime(tts_runtime, voice_config)
                print(f"\n✅ tada_device = {voice_config['tada_device']}\n")
                continue

            if cmd.startswith("tada_language"):
                parts = cmd.split(maxsplit=1)
                if len(parts) != 2:
                    current_value = str(voice_config.get("tada_language", "en")).strip()
                    print(f"\nTADA language is currently: {current_value}")
                    print("Usage: /tada_language <lang>\n")
                    continue
                voice_config["tada_language"] = parts[1].strip().lower()
                save_robot_config(voice_config)
                tts_runtime["active_key"] = None
                if str(voice_config.get("tts_backend", DEFAULT_TTS_BACKEND)).lower() == "tada":
                    ensure_tts_runtime(tts_runtime, voice_config)
                print(f"\n✅ tada_language = {voice_config['tada_language']}\n")
                continue

            if cmd.startswith("repeat"):
                parts = cmd.split(maxsplit=1)
                if len(parts) != 2:
                    print(f"\nrepeat is currently: {bool(voice_config.get('repeat', False))}")
                    print("Usage: /repeat true|false\n")
                    continue
                value = parts[1].strip().lower()
                if value in {"true", "1", "on", "si", "sí", "yes", "y"}:
                    voice_config["repeat"] = True
                    save_robot_config(voice_config)
                    print("\n✅ repeat = True\n")
                    continue
                if value in {"false", "0", "off", "no", "n"}:
                    voice_config["repeat"] = False
                    save_robot_config(voice_config)
                    print("\n✅ repeat = False\n")
                    continue
                print("\n⚠️ Use /repeat true|false\n")
                continue

            if cmd == "audio_inputs":
                sd_mod = stt_runtime.get("sounddevice") or ensure_dependency("sounddevice", "sounddevice", "SoundDevice")
                if sd_mod is None:
                    continue
                stt_runtime["sounddevice"] = sd_mod
                devices = list_audio_input_devices(sd_mod)
                if not devices:
                    print("\nNo audio input devices were detected.\n")
                    continue
                print("\nAudio input devices:\n")
                current_audio = str(voice_config.get("audio_input_device", "")).strip()
                for idx, name, channels in devices:
                    marker = " (current)" if current_audio and (current_audio == str(idx) or current_audio == name) else ""
                    print(f"  - {name} [index={idx}, channels={channels}]{marker}")
                print("")
                continue

            if cmd == "audio_input_select":
                sd_mod = stt_runtime.get("sounddevice") or ensure_dependency("sounddevice", "sounddevice", "SoundDevice")
                if sd_mod is None:
                    continue
                stt_runtime["sounddevice"] = sd_mod
                selected = choose_audio_input_device_interactive(sd_mod, str(voice_config.get("audio_input_device", "")).strip())
                if selected is None:
                    print("\nCancelled.\n")
                    continue
                voice_config["audio_input_device"] = selected[0]
                save_robot_config(voice_config)
                print(f"\n✅ audio_input_device = {selected[1]} [index={selected[0]}]\n")
                continue

            if cmd.startswith("audio_monitor"):
                parts = cmd.split(maxsplit=1)
                if len(parts) != 2:
                    state = "on" if bool(voice_config.get("audio_monitor_enabled", False)) else "off"
                    print(f"\naudio_monitor is currently: {state}")
                    print("Usage: /audio_monitor on|off\n")
                    continue
                value = parts[1].strip().lower()
                if value in {"on", "1", "true", "yes", "y", "si", "sí"}:
                    voice_config["audio_monitor_enabled"] = True
                    save_robot_config(voice_config)
                    start_audio_monitor(auto_listen_runtime)
                    print("\n✅ audio_monitor = on\n")
                    continue
                if value in {"off", "0", "false", "no", "n"}:
                    voice_config["audio_monitor_enabled"] = False
                    save_robot_config(voice_config)
                    stop_audio_monitor(auto_listen_runtime)
                    print("\n✅ audio_monitor = off\n")
                    continue
                print("\n⚠️ Use /audio_monitor on|off\n")
                continue

            if cmd.startswith("audio"):
                parts = cmd.split(maxsplit=1)
                if len(parts) != 2:
                    state = "on" if bool(voice_config.get("audio_enabled", True)) else "off"
                    print(f"\naudio is currently: {state}")
                    print("Usage: /audio on|off\n")
                    continue
                value = parts[1].strip().lower()
                if value in {"on", "1", "true", "yes", "y", "si", "sí"}:
                    voice_config["audio_enabled"] = True
                    save_robot_config(voice_config)
                    print("\n✅ audio = on\n")
                    continue
                if value in {"off", "0", "false", "no", "n"}:
                    voice_config["audio_enabled"] = False
                    save_robot_config(voice_config)
                    print("\n✅ audio = off\n")
                    continue
                print("\n⚠️ Use /audio on|off\n")
                continue

            if cmd.startswith("camera") or cmd.startswith("camara"):
                parts = cmd.split(maxsplit=1)
                if len(parts) != 2:
                    state = "on" if bool(voice_config.get("camera_enabled", False)) else "off"
                    print(f"\ncamera is currently: {state}")
                    print("Usage: /camera on|off\n")
                    continue
                value = parts[1].strip().lower()
                if value in {"on", "1", "true", "yes", "y", "si", "sí"}:
                    started = start_camera_preview(camera_runtime, voice_config)
                    if started:
                        print("Use '/camera off' to disable the camera feed.\n")
                        if camera_runtime.get("thread") is None or not camera_runtime["thread"].is_alive():
                            print("Use '/panel' to open the control panel window.\n")
                    continue
                if value in {"off", "0", "false", "no", "n"}:
                    set_camera_enabled(camera_runtime, voice_config, False)
                    continue
                print("\n⚠️ Use /camera on|off\n")
                continue

            if cmd.startswith("log"):
                parts = cmd.split(maxsplit=1)
                if len(parts) == 1:
                    state = "on" if bool(voice_config.get("vision_log_enabled", False)) else "off"
                    interval_s = float(voice_config.get("vision_log_interval_s", 1.0))
                    print(f"\nvision log is currently: {state}")
                    print(f"vision log interval is currently: {interval_s:.1f}s")
                    print("Usage: /log on|off|<seconds>|interval <seconds>\n")
                    continue
                value = parts[1].strip().lower()
                if value in {"on", "1", "true", "yes", "y", "si", "sí"}:
                    voice_config["vision_log_enabled"] = True
                    save_robot_config(voice_config)
                    camera_runtime["vision_log_enabled"] = True
                    camera_runtime["vision_log_last_ts"] = 0.0
                    print("\n✅ vision log = on\n")
                    continue
                if value in {"off", "0", "false", "no", "n"}:
                    voice_config["vision_log_enabled"] = False
                    save_robot_config(voice_config)
                    camera_runtime["vision_log_enabled"] = False
                    print("\n✅ vision log = off\n")
                    continue
                if value.startswith("interval "):
                    value = value.split(maxsplit=1)[1].strip()
                try:
                    interval_s = max(0.1, float(value))
                except ValueError:
                    print("\n⚠️ Use /log on|off|<seconds>|interval <seconds>\n")
                    continue
                voice_config["vision_log_interval_s"] = interval_s
                save_robot_config(voice_config)
                camera_runtime["vision_log_interval_s"] = interval_s
                camera_runtime["vision_log_last_ts"] = 0.0
                camera_runtime["vision_event_last_ts"] = 0.0
                print(f"\n✅ vision log interval = {interval_s:.1f}s\n")
                continue

            if cmd.startswith("vision_events"):
                parts = cmd.split(maxsplit=1)
                if len(parts) != 2:
                    state = "on" if bool(voice_config.get("vision_event_processing_enabled", True)) else "off"
                    print(f"\nvision_events is currently: {state}")
                    print("Usage: /vision_events on|off\n")
                    continue
                value = parts[1].strip().lower()
                if value in {"on", "1", "true", "yes", "y", "si", "sí"}:
                    voice_config["vision_event_processing_enabled"] = True
                    save_robot_config(voice_config)
                    camera_runtime["vision_event_processing_enabled"] = True
                    camera_runtime["vision_event_last_ts"] = 0.0
                    camera_runtime["vision_last_detection_count"] = 0
                    print("\n✅ vision_events = on\n")
                    continue
                if value in {"off", "0", "false", "no", "n"}:
                    voice_config["vision_event_processing_enabled"] = False
                    save_robot_config(voice_config)
                    camera_runtime["vision_event_processing_enabled"] = False
                    camera_runtime["vision_event_last_ts"] = 0.0
                    camera_runtime["vision_last_detection_count"] = 0
                    print("\n✅ vision_events = off\n")
                    continue
                print("\n⚠️ Use /vision_events on|off\n")
                continue

            if cmd.startswith("vision_model"):
                parts = cmd.split(maxsplit=1)
                if len(parts) != 2:
                    print(f"\nvision_model_path is currently: {voice_config.get('vision_model_path', '') or '(empty)'}")
                    print("Usage: /vision_model <path-to-model.xml>\n")
                    continue
                value = parts[1].strip().strip('"')
                if not value:
                    print("\n⚠️ vision_model path cannot be empty.\n")
                    continue
                voice_config["vision_model_path"] = value
                save_robot_config(voice_config)
                camera_runtime["vision_compiled_model"] = None
                camera_runtime["vision_active_key"] = None
                print(f"\n✅ vision_model_path = {value}\n")
                continue

            if cmd == "vision_models":
                list_vision_models(load_vision_models(), str(voice_config.get("vision_model_id", "")).strip())
                continue

            if cmd == "vision_select":
                models = load_vision_models()
                selected = choose_vision_model_interactive(
                    models,
                    selected_id=str(voice_config.get("vision_model_id", "")).strip(),
                    allow_download=True,
                )
                if selected is None:
                    print("\nCancelled.\n")
                    continue
                selected_device = choose_vision_device_interactive(
                    str(voice_config.get("vision_device", "AUTO")).strip()
                )
                if selected_device is None:
                    print("\nCancelled.\n")
                    continue
                voice_config["vision_model_id"] = selected["id"]
                voice_config["vision_model_path"] = str(selected["xml_path"])
                voice_config["vision_device"] = selected_device
                labels_path = selected["local"] / "labels.txt"
                voice_config["vision_labels_path"] = str(labels_path) if labels_path.exists() else ""
                save_robot_config(voice_config)
                camera_runtime["vision_compiled_model"] = None
                camera_runtime["vision_active_key"] = None
                camera_runtime["vision_labels"] = list(selected.get("labels", []))
                print(f"\n✅ Vision model selected: {selected['display']} [{selected['id']}] on {selected_device}\n")
                continue

            if cmd.startswith("vision_labels"):
                parts = cmd.split(maxsplit=1)
                if len(parts) != 2:
                    print(f"\nvision_labels_path is currently: {voice_config.get('vision_labels_path', '') or '(empty)'}")
                    print("Usage: /vision_labels <path-to-labels.txt>\n")
                    continue
                value = parts[1].strip().strip('"')
                voice_config["vision_labels_path"] = value
                save_robot_config(voice_config)
                camera_runtime["vision_labels"] = load_vision_labels(value)
                print(f"\n✅ vision_labels_path = {value or '(empty)'}\n")
                continue

            if cmd.startswith("vision_device"):
                parts = cmd.split(maxsplit=1)
                if len(parts) != 2:
                    print(f"\nvision_device is currently: {voice_config.get('vision_device', 'AUTO')}")
                    print("Usage: /vision_device CPU|GPU|NPU|AUTO\n")
                    continue
                value = parts[1].strip().upper()
                if value not in {"CPU", "GPU", "NPU", "AUTO"}:
                    print("\n⚠️ Use /vision_device CPU|GPU|NPU|AUTO\n")
                    continue
                voice_config["vision_device"] = value
                save_robot_config(voice_config)
                camera_runtime["vision_compiled_model"] = None
                camera_runtime["vision_active_key"] = None
                print(f"\n✅ vision_device = {value}\n")
                continue

            if cmd.startswith("vision"):
                parts = cmd.split(maxsplit=1)
                if len(parts) != 2:
                    state = "on" if bool(voice_config.get("vision_enabled", False)) else "off"
                    print(f"\nvision is currently: {state}")
                    print("Usage: /vision on|off\n")
                    continue
                value = parts[1].strip().lower()
                if value in {"on", "1", "true", "yes", "y", "si", "sí"}:
                    if not ensure_vision_runtime(camera_runtime, voice_config):
                        continue
                    voice_config["vision_enabled"] = True
                    save_robot_config(voice_config)
                    camera_runtime["vision_enabled"] = True
                    camera_runtime["vision_threshold"] = float(voice_config.get("vision_threshold", 0.4))
                    print("\n✅ vision = on\n")
                    continue
                if value in {"off", "0", "false", "no", "n"}:
                    voice_config["vision_enabled"] = False
                    save_robot_config(voice_config)
                    camera_runtime["vision_enabled"] = False
                    print("\n✅ vision = off\n")
                    continue
                print("\n⚠️ Use /vision on|off\n")
                continue

            if cmd.startswith("max_tokens"):
                parts = cmd.split(maxsplit=1)
                if len(parts) != 2:
                    print(f"\nmax_tokens is currently: {int(voice_config.get('max_new_tokens', 300))}")
                    print("Usage: /max_tokens <1..8192>\n")
                    continue
                try:
                    value = int(parts[1].strip())
                except ValueError:
                    print("\n⚠️ max_tokens must be an integer.\n")
                    continue
                if not (1 <= value <= 8192):
                    print("\n⚠️ max_tokens must be between 1 and 8192.\n")
                    continue
                voice_config["max_new_tokens"] = value
                save_robot_config(voice_config)
                print(f"\n✅ max_tokens = {value}\n")
                continue

            if cmd == "listen":
                run_listen_mode(pipe, current, history, stats, speaker, voice_config, stt_runtime, tts_runtime)
                continue

            if cmd.startswith("auto_listen"):
                parts = cmd.split(maxsplit=1)
                if len(parts) != 2:
                    state = "on" if bool(voice_config.get("auto_listen_enabled", False)) else "off"
                    print(f"\nauto_listen is currently: {state}")
                    print("Usage: /auto_listen on|off\n")
                    continue
                value = parts[1].strip().lower()
                if value in {"on", "1", "true", "yes", "y", "si", "sí"}:
                    voice_config["auto_listen_enabled"] = True
                    save_robot_config(voice_config)
                    refresh_auto_listen_worker(auto_listen_runtime, voice_config)
                    print("\n✅ auto_listen = on\n")
                    continue
                if value in {"off", "0", "false", "no", "n"}:
                    voice_config["auto_listen_enabled"] = False
                    save_robot_config(voice_config)
                    refresh_auto_listen_worker(auto_listen_runtime, voice_config)
                    print("\n✅ auto_listen = off\n")
                    continue
                print("\n⚠️ Use /auto_listen on|off\n")
                continue

            if cmd.startswith("wake_word_enabled") or cmd.startswith("wake_world_enabled"):
                parts = cmd.split(maxsplit=1)
                if len(parts) != 2:
                    state = "on" if bool(voice_config.get("wake_word_enabled", False)) else "off"
                    print(f"\nwake_word_enabled is currently: {state}")
                    print("Usage: /wake_word_enabled true|false\n")
                    continue
                value = parts[1].strip().lower()
                if value in {"on", "1", "true", "yes", "y", "si", "sí"}:
                    voice_config["wake_word_enabled"] = True
                    save_robot_config(voice_config)
                    refresh_auto_listen_worker(auto_listen_runtime, voice_config)
                    print("\n✅ wake_word_enabled = True\n")
                    continue
                if value in {"off", "0", "false", "no", "n"}:
                    voice_config["wake_word_enabled"] = False
                    save_robot_config(voice_config)
                    refresh_auto_listen_worker(auto_listen_runtime, voice_config)
                    print("\n✅ wake_word_enabled = False\n")
                    continue
                print("\n⚠️ Use /wake_word_enabled true|false\n")
                continue

            if cmd.startswith("wake_word_phrase"):
                parts = cmd.split(maxsplit=1)
                if len(parts) != 2:
                    current_value = str(voice_config.get("wake_word_phrase", "hola robot")).strip()
                    print(f"\nwake_word_phrase is currently: {current_value}")
                    print("Usage: /wake_word_phrase <text>\n")
                    continue
                voice_config["wake_word_phrase"] = parts[1].strip()
                save_robot_config(voice_config)
                print(f"\n✅ wake_word_phrase = {voice_config['wake_word_phrase']}\n")
                continue

            if cmd.startswith("wake_word_stop_phrase"):
                parts = cmd.split(maxsplit=1)
                if len(parts) != 2:
                    current_value = str(voice_config.get("wake_word_stop_phrase", "adios robot")).strip()
                    print(f"\nwake_word_stop_phrase is currently: {current_value}")
                    print("Usage: /wake_word_stop_phrase <text>\n")
                    continue
                voice_config["wake_word_stop_phrase"] = parts[1].strip()
                save_robot_config(voice_config)
                print(f"\n✅ wake_word_stop_phrase = {voice_config['wake_word_stop_phrase']}\n")
                continue

            if cmd.startswith("wake_word_on_response"):
                parts = cmd.split(maxsplit=1)
                if len(parts) != 2:
                    current_value = str(voice_config.get("wake_word_on_response", "Te escucho.")).strip()
                    print(f"\nwake_word_on_response is currently: {current_value}")
                    print("Usage: /wake_word_on_response <text>\n")
                    continue
                voice_config["wake_word_on_response"] = parts[1].strip()
                save_robot_config(voice_config)
                print(f"\n✅ wake_word_on_response = {voice_config['wake_word_on_response']}\n")
                continue

            if cmd.startswith("wake_word_off_response"):
                parts = cmd.split(maxsplit=1)
                if len(parts) != 2:
                    current_value = str(voice_config.get("wake_word_off_response", "Modo escucha desactivado.")).strip()
                    print(f"\nwake_word_off_response is currently: {current_value}")
                    print("Usage: /wake_word_off_response <text>\n")
                    continue
                voice_config["wake_word_off_response"] = parts[1].strip()
                save_robot_config(voice_config)
                print(f"\n✅ wake_word_off_response = {voice_config['wake_word_off_response']}\n")
                continue

            if cmd.startswith("vad_silence"):
                parts = cmd.split(maxsplit=1)
                if len(parts) != 2:
                    current_ms = int(voice_config.get("auto_listen_silence_ms", 1600))
                    print(f"\nvad_silence is currently: {current_ms} ms")
                    print("Usage: /vad_silence <milliseconds>\n")
                    continue
                try:
                    value_ms = int(parts[1].strip())
                except ValueError:
                    print("\n⚠️ /vad_silence expects an integer number of milliseconds.\n")
                    continue
                if value_ms < 100:
                    print("\n⚠️ /vad_silence must be at least 100 ms.\n")
                    continue
                voice_config["auto_listen_silence_ms"] = value_ms
                save_robot_config(voice_config)
                if should_run_auto_listen_worker(voice_config):
                    stop_auto_listen(auto_listen_runtime)
                    start_auto_listen(
                        auto_listen_runtime,
                        voice_config,
                        activate_session=bool(voice_config.get("auto_listen_enabled", False)),
                    )
                    print(f"\n✅ vad_silence = {value_ms} ms | auto_listen restarted\n")
                else:
                    print(f"\n✅ vad_silence = {value_ms} ms\n")
                continue

            if cmd.startswith("vad_preroll"):
                parts = cmd.split(maxsplit=1)
                if len(parts) != 2:
                    current_ms = int(voice_config.get("auto_listen_preroll_ms", 350))
                    print(f"\nvad_preroll is currently: {current_ms} ms")
                    print("Usage: /vad_preroll <milliseconds>\n")
                    continue
                try:
                    value_ms = int(parts[1].strip())
                except ValueError:
                    print("\n⚠️ /vad_preroll expects an integer number of milliseconds.\n")
                    continue
                if value_ms < 0:
                    print("\n⚠️ /vad_preroll must be 0 ms or higher.\n")
                    continue
                voice_config["auto_listen_preroll_ms"] = value_ms
                save_robot_config(voice_config)
                if should_run_auto_listen_worker(voice_config):
                    stop_auto_listen(auto_listen_runtime)
                    start_auto_listen(
                        auto_listen_runtime,
                        voice_config,
                        activate_session=bool(voice_config.get("auto_listen_enabled", False)),
                    )
                    print(f"\n✅ vad_preroll = {value_ms} ms | auto_listen restarted\n")
                else:
                    print(f"\n✅ vad_preroll = {value_ms} ms\n")
                continue

            if cmd.startswith("vad_max_segment"):
                parts = cmd.split(maxsplit=1)
                if len(parts) != 2:
                    current_s = float(voice_config.get("auto_listen_max_segment_s", 60.0))
                    print(f"\nvad_max_segment is currently: {current_s:.1f} s")
                    print("Usage: /vad_max_segment <seconds>\n")
                    continue
                try:
                    value_s = float(parts[1].strip())
                except ValueError:
                    print("\n⚠️ /vad_max_segment expects a number of seconds.\n")
                    continue
                if value_s < 1.0:
                    print("\n⚠️ /vad_max_segment must be at least 1 second.\n")
                    continue
                voice_config["auto_listen_max_segment_s"] = value_s
                save_robot_config(voice_config)
                if should_run_auto_listen_worker(voice_config):
                    stop_auto_listen(auto_listen_runtime)
                    start_auto_listen(
                        auto_listen_runtime,
                        voice_config,
                        activate_session=bool(voice_config.get("auto_listen_enabled", False)),
                    )
                    print(f"\n✅ vad_max_segment = {value_s:.1f} s | auto_listen restarted\n")
                else:
                    print(f"\n✅ vad_max_segment = {value_s:.1f} s\n")
                continue

            if cmd == "whisper_models":
                whisper_ov_models = load_whisper_ov_models()
                list_whisper_ov_models(
                    whisper_ov_models,
                    selected_id=str(voice_config.get("whisper_openvino_model_id", "")).strip(),
                )
                continue

            if cmd == "whisper_add":
                whisper_ov_models = load_whisper_ov_models()
                add_whisper_ov_model_interactive(whisper_ov_models)
                continue

            if cmd == "whisper_select":
                whisper_ov_models = load_whisper_ov_models()
                selected_id = str(voice_config.get("whisper_openvino_model_id", "")).strip()
                selected = choose_whisper_ov_model_interactive(
                    whisper_ov_models,
                    selected_id=selected_id,
                    allow_download=False,
                )
                if selected is None:
                    print("\nCancelled.\n")
                    continue
                voice_config["whisper_openvino_model_id"] = selected["id"]
                save_robot_config(voice_config)
                stt_runtime["active_key"] = None
                print(f"\n✅ Whisper OpenVINO model selected: {selected['id']}\n")
                if bool(voice_config.get("whisper_openvino", False)):
                    ensure_stt_runtime(stt_runtime, voice_config)
                continue

            if cmd == "openvino_tts_models":
                ov_models = load_ov_tts_models()
                list_ov_tts_models(
                    ov_models,
                    selected_id=str(voice_config.get("openvino_tts_model_id", "")).strip(),
                )
                continue

            if cmd == "openvino_tts_add":
                ov_models = load_ov_tts_models()
                add_ov_tts_model_interactive(ov_models)
                continue

            if cmd == "openvino_tts_select":
                ov_models = load_ov_tts_models()
                selected_id = str(voice_config.get("openvino_tts_model_id", "")).strip()
                selected = choose_ov_tts_model_interactive(
                    ov_models,
                    selected_id=selected_id,
                    allow_download=False,
                )
                if selected is None:
                    print("\nCancelled.\n")
                    continue
                voice_config["openvino_tts_model_id"] = selected["id"]
                save_robot_config(voice_config)
                tts_runtime["active_key"] = None
                print(f"\n✅ OpenVINO TTS model selected: {selected['id']}\n")
                if str(voice_config.get("tts_backend", DEFAULT_TTS_BACKEND)).lower() == "openvino":
                    ensure_tts_runtime(tts_runtime, voice_config)
                continue

            if cmd == "kokoro_models":
                models = load_kokoro_models()
                _list_repo_models("Kokoro models:", models, str(voice_config.get("kokoro_model_id", "")).strip())
                continue

            if cmd == "kokoro_select":
                models = load_kokoro_models()
                selected = _choose_repo_model_interactive(
                    "Kokoro models:",
                    models,
                    str(voice_config.get("kokoro_model_id", "")).strip(),
                )
                if selected is None:
                    print("\nCancelled.\n")
                    continue
                voice_config["kokoro_model_id"] = selected["id"]
                save_robot_config(voice_config)
                tts_runtime["active_key"] = None
                print(f"\n✅ Kokoro model selected: {selected['id']}\n")
                if str(voice_config.get("tts_backend", DEFAULT_TTS_BACKEND)).lower() == "kokoro":
                    ensure_tts_runtime(tts_runtime, voice_config)
                continue

            if cmd == "babelvox_models":
                models = load_babelvox_models()
                _list_repo_models("BabelVox models:", models, str(voice_config.get("babelvox_model_id", "")).strip())
                continue

            if cmd == "babelvox_select":
                models = load_babelvox_models()
                selected = _choose_repo_model_interactive(
                    "BabelVox models:",
                    models,
                    str(voice_config.get("babelvox_model_id", "")).strip(),
                )
                if selected is None:
                    print("\nCancelled.\n")
                    continue
                voice_config["babelvox_model_id"] = selected["id"]
                save_robot_config(voice_config)
                tts_runtime["active_key"] = None
                print(f"\n✅ BabelVox model selected: {selected['id']}\n")
                if str(voice_config.get("tts_backend", DEFAULT_TTS_BACKEND)).lower() == "babelvox":
                    ensure_tts_runtime(tts_runtime, voice_config)
                continue

            if cmd == "espeak_voices":
                list_espeak_voices()
                continue

            if cmd == "exit":
                break

            if cmd == "start_server":
                if server is not None:
                    print("\nℹ️ Server is already running on http://0.0.0.0:1311\n")
                    continue
                server = start_openai_compatible_server(server_state)
                print("\n✅ Server started at http://0.0.0.0:1311/v1/chat/completions\n")
                continue

            if cmd == "stats":
                print_stats_table(stats)
                continue

            if cmd == "all_models":
                print_all_models_summary(compat)
                continue

            if cmd.startswith("clear_stats"):
                parts = cmd.split()
                if len(parts) == 1:
                    clear_stats(stats)
                    continue
                if len(parts) == 2:
                    if not parts[1].isdigit():
                        print("\n⚠️ Usage: clear_stats [model_number] [device]\n")
                        continue
                    clear_stats(stats, model_number=int(parts[1]))
                    continue
                if len(parts) == 3:
                    if not parts[1].isdigit():
                        print("\n⚠️ Usage: clear_stats [model_number] [device]\n")
                        continue
                    clear_stats(stats, model_number=int(parts[1]), device=parts[2])
                    continue
                print("\n⚠️ Usage: clear_stats [model_number] [device]\n")
                continue

            if cmd == "current_model":
                if current is None:
                    print(
                        f"\n(no model loaded) | DEVICE={ACTIVE_DEVICE} | "
                        f"PERFORMANCE_HINT={ACTIVE_PERFORMANCE_HINT}\n"
                    )
                else:
                    print(
                        f"\nLoaded model: {model_menu_label(current)} | "
                        f"DEVICE={ACTIVE_DEVICE} | PERFORMANCE_HINT={ACTIVE_PERFORMANCE_HINT}\n"
                    )
                continue

            if cmd.startswith("benchmark"):
                parts = cmd.split(maxsplit=1)
                model_number = None
                only_missing_models = False
                if len(parts) == 2:
                    if not parts[1].isdigit():
                        print("\n⚠️ benchmark expects a model number, e.g. '/benchmark 2'.\n")
                        continue
                    model_number = int(parts[1])
                else:
                    only_missing_models = prompt_yes_no(
                        "Run only models missing benchmark stats?",
                        default_yes=True,
                    )

                prompts = collect_benchmark_prompts(5)
                benchmark_models(
                    stats,
                    prompts,
                    model_number=model_number,
                    only_missing_models=only_missing_models,
                    compat=compat,
                )
                continue

            if cmd == "add_model":
                add_model_interactive(MODELS)
                continue

            if cmd == "models":
                new_model = choose_model_interactive(
                    allow_download=True,
                    title="Choose a model to load:",
                    compat=compat,
                )
                if new_model is None:
                    print("\nCancelled.\n")
                    continue

                configure_runtime()
                voice_config["llm_backend"] = "local"
                voice_config["llm_device"] = ACTIVE_DEVICE
                voice_config["llm_performance_hint"] = ACTIVE_PERFORMANCE_HINT
                save_robot_config(voice_config)

                # release previous
                release_llm_pipe(pipe)

                current = new_model
                history = []
                try:
                    pipe = load_pipeline(current)
                    mark_model_device_compat(compat, current["repo"], ACTIVE_DEVICE, True)
                    save_device_compat(compat)
                    server_state["pipe"] = pipe
                    server_state["current"] = current
                    llm_state["pipe"] = pipe
                    llm_state["current"] = current
                    llm_state["history"] = history
                    voice_config["llm_device"] = ACTIVE_DEVICE
                    voice_config["current_model_repo"] = current["repo"]
                    save_robot_config(voice_config)
                except Exception as exc:
                    mark_model_device_compat(compat, current["repo"], ACTIVE_DEVICE, False)
                    save_device_compat(compat)
                    pipe = None
                    current = None
                    history = []
                    server_state["pipe"] = None
                    server_state["current"] = None
                    llm_state["pipe"] = None
                    llm_state["current"] = None
                    llm_state["history"] = history
                    print(f"\n❌ Failed to load model on {ACTIVE_DEVICE}: {exc}\n")
                continue

            if cmd == "delete":
                to_delete = choose_model_interactive(
                    allow_download=False,
                    title="Choose a model to DELETE from disk (it remains in the list):",
                    compat=compat,
                )
                if to_delete is None:
                    print("\nCancelled.\n")
                    continue

                deleting_current = (current is not None and to_delete["repo"] == current["repo"])
                if deleting_current and pipe is not None:
                    release_llm_pipe(pipe)
                    pipe = None
                    server_state["pipe"] = None
                    server_state["current"] = None
                    llm_state["pipe"] = None
                    llm_state["current"] = None

                deleted = delete_model_files(to_delete)

                if deleting_current:
                    if deleted:
                        print("ℹ️ You deleted the active model. Load another one with '/models'.\n")
                        current = None
                        history = []
                        server_state["pipe"] = None
                        server_state["current"] = None
                        llm_state["pipe"] = None
                        llm_state["current"] = None
                        llm_state["history"] = history
                        voice_config["current_model_repo"] = ""
                        save_robot_config(voice_config)
                    else:
                        print("ℹ️ Could not delete the active model.\n")
                continue

            print("\n⚠️ Unknown command. Use '/help'.\n")
            continue

        if bool(voice_config.get("repeat", False)):
            if bool(voice_config.get("audio_enabled", True)):
                try:
                    interrupted, calc_latency_s = speak_text_backend(
                        speaker,
                        user_input,
                        voice_config,
                        tts_runtime,
                        allow_interrupt=True,
                    )
                    print(f"⏱️ text -> audio: {calc_latency_s:0.3f}s")
                    if interrupted:
                        print("⏹️ Audio interrupted by ESC.")
                except Exception as exc:
                    print(f"\n⚠️ TTS failed: {exc}\n")
            else:
                print("⏱️ text -> audio: skipped (audio off)")
            continue

        run_chat_turn(user_input, pipe, current, history, stats, speaker, voice_config, tts_runtime)

    if server is not None:
        server.shutdown()
        server.server_close()
    stop_auto_listen(auto_listen_runtime)
    stop_camera_preview(camera_runtime)
    print("Bye.")


if __name__ == "__main__":
    main()
