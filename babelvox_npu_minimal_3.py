# babelvox_npu_debug_minimal.py
import os
import sys
import time
import traceback
import faulthandler
from pathlib import Path

import soundfile as sf

# Automatically dump the stack if it hangs for more than 120s
faulthandler.enable()
faulthandler.dump_traceback_later(120, repeat=True)

print("[0] Python:", sys.version)
print("[1] PID:", os.getpid())
print("[2] Starting BabelVox NPU-only-style debug run...")

# Optional: force more verbose OpenVINO logs
os.environ.setdefault("OV_LOG_LEVEL", "DEBUG")

try:
    import openvino as ov
    print("[3] OpenVINO imported OK")
    core = ov.Core()
    devices = core.available_devices
    print("[4] OpenVINO devices:", devices)

    if "NPU" not in devices:
        raise RuntimeError("NPU does not appear in openvino.Core().available_devices")

except Exception as e:
    print("[X] Error initializing OpenVINO/NPU:", repr(e))
    traceback.print_exc()
    sys.exit(1)

try:
    from babelvox import BabelVox
    print("[5] BabelVox imported OK")
except Exception as e:
    print("[X] Error importing BabelVox:", repr(e))
    traceback.print_exc()
    sys.exit(1)

CACHE_DIR = Path("./ov_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TEXT = "Hola."
LANG = "Spanish"
OUT = "babelvox_npu_debug.wav"

print("[6] Config:")
print("    device        = NPU")
print("    precision     = int8")
print("    use_cp_kv_cache = False")
print("    talker_buckets  = None")
print("    text            =", repr(TEXT))
print("    language        =", LANG)
print("    cache_dir       =", str(CACHE_DIR.resolve()))

tts = None
audio = None
sr = None

try:
    t0 = time.perf_counter()
    print("[7] Creating BabelVox instance...")
    tts = BabelVox(
        device="NPU",
        precision="int8",
        use_cp_kv_cache=False,   # no hybrid KV cache mode
        talker_buckets=None,     # absolute minimum, no buckets
        cache_dir=str(CACHE_DIR),
    )
    t1 = time.perf_counter()
    print(f"[8] BabelVox instance created in {t1 - t0:.2f}s")

except Exception as e:
    print("[X] Error creando BabelVox:", repr(e))
    traceback.print_exc()
    sys.exit(2)

try:
    print("[9] About to call generate() ...")
    t2 = time.perf_counter()

    # Very short input and no cloned voice
    result = tts.generate(
        TEXT,
        language=LANG,
        max_new_tokens=64,
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        repetition_penalty=1.0,
    )

    t3 = time.perf_counter()
    print(f"[10] generate() returned in {t3 - t2:.2f}s")

    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError(f"generate() returned something unexpected: {type(result)} -> {result!r}")

    audio, sr = result
    print("[11] Audio type:", type(audio))
    print("[12] Sample rate:", sr)

    # Some backends return unusual list/array shapes
    try:
        length = len(audio)
    except Exception:
        length = None
    print("[13] Audio length:", length)

    if length is None or length == 0:
        raise RuntimeError("Empty audio")

except Exception as e:
    print("[X] Error during generate():", repr(e))
    traceback.print_exc()
    sys.exit(3)

try:
    print("[14] Writing WAV:", OUT)
    sf.write(OUT, audio, sr)
    print("[15] WAV written OK:", OUT)
except Exception as e:
    print("[X] Error writing WAV:", repr(e))
    traceback.print_exc()
    sys.exit(4)

finally:
    faulthandler.cancel_dump_traceback_later()

print("[16] DONE")
