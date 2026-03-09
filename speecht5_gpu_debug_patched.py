#!/usr/bin/env python3
"""
SpeechT5 TTS OpenVINO GPU debug script
- Verifies OpenVINO devices
- Attempts to compile each IR on GPU
- Constructs Text2SpeechPipeline(..., "GPU")
- Generates minimal audio
- If the result comes back as a remote GPU tensor, copies it to host before reading

Usage:
    python speecht5_gpu_debug_patched.py "C:\\path\\speech-t5-tts-ov"
    python speecht5_gpu_debug_patched.py "C:\\path\\speech-t5-tts-ov" "Hello Fernando."
"""

import os
import sys
import time
import traceback
import faulthandler
from pathlib import Path

import numpy as np
import soundfile as sf

faulthandler.enable()
faulthandler.dump_traceback_later(120, repeat=True)

def log(msg: str) -> None:
    print(msg, flush=True)

def try_imports():
    import openvino as ov
    import openvino_genai as ov_genai
    return ov, ov_genai

def list_ir_files(model_dir: Path):
    return sorted(model_dir.glob("*.xml"))

def compile_each_ir_on_gpu(ov, model_dir: Path):
    core = ov.Core()
    log(f"[0] available devices: {core.available_devices}")

    xmls = list_ir_files(model_dir)
    if not xmls:
        raise RuntimeError(f"No encontré .xml en {model_dir}")

    for xml in xmls:
        log(f"[A] reading IR: {xml.name}")
        t0 = time.perf_counter()
        model = core.read_model(str(xml))
        t1 = time.perf_counter()
        log(f"    read_model OK in {t1 - t0:.2f}s")

        log(f"[A] compiling on GPU: {xml.name}")
        t2 = time.perf_counter()
        compiled = core.compile_model(model, "GPU")
        t3 = time.perf_counter()
        log(f"    compile_model(GPU) OK in {t3 - t2:.2f}s")
        try:
            outs = [o.get_any_name() if hasattr(o, "get_any_name") else str(o) for o in compiled.outputs]
        except Exception:
            outs = [str(o) for o in compiled.outputs]
        log(f"    outputs: {outs}")

def extract_wave_to_numpy(ov, result):
    """
    Handles several possible output formats from openvino_genai TTS.
    If the tensor is remote on the GPU, it copies it to host before reading.
    """
    speech = None

    # Casos comunes
    if hasattr(result, "speeches"):
        speeches = result.speeches
        if len(speeches) == 0:
            raise RuntimeError("result.speeches is empty")
        speech = speeches[0]
    elif isinstance(result, (list, tuple)):
        if len(result) == 0:
            raise RuntimeError("empty result")
        speech = result[0]
    else:
        speech = result

    log(f"[E] speech object type: {type(speech)}")

    # If it is already a numpy array/list
    if isinstance(speech, np.ndarray):
        wav = speech
        if wav.ndim > 1:
            wav = wav[0]
        return np.array(wav, copy=True)

    if isinstance(speech, (list, tuple)):
        wav = np.array(speech)
        if wav.ndim > 1:
            wav = wav[0]
        return np.array(wav, copy=True)

    # If it is an OpenVINO Tensor
    if hasattr(speech, "shape") and hasattr(speech, "element_type"):
        log(f"[E] tensor shape: {speech.shape}")
        log(f"[E] tensor element_type: {speech.element_type}")

        # Direct attempt
        try:
            data = speech.data
            wav = np.array(data[0] if getattr(data, "ndim", 1) > 1 else data, copy=True)
            log("[E] direct speech.data read OK")
            return wav
        except Exception as e:
            log(f"[E] direct speech.data read failed: {repr(e)}")

        # Explicit copy to a host tensor
        try:
            host_tensor = ov.Tensor(speech.element_type, speech.shape)
            log("[E] copying remote/device tensor -> host tensor ...")
            host_tensor.copy_from(speech)
            data = host_tensor.data
            wav = np.array(data[0] if getattr(data, "ndim", 1) > 1 else data, copy=True)
            log("[E] copy_from(...) OK")
            return wav
        except Exception as e:
            raise RuntimeError(f"Could not extract audio from OpenVINO tensor: {repr(e)}") from e

    raise RuntimeError(f"Unsupported output format: {type(speech)}")

def main():
    if len(sys.argv) < 2:
        print('Usage: python speecht5_gpu_debug_patched.py "C:\\path\\speech-t5-tts-ov" ["text"]')
        sys.exit(1)

    model_dir = Path(sys.argv[1])
    text = sys.argv[2] if len(sys.argv) >= 3 else "Hello."
    out_wav = "speecht5_gpu.wav"

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    os.environ.setdefault("OV_LOG_LEVEL", "DEBUG")

    ov, ov_genai = try_imports()

    log(f"[1] model_dir: {model_dir}")
    log(f"[2] text: {text!r}")

    compile_each_ir_on_gpu(ov, model_dir)

    log("[B] constructing Text2SpeechPipeline(model_dir, 'GPU') ...")
    t0 = time.perf_counter()
    pipe = ov_genai.Text2SpeechPipeline(str(model_dir), "GPU")
    t1 = time.perf_counter()
    log(f"[B] pipeline constructed in {t1 - t0:.2f}s")

    log("[D] calling generate() ...")
    t2 = time.perf_counter()
    result = pipe.generate(text)
    t3 = time.perf_counter()
    log(f"[D] generate() returned in {t3 - t2:.2f}s")

    wav = extract_wave_to_numpy(ov, result)
    log(f"[F] wav shape: {wav.shape}, dtype: {wav.dtype}")

    # Typical SpeechT5 sample rate for this pipeline
    sample_rate = 16000
    sf.write(out_wav, wav, sample_rate)
    log(f"[G] WAV written: {out_wav}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[X] fatal: {repr(e)}", flush=True)
        traceback.print_exc()
        sys.exit(2)
    finally:
        faulthandler.cancel_dump_traceback_later()
