import os
import sys
import time
import traceback
import faulthandler
from pathlib import Path

# Periodically dump Python stack if it appears hung
faulthandler.enable()
faulthandler.dump_traceback_later(120, repeat=True)

# Make OpenVINO chatty
os.environ.setdefault("OV_LOG_LEVEL", "DEBUG")
os.environ.setdefault("OPENVINO_LOG_LEVEL", "DEBUG")


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    if len(sys.argv) < 2:
        print(
            "Usage: python speecht5_gpu_debug.py <model_dir> [text] [speaker_embedding.bin]",
            flush=True,
        )
        return 2

    model_dir = Path(sys.argv[1]).resolve()
    text = sys.argv[2] if len(sys.argv) >= 3 else "Hola."
    speaker_bin = Path(sys.argv[3]).resolve() if len(sys.argv) >= 4 else None

    log(f"[0] Python: {sys.version}")
    log(f"[1] model_dir: {model_dir}")
    log(f"[2] text: {text!r}")
    log(f"[3] speaker_embedding.bin: {speaker_bin if speaker_bin else '<none>'}")

    if not model_dir.exists():
        log("[X] model_dir does not exist")
        return 3

    try:
        import openvino as ov
        import openvino_genai
        import numpy as np
        import soundfile as sf
    except Exception as e:
        log(f"[X] import error: {e!r}")
        traceback.print_exc()
        return 4

    log(f"[4] openvino version: {getattr(ov, '__version__', 'unknown')}")
    log(f"[5] openvino_genai module: {getattr(openvino_genai, '__file__', 'unknown')}")

    try:
        core = ov.Core()
        devices = core.available_devices
        log(f"[6] available_devices: {devices}")
        try:
            log(f"[7] FULL_DEVICE_NAME GPU: {core.get_property('GPU', 'FULL_DEVICE_NAME')}")
        except Exception as e:
            log(f"[7] could not query GPU FULL_DEVICE_NAME: {e!r}")
    except Exception as e:
        log(f"[X] failed to init ov.Core(): {e!r}")
        traceback.print_exc()
        return 5

    xmls = sorted(model_dir.glob("*.xml"))
    bins = sorted(model_dir.glob("*.bin"))
    log(f"[8] xml files: {[x.name for x in xmls]}")
    log(f"[9] bin files: {[x.name for x in bins]}")

    # Step A: try to read and compile each IR on GPU individually.
    # This helps identify which submodel hangs before the GenAI pipeline is even constructed.
    for xml in xmls:
        try:
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

            # Keep object alive a moment and touch outputs to force initialization.
            outs = compiled.outputs
            log(f"    outputs: {[o.get_any_name() if hasattr(o, 'get_any_name') else str(o) for o in outs]}")
        except Exception as e:
            log(f"[X] compile failed for {xml.name}: {e!r}")
            traceback.print_exc()

    # Step B: construct Text2SpeechPipeline on GPU.
    try:
        log("[B] constructing Text2SpeechPipeline(model_dir, 'GPU') ...")
        t4 = time.perf_counter()
        pipe = openvino_genai.Text2SpeechPipeline(str(model_dir), "GPU")
        t5 = time.perf_counter()
        log(f"[B] pipeline constructed in {t5 - t4:.2f}s")
    except Exception as e:
        log(f"[X] pipeline construction failed: {e!r}")
        traceback.print_exc()
        return 6

    speaker_embedding = None
    if speaker_bin is not None:
        try:
            log("[C] loading speaker embedding ...")
            emb = np.fromfile(str(speaker_bin), dtype=np.float32).reshape(1, 512)
            speaker_embedding = ov.Tensor(emb)
            log("[C] speaker embedding loaded")
        except Exception as e:
            log(f"[X] failed to load speaker embedding: {e!r}")
            traceback.print_exc()
            return 7

    # Step C: generate the smallest possible output.
    try:
        log("[D] calling generate() ...")
        t6 = time.perf_counter()
        if speaker_embedding is None:
            result = pipe.generate(text)
        else:
            result = pipe.generate(text, speaker_embedding)
        t7 = time.perf_counter()
        log(f"[D] generate() returned in {t7 - t6:.2f}s")

        speech = result.speeches[0]
        wav = speech.data[0]
        log(f"[D] waveform length: {len(wav)} samples")
        out_path = model_dir / "speecht5_gpu_debug.wav"
        sf.write(str(out_path), wav, samplerate=16000)
        log(f"[D] wrote WAV to: {out_path}")
    except Exception as e:
        log(f"[X] generate failed: {e!r}")
        traceback.print_exc()
        return 8
    finally:
        faulthandler.cancel_dump_traceback_later()

    log("[OK] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
