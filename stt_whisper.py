import argparse
import importlib
import msvcrt
import subprocess
import sys
import time

np = None
sd = None


def _ask_install(display_name):
    while True:
        answer = input(
            f"Missing dependency '{display_name}'. Install now? [Y/n]: "
        ).strip().lower()
        if answer in ("s", "si", "y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("Invalid answer. Type 'y' or 'n'.")


def _ensure_dependency(module_name, pip_name, display_name):
    try:
        return importlib.import_module(module_name)
    except ImportError:
        if not _ask_install(display_name):
            raise SystemExit(
                f"Cancelled by user. Required dependency: {display_name}."
            )

        print(f"Installing dependency: {display_name}")
        cmd = [sys.executable, "-m", "pip", "install", pip_name]
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            raise SystemExit(
                f"Could not install '{pip_name}'. Run: pip install {pip_name}"
            )
        try:
            return importlib.import_module(module_name)
        except ImportError as exc:
            raise SystemExit(f"{display_name} is still unavailable.") from exc


SAMPLE_RATE = 16000
CHANNELS = 1
BLOCKSIZE = 1024


def _clear_keyboard_buffer():
    while msvcrt.kbhit():
        msvcrt.getwch()


def record_until_space(sample_rate=SAMPLE_RATE, channels=CHANNELS, blocksize=BLOCKSIZE):
    frames = []
    _clear_keyboard_buffer()

    def callback(indata, _frames, _time, status):
        if status:
            print(f"[audio] warning: {status}")
        frames.append(indata.copy())

    print("Listening... press SPACE to stop.")
    with sd.InputStream(
        samplerate=sample_rate,
        channels=channels,
        dtype="float32",
        blocksize=blocksize,
        callback=callback,
    ):
        while True:
            if msvcrt.kbhit():
                key = msvcrt.getwch()
                if key == " ":
                    break
            time.sleep(0.01)

    if not frames:
        return np.array([], dtype=np.float32)

    audio = np.concatenate(frames, axis=0).reshape(-1)
    return audio


def transcribe_audio(model, audio, language=None):
    if audio.size == 0:
        return ""

    result = model.transcribe(audio, fp16=False, language=language)
    return result.get("text", "").strip()


def main():
    global np
    global sd

    np = _ensure_dependency("numpy", "numpy", "NumPy")
    sd = _ensure_dependency("sounddevice", "sounddevice", "SoundDevice")
    whisper = _ensure_dependency("whisper", "openai-whisper", "Whisper")

    parser = argparse.ArgumentParser(
        description="Simple Speech-to-Text CLI using Whisper and a microphone."
    )
    parser.add_argument(
        "--model",
        default="base",
        help="Whisper model (tiny, base, small, medium, large). Default: base",
    )
    parser.add_argument(
        "--language",
        default="es",
        help="Expected language (for example: es, en). Use auto-detect if set to auto.",
    )
    args = parser.parse_args()

    language = None if args.language.lower() == "auto" else args.language

    print(f"Loading Whisper model '{args.model}'...")
    model = whisper.load_model(args.model)
    print("Model ready.\n")

    print("Commands:")
    print("  /listen -> listen from the microphone until SPACE is pressed")
    print("  /exit   -> exit\n")

    while True:
        cmd = input("Command: ").strip()

        if cmd == "/exit":
            print("Exiting...")
            return

        if cmd != "/listen":
            print("Invalid command. Use /listen or /exit.\n")
            continue

        audio = record_until_space()
        if audio.size == 0:
            print("No audio captured.\n")
            continue

        print("Transcribing...")
        text = transcribe_audio(model, audio, language=language)

        if not text:
            print("No speech detected.\n")
            continue

        print(f"Text: {text}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(0)
