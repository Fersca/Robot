"""
Minimal BabelVox NPU example for Intel NPU via OpenVINO.

Prereqs:
    pip install babelvox soundfile

What this does:
    - Loads BabelVox on device="NPU"
    - Uses INT8 + code predictor KV cache + talker buckets
    - Writes one WAV file to disk

Notes:
    - First run on NPU can take a long time because OpenVINO compiles and caches models.
    - This example follows the public BabelVox README usage.
"""

from pathlib import Path
import sys

import soundfile as sf
from babelvox import BabelVox


def main() -> int:
    out_path = Path("babelvox_npu_test.wav")
    cache_dir = Path("./ov_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    text = (
        "Hello Fernando. This is a minimal BabelVox test running "
        "on the Intel NPU through OpenVINO."
    )

    try:
        # Minimal NPU configuration shown in the BabelVox README.
        tts = BabelVox(
            device="NPU",
            precision="int8",
            use_cp_kv_cache=True,
            talker_buckets=[64, 128, 256],
            cache_dir=str(cache_dir),
        )

        # Supported languages in the README include Spanish.
        wav, sr = tts.generate(text, language="Spanish")
        sf.write(out_path, wav, sr)

        print(f"OK: audio generated at {out_path.resolve()}")
        print(f"Sample rate: {sr}")
        print("Note: the first NPU run can take a while because of initial compilation.")
        return 0

    except Exception as exc:
        print("ERROR while running BabelVox on NPU:", exc, file=sys.stderr)
        print("Check:", file=sys.stderr)
        print("  1) that you have an Intel Core Ultra/Lunar Lake or another NPU-capable system,", file=sys.stderr)
        print("  2) that NPU drivers are installed,", file=sys.stderr)
        print("  3) that OpenVINO/BabelVox are up to date.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
