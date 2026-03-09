from babelvox import BabelVox
import soundfile as sf

tts = BabelVox(
    device="NPU",
    precision="int8",
    use_cp_kv_cache=False,
    talker_buckets=[64],
    cache_dir="./ov_cache"
)

audio = tts.generate(
    text="Hello.",
    language="Spanish"
)

sf.write("babelvox_npu_test.wav", audio, 22050)
print("Audio generated.")
