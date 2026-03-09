import json
import importlib
import gc
import msvcrt
import os
import subprocess
import sys
import time
import win32com.client

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
DEFAULT_WHISPER_MODEL = "base"

def listar_voces(voces):
    for i in range(voces.Count):
        voz = voces.Item(i)
        print(f"{i}: {voz.GetDescription()}")

def cargar_config():
    if not os.path.exists(CONFIG_FILE):
        return None

    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Could not read {CONFIG_FILE}: {e}")
        return None

def guardar_config(config):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"Configuration saved to {CONFIG_FILE}\n")
    except Exception as e:
        print(f"Could not save configuration: {e}\n")

def preguntar_instalacion(nombre):
    while True:
        valor = input(f"Missing dependency '{nombre}'. Install now? [Y/n]: ").strip().lower()
        if valor in ("s", "si", "y", "yes"):
            return True
        if valor in ("n", "no"):
            return False
        print("Invalid answer. Type 'y' or 'n'.")

def asegurar_dependencia(module_name, pip_name, display_name):
    try:
        return importlib.import_module(module_name)
    except ImportError:
        if not preguntar_instalacion(display_name):
            print(f"Cancelled. Required dependency: {display_name}\n")
            return None

        print(f"Installing dependency: {display_name}")
        result = subprocess.run([sys.executable, "-m", "pip", "install", pip_name], check=False)
        if result.returncode != 0:
            print(f"Could not install '{pip_name}'.\n")
            return None

        try:
            return importlib.import_module(module_name)
        except ImportError:
            print(f"{display_name} is still unavailable.\n")
            return None

def limpiar_buffer_teclado():
    while msvcrt.kbhit():
        msvcrt.getwch()

def grabar_hasta_espacio(sd_mod, np_mod, sample_rate=16000, channels=1, blocksize=1024):
    frames = []
    limpiar_buffer_teclado()

    def callback(indata, _frames, _time, status):
        if status:
            print(f"[audio] warning: {status}")
        frames.append(indata.copy())

    print("Listening... press SPACE to stop.")
    with sd_mod.InputStream(
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
        return np_mod.array([], dtype=np_mod.float32)
    return np_mod.concatenate(frames, axis=0).reshape(-1)

def transcribir_audio(whisper_model, audio):
    if audio.size == 0:
        return ""
    result = whisper_model.transcribe(audio, fp16=False)
    return result.get("text", "").strip()

def ruta_modelo_local_whisper(whisper_mod, model_name):
    modelos = getattr(whisper_mod, "_MODELS", {})
    if model_name not in modelos:
        return None
    url = modelos[model_name]
    return os.path.join(os.path.expanduser("~"), ".cache", "whisper", os.path.basename(url))

def inicializar_stt(runtime, config):
    if runtime.get("numpy") is None:
        runtime["numpy"] = asegurar_dependencia("numpy", "numpy", "NumPy")
        if runtime["numpy"] is None:
            return False

    if runtime.get("sounddevice") is None:
        runtime["sounddevice"] = asegurar_dependencia("sounddevice", "sounddevice", "SoundDevice")
        if runtime["sounddevice"] is None:
            return False

    if runtime.get("whisper") is None:
        runtime["whisper"] = asegurar_dependencia("whisper", "openai-whisper", "Whisper")
        if runtime["whisper"] is None:
            return False

    model_name = config.get("whisper_model", DEFAULT_WHISPER_MODEL)
    if model_name not in WHISPER_MODELS:
        model_name = DEFAULT_WHISPER_MODEL
        config["whisper_model"] = model_name

    if runtime.get("model_name") != model_name or runtime.get("model") is None:
        if runtime.get("model") is not None and runtime.get("model_name"):
            print(f"Releasing previous model from memory: {runtime['model_name']}")
            runtime["model"] = None
            runtime["model_name"] = None
            gc.collect()

        local_path = ruta_modelo_local_whisper(runtime["whisper"], model_name)
        if local_path and os.path.exists(local_path):
            print(f"Whisper model '{model_name}' is already downloaded.")
        else:
            print(f"Whisper model '{model_name}' is not downloaded. Downloading...")

        try:
            print(f"Loading Whisper model '{model_name}'...")
            runtime["model"] = runtime["whisper"].load_model(model_name)
            runtime["model_name"] = model_name
            print(f"Whisper model active: {model_name}\n")
        except Exception as e:
            print(f"Could not load Whisper model '{model_name}': {e}\n")
            runtime["model"] = None
            runtime["model_name"] = None
            return False

    return True

def elegir_voz_inicial(voces):
    print("Available voices:\n")
    listar_voces(voces)

    while True:
        try:
            opcion = int(input("Choose the voice number: "))
            if 0 <= opcion < voces.Count:
                return opcion
            print("Number out of range.")
        except ValueError:
            print("Enter a valid number.")

def cambiar_voz(speaker, voces, config):
    print("\nAvailable voices:\n")
    listar_voces(voces)

    while True:
        valor = input("Choose the voice number: ").strip()

        try:
            opcion = int(valor)
            if 0 <= opcion < voces.Count:
                speaker.Voice = voces.Item(opcion)
                config["voice_index"] = opcion
                print(f"Voice changed to: {voces.Item(opcion).GetDescription()}\n")
                guardar_config(config)
                return
            else:
                print("Number out of range.")
        except ValueError:
            print("Enter a valid number.")

def mostrar_config(config, speaker):
    print("\nCurrent configuration:")
    print(f"  voice   : {speaker.Voice.GetDescription()}")
    print(f"  rate    : {config['rate']}")
    print(f"  volume  : {config['volume']}")
    print(f"  silence : {config['silence']}")
    print(f"  whisper : {config.get('whisper_model', DEFAULT_WHISPER_MODEL)}")
    print()

def configurar(config, speaker, runtime):
    print("\nCurrent config:")
    mostrar_config(config, speaker)

    print("You can change these values:")
    print("  rate    -> voice speed (for example: -5 to 5)")
    print("  volume  -> volume (0 to 100)")
    print("  silence -> initial silence in ms")
    print("  whisper -> Speech-to-Text model")
    print("  show    -> show current config")
    print("  exit    -> exit /config\n")

    cambios = False
    while True:
        clave = input("Parameter (rate / volume / silence / whisper / show / exit): ").strip().lower()

        if clave == "exit":
            if cambios:
                guardar_config(config)
            print()
            return

        if clave == "show":
            mostrar_config(config, speaker)
            continue

        if clave == "whisper":
            print("\nAvailable Whisper models:\n")
            for i, model in enumerate(WHISPER_MODELS):
                actual = " (current)" if config.get("whisper_model", DEFAULT_WHISPER_MODEL) == model else ""
                print(f"{i}: {model}{actual}")
            print()

            valor = input("Choose model number or 'cancel': ").strip().lower()
            if valor == "cancel":
                continue
            try:
                opcion = int(valor)
                if not (0 <= opcion < len(WHISPER_MODELS)):
                    print("Number out of range.")
                    continue
                elegido = WHISPER_MODELS[opcion]
                anterior = config.get("whisper_model", DEFAULT_WHISPER_MODEL)
                config["whisper_model"] = elegido
                print(f"Whisper model changed to: {elegido}")
                if elegido != anterior:
                    print("Applying model change now...")
                if not inicializar_stt(runtime, config):
                    config["whisper_model"] = anterior
                    print(f"Previous model restored: {anterior}\n")
                    continue
                cambios = True
                continue
            except ValueError:
                print("Enter a valid number.")
                continue

        if clave not in ("rate", "volume", "silence"):
            print("Invalid parameter.")
            continue

        valor = input(f"New value for {clave}: ").strip()

        try:
            valor_int = int(valor)

            if clave == "rate":
                config["rate"] = valor_int
                speaker.Rate = valor_int
                print(f"rate updated to {valor_int}")
                cambios = True

            elif clave == "volume":
                if not (0 <= valor_int <= 100):
                    print("volume must be between 0 and 100")
                    continue
                config["volume"] = valor_int
                speaker.Volume = valor_int
                print(f"volume updated to {valor_int}")
                cambios = True

            elif clave == "silence":
                if valor_int < 0:
                    print("silence cannot be negative")
                    continue
                config["silence"] = valor_int
                print(f"silence updated to {valor_int}")
                cambios = True

        except ValueError:
            print("Enter a valid number.")

def hablar(speaker, texto, config):
    xml = f'<speak><silence msec="{config["silence"]}"/>{texto}</speak>'
    speaker.Speak(xml, 1)  # 1 = SPF_IS_XML

def aplicar_config(speaker, voces, config):
    voice_index = config.get("voice_index", 0)
    if 0 <= voice_index < voces.Count:
        speaker.Voice = voces.Item(voice_index)
    else:
        print("The saved voice no longer exists. Voice 0 will be used.")
        config["voice_index"] = 0
        speaker.Voice = voces.Item(0)

    speaker.Rate = int(config.get("rate", -2))
    speaker.Volume = int(config.get("volume", 100))

def main():
    speaker = win32com.client.Dispatch("SAPI.SpVoice")
    voces = speaker.GetVoices()

    if voces.Count == 0:
        print("No voices were found.")
        return

    config = cargar_config()

    if config is None:
        config = {
            "voice_index": elegir_voz_inicial(voces),
            "rate": -2,
            "volume": 100,
            "silence": 600,
            "whisper_model": DEFAULT_WHISPER_MODEL,
        }
        guardar_config(config)
    else:
        print(f"Configuration loaded from {CONFIG_FILE}\n")
        if "whisper_model" not in config:
            config["whisper_model"] = DEFAULT_WHISPER_MODEL
            guardar_config(config)

    aplicar_config(speaker, voces, config)
    stt_runtime = {"numpy": None, "sounddevice": None, "whisper": None, "model": None, "model_name": None}

    print("Type text to speak.")
    print("Commands:")
    print("  /voices  -> list voices and change the current one")
    print("  /config  -> change configuration")
    print("  /listen  -> listen through the microphone and transcribe")
    print("  /exit    -> exit\n")

    while True:
        texto = input("Text: ").strip()

        if texto == "/exit":
            print("Exiting...")
            break

        if texto == "/voices":
            cambiar_voz(speaker, voces, config)
            continue

        if texto == "/config":
            configurar(config, speaker, stt_runtime)
            continue

        if texto == "/listen":
            if not inicializar_stt(stt_runtime, config):
                continue
            audio = grabar_hasta_espacio(stt_runtime["sounddevice"], stt_runtime["numpy"])
            if audio.size == 0:
                print("No audio captured.\n")
                continue
            print("Transcribing...")
            texto = transcribir_audio(stt_runtime["model"], audio)
            if not texto:
                print("No speech detected.\n")
                continue
            print(f"Text: {texto}\n")
            hablar(speaker, texto, config)
            continue

        if not texto:
            continue

        hablar(speaker, texto, config)

class _FakeVoice:
    def __init__(self, description):
        self._description = description

    def GetDescription(self):
        return self._description

class _FakeVoices:
    def __init__(self, items):
        self._items = items
        self.Count = len(items)

    def Item(self, index):
        return self._items[index]

class _FakeSpeaker:
    def __init__(self, voices):
        self._voices = voices
        self.Voice = voices.Item(0)
        self.Rate = 0
        self.Volume = 100
        self.last_spoken = None
        self.last_flags = None

    def GetVoices(self):
        return self._voices

    def Speak(self, text, flags):
        self.last_spoken = text
        self.last_flags = flags

def _run_tests():
    import unittest
    from unittest.mock import patch

    class VocesTests(unittest.TestCase):
        def test_hablar_construye_xml_con_silencio(self):
            voces = _FakeVoices([_FakeVoice("A")])
            speaker = _FakeSpeaker(voces)
            config = {"silence": 600}

            hablar(speaker, "hola", config)

            self.assertEqual(
                speaker.last_spoken,
                '<speak><silence msec="600"/>hola</speak>',
            )
            self.assertEqual(speaker.last_flags, 1)

        def test_aplicar_config_aplica_indices_y_parametros(self):
            voces = _FakeVoices([_FakeVoice("A"), _FakeVoice("B")])
            speaker = _FakeSpeaker(voces)
            config = {"voice_index": 1, "rate": -3, "volume": 80, "silence": 100}

            aplicar_config(speaker, voces, config)

            self.assertEqual(speaker.Voice.GetDescription(), "B")
            self.assertEqual(speaker.Rate, -3)
            self.assertEqual(speaker.Volume, 80)

        def test_aplicar_config_repara_voice_index_fuera_de_rango(self):
            voces = _FakeVoices([_FakeVoice("A"), _FakeVoice("B")])
            speaker = _FakeSpeaker(voces)
            config = {"voice_index": 999, "rate": 1, "volume": 50, "silence": 100}

            aplicar_config(speaker, voces, config)

            self.assertEqual(config["voice_index"], 0)
            self.assertEqual(speaker.Voice.GetDescription(), "A")

        def test_guardar_y_cargar_config_roundtrip(self):
            base_tmp = os.path.join(os.getcwd(), ".tmp_tests")
            os.makedirs(base_tmp, exist_ok=True)
            ruta = os.path.join(base_tmp, "config_test.json")
            try:
                if os.path.exists(ruta):
                    os.remove(ruta)
                data = {"voice_index": 1, "rate": -2, "volume": 100, "silence": 600}
                with patch(__name__ + ".CONFIG_FILE", ruta):
                    guardar_config(data)
                    cargado = cargar_config()
                self.assertEqual(cargado, data)
            finally:
                if os.path.exists(ruta):
                    os.remove(ruta)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(VocesTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    if "--test" in sys.argv:
        raise SystemExit(_run_tests())
    main()
