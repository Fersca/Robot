# Robot

A Windows-oriented desktop conversational assistant that combines local or external LLMs, voice input, voice output, and utilities for working with OpenVINO models on CPU, GPU, and NPU.

The core of the project is [`robot.py`](./robot.py). From an interactive console it can:

- load local LLMs from Hugging Face using OpenVINO GenAI
- use an external OpenAI-compatible model
- transcribe microphone input with classic Whisper or OpenVINO Whisper
- speak responses through multiple TTS backends
- benchmark models and record metrics
- expose an OpenAI-compatible endpoint at `http://0.0.0.0:1311/v1/chat/completions`

## What The Project Does

`robot.py` works as an interactive REPL:

1. It loads configuration from [`robot_config.json`](./robot_config.json).
2. It loads model catalogs from `~/ov_models`.
3. It tries to restore the previously used LLM.
4. It waits for commands (`/models`, `/listen`, `/config`, etc.) or regular prompts.
5. When it receives text:
   - it repeats it through TTS if `repeat=true`
   - otherwise it sends it to the active LLM and plays the response if audio is enabled

It also supports continuous listening: once you enter `/listen`, `SPACE` starts or stops recording and `ESC` exits listen mode.

## Supported Backends

### LLM

- Local via `openvino_genai.LLMPipeline`
- External via an OpenAI-compatible API

### Speech-to-Text

- `openai-whisper`
- `openvino_genai.WhisperPipeline`

### Text-to-Speech

- Windows SAPI
- Parler-TTS on CPU
- OpenVINO Text2SpeechPipeline
- Kokoro ONNX
- BabelVox
- eSpeak NG

## Important Files

- [`robot.py`](./robot.py): main application
- [`robot_config.json`](./robot_config.json): persisted configuration
- [`ov_models/models.json`](./ov_models/models.json): LLM model catalog
- [`voces.py`](./voces.py): older, simpler prototype version
- [`stt_whisper.py`](./stt_whisper.py): minimal Whisper STT CLI
- [`speecht5_gpu_debug.py`](./speecht5_gpu_debug.py): SpeechT5/OpenVINO GPU debug script
- [`speecht5_gpu_debug_patched.py`](./speecht5_gpu_debug_patched.py): patched variant of that debug script
- [`babelvox_npu_minimal.py`](./babelvox_npu_minimal.py): minimal BabelVox NPU test

## Requirements

The project currently does not include a `requirements.txt` or formal packaging. Dependency loading is mixed: some packages are required, while others are imported on demand depending on the selected backend.

Dependencies referenced by the code:

- `openvino_genai`
- `huggingface_hub`
- `pywin32`
- `numpy`
- `sounddevice`
- `openai-whisper`
- `torch`
- `transformers`
- `parler-tts`
- `kokoro-onnx`
- `onnxruntime-openvino`
- `babelvox`

Also:

- Windows is the primary target environment
- `espeakng` requires the `espeak-ng` executable
- gated or private Hugging Face models use `~/ov_models/hf_auth.json`

Expected Hugging Face token format:

```json
{"hf_token":"hf_xxx"}
```

## Quick Start

Create and activate a compatible Python environment, install the dependencies required for the backend you want to use, and then run:

```powershell
python .\robot.py
```

For a first session, the recommended flow is:

1. Run `/models`
2. Choose a local LLM or configure `/llm_backend external`
3. Adjust audio and STT settings with `/config`
4. Try `/listen` or type prompts directly

## Main Commands

```text
/help
/models
/add_model
/delete
/config
/voices
/listen
/llm_backend local|external
/tts_backend windows|parler|openvino|kokoro|babelvox|espeakng
/whisper_models
/whisper_add
/whisper_select
/parler_models
/parler_add
/parler_select
/openvino_tts_models
/openvino_tts_add
/openvino_tts_select
/kokoro_models
/kokoro_select
/babelvox_models
/babelvox_select
/stats
/all_models
/clear_stats
/benchmark
/start_server
/exit
```

## Persisted Configuration

The main configuration lives in [`robot_config.json`](./robot_config.json). Among other things, it stores:

- LLM backend (`local` or `external`)
- current model and device (`CPU`, `GPU`, `NPU`, `AUTO`)
- TTS backend
- Whisper model and language
- audio on/off
- TTS streaming
- system prompt
- `max_new_tokens`

Catalogs, metrics, and compatibility data live under `~/ov_models`.

## Project Status

This repository currently looks more like an advanced experimentation workspace than a fully packaged product. Most of the core logic is concentrated in one large file, and several helper scripts coexist for testing STT, BabelVox, and SpeechT5/OpenVINO behavior.

If the goal is to publish it in a more solid shape on GitHub, the next natural steps would be:

- add a `requirements.txt` or `pyproject.toml`
- split `robot.py` into modules
- document installation per backend
- explicitly version the `ov_models` content
- add tests for `robot.py`, not only for the `voces.py` prototype
