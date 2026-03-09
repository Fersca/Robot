# Robot

A cross-platform desktop conversational assistant for Windows and Linux that combines local or external LLMs, voice input, voice output, and utilities for working with OpenVINO models on CPU, GPU, and NPU.

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

- Windows SAPI on Windows
- Parler-TTS on CPU
- OpenVINO Text2SpeechPipeline
- Kokoro ONNX
- BabelVox
- eSpeak NG

## Important Files

- [`robot.py`](./robot.py): main application
- [`robot_config.json`](./robot_config.json): persisted configuration
- [`ov_models/models.json`](./ov_models/models.json): LLM model catalog

## Screenshots

### Phi-4 Loaded On The NPU

Example of the application after loading the Phi-4 model on the Intel NPU.

![Phi-4 loaded on NPU](./screenshots/phi_4_npu.png)

### NPU Usage

Example showing NPU usage while the assistant is running a model.

![NPU usage](./screenshots/npu_usage.png)

### Model List

The model selection list used to choose which LLM to load.

![Model list](./screenshots/model_list.png)

### Chat Example

Example of a chat session in the interactive console.

![Chat example](./screenshots/chat_example.png)

## Requirements

The repository now ships OS-specific dependency files:

- [`requirements-windows.txt`](./requirements-windows.txt)
- [`requirements-linux.txt`](./requirements-linux.txt)

Also:

- `espeakng` requires the `espeak-ng` executable
- Linux also needs the system libraries required by `sounddevice` and PortAudio
- gated or private Hugging Face models use `~/ov_models/hf_auth.json`

Expected Hugging Face token format:

```json
{"hf_token":"hf_xxx"}
```

## Quick Start

Create and activate a compatible Python environment, install the dependencies for your OS, and then run the app.

### Windows

```powershell
pip install -r .\requirements-windows.txt
python .\robot.py
```

### Linux

Install `espeak-ng` and the PortAudio development/runtime packages with your package manager first, then:

```bash
pip install -r ./requirements-linux.txt
python ./robot.py
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

## Tests

The automated test suite lives under [`tests`](./tests) and uses `pytest`.

Run:

```bash
pytest
```
