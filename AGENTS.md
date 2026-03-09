# Robot Project Context

## Overview

This repository centers on `robot.py`, a cross-platform interactive assistant for Windows and Linux that combines:

- Local LLM inference through `openvino_genai.LLMPipeline`
- External OpenAI-compatible LLMs over HTTP
- Speech-to-text with either classic `openai-whisper` or OpenVINO Whisper
- Text-to-speech through multiple interchangeable backends
- Benchmarking, model catalog management, and a lightweight OpenAI-compatible server

The project is not structured as a package. The runtime is a single large CLI script with helper/prototype scripts beside it.

## Primary Files

- `robot.py`: main application, interactive shell, model management, STT/TTS runtime, OpenAI-compatible server, stats.
- `robot_config.json`: persisted runtime configuration for LLM/STT/TTS behavior.
- `ov_models/models.json`: local catalog of LLM models.
- `tests/`: pytest suite covering cross-platform runtime behavior and packaging sanity.

## Runtime Model

`robot.py` starts an interactive REPL. It loads config, optionally restores a saved LLM, initializes the native voice layer when available, then waits for commands or free-form prompts.

Core flow:

1. Load persisted catalogs and config from JSON files.
2. Optionally auto-load the configured LLM.
3. Accept slash commands for configuration, model selection, listen mode, benchmarking, and server startup.
4. For plain text input:
   - If `repeat` is enabled, send text directly to TTS.
   - Otherwise build a chat prompt from history plus optional system prompt.
   - Stream model output to console.
   - Optionally stream partial output to TTS while generation continues.

## LLM Backends

Two LLM modes exist:

- `local`: OpenVINO GenAI model loaded from the local model catalog and executed on CPU/GPU/NPU/AUTO.
- `external`: OpenAI-compatible HTTP endpoint, default base URL `http://localhost:1234`.

`current_model_repo`, `llm_backend`, `llm_device`, and `llm_performance_hint` are persisted in `robot_config.json`.

The local model catalog is backed by `ov_models/models.json`. Missing local models are downloaded through `huggingface_hub.snapshot_download()`.

## STT Backends

Speech-to-text supports:

- Classic Whisper through `openai-whisper`
- OpenVINO Whisper through `ov_genai.WhisperPipeline`

The `listen` flow records from the microphone until the user presses `SPACE`, transcribes the captured audio, then either:

- repeats it with TTS if `repeat=true`, or
- sends it through the active chat pipeline

Language selection is configurable. OpenVINO Whisper also supports model catalogs stored under `~/ov_models/whisper_models.json`.

## TTS Backends

Supported backends:

- `windows`: Windows SAPI via `win32com.client`
- `parler`: CPU PyTorch Parler-TTS
- `openvino`: `ov_genai.Text2SpeechPipeline`
- `kokoro`: `kokoro-onnx`
- `babelvox`: `babelvox`
- `espeakng`: external `espeak-ng` or `espeak` executable

`speak_text_backend()` dispatches to the selected engine. Several backends lazily install/import optional dependencies the first time they are used.

Important implementation details:

- OpenVINO TTS on GPU can run in an isolated worker process to avoid hangs locking the main process.
- Streaming TTS is supported while the LLM is still generating text.
- `ESC` is used to interrupt generation or audio playback.

## Persisted State

The app stores mutable state in JSON files. The main cache root is `~/ov_models`, while the assistant config lives beside the script.

Common files:

- `~/ov_models/models.json`
- `~/ov_models/whisper_models.json`
- `~/ov_models/parler_models.json`
- `~/ov_models/openvino_tts_models.json`
- `~/ov_models/kokoro_models.json`
- `~/ov_models/babelvox_models.json`
- `~/ov_models/stats.json`
- `~/ov_models/device_compat.json`
- `~/ov_models/benchmark_prompts.json`
- `~/ov_models/hf_auth.json`
- `robot_config.json`

`hf_auth.json` is expected to contain a Hugging Face token as:

```json
{"hf_token":"hf_xxx"}
```

## User Commands

The main command surface includes:

- `/models`, `/add_model`, `/delete`
- `/llm_backend local|external`
- `/tts_backend windows|parler|openvino|kokoro|babelvox|espeakng`
- `/listen`
- `/config`
- `/whisper_models`, `/whisper_add`, `/whisper_select`
- `/parler_models`, `/parler_add`, `/parler_select`
- `/openvino_tts_models`, `/openvino_tts_add`, `/openvino_tts_select`
- `/kokoro_models`, `/kokoro_select`
- `/babelvox_models`, `/babelvox_select`
- `/stats`, `/all_models`, `/clear_stats`, `/benchmark`
- `/start_server`

`/start_server` exposes an OpenAI-compatible endpoint on port `1311` backed by the active assistant state.

## Platform Assumptions

This project now targets both Windows and Linux, but not every helper script is equally portable.

- The main app auto-detects the OS at startup.
- Native Windows voice selection is Windows-only.
- Linux uses `espeakng` as the default native fallback TTS backend.
- Audio interaction still assumes an interactive console.
- Some helper/debug scripts remain Intel/OpenVINO-focused experiments.

## Dependencies

This repo now provides OS-specific dependency files:

- `requirements-windows.txt`
- `requirements-linux.txt`

Dependencies are still partly optional at runtime because several backends are lazy-loaded. Key packages referenced by the code:

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

`espeakng` additionally requires the `espeak-ng` executable installed on the host.

## Tests

- The test suite lives in `tests/`.
- Use `pytest` as the single test entrypoint.
- Do not add new embedded test runners inside application scripts.

## Development Guidance For Codex

- Start by reading `robot.py`. It contains nearly all real behavior.
- Treat sibling scripts as experiments, diagnostics, or earlier prototypes unless the user explicitly wants them updated too.
- Preserve JSON schema compatibility in `robot_config.json` and the `~/ov_models/*.json` catalogs.
- Be careful with blocking audio/model initialization paths; many features lazily initialize and cache runtime objects.
- Avoid breaking keyboard semantics:
  - `SPACE` starts/stops recording in listen mode
  - `ESC` interrupts generation or playback
- If changing TTS/STT/LLM config keys, update both defaults and normalization in `default_robot_config()` and `load_robot_config()`.
- If adding model catalogs, follow the existing `load_*`, `save_*`, `parse_*`, `*_status_line`, and `choose_*_interactive` patterns.
- For documentation, describe the project as an interactive local voice assistant for Intel/OpenVINO-focused Windows and Linux environments.

## Safe Assumptions

- `robot.py` is the production entry point.
- The repository is currently more of an application workspace than a polished distributable package.
- Users are likely experimenting with Intel CPU/GPU/NPU combinations and OpenVINO model compatibility.
