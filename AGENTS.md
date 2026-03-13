# Robot Project Context

## Overview

This repository centers on `robot.py`, a cross-platform interactive assistant for Windows and Linux that combines:

- Local LLM inference through `openvino_genai.LLMPipeline`
- External OpenAI-compatible LLMs over HTTP
- Speech-to-text with either classic `openai-whisper` or OpenVINO Whisper
- Continuous VAD-driven auto-listen with Silero VAD
- Text-to-speech through multiple interchangeable backends
- Optional camera/panel UI with OpenVINO vision events
- Benchmarking, model catalog management, and a lightweight OpenAI-compatible server

The project is not structured as a package. The runtime is a single large CLI script.

## Primary Files

- `robot.py`: main application, interactive shell, model management, STT/TTS runtime, optional panel/camera runtime, OpenAI-compatible server, stats.
- `robot_config.json`: persisted runtime configuration for LLM/STT/TTS/camera behavior.
- `vision_models.json`: local vision model catalog.
- `ov_models/models.json`: local LLM model catalog.
- `tests/`: pytest suite covering runtime behavior and packaging sanity.

## Runtime Model

`robot.py` starts an interactive REPL. It loads config, initializes the native voice layer when available, preloads the configured Whisper backend, optionally restores a saved LLM, and then waits for commands or free-form prompts.

Core flow:

1. Load persisted catalogs and config from JSON files.
2. Preload the active Whisper backend.
3. Optionally auto-load the configured LLM.
4. Accept slash commands for configuration, model selection, panel/camera control, listen mode, benchmarking, and server startup.
5. For plain text input:
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
- Silero VAD for continuous auto-listen segmentation

The app now attempts to preload the active Whisper backend on startup to reduce first-use latency.

The `listen` flow records from the microphone until the user presses `SPACE`, transcribes the captured audio, then either:

- repeats it with TTS if `repeat=true`, or
- sends it through the active chat pipeline

Language selection is configurable. OpenVINO Whisper also supports model catalogs stored under `~/ov_models/whisper_models.json`.

## Camera / Vision / Panel

The runtime has a separate camera worker and an optional OpenCV panel window.

- `/panel opencv|qt` opens the control panel UI with the selected renderer.
- The camera/vision worker can run headless even when the panel is closed.
- `/camera on` can start camera processing without rendering the panel.
- Vision detection can trigger reactive TTS messages when people enter, leave, or disappear.
- If everyone leaves the camera while audio is actively playing, the app interrupts playback and says `me cayo`.
- After that interruption, the next `0 -> 1` presence transition is intentionally suppressed once so the assistant does not immediately greet again.
- Bounding boxes are only drawn when the panel is actually rendering.

## TTS Backends

Supported backends:

- `windows`: Windows SAPI via `win32com.client`
- `openvino`: `ov_genai.Text2SpeechPipeline`
- `kokoro`: `kokoro-onnx`
- `babelvox`: `babelvox`
- `espeakng`: external `espeak-ng` or `espeak` executable
- `tada`: Hume TADA voice-conditioned TTS using a reference audio clip plus transcript

`speak_text_backend()` dispatches to the selected engine. Several backends lazily install/import optional dependencies the first time they are used.

Important implementation details:

- OpenVINO TTS on GPU can run in an isolated worker process to avoid hangs locking the main process.
- Streaming TTS is supported while the LLM is still generating.
- `ESC` is used to interrupt generation or audio playback.
- Audio cancellation can also be triggered internally by vision logic, not only by keyboard input.
- `tada` is experimental and CPU-first in this repo; it currently follows the official PyTorch-based flow rather than an OpenVINO path.

## Persisted State

The app stores mutable state in JSON files. The main cache root is `~/ov_models`, while the assistant config lives beside the script.

Common files:

- `~/ov_models/models.json`
- `~/ov_models/whisper_models.json`
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
- `/tts_backend windows|openvino|kokoro|babelvox|espeakng|tada`
- `/tada_reference_record`, `/tada_reference_audio`, `/tada_reference_text`, `/tada_model`, `/tada_codec`, `/tada_device`, `/tada_language`
- `/panel`
- `/camera on|off`
- `/vision on|off`
- `/vision_events on|off`
- `/vision_models`, `/vision_select`, `/vision_model`, `/vision_labels`, `/vision_device`
- `/audio on|off`, `/audio_inputs`, `/audio_input_select`, `/audio_monitor on|off`
- `/listen`
- `/auto_listen on|off`
- `/config`
- `/whisper_models`, `/whisper_add`, `/whisper_select`
- `/openvino_tts_models`, `/openvino_tts_add`, `/openvino_tts_select`
- `/kokoro_models`, `/kokoro_select`
- `/babelvox_models`, `/babelvox_select`
- `/stats`, `/all_models`, `/clear_stats`, `/benchmark`
- `/start_server`

`/start_server` exposes an OpenAI-compatible endpoint on port `1311` backed by the active assistant state.

## Platform Assumptions

- The main app auto-detects the OS at startup.
- Native Windows voice selection is Windows-only.
- Linux uses `espeakng` as the default native fallback TTS backend.
- Audio interaction still assumes an interactive console.
- The panel is optional; camera/vision can still run headless.

## Dependencies

This repo provides OS-specific dependency files:

- `requirements-windows.txt`
- `requirements-linux.txt`
- `requirements-tada.txt` for the optional Hume TADA backend

Dependencies are still partly optional at runtime because several backends are lazy-loaded. Key packages referenced by the code:

- `openvino_genai`
- `huggingface_hub`
- `pywin32`
- `numpy`
- `sounddevice`
- `silero-vad`
- `openai-whisper`
- `torch`
- `transformers`
- `kokoro-onnx`
- `onnxruntime-openvino`
- `babelvox`

`requirements-tada.txt` is intentionally separate because the Hume TADA backend is optional in this repo.
The code also includes a small compatibility shim for older `huggingface_hub` progress-bar APIs when initializing TADA.

`espeakng` additionally requires the `espeak-ng` executable installed on the host.

## Tests

- The test suite lives in `tests/`.
- Use `pytest` as the single test entrypoint.
- Do not add new embedded test runners inside application scripts.

## Development Guidance For Codex

- Start by reading `robot.py`. It contains nearly all real behavior.
- Preserve JSON schema compatibility in `robot_config.json` and the `~/ov_models/*.json` catalogs.
- Be careful with blocking audio/model initialization paths; many features lazily initialize and cache runtime objects.
- Avoid breaking keyboard semantics:
  - `SPACE` starts/stops recording in listen mode
  - `ESC` interrupts generation or playback
- Preserve the separation between the headless camera worker and the optional `/panel` renderer.
- When changing vision event behavior, remember there is special logic around `me cayo` and suppressing the next join greeting.
- If changing TTS/STT/LLM config keys, update both defaults and normalization in `default_robot_config()` and `load_robot_config()`.
- If adding model catalogs, follow the existing `load_*`, `save_*`, `parse_*`, `*_status_line`, and `choose_*_interactive` patterns.
- For documentation, describe the project as an interactive local voice assistant for Intel/OpenVINO-focused Windows and Linux environments.

## Safe Assumptions

- `robot.py` is the production entry point.
- The repository is currently more of an application workspace than a polished distributable package.
- Users are likely experimenting with Intel CPU/GPU/NPU combinations and OpenVINO model compatibility.
