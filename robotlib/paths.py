from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RobotPaths:
    cache_dir: Path
    stats_file: Path
    auth_file: Path
    benchmark_prompts_file: Path
    device_compat_file: Path
    models_file: Path
    robot_config_file: Path
    whisper_ov_models_file: Path
    ov_tts_models_file: Path
    kokoro_models_file: Path
    babelvox_models_file: Path
    vision_models_file: Path
    vision_event_responses_file: Path


def build_robot_paths(base_dir: Path) -> RobotPaths:
    cache_dir = Path.home() / "ov_models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return RobotPaths(
        cache_dir=cache_dir,
        stats_file=cache_dir / "stats.json",
        auth_file=cache_dir / "hf_auth.json",
        benchmark_prompts_file=cache_dir / "benchmark_prompts.json",
        device_compat_file=cache_dir / "device_compat.json",
        models_file=cache_dir / "models.json",
        robot_config_file=base_dir / "robot_config.json",
        whisper_ov_models_file=cache_dir / "whisper_models.json",
        ov_tts_models_file=cache_dir / "openvino_tts_models.json",
        kokoro_models_file=cache_dir / "kokoro_models.json",
        babelvox_models_file=cache_dir / "babelvox_models.json",
        vision_models_file=base_dir / "vision_models.json",
        vision_event_responses_file=base_dir / "vision_event_responses.json",
    )

