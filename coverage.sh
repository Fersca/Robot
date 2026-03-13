#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "[1/2] Running pytest with coverage..."
python -m pytest \
  --cov=robot \
  --cov-report=term-missing \
  --cov-report=json:.tmp_tests/coverage.json \
  -q \
  -p no:cacheprovider \
  --basetemp .tmp_tests/pytest \
  --ignore pytest-cache-files-6mndw5k6 \
  --ignore pytest-cache-files-io38dnfk

echo
echo "[2/2] Counting code lines..."
python - <<'PY'
import json
from pathlib import Path

root = Path(".").resolve()
coverage_file = root / ".tmp_tests" / "coverage.json"

def is_ignored(path: Path) -> bool:
    parts = set(path.parts)
    return (
        ".tmp_tests" in parts
        or "__pycache__" in parts
        or ".git" in parts
        or ".venv" in parts
    )

def count_lines(path: Path) -> int:
    try:
        return len(path.read_text(encoding="utf-8", errors="ignore").splitlines())
    except Exception:
        return 0

robot_file = root / "robot.py"
tests = sorted(p for p in (root / "tests").rglob("*.py") if not is_ignored(p))
repo_py = sorted(
    p for p in root.rglob("*.py")
    if not is_ignored(p) and "tests" not in p.parts
)

robot_lines = count_lines(robot_file) if robot_file.exists() else 0
tests_lines = sum(count_lines(p) for p in tests)
repo_code_lines = sum(count_lines(p) for p in repo_py)
total_lines = repo_code_lines + tests_lines
coverage_pct = None

if coverage_file.exists():
    try:
        payload = json.loads(coverage_file.read_text(encoding="utf-8"))
        coverage_pct = float(payload.get("totals", {}).get("percent_covered", 0.0))
    except Exception:
        coverage_pct = None

print(f"robot.py lines: {robot_lines}")
print(f"system python lines (excluding tests): {repo_code_lines}")
print(f"test python lines: {tests_lines}")
print(f"total python lines: {total_lines}")
print()
print("Summary:")
print("--------")
if coverage_pct is None:
    print("test coverage: unavailable")
else:
    print(f"test coverage: {coverage_pct:.2f}%")
print(f"system code lines: {repo_code_lines}")
print(f"test code lines: {tests_lines}")
print(f"total code lines: {total_lines}")
PY
