#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v apt >/dev/null 2>&1; then
  echo "This install.sh currently supports Debian/Ubuntu systems with apt."
  echo "Install these manually on your distro, then rerun:"
  echo "  python3 python3-venv python3-pip python3-full portaudio19-dev espeak-ng"
  exit 1
fi

echo "[1/5] Installing system packages..."
sudo apt update
sudo apt install -y \
  python3 \
  python3-venv \
  python3-pip \
  python3-full \
  portaudio19-dev \
  espeak-ng \
  ocl-icd-libopencl1 \
  clinfo

if apt-cache show intel-opencl-icd >/dev/null 2>&1; then
  echo "[1/5] Installing Intel OpenCL runtime..."
  sudo apt install -y intel-opencl-icd
fi

echo "[2/5] Creating virtual environment..."
python3 -m venv .venv

echo "[3/5] Activating virtual environment..."
source .venv/bin/activate

echo "[4/5] Installing Python dependencies..."
python -m pip install --upgrade pip
pip install -r requirements-linux.txt

echo "[5/5] Verifying Silero VAD..."
python -c "from silero_vad import load_silero_vad, VADIterator; model = load_silero_vad(); VADIterator(model, sampling_rate=16000); print('Silero VAD: OK')"

echo ""
echo "OpenCL check:"
if command -v clinfo >/dev/null 2>&1; then
  clinfo >/dev/null 2>&1 && echo "  clinfo: OK" || echo "  clinfo: installed, but no OpenCL platform was detected"
fi

echo ""
echo "Installation complete."
echo ""
echo "Activate the environment with:"
echo "  source .venv/bin/activate"
echo ""
echo "Run the app with:"
echo "  ./run.sh"
echo ""
echo "Run the tests with:"
echo "  ./test.sh"
