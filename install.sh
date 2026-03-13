#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v apt >/dev/null 2>&1; then
  echo "This install.sh currently supports Debian/Ubuntu systems with apt."
  echo "Install these manually on your distro, then rerun:"
  echo "  python3 python3-pip python3-full portaudio19-dev espeak-ng"
  exit 1
fi

echo "[1/4] Installing system packages..."
sudo apt update
sudo apt install -y \
  python3 \
  python3-pip \
  python3-full \
  portaudio19-dev \
  espeak-ng \
  ocl-icd-libopencl1 \
  clinfo

if apt-cache show intel-opencl-icd >/dev/null 2>&1; then
  echo "[1/4] Installing Intel OpenCL runtime..."
  sudo apt install -y intel-opencl-icd
fi

echo "[2/4] Installing Python dependencies into the current Python environment..."
python3 -m pip install --break-system-packages -r requirements-linux.txt

echo "[3/4] Verifying Silero VAD..."
python3 -c "from silero_vad import load_silero_vad, VADIterator; model = load_silero_vad(); VADIterator(model, sampling_rate=16000); print('Silero VAD: OK')"

echo "[4/4] Showing active Python interpreter..."
python3 -c "import sys; print(sys.executable)"

echo ""
echo "OpenCL check:"
if command -v clinfo >/dev/null 2>&1; then
  clinfo >/dev/null 2>&1 && echo "  clinfo: OK" || echo "  clinfo: installed, but no OpenCL platform was detected"
fi

echo ""
echo "Installation complete."
echo ""
echo "Run the app with:"
echo "  ./run.sh"
echo ""
echo "Run the tests with:"
echo "  ./test.sh"
