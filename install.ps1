$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

$python = Get-Command py -ErrorAction SilentlyContinue
if ($python) {
  $pythonCmd = "py"
} else {
  $pythonCmd = "python"
}

Write-Host "[1/3] Installing Python dependencies into the current Python environment..."
& $pythonCmd -m pip install -r requirements-windows.txt
if ($LASTEXITCODE -ne 0) {
  throw "Dependency installation failed."
}

Write-Host "[2/3] Verifying Silero VAD..."
& $pythonCmd -c "from silero_vad import load_silero_vad, VADIterator; model = load_silero_vad(); VADIterator(model, sampling_rate=16000); print('Silero VAD: OK')"
if ($LASTEXITCODE -ne 0) {
  throw "Silero VAD verification failed."
}

Write-Host "[3/3] Showing active Python interpreter..."
& $pythonCmd -c "import sys; print(sys.executable)"
if ($LASTEXITCODE -ne 0) {
  throw "Could not resolve active Python interpreter."
}

Write-Host ""
Write-Host "Installation complete."
Write-Host ""
Write-Host "Run the app with:"
Write-Host "  .\run.ps1"
Write-Host ""
Write-Host "Run the tests with:"
Write-Host "  .\test.ps1"
