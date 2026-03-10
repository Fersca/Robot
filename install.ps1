$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

$python = Get-Command py -ErrorAction SilentlyContinue
if ($python) {
  $pythonCmd = "py"
} else {
  $pythonCmd = "python"
}

Write-Host "[1/5] Creating virtual environment..."
& $pythonCmd -m venv .venv

Write-Host "[2/5] Activating virtual environment..."
$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
  throw "Virtual environment Python was not created at $venvPython"
}

Write-Host "[3/5] Upgrading pip..."
& $venvPython -m pip install --upgrade pip

Write-Host "[4/5] Installing Python dependencies..."
& $venvPython -m pip install -r requirements-windows.txt

Write-Host "[5/5] Verifying Silero VAD..."
& $venvPython -c "from silero_vad import load_silero_vad, VADIterator; model = load_silero_vad(); VADIterator(model, sampling_rate=16000); print('Silero VAD: OK')"

Write-Host ""
Write-Host "Installation complete."
Write-Host ""
Write-Host "Activate the environment with:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Run the app with:"
Write-Host "  .\run.ps1"
Write-Host ""
Write-Host "Run the tests with:"
Write-Host "  .\test.ps1"
