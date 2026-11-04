# scripts/install.ps1
# Creates .venv in project root and installs requirements.
# Run from anywhere:  powershell -ExecutionPolicy Bypass -File scripts\install.ps1

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
Write-Host "[install] project root: $projectRoot"

$venv   = Join-Path $projectRoot ".venv"
$python = Join-Path $venv "Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Host "[install] creating venv..."
    python -m venv .venv
} else {
    Write-Host "[install] reusing existing venv"
}

Write-Host "[install] upgrading pip..."
& $python -m pip install --upgrade pip

Write-Host "[install] installing requirements..."
& $python -m pip install -r (Join-Path $projectRoot "requirements.txt")

Write-Host ""
Write-Host "[install] done."
Write-Host "Activate with:  .\.venv\Scripts\Activate.ps1"