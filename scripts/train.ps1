# scripts/train.ps1
param(
    [int]$Epochs     = 40,
    [int]$BatchSize  = 128,
    [double]$Lr      = 1e-3,
    [double]$ValFrac = 0.15,
    [string]$DataRoot = ""
)
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "venv missing - run scripts\install.ps1 first" }

$pyArgs = @("-m", "src.train",
            "--epochs", $Epochs,
            "--batch-size", $BatchSize,
            "--lr", $Lr,
            "--val-frac", $ValFrac)
if ($DataRoot -ne "") { $pyArgs += @("--data-root", $DataRoot) }

& $python @pyArgs