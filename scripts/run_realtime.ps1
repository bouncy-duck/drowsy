# scripts/run_realtime.ps1
param(
    [int]$Camera = 0,
    [string]$Model = "",
    [string]$Norm  = "",
    [switch]$NoLog
)
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "venv missing - run scripts\install.ps1 first"
}

$pyArgs = @("-m", "src.realtime", "--camera", $Camera)
if ($Model -ne "") { $pyArgs += @("--model", $Model) }
if ($Norm  -ne "") { $pyArgs += @("--norm",  $Norm)  }
if ($NoLog)        { $pyArgs += @("--no-log") }

& $python @pyArgs