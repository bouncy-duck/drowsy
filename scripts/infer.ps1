# scripts/infer.ps1
param(
    [Parameter(Mandatory=$true)][string]$Input,
    [string]$Model  = "",
    [string]$OutDir = ""
)
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "venv missing — run scripts\install.ps1 first" }

$pyArgs = @("-m", "src.infer", "--input", $Input)
if ($Model  -ne "") { $pyArgs += @("--model",  $Model) }
if ($OutDir -ne "") { $pyArgs += @("--out-dir", $OutDir) }

& $python @pyArgs