$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$VenvDir = Join-Path $RepoRoot ".venv"
$PythonExe = Join-Path $VenvDir "Scripts\\python.exe"
$PipExe = Join-Path $VenvDir "Scripts\\pip.exe"

if (-not (Test-Path -LiteralPath $VenvDir)) {
    python -m venv $VenvDir
}

& $PythonExe -m pip install --upgrade pip
& $PipExe install torch torchvision
& $PipExe install transformers accelerate qwen-vl-utils decord sentencepiece huggingface_hub
& $PipExe install -e $RepoRoot
& $PipExe install -e (Join-Path $RepoRoot "lmms-eval")

Write-Host "Environment ready at $VenvDir"
Write-Host "Activate with: $VenvDir\\Scripts\\Activate.ps1"
