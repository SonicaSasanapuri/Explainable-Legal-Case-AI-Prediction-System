$ErrorActionPreference = "Stop"

$projectRoot = $PSScriptRoot
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Error "Virtual environment not found at .venv\Scripts\python.exe. Create it with: py -m venv .venv"
    exit 1
}

& $venvPython -m uvicorn backend.main:app --reload --port 8000
