param(
    [string]$EnvName = ".venv",
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"

$python = "python"
try {
    & $python --version | Out-Null
} catch {
    Write-Error "Python is not installed or not in PATH. Install Python 3.10+ and retry."
    exit 1
}

if ($Recreate -and (Test-Path $EnvName)) {
    Write-Host "Removing existing virtual environment $EnvName ..."
    Remove-Item -Recurse -Force $EnvName
}

if (-not (Test-Path $EnvName)) {
    Write-Host "Creating virtual environment $EnvName ..."
    & $python -m venv $EnvName
}

$activate = Join-Path $EnvName "Scripts\Activate.ps1"
if (-not (Test-Path $activate)) {
    Write-Error "Activation script not found at $activate"
    exit 1
}

Write-Host "Activating virtual environment ..."
. $activate

Write-Host "Upgrading pip/setuptools/wheel ..."
python -m pip install --upgrade pip setuptools wheel

Write-Host "Installing dependencies from requirements.txt ..."
if (-not (Test-Path "requirements.txt")) {
    Write-Error "requirements.txt not found in project root."
    exit 1
}

# Prefer PyTorch CPU wheels from default index; customize if CUDA is desired later
pip install -r requirements.txt

Write-Host "Environment setup complete. To activate later, run:`n. $activate"
