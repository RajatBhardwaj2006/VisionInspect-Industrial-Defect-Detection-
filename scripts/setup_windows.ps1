<#
.SYNOPSIS
    Automated, non-destructive Windows setup script for VisionInspect.
.DESCRIPTION
    Verifies Python 3.12, creates a virtual environment (.venv) if needed,
    installs core dependencies from requirements.txt, and validates model files.
#>

$ErrorActionPreference = "Stop"

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "         VisionInspect Windows Environment Setup         " -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

# 1. Verify Python Installation
Write-Host "`n[1/4] Checking Python version..." -ForegroundColor Yellow
$PythonCmd = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
    $PythonCmd = "py -3.12"
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $PythonCmd = "python"
} else {
    Write-Error "Python was not found. Please install Python 3.12 (64-bit) from https://www.python.org/"
    exit 1
}

$PyVersion = & $PythonCmd --version
Write-Host "Found: $PyVersion" -ForegroundColor Green

# 2. Virtual Environment Setup
Write-Host "`n[2/4] Setting up Python virtual environment..." -ForegroundColor Yellow
$VenvDir = Join-Path $PSScriptRoot "..\.venv"

if (-not (Test-Path $VenvDir)) {
    Write-Host "Creating new virtual environment at .venv..." -ForegroundColor Gray
    & $PythonCmd -m venv $VenvDir
} else {
    Write-Host "Existing virtual environment found at .venv." -ForegroundColor Green
}

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Error "Virtual environment python executable not found at: $VenvPython"
    exit 1
}

# 3. Install Dependencies
Write-Host "`n[3/4] Installing dependencies from requirements.txt..." -ForegroundColor Yellow
$ReqFile = Join-Path $PSScriptRoot "..\requirements.txt"
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r $ReqFile

# 4. Model Integrity Verification
Write-Host "`n[4/4] Verifying production models..." -ForegroundColor Yellow
$VerifyScript = Join-Path $PSScriptRoot "verify_models.py"
& $VenvPython $VerifyScript

Write-Host "`n========================================================" -ForegroundColor Green
Write-Host " VisionInspect Setup Complete! Ready for Deployment." -ForegroundColor Green
Write-Host " To start the system, run:" -ForegroundColor Cyan
Write-Host "     .\scripts\start_visioninspect.ps1" -ForegroundColor White
Write-Host "========================================================`n" -ForegroundColor Green
