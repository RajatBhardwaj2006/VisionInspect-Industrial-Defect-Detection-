<#
.SYNOPSIS
    One-command launcher for VisionInspect (Backend API + Streamlit UI).
.DESCRIPTION
    Activates virtual environment, starts FastAPI backend on port 8000,
    and launches Streamlit UI on port 8501.
#>

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

# Locate Python environment (.venv or visioninspect_py312 or system python)
$PythonExe = $null
if (Test-Path "$ProjectRoot\.venv\Scripts\python.exe") {
    $PythonExe = "$ProjectRoot\.venv\Scripts\python.exe"
} elseif (Test-Path "$ProjectRoot\visioninspect_py312\Scripts\python.exe") {
    $PythonExe = "$ProjectRoot\visioninspect_py312\Scripts\python.exe"
} elseif (Test-Path "$ProjectRoot\venv\Scripts\python.exe") {
    $PythonExe = "$ProjectRoot\venv\Scripts\python.exe"
} else {
    $PythonExe = "python"
}

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "            Starting VisionInspect System               " -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "Using Python: $PythonExe" -ForegroundColor Gray
Write-Host "Project Root: $ProjectRoot" -ForegroundColor Gray

$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONPATH = "$ProjectRoot"

# 1. Start FastAPI Backend in background job
Write-Host "`n[1/2] Launching FastAPI Backend on http://127.0.0.1:8000..." -ForegroundColor Yellow
$BackendJob = Start-Job -ScriptBlock {
    param($py, $root)
    Set-Location $root
    $env:KMP_DUPLICATE_LIB_OK = "TRUE"
    $env:PYTHONPATH = "$root"
    & $py -m uvicorn app.backend:app --host 127.0.0.1 --port 8000
} -ArgumentList $PythonExe, $ProjectRoot

# Wait briefly and verify health
Start-Sleep -Seconds 3
$BackendHealthy = $false
try {
    $resp = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -TimeoutSec 5
    if ($resp.status -eq "ok") {
        $BackendHealthy = $true
        Write-Host "Backend is Healthy! (FastAPI v$($resp.version), Device: $($resp.device))" -ForegroundColor Green
    }
} catch {
    Write-Host "Warning: Backend starting up, proceeding to launch frontend..." -ForegroundColor Yellow
}

# 2. Start Streamlit Frontend
Write-Host "`n[2/2] Launching Streamlit Frontend on http://localhost:8501..." -ForegroundColor Yellow
Write-Host "Press Ctrl+C to terminate both frontend and backend.`n" -ForegroundColor Gray

try {
    & $PythonExe -m streamlit run "$ProjectRoot\app\app.py" --server.port 8501
} finally {
    Write-Host "`nShutting down backend service..." -ForegroundColor Yellow
    Stop-Job -Job $BackendJob -ErrorAction SilentlyContinue
    Remove-Job -Job $BackendJob -ErrorAction SilentlyContinue
    Write-Host "VisionInspect stopped." -ForegroundColor Green
}
