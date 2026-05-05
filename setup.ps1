# MinimalYoga – one-time setup for Windows
# Installs Python 3.12, ffmpeg, and pip dependencies.
# CUDA and the Claude CLI must be installed manually — see README.md.
#
# Run from an elevated PowerShell prompt (right-click -> "Run as administrator"),
# or a normal prompt if winget works without elevation on your machine.

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Ok($msg)   { Write-Host "    OK  $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "    >>  $msg" -ForegroundColor Yellow }

function Refresh-Path {
    $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("PATH", "User")
}

# ── 1. Python 3.12 ────────────────────────────────────────────────────────
Step "Python"
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py -or ($py.Source -like "*WindowsApps*")) {
    Write-Host "    Installing Python 3.12 via winget ..."
    winget install --id Python.Python.3.12 --source winget `
        --accept-package-agreements --accept-source-agreements
    Refresh-Path
    Ok "Python 3.12 installed"
} else {
    Ok $py.Source
}

# ── 2. ffmpeg ─────────────────────────────────────────────────────────────
Step "ffmpeg"
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host "    Installing ffmpeg via winget ..."
    winget install --id Gyan.FFmpeg --source winget `
        --accept-package-agreements --accept-source-agreements
    Refresh-Path
    Ok "ffmpeg installed"
} else {
    Ok "already on PATH"
}

# ── 3. pip dependencies ───────────────────────────────────────────────────
Step "Python dependencies"
python -m pip install --upgrade pip --quiet
python -m pip install -r requirements.txt
Ok "dependencies installed"

# ── 4. Claude CLI check ───────────────────────────────────────────────────
Step "Claude CLI"
$claude = Get-Command claude -ErrorAction SilentlyContinue
if (-not $claude) {
    $localClaude = "$env:USERPROFILE\.local\bin\claude"
    if (Test-Path $localClaude) {
        Ok "found at $localClaude (not on PATH, but pipeline will find it)"
    } else {
        Warn "claude CLI not found."
        Warn "Install from https://claude.ai/claude-code then run: claude login"
    }
} else {
    Ok $claude.Source
}

# ── 5. CUDA note ──────────────────────────────────────────────────────────
Step "CUDA (GPU transcription)"
$nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
if ($nvcc) {
    $ver = & nvcc --version 2>&1 | Select-String "release"
    Ok "CUDA found — $ver"
    if ($ver -notmatch "release 12\.") {
        Warn "ctranslate2 requires CUDA 12.x. GPU transcription may fail."
        Warn "Install CUDA 12 from: https://developer.nvidia.com/cuda-12-6-0-download-archive"
    }
} else {
    Warn "CUDA not found — transcription will run on CPU."
    Warn "For GPU support install CUDA 12: https://developer.nvidia.com/cuda-12-6-0-download-archive"
}

Write-Host "`nSetup complete." -ForegroundColor Green
Write-Host 'Usage:  python pipeline.py videos\my-class.mp4'
