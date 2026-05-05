# MinimalYoga

Strips a yoga video down to a minimal cue track: the instructor's own voice
calling each pose change, at the original timestamps, with silence everywhere
else. Useful once you know a routine and just want timing cues without music
or motivational talk.

## How it works

1. **Extract** audio from the video (ffmpeg)
2. **Transcribe** with word-level timestamps (faster-whisper, GPU if available)
3. **Identify** pose-change cues and count sequences (Claude Haiku, streaming)
4. **Assemble** a new audio file — original voice clips at original positions,
   silence between them, microfades on every cut, leading silence trimmed

Output: `output/<video-stem>_cues.mp3`  
Intermediate files `_transcript.json` and `_cues.json` are cached so
re-running skips expensive steps unless you pass `--force`.

---

## Setup

### 1. CUDA 12 (optional, for GPU transcription)

faster-whisper/ctranslate2 requires **CUDA 12** specifically — CUDA 13+ is not
yet supported. Install the CUDA 12 toolkit alongside any newer driver:

- Download: https://developer.nvidia.com/cuda-12-6-0-download-archive
- Install with default options; it coexists with other CUDA versions
- Verify: `nvcc --version` should report `release 12.x`

Skip this step to run on CPU (~1–1.5× realtime for the `small` model).

### 2. Python, ffmpeg, and pip dependencies

Run the provided setup script from an elevated PowerShell prompt:

```powershell
.\setup.ps1
```

This installs Python 3.12 and ffmpeg via winget (if not already present),
then installs all pip dependencies from `requirements.txt`.

If you prefer to do it manually:

```powershell
winget install Python.Python.3.12
winget install Gyan.FFmpeg
python -m pip install -r requirements.txt
```

### 3. Claude Code CLI

The pipeline calls the `claude` CLI for cue identification. Install it and
log in once:

- Install: https://claude.ai/claude-code
- Authenticate: `claude login`

The script looks for `claude` on PATH and also checks `~/.local/bin/claude`.

### 4. ffmpeg location

If ffmpeg is not on your system PATH, place (or symlink) the ffmpeg `bin/`
folder anywhere inside the project directory — the script will find it
automatically via a recursive search.

---

## Usage

```powershell
# Single video
python pipeline.py videos\my-yoga-class.mp4

# Whole folder
python pipeline.py videos\

# Options
python pipeline.py videos\ --model base     # faster, slightly less accurate
python pipeline.py videos\ --model medium   # slower, more accurate
python pipeline.py videos\ --force          # ignore cached transcript/cues
```

### Whisper model sizes

| Model  | Speed (GPU) | Notes |
|--------|-------------|-------|
| `base` | very fast   | Good enough for clear studio audio |
| `small`| fast        | Default, good balance |
| `medium`| moderate   | Better for accented or noisy audio |

---

## Output details

- **Timing** is preserved exactly — each cue appears at the same moment it
  did in the original video
- **Leading silence** is trimmed so the track starts at the first cue
- **Trailing silence** is kept — useful for queuing up a wake-up song after
  a savasana
- **Microfades** (20 ms) on every cut edge prevent clicks
- **Counts** are included as part of their cue
  ("hold for five, four, three, two, one, and release")
