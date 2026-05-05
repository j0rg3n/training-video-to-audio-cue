#!/usr/bin/env python3
"""
MinimalYoga pipeline
====================
Given yoga video files, produces a minimal cue track: the instructor's own
voice saying each pose-change cue, at the original timing, with silence
everywhere else.

Steps
  1. Extract audio from video (ffmpeg)
  2. Transcribe with word-level timestamps (faster-whisper)
  3. Identify pose-change cues (claude CLI)
  4. Cut cue segments from original audio and reassemble

Usage
  python pipeline.py <video_file_or_directory> [--model base|small|medium] [--force]

Output lands in ./output/<stem>_cues.mp3
Intermediate transcript and cue JSON are kept for inspection/re-runs.
Use --force to ignore cached transcript/cues and redo those steps.
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Tool discovery – prepend local bin dirs so subprocesses find ffmpeg/claude
# ---------------------------------------------------------------------------

def _ensure_on_path(exe_name: str, search_roots: list[Path]) -> None:
    try:
        if subprocess.run([exe_name, "--version" if exe_name == "ffmpeg" else "-v"],
                          capture_output=True).returncode == 0:
            return
    except FileNotFoundError:
        pass
    for root in search_roots:
        for found in root.rglob(f"{exe_name}.exe"):
            os.environ["PATH"] = str(found.parent) + os.pathsep + os.environ.get("PATH", "")
            return

_here = Path(__file__).parent
_home = Path.home()
_ensure_on_path("ffmpeg", [_here])
_ensure_on_path("claude", [_home / ".local" / "bin", _home / "AppData" / "Local"])

import numpy as np                        # noqa: E402
import soundfile as sf                    # noqa: E402
from faster_whisper import WhisperModel   # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(seconds: float) -> str:
    """Format seconds as '45s' or '3m 05s'."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    return f"{s // 60}m {s % 60:02d}s"


def _clear(width: int = 72) -> None:
    """Overwrite the current line with spaces then return to column 0."""
    print(f"\r{' ' * width}\r", end="", flush=True)


# ---------------------------------------------------------------------------
# Step 1 – audio extraction
# ---------------------------------------------------------------------------

def extract_audio(video_path: Path, audio_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn",           # no video
            "-ar", "16000",  # 16 kHz — enough for speech, what whisper prefers
            "-ac", "1",      # mono
            "-f", "wav",
            str(audio_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# ---------------------------------------------------------------------------
# Step 2 – transcription with live progress bar
# ---------------------------------------------------------------------------

def _run_transcription(model: "WhisperModel", audio_path: Path, total_dur: float) -> list[dict]:
    """Iterate the whisper segment generator with a live progress bar."""
    segment_gen, _ = model.transcribe(str(audio_path), word_timestamps=True)

    words: list[dict] = []
    t0         = time.time()
    last_print = 0.0
    BAR        = 28

    for seg in segment_gen:
        now     = time.time()
        elapsed = now - t0
        pos     = seg.end
        pct     = min(pos / total_dur * 100, 100)

        if now - last_print >= 0.15:
            filled = int(BAR * pct / 100)
            bar    = "#" * filled + "." * (BAR - filled)
            eta    = (f"ETA {_fmt((total_dur - pos) / (pos / elapsed))}"
                      if elapsed > 1 and pos > 0 else "ETA --")
            print(f"\r     [{bar}] {pct:5.1f}%  elapsed {_fmt(elapsed)}  {eta}  ",
                  end="", flush=True)
            last_print = now

        if seg.words:
            for w in seg.words:
                text = w.word.strip()
                if text:
                    words.append({"start": round(w.start, 3),
                                  "end":   round(w.end,   3),
                                  "text":  text})

    _clear()
    print(f"     {len(words)} words transcribed in {_fmt(time.time() - t0)}")
    return words


def transcribe(audio_path: Path, model_size: str) -> list[dict]:
    """Load the best available device, then transcribe with a live progress bar."""
    total_dur = sf.info(str(audio_path)).duration

    import ctranslate2
    if ctranslate2.get_cuda_device_count() > 0:
        try:
            model = WhisperModel(model_size, device="cuda", compute_type="float16")
            print(f"     device: GPU (float16)")
            return _run_transcription(model, audio_path, total_dur)
        except RuntimeError as e:
            print(f"     GPU failed ({e}), falling back to CPU ...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    print(f"     device: CPU (int8)")
    return _run_transcription(model, audio_path, total_dur)


# ---------------------------------------------------------------------------
# Step 3 – cue identification via claude CLI, with elapsed-time spinner
# ---------------------------------------------------------------------------

CLAUDE_MODEL  = "claude-haiku-4-5"

PROMPT_TEMPLATE = """\
You are analyzing a yoga class transcript to extract pose-change cues.

The transcript lists every word with its start timestamp in [seconds].

Work through the transcript from start to finish. Each time you find a cue,
immediately emit it on its own line as a JSON object — do not wait until the end:
  {{"start": <float>, "end": <float>, "text": "<cue text>"}}

Output ONLY these JSON lines. No commentary, no array brackets, no markdown.

INCLUDE
- Pose names and moves: "downward dog", "warrior two", "fold forward",
  "come to child's pose", "roll up to standing"
- Side changes: "now the left side", "switch sides", "other side"
- Structural transitions: "let's take a vinyasa", "meet me at the top of the mat"
- Countdowns and release cues attached to a transition: "hold for five four three
  two one and release", "breathe in one two three four breathe out" — include the
  full count sequence as a single cue spanning from the first number to the release

EXCLUDE
- Breathing reminders with no movement and no count ("inhale deeply", "exhale here")
- Motivational or body-awareness talk ("notice how you feel", "great work")
- Micro-alignment cues within a held pose ("soften your knees", "draw your navel in")

For each cue use the tightest word span that conveys the instruction.
Include leading words ("and", "now", "come to") only when the instructor
uses them as part of the call.

Transcript:
{transcript}"""


def _claude_bin() -> str:
    for candidate in ["claude", str(_home / ".local" / "bin" / "claude")]:
        try:
            if subprocess.run([candidate, "--version"], capture_output=True).returncode == 0:
                return candidate
        except FileNotFoundError:
            pass
    raise FileNotFoundError("claude CLI not found — expected at ~/.local/bin/claude")


def identify_cues(words: list[dict]) -> list[dict]:
    """Stream claude CLI output, showing progress notices and collecting the JSON array."""
    transcript = " ".join(f"[{w['start']:.1f}]{w['text']}" for w in words)
    prompt = PROMPT_TEMPLATE.format(transcript=transcript)

    t0 = time.time()

    proc = subprocess.Popen(
        [_claude_bin(), "--model", CLAUDE_MODEL, "-p", prompt],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )

    cues: list[dict] = []
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            cue = json.loads(line)
            if "start" in cue and "end" in cue and "text" in cue:
                cues.append(cue)
                print(f"     {cue['start']:>7.1f}s  {cue['text']}")
        except json.JSONDecodeError:
            pass  # ignore any stray non-JSON lines

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"claude exited with code {proc.returncode}")
    if not cues:
        raise ValueError("No cues parsed from claude output")

    print(f"     {len(cues)} cues in {_fmt(time.time() - t0)}")
    return cues


# ---------------------------------------------------------------------------
# Step 4 – audio assembly
# ---------------------------------------------------------------------------

PRE_PAD_MS   = 120
POST_PAD_MS  = 350
FADE_MS      = 20    # micro-fade length to avoid clicks at cut edges
SILENCE_GATE = 1e-3  # amplitude threshold for leading-silence trim (~-60 dBFS)


def _apply_fades(segment: np.ndarray, fade_len: int) -> np.ndarray:
    """Apply a linear fade-in and fade-out to avoid clicks. Works in-place."""
    n = len(segment)
    fl = min(fade_len, n // 2)
    if fl < 2:
        return segment
    ramp = np.linspace(0.0, 1.0, fl, dtype=np.float32)
    segment = segment.copy()
    segment[:fl]  *= ramp
    segment[-fl:] *= ramp[::-1]
    return segment


def build_cue_track(audio_path: Path, cues: list[dict], output_path: Path) -> None:
    data, sr  = sf.read(str(audio_path), dtype="float32")
    track     = np.zeros_like(data)

    pre        = int(PRE_PAD_MS  * sr / 1000)
    post       = int(POST_PAD_MS * sr / 1000)
    fade_samps = int(FADE_MS     * sr / 1000)

    for cue in cues:
        s = max(0,         int(cue["start"] * sr) - pre)
        e = min(len(data), int(cue["end"]   * sr) + post)
        track[s:e] = _apply_fades(data[s:e], fade_samps)

    # Trim leading silence, preserve all trailing silence
    above = np.where(np.abs(track) > SILENCE_GATE)[0]
    if above.size:
        trim = max(0, above[0] - pre)   # keep a little air before the first cue
        track = track[trim:]

    wav_tmp = output_path.with_suffix(".tmp.wav")
    sf.write(str(wav_tmp), track, sr)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(wav_tmp), "-b:a", "128k", str(output_path)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    wav_tmp.unlink()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def process_video(
    video_path: Path,
    output_dir: Path,
    model_size: str,
    force: bool = False,
) -> None:
    stem            = video_path.stem
    audio_path      = output_dir / f"{stem}_audio.wav"
    transcript_path = output_dir / f"{stem}_transcript.json"
    cues_path       = output_dir / f"{stem}_cues.json"
    output_path     = output_dir / f"{stem}_cues.mp3"

    print(f"\n{'-' * 60}")
    print(f"  {video_path.name}")
    print(f"{'-' * 60}")

    # 1 – audio extraction
    t0 = time.time()
    print("1/4  Extracting audio ...", end="", flush=True)
    extract_audio(video_path, audio_path)
    print(f"  done ({_fmt(time.time() - t0)})")

    # 2 – transcription (cached)
    if transcript_path.exists() and not force:
        print(f"2/4  Transcript cached  ({transcript_path.name})")
        words = json.loads(transcript_path.read_text())
    else:
        print(f"2/4  Transcribing  [whisper/{model_size}]")
        words = transcribe(audio_path, model_size)
        transcript_path.write_text(json.dumps(words, indent=2))

    # 3 – cue identification (cached)
    if cues_path.exists() and not force:
        print(f"3/4  Cues cached  ({cues_path.name})")
        cues = json.loads(cues_path.read_text())
        for c in cues:
            print(f"     {c['start']:>7.1f}s  {c['text']}")
    else:
        print("3/4  Identifying cues  [Claude]")
        cues = identify_cues(words)
        cues_path.write_text(json.dumps(cues, indent=2))

    # 4 – assembly
    t0 = time.time()
    print("4/4  Building cue track ...", end="", flush=True)
    build_cue_track(audio_path, cues, output_path)
    print(f"  done ({_fmt(time.time() - t0)})")
    print(f"     => {output_path}")

    audio_path.unlink(missing_ok=True)


def main() -> None:
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    target     = Path(args[0])
    model_size = "small"
    force      = "--force" in args

    if "--model" in args:
        idx        = args.index("--model")
        model_size = args[idx + 1]

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    if target.is_dir():
        videos = sorted(p for p in target.iterdir()
                        if p.suffix.lower() in VIDEO_EXTS)
    elif target.is_file():
        videos = [target]
    else:
        print(f"Not found: {target}")
        sys.exit(1)

    if not videos:
        print(f"No video files found at {target}")
        sys.exit(1)

    for video in videos:
        process_video(video, output_dir, model_size, force=force)

    print("\nAll done. Output files are in ./output/")


if __name__ == "__main__":
    main()
