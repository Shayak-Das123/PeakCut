"""Shared helpers: folders, settings, running tools, time formats."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT_PROJECT = Path(__file__).resolve().parents[2]
APP = Path(os.environ.get("CLIPFARM_HOME", ROOT_PROJECT / "data"))          # tools, models, settings, ledger
WORK = Path(os.environ.get("CLIPFARM_WORK", ROOT_PROJECT / "work"))           # streams, clips, review pages
MODELS = APP / "models"
VENV = APP / "venv"
SETTINGS = APP / "settings.json"
LEDGER = APP / "ledger.json"
TASTE = APP / "taste.md"
SKILL_DIR = Path(__file__).resolve().parents[2]                            # .../clip-farm

DEFAULTS = {
    "cookies": "",              # "chrome", "safari", "firefox"… for age-restricted YouTube and Kick
    "whisper_model": "small.en",
    "clips_per_stream": 3,
    "max_clips_per_stream": 5,
    "chunk_minutes": 12,
    "min_clip_s": 12,
    "max_clip_s": 59,
    "fps_cap": 60,
    "hashtags": ["#GTA6", "#GTAVI", "#gaming"],
    "credit": True,             # put the streamer's name on every clip
    "live_threshold": 0.35,     # live mode: lower it if a quiet stream never triggers, raise it if it triggers too often
}


def settings() -> dict:
    s = dict(DEFAULTS)
    if SETTINGS.exists():
        try:
            s.update(json.loads(SETTINGS.read_text()))
        except Exception:
            pass
    return s


def save_settings(s: dict) -> None:
    APP.mkdir(parents=True, exist_ok=True)
    SETTINGS.write_text(json.dumps(s, indent=2))


def say(msg: str) -> None:
    print(msg, flush=True)


def warn(msg: str) -> None:
    print(f"! {msg}", file=sys.stderr, flush=True)


def which(name: str) -> str | None:
    exts = ["", ".exe", ".cmd", ".bat"] if sys.platform == "win32" else [""]
    for sub in ("Scripts", "bin"):
        for ext in exts:
            p = VENV / sub / (name + ext)
            if p.exists():
                return str(p)
            p_curr = Path(sys.prefix) / sub / (name + ext)
            if p_curr.exists():
                return str(p_curr)
    if os.environ.get("PEAKCUT_FFMPEG_DIR"):
        fdir = Path(os.environ["PEAKCUT_FFMPEG_DIR"])
        for ext in exts:
            p = fdir / (name + ext)
            if p.exists():
                return str(p)
    return shutil.which(name)


def ytdlp() -> list[str]:
    exe = which("yt-dlp")
    if not exe:
        raise SystemExit("yt-dlp is missing. Run: cf setup")
    return [exe]


def cookie_args(cookies: str | None = None) -> list[str]:
    c = cookies if cookies is not None else settings().get("cookies", "")
    return ["--cookies-from-browser", c] if c else []


_ff_dir = os.environ.get("PEAKCUT_FFMPEG_DIR", r"D:\Shutter Encoder\Library")
if _ff_dir and Path(_ff_dir).exists() and _ff_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _ff_dir + os.pathsep + os.environ.get("PATH", "")


def run(cmd: list[str], check: bool = True, capture: bool = True, timeout: float | None = None,
        env: dict | None = None, quiet: bool = False) -> subprocess.CompletedProcess:
    e = dict(os.environ)
    if env:
        e.update(env)
    real_cmd = list(cmd)
    resolved = which(real_cmd[0])
    if resolved:
        real_cmd[0] = resolved
    p = subprocess.run(real_cmd, capture_output=capture, text=True, errors="replace", timeout=timeout, env=e)
    if check and p.returncode != 0:
        tail = (p.stderr or p.stdout or "").strip().splitlines()[-6:]
        raise RuntimeError(f"{Path(cmd[0]).name} failed ({p.returncode}):\n  " + "\n  ".join(tail))
    return p


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default


def write_json(path: Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    tmp.replace(path)


def hms(sec: float, decimals: int = 0) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec - h * 3600 - m * 60
    s_txt = f"{s:0{3 + decimals}.{decimals}f}" if decimals else f"{int(s):02d}"
    return f"{h}:{m:02d}:{s_txt}" if h else f"{m}:{s_txt}"


def parse_t(v) -> float:
    """'1:02:03', '62:03', '3723', 3723.5, '1:02:03.4' -> seconds."""
    if isinstance(v, (int, float)):
        return float(v)
    v = str(v).strip()
    if re.fullmatch(r"\d+(\.\d+)?", v):
        return float(v)
    parts = v.split(":")
    if not 2 <= len(parts) <= 3:
        raise ValueError(f"bad time: {v!r}")
    parts = [float(p) for p in parts]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def slug(text: str, n: int = 40) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return (s[:n].rstrip("-")) or "clip"


def stream_dir(key: str) -> Path:
    return WORK / "streams" / key


def today() -> str:
    return time.strftime("%Y-%m-%d")


class Timer:
    def __init__(self):
        self.t0 = time.time()
        self.marks: list[tuple[str, float]] = []

    def mark(self, name: str) -> float:
        dt = time.time() - self.t0
        self.marks.append((name, dt))
        return dt

    @staticmethod
    def fmt(sec: float) -> str:
        m, s = divmod(int(round(sec)), 60)
        return f"{m} min {s:02d} s" if m else f"{s} s"
