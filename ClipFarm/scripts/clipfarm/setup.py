"""cf setup / cf doctor: check what this Mac needs, install what's missing, say when it's ready."""
from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path

from .util import APP, MODELS, VENV, WORK, say, settings

BREW_PKGS = {"ffmpeg": "ffmpeg", "whisper-cli": "whisper-cpp", "deno": "deno", "python3.12": "python@3.12"}
PIP_PKGS = ["numpy", "opencv-python-headless", "pillow", "yt-dlp"]
MODEL_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{name}.bin"
EXTRA_MODEL_DIRS = [Path.home() / ".cache/hyperframes/whisper/models", Path("/opt/homebrew/share/whisper-cpp"),
                    Path("/usr/local/share/whisper-cpp")]


def model_path(name: str) -> Path:
    fn = f"ggml-{name}.bin"
    for d in [MODELS] + EXTRA_MODEL_DIRS:
        p = d / fn
        if p.exists() and p.stat().st_size > 10_000_000:
            return p
    return MODELS / fn


YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")


def _brew() -> str | None:
    for p in ("/opt/homebrew/bin/brew", "/usr/local/bin/brew"):
        if Path(p).exists():
            return p
    return shutil.which("brew")


def _have(cmd: str) -> str | None:
    p = shutil.which(cmd)
    if p:
        return p
    for d in ("/opt/homebrew/bin", "/usr/local/bin"):
        if Path(d, cmd).exists():
            return str(Path(d, cmd))
    return None


def _run(cmd: list[str]) -> bool:
    say("  $ " + " ".join(cmd))
    return subprocess.run(cmd).returncode == 0


def setup(doctor_only: bool = False) -> bool:
    ok = True
    say("Clip Farm · " + ("checking" if doctor_only else "setting up") + " this computer\n")
    # 1. the machine
    if sys.platform != "darwin":
        say("✗ This version runs on a Mac. (Windows and Linux: not tested yet.)")
        return False
    arm = platform.machine() == "arm64"
    say(f"{'✓' if arm else '!'} Mac with {'Apple silicon' if arm else 'an Intel chip (it works, the listening is just slower)'}")
    free = shutil.disk_usage(Path.home()).free / 1e9
    say(f"{'✓' if free > 8 else '!'} {free:.0f} GB free disk" + ("" if free > 8 else " (a 4-hour stream needs about 1 GB while it works)"))

    # 2. Homebrew tools
    brew = _brew()
    if not brew:
        say("✗ Homebrew is missing. Install it first (it asks for your Mac password), then run cf setup again:")
        say('  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"')
        return False
    say("✓ Homebrew")
    missing = [pkg for cmd, pkg in BREW_PKGS.items() if not _have(cmd)]
    if missing and not doctor_only:
        ok &= _run([brew, "install"] + missing)
    for cmd, pkg in BREW_PKGS.items():
        have = _have(cmd)
        say(f"{'✓' if have else '✗'} {cmd}" + ("" if have else f"  (brew install {pkg})"))
        ok &= bool(have)

    # 3. the clipper's own Python, so nothing else on your Mac changes
    py = _have("python3.12")
    vpy = VENV / "bin" / "python"
    if not vpy.exists() and not doctor_only and py:
        APP.mkdir(parents=True, exist_ok=True)
        ok &= _run([py, "-m", "venv", str(VENV)])
    if vpy.exists() and not doctor_only:
        _run([str(vpy), "-m", "pip", "install", "-q", "--upgrade", "pip"])
        ok &= _run([str(vpy), "-m", "pip", "install", "-q", "--upgrade"] + PIP_PKGS)
    if vpy.exists():
        p = subprocess.run([str(vpy), "-c", "import numpy, cv2, PIL, yt_dlp; print(yt_dlp.version.__version__)"],
                           capture_output=True, text=True)
        good = p.returncode == 0
        say(f"{'✓' if good else '✗'} clipper's Python ({VENV})" + (f" · yt-dlp {p.stdout.strip()}" if good else ""))
        ok &= good
    else:
        say(f"✗ clipper's Python isn't there yet ({VENV})")
        ok = False

    # 4. models: the ears (Whisper small.en, about 480 MB) and the face finder (230 KB)
    MODELS.mkdir(parents=True, exist_ok=True)
    name = settings()["whisper_model"]
    mp = model_path(name)
    if not mp.exists() and not doctor_only:
        say(f"  downloading the Whisper {name} model (about 480 MB, once)…")
        ok &= _run(["curl", "-L", "--fail", "--progress-bar", "-o", str(MODELS / mp.name), MODEL_URL.format(name=name)])
        mp = model_path(name)
    say(f"{'✓' if mp.exists() else '✗'} Whisper {name} model")
    ok &= mp.exists()
    yn = MODELS / "face_detection_yunet_2023mar.onnx"
    if not yn.exists() and not doctor_only:
        _run(["curl", "-L", "--fail", "-s", "-o", str(yn), YUNET_URL])
    say(f"{'✓' if yn.exists() else '!'} face finder" + ("" if yn.exists() else " (falls back to OpenCV's older one)"))

    WORK.mkdir(parents=True, exist_ok=True)
    say(f"\n{'✓ Ready.' if ok else '✗ Not ready yet: fix the ✗ lines above.'} Clips land in {WORK / 'clips'}")
    if ok:
        c = settings().get("cookies")
        say("  Age-restricted YouTube streams (common with GTA) and Kick need your browser's login: "
            + (f"using {c}." if c else "run  cf settings cookies chrome  if you hit one."))
    return ok
