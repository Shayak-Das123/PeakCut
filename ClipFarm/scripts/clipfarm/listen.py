"""Listening on your own computer: the transcript (whisper.cpp), how loud the streamer got, laughs and music."""
from __future__ import annotations

import json
import os
import re
import subprocess
import wave
from pathlib import Path

import numpy as np

from .util import MODELS, run, say, settings, warn, which

MODEL_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{name}.bin"
EXTRA_MODEL_DIRS = [Path.home() / ".cache/hyperframes/whisper/models", Path("/opt/homebrew/share/whisper-cpp"),
                    Path("/usr/local/share/whisper-cpp")]

LAUGH = re.compile(r"\b(laugh\w*|chuckl\w*|giggl\w*)\b", re.I)
SHOUT = re.compile(r"\b(scream\w*|shout\w*|yell\w*|gasp\w*|cheer\w*|groan\w*|sigh\w*)\b", re.I)
MUSIC = re.compile(r"♪|\bmusic\b|\bsinging\b", re.I)


def model_path(name: str | None = None) -> Path:
    name = name or settings()["whisper_model"]
    fn = f"ggml-{name}.bin"
    for d in [MODELS] + EXTRA_MODEL_DIRS:
        p = d / fn
        if p.exists() and p.stat().st_size > 10_000_000:
            return p
    return MODELS / fn


def whisper_exe() -> str | None:
    return which("whisper-cli") or which("whisper-cpp")


def _threads() -> int:
    return max(2, min(8, (os.cpu_count() or 4) - 2))


def transcribe_sherpa(wav: Path, out_base: Path, show_progress: bool = True, on_progress_callback=None) -> dict:
    import wave
    from studio import transcribe as st_transcribe
    
    with wave.open(str(wav), "rb") as w:
        raw = w.readframes(w.getnframes())
        audio = np.frombuffer(raw, dtype=np.int16)
        
    last_prog = [-10]
    def on_prog(p):
        v = int(p * 100)
        if v >= last_prog[0] + 5:
            last_prog[0] = v
            if show_progress:
                say(f"  listening… {v}%")
            if on_progress_callback:
                try:
                    res = on_progress_callback(v)
                    if res is False:
                        raise KeyboardInterrupt("Stopped by user")
                except KeyboardInterrupt:
                    raise
                except Exception:
                    pass

    words = st_transcribe.transcribe(audio, on_progress=on_prog)
    
    segs = []
    cur_words = []
    for w in words:
        if cur_words and (w["s"] - cur_words[-1]["e"] > 0.8 or len(cur_words) >= 12 or cur_words[-1]["w"].endswith((".", "!", "?"))):
            seg_text = " ".join(x["w"] for x in cur_words)
            segs.append({
                "offsets": {"from": int(cur_words[0]["s"] * 1000), "to": int(cur_words[-1]["e"] * 1000)},
                "text": seg_text,
                "tokens": [{"text": " " + x["w"], "offsets": {"from": int(x["s"] * 1000), "to": int(x["e"] * 1000)}} for x in cur_words]
            })
            cur_words = []
        cur_words.append(w)
    if cur_words:
        seg_text = " ".join(x["w"] for x in cur_words)
        segs.append({
            "offsets": {"from": int(cur_words[0]["s"] * 1000), "to": int(cur_words[-1]["e"] * 1000)},
            "text": seg_text,
            "tokens": [{"text": " " + x["w"], "offsets": {"from": int(x["s"] * 1000), "to": int(x["e"] * 1000)}} for x in cur_words]
        })
        
    res = {"transcription": segs}
    out_json = out_base.with_suffix(".json")
    out_json.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def transcribe(wav: Path, out_base: Path, model: str | None = None, show_progress: bool = True, on_progress_callback=None) -> dict:
    """The whole stream. Greedy decoding: about 20-25x faster than real time on an M-series Mac with small.en.
    No carried-over context (-mc 0): on multi-hour streams it stops whisper sliding into unpunctuated lowercase."""
    out_json = out_base.with_suffix(".json")
    if out_json.exists() and out_json.stat().st_size > 50:
        return json.loads(out_json.read_bytes().decode("utf-8", "replace"))
    
    exe = whisper_exe()
    if not exe:
        # Fall back to sherpa-onnx (pre-downloaded Parakeet 0.6B)
        return transcribe_sherpa(wav, out_base, show_progress=show_progress, on_progress_callback=on_progress_callback)

    mp = model_path(model)
    if not mp.exists():
        return transcribe_sherpa(wav, out_base, show_progress=show_progress, on_progress_callback=on_progress_callback)
    cmd = [exe, "-m", str(mp), "-f", str(wav), "-l", "en", "-oj", "-of", str(out_base),
           "-bs", "1", "-bo", "1", "-mc", "0", "-t", str(_threads()), "-np", "-pp"]
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, errors="replace")
    last = -10
    for line in p.stderr:
        m = re.search(r"progress\s*=\s*(\d+)%", line)
        if m and show_progress:
            v = int(m.group(1))
            if v >= last + 10:
                say(f"  listening… {v}%")
                last = v
    if p.wait() != 0 or not out_json.exists():
        return transcribe_sherpa(wav, out_base, show_progress=show_progress)
    return json.loads(out_json.read_bytes().decode("utf-8", "replace"))


def segments(tr: dict) -> list[dict]:
    """[{s, e, text, kind}] kind: speech | laugh | shout | music | sound"""
    out = []
    prev = None
    for seg in tr.get("transcription") or []:
        s = seg["offsets"]["from"] / 1000.0
        e = seg["offsets"]["to"] / 1000.0
        txt = re.sub(r"\s+", " ", seg.get("text") or "").strip()
        if not txt:
            continue
        kind = "speech"
        bare = re.sub(r"^[\-\s]+", "", txt)
        if MUSIC.search(bare) and ("♪" in bare or re.fullmatch(r"[\(\[].*[\)\]]", bare)):
            kind = "music"
        elif re.fullmatch(r"[\(\[\*].*[\)\]\*]", bare):
            kind = "laugh" if LAUGH.search(bare) else "shout" if SHOUT.search(bare) else "sound"
        bare = re.sub(r"^-\s*", "", bare)
        if kind == "speech":
            # whisper loops on silence; drop exact repeats of the previous line
            if prev and bare.lower() == prev.lower() and s - out[-1]["e"] < 5:
                out[-1]["e"] = e
                continue
            prev = bare
        out.append({"s": round(s, 2), "e": round(max(e, s + 0.2), 2), "text": bare, "kind": kind})
    return out


def energy(wav: Path, hop: float = 0.25) -> dict:
    """Loudness in dB per 0.25 s, and how far above the streamer's OWN normal each moment is (z-score)."""
    with wave.open(str(wav), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        step = int(sr * hop)
        out = np.empty(n // step + 1, dtype=np.float32)
        k = 0
        block = step * 400
        while True:
            raw = w.readframes(block)
            if not raw:
                break
            x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            m = len(x) // step
            if m:
                r = np.sqrt(np.mean(x[: m * step].reshape(m, step) ** 2, axis=1) + 1e-10)
                out[k:k + m] = 20 * np.log10(r)
                k += m
    db = out[:k]
    base, spread = _rolling_stats(db, int(300 / hop))
    z = (db - base) / np.maximum(spread, 1.5)
    return {"hop": hop, "db": db.round(2).tolist(), "z": z.round(2).tolist(), "sr": sr, "duration": n / sr}


def _rolling_stats(x: np.ndarray, block: int):
    n = len(x)
    centers, meds, mads = [], [], []
    for s in range(0, n, max(1, block // 4)):
        seg = x[max(0, s - block // 2): s + block // 2]
        seg = seg[seg > -70]
        if len(seg) < 10:
            continue
        med = float(np.median(seg))
        centers.append(s)
        meds.append(med)
        mads.append(float(np.median(np.abs(seg - med))) * 1.4826)
    if not centers:
        return np.full(n, -40.0), np.full(n, 6.0)
    idx = np.arange(n)
    return np.interp(idx, centers, meds), np.interp(idx, centers, mads)


# ------------------------------------------------------------------ word timings for one clip (captions)

def _split_points(wav: Path, max_len: float = 8.0, min_len: float = 2.5) -> list[tuple[float, float]]:
    """Cut the clip into windows of at most ~8 s at its quietest moments. Whisper's word timings drift on long
    stretches without speech (a streamer silently watching a trailer), and short windows keep them honest."""
    with wave.open(str(wav), "rb") as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    hop = int(sr * 0.02)
    m = len(x) // hop
    if m == 0:
        return [(0.0, len(x) / sr)]
    db = 20 * np.log10(np.sqrt(np.mean(x[: m * hop].reshape(m, hop) ** 2, axis=1)) + 1e-10)
    db = np.convolve(db, np.ones(10) / 10, mode="same")
    total = len(x) / sr
    out, start = [], 0.0
    while total - start > max_len:
        lo, hi = int((start + min_len) / 0.02), int((start + max_len) / 0.02)
        cut = (lo + int(np.argmin(db[lo:hi]))) * 0.02
        out.append((start, cut))
        start = cut
    out.append((start, total))
    return out


def _words_from_json(d: dict, offset: float) -> list[dict]:
    words: list[dict] = []
    for seg in d.get("transcription") or []:
        for tok in seg.get("tokens") or []:
            t = tok.get("text") or ""
            if t.startswith("[_") or t.startswith("<|") or not t.strip():
                continue
            t0 = tok["offsets"]["from"] / 1000.0
            t1 = tok["offsets"]["to"] / 1000.0
            dtw = tok.get("t_dtw")
            if dtw is not None and dtw >= 0:
                t0 = dtw / 100.0
            if t.startswith(" ") or not words:
                words.append({"w": t.strip(), "s": t0 + offset, "e": t1 + offset, "p": tok.get("p", 1.0)})
            else:
                words[-1]["w"] += t
                words[-1]["e"] = t1 + offset
    return words


def clip_words(wav: Path, model: str | None = None) -> list[dict]:
    """Word-by-word timings for a short clip."""
    exe = whisper_exe()
    mp = model_path(model)
    if not exe or not mp.exists():
        import wave
        from studio import transcribe as st_transcribe
        with wave.open(str(wav), "rb") as w:
            raw = w.readframes(w.getnframes())
            audio = np.frombuffer(raw, dtype=np.int16)
        words = st_transcribe.transcribe(audio)
        for i, w in enumerate(words):
            if i + 1 < len(words):
                w["e"] = min(max(w["e"], w["s"] + 0.08), words[i + 1]["s"]) if words[i + 1]["s"] > w["s"] else w["s"] + 0.08
            w["e"] = max(w["e"], w["s"] + 0.08)
        return words

    name = mp.name.replace("ggml-", "").replace(".bin", "")
    wins = _split_points(wav)
    parts = []
    for i, (a, b) in enumerate(wins):
        pw = wav.with_name(f"{wav.stem}_w{i:02d}.wav")
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{a:.3f}", "-t", f"{b - a:.3f}", "-i",
             str(wav), "-c:a", "pcm_s16le", str(pw)])
        parts.append((pw, a))
    files = []
    for pw, _ in parts:
        files += ["-f", str(pw)]
    # one call for every window: the model loads once
    cmd = [whisper_exe(), "-m", str(mp)] + files + ["-l", "en", "-ojf", "-np", "-t", str(_threads()), "-nfa",
                                                     "--dtw", name, "-mc", "0"]
    p = run(cmd, check=False, timeout=900)
    if p.returncode != 0:
        run([c for c in cmd if c not in ("-nfa", "--dtw", name)], timeout=900)
    words: list[dict] = []
    for pw, a in parts:
        j = Path(str(pw) + ".json")
        if j.exists():
            words += _words_from_json(json.loads(j.read_bytes().decode("utf-8", "replace")), a)
    # drop sound tags and music notes from captions
    clean = []
    in_tag = False
    for w in words:
        tx = w["w"]
        if tx.startswith(("(", "[", "*")):
            in_tag = True
        if in_tag:
            if tx.endswith((")", "]", "*")):
                in_tag = False
            continue
        if "♪" in tx or not re.search(r"[A-Za-z0-9]", tx):
            continue
        clean.append(w)
    for i, w in enumerate(clean):          # monotonic, end before the next start
        if i + 1 < len(clean):
            w["e"] = min(max(w["e"], w["s"] + 0.08), clean[i + 1]["s"]) if clean[i + 1]["s"] > w["s"] else w["s"] + 0.08
        w["e"] = max(w["e"], w["s"] + 0.08)
    return clean
