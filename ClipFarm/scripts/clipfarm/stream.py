"""cf scan: everything that happens before the AI reads the stream."""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import chat as chatmod
from . import fetch, listen, signals
from .util import TASTE, Timer, hms, read_json, say, settings, stream_dir, warn, write_json

# one stream listens at a time (one GPU); downloads of the others keep going meanwhile
_LISTEN = threading.Lock()


def _trim_info(info: dict, url: str) -> dict:
    keep = ("id", "title", "channel", "uploader", "uploader_id", "duration", "upload_date", "timestamp",
            "release_timestamp", "live_status", "was_live", "is_live", "view_count", "concurrent_view_count",
            "webpage_url", "extractor_key", "heatmap", "description", "fps", "height", "width")
    d = {k: info.get(k) for k in keep if info.get(k) is not None}
    d["url"] = info.get("webpage_url") or url
    d["platform"] = fetch.platform_of(info)
    d["streamer"] = fetch.streamer_name(info)
    if d.get("description"):
        d["description"] = d["description"][:4000]
    return d


def scan(url: str, cookies: str | None = None, chat_budget: float = 600, force: bool = False, on_progress: Any = None) -> str:
    T = Timer()
    say(f"→ opening {url}")
    info_full = fetch.get_info(url, cookies)
    if info_full.get("is_live") or info_full.get("live_status") == "is_live":
        raise SystemExit("That stream is live right now. For live streams use: cf live start <url>")
    key = fetch.stream_key(info_full)
    d = stream_dir(key)
    d.mkdir(parents=True, exist_ok=True)
    info = _trim_info(info_full, url)
    write_json(d / "info.json", info)
    dur = float(info.get("duration") or 0)
    if dur <= 0:
        raise SystemExit("Couldn't read how long that stream is.")
    plat = info["platform"]
    say(f"  {info['streamer']} · {info.get('title', '')[:70]} · {hms(dur)} · {plat}")
    st = read_json(d / "state.json", {}) if not force else {}
    st.update({"key": key, "url": info["url"], "started": time.time()})

    # 1) audio + chat + crowd clips at the same time
    def job_audio():
        say("  downloading the audio only…")
        w = fetch.download_audio(info["url"], d, cookies)
        say(f"  audio ready ({(w.stat().st_size / 1e6):.0f} MB as 16 kHz WAV) · {Timer.fmt(T.mark('audio'))}")
        return w

    def job_chat():
        try:
            if plat == "youtube" and ("live_chat" in (info_full.get("subtitles") or {})):
                c = chatmod.youtube_replay(info["url"], d, cookies, chat_budget)
            elif plat == "twitch":
                c = fetch.twitch_vod_chat(info["id"], dur)
            else:
                return None
            if c and c.get("ok"):
                say(f"  chat replay: {len(c['messages']):,} messages · {Timer.fmt(T.mark('chat'))}")
                return c
        except Exception as e:
            warn(f"chat replay skipped: {e}")
        return None

    def job_crowd():
        if plat != "twitch":
            return []
        login = fetch.twitch_login(info_full)
        if not login:
            return []
        try:
            cl = fetch.twitch_vod_clips(login, info["id"])
            if cl:
                say(f"  viewers already clipped {len(cl)} moments of this VOD")
            return cl
        except Exception as e:
            warn(f"viewer clips skipped: {e}")
            return []

    def job_program():
        try:
            return fetch.clipping_program(info_full)
        except Exception:
            return []

    # chat keeps downloading while we listen: big chats take minutes, so nothing waits on them
    ex = ThreadPoolExecutor(4)
    fa, fc, fk, fp = ex.submit(job_audio), ex.submit(job_chat), ex.submit(job_crowd), ex.submit(job_program)
    wav = fa.result()

    # 2. listen
    with _LISTEN:
        say(f"  listening to {info['streamer']} on this computer…")
        def _prog_cb(v):
            if on_progress:
                try:
                    return on_progress("transcribe", v, f"Transcribing speech: {v}% (Sherpa-ONNX)")
                except Exception:
                    pass
        tr = listen.transcribe(wav, d / "transcript", on_progress_callback=_prog_cb)
    segs = listen.segments(tr)
    write_json(d / "segments.json", segs)
    chat_raw = fc.result()
    crowd = fk.result()
    program = fp.result()
    ex.shutdown(wait=False)
    if program:
        say("  💰 this streamer mentions a clipping program:")
        for p in program:
            say(f"     {p}")
    write_json(d / "program.json", program)
    write_json(d / "crowd_clips.json", crowd)
    T.mark("transcript")
    words = sum(len(s["text"].split()) for s in segs if s["kind"] == "speech")
    say(f"  transcript: {words:,} words · {Timer.fmt(T.mark('transcript'))}")
    en = listen.energy(wav)
    T.mark("energy")
    ch = chatmod.series(chat_raw, dur) if chat_raw else None
    if ch:
        write_json(d / "chat_series.json", {k: ch[k] for k in ("messages",)})

    # 3) the hype curve
    if on_progress:
        try:
            on_progress("signals", 75, f"Building hype & loudness curves ({words:,} words transcribed)...")
        except Exception:
            pass
    heat = info.get("heatmap") or None
    sig = signals.build(dur, en, ch, heat, crowd, segs)
    pk = signals.peaks(sig)
    np.savez_compressed(d / "signals.npz", hype=sig["hype"], **{f"raw_{k}": v for k, v in sig["raw"].items()})
    write_json(d / "peaks.json", pk)
    write_json(d / "spark.json", signals.sparkline(sig))
    used = sig["used"]
    say(f"  hype curve from: {', '.join(used) if used else 'nothing (no signals)'}")
    music_s = sum(s["e"] - s["s"] for s in segs if s["kind"] == "music")
    laughs = sum(1 for s in segs if s["kind"] == "laugh")

    # 4) chunks for the AI
    if on_progress:
        try:
            on_progress("chunks", 82, "Writing chunk cards for AI curation...")
        except Exception:
            pass
    n = write_chunks(d, info, segs, sig, pk, settings()["chunk_minutes"])
    st.update({"scanned": time.time(), "duration": dur, "streamer": info["streamer"], "title": info.get("title"),
               "platform": plat, "signals": used, "chunks": n, "words": words, "laughs": laughs,
               "music_seconds": round(music_s), "chat_messages": (ch or {}).get("messages", 0),
               "crowd_clips": len(crowd), "program": program, "scan_seconds": round(time.time() - T.t0)})
    write_json(d / "state.json", st)
    say(f"✓ scanned in {Timer.fmt(time.time() - T.t0)} · {n} chunks to read in {d / 'chunks'}")
    return key


def _lines(segs: list[dict]) -> list[dict]:
    """Phrases → readable lines of up to ~20 words / 12 s, broken at sentence ends."""
    lines: list[dict] = []
    cur = None
    for s in segs:
        if s["kind"] != "speech":
            if cur:
                lines.append(cur)
                cur = None
            tag = {"laugh": "(laughing)", "shout": f"({s['text'].strip('()[]* ').lower()})",
                   "music": "♪ music ♪", "sound": f"({s['text'].strip('()[]* ').lower()})"}[s["kind"]]
            if lines and lines[-1]["text"] == tag and s["s"] - lines[-1]["e"] < 4:
                lines[-1]["e"] = s["e"]
            else:
                lines.append({"s": s["s"], "e": s["e"], "text": tag, "kind": s["kind"]})
            continue
        if cur and (s["s"] - cur["e"] > 1.2 or len(cur["text"].split()) > 22 or s["e"] - cur["s"] > 12
                    or (cur["text"].rstrip().endswith((".", "?", "!")) and len(cur["text"].split()) >= 8)):
            lines.append(cur)
            cur = None
        if cur:
            cur["text"] += " " + s["text"]
            cur["e"] = s["e"]
        else:
            cur = {"s": s["s"], "e": s["e"], "text": s["text"], "kind": "speech"}
    if cur:
        lines.append(cur)
    return lines


def write_chunks(d: Path, info: dict, segs: list[dict], sig: dict, pk: list[dict], minutes: float) -> int:
    lines = _lines(segs)
    write_json(d / "lines.json", lines)
    hype = sig["hype"]
    order = np.sort(hype)
    p97 = order[int(len(order) * 0.97)] if len(order) else 1
    p92 = order[int(len(order) * 0.92)] if len(order) else 1
    dur = float(info.get("duration") or 0)
    size = minutes * 60
    overlap = 30
    starts = list(np.arange(0, max(1, dur - overlap), size - overlap)) or [0]
    cdir = d / "chunks"
    cdir.mkdir(exist_ok=True)
    for f in cdir.glob("chunk_*.md"):
        f.unlink()
    taste = TASTE.read_text().strip() if TASTE.exists() else ""
    total = len(starts)
    for ci, a in enumerate(starts, 1):
        b = min(dur, a + size)
        mine = [p for p in pk if a <= p["t"] < b]
        mine.sort(key=lambda p: -p["hype"])
        out = [f"# {info['streamer']} · {info.get('title', '')[:90]}",
               f"chunk {ci}/{total} · {hms(a)}–{hms(b)} · stream key {d.name}", ""]
        out.append("Rules: reference/picking.md. Write your picks to "
                   f"picks/chunk_{ci:02d}.json (an empty list is a fine answer).")
        if taste:
            out.append("Your channel's results so far: " + taste.splitlines()[0][:300])
        out.append("")
        if mine:
            out.append("Hottest moments by the signals (🔥🔥 top 3 % of this stream, 🔥 top 8 %):")
            for p in mine[:6]:
                out.append(f"  {hms(p['t'])}  hype {p['hype']:.2f}  {p['why']}")
        else:
            out.append("No signal peaks in this chunk. Only pick something the words alone make great.")
        out.append("")
        out.append("Transcript ([h:mm:ss] = when the line starts):")
        for ln in lines:
            if ln["e"] < a or ln["s"] >= b:
                continue
            i0, i1 = int(ln["s"]), int(ln["e"]) + 1
            hmax = float(np.max(hype[i0:i1])) if i1 > i0 and i0 < len(hype) else 0.0
            mark = "🔥🔥 " if hmax >= p97 else "🔥 " if hmax >= p92 else ""
            extra = ""
            if mark.startswith("🔥🔥"):
                w = signals.explain(sig, int((ln["s"] + ln["e"]) / 2))
                extra = f"   ⟵ {w}" if w else ""
            out.append(f"[{hms(ln['s'])}] {mark}{ln['text']}{extra}")
        (cdir / f"chunk_{ci:02d}.md").write_text("\n".join(out) + "\n")
    (d / "picks").mkdir(exist_ok=True)
    return total
