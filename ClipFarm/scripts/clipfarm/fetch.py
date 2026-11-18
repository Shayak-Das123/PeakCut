"""Getting things from YouTube, Twitch and Kick: stream info, audio only, the minutes a clip needs."""
from __future__ import annotations

import json
import re
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .util import cookie_args, run, warn, ytdlp, write_json, read_json

# Twitch's own website uses this public client id for anonymous requests.
TWITCH_CLIENT_ID = "kimne78kx3ncx6brgo4mv6wki5h1ko"


def platform_of(info: dict) -> str:
    k = (info.get("extractor_key") or info.get("extractor") or "").lower()
    for p in ("youtube", "twitch", "kick"):
        if p in k:
            return p
    return "other"


def stream_key(info: dict) -> str:
    return f"{platform_of(info)}-{info.get('id')}"


def get_info(url: str, cookies: str | None = None) -> dict:
    cmd = ytdlp() + ["-J", "--no-warnings", "--no-playlist"] + cookie_args(cookies) + [url]
    p = run(cmd, check=False, timeout=180)
    if p.returncode != 0:
        err = (p.stderr or "").strip().splitlines()
        msg = err[-1] if err else "unknown error"
        if "Sign in to confirm your age" in msg or "age" in msg.lower() and "restrict" in msg.lower():
            raise SystemExit("This stream is age-restricted. Log in to YouTube in Chrome, then run: cf settings cookies chrome")
        if "Sign in to confirm you" in msg:
            raise SystemExit("YouTube wants a login for this one. Log in to YouTube in your browser, then: cf settings cookies chrome")
        raise SystemExit(f"Couldn't open that link: {msg}")
    return json.loads(p.stdout)


def streamer_name(info: dict) -> str:
    for k in ("channel", "uploader", "creator"):
        v = info.get(k)
        if v:
            return re.sub(r"\s+(VODs?|Live|Clips|Full Streams?|Archives?|Gaming|Highlights|Uncut|TV)$", "", str(v),
                          flags=re.I).strip()
    return "streamer"


def download_audio(url: str, folder: Path, cookies: str | None = None) -> Path:
    """Audio only: a 4-hour stream is a few hundred MB instead of many GB. Returns a 16 kHz mono WAV."""
    folder.mkdir(parents=True, exist_ok=True)
    wav = folder / "audio16k.wav"
    if wav.exists() and wav.stat().st_size > 1000:
        return wav
    for old in folder.glob("audio_src.*"):
        old.unlink()
    cmd = ytdlp() + ["-f", "bestaudio/ba*/worst", "--no-playlist", "--no-warnings", "--quiet", "--no-progress",
                     "-N", "4", "-o", str(folder / "audio_src.%(ext)s")] + cookie_args(cookies) + [url]
    run(cmd, timeout=7200)
    src = next((p for p in folder.glob("audio_src.*") if not p.name.endswith(".part")), None)
    if not src:
        raise RuntimeError("audio download produced no file")
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", str(wav)], timeout=3600)
    return wav


def download_section(url: str, start: float, end: float, out: Path, cookies: str | None = None,
                     max_h: int = 1080) -> Path:
    """Only the minutes a clip needs, at up to 1080p."""
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 10000:
        return out
    fmt = f"bv*[height<={max_h}][fps<=60]+ba/b[height<={max_h}]/bv*+ba/best/b"
    cmd = ytdlp() + ["--download-sections", f"*{max(0, start):.2f}-{end:.2f}", "-f", fmt,
                     "--merge-output-format", "mp4", "--no-playlist", "--no-warnings", "--quiet", "--no-progress",
                     "-o", str(out)] + cookie_args(cookies) + [url]
    run(cmd, timeout=900)
    if not out.exists():
        cands = sorted(out.parent.glob(out.stem + "*"), key=lambda p: p.stat().st_mtime)
        cands = [c for c in cands if c.suffix in (".mp4", ".mkv", ".webm") and not c.name.endswith(".part")]
        if not cands:
            raise RuntimeError("section download produced no file")
        cands[-1].rename(out)
    return out


# ---------------------------------------------------------------- Twitch (public web endpoints, no key)

def twitch_gql(query: str, variables: dict | None = None, timeout: float = 20) -> dict:
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    req = urllib.request.Request("https://gql.twitch.tv/gql", data=body, headers={
        "Client-ID": TWITCH_CLIENT_ID, "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def twitch_login(info: dict) -> str | None:
    for k in ("uploader_id", "channel_id", "uploader"):
        v = info.get(k)
        if v and re.fullmatch(r"[A-Za-z0-9_]{3,25}", str(v)):
            return str(v).lower()
    return None


def twitch_vod_clips(login: str, vod_id: str) -> list[dict]:
    """Clips that viewers already made from this VOD, with where they sit in it. The crowd's own picks."""
    vod_id = str(vod_id).lstrip("v")
    found: dict[str, dict] = {}
    for period in ("LAST_DAY", "LAST_WEEK", "LAST_MONTH", "ALL_TIME"):
        cursor = None
        for _ in range(6):
            after = f', after: "{cursor}"' if cursor else ""
            q = ('query { user(login: "%s") { clips(first: 100%s, criteria: {period: %s, sort: VIEWS_DESC}) '
                 '{ edges { cursor node { id title viewCount durationSeconds videoOffsetSeconds createdAt video { id } } } '
                 'pageInfo { hasNextPage } } } }') % (login, after, period)
            try:
                d = twitch_gql(q)
            except Exception as e:
                warn(f"twitch clips: {e}")
                return list(found.values())
            clips = (((d.get("data") or {}).get("user") or {}).get("clips") or {})
            edges = clips.get("edges") or []
            for e in edges:
                n = e.get("node") or {}
                if (n.get("video") or {}).get("id") == vod_id and n.get("videoOffsetSeconds") is not None:
                    found[n["id"]] = {"t": float(n["videoOffsetSeconds"]), "dur": float(n.get("durationSeconds") or 20),
                                      "views": int(n.get("viewCount") or 0), "title": n.get("title") or ""}
            if not edges or not (clips.get("pageInfo") or {}).get("hasNextPage"):
                break
            cursor = edges[-1].get("cursor")
        if len(found) >= 40:
            break
    return sorted(found.values(), key=lambda c: c["t"])


def twitch_vod_chat(vod_id: str, duration: float, step: float = 15.0, workers: int = 8) -> dict:
    """Samples the VOD's chat every `step` seconds. Returns messages and a messages-per-second estimate."""
    vod_id = str(vod_id).lstrip("v")
    q = ('query($id: ID!, $t: Int) { video(id: $id) { comments(contentOffsetSeconds: $t) { edges { node { '
         'contentOffsetSeconds message { fragments { text } } } } } } }')

    def one(t: int):
        try:
            d = twitch_gql(q, {"id": vod_id, "t": t}, timeout=15)
            edges = ((((d.get("data") or {}).get("video") or {}).get("comments") or {}).get("edges")) or []
            return t, [(float(e["node"]["contentOffsetSeconds"]),
                        "".join(f.get("text") or "" for f in e["node"]["message"]["fragments"])) for e in edges]
        except Exception:
            return t, None

    offsets = list(range(0, int(duration), int(step)))
    msgs: dict[tuple, None] = {}
    rate: list[tuple[float, float]] = []
    ok = 0
    with ThreadPoolExecutor(workers) as ex:
        for t, batch in ex.map(one, offsets):
            if batch is None:
                continue
            ok += 1
            for m in batch:
                msgs[m] = None
            if batch:
                # one page of messages from this offset: how long it took chat to write them is the local speed
                span = max(1.0, batch[-1][0] - batch[0][0])
                rate.append((t + step / 2, min(len(batch) / span, 200.0)))
            else:
                rate.append((t + step / 2, 0.0))
    if not ok:
        return {"messages": [], "rate": [], "source": "twitch-vod", "ok": False}
    return {"messages": sorted(msgs), "rate": rate, "source": "twitch-vod", "ok": True}


def twitch_panels(login: str) -> list[dict]:
    q = ('query { user(login: "%s") { description panels { ... on DefaultPanel { title description linkURL } } } }'
         % login)
    try:
        d = twitch_gql(q)
    except Exception:
        return []
    u = (d.get("data") or {}).get("user") or {}
    out = [{"title": "about", "description": u.get("description") or "", "linkURL": ""}]
    for p in u.get("panels") or []:
        if p:
            out.append(p)
    return out


CLIP_PROGRAM = re.compile(
    r"(clipp?ing (program|campaign|contest)|clip (program|campaign|contest)|clippers?\b|"
    r"whop\.com/[\w\-/]+|contentrewards|content rewards|get paid to clip|paid per view|clip farm)", re.I)


def clipping_program(info: dict) -> list[str]:
    """Looks in the stream's description (and a Twitch channel's panels) for a clipping program."""
    texts = [info.get("description") or ""]
    if platform_of(info) == "twitch":
        login = twitch_login(info)
        if login:
            for p in twitch_panels(login):
                texts.append(" ".join(str(p.get(k) or "") for k in ("title", "description", "linkURL")))
    hits = []
    for t in texts:
        for line in re.split(r"[\n\r]+", t):
            if CLIP_PROGRAM.search(line):
                hits.append(line.strip()[:200])
    seen, out = set(), []
    for h in hits:
        if h.lower() not in seen:
            seen.add(h.lower())
            out.append(h)
    return out[:5]
