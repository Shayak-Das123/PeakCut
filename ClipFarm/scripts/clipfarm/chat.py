"""Chat: how fast it moves and what it says. "CLIP IT" ×14 in ten seconds means something just happened."""
from __future__ import annotations

import json
import re
import signal
import subprocess
import time
from pathlib import Path

import numpy as np

from .util import cookie_args, warn, ytdlp

CHAT_DELAY = 4.0   # people type a few seconds after they see it

# weight, pattern. Matched once per message.
LEXICON = [
    (3.0, re.compile(r"\bclip(?:\s*(?:it|that|this))?\b|\bclipped\b|\bclipper\b", re.I)),
    (1.6, re.compile(r"\b(?:lmao+|lmfao+|lol+|kekw|omegalul|lulw?|icant|i can'?t|haha+h*|hehe+|xd+)\b|😂|🤣|💀|😭", re.I)),
    (1.4, re.compile(r"\b(?:no ?way+|what+(?:\?|!)|wtf+|holy|omg+|bro+|nah+|yo+o+|pog(?:gers|champ)?|insane|crazy)\b|🔥|😱|😳|‼", re.I)),
    (1.2, re.compile(r"^(?:w+|l+|ww+|ll+)$|\bw (?:streamer|rockstar|tim|kai)\b", re.I)),
    (1.0, re.compile(r"\?{3,}|!{3,}")),
    (0.8, re.compile(r"\b(?:rip|f|ff+|o7|sadge)\b", re.I)),
]


def lex_score(text: str) -> float:
    s = 0.0
    for w, rx in LEXICON:
        if rx.search(text):
            s += w
    if len(text) > 4 and text.isupper():
        s += 0.4
    return s


def is_clip_call(text: str) -> bool:
    return bool(LEXICON[0][1].search(text))


def _yt_message(line: str):
    try:
        d = json.loads(line)
    except Exception:
        return None
    r = d.get("replayChatItemAction") or {}
    t = r.get("videoOffsetTimeMsec")
    if t is None:
        return None
    for a in r.get("actions") or []:
        item = ((a.get("addChatItemAction") or {}).get("item")) or {}
        m = item.get("liveChatTextMessageRenderer") or item.get("liveChatPaidMessageRenderer")
        if not m:
            continue
        runs = ((m.get("message") or {}).get("runs")) or []
        txt = ""
        for run in runs:
            if "text" in run:
                txt += run["text"]
            elif "emoji" in run:
                sc = (run["emoji"].get("shortcuts") or [""])[0]
                eid = run["emoji"].get("emojiId", "")
                txt += eid if len(eid) <= 2 else f" {sc} "
        return float(t) / 1000.0, txt.strip()
    return None


def youtube_replay(url: str, folder: Path, cookies: str | None = None, budget_s: float = 900) -> dict:
    """Downloads the live chat replay (YouTube streams only). Stops after `budget_s` and keeps what it has."""
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / "chat.live_chat.json"
    if not out.exists():
        cmd = ytdlp() + ["--skip-download", "--write-subs", "--sub-langs", "live_chat", "--no-warnings", "--quiet",
                         "--no-progress", "-o", str(folder / "chat")] + cookie_args(cookies) + [url]
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        t0 = time.time()
        while p.poll() is None and time.time() - t0 < budget_s:
            time.sleep(1)
        if p.poll() is None:
            p.send_signal(signal.SIGINT)
            try:
                p.wait(10)
            except Exception:
                p.kill()
            warn(f"chat replay is huge; used the first {int(budget_s / 60)} min of downloading")
    src = out if out.exists() else next(iter(folder.glob("chat.live_chat.json*")), None)
    if not src:
        return {"messages": [], "source": "youtube-replay", "ok": False}
    msgs = []
    with open(src, errors="ignore") as f:
        for line in f:
            m = _yt_message(line)
            if m and m[1]:
                msgs.append(m)
    msgs.sort()
    return {"messages": msgs, "source": "youtube-replay", "ok": bool(msgs)}


def series(chat: dict, duration: float) -> dict:
    """Per-second arrays: speed relative to the stream's own normal, and what chat said."""
    n = int(np.ceil(duration)) + 1
    count = np.zeros(n)
    lex = np.zeros(n)
    clipcalls = np.zeros(n)
    for t, txt in chat.get("messages") or []:
        i = int(t - CHAT_DELAY)
        if 0 <= i < n:
            count[i] += 1
            lex[i] += lex_score(txt)
            if is_clip_call(txt):
                clipcalls[i] += 1
    if chat.get("rate"):   # sampled source (Twitch VOD): use the sampled speed instead of the raw count
        ts = np.array([r[0] for r in chat["rate"]]) - CHAT_DELAY
        vs = np.array([r[1] for r in chat["rate"]])
        speed = np.interp(np.arange(n), ts, vs, left=vs[0] if len(vs) else 0, right=vs[-1] if len(vs) else 0)
    else:
        k = np.ones(5) / 5
        speed = np.convolve(count, k, mode="same")
    base = rolling_block_median(speed, 300)
    ratio = speed / np.maximum(base, 0.2)
    k = np.ones(8)
    return {
        "speed": speed.tolist(),
        "ratio": ratio.tolist(),
        "lex": np.convolve(lex, k, mode="same").tolist(),
        "clipcalls": np.convolve(clipcalls, k, mode="same").tolist(),
        "messages": int(count.sum()),
    }


def rolling_block_median(x: np.ndarray, block: int) -> np.ndarray:
    """A cheap rolling baseline: the median of each block, interpolated."""
    n = len(x)
    if n == 0:
        return x
    centers, meds = [], []
    for s in range(0, n, max(1, block // 2)):
        seg = x[max(0, s - block // 2): s + block // 2]
        if len(seg):
            centers.append(s)
            meds.append(float(np.median(seg)))
    return np.interp(np.arange(n), centers, meds)
