"""cf live: clip a stream while it's still live. Records in the background, listens every 20 seconds, and hands
the AI each moment where the stream gets hot. Clips land a few minutes after the moment happened."""
from __future__ import annotations

import csv
import json
import os
import random
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from . import fetch, listen
from .util import WORK, cookie_args, hms, parse_t, read_json, say, settings, today, warn, write_json, ytdlp

TICK = 20.0          # seconds between listening passes
SEG = 10             # seconds per recorded segment
AFTER = 20.0         # how much has to have happened after a peak before we look at it
COOLDOWN = 75.0      # minimum gap between two moments


def live_root() -> Path:
    return WORK / "live"


def _dir(key: str) -> Path:
    return live_root() / key


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _latest_key(target: str | None) -> str:
    root = live_root()
    if target and (root / target).exists():
        return target
    cands = []
    for p in root.glob("*"):
        st = read_json(p / "state.json", {})
        if st and (not target or target in p.name or target in str(st.get("url", ""))):
            cands.append((st.get("started", 0), p.name))
    if not cands:
        raise SystemExit("No live session. Start one with: cf live start <live url>")
    return sorted(cands)[-1][1]


# ------------------------------------------------------------------ CLI entry

def main(a) -> None:
    if a.action == "start":
        if not a.target:
            raise SystemExit("usage: cf live start <live url>")
        start(a.target, a.keep_minutes, a.cookies)
    elif a.action == "next":
        nxt(_latest_key(None if (a.target or "").isdigit() else a.target), a.wait)
    elif a.action == "cut":
        key, n = _key_and_n(a.target)
        cut(key, n)
    elif a.action == "skip":
        key, n = _key_and_n(a.target)
        d = _dir(key) / "moments"
        (d / f"moment_{n:03d}.skip").write_text("skipped\n")
        say(f"✓ skipped moment {n}")
    elif a.action == "stop":
        stop(_latest_key(a.target))
    elif a.action == "status":
        status(_latest_key(a.target) if a.target else None)


def _key_and_n(target: str | None):
    if not target:
        raise SystemExit("which moment? e.g. cf live cut 3")
    if ":" in target:
        k, n = target.split(":", 1)
        return _latest_key(k), int(n)
    return _latest_key(None), int(target)


def start(url: str, keep_minutes: float, cookies: str | None) -> None:
    info = fetch.get_info(url, cookies)
    if not (info.get("is_live") or info.get("live_status") == "is_live"):
        raise SystemExit("That stream isn't live right now. For a finished stream use: cf scan <url>")
    key = f"{fetch.platform_of(info)}-{(fetch.twitch_login(info) or info.get('id'))}-{time.strftime('%H%M')}"
    d = _dir(key)
    (d / "rec").mkdir(parents=True, exist_ok=True)
    (d / "moments").mkdir(exist_ok=True)
    meta = {"url": info.get("webpage_url") or url, "platform": fetch.platform_of(info),
            "streamer": fetch.streamer_name(info), "title": info.get("title") or "", "id": key,
            "login": fetch.twitch_login(info) if fetch.platform_of(info) == "twitch" else None}
    write_json(d / "info.json", meta)
    write_json(d / "state.json", {"url": meta["url"], "started": time.time(), "keep_minutes": keep_minutes,
                                  "cookies": cookies if cookies is not None else settings().get("cookies", ""),
                                  "status": "starting", "streamer": meta["streamer"]})
    log = open(d / "runner.log", "a")
    p = subprocess.Popen([sys.executable, str(Path(__file__).resolve().parents[1] / "cf.py"), "live-run", key],
                         stdout=log, stderr=log, start_new_session=True)
    st = read_json(d / "state.json", {})
    st["pid"] = p.pid
    write_json(d / "state.json", st)
    say(f"→ recording {meta['streamer']} live ({meta['platform']}) · session {key}")
    t0 = time.time()
    while time.time() - t0 < 90:
        if len(list((d / "rec").glob("seg_*.ts"))) >= 2:
            say("✓ recording and listening. Next: cf live next   (it waits until something happens)")
            return
        if p.poll() is not None:
            raise SystemExit("The recorder stopped right away. Look at " + str(d / "runner.log"))
        time.sleep(2)
    say("… still waiting for the first seconds of video (slow stream?). cf live status shows progress.")


def stop(key: str) -> None:
    d = _dir(key)
    st = read_json(d / "state.json", {})
    pid = st.get("pid")
    if _alive(pid):
        try:
            os.killpg(pid, signal.SIGTERM)
        except Exception:
            os.kill(pid, signal.SIGTERM)
    st["status"] = "stopped"
    write_json(d / "state.json", st)
    done = len(list((d / "moments").glob("*.done")))
    say(f"✓ stopped {st.get('streamer', key)} · {hms(st.get('recorded', 0))} recorded · {done} clips made")


def status(key: str | None) -> None:
    keys = [key] if key else sorted(p.name for p in live_root().glob("*") if (p / "state.json").exists())
    if not keys:
        say("no live sessions")
    for k in keys:
        d = _dir(k)
        st = read_json(d / "state.json", {})
        m = d / "moments"
        found = len(list(m.glob("moment_*.json"))) - len(list(m.glob("*.decision.json")))
        say(f"{k}: {'running' if _alive(st.get('pid')) else st.get('status', 'stopped')} · recorded "
            f"{hms(st.get('recorded', 0))} · listened to {hms(st.get('analyzed', 0))} · chat {st.get('chat', 0):,} msgs · "
            f"moments {found} found, {len(list(m.glob('*.done')))} clipped, {len(list(m.glob('*.skip')))} skipped")


# ------------------------------------------------------------------ waiting for the next moment

def nxt(key: str, wait: float) -> None:
    d = _dir(key)
    m = d / "moments"
    t0 = time.time()
    while True:
        for j in sorted(m.glob("moment_*.json")):
            if j.name.endswith(".decision.json"):
                continue
            n = int(re.findall(r"\d+", j.stem)[0])
            if any((m / f"moment_{n:03d}{s}").exists() for s in (".skip", ".done", ".decision.json")):
                continue
            say((m / f"moment_{n:03d}.md").read_text())
            return
        st = read_json(d / "state.json", {})
        if not _alive(st.get("pid")):
            raise SystemExit(f"The live session isn't running ({st.get('status')}). See {d / 'runner.log'}")
        if time.time() - t0 > wait:
            say(f"Nothing big in the last {int(wait / 60)} min. Recorded {hms(st.get('recorded', 0))} so far. "
                "Run cf live next again to keep watching.")
            return
        time.sleep(3)


# ------------------------------------------------------------------ cutting a clip from the recording

def _segments(d: Path) -> list[dict]:
    idx = read_json(d / "rec" / "index.json", []) or []
    return [s for s in idx if (d / "rec" / s["file"]).exists()]


def cut(key: str, n: int) -> None:
    from . import render, review
    d = _dir(key)
    m = d / "moments"
    dec = read_json(m / f"moment_{n:03d}.decision.json")
    if not dec:
        raise SystemExit(f"Write {m / f'moment_{n:03d}.decision.json'} first (start, end, hook, caption, hashtags).")
    s, e = parse_t(dec["start"]), parse_t(dec["end"])
    segs = [x for x in _segments(d) if x["end"] > s - 4 and x["start"] < e + 4]
    if not segs:
        raise SystemExit("Those minutes aren't in the recording anymore (only the last 45 min are kept).")
    sec_dir = d / "sections"
    sec_dir.mkdir(exist_ok=True)
    lst = sec_dir / f"m{n:03d}.txt"
    lst.write_text("".join(f"file '{(d / 'rec' / x['file']).as_posix()}'\n" for x in segs))
    sec = sec_dir / f"m{n:03d}.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c", "copy", "-bsf:a", "aac_adtstoasc", "-movflags", "+faststart", str(sec)], check=True)
    info = read_json(d / "info.json", {})
    clip = dict(dec, id=f"m{n:03d}")
    out_dir = WORK / "clips" / today()
    meta = render.render_one(info, d, clip, render._next_index(out_dir), out_dir, None,
                             section=(sec, segs[0]["start"]))
    meta["live"] = True
    (out_dir / meta["file"]).with_suffix(".json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    render._ledger_add([meta])
    (m / f"moment_{n:03d}.done").write_text(meta["file"] + "\n")
    page = review.build(today())
    st = read_json(d / "state.json", {})
    lag = time.time() - (st.get("started", time.time()) + s)
    flag = "✓" if meta["qc"]["ok"] else "⚠ " + "; ".join(meta["qc"]["issues"])
    say(f"{flag} {meta['file']} · {meta['length']:.1f} s · {meta['layout']} · on disk {lag / 60:.1f} min after the "
        f"moment\n  review page: {page}\nNext: cf live next")


# ------------------------------------------------------------------ the background runner

class Runner:
    def __init__(self, key: str):
        self.key = key
        self.d = _dir(key)
        self.info = read_json(self.d / "info.json", {})
        self.st = read_json(self.d / "state.json", {})
        self.started = self.st["started"]
        self.stop = False
        self.chat: list[tuple[float, str]] = []
        self.index: list[dict] = read_json(self.d / "rec" / "index.json", []) or []
        self.seen = {x["file"] for x in self.index}
        self.db = np.zeros(0, dtype=np.float32)     # 0.25 s steps on the recording timeline
        self.segs: list[dict] = []                  # transcript lines on the recording timeline
        self.pending: list[dict] = []               # recorded segments not yet listened to
        self.last_peak = -1e9
        self.analyzed = 0.0
        self.n = len(list((self.d / "moments").glob("moment_*.md")))
        self.run_offset = 0.0
        self.proc = None

    def save_state(self, **kw):
        st = read_json(self.d / "state.json", {})
        st.update(kw)
        write_json(self.d / "state.json", st)

    # recorder: yt-dlp → ffmpeg, 10-second segments; restarts when the stream hiccups
    def record(self):
        tries = 0
        while not self.stop and tries < 30:
            self.run_offset = time.time() - self.started
            num = len(self.seen) + 1
            ytd = ytdlp() + ["-f", "best[height<=1080][fps<=60]/best", "--no-part", "-q", "--no-warnings",
                             "--hls-use-mpegts", "-o", "-"] + cookie_args(self.st.get("cookies")) + [self.info["url"]]
            ff = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-map", "0:v:0?", "-map", "0:a:0?",
                  "-c", "copy", "-f", "segment", "-segment_time", str(SEG), "-segment_start_number", str(num),
                  "-segment_list", str(self.d / "rec" / f"run_{num:06d}.csv"), "-segment_list_type", "csv",
                  "-reset_timestamps", "1", str(self.d / "rec" / "seg_%06d.ts")]
            y = subprocess.Popen(ytd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            f = subprocess.Popen(ff, stdin=y.stdout, stderr=subprocess.DEVNULL)
            self.proc = (y, f)
            f.wait()
            y.kill()
            tries += 1
            if not self.stop:
                print(f"[live] recorder ended, restarting ({tries})", flush=True)
                time.sleep(5)

    def collect(self):
        """New finished segments from the csv lists, placed on the recording timeline."""
        for lst in sorted((self.d / "rec").glob("run_*.csv")):
            off = self.st.get("offsets", {}).get(lst.name)
            if off is None:
                off = self.run_offset if lst.name == sorted((self.d / "rec").glob("run_*.csv"))[-1].name else 0.0
                offs = self.st.setdefault("offsets", {})
                offs[lst.name] = off
                self.save_state(offsets=offs)
            try:
                rows = list(csv.reader(open(lst)))
            except Exception:
                continue
            for r in rows:
                if len(r) < 3 or r[0] in self.seen:
                    continue
                seg = {"file": r[0], "start": round(off + float(r[1]), 3), "end": round(off + float(r[2]), 3)}
                self.seen.add(r[0])
                self.index.append(seg)
                self.pending.append(seg)
        write_json(self.d / "rec" / "index.json", self.index)

    def chat_irc(self):
        login = self.info.get("login")
        if not login:
            return
        while not self.stop:
            try:
                s = socket.create_connection(("irc.chat.twitch.tv", 6667), timeout=30)
                s.sendall(f"PASS SCHMOOPIIE\r\nNICK justinfan{random.randint(10000, 99999)}\r\nJOIN #{login}\r\n".encode())
                buf = b""
                while not self.stop:
                    data = s.recv(4096)
                    if not data:
                        break
                    buf += data
                    while b"\r\n" in buf:
                        line, buf = buf.split(b"\r\n", 1)
                        t = line.decode("utf-8", "ignore")
                        if t.startswith("PING"):
                            s.sendall(b"PONG :tmi.twitch.tv\r\n")
                        elif " PRIVMSG " in t:
                            msg = t.split(" :", 1)[-1] if " :" in t else ""
                            self.chat.append((time.time() - self.started, msg))
            except Exception as e:
                print(f"[live] chat reconnect: {e}", flush=True)
                time.sleep(5)

    def listen_new(self):
        if not self.pending:
            return
        batch, self.pending = self.pending, []
        wav = self.d / "work.wav"
        lst = self.d / "work.txt"
        lst.write_text("".join(f"file '{(self.d / 'rec' / x['file']).as_posix()}'\n" for x in batch))
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i",
                        str(lst), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)])
        if not wav.exists():
            return
        t0 = batch[0]["start"]
        en = listen.energy(wav)
        need = int((batch[-1]["end"]) / 0.25) + 1
        if len(self.db) < need:
            self.db = np.concatenate([self.db, np.full(need - len(self.db), -90.0, dtype=np.float32)])
        i0 = int(t0 / 0.25)
        vals = np.asarray(en["db"], dtype=np.float32)
        self.db[i0:i0 + len(vals)] = vals[: max(0, len(self.db) - i0)]
        base = self.d / "work_tr"
        for ext in (".json",):
            (base.with_suffix(ext)).unlink(missing_ok=True)
        try:
            tr = listen.transcribe(wav, base, show_progress=False)
            for sgm in listen.segments(tr):
                sgm["s"] += t0
                sgm["e"] += t0
                self.segs.append(sgm)
        except Exception as e:
            print(f"[live] listen failed: {e}", flush=True)
        self.analyzed = max(self.analyzed, batch[-1]["end"])
        with open(self.d / "transcript.jsonl", "a") as f:
            for sgm in self.segs[-200:]:
                if sgm["s"] >= t0:
                    f.write(json.dumps(sgm) + "\n")

    def hype(self) -> np.ndarray:
        n = int(self.analyzed) + 1
        parts, weights = [], []
        if len(self.db):
            db = np.pad(self.db[: n * 4], (0, max(0, n * 4 - len(self.db[: n * 4]))), constant_values=-90)
            hist = db[db > -70][-2400:]          # the last 10 minutes, in 0.25 s steps (same scale as cf scan)
            if len(hist) > 240:
                med = float(np.median(hist))
                mad = float(np.median(np.abs(hist - med))) * 1.4826
                z = ((db - med) / max(mad, 1.5)).reshape(n, 4).max(axis=1)
                parts.append(np.clip(z - 1.0, 0, 5) / 5)
                weights.append(0.4)
        if len(self.chat) > 30:
            cnt = np.zeros(n)
            for t, _ in self.chat:
                i = int(t - 4)
                if 0 <= i < n:
                    cnt[i] += 1
            sp = np.convolve(cnt, np.ones(5) / 5, mode="same")
            med = max(0.2, float(np.median(sp[-1800:])))
            parts.append(np.clip(np.log2(np.maximum(sp / med, 1e-3)), 0, 3) / 3)
            weights.append(0.35)
        lg = np.zeros(n)
        for sgm in self.segs:
            if sgm["kind"] in ("laugh", "shout"):
                lg[max(0, int(sgm["s"]) - 1):min(n, int(sgm["e"]) + 2)] = 1.0
        parts.append(lg)
        weights.append(0.25)
        h = sum(w * p for w, p in zip(weights, parts)) / sum(weights)
        k = np.exp(-0.5 * (np.arange(-6, 7) / 3) ** 2)
        return np.convolve(h, k / k.sum(), mode="same")

    def detect(self):
        if self.analyzed < 180:
            return
        h = self.hype()
        upto = int(self.analyzed - AFTER)
        if upto <= 0:
            return
        thr = max(float(settings().get("live_threshold", 0.35)), float(np.percentile(h[:upto], 96)))
        lo = int(max(0, self.last_peak + COOLDOWN, upto - 3 * TICK - AFTER))
        if upto <= lo:
            return
        k = lo + int(np.argmax(h[lo:upto]))
        self.save_state(hype_now=round(float(h[k]), 3), hype_needed=round(thr, 3))
        if h[k] < thr:
            return
        self.last_peak = k
        self.n += 1
        self.write_moment(self.n, k, float(h[k]), h)

    def write_moment(self, n: int, peak: int, val: float, h: np.ndarray):
        a, b = max(0.0, peak - 75), min(self.analyzed, peak + 35)
        why = []
        if len(self.chat) > 30:
            near = [m for t, m in self.chat if peak - 2 <= t - 4 <= peak + 10]
            clips = sum(1 for m in near if re.search(r"\bclip", m, re.I))
            why.append(f"{len(near)} chat messages in 12 s" + (f", \"clip\" ×{clips}" if clips else ""))
        if any(s["kind"] == "laugh" and peak - 3 <= s["s"] <= peak + 6 for s in self.segs):
            why.append("laughing")
        lines = [s for s in self.segs if a <= s["s"] < b]
        md = [f"# Live moment {n} · {self.info.get('streamer')} · peak at {hms(peak)} of the recording",
              f"Signals: {' · '.join(why) or 'loudness'} (hype {val:.2f})", "",
              "Decide with reference/final-round.md. Not worth it → cf live skip " + str(n),
              f"Worth it → write {self.d / 'moments' / f'moment_{n:03d}.decision.json'} with start, end (times as below),"
              " hook, caption, hashtags, type; then cf live cut " + str(n), "", "Transcript:"]
        for s in lines:
            mark = "🔥 " if abs(s["s"] - peak) < 6 else ""
            txt = s["text"] if s["kind"] == "speech" else f"({s['kind']})" if s["kind"] != "music" else "♪ music ♪"
            md.append(f"[{hms(s['s'])}] {mark}{txt}")
        (self.d / "moments" / f"moment_{n:03d}.md").write_text("\n".join(md) + "\n")
        write_json(self.d / "moments" / f"moment_{n:03d}.json", {"n": n, "peak": peak, "hype": val,
                                                                    "from": a, "to": b, "why": why})
        print(f"[live] moment {n} at {hms(peak)} ({val:.2f})", flush=True)

    def prune(self, keep_minutes: float):
        cutoff = self.analyzed - keep_minutes * 60
        pending = [read_json(j, {}) for j in (self.d / "moments").glob("moment_*.json")
                   if not j.name.endswith(".decision.json")]
        for p in pending:
            n = p.get("n")
            if n and not any((self.d / "moments" / f"moment_{n:03d}{s}").exists() for s in (".done", ".skip")):
                cutoff = min(cutoff, p.get("from", cutoff) - 10)
        for x in self.index:
            if x["end"] < cutoff:
                (self.d / "rec" / x["file"]).unlink(missing_ok=True)

    def run(self):
        def bye(*_):
            self.stop = True
            for p in self.proc or ():
                try:
                    p.kill()
                except Exception:
                    pass
            self.save_state(status="stopped")
            sys.exit(0)
        signal.signal(signal.SIGTERM, bye)
        threading.Thread(target=self.record, daemon=True).start()
        threading.Thread(target=self.chat_irc, daemon=True).start()
        self.save_state(status="recording")
        keep = float(self.st.get("keep_minutes", 45))
        while not self.stop:
            time.sleep(TICK)
            self.collect()
            # the newest segment is still being written: leave it for the next pass
            if self.pending:
                newest = max(self.index, key=lambda x: x["end"])
                ready = [x for x in self.pending if x is not newest]
                rest = [x for x in self.pending if x is newest]
                self.pending = ready
                self.listen_new()
                self.pending = rest
            self.detect()
            self.prune(keep)
            rec = max((x["end"] for x in self.index), default=0)
            self.save_state(recorded=rec, analyzed=self.analyzed, chat=len(self.chat), status="recording")


def run_background(key: str) -> None:
    Runner(key).run()
