"""cf learn: reads YOUR channel's Shorts views and finds out which kinds of clips work for you."""
from __future__ import annotations

import difflib
import json
import re
import statistics
import subprocess

from .util import LEDGER, TASTE, read_json, say, write_json, ytdlp


def _shorts(channel: str) -> list[dict]:
    base = channel.rstrip("/")
    if not base.startswith("http"):
        base = "https://www.youtube.com/" + (base if base.startswith("@") else "@" + base)
    url = re.sub(r"/(shorts|videos|featured|streams)$", "", base) + "/shorts"
    p = subprocess.run(ytdlp() + ["--flat-playlist", "-J", "--playlist-end", "200", url], capture_output=True,
                       text=True, errors="replace", timeout=300)
    if p.returncode != 0:
        raise SystemExit("Couldn't read that channel's Shorts. Paste the channel link, like https://www.youtube.com/@you")
    return [{"title": e.get("title") or "", "views": e.get("view_count") or 0,
             "url": e.get("url") or f"https://www.youtube.com/shorts/{e.get('id')}"}
            for e in json.loads(p.stdout).get("entries") or []]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", s.lower()).strip()


def run(channel: str) -> None:
    shorts = _shorts(channel)
    if not shorts:
        raise SystemExit("No Shorts on that channel yet. Post a few clips, then run this again.")
    led = read_json(LEDGER, []) or []
    views = [s["views"] for s in shorts if s["views"]]
    med = statistics.median(views) if views else 0
    matched = 0
    for c in led:
        hook = _norm(c.get("hook", ""))
        if not hook:
            continue
        best, score = None, 0.0
        for s in shorts:
            r = difflib.SequenceMatcher(None, hook, _norm(s["title"])).ratio()
            if r > score:
                best, score = s, r
        posted_url = (c.get("posted") or {}).get("youtube")
        if posted_url:
            best = next((s for s in shorts if s["url"].split("/")[-1] in posted_url), best)
            score = 1.0 if best else score
        if best and score >= 0.62:
            c["views"] = best["views"]
            c["matched_title"] = best["title"]
            matched += 1
    write_json(LEDGER, led)
    rows = [c for c in led if c.get("views") is not None]
    if len(rows) < 5:
        say(f"Matched {len(rows)} of your clips to your Shorts (channel median {med:,.0f} views). "
            "Come back after you've posted at least 5 clips from Clip Farm.")
        return

    def table(key):
        groups: dict[str, list[int]] = {}
        for c in rows:
            groups.setdefault((c.get(key) or "other").lower(), []).append(c["views"])
        out = []
        for k, v in groups.items():
            m = statistics.median(v)
            out.append((k, len(v), m, (m / med) if med else 0))
        return sorted(out, key=lambda r: -r[3])

    by_type, by_streamer = table("type"), table("streamer")
    lengths = sorted(rows, key=lambda c: c.get("length") or 0)
    half = len(lengths) // 2
    short_med = statistics.median([c["views"] for c in lengths[:half]]) if half else 0
    long_med = statistics.median([c["views"] for c in lengths[half:]]) if half else 0
    line = "On this channel: " + ", ".join(f"{k} {x:.1f}× median" for k, n, m, x in by_type[:6] if n >= 2)
    if half:
        line += f"; shorter clips {short_med:,.0f} vs longer {long_med:,.0f} median views"
    lines = [line, "",
             f"# What works on your channel ({len(rows)} clips matched, channel median {med:,.0f} views)", "",
             "| Moment type | Clips | Median views | × your median |", "|---|---|---|---|"]
    lines += [f"| {k} | {n} | {m:,.0f} | {x:.1f}× |" for k, n, m, x in by_type]
    lines += ["", "| Streamer | Clips | Median views | × your median |", "|---|---|---|---|"]
    lines += [f"| {k} | {n} | {m:,.0f} | {x:.1f}× |" for k, n, m, x in by_streamer]
    lines += ["", "Picking uses the first line above: give a few more points to the types and streamers that win here,"
              " and be stricter with the ones that don't. Two clips is not a pattern; five starts to be one."]
    TASTE.write_text("\n".join(lines) + "\n")
    say("\n".join(lines))
    say(f"\n✓ saved to {TASTE}. The next scan's chunks carry it, so the picks lean toward what works for you.")
