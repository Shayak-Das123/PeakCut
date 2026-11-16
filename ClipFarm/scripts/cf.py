#!/usr/bin/env python3
"""Clip Farm: long streams in, ready-to-post vertical clips out.

  cf setup                      install what's missing, then check everything
  cf doctor                     check everything, change nothing
  cf settings [key value]       show or change a setting (cookies chrome, clips_per_stream 3…)
  cf find [--game "Grand Theft Auto VI"] [--limit 10]   today's biggest streams (Twitch + YouTube)
  cf scan <url> [<url>…]        audio, transcript, chat, replays, viewer clips → the hype curve → chunks
  cf check <key>                are all the chunk picks there and valid?
  cf shortlist <key>            merge the picks with the signals → shortlist.md for the final round
  cf render <key> [--only c02]  make the clips listed in final.json
  cf review [date]              open the review page of a day's clips
  cf live start|next|cut|stop|status   clip a stream while it's still live
  cf learn <your channel url>   read your own Shorts' views and learn what works for you
  cf posted <clip file> <url>   mark a clip as posted
  cf status                     what's in progress
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clipfarm import util  # noqa: E402
from clipfarm.util import WORK, hms, read_json, say, settings, stream_dir  # noqa: E402


def resolve_key(arg: str) -> str:
    """A stream key, a URL we already scanned, or 'last'."""
    root = WORK / "streams"
    if arg == "last":
        ds = sorted((p for p in root.glob("*") if (p / "state.json").exists()),
                    key=lambda p: (p / "state.json").stat().st_mtime)
        if not ds:
            raise SystemExit("nothing scanned yet")
        return ds[-1].name
    if (root / arg).exists():
        return arg
    for p in root.glob("*"):
        st = read_json(p / "state.json", {})
        if st.get("url") == arg or arg in str(st.get("url", "")) or p.name.endswith(arg):
            return p.name
    raise SystemExit(f"no scanned stream matches {arg!r} (cf status lists them)")


def cmd_scan(a):
    from clipfarm import stream
    keys = []
    if len(a.urls) == 1:
        keys.append(stream.scan(a.urls[0], a.cookies, force=a.force))
    else:
        # several streams: downloads overlap, the listening runs one after the other (one GPU)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(min(3, len(a.urls))) as ex:
            for k in ex.map(lambda u: _safe_scan(stream, u, a), a.urls):
                if k:
                    keys.append(k)
    say("\nNext: read every chunk and write the picks (reference/picking.md), then: cf shortlist <key>")
    for k in keys:
        st = read_json(stream_dir(k) / "state.json", {})
        say(f"  {k}  {st.get('chunks')} chunks  ({stream_dir(k) / 'chunks'})")


def _safe_scan(stream, url, a):
    try:
        return stream.scan(url, a.cookies, force=a.force)
    except SystemExit as e:
        say(f"! {url}: {e}")
    except Exception as e:
        say(f"! {url}: {e}")
    return None


def cmd_check(a):
    from clipfarm import shortlist
    d = stream_dir(resolve_key(a.key))
    problems, moments = shortlist.check_picks(d)
    if problems:
        say("✗ " + "\n✗ ".join(problems))
        sys.exit(1)
    say(f"✓ all picks files valid · {len(moments)} moments")


def cmd_shortlist(a):
    from clipfarm import shortlist
    d = stream_dir(resolve_key(a.key))
    kept = shortlist.build(d, a.top)
    say(f"✓ {len(kept)} candidates → {d / 'shortlist.md'}")
    for c in kept:
        say(f"  {c['id']}  {hms(c['start'])}–{hms(c['end'])}  {c['combined']:5.1f}  {c['type']:<14} {c.get('title', '')[:60]}")
    say("Next: the final round (reference/final-round.md) → final.json, then: cf render " + d.name)


def cmd_render(a):
    from clipfarm import render
    key = resolve_key(a.key)
    render.render_stream(key, only=a.only, layout=a.layout)


def cmd_review(a):
    from clipfarm import review
    p = review.build(a.date or util.today())
    say(f"✓ {p}")
    if not a.no_open:
        import subprocess
        subprocess.run(["open", str(p)])


def cmd_find(a):
    from clipfarm import find
    find.run(a.game, a.query, a.limit, a.lang)


def cmd_live(a):
    from clipfarm import live
    live.main(a)


def cmd_learn(a):
    from clipfarm import learn
    learn.run(a.channel)


def cmd_posted(a):
    from clipfarm import review
    review.mark_posted(a.clip, a.url)


def cmd_setup(a):
    from clipfarm import setup
    setup.setup(doctor_only=False)


def cmd_doctor(a):
    from clipfarm import setup
    ok = setup.setup(doctor_only=True)
    sys.exit(0 if ok else 1)


def cmd_settings(a):
    s = settings()
    if a.key:
        if a.value is None:
            say(json.dumps(s.get(a.key), indent=1))
            return
        v = a.value
        cur = util.DEFAULTS.get(a.key, s.get(a.key))
        if isinstance(cur, bool):
            v = v.lower() in ("1", "true", "yes", "on")
        elif isinstance(cur, int):
            v = int(v)
        elif isinstance(cur, float):
            v = float(v)
        elif isinstance(cur, list):
            v = [x.strip() for x in v.split(",") if x.strip()]
        elif a.key == "cookies" and v.lower() in ("none", "off", "no", ""):
            v = ""
        s[a.key] = v
        util.save_settings(s)
        say(f"✓ {a.key} = {v!r}")
    else:
        say(json.dumps(s, indent=1))


def cmd_status(a):
    root = WORK / "streams"
    rows = []
    for p in sorted(root.glob("*")):
        st = read_json(p / "state.json", {})
        if not st:
            continue
        picks = len(list((p / "picks").glob("chunk_*.json"))) if (p / "picks").exists() else 0
        stage = ("rendered" if st.get("rendered") else "final" if (p / "final.json").exists() else
                 "shortlist" if (p / "shortlist.json").exists() else
                 f"picking {picks}/{st.get('chunks')}" if st.get("chunks") else "scanning")
        rows.append((st.get("scanned") or st.get("started") or 0, p.name, st.get("streamer", "?"),
                     hms(st.get("duration", 0)), stage))
    if not rows:
        say("nothing yet. Start with: cf scan <url>  (or cf find)")
    for _, k, s, d, stage in sorted(rows):
        say(f"  {k:<28} {s[:22]:<22} {d:>8}  {stage}")


def main():
    ap = argparse.ArgumentParser(prog="cf", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup").set_defaults(f=cmd_setup)
    sub.add_parser("doctor").set_defaults(f=cmd_doctor)
    p = sub.add_parser("settings"); p.add_argument("key", nargs="?"); p.add_argument("value", nargs="?")
    p.set_defaults(f=cmd_settings)
    p = sub.add_parser("find"); p.add_argument("--game", default="Grand Theft Auto VI")
    p.add_argument("--query", default=""); p.add_argument("--limit", type=int, default=10)
    p.add_argument("--lang", default="en"); p.set_defaults(f=cmd_find)
    p = sub.add_parser("scan"); p.add_argument("urls", nargs="+"); p.add_argument("--cookies", default=None)
    p.add_argument("--force", action="store_true"); p.set_defaults(f=cmd_scan)
    p = sub.add_parser("check"); p.add_argument("key"); p.set_defaults(f=cmd_check)
    p = sub.add_parser("shortlist"); p.add_argument("key"); p.add_argument("--top", type=int, default=12)
    p.set_defaults(f=cmd_shortlist)
    p = sub.add_parser("render"); p.add_argument("key"); p.add_argument("--only", default=None)
    p.add_argument("--layout", default=None, choices=[None, "stack", "fullcam", "gameplay"]); p.set_defaults(f=cmd_render)
    p = sub.add_parser("review"); p.add_argument("date", nargs="?"); p.add_argument("--no-open", action="store_true")
    p.set_defaults(f=cmd_review)
    p = sub.add_parser("live"); p.add_argument("action", choices=["start", "next", "cut", "stop", "status", "skip"])
    p.add_argument("target", nargs="?"); p.add_argument("--wait", type=float, default=900)
    p.add_argument("--keep-minutes", type=float, default=45); p.add_argument("--cookies", default=None)
    p.set_defaults(f=cmd_live)
    p = sub.add_parser("learn"); p.add_argument("channel"); p.set_defaults(f=cmd_learn)
    p = sub.add_parser("posted"); p.add_argument("clip"); p.add_argument("url"); p.set_defaults(f=cmd_posted)
    sub.add_parser("status").set_defaults(f=cmd_status)
    if len(sys.argv) >= 3 and sys.argv[1] == "live-run":      # the background recorder (started by cf live start)
        from clipfarm import live
        live.run_background(sys.argv[2])
        return
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
