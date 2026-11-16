"""cf find: the biggest streams of a game right now, on Twitch and YouTube, ranked by viewers."""
from __future__ import annotations

import json
import re
import subprocess
import urllib.parse

from .fetch import twitch_gql
from .util import WORK, say, today, write_json, ytdlp

SKIP = re.compile(r"24/7|countdown|giveaway|!drop|rerun|re-?stream|music|lofi|radio|asmr", re.I)


def twitch(game: str, limit: int, lang: str) -> list[dict]:
    q = ('query { game(name: "%s") { viewersCount streams(first: %d, options: {languages: ["%s"]}) { edges { node { '
         'title viewersCount createdAt broadcaster { login displayName } archiveVideo { id } } } } } }'
         % (game.replace('"', ""), min(100, limit * 3), lang))
    try:
        d = twitch_gql(q)
    except Exception as e:
        say(f"! Twitch didn't answer: {e}")
        return []
    g = (d.get("data") or {}).get("game")
    if not g:
        say(f"! Twitch has no category called {game!r}")
        return []
    out = []
    for e in (g.get("streams") or {}).get("edges") or []:
        n = e["node"]
        b = n.get("broadcaster") or {}
        out.append({"platform": "twitch", "streamer": b.get("displayName") or b.get("login"),
                    "title": n.get("title") or "", "viewers": n.get("viewersCount") or 0, "live": True,
                    "url": f"https://www.twitch.tv/{b.get('login')}", "started": n.get("createdAt"),
                    "so_far": (f"https://www.twitch.tv/videos/{n['archiveVideo']['id']}"
                               if n.get("archiveVideo") else None)})
    return out


def youtube(query: str, limit: int) -> list[dict]:
    url = "https://www.youtube.com/results?search_query=" + urllib.parse.quote_plus(query) + "&sp=EgJAAQ%253D%253D"
    p = subprocess.run(ytdlp() + ["--flat-playlist", "-J", "--playlist-end", str(limit * 3), url],
                       capture_output=True, text=True, errors="replace", timeout=120)
    if p.returncode != 0:
        say("! YouTube search failed (cf setup updates yt-dlp)")
        return []
    out = []
    for e in json.loads(p.stdout).get("entries") or []:
        out.append({"platform": "youtube", "streamer": e.get("channel") or "", "title": e.get("title") or "",
                    "viewers": e.get("concurrent_view_count") or 0, "live": e.get("live_status") == "is_live",
                    "url": e.get("url") or f"https://www.youtube.com/watch?v={e.get('id')}", "so_far": None})
    return out


def run(game: str, query: str, limit: int, lang: str) -> list[dict]:
    q = query or ("GTA 6" if "Grand Theft Auto VI" in game else game)
    rows = twitch(game, limit, lang) + youtube(q, limit)
    rows = [r for r in rows if not SKIP.search(r["title"])]
    rows.sort(key=lambda r: -r["viewers"])
    rows = rows[: limit]
    path = WORK / "found" / f"{today()}.json"
    write_json(path, rows)
    if not rows:
        say("Nothing live right now. Try --game \"Grand Theft Auto V\" or paste a link to a stream you like.")
        return rows
    say(f"Biggest {game} streams right now ({len(rows)}):")
    for i, r in enumerate(rows, 1):
        say(f"{i:>2}. {r['viewers']:>7,}  {r['platform']:<7} {r['streamer'][:22]:<22} {r['title'][:60]}")
        say(f"     live: {r['url']}" + (f"   ·   everything so far: {r['so_far']}" if r.get("so_far") else ""))
    say(f"\nSaved to {path}. Clip what already happened with  cf scan <so-far link>,"
        " or follow one as it happens with  cf live start <live link>.")
    return rows
