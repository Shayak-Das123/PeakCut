"""Build and render curated viral reels for live streams (CaseOh, Kai Cenat, etc.).
Enforces the 9-point viral quality standard:
1. 100% audio/video/caption synchronization with 0 delay.
2. Hook card duration set to exactly 50% of the clip length.
3. Hook Y = 800 (seam), Caption Y = 1420 (lower third), with zero vertical overlap.
4. Editorial hooks: punchy, bold, max 8 words, present tense, NO raw broken quotes.
5. Subagent-driven curation with Setup + Climax + Payoff.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ClipFarm" / "scripts"))
sys.path.insert(0, str(ROOT))

from clipfarm import captions, render as cf_render, util
from clipfarm.util import read_json, write_json, stream_dir
from scripts.render_elite_clips import load_stream_segments, render_curated_reel

def sanitize_hook(hook: str) -> str:
    h = hook.strip()
    h = re.sub(r'^["\']|["\']$', '', h)
    h = re.sub(r'\s*-\s*[A-Za-z0-9_ ]+$', '', h)
    h = h.replace('"', '').replace("'", "")
    h = h.rstrip('.!?,;:')
    words = h.split()
    if len(words) > 8:
        h = " ".join(words[:8])
    return h

def build_stream_reels(stream_key: str):
    d = stream_dir(stream_key)
    info = read_json(d / "info.json", {})
    streamer = info.get("streamer") or "Streamer"
    
    curated_file = d / "curated_picks.json"
    if not curated_file.exists():
        print(f"[{stream_key}] curated_picks.json not found yet.")
        return False
        
    picks_data = read_json(curated_file, {})
    picks = picks_data.get("curated_picks") or picks_data.get("picks") or []
    if not picks and isinstance(picks_data, list):
        picks = picks_data
        
    if not picks:
        print(f"[{stream_key}] No picks in {curated_file}")
        return False

    all_words = read_json(d / "words.json", []) or []
    segments = load_stream_segments(d)
    if not segments:
        print(f"[{stream_key}] No buffer segments found in {d / 'buffer'}")
        return False

    print(f"[{stream_key}] Processing {len(picks)} curated picks from subagent...")
    final_picks = []

    for idx, p in enumerate(picks, 1):
        cid = f"clip_{idx:02d}"
        s = float(p.get("start", 0))
        e = float(p.get("end", s + 25))
        dur = round(e - s, 2)
        
        hook = sanitize_hook(p.get("hook") or p.get("title") or f"{streamer} GTA 6 Moment")
        caption = p.get("caption") or f"{streamer} has an insane moment in GTA 6!"
        hashtags = p.get("hashtags") or [f"#{streamer.lower().replace(' ', '')}", "#gta6", "#gaming", "#viral"]
        score = int(p.get("score", 95))

        # Words in clip (relative to clip start)
        words_in_clip = [w for w in all_words if w["e"] >= s and w["s"] <= e]
        rel_words = [
            {
                "w": w["w"],
                "s": round(max(0.0, w["s"] - s), 2),
                "e": round(max(0.0, w["e"] - s), 2)
            }
            for w in words_in_clip
        ]
        
        # Render reel
        meta = render_curated_reel(
            stream_key=stream_key,
            streamer=streamer,
            cid=cid,
            start_sec=s,
            end_sec=e,
            hook=hook,
            caption=caption,
            hashtags=hashtags,
            score=score,
            words_in_clip=words_in_clip,
            segments=segments
        )
        
        if meta:
            final_picks.append({
                "id": cid,
                "file": meta.get("file"),
                "stream": stream_key,
                "streamer": streamer,
                "platform": info.get("platform", "twitch"),
                "title": hook,
                "start": s,
                "end": e,
                "length": dur,
                "hook": hook,
                "hook_duration": round(dur * 0.50, 1),
                "hook_y": 800,
                "cap_y": 1420,
                "caption": caption,
                "hashtags": hashtags,
                "score": score,
                "layout": "stack",
                "words": len(rel_words),
                "words_list": rel_words
            })

    # Save final picks & progress
    write_json(d / "picks.json", {"curated_picks": final_picks})
    write_json(d / "final.json", {"curated_picks": final_picks})
    write_json(d / "progress.json", {
        "status": "Done curating with Antigravity Subagent",
        "time": time.time(),
        "total_clips": len(final_picks)
    })
    print(f"[{stream_key}] Successfully rendered {len(final_picks)}/{len(picks)} elite reels!")
    return True

if __name__ == "__main__":
    sk = sys.argv[1] if len(sys.argv) > 1 else "twitch-caseoh"
    build_stream_reels(sk)
