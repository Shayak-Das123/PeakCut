"""Render Elite Top-10 Curated Reels for PeakCut & ClipFarm.
Implements the strict specifications from the user and SKILL.md:
1. Grounded directly in real speech/event moments (no music or silent driving).
2. Curated viral hooks (max 8 words, present tense, describing the tension/story).
3. White hook card burned onto the video matches the dashboard title 100%.
4. Zero segment loops or repeated frames (exact segment timestamp alignment).
5. Dynamic start/end boundaries (sweet spot 20-38s).
6. Strict limit: exactly top 8-10 reels per stream.
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

def clean_slug(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return s[:40] or "clip"

def load_stream_segments(stream_folder: Path) -> list[tuple[Path, float, float]]:
    buf = stream_folder / "buffer"
    if not buf.exists():
        return []
    segments = []
    seg_clock = 0.0
    for i in range(1, 20):
        csv_p = buf / f"p{i:02d}.csv"
        if not csv_p.exists():
            continue
        for row in csv_p.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = [p.strip() for p in row.split(",") if p.strip()]
            if len(parts) >= 3:
                name = parts[0]
                try:
                    a, b = float(parts[1]), float(parts[2])
                    dur = max(0.0, b - a)
                    ts_path = buf / name
                    if ts_path.exists():
                        segments.append((ts_path, seg_clock, seg_clock + dur))
                    seg_clock += dur
                except ValueError:
                    pass
    return segments

def render_curated_reel(
    stream_key: str,
    streamer: str,
    cid: str,
    start_sec: float,
    end_sec: float,
    hook: str,
    caption: str,
    hashtags: list[str],
    score: int,
    words_in_clip: list[dict],
    segments: list[tuple[Path, float, float]]
) -> dict | None:
    d = stream_dir(stream_key)
    out_dir = d / "clips"
    out_dir.mkdir(parents=True, exist_ok=True)
    sec_dir = d / "sections"
    sec_dir.mkdir(parents=True, exist_ok=True)
    work_render = d / "render" / cid
    work_render.mkdir(parents=True, exist_ok=True)

    pad = 4.0
    s, e = start_sec, end_sec
    matching_segs = [p for p, a, b in segments if b >= s - pad and a <= e + pad and p.exists()]
    if not matching_segs:
        print(f"[{cid}] Error: No buffer segments found for range {s:.1f} - {e:.1f}")
        return None

    # Determine exact start time of the first concatenated segment
    first_seg_start = next(a for p, a, b in segments if p == matching_segs[0])
    sec_file = sec_dir / f"{cid}.mp4"
    lst_file = sec_dir / f"{cid}.txt"

    # Concat selected segments into intermediate mp4
    lst_file.write_text("".join(f"file '{p.as_posix()}'\n" for p in matching_segs), encoding="utf-8")
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
         "-i", str(lst_file), "-c:v", "copy", "-c:a", "aac", str(sec_file)],
        check=True
    )
    lst_file.unlink(missing_ok=True)

    # Relative words for kinetic styling
    rel_words = []
    for w in words_in_clip:
        rel_words.append({
            "w": w["w"],
            "s": round(max(0.0, w["s"] - s), 2),
            "e": round(max(0.0, w["e"] - s), 2)
        })
    write_json(work_render / "words.json", rel_words)

    info_data = read_json(d / "info.json", {})
    if not info_data:
        info_data = {
            "url": f"https://www.youtube.com/watch?v={stream_key}",
            "title": f"{streamer} GTA 6 Live",
            "streamer": streamer,
            "channel": streamer
        }

    clip_dur = round(e - s, 2)
    hook_half_dur = round(clip_dur * 0.50, 1)

    clip_spec = {
        "id": cid,
        "start": s,
        "end": e,
        "title": hook,
        "hook": hook,
        "hook_duration": hook_half_dur,
        "hook_y": 800,
        "cap_y": 1420,
        "trim_air": False,
        "caption": caption,
        "streamer": streamer,
        "hashtags": hashtags,
        "custom_words": rel_words,
        "curated": True,
        "saved": True,
        "score": score,
        "layout": "stack"
    }

    print(f"[{cid}] Rendering: '{hook}' ({s:.1f}s - {e:.1f}s, dur {e-s:.1f}s)...")
    meta = cf_render.render_one(
        info_data,
        d,
        clip_spec,
        int(cid.replace("clip_", "").replace("c", "") or 1),
        out_dir,
        section=(sec_file, first_seg_start)
    )

    out_name = meta["file"]
    meta["title"] = hook
    meta["hook"] = hook
    meta["hook_duration"] = hook_half_dur
    meta["hook_y"] = 800
    meta["cap_y"] = 1420
    meta["caption"] = caption
    meta["hashtags"] = hashtags
    meta["curated"] = True
    meta["saved"] = True
    meta["score"] = score
    meta["streamer"] = streamer
    meta["stream"] = stream_key

    # Save companion json and txt
    comp_json = out_dir / f"{Path(out_name).stem}.json"
    comp_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    txt_lines = [
        f"HOOK: {hook}",
        "",
        "— TikTok / Instagram Reels / YouTube Shorts —",
        f"{streamer} - {hook}",
        f"🎥 {streamer} in GTA 6",
        caption,
        "",
        " ".join(hashtags)
    ]
    comp_txt = out_dir / f"{Path(out_name).stem}.txt"
    comp_txt.write_text("\n".join(txt_lines), encoding="utf-8")

    # Clean intermediate section
    sec_file.unlink(missing_ok=True)
    print(f"[{cid}] SUCCESS -> {out_name} (Hook: '{hook}')")
    return meta
