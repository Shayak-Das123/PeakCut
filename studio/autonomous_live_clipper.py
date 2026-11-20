"""Autonomous Live Stream Clipper & Viral Reel Producer for PeakCut / ClipFarm.
Monitors top streamers (Kai Cenat, IShowSpeed, Clix, CaseOh, xQc, etc.) around the clock.
When a streamer goes live, it:
1. Buffers live 6-second segments and transcribes speech in real-time via Sherpa-ONNX.
2. Detects excitement peaks (screams, laughter, high-volume moments).
3. Uses Gemini / Subagent to curate viral narrative moments (Setup + Action + Payoff).
4. Crafts bold present-tense editorial hooks (max 8 words, no raw quotes).
5. Renders 1080x1920 60fps vertical reels using RTX 4050 GPU NVENC:
   - Hook duration: exactly 50% of clip length.
   - Hook Y: 800, Caption Y: 1420 (zero overlap, 620px clearance).
   - Captions: 100% synchronized with spoken audio (0s delay).
6. Preserves all existing library clips and adds new ones continuously.
7. Automatically cleans raw temporary buffer chunks to maintain disk headroom.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ClipFarm" / "scripts"))
sys.path.insert(0, str(ROOT))

import numpy as np
from studio import ai, config, ctx, scheduler
from studio import transcribe as st_transcribe
from clipfarm.util import read_json, write_json, stream_dir
from scripts.render_elite_clips import load_stream_segments, render_curated_reel

STREAMLINK_BIN = str(ROOT / ".venv" / "Scripts" / "streamlink.exe")
WORK_DIR = ROOT / "ClipFarm" / "work" / "streams"
WORK_DIR.mkdir(parents=True, exist_ok=True)

TARGET_STREAMERS = [
    {
        "id": "twitch-kaicenat",
        "streamer": "Kai Cenat",
        "platform": "twitch",
        "url": "https://www.twitch.tv/kaicenat",
        "channel": "kaicenat",
        "layout": "stack"
    },
    {
        "id": "youtube-ishowspeed",
        "streamer": "IShowSpeed",
        "platform": "youtube",
        "url": "https://www.youtube.com/@IShowSpeed/live",
        "channel": "IShowSpeed",
        "layout": "stack"
    },
    {
        "id": "twitch-clix",
        "streamer": "Clix",
        "platform": "twitch",
        "url": "https://www.twitch.tv/clix",
        "channel": "clix",
        "layout": "stack"
    },
    {
        "id": "twitch-caseoh",
        "streamer": "CaseOh",
        "platform": "twitch",
        "url": "https://www.twitch.tv/caseoh_",
        "channel": "caseoh_",
        "layout": "stack"
    },
    {
        "id": "twitch-xqc",
        "streamer": "xQc",
        "platform": "twitch",
        "url": "https://www.twitch.tv/xqc",
        "channel": "xqc",
        "layout": "stack"
    },
    {
        "id": "youtube-carryminati",
        "streamer": "CarryMinati",
        "platform": "youtube",
        "url": "https://www.youtube.com/@CarryisLive/live",
        "channel": "CarryisLive",
        "language": "hindi",
        "layout": "stack"
    },
    {
        "id": "youtube-technogamerz",
        "streamer": "Techno Gamerz",
        "platform": "youtube",
        "url": "https://www.youtube.com/@TechnoGamerzOfficial/live",
        "channel": "TechnoGamerzOfficial",
        "language": "hindi",
        "layout": "stack"
    },
    {
        "id": "youtube-totalgaming",
        "streamer": "Total Gaming",
        "platform": "youtube",
        "url": "https://www.youtube.com/@TotalGaming093/live",
        "channel": "TotalGaming093",
        "language": "hindi",
        "layout": "stack"
    },
    {
        "id": "youtube-scout",
        "streamer": "Scout",
        "platform": "youtube",
        "url": "https://www.youtube.com/@scoutOP/live",
        "channel": "scoutOP",
        "language": "hindi",
        "layout": "stack"
    },
    {
        "id": "youtube-mortal",
        "streamer": "Mortal",
        "platform": "youtube",
        "url": "https://www.youtube.com/@Morticall/live",
        "channel": "Morticall",
        "language": "hindi",
        "layout": "stack"
    },
    {
        "id": "youtube-liveinsaan",
        "streamer": "Live Insaan",
        "platform": "youtube",
        "url": "https://www.youtube.com/@LiveInsaan/live",
        "channel": "LiveInsaan",
        "language": "hindi",
        "layout": "stack"
    },
    {
        "id": "twitch-jynxzi",
        "streamer": "Jynxzi",
        "platform": "twitch",
        "url": "https://www.twitch.tv/jynxzi",
        "channel": "jynxzi",
        "layout": "stack"
    },
    {
        "id": "twitch-tarik",
        "streamer": "Tarik",
        "platform": "twitch",
        "url": "https://www.twitch.tv/tarik",
        "channel": "tarik",
        "layout": "stack"
    }
]

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

def get_live_hls(url: str) -> str | None:
    try:
        p = subprocess.run(
            [STREAMLINK_BIN, "--stream-url", url, "720p,best"],
            capture_output=True,
            text=True,
            timeout=15
        )
        out = p.stdout.strip()
        if out.startswith("http"):
            return out
    except Exception as e:
        print(f"[Streamlink] Error checking {url}: {e}")
    return None

def get_stream_meta(url: str) -> dict | None:
    try:
        p = subprocess.run(
            [STREAMLINK_BIN, "--json", url],
            capture_output=True,
            text=True,
            timeout=12
        )
        data = json.loads(p.stdout)
        meta = data.get("metadata", {})
        if meta and meta.get("title"):
            return meta
    except Exception:
        pass
    return None

def record_stream_buffer(stream_cfg: dict, hls_url: str, meta: dict, record_seconds: int = 300) -> Path:
    stream_id = stream_cfg["id"]
    timestamp_slug = time.strftime("%Y%m%d_%H%M%S")
    session_key = f"{stream_id}_{timestamp_slug}"
    s_dir = WORK_DIR / session_key
    buf_dir = s_dir / "buffer"
    clips_dir = s_dir / "clips"
    buf_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    info_data = {
        "id": session_key,
        "base_stream": stream_id,
        "title": meta.get("title") or f"{stream_cfg['streamer']} Live",
        "streamer": stream_cfg["streamer"],
        "channel": stream_cfg["channel"],
        "platform": stream_cfg["platform"],
        "url": stream_cfg["url"],
        "category": meta.get("category", "Grand Theft Auto VI"),
        "layout": stream_cfg.get("layout", "stack"),
        "is_live": True,
        "created_at": time.time()
    }
    write_json(s_dir / "info.json", info_data)

    csv_file = "seg.csv"
    ts_pattern = "seg_%05d.ts"
    ffmpeg_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning",
        "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "10",
        "-i", hls_url,
        "-map", "0:v:0", "-map", "0:a:0", "-c", "copy",
        "-f", "segment", "-segment_time", "6",
        "-reset_timestamps", "1", "-segment_format", "mpegts",
        "-segment_list", csv_file, "-segment_list_type", "csv",
        ts_pattern,
        "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "pipe:1"
    ]

    print(f"[{stream_cfg['streamer']}] Ingesting {record_seconds}s of live stream into {session_key}...")
    proc = subprocess.Popen(
        ffmpeg_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(buf_dir)
    )

    all_words = []
    peaks = []
    chunk_size = 16000 * 2 * 6
    base_sec = 0.0
    t0 = time.time()

    try:
        while proc.poll() is None:
            if time.time() - t0 > record_seconds:
                print(f"[{stream_cfg['streamer']}] Reached target duration ({record_seconds}s). Stopping recorder.")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()
                break

            raw_audio = proc.stdout.read(chunk_size)
            if not raw_audio:
                break
            samples = len(raw_audio) // 2
            dur_chunk = samples / 16000.0
            audio_np = np.frombuffer(raw_audio, dtype=np.int16)

            slice_samples = int(16000 * 0.25)
            for i in range(0, samples, slice_samples):
                sl = audio_np[i : i + slice_samples]
                if len(sl) > 0:
                    rms = float(np.sqrt(np.mean(sl.astype(np.float32) ** 2)))
                    t_slice = round(base_sec + (i / 16000.0), 2)
                    if rms > 3000:
                        peaks.append({"t": t_slice, "rms": round(rms, 1)})

            try:
                words = st_transcribe.transcribe(audio_np)
                for w in words:
                    w["s"] = round(w["s"] + base_sec, 2)
                    w["e"] = round(w["e"] + base_sec, 2)
                all_words.extend(words)

                write_json(s_dir / "words.json", all_words)
                write_json(s_dir / "peaks.json", peaks)

                lines = []
                cur_line = []
                for w in all_words:
                    cur_line.append(w)
                    if w["w"].endswith((".", "!", "?")) or len(cur_line) >= 12:
                        lines.append({
                            "text": " ".join(x["w"] for x in cur_line),
                            "start": cur_line[0]["s"],
                            "end": cur_line[-1]["e"]
                        })
                        cur_line = []
                if cur_line:
                    lines.append({
                        "text": " ".join(x["w"] for x in cur_line),
                        "start": cur_line[0]["s"],
                        "end": cur_line[-1]["e"]
                    })
                write_json(s_dir / "lines.json", lines)
            except Exception as e:
                print(f"[{stream_cfg['streamer']}] Transcribe error: {e}")

            base_sec += dur_chunk
    except Exception as e:
        print(f"[{stream_cfg['streamer']}] Ingest loop interrupted: {e}")
        try:
            proc.terminate()
        except Exception:
            pass

    return s_dir

def curate_picks_with_ai(s_dir: Path, stream_cfg: dict, meta: dict) -> list[dict]:
    words = read_json(s_dir / "words.json", [])
    lines = read_json(s_dir / "lines.json", [])
    peaks = read_json(s_dir / "peaks.json", [])
    segments = load_stream_segments(s_dir)

    if not segments or not words:
        print(f"[{stream_cfg['streamer']}] No words or segments to curate.")
        return []

    max_video_time = segments[-1][2]
    streamer = stream_cfg["streamer"]
    title = meta.get("title", f"{streamer} Live")

    dialogue_summary = "\n".join(f"[{l['start']:.1f}s - {l['end']:.1f}s] {l['text']}" for l in lines[:60])
    top_peaks = sorted(peaks, key=lambda x: x["rms"], reverse=True)[:15]
    peaks_summary = ", ".join(f"{p['t']}s (RMS: {p['rms']:.0f})" for p in top_peaks)

    is_hindi = stream_cfg.get("language") == "hindi" or any(k in streamer.lower() for k in ["carry", "techno", "gaming", "scout", "mortal", "insaan"])
    lang_guidance = ""
    if is_hindi:
        lang_guidance = f"""
Language & Cultural Context:
The streamer speaks Hindi / Hinglish. Understand Hindi commentary, comedic roasts, drama, gaming slang (e.g., 'bhai', 'arre yaar', 'op', 'gaadi', 'scam', 'police piche pad gayi').
- Craft high-retention on-screen title cards in punchy English or Hinglish (max 7-8 words, bold present tense, e.g. '{streamer} Roasts GTA 6 Police', '{streamer} Ne Cops Ko Bhaga Diya').
- Write engaging post captions in lively Hinglish/English with 1 comment question.
- Hashtags should include Hindi/Indian gaming tags (e.g. ['#{streamer.lower().replace(' ', '')}', '#gta6', '#gamingindia', '#viral', '#carryminati']).
"""

    prompt = f"""You are the Lead Viral Clip Director for YouTube Shorts, TikTok, and Instagram Reels.
Streamer: {streamer}
Stream Title: {title}
Max Available Video Length: {max_video_time:.1f} seconds.
{lang_guidance}

SPEAKER & NARRATIVE DIARIZATION (Streamer vs Character):
In gameplay livestreams (such as GTA 6):
- GAME CHARACTERS / NPCs: Dialogue from in-game characters (Lucia, Jason, Police, Pedestrians, Cutscenes).
- STREAMER: Live voice over microphone, screams, laughter, live commentary, panic, talking to viewers/chat.
VIRAL SELECTION CRITERIA:
1. Understand WHO is speaking: identify whether dialogue belongs to the game character or streamer.
2. The most viral moments feature the STREAMER having an intense, hilarious, or shocked reaction to what an in-game character says or does (e.g. cop chase, roast, unexpected dialogue).
3. Set "speaker_type" to either "streamer_reaction" (streamer reacting to game event/character) or "character_interaction" (streamer conversing/arguing with character).
4. The hook MUST be in bold present tense describing the moment (e.g. '{streamer} Can't Believe Lucia's Answer', '{streamer} Instantly Panics at GTA 6 Cops'). Max 7-8 words, NO raw syllable quotes!

Dialogue Transcript with timestamps:
{dialogue_summary}

Excitement & Loudness Peaks:
{peaks_summary}

Select exactly 3 to 4 viral clips following the 9-point criteria:
1. Duration: 20 to 38 seconds.
2. Clean start and end boundaries: Never cut mid-sentence, never cut off punchlines.
3. Editorial Hook: Max 7-8 words, bold present tense, explaining what happens.
4. Caption: 1-2 punchy sentences describing the moment + 1 question driving viewer comments.
5. Hashtags: 4-6 viral tags (e.g. ['#{streamer.lower().replace(' ', '')}', '#gta6', '#gaming', '#Shorts', '#viral']).
6. Score: 92 to 99.
7. Layout: 'stack'.

All 'start' and 'end' timestamps must be between 0.0 and {max_video_time:.1f}.
"""
    schema = {
        "type": "OBJECT",
        "properties": {
            "curated_picks": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "id": {"type": "STRING"},
                        "start": {"type": "NUMBER"},
                        "end": {"type": "NUMBER"},
                        "hook": {"type": "STRING"},
                        "title": {"type": "STRING"},
                        "caption": {"type": "STRING"},
                        "hashtags": {"type": "ARRAY", "items": {"type": "STRING"}},
                        "speaker_type": {"type": "STRING"},
                        "score": {"type": "NUMBER"},
                        "layout": {"type": "STRING"}
                    },
                    "required": ["start", "end", "hook", "caption", "hashtags", "score"]
                }
            }
        },
        "required": ["curated_picks"]
    }

    picks = []
    models = ai.models() or ["gemini-2.5-flash", "gemini-3.5-flash-lite"]
    for m in models:
        try:
            res, _, _ = ai.call(m, prompt, schema)
            if isinstance(res, dict) and res.get("curated_picks"):
                picks = res["curated_picks"]
                break
        except Exception as e:
            print(f"[{streamer}] AI curation error on model {m}: {e}")

    # Fallback heuristic if AI unavailable
    if not picks:
        print(f"[{streamer}] Using heuristic fallback curation...")
        step = max(30.0, max_video_time / 4)
        cur_t = 2.0
        idx = 1
        while cur_t + 25.0 <= max_video_time and idx <= 4:
            s_cand = cur_t
            e_cand = min(max_video_time, cur_t + 28.0)
            picks.append({
                "id": f"clip_{idx:02d}",
                "start": s_cand,
                "end": e_cand,
                "hook": f"{streamer} Live GTA 6 Moment {idx}",
                "title": f"{streamer} Live GTA 6 Moment {idx}",
                "caption": f"{streamer} with an intense moment live in GTA 6! What would you do in this situation?",
                "hashtags": [f"#{streamer.lower().replace(' ', '')}", "#gta6", "#gaming", "#viral"],
                "score": 93,
                "layout": "stack"
            })
            cur_t += step
            idx += 1

    write_json(s_dir / "curated_picks.json", {"curated_picks": picks})
    return picks

def render_session_picks(s_dir: Path, stream_cfg: dict, picks: list[dict]):
    streamer = stream_cfg["streamer"]
    stream_key = s_dir.name
    all_words = read_json(s_dir / "words.json", [])
    segments = load_stream_segments(s_dir)

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

        words_in_clip = [w for w in all_words if w["e"] >= s and w["s"] <= e]
        rel_words = [
            {"w": w["w"], "s": round(max(0.0, w["s"] - s), 2), "e": round(max(0.0, w["e"] - s), 2)}
            for w in words_in_clip
        ]

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
            clip_record = {
                "id": cid,
                "file": meta.get("file"),
                "stream": stream_key,
                "streamer": streamer,
                "platform": stream_cfg["platform"],
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
                "speaker_type": p.get("speaker_type", "streamer_reaction"),
                "layout": "stack",
                "words": len(rel_words),
                "words_list": rel_words
            }
            final_picks.append(clip_record)

            # Automatically enqueue into 1-hour schedule (Approved if Autopilot ON, Review Queue if OFF)
            try:
                scheduler.add_rendered_clip_to_schedule({
                    "id": cid,
                    "filename": meta.get("file"),
                    "file": meta.get("file"),
                    "filepath": str(out_dir / meta.get("file")),
                    "streamer": streamer,
                    "hook": hook,
                    "title": hook,
                    "caption": caption,
                    "hashtags": hashtags,
                    "score": score,
                    "duration": dur,
                    "speaker_type": p.get("speaker_type", "streamer_reaction")
                })
            except Exception as sched_err:
                print(f"[{streamer}] Scheduler enqueue error: {sched_err}")

    write_json(s_dir / "picks.json", {"curated_picks": final_picks})
    write_json(s_dir / "final.json", {"curated_picks": final_picks})
    write_json(s_dir / "progress.json", {
        "status": "Complete",
        "time": time.time(),
        "total_clips": len(final_picks)
    })

    # Clean intermediate segment buffer to free up disk space
    buf_dir = s_dir / "buffer"
    if buf_dir.exists():
        for ts_f in buf_dir.glob("*.ts"):
            try:
                ts_f.unlink(missing_ok=True)
            except Exception:
                pass
        print(f"[{streamer}] Cleaned raw segment files in {buf_dir.name} to preserve disk space.")

    print(f"[{streamer}] Successfully created and published {len(final_picks)} new reels!")
    try:
        ctx.bump()
    except Exception:
        pass

def auto_free_space(min_free_gb: float = 6.0):
    try:
        free_gb = shutil.disk_usage(str(WORK_DIR)).free / (1024 ** 3)
        if free_gb < min_free_gb:
            print(f"[Disk Safeguard] Free space is {free_gb:.1f} GB (< {min_free_gb} GB). Purging old .ts chunks...")
            cutoff = time.time() - 900  # 15 minutes ago
            for ts_path in WORK_DIR.glob("**/*.ts"):
                try:
                    if ts_path.stat().st_mtime < cutoff:
                        ts_path.unlink(missing_ok=True)
                except Exception:
                    pass
    except Exception as e:
        print(f"[Disk Safeguard Error] {e}")

def run_autonomous_clipping_cycle():
    """Single pass of checking live streamers and clipping active broadcasts."""
    auto_free_space(min_free_gb=6.0)
    print(f"[Daemon] Checking live streamers at {time.strftime('%Y-%m-%d %H:%M:%S')}...")
    active_found = []

    for stream_cfg in TARGET_STREAMERS:
        name = stream_cfg["streamer"]
        url = stream_cfg["url"]
        meta = get_stream_meta(url)

        if meta and meta.get("title"):
            hls = get_live_hls(url)
            if hls:
                active_found.append((stream_cfg, hls, meta))
                print(f"[Daemon] FOUND LIVE: {name} -> '{meta.get('title')}'")

    if not active_found:
        print("[Daemon] No target streams are currently live. Sleeping...")
        return

    for stream_cfg, hls_url, meta in active_found:
        name = stream_cfg["streamer"]
        try:
            # Ingest 240s of fresh live footage
            s_dir = record_stream_buffer(stream_cfg, hls_url, meta, record_seconds=240)
            # Curate with AI
            picks = curate_picks_with_ai(s_dir, stream_cfg, meta)
            if picks:
                # Render reels with GPU NVENC
                render_session_picks(s_dir, stream_cfg, picks)
            else:
                # Clean up empty session buffer
                buf_dir = s_dir / "buffer"
                if buf_dir.exists():
                    shutil.rmtree(buf_dir, ignore_errors=True)
                clips_f = list((s_dir / "clips").glob("*.mp4")) if (s_dir / "clips").exists() else []
                if not clips_f:
                    shutil.rmtree(s_dir, ignore_errors=True)
        except Exception as err:
            print(f"[{name}] Error processing session: {err}")

def daemon_main():
    print("[Autonomous Live Clipper Daemon Started]")
    while True:
        try:
            run_autonomous_clipping_cycle()
        except Exception as e:
            print(f"[Daemon Error] {e}")
        time.sleep(60)

if __name__ == "__main__":
    daemon_main()
