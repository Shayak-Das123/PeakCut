"""ClipFarm Web Server & Dashboard
Runs ClipFarm with a dedicated web interface, GPU hardware acceleration (RTX 4050 NVENC),
local speech transcription (sherpa-onnx / CUDA), OpenCV YuNet FaceCam detection,
kinetic subtitle motion graphics, and social posting integration.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import uvicorn
from fastapi import BackgroundTasks, Body, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

# Setup paths and environment
ROOT = Path(__file__).resolve().parent.parent
CLIPFARM_DIR = ROOT / "ClipFarm"
SCRIPTS_DIR = CLIPFARM_DIR / "scripts"
WORK_DIR = ROOT / "ClipFarm" / "work"
MODELS_DIR = ROOT / "ClipFarm" / "models"
DATA_MODELS_DIR = ROOT / "ClipFarm" / "data" / "models"
WEB_DIR = ROOT / "web"

sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(ROOT))

os.environ["CLIPFARM_HOME"] = str(CLIPFARM_DIR)
os.environ["CLIPFARM_WORK"] = str(WORK_DIR)
if not os.environ.get("PEAKCUT_FFMPEG_DIR"):
    os.environ["PEAKCUT_FFMPEG_DIR"] = r"D:\Shutter Encoder\Library"

WORK_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

from clipfarm import captions, face, fetch, listen, render, shortlist, signals, stream, util
from clipfarm.util import hms, read_json, save_settings, settings, stream_dir, write_json

from urllib.parse import quote
from fastapi.responses import RedirectResponse
from studio import accounts, config, ctx

# Configure server public URL and default port for OAuth callbacks
config.PORT = int(os.environ.get("CLIPFARM_PORT", 8767))
config.PUBLIC_URL = f"http://127.0.0.1:{config.PORT}"

AUTOPOST_SETTINGS_FILE = ROOT / "ClipFarm" / "data" / "autopost.json"
AUTOPOST_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)

DEFAULT_AUTOPOST_SETTINGS = {
    "enabled": False,
    "post_youtube": False,
    "post_instagram": False,
    "youtube_privacy": "public",
    "max_clips_per_stream": 5,
    "delay_seconds": 20,
    "history": []
}


def get_autopost_settings() -> dict:
    if not AUTOPOST_SETTINGS_FILE.exists():
        return dict(DEFAULT_AUTOPOST_SETTINGS)
    try:
        data = read_json(AUTOPOST_SETTINGS_FILE, DEFAULT_AUTOPOST_SETTINGS)
        return {**DEFAULT_AUTOPOST_SETTINGS, **data}
    except Exception:
        return dict(DEFAULT_AUTOPOST_SETTINGS)


def save_autopost_settings(data: dict) -> dict:
    cur = get_autopost_settings()
    cur.update(data)
    write_json(AUTOPOST_SETTINGS_FILE, cur)
    return cur


def record_post_history(platform: str, filename: str, url: str, post_id: str, title: str, error: str = ""):
    cur = get_autopost_settings()
    hist = cur.get("history", [])
    entry = {
        "timestamp": time.time(),
        "platform": platform,
        "filename": filename,
        "url": url,
        "post_id": post_id,
        "title": title,
        "error": error,
        "success": bool(url and not error)
    }
    hist.insert(0, entry)
    cur["history"] = hist[:100]
    write_json(AUTOPOST_SETTINGS_FILE, cur)


def get_clip_metadata_for_file(filename: str) -> dict:
    if not filename:
        return {}
    cid = filename.split("_")[0]
    for s_dir in (WORK_DIR / "streams").glob("*"):
        final_f = s_dir / "final.json"
        if final_f.exists():
            final_data = read_json(final_f, {})
            for c in final_data.get("clips", []):
                if match_clip_id(c.get("id"), cid) or c.get("file") == filename:
                    return c
    return {}


def trigger_autopost_for_stream(stream_key: str):
    """
    Autonomous Zero-Supervision Auto-Poster:
    Called as soon as a stream finishes rendering clips.
    Inspects autopost settings, queries connected platforms,
    and automatically uploads new clips without user intervention.
    """
    ctx.set_uid("u_local")
    st = get_autopost_settings()
    if not st.get("enabled"):
        return

    post_yt = bool(st.get("post_youtube", False))
    post_ig = bool(st.get("post_instagram", False))
    if not post_yt and not post_ig:
        return

    all_accs = accounts._all()
    yt_acc = next((a for a in all_accs if a.get("platform") == "youtube"), None)
    ig_acc = next((a for a in all_accs if a.get("platform") == "instagram"), None)

    if post_yt and not yt_acc:
        print(f"[AutoPost] YouTube auto-posting enabled, but no YouTube account connected.")
    if post_ig and not ig_acc:
        print(f"[AutoPost] Instagram auto-posting enabled, but no Instagram account connected.")

    if not yt_acc and not ig_acc:
        return

    d = stream_dir(stream_key)
    final_f = d / "final.json"
    if not final_f.exists():
        return

    final_data = read_json(final_f, {"clips": []})
    clips = final_data.get("clips", [])
    if not clips:
        return

    autopost_log_f = d / "autopost_log.json"
    autopost_log = read_json(autopost_log_f, {})

    max_clips = int(st.get("max_clips_per_stream", 5))
    delay_s = int(st.get("delay_seconds", 20))
    posted_count = 0

    print(f"\n[AutoPost] 🤖 Triggering autonomous zero-supervision auto-posting for stream '{stream_key}' ({len(clips)} total clips)...")

    for clip in clips[:max_clips]:
        cid = clip.get("id") or "c01"
        rendered_file = clip.get("file")
        if not rendered_file:
            continue

        p = find_file_in_work(rendered_file)
        if not p or not p.is_file():
            p = d / "clips" / rendered_file
            if not p.is_file():
                continue

        clip_posted = autopost_log.get(cid, {})
        title = clip.get("hook") or clip.get("title") or f"Viral Highlight {cid}"
        caption = clip.get("caption") or ""
        tags = clip.get("hashtags") or ["Shorts", "Gaming", "Viral"]
        if isinstance(tags, str):
            tags = [t.strip().lstrip("#") for t in tags.split() if t.strip()]
        else:
            tags = [str(t).strip().lstrip("#") for t in tags]
        if "Shorts" not in tags:
            tags.append("Shorts")

        # YouTube Shorts Auto-Post
        if post_yt and yt_acc and not clip_posted.get("youtube"):
            try:
                print(f"[AutoPost] 🚀 Uploading clip {cid} to YouTube Shorts: '{title}'...")
                desc = f"{caption}\n\n{' '.join('#' + t for t in tags)}\n\n#Shorts\n\nClipped autonomously with PeakCut ClipFarm"
                privacy = st.get("youtube_privacy", "public")
                vid_id, shorts_url = accounts.youtube_upload(yt_acc, p, title=title[:100], description=desc[:4900], tags=tags[:15], privacy=privacy)
                clip["youtube_url"] = shorts_url
                clip_posted["youtube"] = {"url": shorts_url, "id": vid_id, "time": time.time()}
                record_post_history("youtube", rendered_file, shorts_url, vid_id, title)
                print(f"[AutoPost] ✅ Published YouTube Short: {shorts_url}")
                posted_count += 1
            except Exception as e:
                print(f"[AutoPost] ❌ YouTube auto-post failed for clip {cid}: {e}")
                record_post_history("youtube", rendered_file, "", "", title, error=str(e))

        # Instagram Reels Auto-Post
        if post_ig and ig_acc and not clip_posted.get("instagram"):
            try:
                print(f"[AutoPost] 🚀 Uploading clip {cid} to Instagram Reels: '{title}'...")
                full_caption = f"{title}\n\n{caption}\n\n{' '.join('#' + t for t in tags)}"
                media_id, reel_url = accounts.instagram_upload(ig_acc, p, caption=full_caption[:2200])
                clip["instagram_url"] = reel_url
                clip_posted["instagram"] = {"url": reel_url, "id": media_id, "time": time.time()}
                record_post_history("instagram", rendered_file, reel_url, media_id, title)
                print(f"[AutoPost] ✅ Published Instagram Reel: {reel_url or 'Reel Published'}")
                posted_count += 1
            except Exception as e:
                print(f"[AutoPost] ❌ Instagram auto-post failed for clip {cid}: {e}")
                record_post_history("instagram", rendered_file, "", "", title, error=str(e))

        autopost_log[cid] = clip_posted
        write_json(autopost_log_f, autopost_log)
        write_json(final_f, final_data)

        if posted_count > 0:
            time.sleep(delay_s)

    print(f"[AutoPost] 🎯 Autonomous auto-posting finished for stream '{stream_key}'.\n")


app = FastAPI(title="ClipFarm Studio", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def ensure_user_context(request: Request, call_next):
    ctx.set_uid("u_local")
    return await call_next(request)

ACTIVE_JOBS = {}


def get_gpu_info() -> dict:
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2
        )
        if p.returncode == 0 and p.stdout.strip():
            parts = [x.strip() for x in p.stdout.strip().split(",")]
            return {
                "available": True,
                "name": parts[0],
                "memory_total_mb": parts[1] if len(parts) > 1 else "",
                "utilization": parts[2] if len(parts) > 2 else "0",
                "encoder": "NVENC (h264_nvenc)"
            }
    except Exception:
        pass
    return {"available": False, "name": "None", "encoder": "CPU (libx264)"}


def get_system_status():
    free_gb = round(shutil.disk_usage(str(ROOT)).free / (1024**3), 1)
    yunet_ok = (MODELS_DIR / "face_detection_yunet_2023mar.onnx").exists() or (DATA_MODELS_DIR / "face_detection_yunet_2023mar.onnx").exists()
    ffmpeg_ok = bool(util.which("ffmpeg"))
    ytdlp_ok = bool(util.which("yt-dlp"))
    gpu = get_gpu_info()
    return {
        "status": "online",
        "disk_free_gb": free_gb,
        "gpu": gpu,
        "facecam": "YuNet ONNX (GPU Accelerated)",
        "yunet_ready": yunet_ok,
        "ffmpeg_ready": ffmpeg_ok,
        "ytdlp_ready": ytdlp_ok,
        "active_jobs": len(ACTIVE_JOBS),
    }


@app.get("/healthz")
def healthz():
    return {"ok": True, "app": "ClipFarm"}


@app.get("/api/status")
def api_status():
    return get_system_status()


# --- HTTP Range Streaming for Video Playback ---
def range_stream(file_path: Path, range_header: Optional[str], media_type: str = "video/mp4"):
    file_size = file_path.stat().st_size
    if not range_header:
        return FileResponse(str(file_path), media_type=media_type, headers={"Accept-Ranges": "bytes"})
    
    range_match = re.match(r"bytes=(\d+)-(\d*)", range_header)
    if not range_match:
        return FileResponse(str(file_path), media_type=media_type)

    start = int(range_match.group(1))
    end_val = range_match.group(2)
    end = int(end_val) if end_val else file_size - 1
    end = min(end, file_size - 1)
    content_length = end - start + 1

    def iter_range():
        with open(file_path, "rb") as f:
            f.seek(start)
            remaining = content_length
            while remaining > 0:
                chunk_size = min(remaining, 1024 * 512)
                data = f.read(chunk_size)
                if not data:
                    break
                remaining -= len(data)
                yield data

    return StreamingResponse(
        iter_range(),
        status_code=206,
        media_type=media_type,
        headers={
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(content_length),
            "Content-Type": media_type,
        }
    )


@app.get("/api/download/{filename}")
def download_clip_file(filename: str):
    p = find_file_in_work(filename)
    if not p or not p.exists():
        raise HTTPException(status_code=404, detail="Clip not found")
    return FileResponse(
        path=str(p),
        media_type="video/mp4",
        filename=filename,
        content_disposition_type="attachment"
    )


def find_file_in_work(filename: str) -> Optional[Path]:
    if not filename:
        return None
    # 1. Look in streams clips
    for p in (WORK_DIR / "streams").glob(f"*/clips/{filename}"):
        if p.exists():
            return p
    # 2. Look in dated clips
    for p in (WORK_DIR / "clips").glob(f"*/{filename}"):
        if p.exists():
            return p
    # 3. Look in stream sections
    for p in (WORK_DIR / "streams").glob(f"*/sections/{filename}"):
        if p.exists():
            return p
    # 4. Look directly in stream root (e.g. audio16k.wav, live_stream.mp4)
    for p in (WORK_DIR / "streams").glob(f"*/{filename}"):
        if p.exists():
            return p
    # 5. Look in stream music folders
    for p in (WORK_DIR / "streams").glob(f"*/music/{filename}"):
        if p.exists():
            return p
    # 6. Look in ClipFarm/data/music (built-in presets)
    preset_p = CLIPFARM_DIR / "data" / "music" / filename
    if preset_p.exists():
        return preset_p
    # 7. Look in render subfolders (e.g. clip16k.wav)
    for p in (WORK_DIR / "streams").glob(f"*/render/*/{filename}"):
        if p.exists():
            return p
    return None


@app.get("/media/video/{filename}")
def stream_video(filename: str):
    p = find_file_in_work(filename)
    if not p or not p.exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    media_type = "video/mp4"
    if p.suffix in (".wav", ".mp3", ".aac", ".ogg"):
        media_type = f"audio/{p.suffix.lstrip('.')}"
    return FileResponse(
        str(p),
        media_type=media_type,
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "public, max-age=3600"
        }
    )


@app.get("/media/audio/{filename}")
def stream_audio(filename: str):
    p = find_file_in_work(filename)
    if not p or not p.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    media_type = "audio/wav" if filename.endswith(".wav") else "audio/mpeg" if filename.endswith(".mp3") else "audio/mp4"
    return FileResponse(
        str(p),
        media_type=media_type,
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "public, max-age=3600"
        }
    )


@app.get("/media/thumb/{filename}")
def serve_thumbnail(filename: str):
    p = find_file_in_work(filename)
    if not p or not p.exists():
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    media_type = "image/jpeg" if filename.endswith((".jpg", ".jpeg")) else "image/png"
    return FileResponse(str(p), media_type=media_type)


@app.get("/media/clips/{key}/{filename}")
def serve_clip_file_legacy(key: str, filename: str):
    p = stream_dir(key) / "clips" / filename
    if not p.exists():
        p = find_file_in_work(filename)
    if not p or not p.exists():
        raise HTTPException(status_code=404, detail="Clip not found")
    if filename.endswith(".mp4"):
        return FileResponse(
            str(p),
            media_type="video/mp4",
            headers={
                "Accept-Ranges": "bytes",
                "Cache-Control": "public, max-age=3600"
            }
        )
    media_type = "image/jpeg" if filename.endswith((".jpg", ".jpeg")) else "image/png"
    return FileResponse(str(p), media_type=media_type)


def ensure_clean_video(key: str, cid: str, force: bool = False) -> Optional[Path]:
    """Ensures clean 9:16 vertical footage (clean.mp4) exists with NO burned-in subtitles or hook."""
    d = stream_dir(key)
    render_d = d / "render" / cid
    clean_p = render_d / "clean.mp4"
    if not force and clean_p.exists() and clean_p.stat().st_size > 10000:
        return clean_p

    filter_file = render_d / "filter.txt"
    sec_file = d / "sections" / f"{cid}.mp4"
    if not sec_file.exists():
        for f in (d / "sections").glob("*.mp4"):
            if match_clip_id(f.stem, cid) and f.stat().st_size > 10000:
                sec_file = f
                break

    # If section is missing but info.json is available, download the needed section
    if (not sec_file.exists() or sec_file.stat().st_size < 10000) and (d / "info.json").exists():
        info = read_json(d / "info.json", {})
        final_file = d / "final.json"
        if final_file.exists():
            final_data = read_json(final_file, {"clips": []})
            c = next((x for x in final_data.get("clips", []) if match_clip_id(x.get("id"), cid)), None)
            if c and info.get("url"):
                try:
                    s_t = max(0.0, util.parse_t(c["start"]) - 4)
                    e_t = util.parse_t(c["end"]) + 4
                    sec_file.parent.mkdir(parents=True, exist_ok=True)
                    fetch.download_section(info["url"], s_t, e_t, sec_file)
                except Exception as ex:
                    print(f"Failed to download section for clean video {cid}:", ex)

    if filter_file.exists() and sec_file.exists() and sec_file.stat().st_size > 10000:
        try:
            filter_txt = filter_file.read_text(encoding="utf-8")
            lines = filter_txt.strip().split(";\n")
            clean_lines = [l for l in lines if not any(tag in l for tag in ['[cap]', '[hk]', '[cr]', 'overlay'])]
            clean_script = render_d / "filter_clean.txt"
            render_d.mkdir(parents=True, exist_ok=True)
            clean_script.write_text(";\n".join(clean_lines), encoding="utf-8")

            ffmpeg_bin = os.environ.get("PEAKCUT_FFMPEG_DIR", "")
            ffmpeg_exe = os.path.join(ffmpeg_bin, "ffmpeg.exe") if ffmpeg_bin else "ffmpeg"
            if not os.path.exists(ffmpeg_exe):
                ffmpeg_exe = shutil.which("ffmpeg") or "ffmpeg"

            cmd = [
                ffmpeg_exe, "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(sec_file),
                "-filter_complex_script", str(clean_script),
                "-map", "[base]", "-map", "[aout]",
                "-c:v", "h264_nvenc", "-preset", "p4", "-b:v", "10M",
                "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
                str(clean_p)
            ]
            res = subprocess.run(cmd, capture_output=True)
            if res.returncode != 0:
                v_idx = cmd.index("-c:v") if "-c:v" in cmd else -1
                if v_idx != -1:
                    cmd[v_idx + 1] = "libx264"
                    cmd[v_idx + 2] = "-preset"
                    cmd[v_idx + 3] = "fast"
                    subprocess.run(cmd, check=False)
            if clean_p.exists() and clean_p.stat().st_size > 10000:
                return clean_p
            elif clean_p.exists():
                clean_p.unlink(missing_ok=True)
        except Exception as ex:
            print(f"Error ensuring clean video for {key}/{cid}:", ex)

    # Fallback to rendered clip if clean.mp4 could not be built
    for f in (d / "clips").glob("*.mp4"):
        if match_clip_id(f.stem.split("_")[0], cid) or cid in f.stem:
            return f
    return None


@app.get("/media/clean/{key}/{cid}.mp4")
@app.head("/media/clean/{key}/{cid}.mp4")
def serve_clean_video(key: str, cid: str):
    d = stream_dir(key)
    clean_p = d / "render" / cid / "clean.mp4"
    if clean_p.exists() and clean_p.stat().st_size > 10000:
        return FileResponse(
            str(clean_p),
            media_type="video/mp4",
            headers={
                "Accept-Ranges": "bytes",
                "Cache-Control": "public, max-age=3600"
            }
        )
    # Instant fallback to rendered clip from d/clips or WORK_DIR/clips
    for f in (d / "clips").glob("*.mp4"):
        if match_clip_id(f.stem.split("_")[0], cid) or cid in f.stem:
            if f.stat().st_size > 10000:
                return FileResponse(
                    str(f),
                    media_type="video/mp4",
                    headers={
                        "Accept-Ranges": "bytes",
                        "Cache-Control": "public, max-age=3600"
                    }
                )
    for c_dir in (WORK_DIR / "clips").glob("*"):
        if c_dir.is_dir():
            for f in c_dir.glob("*.mp4"):
                if match_clip_id(f.stem.split("_")[0], cid) or cid in f.stem:
                    if f.stat().st_size > 10000:
                        return FileResponse(
                            str(f),
                            media_type="video/mp4",
                            headers={
                                "Accept-Ranges": "bytes",
                                "Cache-Control": "public, max-age=3600"
                            }
                        )
    p = ensure_clean_video(key, cid)
    if not p or not p.exists() or p.stat().st_size < 10000:
        raise HTTPException(status_code=404, detail="Clean video not found")
    return FileResponse(
        str(p),
        media_type="video/mp4",
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "public, max-age=3600"
        }
    )


def ensure_peekcut_video(key: str, cid: str, force: bool = False) -> Optional[Path]:
    """Ensures 9:16 vertical PeekCut footage (peekcut.mp4) with 16:9 uncropped stream centered and blurred background."""
    d = stream_dir(key)
    render_d = d / "render" / cid
    peek_p = render_d / "peekcut.mp4"
    if not force and peek_p.exists() and peek_p.stat().st_size > 10000:
        return peek_p

    filter_file = render_d / "filter.txt"
    sec_file = d / "sections" / f"{cid}.mp4"
    if not sec_file.exists():
        for f in (d / "sections").glob("*.mp4"):
            if match_clip_id(f.stem, cid) and f.stat().st_size > 10000:
                sec_file = f
                break

    if (not sec_file.exists() or sec_file.stat().st_size < 10000) and (d / "info.json").exists():
        info = read_json(d / "info.json", {})
        final_file = d / "final.json"
        if final_file.exists():
            final_data = read_json(final_file, {"clips": []})
            c = next((x for x in final_data.get("clips", []) if match_clip_id(x.get("id"), cid)), None)
            if c and info.get("url"):
                try:
                    s_t = max(0.0, util.parse_t(c["start"]) - 4)
                    e_t = util.parse_t(c["end"]) + 4
                    sec_file.parent.mkdir(parents=True, exist_ok=True)
                    fetch.download_section(info["url"], s_t, e_t, sec_file)
                except Exception as ex:
                    print(f"Failed to download section for peekcut video {cid}:", ex)

    if sec_file.exists() and sec_file.stat().st_size > 10000:
        try:
            render_d.mkdir(parents=True, exist_ok=True)
            script_p = render_d / "filter_peekcut.txt"

            if filter_file.exists():
                filter_txt = filter_file.read_text(encoding="utf-8")
                lines = [l.strip() for l in filter_txt.split(";") if l.strip()]
                trims = []
                for l in lines:
                    m = re.search(r"\[vs(\d+)\]trim=start=([\d.]+):end=([\d.]+)", l)
                    if m:
                        trims.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))

                if trims:
                    K = len(trims)
                    fc = [
                        f"[0:v]split={K}" + "".join(f"[vs{i}]" for i in range(K)) if K > 1 else "[0:v]null[vs0]",
                        f"[0:a]asplit={K}" + "".join(f"[as{i}]" for i in range(K)) if K > 1 else "[0:a]anull[as0]"
                    ]
                    for idx, s, e in trims:
                        fc.append(f"[vs{idx}]trim=start={s:.3f}:end={e:.3f},setpts=PTS-STARTPTS,split=2[pa{idx}][pb{idx}]")
                        fc.append(f"[pa{idx}]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=20:5,eq=brightness=-0.12:saturation=0.85[pg{idx}]")
                        fc.append(f"[pb{idx}]scale=1080:-2:flags=bicubic,setsar=1[pf{idx}]")
                        fc.append(f"[pg{idx}][pf{idx}]overlay=0:(H-h)/2,format=yuv420p,setsar=1[v{idx}]")

                    for l in lines:
                        if re.search(r"\[as\d+\]atrim=", l):
                            fc.append(l)

                    fc.append("".join(f"[v{i}][a{i}]" for i in range(K)) + f"concat=n={K}:v=1:a=1[base][ac]")

                    for l in lines:
                        if "[aout]" in l:
                            fc.append(l)

                    script_p.write_text(";\n".join(fc), encoding="utf-8")
                else:
                    fc = [
                        "[0:v]split=2[pa][pb]",
                        "[pa]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=20:5,eq=brightness=-0.12:saturation=0.85[pg]",
                        "[pb]scale=1080:-2:flags=bicubic,setsar=1[pf]",
                        "[pg][pf]overlay=0:(H-h)/2,format=yuv420p,setsar=1[base]"
                    ]
                    script_p.write_text(";\n".join(fc), encoding="utf-8")
            else:
                fc = [
                    "[0:v]split=2[pa][pb]",
                    "[pa]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=20:5,eq=brightness=-0.12:saturation=0.85[pg]",
                    "[pb]scale=1080:-2:flags=bicubic,setsar=1[pf]",
                    "[pg][pf]overlay=0:(H-h)/2,format=yuv420p,setsar=1[base]"
                ]
                script_p.write_text(";\n".join(fc), encoding="utf-8")

            ffmpeg_bin = os.environ.get("PEAKCUT_FFMPEG_DIR", "")
            ffmpeg_exe = os.path.join(ffmpeg_bin, "ffmpeg.exe") if ffmpeg_bin else "ffmpeg"
            if not os.path.exists(ffmpeg_exe):
                ffmpeg_exe = shutil.which("ffmpeg") or "ffmpeg"

            cmd = [
                ffmpeg_exe, "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(sec_file),
                "-filter_complex_script", str(script_p),
                "-map", "[base]",
                "-c:v", "h264_nvenc", "-preset", "p4", "-b:v", "8M",
                "-movflags", "+faststart"
            ]
            if "[aout]" in script_p.read_text(encoding="utf-8"):
                cmd.extend(["-map", "[aout]", "-c:a", "aac", "-b:a", "192k"])
            else:
                cmd.extend(["-map", "0:a?", "-c:a", "aac", "-b:a", "192k"])
            cmd.append(str(peek_p))

            res = subprocess.run(cmd, capture_output=True)
            if res.returncode != 0:
                cmd_cpu = [c if c != "h264_nvenc" else "libx264" for c in cmd]
                subprocess.run(cmd_cpu, check=False)

            if peek_p.exists() and peek_p.stat().st_size > 10000:
                return peek_p
        except Exception as ex:
            print(f"Error ensuring peekcut video for {key}/{cid}:", ex)

    return None


@app.get("/media/peekcut/{key}/{cid}.mp4")
@app.head("/media/peekcut/{key}/{cid}.mp4")
def serve_peekcut_video(key: str, cid: str):
    d = stream_dir(key)
    peek_p = d / "render" / cid / "peekcut.mp4"
    if peek_p.exists() and peek_p.stat().st_size > 10000:
        return FileResponse(
            str(peek_p),
            media_type="video/mp4",
            headers={
                "Accept-Ranges": "bytes",
                "Cache-Control": "public, max-age=3600"
            }
        )
    # Instant fallback to rendered clip from d/clips or WORK_DIR/clips
    for f in (d / "clips").glob("*.mp4"):
        if match_clip_id(f.stem.split("_")[0], cid) or cid in f.stem:
            if f.stat().st_size > 10000:
                return FileResponse(
                    str(f),
                    media_type="video/mp4",
                    headers={
                        "Accept-Ranges": "bytes",
                        "Cache-Control": "public, max-age=3600"
                    }
                )
    for c_dir in (WORK_DIR / "clips").glob("*"):
        if c_dir.is_dir():
            for f in c_dir.glob("*.mp4"):
                if match_clip_id(f.stem.split("_")[0], cid) or cid in f.stem:
                    if f.stat().st_size > 10000:
                        return FileResponse(
                            str(f),
                            media_type="video/mp4",
                            headers={
                                "Accept-Ranges": "bytes",
                                "Cache-Control": "public, max-age=3600"
                            }
                        )
    p = ensure_peekcut_video(key, cid)
    if not p or not p.exists() or p.stat().st_size < 10000:
        raise HTTPException(status_code=404, detail="PeekCut video not found")
    return FileResponse(
        str(p),
        media_type="video/mp4",
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "public, max-age=3600"
        }
    )


def _enrich_job_info(d: Path, job_info: Optional[dict], info: dict, st: dict, final: dict) -> Optional[dict]:
    if not job_info:
        return None
    res = dict(job_info)
    dur = float(info.get("duration") or st.get("duration") or 0.0)
    
    # Calculate audio downloaded MB & HMS
    audio_mb = 0.0
    for af in [d / "audio_src.mp4", d / "audio_src.webm", d / "audio16k.wav"]:
        if af.exists():
            audio_mb = max(audio_mb, round(af.stat().st_size / (1024 * 1024), 1))
    
    if "audio_downloaded_mb" not in res:
        res["audio_downloaded_mb"] = audio_mb
    if "audio_downloaded_hms" not in res:
        res["audio_downloaded_hms"] = hms(dur) if dur > 0 else f"{audio_mb} MB"
    if "audio_downloaded_sec" not in res:
        res["audio_downloaded_sec"] = dur

    # Calculate audio transcribed HMS & PCT
    is_fully_scanned = bool(st.get("chunks") or st.get("words") or (d / "segments.json").exists())
    pct = 100 if is_fully_scanned else res.get("pct", 0)
    if "audio_transcribed_pct" not in res:
        res["audio_transcribed_pct"] = pct
    if "audio_transcribed_sec" not in res:
        res["audio_transcribed_sec"] = dur if is_fully_scanned else round(dur * (pct / 100.0))
    if "audio_transcribed_hms" not in res:
        res["audio_transcribed_hms"] = hms(dur) if (is_fully_scanned and dur > 0) else hms(round(dur * (pct / 100.0))) if dur > 0 else f"{pct}%"

    # Clips found count
    clips_list = final.get("clips") or []
    if "clips_found_count" not in res:
        res["clips_found_count"] = len(clips_list) or len(read_json(d / "shortlist.json", []))

    return res


# --- Streams & Clips APIs ---
@app.get("/api/streams")
def list_streams():
    streams_root = WORK_DIR / "streams"
    if not streams_root.exists():
        return []
    items = []
    for d in sorted(streams_root.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
        st = read_json(d / "state.json", {})
        info = read_json(d / "info.json", {})
        final = read_json(d / "final.json", {})
        clips_count = len(list((d / "clips").glob("*.mp4"))) if (d / "clips").exists() else 0
        live_s = ACTIVE_LIVE_SESSIONS.get(d.name)
        raw_job = (live_s.get_status() if hasattr(live_s, "get_status") else None) or ACTIVE_JOBS.get(d.name)
        if not raw_job and (d / "progress.json").exists():
            pj = read_json(d / "progress.json", None)
            if pj and not pj.get("done") and pj.get("action") not in ("stopped", "idle", "complete"):
                raw_job = pj
        job_info = _enrich_job_info(d, raw_job, info, st, final) if raw_job else None
        stage_val = (
            live_s.status if live_s
            else "rendered" if clips_count > 0
            else "ready_to_render" if final.get("clips")
            else "shortlisted" if (d / "shortlist.json").exists()
            else "scanned" if st.get("chunks")
            else "processing" if job_info and not job_info.get("done")
            else "scanned" if (d / "segments.json").exists()
            else "pending"
        )
        items.append({
            "key": d.name,
            "title": info.get("title", d.name),
            "streamer": info.get("streamer", st.get("streamer", "Unknown")),
            "platform": info.get("platform", "web"),
            "duration": info.get("duration", 0),
            "duration_hms": hms(info.get("duration", 0)),
            "url": info.get("url", st.get("url", "")),
            "chunks": st.get("chunks", 0),
            "stage": stage_val,
            "clips_count": clips_count,
            "job": job_info
        })
    return items


@app.get("/api/streams/{key}")
def get_stream(key: str):
    d = stream_dir(key)
    if not d.exists():
        raise HTTPException(status_code=404, detail="Stream not found")
    st = read_json(d / "state.json", {})
    info = read_json(d / "info.json", {})
    shortlist_data = read_json(d / "shortlist.json", [])
    final_data = read_json(d / "final.json", {"clips": []})
    
    clips = []
    clips_dir = d / "clips"
    if clips_dir.exists():
        for f in sorted(clips_dir.glob("*.mp4")):
            facecam = f.with_suffix(".facecam.jpg")
            thumb = f.with_suffix(".jpg")
            meta = read_json(f.with_suffix(".json"), {})
            txt = ""
            txt_file = f.with_suffix(".txt")
            if txt_file.exists():
                txt = txt_file.read_text(encoding="utf-8", errors="replace")
            prefix = f.stem.split("_")[0]
            cid_val = meta.get("id") or (prefix if (prefix.isdigit() or prefix.startswith("c")) else f.stem)
            f_clip = next((c for c in final_data.get("clips", []) if match_clip_id(c.get("id"), cid_val)), None)
            if f_clip:
                if f_clip.get("hook"):
                    meta["hook"] = f_clip["hook"]
                if f_clip.get("caption"):
                    meta["caption"] = f_clip["caption"]
                if f_clip.get("title"):
                    meta["title"] = f_clip["title"]
                if f_clip.get("hashtags"):
                    meta["hashtags"] = f_clip["hashtags"]
            clips.append({
                "filename": f.name,
                "cid": cid_val,
                "video_url": f"/media/video/{f.name}",
                "download_url": f"/api/download/{f.name}",
                "thumb_url": f"/media/thumb/{thumb.name}" if thumb.exists() else None,
                "facecam_url": f"/media/thumb/{facecam.name}" if facecam.exists() else None,
                "meta": meta,
                "post_text": txt,
                "youtube_url": f_clip.get("youtube_url") if f_clip else meta.get("youtube_url"),
                "instagram_url": f_clip.get("instagram_url") if f_clip else meta.get("instagram_url"),
                "saved": meta.get("saved", False),
                "size_mb": round(f.stat().st_size / (1024 * 1024), 2)
            })

    live_s = ACTIVE_LIVE_SESSIONS.get(key)
    raw_job = (live_s.get_status() if hasattr(live_s, "get_status") else None) or ACTIVE_JOBS.get(key)
    if not raw_job and (d / "progress.json").exists():
        pj = read_json(d / "progress.json", None)
        if pj and not pj.get("done") and pj.get("action") not in ("stopped", "idle", "complete"):
            raw_job = pj
    job_info = _enrich_job_info(d, raw_job, info, st, final_data) if raw_job else None

    return {
        "key": key,
        "info": info,
        "state": st,
        "shortlist": shortlist_data,
        "final": final_data,
        "clips": clips,
        "job": job_info
    }


@app.get("/api/clips")
def list_all_clips():
    clips = []
    seen = set()

    def add_clip(f: Path, stream_key: str):
        if f.name in seen:
            return
        seen.add(f.name)
        meta = read_json(f.with_suffix(".json"), {})
        txt = ""
        txt_file = f.with_suffix(".txt")
        if txt_file.exists():
            txt = txt_file.read_text(encoding="utf-8", errors="replace")
        thumb = f.with_suffix(".jpg")
        facecam = f.with_suffix(".facecam.jpg")
        cid_val = meta.get("id")
        if not cid_val:
            prefix = f.stem.split("_")[0]
            cid_val = prefix if (prefix.isdigit() or prefix.startswith("c")) else f.stem
        
        stream_name = stream_key if stream_key != "vod" else (meta.get("stream") or "")
        effective_stream_key = stream_name or stream_key
        clip_uid = f"{effective_stream_key}:{cid_val}"
        if clip_uid in seen:
            # Obsolete older version of this clip! Automatically purge from disk to free storage
            if stream_key == "vod":
                try:
                    for old_f in f.parent.glob(f"{f.stem}.*"):
                        old_f.unlink(missing_ok=True)
                except Exception:
                    pass
            return
        seen.add(clip_uid)
        if stream_name:
            final_p = WORK_DIR / "streams" / stream_name / "final.json"
            stream_final = read_json(final_p, {}) if final_p.exists() else {}
            f_clip = next((c for c in stream_final.get("clips", []) if match_clip_id(c.get("id"), cid_val)), None)
            if not f_clip:
                f_clip = next((c for c in stream_final.get("clips", []) if c.get("file") == f.name), None)
            if f_clip:
                cid_val = f_clip.get("id") or cid_val
                if f_clip.get("hook"):
                    meta["hook"] = f_clip["hook"]
                if f_clip.get("caption") and "Would you have reacted" not in f_clip["caption"]:
                    meta["caption"] = f_clip["caption"]
                elif not meta.get("caption") or "Would you have reacted" in meta.get("caption", ""):
                    # Dynamically generate Gemini viral caption if missing or generic
                    dlg = ""
                    words_p = WORK_DIR / "streams" / stream_name / "render" / cid_val / "words.json"
                    if words_p.exists():
                        dlg = " ".join(w.get("w", "") for w in read_json(words_p, [])).strip()
                    if not dlg and f_clip.get("custom_words"):
                        dlg = " ".join(w.get("w", "") for w in f_clip["custom_words"]).strip()
                    pkg = generate_ai_clip_hook_and_caption(
                        streamer=meta.get("streamer") or "CaseOh",
                        stream_title=meta.get("title") or "Live Stream",
                        dialogue=dlg,
                        why_signals=f_clip.get("why") or "",
                        cand_title=f_clip.get("title") or f_clip.get("hook") or ""
                    )
                    f_clip["caption"] = pkg["caption"]
                    if not f_clip.get("hook") or f_clip["hook"].isupper() or len(f_clip["hook"].split()) <= 2:
                        f_clip["hook"] = pkg["hook"]
                    f_clip["hashtags"] = pkg["hashtags"]
                    write_json(final_p, stream_final)
                    meta["hook"] = f_clip["hook"]
                    meta["caption"] = f_clip["caption"]
                    meta["hashtags"] = f_clip["hashtags"]
                if f_clip.get("title"):
                    meta["title"] = f_clip["title"]
                if f_clip.get("hashtags"):
                    meta["hashtags"] = f_clip["hashtags"]
        
        # Expose clean (no burned text) video URL so the library preview matches
        # exactly what loads in the editor — same source, no confusion.
        # ensure_clean_video stores files at render/{cid}/clean.mp4
        clean_p = None
        if stream_name:
            _cp = WORK_DIR / "streams" / stream_name / "render" / cid_val / "clean.mp4"
            if _cp.exists() and _cp.stat().st_size > 10000:
                clean_p = _cp
        clean_video_url = f"/media/clean/{stream_name}/{cid_val}.mp4" if clean_p else f"/media/video/{f.name}"

        clips.append({
            "filename": f.name,
            "stream_key": effective_stream_key,
            "cid": cid_val,
            "streamer": meta.get("streamer") or "Creator",
            "title": meta.get("title") or meta.get("hook") or f.stem,
            "hook": meta.get("hook") or meta.get("title") or "",
            "caption": meta.get("caption") or "",
            "hashtags": meta.get("hashtags") or [],
            "post_text": txt,
            "video_url": f"/media/video/{f.name}",
            "clean_video_url": clean_video_url,
            "download_url": f"/api/download/{f.name}",
            "thumb_url": f"/media/thumb/{thumb.name}" if thumb.exists() else None,
            "facecam_url": f"/media/thumb/{facecam.name}" if facecam.exists() else None,
            "youtube_url": f_clip.get("youtube_url") if f_clip else meta.get("youtube_url"),
            "instagram_url": f_clip.get("instagram_url") if f_clip else meta.get("instagram_url"),
            "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
            "saved": meta.get("saved", False),
            "created_at": f.stat().st_mtime
        })

    # Search streams
    for d in (WORK_DIR / "streams").glob("*"):
        for f in (d / "clips").glob("*.mp4"):
            add_clip(f, d.name)

    # Search dated clips
    for d in (WORK_DIR / "clips").glob("*"):
        if d.is_dir():
            for f in d.glob("*.mp4"):
                add_clip(f, "vod")

    clips.sort(key=lambda x: x["created_at"], reverse=True)
    return clips


@app.post("/api/clips/regenerate-all-ai")
def regenerate_all_ai_clips():
    """Batch regenerates catchy AI hooks and viral captions for all clips using Gemini."""
    results = []

    # 1. Process all stream final.json files
    for d in (WORK_DIR / "streams").glob("*"):
        final_file = d / "final.json"
        if not final_file.exists():
            continue
        final_data = read_json(final_file, {"clips": []})
        clips = final_data.get("clips", [])
        info = read_json(d / "info.json", {})
        streamer = info.get("streamer") or "Streamer"
        stream_title = info.get("title") or "Live Stream"

        modified = False
        for c in clips:
            cid = c.get("id")
            dialogue = ""
            render_words_file = d / "render" / cid / "words.json"
            if render_words_file.exists():
                w_list = read_json(render_words_file, [])
                dialogue = " ".join(w.get("w", "") for w in w_list).strip()
            if not dialogue and c.get("custom_words"):
                dialogue = " ".join(w.get("w", "") for w in c["custom_words"]).strip()

            pkg = generate_ai_clip_hook_and_caption(
                streamer=streamer,
                stream_title=stream_title,
                dialogue=dialogue,
                why_signals=c.get("why") or "",
                cand_title=c.get("title") or c.get("hook") or ""
            )
            c["hook"] = pkg["hook"]
            c["caption"] = pkg["caption"]
            c["hashtags"] = pkg["hashtags"]
            modified = True

            # Sync sidecar .json files in stream clips and saved_clips
            for c_dir in [d / "clips", WORK_DIR / "saved_clips"]:
                if c_dir.exists():
                    for f in c_dir.glob("*.mp4"):
                        prefix = f.stem.split("_")[0]
                        if match_clip_id(prefix, cid) or cid in f.stem:
                            side_meta = read_json(f.with_suffix(".json"), {})
                            side_meta["hook"] = pkg["hook"]
                            side_meta["caption"] = pkg["caption"]
                            side_meta["hashtags"] = pkg["hashtags"]
                            write_json(f.with_suffix(".json"), side_meta)

            results.append({"stream": d.name, "cid": cid, "hook": pkg["hook"], "caption": pkg["caption"]})

        if modified:
            write_json(final_file, final_data)

    # 2. Process all dated clips in WORK_DIR / "clips"
    for folder in (WORK_DIR / "clips").glob("*"):
        if folder.is_dir():
            for f in folder.glob("*.mp4"):
                meta_file = f.with_suffix(".json")
                meta = read_json(meta_file, {})
                stream_name = meta.get("stream", "")
                stream_dir_path = WORK_DIR / "streams" / stream_name if stream_name else None
                stream_final = read_json(stream_dir_path / "final.json", {}) if stream_dir_path and (stream_dir_path / "final.json").exists() else {}
                cid_val = meta.get("id") or f.stem.split("_")[0]
                matched_clip = next((c for c in stream_final.get("clips", []) if match_clip_id(c.get("id"), cid_val)), None)
                if matched_clip:
                    meta["hook"] = matched_clip.get("hook") or meta.get("hook")
                    meta["caption"] = matched_clip.get("caption") or meta.get("caption")
                    meta["hashtags"] = matched_clip.get("hashtags") or meta.get("hashtags")
                    write_json(meta_file, meta)

    return {"ok": True, "count": len(results), "regenerated": results}


# --- Clip Storage, Retention & Audio Cleanup APIs ---
@app.post("/api/clips/{filename}/save")
def toggle_save_clip(filename: str):
    p = find_file_in_work(filename)
    if not p or not p.exists():
        raise HTTPException(status_code=404, detail="Clip not found")
    meta_path = p.with_suffix(".json")
    meta = read_json(meta_path, {})
    new_state = not meta.get("saved", False)
    meta["saved"] = new_state
    write_json(meta_path, meta)

    saved_dir = WORK_DIR / "saved_clips"
    saved_dir.mkdir(parents=True, exist_ok=True)
    if new_state:
        for ext in [".mp4", ".jpg", ".json", ".txt", ".words.json", ".facecam.jpg"]:
            src_f = p.with_suffix(ext)
            if src_f.exists():
                shutil.copy2(src_f, saved_dir / src_f.name)
    else:
        for ext in [".mp4", ".jpg", ".json", ".txt", ".words.json", ".facecam.jpg"]:
            (saved_dir / (p.stem + ext)).unlink(missing_ok=True)

    return {"ok": True, "saved": new_state, "filename": filename}


@app.delete("/api/clips/{filename}")
def delete_clip(filename: str):
    p = find_file_in_work(filename)
    if not p or not p.exists():
        raise HTTPException(status_code=404, detail="Clip not found")

    stem = p.stem
    parent = p.parent
    for item in parent.glob(f"{stem}.*"):
        item.unlink(missing_ok=True)

    for d in (WORK_DIR / "clips").glob("*"):
        for item in d.glob(f"{stem}.*"):
            item.unlink(missing_ok=True)

    return {"ok": True, "deleted": filename}


@app.post("/api/clips/purge-unsaved")
def purge_unsaved_clips():
    deleted = 0
    for d in (WORK_DIR / "streams").glob("*"):
        for f in (d / "clips").glob("*.mp4"):
            meta = read_json(f.with_suffix(".json"), {})
            if not meta.get("saved", False):
                stem = f.stem
                for item in f.parent.glob(f"{stem}.*"):
                    item.unlink(missing_ok=True)
                deleted += 1

    for d in (WORK_DIR / "clips").glob("*"):
        if d.is_dir() and d.name != "saved_clips":
            for f in d.glob("*.mp4"):
                meta = read_json(f.with_suffix(".json"), {})
                if not meta.get("saved", False):
                    stem = f.stem
                    for item in f.parent.glob(f"{stem}.*"):
                        item.unlink(missing_ok=True)
                    deleted += 1

    return {"ok": True, "deleted_count": deleted}


@app.post("/api/streams/{key}/cleanup-audio")
def cleanup_stream_audio(key: str):
    d = stream_dir(key)
    if not d.exists():
        raise HTTPException(status_code=404, detail="Stream not found")
    freed = 0
    for fname in ["audio16k.wav", "audio_src.webm", "work.wav"]:
        af = d / fname
        if af.exists():
            freed += af.stat().st_size
            af.unlink(missing_ok=True)

    live_vid = d / "live_stream.mp4"
    if live_vid.exists() and (d / "clips").exists() and len(list((d / "clips").glob("*.mp4"))) > 0:
        freed += live_vid.stat().st_size
        live_vid.unlink(missing_ok=True)

    sec_dir = d / "sections"
    if sec_dir.exists():
        for sf in sec_dir.glob("*.mp4"):
            freed += sf.stat().st_size
            sf.unlink(missing_ok=True)

    freed_mb = round(freed / (1024 * 1024), 1)
    return {"ok": True, "freed_mb": freed_mb}


@app.get("/api/storage/overview")
def get_storage_overview():
    try:
        total_b, used_b, free_b = shutil.disk_usage(WORK_DIR)
        disk_total_gb = round(total_b / (1024**3), 1)
        disk_free_gb = round(free_b / (1024**3), 1)
        disk_used_gb = round(used_b / (1024**3), 1)
        free_pct = round(free_b / max(1, total_b) * 100, 1)
    except Exception:
        disk_total_gb, disk_free_gb, disk_used_gb, free_pct = 0, 0, 0, 0

    streams_root = WORK_DIR / "streams"
    streams_data = []
    all_audio_files = []
    all_clip_files = []
    all_section_files = []

    total_audio_bytes = 0
    total_clips_bytes = 0
    total_sections_bytes = 0

    if streams_root.exists():
        for sdir in sorted(streams_root.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
            if not sdir.is_dir():
                continue
            s_key = sdir.name
            info = read_json(sdir / "info.json", {})
            st = read_json(sdir / "state.json", {})
            streamer_name = info.get("streamer") or st.get("streamer") or s_key

            s_audio_bytes = 0
            s_clips_bytes = 0
            s_sections_bytes = 0
            s_audio_count = 0
            s_clips_count = 0
            s_sections_count = 0

            # 1. Audio files in stream folder & render/
            audio_pats = ["audio16k.wav", "audio_src.*", "work.wav", "render/*/clip16k.wav"]
            for pat in audio_pats:
                for af in sdir.glob(pat):
                    if af.is_file() and af.exists():
                        sz = af.stat().st_size
                        s_audio_bytes += sz
                        total_audio_bytes += sz
                        s_audio_count += 1
                        all_audio_files.append({
                            "name": af.name,
                            "rel_path": str(af.relative_to(WORK_DIR)).replace("\\", "/"),
                            "stream_key": s_key,
                            "streamer": streamer_name,
                            "size_mb": round(sz / (1024 * 1024), 2),
                            "size_bytes": sz,
                            "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(af.stat().st_mtime)),
                            "url": f"/media/audio/{af.name}"
                        })

            # 2. Clips in stream/clips
            clips_dir = sdir / "clips"
            if clips_dir.exists():
                for cf in sorted(clips_dir.glob("*.mp4")):
                    sz = cf.stat().st_size
                    s_clips_bytes += sz
                    total_clips_bytes += sz
                    s_clips_count += 1
                    cmeta = read_json(cf.with_suffix(".json"), {})
                    prefix = cf.stem.split("_")[0]
                    cid_val = cmeta.get("id") or (prefix if (prefix.isdigit() or prefix.startswith("c")) else cf.stem)
                    all_clip_files.append({
                        "name": cf.name,
                        "cid": cid_val,
                        "rel_path": str(cf.relative_to(WORK_DIR)).replace("\\", "/"),
                        "stream_key": s_key,
                        "streamer": streamer_name,
                        "hook": cmeta.get("hook") or cmeta.get("title") or cf.stem,
                        "size_mb": round(sz / (1024 * 1024), 2),
                        "size_bytes": sz,
                        "saved": cmeta.get("saved", False),
                        "duration": cmeta.get("length") or 30.0,
                        "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(cf.stat().st_mtime)),
                        "video_url": f"/media/video/{cf.name}"
                    })

            # 3. Sections & buffer files
            sec_dir = sdir / "sections"
            if sec_dir.exists():
                for sf in sorted(sec_dir.glob("*")):
                    if sf.is_file() and sf.suffix in (".mp4", ".ts", ".mkv", ".webm"):
                        sz = sf.stat().st_size
                        s_sections_bytes += sz
                        total_sections_bytes += sz
                        s_sections_count += 1
                        all_section_files.append({
                            "name": sf.name,
                            "rel_path": str(sf.relative_to(WORK_DIR)).replace("\\", "/"),
                            "stream_key": s_key,
                            "streamer": streamer_name,
                            "size_mb": round(sz / (1024 * 1024), 2),
                            "size_bytes": sz,
                            "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(sf.stat().st_mtime)),
                            "video_url": f"/media/video/{sf.name}" if sf.suffix == ".mp4" else None
                        })

            # Live stream file if present
            live_f = sdir / "live_stream.mp4"
            if live_f.exists():
                sz = live_f.stat().st_size
                s_sections_bytes += sz
                total_sections_bytes += sz
                s_sections_count += 1
                all_section_files.append({
                    "name": live_f.name,
                    "rel_path": str(live_f.relative_to(WORK_DIR)).replace("\\", "/"),
                    "stream_key": s_key,
                    "streamer": streamer_name,
                    "size_mb": round(sz / (1024 * 1024), 2),
                    "size_bytes": sz,
                    "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(live_f.stat().st_mtime)),
                    "video_url": f"/media/video/{live_f.name}"
                })

            streams_data.append({
                "key": s_key,
                "title": info.get("title") or s_key,
                "streamer": streamer_name,
                "platform": info.get("platform", "web"),
                "audio_mb": round(s_audio_bytes / (1024 * 1024), 1),
                "audio_count": s_audio_count,
                "clips_mb": round(s_clips_bytes / (1024 * 1024), 1),
                "clips_count": s_clips_count,
                "sections_mb": round(s_sections_bytes / (1024 * 1024), 1),
                "sections_count": s_sections_count,
                "total_mb": round((s_audio_bytes + s_clips_bytes + s_sections_bytes) / (1024 * 1024), 1),
            })

    # Also check dated clips folder
    dated_clips_dir = WORK_DIR / "clips"
    if dated_clips_dir.exists():
        for cf in dated_clips_dir.glob("*/*.mp4"):
            if cf.name not in [c["name"] for c in all_clip_files]:
                sz = cf.stat().st_size
                total_clips_bytes += sz
                cmeta = read_json(cf.with_suffix(".json"), {})
                all_clip_files.append({
                    "name": cf.name,
                    "cid": cf.stem.split("_")[0],
                    "rel_path": str(cf.relative_to(WORK_DIR)).replace("\\", "/"),
                    "stream_key": "dated",
                    "streamer": cmeta.get("streamer") or "Creator",
                    "hook": cmeta.get("hook") or cmeta.get("title") or cf.stem,
                    "size_mb": round(sz / (1024 * 1024), 2),
                    "size_bytes": sz,
                    "saved": cmeta.get("saved", False),
                    "duration": cmeta.get("length") or 30.0,
                    "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(cf.stat().st_mtime)),
                    "video_url": f"/media/video/{cf.name}"
                })

    all_clip_files.sort(key=lambda x: x["size_bytes"], reverse=True)
    all_audio_files.sort(key=lambda x: x["size_bytes"], reverse=True)
    all_section_files.sort(key=lambda x: x["size_bytes"], reverse=True)

    total_media_bytes = total_audio_bytes + total_clips_bytes + total_sections_bytes

    return {
        "disk": {
            "total_gb": disk_total_gb,
            "free_gb": disk_free_gb,
            "used_gb": disk_used_gb,
            "free_pct": free_pct
        },
        "summary": {
            "total_media_mb": round(total_media_bytes / (1024 * 1024), 1),
            "total_media_gb": round(total_media_bytes / (1024**3), 2),
            "clips_mb": round(total_clips_bytes / (1024 * 1024), 1),
            "clips_count": len(all_clip_files),
            "audio_mb": round(total_audio_bytes / (1024 * 1024), 1),
            "audio_count": len(all_audio_files),
            "sections_mb": round(total_sections_bytes / (1024 * 1024), 1),
            "sections_count": len(all_section_files),
        },
        "streams": streams_data,
        "files": {
            "audio": all_audio_files,
            "clips": all_clip_files,
            "sections": all_section_files
        }
    }


@app.post("/api/storage/clean-stream-sections/{key}")
def clean_stream_sections(key: str):
    d = stream_dir(key)
    if not d.exists():
        raise HTTPException(status_code=404, detail="Stream not found")
    sec_dir = d / "sections"
    freed = 0
    if sec_dir.exists():
        for sf in sec_dir.glob("*"):
            if sf.is_file():
                freed += sf.stat().st_size
                sf.unlink(missing_ok=True)
    return {"ok": True, "freed_mb": round(freed / (1024 * 1024), 1)}


@app.post("/api/storage/delete-stream/{key}")
def delete_stream_full(key: str):
    d = stream_dir(key)
    if not d.exists():
        raise HTTPException(status_code=404, detail="Stream not found")
    shutil.rmtree(d, ignore_errors=True)
    return {"ok": True, "message": f"Stream {key} data deleted"}


# --- Settings & Subtitle / Kinetic Typography APIs ---
@app.get("/api/settings")
def api_get_settings():
    s = settings()
    current_style = captions.get_current_style()
    return {
        "settings": s,
        "style": current_style,
        "available_presets": list(captions.STYLES.keys()),
        "presets_detail": captions.STYLES
    }


@app.post("/api/settings")
def api_save_settings(payload: dict):
    s = settings()
    s.update(payload)
    save_settings(s)
    return {"ok": True, "settings": s, "style": captions.get_current_style()}


# --- Edit Clip Hook & Transcripts ---
@app.get("/api/streams/{key}/clips/{cid}/details")
def get_clip_details(key: str, cid: str):
    d = stream_dir(key)
    final_file = d / "final.json"
    if not final_file.exists():
        raise HTTPException(status_code=404, detail="Stream final.json not found")
    final_data = read_json(final_file, {"clips": []})
    clip = next((c for c in final_data.get("clips", []) if c.get("id") == cid), None)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found in final.json")
    
    # Locate words
    render_words_file = d / "render" / cid / "words.json"
    words = []
    if render_words_file.exists():
        words = read_json(render_words_file, [])
    else:
        for f in (d / "clips").glob(f"*{cid}*.words.json"):
            if f.exists():
                words = read_json(f, [])
                break
    
    if not words:
        wav = d / "render" / cid / "clip16k.wav"
        if wav.exists():
            try:
                words = listen.clip_words(wav)
                render_words_file.parent.mkdir(parents=True, exist_ok=True)
                write_json(render_words_file, words)
            except Exception as e:
                print("Error extracting words:", e)

    groups = captions.group_words(words, max_words=2) if words else []
    cues = []
    for g in groups:
        if g:
            cues.append({
                "start": g[0]["s"],
                "end": g[-1]["e"],
                "start_hms": util.hms(g[0]["s"]),
                "end_hms": util.hms(g[-1]["e"]),
                "text": " ".join(x.get("t", x.get("w", "")) for x in g),
                "words": g
            })

    full_transcript = " ".join(w.get("w", w.get("t", "")) for w in words)

    return {
        "key": key,
        "cid": cid,
        "hook": clip.get("hook", ""),
        "caption": clip.get("caption", ""),
        "type": clip.get("type", "highlight"),
        "hashtags": clip.get("hashtags", []),
        "words": words,
        "cues": cues,
        "full_transcript": full_transcript,
        "style": captions.get_current_style()
    }


def match_clip_id(target_id: str, candidate_id: str) -> bool:
    if not target_id or not candidate_id:
        return False
    t = str(target_id).strip().lower().replace("_", "").replace("-", "")
    c = str(candidate_id).strip().lower().replace("_", "").replace("-", "")
    if t == c:
        return True
    t_num = re.sub(r"[^\d]", "", t)
    c_num = re.sub(r"[^\d]", "", c)
    if t_num and c_num and int(t_num) == int(c_num):
        return True
    return False


@app.get("/api/streams/{key}/clips/{cid}/editor-data")
def get_clip_editor_data(key: str, cid: str):
    d = stream_dir(key)
    actual_key = key
    clip = None

    # 1. Look in specified stream directory
    if d.exists():
        final_file = d / "final.json"
        final_data = read_json(final_file, {"clips": []}) if final_file.exists() else {"clips": []}
        clip = next((c for c in final_data.get("clips", []) if match_clip_id(c.get("id"), cid)), None)
        if not clip:
            shortlist = read_json(d / "shortlist.json", [])
            clip = next((c for c in shortlist if match_clip_id(c.get("id"), cid)), None)

    # 2. If not found or key is 'vod', search all streams
    if not clip:
        for s_dir in (WORK_DIR / "streams").glob("*"):
            fj = s_dir / "final.json"
            if fj.exists():
                c_list = read_json(fj, {}).get("clips", [])
                match = next((c for c in c_list if match_clip_id(c.get("id"), cid)), None)
                if match:
                    clip = match
                    d = s_dir
                    actual_key = s_dir.name
                    break

    # 3. Check dated clips metadata if still not found
    if not clip:
        for c_dir in (WORK_DIR / "clips").glob("*"):
            if c_dir.is_dir():
                for mf in c_dir.glob("*.json"):
                    m = read_json(mf, {})
                    if match_clip_id(m.get("id"), cid) or match_clip_id(mf.stem.split("_")[0], cid):
                        s_name = m.get("stream")
                        if s_name and (WORK_DIR / "streams" / s_name).exists():
                            d = WORK_DIR / "streams" / s_name
                            actual_key = s_name
                            fj = d / "final.json"
                            c_list = read_json(fj, {}).get("clips", [])
                            clip = next((c for c in c_list if match_clip_id(c.get("id"), cid)), m)
                            break
            if clip:
                break

    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

    actual_cid = clip.get("id") or cid

    video_url = None
    rendered_file = None
    if (d / "clips").exists():
        for f in (d / "clips").glob("*.mp4"):
            meta = read_json(f.with_suffix(".json"), {})
            if match_clip_id(meta.get("id"), actual_cid) or match_clip_id(f.stem.split("_")[0], actual_cid):
                rendered_file = f
                video_url = f"/media/video/{f.name}"
                break

    words = []
    render_words_file = d / "render" / actual_cid / "words.json"
    if render_words_file.exists():
        words = read_json(render_words_file, [])
    elif rendered_file and rendered_file.with_suffix(".words.json").exists():
        words = read_json(rendered_file.with_suffix(".words.json"), [])
    else:
        for f in (d / "clips").glob("*.words.json"):
            stem = f.name.replace(".words.json", "")
            if match_clip_id(stem.split("_")[0], actual_cid) or actual_cid in stem:
                words = read_json(f, [])
                break
    
    if not words:
        wav = d / "render" / actual_cid / "clip16k.wav"
        if wav.exists():
            try:
                words = listen.clip_words(wav)
                render_words_file.parent.mkdir(parents=True, exist_ok=True)
                write_json(render_words_file, words)
            except Exception as e:
                print("Error extracting words:", e)

    raw_video_url = None
    # 1. Direct .mp4 match in sections
    for sec_file in [d / "sections" / f"{actual_cid}.mp4", d / "sections" / f"{cid}.mp4"]:
        if sec_file.exists() and sec_file.stat().st_size > 10000:
            raw_video_url = f"/media/video/{sec_file.name}"
            break

    # 2. Match by number (e.g. clip_01 vs c01)
    if not raw_video_url and (d / "sections").exists():
        for f in (d / "sections").glob("*.mp4"):
            if match_clip_id(f.stem, actual_cid) and f.stat().st_size > 10000:
                raw_video_url = f"/media/video/{f.name}"
                break

    # 3. If only .ts exists, remux to .mp4 instantly
    if not raw_video_url and (d / "sections").exists():
        for f in (d / "sections").glob("*.ts"):
            if match_clip_id(f.stem, actual_cid) and f.stat().st_size > 10000:
                mp4_target = f.with_suffix(".mp4")
                if not mp4_target.exists():
                    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(f), "-c", "copy", "-movflags", "+faststart", str(mp4_target)], check=False)
                if mp4_target.exists():
                    raw_video_url = f"/media/video/{mp4_target.name}"
                    break

    # 4. Fallback if sections missing: cut 16:9 from live_stream.mp4
    if not raw_video_url and (d / "live_stream.mp4").exists() and "start" in clip and "end" in clip:
        s_t = max(0.0, util.parse_t(clip["start"]) - 4)
        e_t = util.parse_t(clip["end"]) + 4
        sec_cut = d / "sections" / f"{actual_cid}.mp4"
        sec_cut.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{s_t:.2f}", "-to", f"{e_t:.2f}", "-i", str(d / "live_stream.mp4"), "-c", "copy", "-movflags", "+faststart", str(sec_cut)], check=False)
        if sec_cut.exists():
            raw_video_url = f"/media/video/{sec_cut.name}"

    # Clean 9:16 vertical video without burned-in subtitles or hook:
    clean_p = d / "render" / actual_cid / "clean.mp4"
    if clean_p.exists() and clean_p.stat().st_size < 10000:
        clean_p.unlink(missing_ok=True)
        clean_p = None

    if not clean_p or not clean_p.exists():
        clean_p = ensure_clean_video(actual_key, actual_cid)
    clean_video_url = f"/media/clean/{actual_key}/{actual_cid}.mp4" if (clean_p and clean_p.exists() and clean_p.stat().st_size > 10000) else (video_url or raw_video_url)

    # PeekCut 9:16 video (16:9 uncropped centered + blurred background):
    peek_p = d / "render" / actual_cid / "peekcut.mp4"
    if peek_p.exists() and peek_p.stat().st_size < 10000:
        peek_p.unlink(missing_ok=True)
        peek_p = None

    if not peek_p or not peek_p.exists():
        peek_p = ensure_peekcut_video(actual_key, actual_cid)
    peekcut_video_url = f"/media/peekcut/{actual_key}/{actual_cid}.mp4" if (peek_p and peek_p.exists() and peek_p.stat().st_size > 10000) else (video_url or clean_video_url)

    # Master editor video URL MUST be the real rendered clip so it always plays immediately:
    master_video_url = video_url or clean_video_url or raw_video_url

    duration = 30.0
    if clean_p and clean_p.exists():
        try:
            pv = util.probe_video(clean_p)
            duration = float(pv.get("dur", 30.0))
        except Exception:
            pass
    elif peek_p and peek_p.exists():
        try:
            pv = util.probe_video(peek_p)
            duration = float(pv.get("dur", 30.0))
        except Exception:
            pass
    elif rendered_file and rendered_file.exists():
        try:
            meta = read_json(rendered_file.with_suffix(".json"), {})
            duration = meta.get("length", 30.0)
        except Exception:
            pass
    elif "start" in clip and "end" in clip:
        try:
            duration = util.parse_t(clip["end"]) - util.parse_t(clip["start"])
        except Exception:
            pass

    return {
        "key": key,
        "cid": actual_cid,
        "title": clip.get("title") or clip.get("hook", f"Clip {actual_cid}"),
        "hook": clip.get("hook", ""),
        "hook_duration": float(clip.get("hook_duration", 3.5)),
        "hook_start": float(clip.get("hook_start", 0.0)),
        "hook_y": clip.get("hook_y"),
        "cap_y": clip.get("cap_y"),
        "layout": clip.get("layout", "stack"),
        "gameplay_zoom": float(clip.get("gameplay_zoom", 1.0)),
        "gameplay_pan_x": float(clip.get("gameplay_pan_x", 0.0)),
        "facecam_zoom": float(clip.get("facecam_zoom", 1.0)),
        "facecam_pan_x": float(clip.get("facecam_pan_x", 0.0)),
        "preset": clip.get("preset", "pop_glow"),
        "music_file": clip.get("music_file", ""),
        "music_url": f"/media/audio/{clip.get('music_file')}" if clip.get("music_file") else "",
        "music_volume": float(clip.get("music_volume", 0.20)),
        "music_ducking": bool(clip.get("music_ducking", True)),
        "video_url": master_video_url,
        "clean_video_url": clean_video_url or master_video_url,
        "peekcut_video_url": peekcut_video_url or master_video_url,
        "raw_video_url": raw_video_url or master_video_url,
        "rendered_video_url": video_url,
        "has_clean_video": bool(clean_video_url),
        "has_peekcut_video": bool(peek_p and peek_p.exists()),
        "duration": round(duration, 2),
        "words": words,
        "presets": captions.STYLES
    }


@app.post("/api/streams/{key}/clips/{cid}/adjust-bounds")
def adjust_clip_bounds(key: str, cid: str, payload: dict):
    """
    Adjusts the start and/or end time of a clip (e.g. start_delta = -5.0s, end_delta = +5.0s).
    Re-cuts the section from the source video, re-transcribes words using GPU,
    and updates final.json.
    """
    start_delta = float(payload.get("start_delta", 0.0))
    end_delta = float(payload.get("end_delta", 0.0))
    if abs(start_delta) < 0.05 and abs(end_delta) < 0.05:
        return {"ok": True, "message": "No change"}

    d = stream_dir(key)
    actual_key = key
    if not d.exists() or key == "vod":
        for s_dir in (WORK_DIR / "streams").glob("*"):
            fj = s_dir / "final.json"
            if fj.exists():
                c_list = read_json(fj, {}).get("clips", [])
                match = next((c for c in c_list if match_clip_id(c.get("id"), cid)), None)
                if match:
                    d = s_dir
                    actual_key = s_dir.name
                    break

    final_file = d / "final.json"
    if not final_file.exists():
        raise HTTPException(status_code=404, detail="Stream final.json not found")

    final_data = read_json(final_file, {"clips": []})
    clip = next((c for c in final_data.get("clips", []) if match_clip_id(c.get("id"), cid)), None)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

    actual_cid = clip.get("id") or cid
    orig_start = util.parse_t(clip.get("start", "00:00"))
    orig_end = util.parse_t(clip.get("end", "00:30"))

    new_start = max(0.0, orig_start + start_delta)
    new_end = max(new_start + 4.0, orig_end + end_delta)
    new_dur = round(new_end - new_start, 2)

    clip["start"] = util.hms(new_start)
    clip["end"] = util.hms(new_end)
    clip["duration"] = new_dur
    write_json(final_file, final_data)

    sec_file = d / "sections" / f"{actual_cid}.mp4"
    sec_file.parent.mkdir(parents=True, exist_ok=True)
    live_stream = d / "live_stream.mp4"

    ffmpeg_bin = os.environ.get("PEAKCUT_FFMPEG_DIR", "")
    ffmpeg_exe = os.path.join(ffmpeg_bin, "ffmpeg.exe") if ffmpeg_bin else (shutil.which("ffmpeg") or "ffmpeg")

    if live_stream.exists():
        s_t = max(0.0, new_start - 2.0)
        e_t = new_end + 2.0
        subprocess.run([
            ffmpeg_exe, "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{s_t:.2f}", "-to", f"{e_t:.2f}",
            "-i", str(live_stream),
            "-c", "copy", "-movflags", "+faststart",
            str(sec_file)
        ], check=False)
    else:
        # VOD stream: remove previous section file so fetch.download_section re-downloads exact new boundary
        if sec_file.exists():
            try: sec_file.unlink()
            except Exception: pass

    # Clean previous rendered clip file across stream clips and all dated folders so old version is deleted!
    target_clips = d / "clips"
    target_clips.mkdir(parents=True, exist_ok=True)
    for old_f in list(target_clips.glob("*")):
        if match_clip_id(old_f.stem.split("_")[0], actual_cid) or old_f.stem.startswith(f"{actual_cid}_"):
            try: old_f.unlink()
            except Exception: pass

    dated_root = WORK_DIR / "clips"
    if dated_root.exists():
        for folder in dated_root.glob("*"):
            if folder.is_dir() and folder.name != "saved_clips":
                for old_df in list(folder.glob("*")):
                    if match_clip_id(old_df.stem.split("_")[0], actual_cid) or old_df.stem.startswith(f"{actual_cid}_"):
                        m_f = old_df.with_suffix(".json")
                        m = read_json(m_f, {}) if m_f.exists() else {}
                        if not m or m.get("stream") == actual_key or actual_key in old_df.name:
                            try: old_df.unlink()
                            except Exception: pass

    render_d = d / "render" / actual_cid
    render_d.mkdir(parents=True, exist_ok=True)
    for old_rf in [render_d / "clean.mp4", render_d / "peekcut.mp4"]:
        if old_rf.exists():
            try: old_rf.unlink()
            except Exception: pass

    # Re-render with RTX GPU NVENC!
    from clipfarm import render
    made = render.render_stream(actual_key, only=actual_cid)
    rendered_file = ""
    new_words = []
    if made:
        rendered_file = made[0].get("file", "")
        stem = Path(rendered_file).stem
        if dated_root.exists():
            for folder in dated_root.glob("*"):
                if folder.is_dir():
                    for item in folder.glob(f"{stem}*"):
                        shutil.copy2(item, target_clips / item.name)
                    if folder.name != "saved_clips":
                        for old_df in list(folder.glob("*")):
                            if (match_clip_id(old_df.stem.split("_")[0], actual_cid) or old_df.stem.startswith(f"{actual_cid}_")) and not old_df.name.startswith(stem):
                                try: old_df.unlink()
                                except Exception: pass
        clip["file"] = rendered_file
        new_words = made[0].get("words_list") or []
        write_json(final_file, final_data)

    threading.Thread(target=ensure_clean_video, args=(actual_key, actual_cid, True), daemon=True).start()

    return {
        "ok": True,
        "cid": actual_cid,
        "start": clip["start"],
        "end": clip["end"],
        "duration": new_dur,
        "rendered": bool(rendered_file),
        "file": rendered_file,
        "rendered_video_url": f"/api/streams/{actual_key}/clips/{actual_cid}/video?t={time.time()}",
        "words": new_words
    }


@app.post("/api/streams/{key}/clips/{cid}/agent-generate")
def agent_generate_clip_package(key: str, cid: str):
    """
    Autonomous creative agent for high-converting hook, alt hooks, and captions
    using transcript, heatmaps, and signal context.
    """
    d = stream_dir(key)
    if not d.exists() or key == "vod":
        for s_dir in (WORK_DIR / "streams").glob("*"):
            fj = s_dir / "final.json"
            if fj.exists():
                c_list = read_json(fj, {}).get("clips", [])
                match = next((c for c in c_list if match_clip_id(c.get("id"), cid)), None)
                if match:
                    d = s_dir
                    break

    final_file = d / "final.json"
    final_data = read_json(final_file, {"clips": []}) if final_file.exists() else {"clips": []}
    clip = next((c for c in final_data.get("clips", []) if match_clip_id(c.get("id"), cid)), None)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

    actual_cid = clip.get("id") or cid
    info = read_json(d / "info.json", {})
    streamer = info.get("streamer") or "Streamer"
    stream_title = info.get("title") or "Live Stream"

    words_file = d / "render" / actual_cid / "words.json"
    dlg = ""
    if words_file.exists():
        w_list = read_json(words_file, [])
        dlg = " ".join(w.get("w", "") for w in w_list).strip()

    pkg = generate_ai_clip_hook_and_caption(
        streamer=streamer,
        stream_title=stream_title,
        dialogue=dlg,
        why_signals=clip.get("why") or "High audience engagement moment",
        cand_title=clip.get("title") or clip.get("hook") or ""
    )

    clip["hook"] = pkg["hook"]
    clip["caption"] = pkg["caption"]
    clip["hashtags"] = pkg.get("hashtags", [])
    if final_file.exists():
        write_json(final_file, final_data)

    return {
        "ok": True,
        "hook": pkg["hook"],
        "alt_hooks": pkg.get("alt_hooks", []),
        "caption": pkg["caption"],
        "hashtags": pkg.get("hashtags", [])
    }


@app.get("/api/music/presets")
def get_music_presets():
    return [
        {"id": "none", "name": "None (Stream Audio Only)", "filename": "", "url": ""},
        {"id": "lofi", "name": "🎵 Chill Lo-Fi Beat", "filename": "lofi_chill_beat.mp3", "url": "/media/audio/lofi_chill_beat.mp3"},
        {"id": "phonk", "name": "🔥 Phonk & Bass Energy", "filename": "phonk_hype_energy.mp3", "url": "/media/audio/phonk_hype_energy.mp3"},
        {"id": "ambient", "name": "✨ Ambient Glow Synth", "filename": "ambient_glow_synth.mp3", "url": "/media/audio/ambient_glow_synth.mp3"},
    ]


@app.post("/api/streams/{key}/clips/{cid}/upload-music")
async def upload_clip_music(key: str, cid: str, file: UploadFile = File(...)):
    d = stream_dir(key)
    music_dir = d / "music"
    music_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", file.filename)
    dest = music_dir / safe_name
    content = await file.read()
    dest.write_bytes(content)
    return {
        "ok": True,
        "filename": safe_name,
        "url": f"/media/audio/{safe_name}",
        "size_mb": round(len(content) / (1024 * 1024), 2)
    }


@app.post("/api/streams/{key}/clips/{cid}/save-draft")
def save_clip_draft(key: str, cid: str, payload: dict):
    d = stream_dir(key)
    actual_key = key
    clip = None
    if d.exists():
        final_file = d / "final.json"
        if final_file.exists():
            final_data = read_json(final_file, {"clips": []})
            clip = next((c for c in final_data.get("clips", []) if match_clip_id(c.get("id"), cid)), None)

    if not clip:
        for s_dir in (WORK_DIR / "streams").glob("*"):
            fj = s_dir / "final.json"
            if fj.exists():
                c_list = read_json(fj, {}).get("clips", [])
                match = next((c for c in c_list if match_clip_id(c.get("id"), cid)), None)
                if match:
                    clip = match
                    d = s_dir
                    actual_key = s_dir.name
                    final_file = fj
                    final_data = read_json(final_file, {"clips": []})
                    break

    if not clip:
        for c_dir in (WORK_DIR / "clips").glob("*"):
            if c_dir.is_dir():
                for mf in c_dir.glob("*.json"):
                    if not mf.name.endswith(".words.json"):
                        m = read_json(mf, {})
                        if match_clip_id(m.get("id"), cid) or match_clip_id(mf.stem.split("_")[0], cid):
                            s_name = m.get("stream")
                            if s_name and (WORK_DIR / "streams" / s_name).exists():
                                d = WORK_DIR / "streams" / s_name
                                actual_key = s_name
                                final_file = d / "final.json"
                                final_data = read_json(final_file, {"clips": []}) if final_file.exists() else {"clips": []}
                                clip = next((c for c in final_data.get("clips", []) if match_clip_id(c.get("id"), cid)), m)
                                break
            if clip:
                break

    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

    actual_cid = clip.get("id") or cid

    if "hook" in payload:
        clip["hook"] = (payload["hook"] or "").strip()
        clip["title"] = clip["hook"]
    if "caption" in payload:
        clip["caption"] = (payload["caption"] or "").strip()
    if "layout" in payload and payload["layout"]:
        clip["layout"] = payload["layout"]
    if "hook_duration" in payload:
        try: clip["hook_duration"] = float(payload["hook_duration"])
        except (ValueError, TypeError): pass
    if "hook_start" in payload:
        try: clip["hook_start"] = max(0.0, float(payload["hook_start"]))
        except (ValueError, TypeError): pass
    if "hook_y" in payload and payload["hook_y"] is not None:
        try: clip["hook_y"] = int(payload["hook_y"])
        except (ValueError, TypeError): pass
    if "cap_y" in payload and payload["cap_y"] is not None:
        try: clip["cap_y"] = int(payload["cap_y"])
        except (ValueError, TypeError): pass
    if "gameplay_zoom" in payload:
        try: clip["gameplay_zoom"] = float(payload["gameplay_zoom"])
        except (ValueError, TypeError): pass
    if "gameplay_pan_x" in payload:
        try: clip["gameplay_pan_x"] = float(payload["gameplay_pan_x"])
        except (ValueError, TypeError): pass
    if "facecam_zoom" in payload:
        try: clip["facecam_zoom"] = float(payload["facecam_zoom"])
        except (ValueError, TypeError): pass
    if "facecam_pan_x" in payload:
        try: clip["facecam_pan_x"] = float(payload["facecam_pan_x"])
        except (ValueError, TypeError): pass
    if "preset" in payload and payload["preset"]:
        clip["preset"] = payload["preset"]
    if "music_file" in payload:
        clip["music_file"] = (payload["music_file"] or "").strip()
    if "music_volume" in payload:
        try: clip["music_volume"] = float(payload["music_volume"])
        except (ValueError, TypeError): pass
    if "music_ducking" in payload:
        clip["music_ducking"] = bool(payload["music_ducking"])

    if "words" in payload and isinstance(payload["words"], list) and payload["words"]:
        existing_words_file = d / "render" / actual_cid / "words.json"
        existing_words_file.parent.mkdir(parents=True, exist_ok=True)
        write_json(existing_words_file, payload["words"])
        clip["custom_words"] = payload["words"]

    if 'final_file' in locals() and final_file.exists():
        write_json(final_file, final_data)

    # Also update sidecar metadata in stream clips and dated clips
    for c_dir in [d / "clips"] + list((WORK_DIR / "clips").glob("*")):
        if c_dir.exists() and c_dir.is_dir():
            for mf in c_dir.glob("*.json"):
                if not mf.name.endswith(".words.json"):
                    m = read_json(mf, {})
                    if match_clip_id(m.get("id"), actual_cid) or match_clip_id(mf.stem.split("_")[0], actual_cid):
                        if "hook" in payload:
                            m["hook"] = clip["hook"]
                            m["title"] = clip["hook"]
                        if "caption" in payload:
                            m["caption"] = clip["caption"]
                        if "preset" in payload:
                            m["preset"] = clip["preset"]
                        if "layout" in payload:
                            m["layout"] = clip["layout"]
                        write_json(mf, m)

    return {"ok": True, "message": "Draft saved successfully!"}


@app.post("/api/streams/{key}/clips/{cid}/rerender-custom")
def rerender_clip_custom(key: str, cid: str, payload: dict):
    d = stream_dir(key)
    final_file = d / "final.json"
    if not final_file.exists():
        raise HTTPException(status_code=404, detail="final.json not found")
    final_data = read_json(final_file, {"clips": []})
    clip = next((c for c in final_data.get("clips", []) if match_clip_id(c.get("id"), cid)), None)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

    actual_cid = clip.get("id") or cid

    new_hook = (payload.get("hook") or clip.get("hook", "")).strip()
    new_caption = (payload.get("caption") or clip.get("caption", "")).strip()
    clip["hook"] = new_hook
    clip["caption"] = new_caption

    if "layout" in payload and payload["layout"]:
        clip["layout"] = payload["layout"]
    if "hook_duration" in payload:
        try:
            clip["hook_duration"] = float(payload["hook_duration"])
        except (ValueError, TypeError):
            pass
    if "hook_start" in payload:
        try:
            clip["hook_start"] = max(0.0, float(payload["hook_start"]))
        except (ValueError, TypeError):
            pass
    if "hook_y" in payload and payload["hook_y"] is not None:
        try:
            clip["hook_y"] = int(payload["hook_y"])
        except (ValueError, TypeError):
            pass
    if "cap_y" in payload and payload["cap_y"] is not None:
        try:
            clip["cap_y"] = int(payload["cap_y"])
        except (ValueError, TypeError):
            pass
    if "gameplay_zoom" in payload:
        try:
            clip["gameplay_zoom"] = float(payload["gameplay_zoom"])
        except (ValueError, TypeError):
            pass
    if "gameplay_pan_x" in payload:
        try:
            clip["gameplay_pan_x"] = float(payload["gameplay_pan_x"])
        except (ValueError, TypeError):
            pass
    if "facecam_zoom" in payload:
        try:
            clip["facecam_zoom"] = float(payload["facecam_zoom"])
        except (ValueError, TypeError):
            pass
    if "facecam_pan_x" in payload:
        try:
            clip["facecam_pan_x"] = float(payload["facecam_pan_x"])
        except (ValueError, TypeError):
            pass
    if "preset" in payload and payload["preset"]:
        clip["preset"] = payload["preset"]

    existing_words_file = d / "render" / cid / "words.json"
    existing_words = read_json(existing_words_file, []) if existing_words_file.exists() else []

    if "words" in payload and isinstance(payload["words"], list) and payload["words"]:
        updated_words = payload["words"]
    elif "cues" in payload and isinstance(payload["cues"], list) and payload["cues"]:
        updated_words = []
        for cue in payload["cues"]:
            cue_words = [w.strip() for w in cue.get("text", "").split() if w.strip()]
            if not cue_words:
                continue
            s_val = float(cue.get("start", 0.0))
            e_val = float(cue.get("end", s_val + 1.0))
            step = (e_val - s_val) / len(cue_words)
            for i, cw in enumerate(cue_words):
                ws = round(s_val + i * step, 2)
                we = round(ws + step * 0.9, 2)
                updated_words.append({"w": cw, "s": ws, "e": we})
    elif "full_transcript" in payload and payload["full_transcript"].strip():
        new_text = payload["full_transcript"].strip()
        new_tokens = [w.strip() for w in new_text.split() if w.strip()]
        if existing_words:
            t0 = existing_words[0]["s"]
            t1 = existing_words[-1]["e"]
            step = max(0.1, (t1 - t0) / len(new_tokens))
            updated_words = []
            for i, tok in enumerate(new_tokens):
                ws = round(t0 + i * step, 2)
                we = round(ws + step * 0.88, 2)
                updated_words.append({"w": tok, "s": ws, "e": we})
        else:
            updated_words = [{"w": tok, "s": round(i * 0.5, 2), "e": round(i * 0.5 + 0.45, 2)} for i, tok in enumerate(new_tokens)]
    else:
        updated_words = existing_words

    existing_words_file.parent.mkdir(parents=True, exist_ok=True)
    write_json(existing_words_file, updated_words)
    clip["custom_words"] = updated_words
    write_json(final_file, final_data)

    if payload.get("preset"):
        s = settings()
        s["subtitle_preset"] = payload["preset"]
        save_settings(s)
    if payload.get("style_overrides"):
        s = settings()
        s.update(payload["style_overrides"])
        save_settings(s)

    made = render.render_stream(key, only=actual_cid)
    try:
        ensure_clean_video(key, actual_cid, force=True)
    except Exception as ex:
        print("Warning updating clean video after render:", ex)
    
    target_clips = d / "clips"
    target_clips.mkdir(parents=True, exist_ok=True)
    if made:
        rendered_name = made[0]["file"]
        stem = Path(rendered_name).stem
        # Clean up any obsolete/previous clip files for this cid in target_clips
        for old_f in list(target_clips.glob("*")):
            if (match_clip_id(old_f.stem.split("_")[0], actual_cid) or old_f.stem.startswith(f"{actual_cid}_")) and not old_f.name.startswith(stem):
                try: old_f.unlink()
                except Exception: pass

        dated_root = WORK_DIR / "clips"
        if dated_root.exists():
            for folder in dated_root.glob("*"):
                if folder.is_dir():
                    for item in folder.glob(f"{stem}*"):
                        shutil.copy2(item, target_clips / item.name)
                    if folder.name != "saved_clips":
                        for old_df in list(folder.glob("*")):
                            if (match_clip_id(old_df.stem.split("_")[0], actual_cid) or old_df.stem.startswith(f"{actual_cid}_")) and not old_df.name.startswith(stem):
                                try: old_df.unlink()
                                except Exception: pass

        clip["file"] = rendered_name
        write_json(final_file, final_data)

        # Also sync to saved_clips if marked as saved
        if clip.get("saved"):
            saved_dir = WORK_DIR / "saved_clips"
            saved_dir.mkdir(parents=True, exist_ok=True)
            for old_s in list(saved_dir.glob("*")):
                if (match_clip_id(old_s.stem.split("_")[0], actual_cid) or old_s.stem.startswith(f"{actual_cid}_")) and not old_s.name.startswith(stem):
                    try: old_s.unlink()
                    except Exception: pass
            for item in target_clips.glob(f"{stem}*"):
                try: shutil.copy2(item, saved_dir / item.name)
                except Exception: pass

        # Mix background music if selected
        music_file = payload.get("music_file") or clip.get("music_file")
        if music_file:
            m_path = find_file_in_work(music_file)
            if m_path and m_path.exists():
                rendered_path = target_clips / rendered_name
                if rendered_path.exists():
                    temp_out = rendered_path.with_name(f"bgm_{rendered_name}")
                    music_vol = float(payload.get("music_volume", 0.20))
                    duck = bool(payload.get("music_ducking", True))
                    dur = 30.0
                    try:
                        dur_res = subprocess.run([
                            "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(rendered_path)
                        ], capture_output=True, text=True)
                        dur = float(dur_res.stdout.strip() or 30.0)
                    except Exception:
                        pass

                    if duck:
                        fc_filter = (
                            f"[1:a]aloop=loop=-1:size=2e+09,atrim=0:{dur:.2f},volume={music_vol:.2f}[bgm];"
                            f"[bgm][0:a]sidechaincompress=threshold=0.08:ratio=4:attack=50:release=350[ducked_bgm];"
                            f"[0:a][ducked_bgm]amix=inputs=2:duration=first:dropout_transition=2[aout]"
                        )
                    else:
                        fc_filter = (
                            f"[1:a]aloop=loop=-1:size=2e+09,atrim=0:{dur:.2f},volume={music_vol:.2f}[bgm];"
                            f"[0:a][bgm]amix=inputs=2:duration=first:dropout_transition=2[aout]"
                        )

                    subprocess.run([
                        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-i", str(rendered_path),
                        "-i", str(m_path),
                        "-filter_complex", fc_filter,
                        "-map", "0:v", "-map", "[aout]",
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                        str(temp_out)
                    ], check=False)

                    if temp_out.exists() and temp_out.stat().st_size > 10000:
                        shutil.move(str(temp_out), str(rendered_path))
                        # Also sync back to dated clips
                        if dated_root.exists():
                            for folder in dated_root.glob("*"):
                                if (folder / rendered_name).exists():
                                    shutil.copy2(rendered_path, folder / rendered_name)

    rendered_name = made[0]["file"] if made else f"{cid}.mp4"
    return {
        "ok": True,
        "message": "Clip re-rendered successfully with custom hook & subtitles!",
        "filename": rendered_name,
        "video_url": f"/media/video/{rendered_name}",
        "download_url": f"/api/download/{rendered_name}",
        "meta": made[0] if made else {}
    }


@app.post("/api/streams/{key}/clips/{cid}/update")
def update_clip_metadata(key: str, cid: str, payload: dict):
    d = stream_dir(key)
    final_file = d / "final.json"
    if not final_file.exists():
        raise HTTPException(status_code=404, detail="final.json not found")
    data = read_json(final_file, {"clips": []})
    found = False
    for c in data.get("clips", []):
        if c.get("id") == cid:
            found = True
            if "hook" in payload:
                c["hook"] = payload["hook"]
            if "caption" in payload:
                c["caption"] = payload["caption"]
            if "start" in payload:
                c["start"] = payload["start"]
            if "end" in payload:
                c["end"] = payload["end"]
            if "hashtags" in payload:
                c["hashtags"] = payload["hashtags"]
            break
    if not found:
        raise HTTPException(status_code=404, detail="Clip ID not found")
    write_json(final_file, data)
    return {"ok": True, "clips": data["clips"]}


# --- AI Curation & Viral Hook Generation ---
def generate_ai_clip_hook_and_caption(streamer: str, stream_title: str, dialogue: str, why_signals: str = "", cand_title: str = "") -> dict:
    from studio import ai
    
    clean_streamer = streamer or "Streamer"
    if clean_streamer.lower() == "morecaseoh":
        clean_streamer = "CaseOh"

    text_content = dialogue.strip()
    if not text_content and cand_title:
        text_content = f"Event: {cand_title}"
    if not text_content:
        text_content = f"{clean_streamer} intense stream highlight"

    # Smart contextual generator for fallback/offline when AI API is unavailable
    dlg_lower = text_content.lower()
    subject = ""
    if "daughter" in dlg_lower and "ugly" in dlg_lower:
        subject = "the game insulted his daughter"
    elif "granny" in dlg_lower:
        subject = "Granny caught him off guard"
    elif "jumpscare" in dlg_lower or "scared" in dlg_lower or "scream" in dlg_lower:
        subject = "this jumpscare completely broke him"
    elif "boat" in dlg_lower or "faster" in dlg_lower:
        subject = "he claimed he was faster than boats"
    elif "die" in dlg_lower or "died" in dlg_lower:
        subject = "nobody believed he survived that"
    elif cand_title and len(cand_title.split()) >= 3 and not cand_title.isupper():
        subject = cand_title
    else:
        # Extract the most impactful 4-5 words from dialogue
        words_clean = [w for w in re.sub(r"[^a-zA-Z\s]", "", text_content).split() if len(w) > 3][:6]
        subject = "this unexpected moment" if not words_clean else f"what happened with {' '.join(words_clean[:3])}"

    if cand_title and not cand_title.isupper() and len(cand_title.split()) >= 3:
        fallback_hook = cand_title
    else:
        fallback_hook = f"{clean_streamer} Can't Believe {subject.capitalize()}" if not subject.startswith(clean_streamer) else subject

    fallback_caption = f"{clean_streamer} was NOT ready for what happened here live on stream 😭 Is this the wildest moment yet?"
    fallback_tags = [f"#{clean_streamer.replace(' ', '')}", "#gaming", "#viral", "#clips", "#funny"]

    if not ai.ready():
        return {
            "hook": fallback_hook,
            "alt_hooks": [f"Nobody expected {clean_streamer} to do this", f"{clean_streamer} loses it on stream"],
            "caption": fallback_caption,
            "hashtags": fallback_tags
        }

    prompt = f"""You are an elite viral clip editor scouting highlights for YouTube Shorts, TikTok, and Instagram Reels.
Streamer: {clean_streamer}
Stream Context: {stream_title or 'Live Stream'}
Moment Signals: {why_signals or 'Audience reaction peak'}

Spoken dialogue / action in this moment:
\"\"\"{text_content[:2000]}\"\"\"

Generate high-converting, viral packaging for this clip:
1. hook: The big on-screen title card shown in the first 3 seconds (MAX 7 WORDS).
   - MUST explain the drama, comedic irony, or tension so a scrolling viewer stops immediately.
   - Use proven viral formulas:
     * "{clean_streamer} finds..."
     * "{clean_streamer} can't believe..."
     * "{clean_streamer} defends..."
     * "{clean_streamer} loses it when..."
     * "Nobody expected {clean_streamer} to..."
     * "This detail broke {clean_streamer}"
   - NEVER quote raw literal speech syllables like "I have a plastic bucket on" or "What I hit". Explain WHAT is happening!
2. alt_hooks: 2 alternative punchy hook titles (max 7 words each).
3. caption: 1-2 punchy sentences for the post description + 1 engaging question for comments.
4. hashtags: 3-5 relevant viral hashtags without # signs.
"""
    schema = {
        "type": "OBJECT",
        "properties": {
            "hook": {"type": "STRING"},
            "alt_hooks": {"type": "ARRAY", "items": {"type": "STRING"}},
            "caption": {"type": "STRING"},
            "hashtags": {"type": "ARRAY", "items": {"type": "STRING"}}
        },
        "required": ["hook", "caption", "hashtags"]
    }

    models_to_try = []
    if ai.models():
        for m in ai.models():
            if m not in models_to_try:
                models_to_try.append(m)
    for fallback_m in ["gemini-2.5-flash", "gemini-3.5-flash-lite", "gemini-3-flash-preview"]:
        if fallback_m not in models_to_try:
            models_to_try.append(fallback_m)

    for m in models_to_try:
        try:
            res, sent, wrote = ai.call(m, prompt, schema)
            if isinstance(res, dict) and res.get("hook"):
                hook = re.sub(r'[\"“”]', '', str(res["hook"])).strip()
                caption = str(res.get("caption", "")).strip()
                if not caption or "Would you have reacted" in caption:
                    caption = f"{clean_streamer} had an unbelievable moment that had everyone talking! What would you have done here? 😱"
                raw_tags = res.get("hashtags", fallback_tags)
                tags = [f"#{re.sub(r'[^a-zA-Z0-9]', '', str(t))}" for t in raw_tags if str(t).strip()]
                alts = [re.sub(r'[\"“”]', '', str(h)).strip() for h in res.get("alt_hooks", []) if str(h).strip()]
                all_hooks = [hook] + alts
                return {
                    "hook": hook,
                    "caption": caption,
                    "hashtags": tags if tags else fallback_tags,
                    "hooks": all_hooks[:3]
                }
        except Exception as e:
            print(f"[generate_ai_clip_hook_and_caption] Model '{m}' error: {e}. Trying next model...", flush=True)
            continue

    # Claude / Anthropic fallback if Gemini quota is depleted and key is available
    claude_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY") or os.environ.get("PEAKCUT_CLAUDE_KEY")
    if claude_key:
        try:
            print("[generate_ai_clip_hook_and_caption] Trying Anthropic Claude fallback...", flush=True)
            import httpx
            claude_res = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": claude_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json={
                    "model": "claude-3-5-haiku-20241022",
                    "max_tokens": 500,
                    "system": "Return ONLY a valid JSON object matching the requested schema with keys: hook, alt_hooks, caption, hashtags.",
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=20.0
            )
            if claude_res.status_code == 200:
                cdata = claude_res.json()
                text = "".join(b.get("text", "") for b in cdata.get("content", []) if b.get("type") == "text")
                match = re.search(r"\{.*\}", text, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(0))
                    if parsed.get("hook"):
                        hook = re.sub(r'[\"“”]', '', str(parsed["hook"])).strip()
                        caption = str(parsed.get("caption", "")).strip() or f"{clean_streamer} had an unbelievable moment! What would you have done? 😱"
                        raw_tags = parsed.get("hashtags", fallback_tags)
                        tags = [f"#{re.sub(r'[^a-zA-Z0-9]', '', str(t))}" for t in raw_tags if str(t).strip()]
                        alts = [re.sub(r'[\"“”]', '', str(h)).strip() for h in parsed.get("alt_hooks", []) if str(h).strip()]
                        return {
                            "hook": hook,
                            "caption": caption,
                            "hashtags": tags if tags else fallback_tags,
                            "hooks": ([hook] + alts)[:3]
                        }
        except Exception as ce:
            print(f"[generate_ai_clip_hook_and_caption] Claude fallback error: {ce}", flush=True)

    return {
        "hook": fallback_hook,
        "caption": f"{clean_streamer} had an unbelievable moment that had everyone talking! What would you have done here? 😱",
        "hashtags": fallback_tags,
        "hooks": [fallback_hook, f"Nobody expected {clean_streamer} to do this", f"{clean_streamer} can't believe what happened"]
    }


def _run_ai_curation(key: str):
    d = stream_dir(key)
    ACTIVE_JOBS[key] = {"action": "ai_curation", "progress": "Analyzing semantic transcript & narrative setup-payoff arcs..."}
    try:
        st = read_json(d / "state.json", {})
        chunks_count = st.get("chunks", 0)
        picks_dir = d / "picks"
        picks_dir.mkdir(parents=True, exist_ok=True)
        info = read_json(d / "info.json", {})
        streamer = info.get("streamer") or "Streamer"
        total_dur = float(info.get("duration") or 3600.0)

        # 1. Load full semantic transcript and sentence lines
        from studio import brain
        segs = read_json(d / "segments.json", [])
        speech_segs = [s for s in segs if s.get("kind") == "speech" and s.get("text", "").strip()]

        tr_data = read_json(d / "transcript.json", {})
        stream_words = tr_data.get("words", [])
        if not stream_words:
            # Check sherpa / whisper transcription tokens format
            for seg in tr_data.get("transcription", []):
                for tok in seg.get("tokens", []):
                    w_txt = (tok.get("text") or "").strip()
                    offs = tok.get("offsets") or {}
                    if w_txt and "from" in offs and "to" in offs:
                        stream_words.append({
                            "w": w_txt,
                            "s": round(offs["from"] / 1000.0, 2),
                            "e": round(offs["to"] / 1000.0, 2)
                        })
        if not stream_words:
            for s in speech_segs:
                words_in_seg = s.get("text", "").split()
                s_t = s.get("s", 0)
                e_t = s.get("e", s_t + 1)
                step = (e_t - s_t) / max(1, len(words_in_seg))
                for idx, w in enumerate(words_in_seg):
                    stream_words.append({
                        "w": w,
                        "s": round(s_t + idx * step, 2),
                        "e": round(s_t + (idx + 1) * step, 2)
                    })

        # 2. Process chunks with dialogue-first semantic snapping
        for i in range(1, chunks_count + 1):
            chunk_file = d / "chunks" / f"chunk_{i:02d}.md"
            out_file = picks_dir / f"chunk_{i:02d}.json"
            if not chunk_file.exists():
                continue
            lines = chunk_file.read_text(encoding="utf-8", errors="replace").splitlines()
            hot_moments = []
            for line in lines:
                if "🔥🔥" in line or "🔥" in line:
                    m = re.search(r"(\d+:\d+(?::\d+)?)", line)
                    if m:
                        t_str = m.group(1)
                        t_sec = util.parse_t(t_str)

                        # Look for narrative setup (speech starting 6-14s before peak)
                        setup_seg = next((s for s in reversed(speech_segs) if (t_sec - 14) <= s.get("s", 0) <= (t_sec - 2)), None)
                        raw_start = setup_seg.get("s", t_sec - 8) if setup_seg else max(0.0, t_sec - 8)

                        # Look for narrative payoff (speech ending 6-18s after peak)
                        payoff_seg = next((s for s in speech_segs if (t_sec + 4) <= s.get("e", 0) <= (t_sec + 22)), None)
                        raw_end = payoff_seg.get("e", t_sec + 18) if payoff_seg else min(total_dur, t_sec + 18)

                        # Use brain._snap to align to sentence start and end with natural breathing tail
                        snap_s, snap_e = brain._snap(
                            {"start": raw_start, "end": raw_end},
                            speech_segs,
                            stream_words,
                            min_len=18.0,
                            max_len=55.0,
                            duration=total_dur
                        )
                        if snap_e <= snap_s:
                            snap_s, snap_e = raw_start, raw_end

                        # Extract actual dialogue spoken in this semantic moment
                        moment_dialogue = " ".join(
                            s["text"] for s in speech_segs if snap_s <= s.get("s", 0) <= snap_e
                        ).strip()

                        hot_moments.append({
                            "start": util.hms(snap_s),
                            "end": util.hms(snap_e),
                            "peak": t_str,
                            "type": "discovery" if "🔥🔥" in line else "funny",
                            "score": 95 if ("🔥🔥" in line and len(moment_dialogue) > 20) else 82,
                            "standalone": True,
                            "title": f"{streamer} highlight",
                            "dialogue": moment_dialogue,
                            "why": "Setup & payoff peak with narrative sentence boundaries."
                        })
            write_json(out_file, {"chunk": i, "moments": hot_moments[:3]})

        ACTIVE_JOBS[key]["progress"] = "Shortlisting best narrative candidates & generating viral AI hooks..."
        kept = shortlist.build(d, top=10)
        final_clips = []
        for cand in kept[:6]:
            s_val = cand.get("start", 0)
            e_val = cand.get("end", s_val + 35)
            s_sec = util.parse_t(s_val) if isinstance(s_val, str) else float(s_val)
            e_sec = util.parse_t(e_val) if isinstance(e_val, str) else float(e_val)

            # Snap to sentence boundaries if needed
            s_sec, e_sec = brain._snap(
                {"start": s_sec, "end": e_sec},
                speech_segs,
                stream_words,
                min_len=18.0,
                max_len=55.0,
                duration=total_dur
            )

            # Find dialogue spoken in this snapped moment
            spoken = [x["text"] for x in speech_segs if s_sec <= x.get("s", 0) <= e_sec]
            dialogue = " ".join(spoken).strip()

            why_info = cand.get("why_signals") or cand.get("why") or "Excitement & punchline peak"

            # Use AI brain / agent for viral packaging
            ai_pkg = generate_ai_clip_hook_and_caption(
                streamer=streamer,
                stream_title=info.get("title") or "",
                dialogue=dialogue,
                why_signals=why_info,
                cand_title=cand.get("title") or ""
            )

            final_clips.append({
                "id": cand.get("id") or f"c{len(final_clips)+1:02d}",
                "start": util.hms(s_sec),
                "end": util.hms(e_sec),
                "peak": cand.get("peak") or util.hms(s_sec),
                "type": cand.get("type", "highlight"),
                "score": cand.get("score", 90),
                "hook": ai_pkg["hook"],
                "caption": ai_pkg["caption"],
                "hashtags": ai_pkg["hashtags"],
                "why": why_info
            })
        write_json(d / "final.json", {"clips": final_clips})
        job_data = {
            "action": "render",
            "progress": f"Detected {len(final_clips)} viral moments! Automatically rendering vertical clips on RTX 4050 GPU...",
            "pct": 10,
            "clips_found_count": len(final_clips),
            "done": False
        }
        ACTIVE_JOBS[key] = job_data
        write_json(d / "progress.json", job_data)
        # Automatically start rendering immediately upon detecting viral moments!
        _run_render_stream(key)
    except Exception as e:
        print(f"[_run_ai_curation] Error: {e}", flush=True)
        ACTIVE_JOBS[key] = {"action": "ai_curation", "error": str(e), "failed": True}


@app.post("/api/streams/{key}/clips/{cid}/generate-ai-hook")
def api_generate_ai_hook(key: str, cid: str):
    d = stream_dir(key)
    final_file = d / "final.json"
    final_data = read_json(final_file, {"clips": []}) if final_file.exists() else {"clips": []}
    clip = next((c for c in final_data.get("clips", []) if match_clip_id(c.get("id"), cid)), None)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

    info = read_json(d / "info.json", {})
    streamer = info.get("streamer") or clip.get("streamer") or "Streamer"
    stream_title = info.get("title") or ""

    # Extract dialogue
    s_val = util.parse_t(clip.get("start", 0))
    e_val = util.parse_t(clip.get("end", s_val + 35))
    segs = read_json(d / "segments.json", [])
    spoken = [x["text"] for x in segs if x.get("kind") == "speech" and s_val <= x.get("s", 0) <= e_val]
    dialogue = " ".join(spoken).strip()

    # If segments has no speech, try render words.json
    if not dialogue:
        render_words_file = d / "render" / (clip.get("id") or cid) / "words.json"
        if render_words_file.exists():
            w_list = read_json(render_words_file, [])
            dialogue = " ".join(w.get("w", "") for w in w_list).strip()

    pkg = generate_ai_clip_hook_and_caption(
        streamer=streamer,
        stream_title=stream_title,
        dialogue=dialogue,
        why_signals=clip.get("why") or "",
        cand_title=clip.get("title") or ""
    )

    # Update in final.json
    clip["hook"] = pkg["hook"]
    clip["caption"] = pkg["caption"]
    clip["hashtags"] = pkg["hashtags"]
    write_json(final_file, final_data)

    # Sync into rendered clip sidecar .json files so clips library is updated immediately
    actual_cid = clip.get("id") or cid
    for c_dir in [d / "clips", WORK_DIR / "saved_clips"]:
        if c_dir.exists():
            for f in c_dir.glob("*.mp4"):
                prefix = f.stem.split("_")[0]
                if match_clip_id(prefix, actual_cid) or actual_cid in f.stem:
                    side_meta = read_json(f.with_suffix(".json"), {})
                    side_meta["hook"] = pkg["hook"]
                    side_meta["caption"] = pkg["caption"]
                    side_meta["hashtags"] = pkg["hashtags"]
                    write_json(f.with_suffix(".json"), side_meta)

    return {
        "ok": True,
        "hook": pkg["hook"],
        "alt_hooks": pkg.get("hooks", [pkg["hook"]]),
        "caption": pkg["caption"],
        "hashtags": pkg["hashtags"]
    }


@app.post("/api/streams/{key}/ai-pick")
def ai_pick_stream(key: str, background_tasks: BackgroundTasks):
    d = stream_dir(key)
    if not d.exists():
        raise HTTPException(status_code=404, detail="Stream not found")
    background_tasks.add_task(_run_ai_curation, key)
    return {"ok": True, "message": "AI curation started"}


def _run_render_stream(key: str, only: Optional[str] = None):
    d = stream_dir(key)

    def on_prog(cur, total, hook, stage, clip_meta):
        if key in CANCEL_JOBS:
            raise KeyboardInterrupt("Stopped by user")
        pct = round(((cur - 1 if stage == "rendering" else cur) / max(1, total)) * 100)
        prog_msg = f"Rendering clip {cur}/{total}: {hook}..." if stage == "rendering" else f"Rendered clip {cur}/{total}: {hook} ✓"
        job_data = {
            "action": "render",
            "progress": prog_msg,
            "pct": pct,
            "clip_current": cur,
            "clip_total": total,
            "current_hook": hook,
            "updated_at": time.time(),
            "done": False
        }
        ACTIVE_JOBS[key] = job_data
        write_json(d / "progress.json", job_data)

    job_data = {
        "action": "render",
        "progress": "Preparing RTX 4050 GPU NVENC & kinetic caption pipeline...",
        "pct": 5,
        "done": False
    }
    ACTIVE_JOBS[key] = job_data
    write_json(d / "progress.json", job_data)
    try:
        render.render_stream(key, only=only, on_progress=on_prog)
        job_data = {
            "action": "render",
            "progress": "Render finished successfully! All clips ready in dashboard.",
            "pct": 100,
            "done": True
        }
        ACTIVE_JOBS[key] = job_data
        write_json(d / "progress.json", job_data)
        if not settings().get("keep_audio", False):
            cleanup_stream_audio(key)
        # Autonomous Zero-Supervision Auto-Pilot trigger:
        threading.Thread(target=trigger_autopost_for_stream, args=(key,), daemon=True).start()
    except BaseException as e:
        print(f"[_run_render_stream] Error: {e}", flush=True)
        ACTIVE_JOBS[key] = {"action": "render", "error": str(e), "failed": True, "done": True}


@app.post("/api/streams/{key}/render")
def render_stream_endpoint(key: str, payload: dict, background_tasks: BackgroundTasks):
    d = stream_dir(key)
    if not d.exists():
        raise HTTPException(status_code=404, detail="Stream not found")
    final = read_json(d / "final.json", {})
    if not final.get("clips"):
        raise HTTPException(status_code=400, detail="No clips in final.json to render. Run AI curation first.")
    only = payload.get("only") if payload else None
    background_tasks.add_task(_run_render_stream, key, only)
    return {"ok": True, "message": "Rendering started"}


# --- Live Stream Watcher & Auto-Clipper (PeakCut Segmented Rolling Architecture) ---
from . import live_pipeline

ACTIVE_LIVE_SESSIONS: dict[str, Any] = {}


def _start_live_pipeline(key: str, url: str, info: dict, auto_clip: bool = True):
    sess = live_pipeline.LivePipelineSession(key, url, info, auto_clip=auto_clip)
    ACTIVE_LIVE_SESSIONS[key] = sess
    sess.start()
    return sess


def _process_live_recording(key: str, dest_video: Path, info: dict):
    d = stream_dir(key)
    ACTIVE_JOBS[key] = {"action": "auto_clip", "progress": "Extracting audio and analyzing peaks for auto-clipping..."}
    try:
        # 1. Extract audio
        wav = d / "audio16k.wav"
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(dest_video),
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            str(wav)
        ], check=True)

        # 2. Get duration
        pv = util.probe_video(dest_video)
        dur = float(pv.get("dur") or 0)
        info["duration"] = dur
        write_json(d / "info.json", info)

        # 3. Transcribe speech
        ACTIVE_JOBS[key]["progress"] = "Transcribing live stream speech (Sherpa-ONNX)..."
        tr = listen.transcribe(wav, d / "transcript")
        segs = listen.segments(tr)
        write_json(d / "segments.json", segs)

        # 4. Energy & Signals
        ACTIVE_JOBS[key]["progress"] = "Detecting loudness & laughter spikes..."
        en = listen.energy(wav)
        sig = signals.build(dur, en, None, None, [], segs)
        pk = signals.peaks(sig)
        np.savez_compressed(d / "signals.npz", hype=sig["hype"], **{f"raw_{k}": v for k, v in sig["raw"].items()})
        write_json(d / "peaks.json", pk)
        write_json(d / "spark.json", signals.sparkline(sig))

        # 5. Write chunks & Run AI curation
        ACTIVE_JOBS[key]["progress"] = "Curating top viral moments..."
        n_chunks = stream.write_chunks(d, info, segs, sig, pk, 15.0)
        st = {
            "key": key,
            "url": info.get("url", ""),
            "scanned": time.time(),
            "duration": dur,
            "streamer": info.get("streamer", "Streamer"),
            "title": info.get("title", ""),
            "platform": info.get("platform", "web"),
            "chunks": n_chunks
        }
        write_json(d / "state.json", st)
        _run_ai_curation(key)

        # 6. Render vertical clips with RTX 4050 NVENC
        ACTIVE_JOBS[key]["progress"] = "Rendering finished 9:16 vertical clips with RTX 4050 NVENC..."
        _run_render_stream(key)

        ACTIVE_JOBS[key] = {
            "action": "live_record",
            "progress": "✅ Live stream finished and vertical clips created!",
            "done": True
        }
    except Exception as e:
        ACTIVE_JOBS[key] = {"action": "auto_clip", "error": str(e), "failed": True}


@app.get("/api/live/status")
def get_live_sessions():
    active = []
    for k, s in list(ACTIVE_LIVE_SESSIONS.items()):
        if hasattr(s, "get_status"):
            active.append(s.get_status())
        elif isinstance(s, dict):
            elapsed = int(time.time() - s.get("started_at", time.time()))
            active.append({
                "key": k,
                "url": s.get("url", ""),
                "streamer": s.get("streamer", "Streamer"),
                "title": s.get("title", "Live Stream"),
                "platform": s.get("platform", "web"),
                "status": s.get("status", "unknown"),
                "elapsed_hms": util.hms(elapsed),
                "audio_downloaded_sec": 0,
                "audio_transcribed_sec": 0,
                "transcribe_pct": 0,
                "clips_found_count": 0,
                "clips_editing": [],
                "clips_ready": []
            })
    return {"sessions": active}


@app.post("/api/live/start")
def start_live_session(payload: dict = Body(...), background_tasks: BackgroundTasks = None):
    url = (payload.get("url") or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL required")
    auto_clip = payload.get("auto_clip", True)

    def _run_start():
        info_full = fetch.get_info(url)
        key = fetch.stream_key(info_full)
        info = stream._trim_info(info_full, url)
        _start_live_pipeline(key, url, info, auto_clip=auto_clip)

    if background_tasks:
        background_tasks.add_task(_run_start)
    else:
        threading.Thread(target=_run_start, daemon=True).start()
    return {"ok": True, "message": "Live pipeline initialized"}


@app.post("/api/live/{key}/stop")
def stop_live_session(key: str, payload: dict = Body({})):
    session = ACTIVE_LIVE_SESSIONS.get(key)
    if not session:
        raise HTTPException(status_code=404, detail="Live session not found")
    if hasattr(session, "stop"):
        session.stop("Stopped by user")
        return {"ok": True, "message": "Live pipeline stopping. Queued renders will complete."}
    if isinstance(session, dict):
        proc = session.get("proc")
        if proc and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
        session["status"] = "stopped"
    return {"ok": True, "message": "Session stopped."}


CANCEL_JOBS: set[str] = set()


def _run_scan_thread(url: str, force_live: bool = False):
    try:
        ACTIVE_JOBS[url] = {"action": "scan", "progress": "Checking stream status...", "error": None}
        info_full = fetch.get_info(url)
        is_live = bool(info_full.get("is_live") or info_full.get("live_status") == "is_live" or force_live)
        key = fetch.stream_key(info_full)
        d = stream_dir(key)
        d.mkdir(parents=True, exist_ok=True)
        info = stream._trim_info(info_full, url)
        write_json(d / "info.json", info)

        if is_live:
            # LIVE stream detected! Launch PeakCut rolling catch-up pipeline
            ACTIVE_JOBS[url] = {"action": "live_pipeline", "progress": f"Connecting live pipeline to {info.get('streamer')}...", "live": True}
            sess = _start_live_pipeline(key, url, info, auto_clip=True)
            ACTIVE_JOBS[key] = {
                "action": "live_pipeline",
                "progress": f"🔴 Live pipeline active ({info.get('streamer')}). Catching up and auto-clipping...",
                "live": True
            }
            write_json(d / "progress.json", ACTIVE_JOBS[key])
            ACTIVE_JOBS.pop(url, None)
        else:
            # VOD processing with continuous progress tracking, auto-curation, and auto-render!
            def on_scan_prog(stage: str, pct: int, msg: str):
                if key in CANCEL_JOBS:
                    CANCEL_JOBS.discard(key)
                    job_data = {"action": "stopped", "progress": "Stopped by user", "done": True}
                    ACTIVE_JOBS[key] = job_data
                    write_json(d / "progress.json", job_data)
                    return False
                job_data = {
                    "action": stage,
                    "progress": msg,
                    "pct": pct,
                    "updated_at": time.time()
                }
                ACTIVE_JOBS[key] = job_data
                write_json(d / "progress.json", job_data)
                return True

            on_scan_prog("audio", 10, f"Downloading stream audio ({info.get('streamer')})...")
            key = stream.scan(url, on_progress=on_scan_prog)

            if key in CANCEL_JOBS:
                return

            on_scan_prog("curate", 85, "Curating top viral moments and generating kinetic hooks...")
            _run_ai_curation(key)

            if key in CANCEL_JOBS:
                return

            on_scan_prog("render", 90, "Rendering finished 9:16 vertical clips on RTX 4050 GPU NVENC...")
            _run_render_stream(key)

            on_scan_prog("done", 100, "All vertical clips rendered and ready in library!")
            ACTIVE_JOBS.pop(url, None)
    except Exception as e:
        ACTIVE_JOBS[url] = {"action": "scan", "error": str(e), "failed": True}


PAUSED_PIPELINE: bool = False
PAUSED_JOBS_CHECKPOINTS: dict[str, dict] = {}


@app.post("/api/pipeline/pause")
def pause_all_pipelines():
    """
    Globally pauses all active AI/GPU pipelines, speech transcription, rendering, and watcher daemons.
    Frees all GPU resources immediately while preserving all progress.
    """
    global PAUSED_PIPELINE
    PAUSED_PIPELINE = True

    try:
        from studio import watched_creators
        watched_creators.stop_watcher_daemon()
    except Exception:
        pass

    paused_streams = []
    for key, job in list(ACTIVE_JOBS.items()):
        CANCEL_JOBS.add(key)
        d = stream_dir(key)
        if d.exists():
            prog_file = d / "progress.json"
            current_prog = read_json(prog_file, {}) if prog_file.exists() else job
            paused_state = {
                "action": "paused",
                "paused_stage": current_prog.get("action", "processing"),
                "progress": "⏸️ Pipeline paused by user · GPU VRAM & NVENC freed · Progress saved",
                "pct": current_prog.get("pct", 0),
                "clip_current": current_prog.get("clip_current", 1),
                "clip_total": current_prog.get("clip_total", 6),
                "paused": True,
                "updated_at": time.time(),
                "done": False
            }
            PAUSED_JOBS_CHECKPOINTS[key] = paused_state
            write_json(prog_file, paused_state)
            ACTIVE_JOBS[key] = paused_state
            paused_streams.append(key)

    for s_dir in (WORK_DIR / "streams").glob("*"):
        prog_f = s_dir / "progress.json"
        if prog_f.exists():
            cur = read_json(prog_f, {})
            if not cur.get("done") and cur.get("action") not in ("done", "stopped"):
                paused_state = {
                    "action": "paused",
                    "paused_stage": cur.get("action", "processing"),
                    "progress": "⏸️ Pipeline paused by user · GPU VRAM & NVENC freed · Progress saved",
                    "pct": cur.get("pct", 0),
                    "clip_current": cur.get("clip_current", 1),
                    "clip_total": cur.get("clip_total", 6),
                    "paused": True,
                    "updated_at": time.time(),
                    "done": False
                }
                PAUSED_JOBS_CHECKPOINTS[s_dir.name] = paused_state
                write_json(prog_f, paused_state)
                ACTIVE_JOBS[s_dir.name] = paused_state
                if s_dir.name not in paused_streams:
                    paused_streams.append(s_dir.name)

    for lk, sess in list(ACTIVE_LIVE_SESSIONS.items()):
        try:
            if hasattr(sess, "stop"):
                sess.stop("Paused by user")
        except Exception:
            pass

    try:
        subprocess.run(["taskkill", "/F", "/IM", "ffmpeg.exe"], capture_output=True, check=False)
    except Exception:
        pass

    return {
        "ok": True,
        "paused": True,
        "message": "All GPU pipelines paused! NVENC and GPU memory freed. All progress is saved.",
        "paused_streams": paused_streams
    }


@app.post("/api/pipeline/resume")
def resume_all_pipelines(background_tasks: BackgroundTasks):
    """
    Resumes paused pipelines from their exact saved checkpoint without starting over.
    """
    global PAUSED_PIPELINE
    PAUSED_PIPELINE = False

    try:
        from studio import watched_creators
        watched_creators.start_watcher_daemon()
    except Exception:
        pass

    resumed_streams = []
    for s_dir in (WORK_DIR / "streams").glob("*"):
        prog_f = s_dir / "progress.json"
        if prog_f.exists():
            cur = read_json(prog_f, {})
            if cur.get("paused") or cur.get("action") == "paused":
                key = s_dir.name
                CANCEL_JOBS.discard(key)
                paused_stage = cur.get("paused_stage", "render")

                info = read_json(s_dir / "info.json", {})
                url = info.get("url")

                if paused_stage in ("render", "curate") or (s_dir / "final.json").exists() or (s_dir / "clips").exists():
                    cur["action"] = "render"
                    cur["progress"] = "Resuming clip rendering on RTX 4050 GPU NVENC..."
                    cur["paused"] = False
                    cur["done"] = False
                    write_json(prog_f, cur)
                    ACTIVE_JOBS[key] = cur
                    background_tasks.add_task(_run_render_stream, key)
                    resumed_streams.append(key)
                elif url:
                    cur["action"] = "scan"
                    cur["progress"] = "Resuming stream processing..."
                    cur["paused"] = False
                    cur["done"] = False
                    write_json(prog_f, cur)
                    ACTIVE_JOBS[key] = cur
                    background_tasks.add_task(_run_scan_thread, url)
                    resumed_streams.append(key)

    return {
        "ok": True,
        "paused": False,
        "message": f"Resumed {len(resumed_streams)} pipeline(s) from saved checkpoint!",
        "resumed_streams": resumed_streams
    }


@app.get("/api/pipeline/status")
def get_pipeline_status():
    paused_count = 0
    active_count = 0
    for s_dir in (WORK_DIR / "streams").glob("*"):
        prog_f = s_dir / "progress.json"
        if prog_f.exists():
            cur = read_json(prog_f, {})
            if cur.get("paused") or cur.get("action") == "paused":
                paused_count += 1
            elif not cur.get("done") and cur.get("action") and cur.get("action") not in ("done", "stopped"):
                active_count += 1
    return {
        "paused": PAUSED_PIPELINE or paused_count > 0,
        "paused_count": paused_count,
        "active_count": active_count
    }


@app.post("/api/streams/{key}/pause")
def pause_stream_processing(key: str):
    CANCEL_JOBS.add(key)
    d = stream_dir(key)
    prog_p = d / "progress.json"
    cur = read_json(prog_p, {}) if prog_p.exists() else {}
    paused_state = {
        "action": "paused",
        "paused_stage": cur.get("action", "render"),
        "progress": "⏸️ Pipeline paused by user · GPU VRAM & NVENC freed · Progress saved",
        "pct": cur.get("pct", 50),
        "clip_current": cur.get("clip_current", 1),
        "clip_total": cur.get("clip_total", 6),
        "paused": True,
        "updated_at": time.time(),
        "done": False
    }
    ACTIVE_JOBS[key] = paused_state
    if d.exists():
        write_json(prog_p, paused_state)
    try:
        subprocess.run(["taskkill", "/F", "/IM", "ffmpeg.exe"], capture_output=True, check=False)
    except Exception:
        pass
    return {"ok": True, "message": "Stream pipeline paused! GPU resources freed and progress saved."}


@app.post("/api/streams/{key}/stop")
def stop_stream_processing(key: str):
    live_sess = ACTIVE_LIVE_SESSIONS.pop(key, None)
    if live_sess and hasattr(live_sess, "stop"):
        try:
            live_sess.stop("Stopped by user")
        except Exception:
            pass

    CANCEL_JOBS.add(key)
    ACTIVE_JOBS.pop(key, None)

    d = stream_dir(key)
    if d.exists():
        prog_p = d / "progress.json"
        if prog_p.exists():
            try:
                prog_p.unlink()
            except OSError:
                write_json(prog_p, {"action": "stopped", "progress": "Stopped by user", "done": True})

    return {"ok": True, "message": "Pipeline stopped and cleared successfully."}


@app.post("/api/streams/{key}/resume")
def resume_stream_processing(key: str, background_tasks: BackgroundTasks):
    d = stream_dir(key)
    if not d.exists():
        raise HTTPException(status_code=404, detail="Stream not found")
    CANCEL_JOBS.discard(key)
    prog_p = d / "progress.json"
    cur = read_json(prog_p, {}) if prog_p.exists() else {}
    paused_stage = cur.get("paused_stage") or cur.get("action", "render")

    info = read_json(d / "info.json", {})
    url = info.get("url")

    if paused_stage in ("render", "curate") or (d / "final.json").exists() or (d / "clips").exists():
        cur["action"] = "render"
        cur["progress"] = "Resuming clip rendering on RTX 4050 GPU NVENC..."
        cur["paused"] = False
        cur["done"] = False
        write_json(prog_p, cur)
        ACTIVE_JOBS[key] = cur
        background_tasks.add_task(_run_render_stream, key)
        return {"ok": True, "message": "Resumed clip rendering from saved checkpoint (skipping completed clips)!"}
    elif url:
        cur["action"] = "scan"
        cur["progress"] = "Resuming stream processing..."
        cur["paused"] = False
        cur["done"] = False
        write_json(prog_p, cur)
        ACTIVE_JOBS[key] = cur
        background_tasks.add_task(_run_scan_thread, url)
        return {"ok": True, "message": "Stream pipeline resumed from saved checkpoint!"}
    raise HTTPException(status_code=400, detail="Cannot resume stream")


@app.post("/api/scan")
def scan_url(payload: dict, background_tasks: BackgroundTasks):
    url = (payload.get("url") or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL required")
    force_live = bool(payload.get("force_live", False))
    background_tasks.add_task(_run_scan_thread, url, force_live)
    return {"ok": True, "message": "Stream monitor started in background", "url": url}


# --- Watched Creators Auto-Clipper APIs ---
from studio import watched_creators

@app.get("/api/creators")
def api_list_creators():
    return watched_creators.list_creators()

@app.post("/api/creators")
def api_add_creator(payload: dict = Body(...)):
    url = (payload.get("url") or payload.get("channel_url") or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="Channel URL is required")
    auto_clip = bool(payload.get("auto_clip", True))
    try:
        c = watched_creators.add_creator(url, auto_clip=auto_clip)
        return {"ok": True, "creator": c}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/api/creators/{cid}")
def api_remove_creator(cid: str):
    ok = watched_creators.remove_creator(cid)
    if not ok:
        raise HTTPException(status_code=404, detail="Creator not found")
    return {"ok": True, "deleted": cid}

@app.post("/api/creators/{cid}/toggle")
def api_toggle_creator(cid: str, payload: dict = Body({})):
    val = payload.get("enabled")
    c = watched_creators.toggle_auto_clip(cid, val)
    if not c:
        raise HTTPException(status_code=404, detail="Creator not found")
    return {"ok": True, "creator": c}

@app.post("/api/creators/{cid}/check-now")
def api_check_creator(cid: str):
    c = watched_creators.check_creator_now(cid)
    if not c:
        raise HTTPException(status_code=404, detail="Creator not found")
    return {"ok": True, "creator": c}


# --- Social Posting Integration (YouTube Shorts & Instagram Reels) ---
@app.get("/api/post/status")
def get_posting_status():
    ctx.set_uid("u_local")
    accs = accounts.list_accounts()
    has_yt = any(a.get("platform") == "youtube" for a in accs)
    has_ig = any(a.get("platform") == "instagram" for a in accs)
    apps = accounts.apps_public()
    autopost = get_autopost_settings()
    return {
        "youtube": {"configured": apps.get("youtube", {}).get("ready", False), "connected": has_yt},
        "instagram": {"configured": apps.get("instagram", {}).get("ready", False), "connected": has_ig},
        "accounts": accs,
        "apps": apps,
        "autopost": autopost,
        "redirect_base": config.PUBLIC_URL
    }


@app.get("/api/accounts")
def get_accounts_api():
    ctx.set_uid("u_local")
    return {
        "ok": True,
        "accounts": accounts.list_accounts(),
        "apps": accounts.apps_public(),
        "redirect_base": config.PUBLIC_URL
    }


@app.post("/api/accounts/save-app")
def api_save_app(payload: dict):
    ctx.set_uid("u_local")
    platform = payload.get("platform")
    client_id = (payload.get("client_id") or "").strip()
    client_secret = (payload.get("client_secret") or "").strip()
    if not platform:
        raise HTTPException(status_code=400, detail="Platform is required")
    try:
        apps = accounts.save_app(platform, client_id, client_secret)
        return {"ok": True, "apps": apps}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/accounts/upload-secrets")
async def api_upload_secrets(file: UploadFile = File(...)):
    ctx.set_uid("u_local")
    try:
        content = await file.read()
        data = json.loads(content.decode("utf-8"))
        app_data = data.get("web") or data.get("installed")
        if not app_data:
            raise ValueError("Invalid client_secrets.json: missing 'web' or 'installed' object")
        client_id = app_data.get("client_id", "").strip()
        client_secret = app_data.get("client_secret", "").strip()
        if not client_id or not client_secret:
            raise ValueError("client_secrets.json must contain both client_id and client_secret")
        apps = accounts.save_app("youtube", client_id, client_secret)
        return {"ok": True, "client_id": client_id, "apps": apps}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process client_secrets.json: {e}")


@app.post("/api/accounts/direct-token")
def api_direct_token(payload: dict):
    ctx.set_uid("u_local")
    platform = payload.get("platform")
    token = (payload.get("token") or "").strip()
    if not token:
        raise HTTPException(status_code=400, detail="Token cannot be empty")
    if platform == "instagram":
        try:
            acc = accounts.instagram_add(token)
            return {"ok": True, "account": acc}
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Instagram connection failed: {e}")
    else:
        raise HTTPException(status_code=400, detail=f"Direct token connection not supported for {platform}")


@app.delete("/api/accounts/{aid}")
def api_remove_account(aid: str):
    ctx.set_uid("u_local")
    accounts.remove(aid)
    return {"ok": True, "removed": aid}


@app.get("/oauth/{platform}/start")
def connect_start(platform: str):
    ctx.set_uid("u_local")
    if platform not in accounts.PLATFORM_NAMES:
        raise HTTPException(status_code=404, detail="Unknown platform")
    try:
        url = accounts.connect_url(platform)
        return RedirectResponse(url, status_code=303)
    except Exception as e:
        return RedirectResponse(f"/?connect_error={quote(str(e))}#social", status_code=303)


@app.get("/oauth/{platform}/callback")
def connect_callback(platform: str, code: str = "", state: str = "", error: str = "", error_description: str = "", error_reason: str = ""):
    ctx.set_uid("u_local")
    if platform not in accounts.PLATFORM_NAMES:
        raise HTTPException(status_code=404, detail="Unknown platform")
    name = accounts.PLATFORM_NAMES[platform]
    if error or not code:
        why = "cancelled by user" if error in ("access_denied",) or error_reason == "user_denied" else (error_description or error or "no authorization code returned")
        return RedirectResponse(f"/?connect_error={quote(f'{name} was not connected: {why[:160]}')}#social", status_code=303)
    try:
        acc = accounts.connect_finish(platform, code, state)
        return RedirectResponse(f"/?connected={platform}&acc_name={quote(acc.get('name', ''))}#social", status_code=303)
    except Exception as e:
        return RedirectResponse(f"/?connect_error={quote(f'{name} was not connected: {str(e)[:160]}')}#social", status_code=303)


@app.get("/pub/{token}.mp4")
def public_clip_fetch(token: str):
    ctx.set_uid("u_local")
    got = accounts.read_clip_link(token)
    if not got:
        raise HTTPException(status_code=404, detail="Clip link expired or invalid")
    uid, job_id, name = got
    p = find_file_in_work(name)
    if not p or not p.is_file():
        raise HTTPException(status_code=404, detail="Clip file not found")
    return FileResponse(p, media_type="video/mp4", headers={"Cache-Control": "private, no-store"})


@app.get("/api/autopost/settings")
def api_get_autopost():
    return {"ok": True, "settings": get_autopost_settings()}


@app.post("/api/autopost/settings")
def api_save_autopost(payload: dict):
    res = save_autopost_settings(payload)
    return {"ok": True, "settings": res}


@app.post("/api/post/clip")
def post_clip_action(payload: dict):
    ctx.set_uid("u_local")
    platform = payload.get("platform")
    filename = payload.get("filename")
    if not platform or not filename:
        raise HTTPException(status_code=400, detail="Platform and filename are required.")

    p = find_file_in_work(filename)
    if not p or not p.is_file():
        raise HTTPException(status_code=404, detail=f"Clip file '{filename}' not found on disk.")

    all_accs = [a for a in accounts._all() if a.get("platform") == platform]
    if not all_accs:
        plat_label = accounts.PLATFORM_NAMES.get(platform, platform)
        raise HTTPException(
            status_code=400,
            detail=f"No connected {plat_label} account found. Please connect your account in the Social tab first."
        )

    acc = all_accs[0]
    meta = get_clip_metadata_for_file(filename)

    title = payload.get("title", "").strip() or meta.get("hook") or meta.get("title") or filename
    caption = payload.get("caption", "").strip() or meta.get("caption") or ""
    tags = meta.get("hashtags") or ["Shorts", "Gaming", "Viral"]
    if isinstance(tags, str):
        tags = [t.strip().lstrip("#") for t in tags.split() if t.strip()]
    else:
        tags = [str(t).strip().lstrip("#") for t in tags]
    if "Shorts" not in tags:
        tags.append("Shorts")

    if platform == "youtube":
        desc = f"{caption}\n\n{' '.join('#' + t for t in tags)}\n\n#Shorts\n\nClipped autonomously with PeakCut ClipFarm"
        privacy = payload.get("privacy", "public")
        try:
            vid_id, shorts_url = accounts.youtube_upload(acc, p, title=title[:100], description=desc[:4900], tags=tags[:15], privacy=privacy)
            record_post_history("youtube", filename, shorts_url, vid_id, title)
            # update final.json if available
            for s_dir in (WORK_DIR / "streams").glob("*"):
                final_f = s_dir / "final.json"
                if final_f.exists():
                    f_data = read_json(final_f, {})
                    updated = False
                    for c in f_data.get("clips", []):
                        if c.get("file") == filename or match_clip_id(c.get("id"), filename.split("_")[0]):
                            c["youtube_url"] = shorts_url
                            updated = True
                    if updated:
                        write_json(final_f, f_data)
            return {
                "ok": True,
                "platform": "youtube",
                "url": shorts_url,
                "id": vid_id,
                "message": f"Successfully posted to YouTube Shorts: {shorts_url}"
            }
        except Exception as e:
            record_post_history("youtube", filename, "", "", title, error=str(e))
            raise HTTPException(status_code=500, detail=f"YouTube upload failed: {str(e)}")

    elif platform == "instagram":
        full_caption = f"{title}\n\n{caption}\n\n{' '.join('#' + t for t in tags)}"
        try:
            media_id, reel_url = accounts.instagram_upload(acc, p, caption=full_caption[:2200])
            record_post_history("instagram", filename, reel_url, media_id, title)
            # update final.json if available
            for s_dir in (WORK_DIR / "streams").glob("*"):
                final_f = s_dir / "final.json"
                if final_f.exists():
                    f_data = read_json(final_f, {})
                    updated = False
                    for c in f_data.get("clips", []):
                        if c.get("file") == filename or match_clip_id(c.get("id"), filename.split("_")[0]):
                            c["instagram_url"] = reel_url
                            updated = True
                    if updated:
                        write_json(final_f, f_data)
            return {
                "ok": True,
                "platform": "instagram",
                "url": reel_url,
                "id": media_id,
                "message": f"Successfully posted to Instagram Reels: {reel_url or 'Reel published'}"
            }
        except Exception as e:
            record_post_history("instagram", filename, "", "", title, error=str(e))
            raise HTTPException(status_code=500, detail=f"Instagram upload failed: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported platform: {platform}")


@app.get("/", response_class=HTMLResponse)
def index_page():
    ui_path = WEB_DIR / "clipfarm.html"
    if ui_path.exists():
        return HTMLResponse(ui_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>ClipFarm UI not found</h1>")


def main():
    watched_creators.start_watcher_daemon()
    port = int(os.environ.get("CLIPFARM_PORT", 8767))
    print(f"\n  ClipFarm Studio is running: http://127.0.0.1:{port}\n")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
