"""CarryMinati live stream recorder and auto-clipper using streamlink and Sherpa-ONNX."""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ClipFarm" / "scripts"))

from studio import transcribe as st_transcribe
import numpy as np

def get_carry_hls_url():
    streamlink_exe = str(ROOT / ".venv" / "Scripts" / "streamlink.exe")
    cmd = [streamlink_exe, "--stream-url", "https://www.youtube.com/@CarryisLive/live", "720p,best"]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    url = res.stdout.strip()
    if url.startswith("http"):
        return url
    return None

def start_carry_recorder():
    s_dir = ROOT / "ClipFarm" / "work" / "streams" / "youtube-carryminati"
    buf_dir = s_dir / "buffer"
    buf_dir.mkdir(parents=True, exist_ok=True)
    (s_dir / "clips").mkdir(parents=True, exist_ok=True)

    info = {
        "id": "carryminati",
        "title": "CARRYMINATI PLAYS GTA VI",
        "streamer": "CarryMinati",
        "channel": "CarryisLive",
        "uploader": "CarryisLive",
        "platform": "youtube",
        "url": "https://www.youtube.com/@CarryisLive",
        "is_live": True
    }
    (s_dir / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    hls_url = get_carry_hls_url()
    if not hls_url:
        print("[CarryMinati] Streamlink could not get live HLS URL right now.")
        return

    print(f"[CarryMinati] Resolved live HLS URL. Starting ffmpeg recorder...")
    csv_file = "p01.csv"
    ts_pattern = "p01_%05d.ts"
    
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

    proc = subprocess.Popen(
        ffmpeg_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(buf_dir)
    )

    print("[CarryMinati] Ingest active in background. PID:", proc.pid)
    # Read audio chunks and transcribe
    all_words = []
    chunk_size = 16000 * 2 * 12 # 12 seconds
    base_sec = 0.0

    while proc.poll() is None:
        raw_audio = proc.stdout.read(chunk_size)
        if not raw_audio:
            break
        samples = len(raw_audio) // 2
        dur_chunk = samples / 16000.0
        audio_np = np.frombuffer(raw_audio, dtype=np.int16)
        try:
            words = st_transcribe.transcribe(audio_np)
            for w in words:
                w["s"] = round(w["s"] + base_sec, 2)
                w["e"] = round(w["e"] + base_sec, 2)
            all_words.extend(words)
            (s_dir / "words.json").write_text(json.dumps(all_words, indent=2), encoding="utf-8")
        except Exception as e:
            pass
        base_sec += dur_chunk

if __name__ == "__main__":
    start_carry_recorder()
