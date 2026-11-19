"""Kai Cenat live Twitch stream recorder and auto-clipper using streamlink and Sherpa-ONNX."""
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

def get_kai_hls_url():
    streamlink_exe = str(ROOT / ".venv" / "Scripts" / "streamlink.exe")
    cmd = [streamlink_exe, "--stream-url", "https://www.twitch.tv/kaicenat", "720p,best"]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    url = res.stdout.strip()
    if url.startswith("http"):
        return url
    return None

def start_kai_recorder(max_seconds=360):
    s_dir = ROOT / "ClipFarm" / "work" / "streams" / "twitch-kaicenat"
    buf_dir = s_dir / "buffer"
    buf_dir.mkdir(parents=True, exist_ok=True)
    (s_dir / "clips").mkdir(parents=True, exist_ok=True)

    info = {
        "id": "twitch-kaicenat",
        "title": "🌴PLAYING GTA 6 FOR THE FIRST TIME🌴",
        "streamer": "Kai Cenat",
        "channel": "kaicenat",
        "uploader": "Kai Cenat",
        "platform": "twitch",
        "url": "https://www.twitch.tv/kaicenat",
        "category": "Grand Theft Auto VI",
        "is_live": True
    }
    (s_dir / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    hls_url = get_kai_hls_url()
    if not hls_url:
        print("[Kai Cenat] Streamlink could not get live HLS URL right now.")
        return

    print(f"[Kai Cenat] Resolved live HLS URL. Starting ffmpeg recorder...")
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

    proc = subprocess.Popen(
        ffmpeg_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(buf_dir)
    )

    print(f"[Kai Cenat] Ingest active in background. PID: {proc.pid}. Recording for ~{max_seconds}s...")
    all_words = []
    peaks = []
    chunk_size = 16000 * 2 * 6 # 6 seconds chunks
    base_sec = 0.0
    t_start = time.time()

    try:
        while proc.poll() is None:
            if time.time() - t_start > max_seconds:
                print(f"[Kai Cenat] Reached target recording duration ({max_seconds}s). Stopping ingest.")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except:
                    proc.kill()
                break

            raw_audio = proc.stdout.read(chunk_size)
            if not raw_audio:
                break
            samples = len(raw_audio) // 2
            dur_chunk = samples / 16000.0
            audio_np = np.frombuffer(raw_audio, dtype=np.int16)

            # Compute loudness / energy peaks
            slice_samples = int(16000 * 0.25)
            for i in range(0, samples, slice_samples):
                sl = audio_np[i : i + slice_samples]
                if len(sl) > 0:
                    rms = float(np.sqrt(np.mean(sl.astype(np.float32) ** 2)))
                    t_slice = round(base_sec + (i / 16000.0), 2)
                    if rms > 3000:
                        peaks.append({"t": t_slice, "rms": round(rms, 1)})

            # Transcribe
            try:
                words = st_transcribe.transcribe(audio_np)
                for w in words:
                    w["s"] = round(w["s"] + base_sec, 2)
                    w["e"] = round(w["e"] + base_sec, 2)
                all_words.extend(words)
                
                (s_dir / "words.json").write_text(json.dumps(all_words, indent=2), encoding="utf-8")
                (s_dir / "peaks.json").write_text(json.dumps(peaks, indent=2), encoding="utf-8")
                
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
                (s_dir / "lines.json").write_text(json.dumps(lines, indent=2), encoding="utf-8")
            except Exception as e:
                print(f"[Kai Cenat] Transcribe error: {e}")

            base_sec += dur_chunk

    except KeyboardInterrupt:
        print("\n[Kai Cenat] Stopping recorder...")
        proc.terminate()

    print(f"\n[Kai Cenat] Finished recording. Total elapsed: {base_sec:.1f}s, total words: {len(all_words)}")

if __name__ == "__main__":
    dur = int(sys.argv[1]) if len(sys.argv) > 1 else 360
    start_kai_recorder(max_seconds=dur)
