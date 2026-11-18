"""Continuous Catch-up & Real-Time Live Stream Clipper Engine for ClipFarm.
Implements the PeakCut segmented rolling buffer architecture:
- Slices incoming broadcast into 6-second MPEG-TS segments (zero file locks).
- Simultaneously streams 16kHz mono audio into a continuous memory pipe.
- Catch-up transcription worker processes speech at maximum speed from t=0.
- Rolling excitement / loudness peak detector flags viral moments on-the-fly.
- Asynchronous RTX 4050 NVENC GPU render worker cuts and finishes 9:16 vertical clips
  with kinetic typography and safe-zone layouts while the stream is actively broadcasting.
"""
from __future__ import annotations

import collections
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Ensure ClipFarm/scripts is in sys.path
_scripts_dir = Path(__file__).resolve().parent.parent / "ClipFarm" / "scripts"
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

from clipfarm import face, listen, render, shortlist, signals, util
from clipfarm.util import hms, read_json, settings, stream_dir, write_json

SEG_SEC = 6
SR = 16000
PAD = 4


def _clean_slug(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return s[:40] or "clip"


class LivePipelineSession:
    def __init__(self, key: str, url: str, info: dict, auto_clip: bool = True):
        self.key = key
        self.url = url
        self.info = info
        self.auto_clip = auto_clip
        self.streamer = info.get("streamer") or info.get("uploader") or "Live Streamer"
        self.title = info.get("title") or "Live Stream"
        self.platform = info.get("platform") or "web"

        self.dir = stream_dir(key)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.buf = self.dir / "buffer"
        self.buf.mkdir(parents=True, exist_ok=True)
        self.clips_dir = self.dir / "clips"
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.sections_dir = self.dir / "sections"
        self.sections_dir.mkdir(parents=True, exist_ok=True)

        write_json(self.dir / "info.json", self.info)

        self.lock = threading.RLock()
        self.stop_flag = threading.Event()
        self.user_stopped = False
        self.ended = False
        self.status = "starting"  # starting | catching_up | live_monitoring | stopped | finished

        # Audio tracking
        self.samples = 0
        self.audio_bytes = 0
        self.pending_audio = bytearray()
        self.pending_base_sample = 0
        self.db: list[float] = []
        self.t_transcribed = 0.0
        self.words: list[dict] = []

        # Segment tracking
        self.segments: list[tuple[Path, float, float]] = []
        self.seg_clock = 0.0
        self.part = 0
        self.csv_seen: dict[int, int] = {}

        # Clipping tracking
        self.clips_found: list[dict] = []
        self.clips_editing: list[dict] = []
        self.clips_ready: list[dict] = []
        self.render_queue: collections.deque[dict] = collections.deque()
        self.last_clip_end = 0.0

        # Timers & metrics
        self.started_at = time.time()
        self.ingest_proc: Optional[subprocess.Popen] = None
        self.ffmpeg_proc: Optional[subprocess.Popen] = None
        self.last_speed_calc = time.time()
        self.last_transcribed_val = 0.0
        self.current_speed = "1.0x"

        # Threads
        self.threads: list[threading.Thread] = []

    def start(self):
        t_ingest = threading.Thread(target=self._ingest_loop, name=f"LiveIngest-{self.key}", daemon=True)
        t_trans = threading.Thread(target=self._transcribe_worker, name=f"LiveTrans-{self.key}", daemon=True)
        t_peaks = threading.Thread(target=self._peak_detector_worker, name=f"LivePeaks-{self.key}", daemon=True)
        t_rend = threading.Thread(target=self._render_worker, name=f"LiveRender-{self.key}", daemon=True)

        self.threads = [t_ingest, t_trans, t_peaks, t_rend]
        for t in self.threads:
            t.start()

    def stop(self, note: str = "Stopped by user"):
        self.user_stopped = True
        self.stop_flag.set()
        self.status = "stopped"
        if self.ffmpeg_proc and self.ffmpeg_proc.poll() is None:
            try:
                self.ffmpeg_proc.terminate()
            except Exception:
                pass
        if self.ingest_proc and self.ingest_proc.poll() is None:
            try:
                self.ingest_proc.terminate()
            except Exception:
                pass

    def _scan_segments(self):
        with self.lock:
            for i in range(1, self.part + 1):
                csv_file = self.buf / f"p{i:02d}.csv"
                if not csv_file.exists():
                    continue
                try:
                    lines = csv_file.read_text(encoding="utf-8", errors="ignore").splitlines()
                    seen = self.csv_seen.get(i, 0)
                    for row in lines[seen:]:
                        parts = [p.strip() for p in row.split(",") if p.strip()]
                        if len(parts) >= 3:
                            name = parts[0]
                            try:
                                a, b = float(parts[1]), float(parts[2])
                                dur = max(0.0, b - a)
                                ts_path = self.buf / name
                                self.segments.append((ts_path, self.seg_clock, self.seg_clock + dur))
                                self.seg_clock += dur
                            except ValueError:
                                continue
                    self.csv_seen[i] = len(lines)
                except Exception:
                    pass

    def _ingest_loop(self):
        """Streams live video directly via ffmpeg segmenter using resolved stream URLs."""
        from studio import live as st_live
        fails = 0
        errlog_path = self.dir / "ffmpeg_ingest.log"

        while not self.stop_flag.is_set():
            try:
                try:
                    info = st_live.resolve(self.url)
                    urls = info.get("urls", [])
                    if info.get("title") and not self.title:
                        self.title = info["title"]
                    if info.get("uploader") and not self.streamer:
                        self.streamer = info["uploader"]
                except Exception:
                    urls = [self.url]

                if not urls:
                    fails += 1
                    if fails >= 5:
                        break
                    time.sleep(3)
                    continue

                self.part += 1
                with self.lock:
                    self.seg_clock = max(self.seg_clock, self.samples / SR)
                p = f"p{self.part:02d}"

                ffmpeg_cmd = [
                    "ffmpeg", "-hide_banner", "-loglevel", "warning",
                    "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "10",
                ]
                for u in urls:
                    ffmpeg_cmd += ["-i", u]

                a_in = "1:a:0" if len(urls) > 1 else "0:a:0"
                ffmpeg_cmd += [
                    "-map", "0:v:0", "-map", a_in, "-c", "copy",
                    "-f", "segment", "-segment_time", str(SEG_SEC),
                    "-reset_timestamps", "1", "-segment_format", "mpegts",
                    "-segment_list", f"{p}.csv", "-segment_list_type", "csv",
                    f"{p}_%05d.ts",
                    "-map", a_in, "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "pipe:1"
                ]

                with open(errlog_path, "ab") as errlog:
                    self.ffmpeg_proc = subprocess.Popen(
                        ffmpeg_cmd,
                        stdout=subprocess.PIPE,
                        stderr=errlog,
                        cwd=str(self.buf)
                    )

                    chunk_size = SR * 2  # 1 second of 16kHz mono audio
                    while not self.stop_flag.is_set():
                        data = self.ffmpeg_proc.stdout.read(chunk_size)
                        if not data:
                            break
                        data = data[: len(data) // 2 * 2]
                        x = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                        level = float(20 * np.log10(np.sqrt((x ** 2).mean() + 1e-10))) if len(x) else -100.0

                        with self.lock:
                            self.pending_audio += data
                            self.audio_bytes += len(data)
                            self.samples += len(data) // 2
                            self.db.append(level)

                        self._scan_segments()

                    self.ffmpeg_proc.wait()

                fails = 0
            except Exception as e:
                fails += 1
                print(f"[LiveIngest] Exception: {e}", flush=True)
                if fails >= 5:
                    break
                time.sleep(3)

        self.ended = True
        if not self.user_stopped:
            self.status = "finished"

    def _transcribe_worker(self):
        """Transcribes incoming audio in batches, catching up as fast as possible."""
        from studio import transcribe as st_transcribe

        while not self.stop_flag.is_set():
            with self.lock:
                avail_bytes = len(self.pending_audio)
                ended = self.ended

            if ended and avail_bytes == 0:
                break

            min_bytes = SR * 2 * 6  # at least 6 seconds
            if avail_bytes < min_bytes and not ended:
                time.sleep(1)
                continue

            # Batch dynamically: if we have a lot of audio pending (catching up), take up to 24s at once
            take_sec = 24.0 if avail_bytes > SR * 2 * 30 else 12.0
            take_samples = min(int(take_sec * SR), avail_bytes // 2)

            with self.lock:
                raw_chunk = bytes(self.pending_audio[: take_samples * 2])
                base_sample = self.pending_base_sample
                base_sec = base_sample / SR

            audio_np = np.frombuffer(raw_chunk, dtype=np.int16)

            # Transcribe with Sherpa-ONNX model
            words_chunk = []
            try:
                words_chunk = st_transcribe.transcribe(audio_np)
            except Exception as e:
                print(f"[LiveTranscribe] Error: {e}", flush=True)

            for w in words_chunk:
                w["s"] = round(w["s"] + base_sec, 2)
                w["e"] = round(w["e"] + base_sec, 2)

            with self.lock:
                del self.pending_audio[: take_samples * 2]
                self.pending_base_sample += take_samples
                self.t_transcribed = self.pending_base_sample / SR
                self.words += words_chunk

                downloaded_sec = self.samples / SR
                # Calculate speed & status
                now = time.time()
                dt = now - self.last_speed_calc
                if dt >= 3.0:
                    delta_trans = self.t_transcribed - self.last_transcribed_val
                    spd = (delta_trans / dt) if dt > 0 else 1.0
                    self.current_speed = f"{spd:.1f}x"
                    self.last_transcribed_val = self.t_transcribed
                    self.last_speed_calc = now

                if downloaded_sec - self.t_transcribed < 15.0:
                    self.status = "live_monitoring"
                else:
                    self.status = "catching_up"

            # Persist words.json
            write_json(self.dir / "words.json", self.words)
            time.sleep(0.1)

    def _peak_detector_worker(self):
        """Monitors loudness and transcript cadence to identify viral clips in rolling windows."""
        while not self.stop_flag.is_set():
            time.sleep(5)
            with self.lock:
                t_trans = self.t_transcribed
                n_db = len(self.db)
                last_end = self.last_clip_end

            # Only look for clips if we have at least 45 seconds of transcribed audio beyond last clip
            if t_trans - last_end < 45.0:
                continue

            # Check loudness spikes in the window [last_end, t_trans]
            ws = max(0, int(last_end))
            we = min(n_db, int(t_trans))
            if we - ws < 20:
                continue

            sub_db = np.array(self.db[ws:we], dtype=np.float32)
            base_loudness = np.median(sub_db)
            peak_idx = int(np.argmax(sub_db))
            peak_db = float(sub_db[peak_idx])
            peak_t = ws + peak_idx

            # Spike condition: peak is at least 6 dB above median (roughly 2x louder)
            jump_db = peak_db - base_loudness
            if jump_db < 5.0 and (we - ws < 120.0):
                # Wait until either a jump occurs or 2 minutes elapse without a clip
                continue

            # Cap at strictly 10 reels per stream as instructed
            with self.lock:
                if len(self.clips_ready) + len(self.render_queue) >= 10:
                    time.sleep(15)
                    continue

            # Formulate clip boundaries: 22-38s clip centered around peak with natural narrative arc
            clip_start = max(last_end, peak_t - 18.0)
            clip_end = min(t_trans, clip_start + 32.0)

            if clip_end - clip_start < 18.0:
                continue

            # Extract words for this clip
            with self.lock:
                clip_words = [w for w in self.words if clip_start <= w["s"] <= clip_end]

            if len(clip_words) < 8:
                # Insufficient speech/action, skip forward
                self.last_clip_end = clip_end
                continue

            # Full dialogue text for contextual hook and story caption generation
            dialogue_text = " ".join(w["w"] for w in clip_words).strip()
            pkg = self._generate_hook(dialogue_text, jump_db)
            hook = pkg["hook"]
            caption = pkg["caption"]
            hashtags = pkg["hashtags"]

            cid = f"clip_{len(self.clips_found) + 1:02d}"
            clip_meta = {
                "id": cid,
                "start": round(clip_start, 2),
                "end": round(clip_end, 2),
                "duration": round(clip_end - clip_start, 2),
                "title": hook,
                "hook": hook,
                "caption": caption,
                "hashtags": hashtags,
                "curated": True,
                "saved": True,
                "score": min(99, 90 + int(min(9, jump_db))),
                "reason": f"Excitement spike (+{round(jump_db)} dB loud) during live broadcast",
                "status": "queued",
                "progress": 0,
                "found_at": time.time(),
                "words_count": len(clip_words)
            }

            with self.lock:
                self.clips_found.append(clip_meta)
                self.clips_editing.append(clip_meta)
                self.render_queue.append(clip_meta)
                self.last_clip_end = clip_end

            # Update final.json so frontend can see clips immediately
            self._save_final_json()

    def _generate_hook(self, speech_sample: str, jump_db: float) -> dict:
        """Generates high-converting, viral packaging. Never returns raw spoken phrases."""
        from studio import clipfarm_server
        streamer_name = self.streamer or "Streamer"
        return clipfarm_server.generate_ai_clip_hook_and_caption(
            streamer=streamer_name,
            stream_title=self.title or "GTA 6 Live Stream",
            dialogue=speech_sample,
            why_signals=f"Excitement jump +{round(jump_db)} dB loud"
        )

    def _render_worker(self):
        """Asynchronously renders vertical 9:16 clips on RTX 4050 GPU using NVENC."""
        while not self.stop_flag.is_set():
            if not self.render_queue:
                time.sleep(1)
                continue

            clip = self.render_queue.popleft()
            cid = clip["id"]
            s, e = clip["start"], clip["end"]

            clip["status"] = "rendering"
            clip["progress"] = 10

            # 1. Wait for segment pieces to be flushed to disk (up to 45s)
            deadline = time.time() + 45
            while time.time() < deadline and not self.stop_flag.is_set():
                self._scan_segments()
                if self.ended or (self.segments and self.segments[-1][2] >= e + 1):
                    break
                time.sleep(1)

            with self.lock:
                segs = [p for p, a, b in self.segments if b >= s - PAD and a <= e + PAD and p.exists()]

            if not segs:
                clip["status"] = "error"
                clip["error"] = "Segments not found in buffer"
                continue

            first_seg_start = next(a for p, a, b in self.segments if p == segs[0])
            clip["progress"] = 25
            sec_file = self.sections_dir / f"{cid}.mp4"
            lst_file = self.sections_dir / f"{cid}.txt"

            try:
                # Concat segments into one clean MP4 section so OpenCV can decode every frame cleanly
                lst_file.write_text("".join(f"file '{p.as_posix()}'\n" for p in segs), encoding="utf-8")
                subprocess.run(
                    ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
                     "-i", str(lst_file), "-c:v", "copy", "-c:a", "aac", str(sec_file)],
                    check=True
                )
                lst_file.unlink(missing_ok=True)
                clip["progress"] = 40

                # 2. Extract words and audio directly for clip
                work_render = self.dir / "render" / cid
                work_render.mkdir(parents=True, exist_ok=True)
                cw = work_render / "clip16k.wav"

                subprocess.run(
                    ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(sec_file),
                     "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(cw)],
                    check=True
                )
                clip["progress"] = 55

                # Millisecond-accurate local speech-to-text transcription via Sherpa-ONNX
                from ClipFarm.scripts.clipfarm import listen as cf_listen
                clip_words = cf_listen.clip_words(cw)
                if not clip_words:
                    with self.lock:
                        clip_words = [{**w, "s": round(w["s"] - s, 2), "e": round(w["e"] - s, 2)}
                                      for w in self.words if s <= w["s"] <= e]

                words_file = work_render / "words.json"
                write_json(words_file, clip_words)

                # 3. Call ClipFarm's authentic render_one pipeline:
                from ClipFarm.scripts.clipfarm import render as cf_render
                clip_spec = {
                    "id": cid,
                    "start": s,
                    "end": e,
                    "title": clip.get("title") or clip["hook"],
                    "hook": clip["hook"],
                    "caption": clip.get("caption") or f"{self.streamer} - {clip['hook']}",
                    "streamer": self.streamer,
                    "hashtags": clip.get("hashtags") or ["#shorts", "#streamer", "#viral", "#gaming", f"#{_clean_slug(self.streamer)}"],
                    "custom_words": clip_words,
                    "curated": True,
                    "saved": True,
                    "score": clip.get("score", 95)
                }
                out_dir = self.clips_dir
                out_dir.mkdir(parents=True, exist_ok=True)
                meta = cf_render.render_one(
                    self.info,
                    self.dir,
                    clip_spec,
                    len(self.clips_ready) + 1,
                    out_dir,
                    section=(sec_file, first_seg_start)
                )
                out_name = meta["file"]
                out_path = out_dir / out_name

                # Keep clean raw 16:9 intermediate section so Live Video Editor can re-edit anytime
                # sec_file.unlink(missing_ok=True)
                cw.unlink(missing_ok=True)

                clip["status"] = "ready"
                clip["progress"] = 100
                clip["filename"] = out_name
                clip["video_url"] = f"/media/video/{out_name}"
                clip["download_url"] = f"/api/download/{out_name}"

                with self.lock:
                    if clip in self.clips_editing:
                        self.clips_editing.remove(clip)
                    self.clips_ready.append(clip)

                self._save_final_json()

                # Autonomous Zero-Supervision Auto-Pilot posting hook for rolling live clips
                try:
                    from studio import clipfarm_server
                    threading.Thread(target=clipfarm_server.trigger_autopost_for_stream, args=(self.key,), daemon=True).start()
                except Exception as ex:
                    print(f"[LiveRender] Warning triggering autopost: {ex}", flush=True)

            except Exception as e:
                clip["status"] = "error"
                clip["error"] = str(e)
                print(f"[LiveRender] Error rendering {cid}: {e}", flush=True)

    def _build_kinetic_ass(self, words: list[dict], hook: str, ass_path: Path):
        """Generates ASS kinetic subtitles matching charlie947/motion-graphics-skills safe zones."""
        header = (
            "[Script Info]\n"
            "ScriptType: v4.00+\n"
            "PlayResX: 1080\n"
            "PlayResY: 1920\n"
            "ScaledBorderAndShadow: yes\n\n"
            "[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
            "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding\n"
            "Style: Hook,Arial Black,72,&H0000FFFF,&H000000FF,&H00000000,&H90000000,-1,0,0,0,100,100,0,0,1,8,4,8,40,40,400,1\n"
            "Style: Kinetic,Arial Black,68,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,1,0,1,7,3,2,40,40,550,1\n\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        )

        def _fmt_ass_time(sec: float) -> str:
            sec = max(0.0, sec)
            h = int(sec // 3600)
            m = int((sec % 3600) // 60)
            s = sec % 60
            return f"{h}:{m:02d}:{s:05.2f}"

        events = []
        # Big Hook Title card for first 3.5 seconds
        events.append(f"Dialogue: 1,0:00:00.00,0:00:03.50,Hook,,0,0,0,,{{\\fad(200,200)}}{hook}")

        # Group words into 3-4 word phrases with active word pop
        chunk_sz = 3
        for i in range(0, len(words), chunk_sz):
            group = words[i:i + chunk_sz]
            g_start = group[0]["s"]
            g_end = group[-1]["e"] + 0.3

            for active_idx, w in enumerate(group):
                w_start = w["s"]
                w_end = w["e"]
                rendered_words = []
                for idx, item in enumerate(group):
                    if idx == active_idx:
                        # Cyan pop highlight
                        rendered_words.append(r"{\c&H00F0FF&\fscx115\fscy115}" + item["w"] + r"{\rKinetic}")
                    else:
                        rendered_words.append(item["w"])
                line_text = " ".join(rendered_words)
                events.append(f"Dialogue: 0,{_fmt_ass_time(w_start)},{_fmt_ass_time(w_end)},Kinetic,,0,0,0,,{line_text}")

        ass_path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")

    def _save_final_json(self):
        final_data = {
            "key": self.key,
            "clips": self.clips_found,
            "ready_clips": self.clips_ready
        }
        write_json(self.dir / "final.json", final_data)

    def get_status(self) -> dict[str, Any]:
        with self.lock:
            elapsed = int(time.time() - self.started_at)
            downloaded_sec = self.samples / SR
            transcribed_sec = self.t_transcribed
            pct = round((transcribed_sec / max(1.0, downloaded_sec)) * 100, 1) if downloaded_sec > 0 else 0.0
            mb_down = round(self.audio_bytes / (1024 * 1024), 1)

            return {
                "key": self.key,
                "url": self.url,
                "streamer": self.streamer,
                "title": self.title,
                "platform": self.platform,
                "status": self.status,
                "elapsed_sec": elapsed,
                "elapsed_hms": hms(elapsed),
                "audio_downloaded_sec": round(downloaded_sec, 1),
                "audio_downloaded_hms": hms(int(downloaded_sec)),
                "audio_downloaded_mb": mb_down,
                "audio_transcribed_sec": round(transcribed_sec, 1),
                "audio_transcribed_hms": hms(int(transcribed_sec)),
                "transcribe_pct": min(100.0, pct),
                "is_caught_up": (self.status == "live_monitoring"),
                "catchup_speed": self.current_speed,
                "clips_found_count": len(self.clips_found),
                "clips_editing": list(self.clips_editing),
                "clips_ready": list(self.clips_ready),
                "words_count": len(self.words),
                "segments_count": len(self.segments)
            }
