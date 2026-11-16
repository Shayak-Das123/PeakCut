"""Watched Creators Background Auto-Clipper Daemon for ClipFarm.
Allows users to add streamer channels (Twitch, YouTube, Kick).
A lightweight background loop polls their live status every 45-60 seconds.
The moment any watched creator goes live:
- Automatically resolves the stream URL.
- Spawns the PeakCut-style continuous segmented rolling buffer from t=0.
- Transcribes speech, detects excitement spikes, and auto-renders vertical 9:16 clips
  on the RTX 4050 GPU while the streamer is still broadcasting.
"""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Optional

from studio import profiles

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "ClipFarm" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CREATORS_FILE = DATA_DIR / "creators.json"

_lock = threading.RLock()
_poll_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()


def _read_creators() -> list[dict]:
    with _lock:
        if not CREATORS_FILE.exists():
            return []
        try:
            return json.loads(CREATORS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []


def _save_creators(creators: list[dict]):
    with _lock:
        CREATORS_FILE.write_text(json.dumps(creators, indent=2, ensure_ascii=False), encoding="utf-8")


def list_creators() -> list[dict]:
    return _read_creators()


def add_creator(channel_url: str, auto_clip: bool = True) -> dict:
    url = channel_url.strip()
    if "://" not in url:
        url = "https://" + url

    # Fetch creator profile metadata via PeakCut profiles engine
    meta = profiles.preview(url)
    plat = meta.get("platform", "Twitch")
    name = meta.get("name") or "Streamer"
    followers = meta.get("followers", "")
    avatar = meta.get("pic", "")

    # Clean ID
    cid = f"c_{re.sub(r'[^a-zA-Z0-9]+', '_', name.lower()).strip('_')}"
    creators = _read_creators()

    # Check if already exists
    for c in creators:
        if c.get("id") == cid or c.get("channel_url").lower() == url.lower():
            c["auto_clip"] = auto_clip
            _save_creators(creators)
            return c

    # Check initial live status
    live_info = profiles.live_now(plat, url)
    creator_item = {
        "id": cid,
        "name": name,
        "platform": plat.lower(),
        "channel_url": url,
        "followers": followers,
        "avatar": avatar,
        "auto_clip": auto_clip,
        "is_live": bool(live_info.get("live")),
        "live_title": live_info.get("title", ""),
        "viewers": live_info.get("viewers", 0),
        "last_checked": time.time(),
        "last_live": time.time() if live_info.get("live") else None,
        "clips_made": 0
    }
    creators.append(creator_item)
    _save_creators(creators)
    return creator_item


def remove_creator(cid: str) -> bool:
    with _lock:
        creators = _read_creators()
        new_list = [c for c in creators if c.get("id") != cid]
        if len(new_list) != len(creators):
            _save_creators(new_list)
            return True
        return False


def toggle_auto_clip(cid: str, enabled: Optional[bool] = None) -> Optional[dict]:
    with _lock:
        creators = _read_creators()
        target = None
        for c in creators:
            if c.get("id") == cid:
                c["auto_clip"] = (not c.get("auto_clip", True)) if enabled is None else enabled
                target = c
                break
        if target:
            _save_creators(creators)
        return target


def check_creator_now(cid: str) -> Optional[dict]:
    creators = _read_creators()
    for c in creators:
        if c.get("id") == cid:
            plat = c.get("platform", "twitch").title()
            url = c.get("channel_url")
            try:
                live_info = profiles.live_now(plat, url)
                c["is_live"] = bool(live_info.get("live"))
                c["live_title"] = live_info.get("title", "")
                c["viewers"] = live_info.get("viewers", 0)
                c["last_checked"] = time.time()
                if c["is_live"]:
                    c["last_live"] = time.time()
                _save_creators(creators)
            except Exception:
                pass
            return c
    return None


def _background_poller():
    """Background polling daemon: checks watched creators every 45-60 seconds.
    If a creator goes live and auto_clip is enabled, triggers the rolling live pipeline!
    """
    from studio import clipfarm_server

    while not _stop_event.is_set():
        try:
            creators = _read_creators()
            for c in creators:
                if not c.get("auto_clip", True):
                    continue

                plat = c.get("platform", "twitch").title()
                url = c.get("channel_url", "")
                if not url:
                    continue

                try:
                    live_info = profiles.live_now(plat, url)
                    is_live = bool(live_info.get("live"))
                    c["is_live"] = is_live
                    c["live_title"] = live_info.get("title", "")
                    c["viewers"] = live_info.get("viewers", 0)
                    c["last_checked"] = time.time()

                    if is_live:
                        c["last_live"] = time.time()
                        # Check if this stream is already being recorded
                        embed_key = live_info.get("embed") or c.get("id")
                        already_active = False
                        for active_key, sess in clipfarm_server.ACTIVE_LIVE_SESSIONS.items():
                            if embed_key in active_key or c.get("name", "").lower() in active_key.lower():
                                if getattr(sess, "status", "") in ("starting", "catching_up", "live_monitoring", "recording"):
                                    already_active = True
                                    break

                        if not already_active:
                            # Streamer just went LIVE! Auto-launch rolling live pipeline
                            print(f"[WatchedCreators] 🔴 Creator {c.get('name')} went LIVE: {c.get('live_title')}! Launching auto-clipper...", flush=True)
                            key = f"{c.get('platform', 'twitch')}-{embed_key}"
                            info = {
                                "id": embed_key,
                                "streamer": c.get("name"),
                                "title": c.get("live_title") or f"{c.get('name')} Live",
                                "platform": c.get("platform", "twitch"),
                                "url": url,
                                "is_live": True
                            }
                            clipfarm_server._start_live_pipeline(key, url, info, auto_clip=True)
                            c["clips_made"] = c.get("clips_made", 0) + 1

                except Exception as e:
                    # Individual channel check failed; continue to next
                    pass

            _save_creators(creators)

        except Exception as e:
            print(f"[WatchedCreatorsPoller] Loop error: {e}", flush=True)

        # Sleep 50 seconds before next check
        _stop_event.wait(50)


def start_watcher_daemon():
    global _poll_thread
    if _poll_thread is None or not _poll_thread.is_alive():
        _stop_event.clear()
        _poll_thread = threading.Thread(target=_background_poller, name="WatchedCreatorsPoller", daemon=True)
        _poll_thread.start()
        print("[WatchedCreators] Daemon started: polling watched streamers every 50s.", flush=True)


def stop_watcher_daemon():
    _stop_event.set()
