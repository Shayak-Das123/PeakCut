"""Autonomous 1-Hour Posting Scheduler & Approval Queue Engine for PeakCut / ClipFarm.
Manages:
- Autopilot mode: When ON, newly rendered clips are auto-approved and scheduled 1 hour after the previous slot.
  When OFF, newly rendered clips wait in the Review Queue for user approval.
- Approved Schedule Timeline: Clips queued sequentially 1 hour apart (3600 seconds).
- Background Worker: Checks due posts every 20 seconds and uploads them directly to YouTube Shorts.
- Manual actions: Approve, Disapprove, Post Now, Reschedule, Delete.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ClipFarm" / "scripts"))
sys.path.insert(0, str(ROOT))

from studio import accounts, config, ctx
from clipfarm.util import read_json, write_json
SCHEDULE_FILE = ROOT / "ClipFarm" / "data" / "schedule.json"
SCHEDULE_FILE.parent.mkdir(parents=True, exist_ok=True)

_lock = threading.RLock()

DEFAULT_SCHEDULE = {
    "autopilot": True,
    "interval_minutes": 60,
    "posts": []
}


def _load_data() -> dict:
    if not SCHEDULE_FILE.exists():
        return dict(DEFAULT_SCHEDULE)
    try:
        data = read_json(SCHEDULE_FILE, DEFAULT_SCHEDULE)
        return {**DEFAULT_SCHEDULE, **data}
    except Exception:
        return dict(DEFAULT_SCHEDULE)


def _save_data(data: dict):
    with _lock:
        write_json(SCHEDULE_FILE, data)


def get_schedule() -> dict:
    """Returns schedule data with posts sorted and categorized."""
    data = _load_data()
    posts = data.get("posts", [])

    # Format human-readable scheduled strings
    now = time.time()
    for p in posts:
        s_at = p.get("scheduled_at")
        if s_at:
            dt = datetime.fromtimestamp(s_at)
            diff = s_at - now
            if diff < 0 and p.get("status") == "approved":
                rel = "Due now"
            elif diff < 3600:
                rel = f"in {max(1, int(diff / 60))} mins"
            else:
                rel = f"in {int(diff / 3600)}h {int((diff % 3600) / 60)}m"
            p["scheduled_time_str"] = dt.strftime("%b %d, %I:%M %p")
            p["relative_time"] = rel
        else:
            p["scheduled_time_str"] = "Pending Review"
            p["relative_time"] = "Awaiting Approval"

    pending = [p for p in posts if p.get("status") == "pending"]
    approved = [p for p in posts if p.get("status") == "approved"]
    posted = [p for p in posts if p.get("status") == "posted"]
    rejected = [p for p in posts if p.get("status") == "rejected"]

    # Sort approved by scheduled_at ascending (timeline order)
    approved.sort(key=lambda x: x.get("scheduled_at", 0))
    # Sort posted by posted_at descending
    posted.sort(key=lambda x: x.get("posted_at", 0), reverse=True)
    # Sort pending by created_at descending
    pending.sort(key=lambda x: x.get("created_at", 0), reverse=True)

    return {
        "ok": True,
        "autopilot": bool(data.get("autopilot", True)),
        "interval_minutes": int(data.get("interval_minutes", 60)),
        "total": len(posts),
        "pending_count": len(pending),
        "approved_count": len(approved),
        "posted_count": len(posted),
        "rejected_count": len(rejected),
        "next_post": approved[0] if approved else None,
        "pending": pending,
        "approved": approved,
        "posted": posted,
        "rejected": rejected
    }


def toggle_autopilot(enabled: bool | None = None) -> bool:
    with _lock:
        data = _load_data()
        if enabled is None:
            data["autopilot"] = not data.get("autopilot", True)
        else:
            data["autopilot"] = bool(enabled)
        _save_data(data)
        return data["autopilot"]


def compute_next_slot(interval_minutes: int = 60) -> float:
    """Computes the next available timestamp, spaced interval_minutes apart from the last approved post."""
    data = _load_data()
    now = time.time()
    approved = [p for p in data.get("posts", []) if p.get("status") == "approved"]
    if not approved:
        # If queue is empty, schedule first post 5 minutes from now (or immediate)
        return round(now + 300.0, 0)

    latest_time = max((p.get("scheduled_at", 0) for p in approved), default=0)
    slot_gap = interval_minutes * 60.0

    if latest_time > now:
        return round(latest_time + slot_gap, 0)
    else:
        return round(now + slot_gap, 0)


def add_rendered_clip_to_schedule(clip_meta: dict) -> dict:
    """Called when a new reel finishes rendering to enqueue it into the schedule."""
    with _lock:
        data = _load_data()
        is_auto = bool(data.get("autopilot", True))
        interval = int(data.get("interval_minutes", 60))

        filename = clip_meta.get("filename") or clip_meta.get("file") or ""
        # Avoid duplicate scheduling of the same filename
        existing = next((p for p in data.get("posts", []) if p.get("file") == filename), None)
        if existing:
            return existing

        post_id = f"post_{int(time.time())}_{secrets.token_hex(4)}"
        status = "approved" if is_auto else "pending"
        scheduled_at = compute_next_slot(interval) if is_auto else None

        title = clip_meta.get("hook") or clip_meta.get("title") or "Viral Moment"
        streamer = clip_meta.get("streamer") or "Streamer"
        caption = clip_meta.get("caption") or f"{streamer} live in GTA 6! What would you do in this chase?"
        tags = clip_meta.get("hashtags") or [f"#{streamer.lower().replace(' ', '')}", "#gta6", "#gaming", "#Shorts", "#viral"]
        if "#Shorts" not in tags and "Shorts" not in tags:
            tags.append("#Shorts")

        post = {
            "id": post_id,
            "clip_id": clip_meta.get("id") or post_id,
            "file": filename,
            "filepath": str(clip_meta.get("filepath", "")),
            "streamer": streamer,
            "title": title[:100],
            "hook": title[:100],
            "caption": caption[:4900],
            "hashtags": tags,
            "score": clip_meta.get("score", 95),
            "duration": clip_meta.get("duration", 25.0),
            "speaker_type": clip_meta.get("speaker_type", "reaction"),
            "status": status,
            "created_at": time.time(),
            "scheduled_at": scheduled_at,
            "posted_at": None,
            "youtube_url": None,
            "youtube_id": None,
            "error": None
        }

        data.setdefault("posts", []).append(post)
        _save_data(data)
        mode_str = "Auto-Approved & Scheduled" if is_auto else "Sent to Review Queue (Pending Approval)"
        print(f"[Scheduler] Clip '{filename}' {mode_str} (ID: {post_id})")
        return post


def approve_post(post_id: str) -> dict:
    with _lock:
        data = _load_data()
        post = next((p for p in data.get("posts", []) if p.get("id") == post_id), None)
        if not post:
            raise ValueError(f"Post {post_id} not found.")

        interval = int(data.get("interval_minutes", 60))
        post["status"] = "approved"
        post["scheduled_at"] = compute_next_slot(interval)
        post["error"] = None
        _save_data(data)
        print(f"[Scheduler] Post {post_id} APPROVED -> Scheduled for {post['scheduled_at']}")
        return post


def disapprove_post(post_id: str) -> dict:
    with _lock:
        data = _load_data()
        post = next((p for p in data.get("posts", []) if p.get("id") == post_id), None)
        if not post:
            raise ValueError(f"Post {post_id} not found.")

        post["status"] = "rejected"
        _save_data(data)
        print(f"[Scheduler] Post {post_id} DISAPPROVED -> Rejected.")
        return post


def reschedule_post(post_id: str, new_timestamp: float) -> dict:
    with _lock:
        data = _load_data()
        post = next((p for p in data.get("posts", []) if p.get("id") == post_id), None)
        if not post:
            raise ValueError(f"Post {post_id} not found.")

        post["scheduled_at"] = float(new_timestamp)
        post["status"] = "approved"
        _save_data(data)
        return post


def delete_post(post_id: str) -> bool:
    with _lock:
        data = _load_data()
        before = len(data.get("posts", []))
        data["posts"] = [p for p in data.get("posts", []) if p.get("id") != post_id]
        _save_data(data)
        return len(data["posts"]) < before


def find_video_path(filename: str, fallback_path: str = "") -> Path | None:
    if fallback_path and Path(fallback_path).is_file():
        return Path(fallback_path)
    work = ROOT / "ClipFarm" / "work" / "streams"
    for s_dir in work.glob("*"):
        candidate = s_dir / "clips" / filename
        if candidate.is_file():
            return candidate
    return None


def execute_upload_post(post: dict) -> dict:
    """Executes YouTube upload for a scheduled post."""
    ctx.set_uid("u_local")
    filename = post.get("file", "")
    p = find_video_path(filename, post.get("filepath", ""))
    if not p or not p.is_file():
        err_msg = f"Video file '{filename}' not found on disk."
        post["error"] = err_msg
        post["status"] = "error"
        return {"ok": False, "error": err_msg}

    all_accs = [a for a in accounts._all() if a.get("platform") == "youtube"]
    if not all_accs:
        err_msg = "No connected YouTube account found. Please connect in Social tab."
        post["error"] = err_msg
        post["status"] = "error"
        return {"ok": False, "error": err_msg}

    acc = all_accs[0]
    title = post.get("title", "") or post.get("hook", "") or "Clip"
    caption = post.get("caption", "")
    tags = post.get("hashtags", ["#Shorts", "#Gaming", "#Viral"])
    desc = f"{caption}\n\n{' '.join('#' + t.lstrip('#') for t in tags)}\n\n#Shorts\n\nClipped autonomously with PeakCut ClipFarm"

    try:
        print(f"[Scheduler] Uploading '{filename}' to YouTube channel '{acc.get('name')}'...")
        vid_id, shorts_url = accounts.youtube_upload(
            acc, p,
            title=title[:100],
            description=desc[:4900],
            tags=[t.lstrip("#") for t in tags[:15]],
            privacy="public"
        )
        post["status"] = "posted"
        post["posted_at"] = time.time()
        post["youtube_url"] = shorts_url
        post["youtube_id"] = vid_id
        post["error"] = None
        print(f"[Scheduler] SUCCESS! Uploaded: {shorts_url}")
        return {"ok": True, "url": shorts_url, "id": vid_id}
    except Exception as e:
        err = str(e)
        post["error"] = err
        print(f"[Scheduler] YouTube Upload Failed for '{filename}': {err}")
        return {"ok": False, "error": err}


def post_now(post_id: str) -> dict:
    with _lock:
        data = _load_data()
        post = next((p for p in data.get("posts", []) if p.get("id") == post_id), None)
        if not post:
            raise ValueError(f"Post {post_id} not found.")

        res = execute_upload_post(post)
        _save_data(data)
        return res


def clear_all_schedule():
    with _lock:
        data = {
            "autopilot": True,
            "interval_minutes": 60,
            "posts": []
        }
        _save_data(data)


# -------------------------------------------------------------------------
# Background Worker Thread
# -------------------------------------------------------------------------
_worker_running = False

def start_scheduler_worker():
    global _worker_running
    if _worker_running:
        return
    _worker_running = True

    def _loop():
        print("[Scheduler Daemon] Started. Checking due posts every 20 seconds...")
        while _worker_running:
            try:
                now = time.time()
                data = _load_data()
                posts = data.get("posts", [])
                due = [p for p in posts if p.get("status") == "approved" and p.get("scheduled_at") and p["scheduled_at"] <= now]
                if due:
                    # Upload first due post
                    p = due[0]
                    print(f"[Scheduler Daemon] Post {p['id']} is due! Initiating automatic upload...")
                    res = execute_upload_post(p)
                    with _lock:
                        # Re-load in case of concurrent modifications
                        cur = _load_data()
                        for idx, item in enumerate(cur.get("posts", [])):
                            if item.get("id") == p["id"]:
                                cur["posts"][idx] = p
                                break
                        _save_data(cur)
            except Exception as e:
                print(f"[Scheduler Daemon] Loop error: {e}")
            time.sleep(20)

    t = threading.Thread(target=_loop, daemon=True, name="SchedulerWorkerThread")
    t.start()
