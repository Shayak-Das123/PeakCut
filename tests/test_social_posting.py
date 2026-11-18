"""Test suite for ClipFarm social posting, autonomous auto-pilot, and account endpoints."""
import json
from pathlib import Path
from fastapi.testclient import TestClient

from studio.clipfarm_server import app, get_autopost_settings, save_autopost_settings, AUTOPOST_SETTINGS_FILE
from studio import accounts, config, ctx


def test_accounts_api():
    client = TestClient(app)
    r = client.get("/api/accounts")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert isinstance(data["accounts"], list)
    assert "youtube" in data["apps"]
    assert "instagram" in data["apps"]
    assert "8767" in data["redirect_base"] or "http" in data["redirect_base"]


def test_autopost_settings_crud():
    client = TestClient(app)
    # 1. Read
    r1 = client.get("/api/autopost/settings")
    assert r1.status_code == 200
    assert r1.json()["ok"] is True

    # 2. Save
    payload = {
        "enabled": True,
        "post_youtube": True,
        "post_instagram": False,
        "youtube_privacy": "unlisted",
        "max_clips_per_stream": 3
    }
    r2 = client.post("/api/autopost/settings", json=payload)
    assert r2.status_code == 200
    saved = r2.json()["settings"]
    assert saved["enabled"] is True
    assert saved["post_youtube"] is True
    assert saved["youtube_privacy"] == "unlisted"
    assert saved["max_clips_per_stream"] == 3

    # Reset
    save_autopost_settings({"enabled": False, "post_youtube": False, "post_instagram": False})


def test_save_app_and_oauth_start():
    client = TestClient(app)
    # Save YouTube Google App credentials
    r = client.post("/api/accounts/save-app", json={
        "platform": "youtube",
        "client_id": "test-123.apps.googleusercontent.com",
        "client_secret": "GOCSPX-secret456"
    })
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert r.json()["apps"]["youtube"]["ready"] is True

    # Verify OAuth start endpoint redirects to Google
    r_oauth = client.get("/oauth/youtube/start", follow_redirects=False)
    assert r_oauth.status_code == 303
    assert "accounts.google.com" in r_oauth.headers["location"]
    assert "test-123.apps.googleusercontent.com" in r_oauth.headers["location"]

    # Clean up test credentials
    f = accounts._apps_file()
    if f.exists():
        d = json.loads(f.read_text("utf-8"))
        d.pop("google", None)
        f.write_text(json.dumps(d), "utf-8")
        accounts._apps_seen["at"] = None


def test_post_clip_validation():
    client = TestClient(app)
    # No platform/filename
    r_empty = client.post("/api/post/clip", json={})
    assert r_empty.status_code == 400

    # Clip not found
    r_404 = client.post("/api/post/clip", json={"platform": "youtube", "filename": "non_existent_clip_12345.mp4"})
    assert r_404.status_code == 404

    # No account connected
    r_noacc = client.post("/api/post/clip", json={"platform": "youtube", "filename": "c01_morecaseoh_caseoh-destroys-griefer-with-savag.mp4"})
    assert r_noacc.status_code == 400
    assert "No connected YouTube account found" in r_noacc.json()["detail"]


if __name__ == "__main__":
    test_accounts_api()
    test_autopost_settings_crud()
    test_save_app_and_oauth_start()
    test_post_clip_validation()
    print("ALL SOCIAL POSTING & AUTO-PILOT TESTS PASSED!")
