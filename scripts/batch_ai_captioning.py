import json
import glob
import os
import sys
from pathlib import Path

# Ensure root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import studio.ai as ai

SCHEMA = {
    "type": "object",
    "properties": {
        "hook": {"type": "string", "description": "Punchy 3 to 6 word viral title in Title Case (e.g. 'Speed Flips Out Over Lucia!')"},
        "caption": {"type": "string", "description": "Engaging 2-sentence description that asks a debate-inducing question to drive comments"},
        "hashtags": {"type": "array", "items": {"type": "string"}, "description": "4-6 viral hashtags starting with #"},
        "score": {"type": "integer", "description": "Viral potential score from 80 to 99"}
    },
    "required": ["hook", "caption", "hashtags", "score"]
}

def get_clip_dialogue(json_path: Path) -> str:
    # 1. Try words.json companion
    words_f = json_path.with_suffix(".words.json")
    if words_f.exists():
        try:
            w_data = json.load(open(words_f, encoding="utf-8"))
            if isinstance(w_data, list):
                words = [w.get("w", "") for w in w_data if isinstance(w, dict) and w.get("w")]
                if words:
                    return " ".join(words)
        except Exception:
            pass
    # 2. Try inside the clip json itself
    try:
        data = json.load(open(json_path, encoding="utf-8"))
        if isinstance(data, dict):
            wl = data.get("words_list")
            if wl and isinstance(wl, list):
                words = [w.get("w", "") for w in wl if isinstance(w, dict) and w.get("w")]
                if words:
                    return " ".join(words)
    except Exception:
        pass
    return ""

def process_stream(stream_key: str, streamer_name: str, limit: int = 15):
    stream_dir = Path("ClipFarm/work/streams") / stream_key
    clips_dir = stream_dir / "clips"
    if not clips_dir.exists():
        print(f"Directory {clips_dir} does not exist.")
        return

    json_files = sorted(clips_dir.glob("*.json"))
    candidates = []

    for jf in json_files:
        if jf.name.endswith(".words.json"):
            continue
        mp4_f = jf.with_suffix(".mp4")
        if not mp4_f.exists() or mp4_f.stat().st_size < 100000:
            continue
        dlg = get_clip_dialogue(jf)
        word_count = len(dlg.split())
        if word_count >= 15:
            candidates.append((jf, mp4_f, dlg, word_count))

    print(f"Found {len(candidates)} quality candidates for {streamer_name} ({stream_key})")
    # Sort candidates by dialogue length / substance
    candidates.sort(key=lambda x: x[3], reverse=True)
    selected = candidates[:limit]

    curated_clips = []

    for jf, mp4_f, dlg, wc in selected:
        print(f"\nProcessing {jf.name} ({wc} words)...")
        prompt = f"""You are a master viral TikTok & YouTube Shorts editor.
Streamer: {streamer_name} playing Grand Theft Auto VI (GTA 6).
Spoken Dialogue in Clip:
"{dlg}"

Generate a high-converting, curiosity-inducing viral hook title (Title Case), a 2-sentence storytelling description asking a debate question to drive viewer comments, viral hashtags, and a viral score (80-99).
NEVER use boring transcript snippets like 'I DONT WANT TO' as the hook. Write a real compelling YouTube Shorts title!"""

        try:
            res, _, _ = ai.call("gemini-2.5-flash", prompt, SCHEMA)
            print(f"  -> Hook: {res['hook']}")
            print(f"  -> Caption: {res['caption'][:80]}...")

            # Update clip json
            clip_data = json.load(open(jf, encoding="utf-8"))
            clip_data["hook"] = res["hook"]
            clip_data["title"] = res["hook"]
            clip_data["caption"] = res["caption"]
            clip_data["hashtags"] = res["hashtags"]
            clip_data["score"] = res["score"]
            clip_data["saved"] = True
            clip_data["curated"] = True

            with open(jf, "w", encoding="utf-8") as f:
                json.dump(clip_data, f, indent=1)

            # Update post text companion
            txt_f = jf.with_suffix(".txt")
            txt_content = f"""HOOK (on screen): {res['hook']}

— TikTok / Instagram Reels —
{res['hook']}
{res['caption']}
🎥 {streamer_name} on {clip_data.get('platform', 'stream').title()}
{' '.join(res['hashtags'])}

— YouTube Shorts —
Title: {res['hook']}
Description: {res['caption']}
🎥 {streamer_name} on {clip_data.get('platform', 'stream').title()}
{' '.join(res['hashtags'])}
"""
            with open(txt_f, "w", encoding="utf-8") as f:
                f.write(txt_content)

            curated_clips.append({
                "id": clip_data.get("id") or jf.stem.split("_")[0],
                "file": mp4_f.name,
                "filename": mp4_f.name,
                "video_url": f"/media/video/{mp4_f.name}",
                "download_url": f"/api/download/{mp4_f.name}",
                "hook": res["hook"],
                "title": res["hook"],
                "caption": res["caption"],
                "hashtags": res["hashtags"],
                "score": res["score"],
                "duration": clip_data.get("length") or clip_data.get("duration") or 30.0,
                "curated": True,
                "saved": True
            })
        except Exception as e:
            print(f"  Error generating AI caption for {jf.name}: {e}")

    # Update final.json
    final_p = stream_dir / "final.json"
    final_data = {"key": stream_key, "clips": curated_clips}
    with open(final_p, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=1)
    print(f"Saved {len(curated_clips)} curated moments to {final_p}")

if __name__ == "__main__":
    print("=== Regenerating AI Viral Captions via Gemini API Fallback ===")
    process_stream("youtube-qSyo5d-dVzA", "IShowSpeed", limit=10)
    process_stream("twitch-xqc", "xQc", limit=10)
    print("\nDone! All clips updated with high-converting hooks and captions.")
