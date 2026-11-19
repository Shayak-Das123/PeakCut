import json, os
from pathlib import Path

streams = [("twitch-caseoh", "CaseOh"), ("twitch-kaicenat", "Kai Cenat")]
all_valid = True
total_checked = 0

for sk, streamer in streams:
    s_dir = Path("ClipFarm/work/streams") / sk
    clips_dir = s_dir / "clips"
    picks_file = s_dir / "picks.json"
    
    if not picks_file.exists():
        print(f"FAILED: {sk} missing picks.json")
        all_valid = False
        continue
        
    data = json.load(open(picks_file, encoding="utf-8"))
    picks = data.get("curated_picks", [])
    print(f"\n=================== {streamer} ({sk}) ===================")
    
    for p in picks:
        total_checked += 1
        cid = p["id"]
        hook = p["hook"]
        hook_dur = p["hook_duration"]
        length = p["length"]
        ratio = hook_dur / length
        hook_y = p.get("hook_y", 800)
        cap_y = p.get("cap_y", 1420)
        clearance = cap_y - hook_y
        words = p.get("words_list", [])
        
        # Check mp4 file exists and size > 5MB
        mp4_path = clips_dir / p["file"]
        file_ok = mp4_path.exists() and mp4_path.stat().st_size > 5 * 1024 * 1024
        
        # Check hook constraints
        clean_hook = '"' not in hook and "'" not in hook and len(hook.split()) <= 8
        dur_ok = 0.45 <= ratio <= 0.55
        pos_ok = (hook_y == 800 and cap_y == 1420 and clearance >= 400)
        words_ok = len(words) > 0 and all(w["s"] >= 0 and w["e"] <= length + 2.0 for w in words)
        
        print(f"[{cid}] '{hook}'")
        print(f"   Length: {length:.1f}s | HookDur: {hook_dur:.1f}s ({ratio*100:.1f}%) -> dur_ok={dur_ok}")
        print(f"   Hook Y: {hook_y} | Cap Y: {cap_y} | Clearance: {clearance}px -> pos_ok={pos_ok}")
        print(f"   Words: {len(words)} -> words_ok={words_ok} (first: '{words[0]['w']}' at {words[0]['s']}s)")
        print(f"   MP4: {mp4_path.name} ({mp4_path.stat().st_size / (1024**2):.1f} MB) -> file_ok={file_ok}")
        
        if not (clean_hook and dur_ok and pos_ok and words_ok and file_ok):
            print(f"   --> FAILED checks: clean_hook={clean_hook}, dur_ok={dur_ok}, pos_ok={pos_ok}, words_ok={words_ok}, file_ok={file_ok}")
            all_valid = False
        else:
            print("   --> PASSED 100%!")

print(f"\n=======================================================")
print(f"Total Live Reels Verified: {total_checked}. All Valid: {all_valid}")
