import json, os

for sk, name in [("twitch-caseoh", "CaseOh"), ("twitch-kaicenat", "Kai Cenat")]:
    s_dir = os.path.join("ClipFarm/work/streams", sk)
    w_file = os.path.join(s_dir, "words.json")
    l_file = os.path.join(s_dir, "lines.json")
    p_file = os.path.join(s_dir, "peaks.json")
    b_dir = os.path.join(s_dir, "buffer")

    words = json.load(open(w_file, encoding="utf-8")) if os.path.exists(w_file) else []
    lines = json.load(open(l_file, encoding="utf-8")) if os.path.exists(l_file) else []
    peaks = json.load(open(p_file, encoding="utf-8")) if os.path.exists(p_file) else []
    segs = [f for f in os.listdir(b_dir) if f.endswith(".ts")] if os.path.exists(b_dir) else []
    last_t = words[-1]["e"] if words else 0.0

    print(f"=== {name} ({sk}) ===")
    print(f"  Segments: {len(segs)} ({len(segs)*6}s video) | Words: {len(words)} | Lines: {len(lines)} | Audio Peaks: {len(peaks)}")
    print(f"  Time span: 0.0s - {last_t:.1f}s")
    if lines:
        print(f"  First line: {lines[0]}")
        print(f"  Latest line: {lines[-1]}")
