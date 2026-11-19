import subprocess, json

streamlink = ".venv\\Scripts\\streamlink.exe"
indian_streamers = [
    ("CarryMinati", "https://www.youtube.com/@CarryisLive/live"),
    ("Techno Gamerz", "https://www.youtube.com/@TechnoGamerzOfficial/live"),
    ("Total Gaming", "https://www.youtube.com/@TotalGaming093/live"),
    ("Scout", "https://www.youtube.com/@scoutOP/live"),
    ("Mortal", "https://www.youtube.com/@Morticall/live"),
    ("Live Insaan", "https://www.youtube.com/@LiveInsaan/live"),
]

print("Checking Indian / Hindi gaming streamers...")
for name, url in indian_streamers:
    try:
        p = subprocess.run([streamlink, "--json", url], capture_output=True, text=True, timeout=12)
        data = json.loads(p.stdout)
        meta = data.get("metadata", {})
        title = meta.get("title")
        if title:
            print(f"[LIVE NOW] {name}: '{title}'")
        else:
            print(f"[Offline] {name}")
    except Exception as e:
        print(f"[Offline/Error] {name}")
