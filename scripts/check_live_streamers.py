import subprocess, json

streamlink = ".venv\\Scripts\\streamlink.exe"
streamers = [
    ("twitch", "caseoh_", "CaseOh"),
    ("twitch", "kaicenat", "Kai Cenat"),
    ("twitch", "xqc", "xQc"),
    ("twitch", "jynxzi", "Jynxzi"),
    ("twitch", "tarik", "Tarik"),
    ("twitch", "hasanabi", "HasanAbi"),
    ("twitch", "clix", "Clix"),
    ("twitch", "shroud", "Shroud"),
    ("twitch", "flight23white", "FlightReacts"),
    ("youtube", "https://www.youtube.com/@IShowSpeed/live", "IShowSpeed"),
    ("youtube", "https://www.youtube.com/@CarryisLive/live", "CarryMinati"),
]

for plat, ch, name in streamers:
    url = f"https://www.twitch.tv/{ch}" if plat == "twitch" else ch
    try:
        p = subprocess.run([streamlink, "--json", url], capture_output=True, text=True, timeout=12)
        data = json.loads(p.stdout)
        meta = data.get("metadata", {})
        title = meta.get("title")
        if title:
            cat = meta.get("category", "")
            print(f"[LIVE NOW] {name} ({plat}): '{title}' (Category: {cat})")
        else:
            print(f"[Offline] {name}")
    except Exception as e:
        print(f"[Error/Offline {name}]")
