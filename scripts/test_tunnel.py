import urllib.request, json

url = "https://effort-government-runner-arrow.trycloudflare.com/api/clips"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode())
        print(f"Cloudflare Tunnel live! Total clips: {len(data)}")
        caseoh = [c for c in data if c.get("streamer") == "CaseOh"]
        kai = [c for c in data if c.get("streamer") == "Kai Cenat"]
        print(f"CaseOh clips: {len(caseoh)}")
        for c in caseoh:
            print(f"   * {c.get('hook')} -> {c.get('video_url')}")
        print(f"Kai Cenat clips: {len(kai)}")
        for c in kai[-4:]:
            print(f"   * {c.get('hook')} -> {c.get('video_url')}")
except Exception as e:
    print("Tunnel error:", e)
