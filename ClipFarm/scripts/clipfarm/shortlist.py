"""After the AI read every chunk: merge its picks, weigh them with the signals, drop repeats, build the final round."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import signals
from .util import LEDGER, hms, parse_t, read_json, say, settings, write_json

TYPES = {"first reaction", "discovery", "freakout", "funny", "fail", "hot take", "wholesome", "rage", "clutch",
         "story", "reaction", "callout", "other"}


def load_sig(d: Path) -> dict:
    z = np.load(d / "signals.npz")
    raw = {k[4:]: z[k] for k in z.files if k.startswith("raw_")}
    return {"hype": z["hype"], "raw": raw}


def check_picks(d: Path) -> tuple[list[str], list[dict]]:
    """Returns (problems, all moments). Every chunk needs a picks file, every moment the right fields."""
    problems: list[str] = []
    moments: list[dict] = []
    chunks = sorted((d / "chunks").glob("chunk_*.md"))
    for c in chunks:
        p = d / "picks" / (c.stem + ".json")
        if not p.exists():
            problems.append(f"{p.name}: missing")
            continue
        try:
            data = json.loads(p.read_text())
        except Exception as e:
            problems.append(f"{p.name}: not valid JSON ({e})")
            continue
        items = data.get("moments") if isinstance(data, dict) else data
        if not isinstance(items, list):
            problems.append(f"{p.name}: needs {{\"moments\": [...]}}")
            continue
        for i, m in enumerate(items):
            try:
                s, e = parse_t(m["start"]), parse_t(m["end"])
                sc = float(m["score"])
            except Exception:
                problems.append(f"{p.name} #{i + 1}: needs start, end (h:mm:ss) and score (0-100)")
                continue
            if e <= s:
                problems.append(f"{p.name} #{i + 1}: end is before start")
                continue
            moments.append({"start": s, "end": e, "score": max(0.0, min(100.0, sc)),
                            "type": str(m.get("type", "other")).lower(), "title": m.get("title", ""),
                            "why": m.get("why", ""), "standalone": m.get("standalone", True) is not False,
                            "chunk": c.stem})
    return problems, moments


def _iou(a, b) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def ledger() -> list[dict]:
    return read_json(LEDGER, []) or []


def already_clipped(key: str, s: float, e: float) -> bool:
    for c in ledger():
        if c.get("stream") == key and _iou((s, e), (c["start"], c["end"])) > 0.25:
            return True
    return False


def build(d: Path, top: int = 12) -> list[dict]:
    problems, moments = check_picks(d)
    if problems:
        say("! some picks files need fixing first:")
        for p in problems[:20]:
            say(f"  {p}")
        raise SystemExit(1)
    info = read_json(d / "info.json", {})
    lines = read_json(d / "lines.json", [])
    segs = read_json(d / "segments.json", [])
    sig = load_sig(d)
    cfg = settings()
    key = d.name
    dur = float(info.get("duration") or 0)

    # snap to whole lines, so a clip never starts or ends mid-sentence
    def snap(s, e):
        for ln in lines:
            if ln["s"] <= s < ln["e"] and ln["kind"] == "speech" and s - ln["s"] < 6:
                s = ln["s"]
            if ln["s"] < e <= ln["e"] and ln["kind"] == "speech" and ln["e"] - e < 6:
                e = ln["e"]
        return max(0.0, s - 0.3), min(dur, e + 0.5)

    cands = []
    for m in moments:
        if not m["standalone"]:
            continue
        s, e = snap(m["start"], m["end"])
        if e - s < cfg["min_clip_s"]:
            pad = (cfg["min_clip_s"] - (e - s)) / 2
            s, e = max(0, s - pad), min(dur, e + pad)
        sigsc = signals.window_score(sig, s, e)
        music = sum(max(0.0, min(e, x["e"]) - max(s, x["s"])) for x in segs if x["kind"] == "music")
        music_share = music / max(1.0, e - s)
        combined = 0.65 * m["score"] + 0.35 * sigsc - (8 if music_share > 0.4 else 0)
        cands.append({**m, "start": round(s, 2), "end": round(e, 2), "signal": round(sigsc, 1),
                      "combined": round(combined, 1), "music": round(music_share, 2),
                      "why_signals": signals.explain(sig, int(s + int(np.argmax(sig["hype"][int(s):int(e) + 1])))
                                                     if e > s else int(s))})
    cands.sort(key=lambda c: -c["combined"])
    kept: list[dict] = []
    dropped_repeat = 0
    for c in cands:
        if any(_iou((c["start"], c["end"]), (k["start"], k["end"])) > 0.3 for k in kept):
            continue
        if already_clipped(key, c["start"], c["end"]):
            dropped_repeat += 1
            continue
        kept.append(c)
        if len(kept) >= top:
            break
    for i, c in enumerate(kept, 1):
        c["id"] = f"c{i:02d}"
    write_json(d / "shortlist.json", kept)
    _write_md(d, info, lines, kept, len(moments), dropped_repeat)
    return kept


def _write_md(d: Path, info: dict, lines: list[dict], kept: list[dict], n_moments: int, dropped: int) -> None:
    cfg = settings()
    out = [f"# Final round · {info.get('streamer')} · {info.get('title', '')[:80]}", "",
           f"{n_moments} moments came out of the chunks. These are the best {len(kept)} by your score and the signals"
           + (f" ({dropped} already clipped before, dropped)." if dropped else "."),
           f"Pick up to {cfg['max_clips_per_stream']} (default {cfg['clips_per_stream']}). Fewer is fine: never post a weak one.",
           "Rules: reference/final-round.md. Write final.json in this stream's folder.", ""]
    for c in kept:
        out.append(f"## {c['id']} · {hms(c['start'])}–{hms(c['end'])} · {c['end'] - c['start']:.0f} s · "
                   f"{c['type']} · combined {c['combined']:.0f} (you {c['score']:.0f}, signals {c['signal']:.0f})")
        if c.get("title"):
            out.append(f"working title: {c['title']}")
        if c.get("why"):
            out.append(f"why: {c['why']}")
        if c.get("why_signals"):
            out.append(f"signals: {c['why_signals']}")
        if c["music"] > 0.4:
            out.append(f"⚠ music under {int(c['music'] * 100)} % of it: copyrighted songs get Shorts claimed or muted")
        out.append("")
        for ln in lines:
            if ln["e"] > c["start"] - 20 and ln["s"] < c["end"] + 15:
                inside = c["start"] <= ln["s"] < c["end"]
                out.append(f"{'│' if inside else ' '} [{hms(ln['s'])}] {ln['text']}")
        out.append("")
    (d / "shortlist.md").write_text("\n".join(out) + "\n")
