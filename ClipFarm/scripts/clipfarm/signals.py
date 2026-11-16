"""One hype curve per stream, from every signal we have. Each signal is scaled against the stream itself."""
from __future__ import annotations

import numpy as np

WEIGHTS = {            # what each signal is worth when it's there; missing ones drop out
    "loud": 0.22,      # the streamer got loud compared with their own normal
    "chat": 0.18,      # chat sped up compared with its own normal
    "words": 0.14,     # what chat typed: "CLIP IT", LMAO, "NO WAY", ????
    "replayed": 0.18,  # YouTube's "most replayed" graph: people rewound to watch it again
    "crowd": 0.18,     # Twitch viewers already clipped it
    "laugh": 0.10,     # laughing, screaming, gasping in the transcript
}


def _norm(x: np.ndarray, q: float = 99.0) -> np.ndarray:
    x = np.nan_to_num(np.asarray(x, dtype=float))
    top = np.percentile(x, q) if len(x) else 1.0
    if top <= 1e-9:
        return np.zeros_like(x)
    return np.clip(x / top, 0, 1.5) / 1.5


def _smooth(x: np.ndarray, sec: float) -> np.ndarray:
    n = max(1, int(sec))
    k = np.exp(-0.5 * (np.arange(-2 * n, 2 * n + 1) / n) ** 2)
    k /= k.sum()
    return np.convolve(x, k, mode="same")


def build(duration: float, energy: dict | None, chat: dict | None, heatmap: list | None,
          crowd_clips: list | None, segs: list | None) -> dict:
    n = int(np.ceil(duration)) + 1
    t = np.arange(n)
    parts: dict[str, np.ndarray] = {}
    raw: dict[str, np.ndarray] = {}

    if energy and energy.get("z"):
        z = np.asarray(energy["z"], dtype=float)
        hop = energy["hop"]
        per = int(round(1 / hop))
        m = len(z) // per
        zs = z[: m * per].reshape(m, per).max(axis=1) if m else np.zeros(1)
        zs = np.pad(zs, (0, max(0, n - len(zs))))[:n]
        raw["loud"] = zs
        parts["loud"] = np.clip(zs - 1.0, 0, 5) / 5

    if chat and chat.get("messages", 0) > 30:
        ratio = np.asarray(chat["ratio"], dtype=float)[:n]
        ratio = np.pad(ratio, (0, n - len(ratio)), constant_values=1)
        raw["chat"] = ratio
        parts["chat"] = np.clip(np.log2(np.maximum(ratio, 1e-3)), 0, 3) / 3
        lex = np.asarray(chat["lex"], dtype=float)[:n]
        lex = np.pad(lex, (0, n - len(lex)))
        raw["words"] = lex
        parts["words"] = _norm(lex)
        cc = np.asarray(chat["clipcalls"], dtype=float)[:n]
        raw["clipcalls"] = np.pad(cc, (0, n - len(cc)))

    if heatmap:
        xs = [(h["start_time"] + h["end_time"]) / 2 for h in heatmap]
        vs = [h["value"] for h in heatmap]
        hm = np.interp(t, xs, vs)
        raw["replayed"] = hm
        # what matters is being replayed MORE than the parts around it
        base = np.convolve(hm, np.ones(600) / 600, mode="same") if n > 600 else np.full(n, np.median(hm))
        parts["replayed"] = np.clip((hm - base) / max(1e-6, np.percentile(hm, 99) - np.median(hm)), 0, 1)

    if crowd_clips:
        c = np.zeros(n)
        top = max(c_["views"] for c_ in crowd_clips) or 1
        for c_ in crowd_clips:
            a, b = int(c_["t"]), int(c_["t"] + c_["dur"])
            w = np.log10(1 + c_["views"]) / np.log10(1 + top) if top > 0 else 1
            c[max(0, a):min(n, b)] = np.maximum(c[max(0, a):min(n, b)], 0.4 + 0.6 * w)
        raw["crowd"] = c
        parts["crowd"] = c

    if segs:
        lg = np.zeros(n)
        for s in segs:
            if s["kind"] in ("laugh", "shout"):
                a, b = int(s["s"]), int(s["e"]) + 2
                lg[max(0, a - 1):min(n, b)] = 1.0 if s["kind"] == "laugh" else 0.7
        if lg.any():
            raw["laugh"] = lg
            parts["laugh"] = lg

    if not parts:
        hype = np.zeros(n)
    else:
        wsum = sum(WEIGHTS[k] for k in parts)
        hype = sum(WEIGHTS[k] * _smooth(v, 3) for k, v in parts.items()) / wsum
    hype = _smooth(hype, 2)
    hype = hype / max(1e-6, float(hype.max()))
    return {"n": n, "hype": hype, "parts": parts, "raw": raw, "used": sorted(parts)}


def peaks(sig: dict, top: int = 60, gap: float = 40.0) -> list[dict]:
    hype = sig["hype"]
    order = np.argsort(-hype)
    chosen: list[int] = []
    for i in order:
        if hype[i] <= 0:
            break
        if all(abs(i - j) >= gap for j in chosen):
            chosen.append(int(i))
        if len(chosen) >= top:
            break
    rank = _pct_rank(hype)
    out = []
    for i in sorted(chosen):
        out.append({"t": i, "hype": round(float(hype[i]), 3), "pct": round(float(rank[i]), 1),
                    "why": explain(sig, i)})
    return out


def _pct_rank(x: np.ndarray) -> np.ndarray:
    order = x.argsort()
    r = np.empty(len(x))
    r[order] = np.linspace(0, 100, len(x))
    return r


def explain(sig: dict, i: int, win: int = 8) -> str:
    raw = sig["raw"]
    a, b = max(0, i - win), i + win
    bits = []
    if "chat" in raw:
        r = float(np.max(raw["chat"][a:b]))
        if r >= 1.8:
            bits.append(f"chat ×{r:.1f}")
    if "clipcalls" in raw:
        c = int(round(float(np.max(raw["clipcalls"][a:b]))))
        if c >= 2:
            bits.append(f"\"clip it\" ×{c}")
    if "loud" in raw:
        z = float(np.max(raw["loud"][a:b]))
        if z >= 2.0:
            bits.append(f"loud +{z:.1f}σ")
    if "replayed" in raw:
        h = float(np.max(raw["replayed"][a:b]))
        if h >= 0.5:
            bits.append(f"replayed {h:.2f}")
    if "crowd" in raw and float(np.max(raw["crowd"][a:b])) > 0:
        bits.append("viewers clipped it")
    if "laugh" in raw and float(np.max(raw["laugh"][a:b])) >= 1:
        bits.append("laughing")
    return " · ".join(bits)


def window_score(sig: dict, s: float, e: float) -> float:
    """0-100: how hot this window is compared with the rest of the stream (percentile of its peak)."""
    hype = sig["hype"]
    a, b = max(0, int(s)), min(len(hype), int(e) + 1)
    if b <= a:
        return 0.0
    peak = float(np.max(hype[a:b]))
    return float(np.mean(hype <= peak) * 100)


def sparkline(sig: dict, points: int = 600) -> list[float]:
    h = sig["hype"]
    if len(h) <= points:
        return [round(float(v), 3) for v in h]
    m = len(h) // points
    return [round(float(v), 3) for v in h[: m * points].reshape(points, m).max(axis=1)]
