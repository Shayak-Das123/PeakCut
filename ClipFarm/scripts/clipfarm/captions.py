"""Word-by-word captions, the hook title and the credit tag, drawn with Pillow (no special ffmpeg build needed)."""
from __future__ import annotations

import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_CANDIDATES = [
    "C:/Windows/Fonts/ariblk.ttf",
    "C:/Windows/Fonts/impact.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/System/Library/Fonts/Supplemental/Arial Black.ttf",
    "/System/Library/Fonts/Supplemental/Impact.ttf",
    "/Library/Fonts/Arial Black.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
]
CAP_W, CAP_H = 1080, 260
YELLOW = (255, 214, 10, 255)
CYAN = (0, 240, 255, 255)
WHITE = (255, 255, 255, 255)
BLACK = (0, 0, 0, 255)

STYLES = {
    "pop_glow": {
        "label": "Radiant Neon Glow",
        "description": "Vibrant neon yellow aura on the active word, elevated pop baseline, soft ambient glow",
        "active_color": (255, 214, 10, 255),
        "glow_outer": (255, 235, 20, 220),
        "glow_inner": (255, 255, 120, 250),
        "inactive_color": (255, 255, 255, 255),
        "outline_color": (0, 0, 0, 255),
        "pop_offset_y": -6,
    },
    "bounce": {
        "label": "Kinetic Bounce (Cyan)",
        "description": "High-energy electric cyan highlight with sharp motion pop",
        "active_color": (0, 240, 255, 255),
        "glow_outer": (0, 200, 255, 200),
        "glow_inner": (180, 255, 255, 250),
        "inactive_color": (255, 255, 255, 255),
        "outline_color": (0, 0, 0, 255),
        "pop_offset_y": -6,
    },
    "clean": {
        "label": "Clean Bold (Classic)",
        "description": "Heavy white typography with deep drop shadow, no color tint",
        "active_color": (255, 255, 255, 255),
        "glow_outer": None,
        "glow_inner": None,
        "inactive_color": (210, 210, 210, 230),
        "outline_color": (0, 0, 0, 255),
        "pop_offset_y": 0,
    }
}


def get_current_style(name: str | None = None) -> dict:
    try:
        from .util import settings
        s = settings()
        preset = name or s.get("subtitle_preset") or "pop_glow"
    except Exception:
        preset = name or "pop_glow"
    return STYLES.get(preset, STYLES["pop_glow"])


def font(size: int) -> ImageFont.FreeTypeFont:
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size)


def clean_word(w: str) -> str:
    w = re.sub(r"[\"“”…]", "", w)
    w = re.sub(r"[.,;:]+$", "", w)
    return w.upper()


def group_words(words: list[dict], max_words: int = 3, max_chars: int = 16) -> list[list[dict]]:
    groups: list[list[dict]] = []
    cur: list[dict] = []
    for i, w in enumerate(words):
        txt = clean_word(w["w"])
        if not txt:
            continue
        ww = dict(w, t=txt)
        if cur:
            gap = w["s"] - cur[-1]["e"]
            chars = sum(len(x["t"]) for x in cur) + len(txt) + len(cur)
            if len(cur) >= max_words or chars > max_chars or gap > 0.4 or re.search(r"[.!?,]$", cur[-1]["w"]):
                groups.append(cur)
                cur = []
        cur.append(ww)
    if cur:
        groups.append(cur)
    return groups


def _text_layer(parts: list[tuple[str, bool]], size: int, preset: str = "pop_glow") -> Image.Image:
    """Renders one line of words with active word glowing and kinetic pop."""
    style = STYLES.get(preset, STYLES["pop_glow"])
    act_col = style["active_color"]
    inact_col = style["inactive_color"]
    glow_out_col = style["glow_outer"]
    glow_in_col = style["glow_inner"]
    pop_y = style.get("pop_offset_y", -6)

    f = font(size)
    space = f.getlength(" ")
    widths = [f.getlength(t) for t, _ in parts]
    total = sum(widths) + space * (len(parts) - 1)
    
    img = Image.new("RGBA", (CAP_W, CAP_H), (0, 0, 0, 0))
    shadow = Image.new("RGBA", (CAP_W, CAP_H), (0, 0, 0, 0))
    motion_layer = Image.new("RGBA", (CAP_W, CAP_H), (0, 0, 0, 0))
    glow_outer = Image.new("RGBA", (CAP_W, CAP_H), (0, 0, 0, 0)) if glow_out_col else None
    glow_inner = Image.new("RGBA", (CAP_W, CAP_H), (0, 0, 0, 0)) if glow_in_col else None
    
    d = ImageDraw.Draw(img)
    ds = ImageDraw.Draw(shadow)
    dm = ImageDraw.Draw(motion_layer)
    dgo = ImageDraw.Draw(glow_outer) if glow_outer else None
    dgi = ImageDraw.Draw(glow_inner) if glow_inner else None
    
    x = (CAP_W - total) / 2
    asc, desc = f.getmetrics()
    y = (CAP_H - (asc + desc)) / 2
    stroke = max(6, size // 10)
    
    for (t, is_active), w in zip(parts, widths):
        cur_y = y + pop_y if is_active else y
        
        # Deep Gaussian drop shadow behind all words
        ds.text((x + 4, cur_y + 8), t, font=f, fill=(0, 0, 0, 180), stroke_width=stroke + 4, stroke_fill=(0, 0, 0, 180))
        
        if is_active:
            # Kinetic motion blur streaks (vertical smear simulating high-speed pop entry)
            for dy, a_mult in [(5, 0.30), (3, 0.55), (1, 0.75), (-2, 0.35)]:
                c = (*act_col[:3], int(act_col[3] * a_mult))
                dm.text((x, cur_y + dy), t, font=f, fill=c, stroke_width=stroke + 2, stroke_fill=(0, 0, 0, int(160 * a_mult)))

            # Dual-layer radiant neon aura on the active word
            if dgo and glow_out_col:
                dgo.text((x, cur_y), t, font=f, fill=glow_out_col, stroke_width=stroke + 24, stroke_fill=glow_out_col)
            if dgi and glow_in_col:
                dgi.text((x, cur_y), t, font=f, fill=glow_in_col, stroke_width=stroke + 12, stroke_fill=glow_in_col)
            d.text((x, cur_y), t, font=f, fill=act_col, stroke_width=stroke, stroke_fill=BLACK)
        else:
            # Soft ambient light on inactive text
            if dgo and glow_out_col:
                dgo.text((x, cur_y), t, font=f, fill=(255, 255, 255, 50), stroke_width=stroke + 14, stroke_fill=(255, 255, 255, 50))
            d.text((x, cur_y), t, font=f, fill=inact_col, stroke_width=stroke, stroke_fill=BLACK)
            
        x += w + space
        
    shadow = shadow.filter(ImageFilter.GaussianBlur(8))
    motion_layer = motion_layer.filter(ImageFilter.GaussianBlur(3))
    res = shadow
    if glow_outer:
        glow_outer = glow_outer.filter(ImageFilter.GaussianBlur(16))
        res = Image.alpha_composite(res, glow_outer)
    if glow_inner:
        glow_inner = glow_inner.filter(ImageFilter.GaussianBlur(6))
        res = Image.alpha_composite(res, glow_inner)
    res = Image.alpha_composite(res, motion_layer)
    res = Image.alpha_composite(res, img)
    return res


def fit_size(texts: list[str], start: int = 100, max_w: int = 990) -> int:
    size = start
    while size > 44:
        f = font(size)
        if f.getlength(" ".join(texts)) + size * 0.25 <= max_w:
            return size
        size -= 4
    return size


def render_captions(words: list[dict], duration: float, folder: Path, preset: str = "pop_glow") -> Path:
    """PNG for every caption state + an ffconcat list timed to the words. Returns the list's path."""
    folder.mkdir(parents=True, exist_ok=True)
    for f in folder.glob("cap_*.png"):
        f.unlink()
    blank = folder / "cap_blank.png"
    Image.new("RGBA", (CAP_W, CAP_H), (0, 0, 0, 0)).save(blank)
    groups = group_words(words)
    events: list[tuple[float, float, Path]] = []
    k = 0
    for gi, g in enumerate(groups):
        size = fit_size([w["t"] for w in g])
        g_end = g[-1]["e"] + 0.25
        if gi + 1 < len(groups):
            g_end = min(g_end, groups[gi + 1][0]["s"])
        for wi, w in enumerate(g):
            parts = [(x["t"], j == wi) for j, x in enumerate(g)]
            p = folder / f"cap_{k:04d}.png"
            _text_layer(parts, size, preset=preset).save(p, optimize=False, compress_level=1)
            k += 1
            s = g[0]["s"] if wi == 0 else w["s"]
            e = g[wi + 1]["s"] if wi + 1 < len(g) else g_end
            events.append((max(0.0, s), max(s + 0.04, e), p))
    lines = ["ffconcat version 1.0"]
    t = 0.0
    for s, e, p in events:
        if s > t + 0.02:
            lines += [f"file '{blank.name}'", f"duration {s - t:.3f}"]
            t = s
        e = min(e, duration)
        if e - t < 0.02:
            continue
        lines += [f"file '{p.name}'", f"duration {e - t:.3f}"]
        t = e
    if t < duration:
        lines += [f"file '{blank.name}'", f"duration {duration - t + 0.5:.3f}"]
    lines += [f"file '{blank.name}'"]
    lst = folder / "captions.ffconcat"
    lst.write_text("\n".join(lines) + "\n")
    return lst


def hook_card(text: str, path: Path, max_w: int = 960) -> tuple[int, int]:
    """White rounded box, black heavy type: the line that makes people stop scrolling."""
    words = text.strip().split()
    size = 66
    f = font(size)
    lines: list[str] = []
    while True:
        lines, cur = [], ""
        for w in words:
            tryline = (cur + " " + w).strip()
            if f.getlength(tryline) <= max_w - 80:
                cur = tryline
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        if len(lines) <= 3 or size <= 44:
            break
        size -= 4
        f = font(size)
    asc, desc = f.getmetrics()
    lh = int((asc + desc) * 1.08)
    w = int(max(f.getlength(l) for l in lines)) + 80
    h = lh * len(lines) + 56
    img = Image.new("RGBA", (w + 24, h + 24), (0, 0, 0, 0))
    sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle((12, 18, w + 12, h + 18), 28, fill=(0, 0, 0, 120))
    sh = sh.filter(ImageFilter.GaussianBlur(8))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((12, 12, w + 12, h + 12), 28, fill=(255, 255, 255, 255))
    y = 12 + 28
    for l in lines:
        lw = f.getlength(l)
        d.text((12 + (w - lw) / 2, y), l, font=f, fill=(12, 12, 12, 255))
        y += lh
    Image.alpha_composite(sh, img).save(path)
    return img.size


def credit_tag(text: str, path: Path) -> tuple[int, int]:
    f = font(30)
    w = int(f.getlength(text)) + 40
    h = 54
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, w - 1, h - 1), 27, fill=(0, 0, 0, 150))
    d.text((20, (h - sum(f.getmetrics())) / 2), text, font=f, fill=(255, 255, 255, 235))
    img.save(path)
    return img.size
