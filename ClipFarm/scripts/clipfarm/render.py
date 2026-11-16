"""cf render: from the final round to ready-to-post 1080x1920 clips."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

from . import captions, face, fetch, listen, scenes
from .util import (LEDGER, WORK, hms, parse_t, read_json, run, say, settings, slug, stream_dir, today, warn,
                   write_json)

SR = 16000


# ------------------------------------------------------------------ audio helpers

def decode_audio(path: Path, start: float | None = None, dur: float | None = None) -> np.ndarray:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if start is not None:
        cmd += ["-ss", f"{max(0, start):.3f}"]
    cmd += ["-i", str(path)]
    if dur is not None:
        cmd += ["-t", f"{dur:.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"]
    p = subprocess.run(cmd, capture_output=True)
    return np.frombuffer(p.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def read_wav(wav: Path, start: float, dur: float) -> np.ndarray:
    with wave.open(str(wav), "rb") as w:
        sr = w.getframerate()
        a = max(0, int(start * sr))
        w.setpos(min(a, w.getnframes()))
        raw = w.readframes(int(dur * sr))
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def align(section_audio: np.ndarray, stream_wav: Path, approx_start: float, search: float = 12.0) -> tuple[float, float]:
    """Where exactly the downloaded section starts in the stream (seconds), by matching the sound."""
    probe = section_audio[: SR * 20]
    if len(probe) < SR * 2:
        return approx_start, 0.0
    ref_start = max(0.0, approx_start - search)
    ref = read_wav(stream_wav, ref_start, len(probe) / SR + 2 * search)
    if len(ref) < len(probe):
        return approx_start, 0.0
    # same source, two encodes: the waveforms themselves line up, so correlate them at 4 kHz (0.25 ms steps)
    def dec(x):
        m = len(x) // 4
        y = x[: m * 4].reshape(m, 4).mean(axis=1)
        return (y - y.mean()) / (y.std() + 1e-9)
    p, r = dec(probe), dec(ref)
    nfft = 1 << int(np.ceil(np.log2(len(p) + len(r))))
    c = np.fft.irfft(np.fft.rfft(r, nfft) * np.conj(np.fft.rfft(p, nfft)), nfft)[: len(r) - len(p) + 1]
    k = int(np.argmax(c))
    conf = float((c[k] - c.mean()) / (c.std() + 1e-9))
    return ref_start + k / 4000.0, conf


def frames_db(x: np.ndarray, hop: float = 0.02) -> np.ndarray:
    n = int(SR * hop)
    m = len(x) // n
    if m == 0:
        return np.array([-90.0])
    r = np.sqrt(np.mean(x[: m * n].reshape(m, n) ** 2, axis=1) + 1e-10)
    return 20 * np.log10(r)


def loudness(path: Path) -> float:
    p = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-vn", "-af", "ebur128=peak=true",
                        "-f", "null", "-"], capture_output=True, text=True, errors="replace")
    vals = re.findall(r"I:\s+(-?[\d.]+) LUFS", p.stderr)
    return float(vals[-1]) if vals else -99.0


def probe_video(path: Path) -> dict:
    p = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height,r_frame_rate:format=duration", "-of", "json", str(path)])
    d = json.loads(p.stdout)
    st = (d.get("streams") or [{}])[0]
    num, den = (st.get("r_frame_rate") or "30/1").split("/")
    fps = float(num) / max(1.0, float(den))
    return {"w": st.get("width"), "h": st.get("height"), "fps": fps,
            "dur": float((d.get("format") or {}).get("duration") or 0)}


# ------------------------------------------------------------------ the clip

def plan_cut(words: list[dict], db: np.ndarray, a: float, b: float, cap: float, trim_air: bool = True):
    """Clean start and end on whole words, and take the dead air out of the middle."""
    hop = 0.02
    inside = [w for w in words if w["e"] > a and w["s"] < b]
    if inside:
        if inside[0]["s"] < a < inside[0]["e"]:
            a = inside[0]["s"] - 0.08
        if inside[-1]["s"] < b < inside[-1]["e"]:
            b = inside[-1]["e"] + 0.12
    # speech level for this clip
    sp = [db[int(w["s"] / hop):int(w["e"] / hop) + 1] for w in inside]
    sp = np.concatenate(sp) if sp else db
    speech = float(np.median(sp)) if len(sp) else -30.0
    quiet = max(-50.0, speech - 18.0)

    def quiet_between(x0, x1):
        seg = db[int(max(0, x0) / hop):int(max(0, x1) / hop)]
        return len(seg) > 0 and float(np.mean(seg < quiet)) > 0.8

    if inside:
        fw, lw = inside[0], inside[-1]
        if fw["s"] - a > 0.45 and quiet_between(a, fw["s"]):
            a = fw["s"] - 0.12
        if b - lw["e"] > 1.0 and quiet_between(lw["e"], b):
            b = lw["e"] + 0.5
    keep = [(a, b)]
    if trim_air:
        i0, i1 = int(a / hop), int(b / hop)
        q = db[i0:i1] < quiet
        runs, start = [], None
        for k, v in enumerate(q):
            if v and start is None:
                start = k
            if (not v or k == len(q) - 1) and start is not None:
                end = k if not v else k + 1
                if (end - start) * hop >= 0.6:
                    runs.append((a + start * hop, a + end * hop))
                start = None
        keep = []
        cur = a
        for r0, r1 in runs:
            if r0 - cur > 0.05:
                keep.append((cur, r0 + 0.12))
            cur = r1 - 0.12
        if b - cur > 0.05:
            keep.append((cur, b))
    # cap the length on a word boundary
    total = sum(y - x for x, y in keep)
    if total > cap:
        budget, out = cap, []
        for x, y in keep:
            if y - x <= budget:
                out.append((x, y))
                budget -= y - x
            else:
                end = x + budget
                ends = [w["e"] for w in inside if x < w["e"] <= end]
                out.append((x, (max(ends) + 0.15) if ends else end))
                break
        keep = out
    return keep, speech


def remap(words: list[dict], keep: list[tuple[float, float]]) -> list[dict]:
    out, offset = [], 0.0
    for x, y in keep:
        for w in words:
            if w["e"] > x and w["s"] < y:
                s_new = offset + max(0.0, w["s"] - x)
                e_new = offset + min(y - x, w["e"] - x)
                if e_new - s_new > 0.03:
                    out.append({**w, "s": round(s_new, 2), "e": round(e_new, 2)})
        offset += y - x
    out.sort(key=lambda w: w["s"])
    return out


def source_link(info: dict, t: float) -> str:
    url = info.get("url") or ""
    t = int(t)
    vid_id = str(info.get("id", "")).lstrip("v")
    if info.get("platform") == "youtube":
        return f"https://www.youtube.com/watch?v={vid_id}&t={t}s" if vid_id else url
    if info.get("platform") == "twitch":
        h, m, s = t // 3600, (t % 3600) // 60, t % 60
        return f"https://www.twitch.tv/videos/{vid_id}?t={h}h{m}m{s}s" if vid_id else url
    return url


def render_one(info: dict, d: Path, clip: dict, idx: int, out_dir: Path, layout: str | None = None,
               section: tuple[Path, float] | None = None, stream_wav: Path | None = None) -> dict:
    cfg = settings()
    s, e = parse_t(clip["start"]), parse_t(clip["end"])
    cid = clip.get("id") or f"c{idx:02d}"
    work = d / "render" / cid
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 1. only the minutes this clip needs
    if section:
        sec, T0 = section
        conf = 99.0
    elif (d / "live_stream.mp4").exists():
        sec = d / "sections" / f"{cid}.mp4"
        sec.parent.mkdir(parents=True, exist_ok=True)
        t_start = max(0.0, s - 4)
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{t_start:.2f}", "-to", f"{e + 4:.2f}",
             "-i", str(d / "live_stream.mp4"), "-c", "copy", "-movflags", "+faststart", str(sec)], check=False)
        T0 = t_start
        conf = 99.0
    else:
        sec = fetch.download_section(info["url"], s - 4, e + 4, d / "sections" / f"{cid}.mp4")
        sec_audio = decode_audio(sec)
        ref_wav = stream_wav if stream_wav and Path(stream_wav).exists() else (d / "audio16k.wav" if (d / "audio16k.wav").exists() else None)
        if ref_wav:
            T0, conf = align(sec_audio, ref_wav, s - 4)
            if conf < 6:
                warn(f"{cid}: couldn't match the section's sound exactly (conf {conf:.1f}); using the requested time")
                T0 = s - 4
        else:
            T0 = s - 4
    pv = probe_video(sec)
    a, b = s - T0, e - T0
    a, b = max(0.0, a), min(pv["dur"] - 0.05, b)
    full = decode_audio(sec)
    db = frames_db(full)

    # 2. word timings for the captions, from the clip's own audio
    lead = max(0.0, a - 0.6)
    cw = work / "clip16k.wav"
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{lead:.3f}", "-i", str(sec), "-t",
         f"{b - lead + 0.8:.3f}", "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(cw)])
    words = [{**w, "s": w["s"] + lead, "e": w["e"] + lead} for w in listen.clip_words(cw)]

    # 3. clean edges, no dead air, under the length cap
    keep, speech_db = plan_cut(words, db, a, b, float(cfg["max_clip_s"]), trim_air=clip.get("trim_air", True))
    L = sum(y - x for x, y in keep)
    source_words = clip.get("custom_words") or words
    out_words = remap(source_words, keep)
    try:
        (work / "words.json").write_text(json.dumps(out_words, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

    # 4. where's the facecam, scene by scene (streams switch between full camera and gameplay)
    sa = scenes.analyze(sec, keep[0][0], keep[-1][1], work / "facecam.jpg")
    want = layout or clip.get("layout")
    if want:
        if want in ("blur", "peekcut", "horizontal"):
            want = "blur"
            sa["scenes"] = [{"start": keep[0][0], "end": keep[-1][1], "layout": "blur"}]
            sa["why"] = "blur (PeekCut mode)"
        else:
            faces = [sc for sc in sa["scenes"] if sc.get("face")]
            if want in ("stack", "fullcam") and not faces:
                warn(f"{cid}: no face found, so {want} isn't possible; using gameplay")
                want = "gameplay"
            base = dict(faces[0]) if faces else {}
            base.update({"start": keep[0][0], "end": keep[-1][1], "layout": want})
            if want == "stack":
                base["cam"] = next((sc["cam"] for sc in sa["scenes"] if sc.get("cam")), None)
            sa["scenes"] = [base]
            sa["why"] = f"{want} (asked for)"
    W, H = sa["W"], sa["H"]
    # the pieces: every kept stretch, cut again wherever the scene changes
    pieces = []
    for x, y in keep:
        for sc in sa["scenes"]:
            lo, hi = max(x, sc["start"] if sc is not sa["scenes"][0] else x), min(y, sc["end"] if sc is not sa["scenes"][-1] else y)
            if hi - lo > 0.04:
                pieces.append((lo, hi, sc))
    lays = [p[2]["layout"] for p in pieces]
    lay = lays[0] if len(set(lays)) == 1 else "mixed"

    # 5. the graphics
    caplist = captions.render_captions(out_words, L, work / "caps", preset=clip.get("preset", "pop_glow"))
    hook = (clip.get("hook") or "").strip()
    hook_png = work / "hook.png"
    hw, hh = captions.hook_card(hook, hook_png) if hook else (0, 0)
    credit_png = work / "credit.png"
    who = clip.get("streamer") or info.get("streamer") or ""
    plat_name = {"twitch": "Twitch", "youtube": "YouTube", "kick": "Kick"}.get(info.get("platform", ""))
    if cfg.get("credit", True) and who:
        crw, crh = captions.credit_tag(f"{who} on {plat_name}" if plat_name else who, credit_png)
    else:
        crw = crh = 0
    CAP_Y = {
        "stack": face.CAM_H - captions.CAP_H // 2,
        "fullcam": 1180,
        "gameplay": 656 + 608 + 80,
        "blur": 1460,
        "peekcut": 1460
    }
    HOOK_Y = {
        "stack": face.CAM_H + 70,
        "fullcam": 250,
        "gameplay": max(160, 656 - hh - 30),
        "blur": 360,
        "peekcut": 360
    }
    cred = (24, 150)
    hook_y = int(clip.get("hook_y")) if clip.get("hook_y") is not None else HOOK_Y.get(lays[0], 360)
    custom_cap_y = clip.get("cap_y")
    if custom_cap_y is not None:
        cap_expr = str(int(custom_cap_y))
    else:
        # caption height follows the layout of whatever is on screen
        spans, t = [], 0.0
        for lo, hi, sc in pieces:
            y_ = CAP_Y.get(sc["layout"], 1460)
            if spans and spans[-1][2] == y_:
                spans[-1][1] = t + hi - lo
            else:
                spans.append([t, t + hi - lo, y_])
            t += hi - lo
        cap_expr = str(spans[-1][2])
        for t0_, t1_, y_ in reversed(spans[:-1]):
            cap_expr = f"if(lt(t\\,{t1_:.3f})\\,{y_}\\,{cap_expr})"

    # 6. one ffmpeg pass
    fps = min(float(cfg["fps_cap"]), pv["fps"] if pv["fps"] > 1 else 30.0)
    K = len(pieces)
    fc = [f"[0:v]split={K}" + "".join(f"[vs{i}]" for i in range(K)) if K > 1 else "[0:v]null[vs0]",
          f"[0:a]asplit={K}" + "".join(f"[as{i}]" for i in range(K)) if K > 1 else "[0:a]anull[as0]"]
    for i, (x, y, sc) in enumerate(pieces):
        dur = y - x
        cr = scenes.crops_for(sc, W, H)
        head = f"[vs{i}]trim=start={x:.3f}:end={y:.3f},setpts=PTS-STARTPTS"
        zoom = max(1.0, min(2.5, float(clip.get("gameplay_zoom", 1.0))))
        pan_x = max(-1.0, min(1.0, float(clip.get("gameplay_pan_x", 0.0))))
        cam_zoom = max(1.0, min(2.5, float(clip.get("facecam_zoom", 1.0))))
        cam_pan_x = max(-1.0, min(1.0, float(clip.get("facecam_pan_x", 0.0))))
        if sc["layout"] == "stack" and cr.get("cam"):
            cx, cy, cw_, ch_ = cr["cam"]
            gx, gy, gw, gh = cr["game"]

            # Facecam independent framing
            if cam_zoom > 1.001 or abs(cam_pan_x) > 0.01:
                cw_z = int(cw_ / cam_zoom) // 2 * 2
                ch_z = int(ch_ / cam_zoom) // 2 * 2
                c_shift_x = cam_pan_x * ((cw_ - cw_z) / 2 if cam_zoom > 1.0 else 0.1 * W)
                cx_z = int(max(0, min(W - cw_z, cx + (cw_ - cw_z) / 2 + c_shift_x))) // 2 * 2
                cy_z = int(max(0, min(H - ch_z, cy + (ch_ - ch_z) / 2))) // 2 * 2
                cam_crop_cmd = f"crop={cw_z}:{ch_z}:{cx_z}:{cy_z},scale=1080:{face.CAM_H}:flags=lanczos,setsar=1"
            else:
                cam_crop_cmd = f"crop={cw_}:{ch_}:{cx}:{cy},scale=1080:{face.CAM_H}:flags=lanczos,setsar=1"

            # Gameplay independent framing
            if zoom > 1.001 or abs(pan_x) > 0.01:
                gw_z = int(gw / zoom) // 2 * 2
                gh_z = int(gh / zoom) // 2 * 2
                shift_x = pan_x * ((gw - gw_z) / 2 if zoom > 1.0 else 0.1 * W)
                gx_z = int(max(0, min(W - gw_z, gx + (gw - gw_z) / 2 + shift_x))) // 2 * 2
                gy_z = int(max(0, min(H - gh_z, gy + (gh - gh_z) / 2))) // 2 * 2
                game_crop_cmd = f"crop={gw_z}:{gh_z}:{gx_z}:{gy_z},scale=1080:{face.GAME_H}:flags=lanczos,setsar=1"
            else:
                game_crop_cmd = f"crop={gw}:{gh}:{gx}:{gy},scale=1080:{face.GAME_H}:flags=lanczos,setsar=1"

            fc.append(f"{head},split=2[pa{i}][pb{i}]")
            fc.append(f"[pa{i}]{cam_crop_cmd}[pt{i}]")
            fc.append(f"[pb{i}]{game_crop_cmd}[pu{i}]")
            fc.append(f"[pt{i}][pu{i}]vstack=inputs=2,format=yuv420p,setsar=1[v{i}]")
        elif sc["layout"] == "fullcam" and cr.get("full"):
            x_, y_, w_, h_ = cr["full"]
            fc.append(f"{head},crop={w_}:{h_}:{x_}:{y_},scale=1080:1920:flags=lanczos,format=yuv420p,setsar=1[v{i}]")
        elif sc["layout"] in ("blur", "peekcut"):
            # PeekCut layout: blurred 16:9 full background + center 16:9 gameplay stream (1080x608 at y=656)
            fc.append(f"{head},split=2[pa{i}][pb{i}]")
            fc.append(f"[pa{i}]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,boxblur=6:2,"
                      f"scale=1080:1920,eq=brightness=-0.14:saturation=1.15[pg{i}]")
            if zoom > 1.001 or abs(pan_x) > 0.01:
                cw_src = int(W / zoom) // 2 * 2
                ch_src = int(H / zoom) // 2 * 2
                max_shift_x = (W - cw_src) / 2
                cx_src = int((W - cw_src) / 2 + pan_x * max_shift_x) // 2 * 2
                cx_src = max(0, min(W - cw_src, cx_src))
                cy_src = int((H - ch_src) / 2) // 2 * 2
                fc.append(f"[pb{i}]crop={cw_src}:{ch_src}:{cx_src}:{cy_src},scale=1080:-2:flags=lanczos,setsar=1[pf{i}]")
            else:
                fc.append(f"[pb{i}]scale=1080:-2:flags=lanczos,setsar=1[pf{i}]")
            fc.append(f"[pg{i}][pf{i}]overlay=0:(H-h)/2,format=yuv420p,setsar=1[v{i}]")
        else:
            fc.append(f"{head},split=2[pa{i}][pb{i}]")
            fc.append(f"[pa{i}]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,boxblur=6:2,"
                      f"scale=1080:1920,eq=brightness=-0.14:saturation=1.15[pg{i}]")
            fc.append(f"[pb{i}]scale=1080:-2:flags=lanczos,setsar=1[pf{i}]")
            fc.append(f"[pg{i}][pf{i}]overlay=0:(H-h)/2,format=yuv420p,setsar=1[v{i}]")
        fc.append(f"[as{i}]atrim=start={x:.3f}:end={y:.3f},asetpts=PTS-STARTPTS,"
                  f"afade=t=in:d=0.012,afade=t=out:st={max(0, dur - 0.012):.3f}:d=0.012[a{i}]")
    fc.append("".join(f"[v{i}][a{i}]" for i in range(K)) + f"concat=n={K}:v=1:a=1[base][ac]")
    fc.append(f"[1:v]format=rgba[cap];[base][cap]overlay=x='6*sin(2.4*t)':y='({cap_expr})+4*cos(1.8*t)':eval=frame:eof_action=pass[o1]")
    last = "o1"
    inputs = ["-i", str(sec), "-f", "concat", "-safe", "0", "-i", str(caplist)]
    n_in = 2
    if hook:
        hook_dur = float(clip.get("hook_duration", 3.5))
        hook_start = max(0.0, float(clip.get("hook_start", 0.0)))
        hook_dur = max(0.5, min(L, hook_dur))
        hook_end = min(L, hook_start + hook_dur)
        inputs += ["-loop", "1", "-t", f"{L + 0.5:.2f}", "-i", str(hook_png)]
        fc.append(f"[{n_in}:v]format=rgba[hk];[{last}][hk]overlay=x='(W-w)/2+4*sin(2.0*t)':y='{hook_y}+3*cos(2.2*t)':eval=frame:enable='between(t,{hook_start:.2f},{hook_end:.2f})'[o2]")
        last = "o2"
        n_in += 1
    if crw:
        inputs += ["-loop", "1", "-t", f"{L + 0.5:.2f}", "-i", str(credit_png)]
        fc.append(f"[{n_in}:v]format=rgba[cr];[{last}][cr]overlay={cred[0]}:{cred[1]}[o3]")
        last = "o3"
        n_in += 1
    gain = max(-12.0, min(18.0, -14.0 - _measure_keep(sec, keep)))
    fc.append(f"[ac]volume={gain:.2f}dB,alimiter=limit=0.89:level=false,aresample=48000[aout]")
    cid_prefix = re.sub(r"[^\w]", "", cid)
    name = f"{cid_prefix}_{slug(who, 18)}_{slug(hook or clip.get('title', cid), 34)}"
    out = out_dir / f"{name}.mp4"
    out_dir.mkdir(parents=True, exist_ok=True)
    script = work / "filter.txt"
    script.write_text(";\n".join(fc))
    v_codec = ["-c:v", "h264_nvenc", "-preset", "p4", "-b:v", "5M", "-maxrate", "7M", "-bufsize", "10M", "-g", "30", "-keyint_min", "30"] if sys.platform == "win32" else [
        "-c:v", "h264_videotoolbox", "-b:v", "6M", "-maxrate", "8M", "-g", "30", "-profile:v", "high"
    ] if sys.platform == "darwin" else ["-c:v", "libx264", "-crf", "20", "-preset", "fast", "-g", "30", "-keyint_min", "30"]

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"] + inputs + [
        "-filter_complex_script", str(script), "-map", f"[{last}]", "-map", "[aout]", "-r", f"{fps:.3f}"] + v_codec + [
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-t", f"{L:.3f}", str(out)]
    p = run(cmd, check=False, timeout=1200)
    if p.returncode != 0:
        # fallback to libx264 or alt filter_complex arg
        cmd_cpu = [c if c != "h264_nvenc" and c != "h264_videotoolbox" else "libx264" for c in cmd]
        p2 = run(cmd_cpu, check=False, timeout=1200)
        if p2.returncode != 0:
            cmd_alt = [c if c != "-filter_complex_script" else "-/filter_complex" for c in cmd_cpu]
            p3 = run(cmd_alt, check=False, timeout=1200)
            if p3.returncode != 0:
                raise RuntimeError("ffmpeg failed:\n" + (p.stderr or "")[-800:] + (p2.stderr or "")[-800:] + (p3.stderr or "")[-800:])

    # 7. its own check, a cover frame and the post kit
    qc = check_clip(out, L, out_words, sa)
    if -40 < qc["lufs"] and abs(qc["lufs"] + 14) > 0.8:
        # the limiter shaved too much on a stream with loud spikes: one more pass on the sound only
        fix = out.with_suffix(".fix.mp4")
        p3 = run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(out), "-c:v", "copy", "-af",
                  f"volume={-14 - qc['lufs']:.2f}dB,alimiter=limit=0.89:level=false", "-c:a", "aac", "-b:a", "192k",
                  "-movflags", "+faststart", str(fix)], check=False)
        if p3.returncode == 0 and fix.exists():
            fix.replace(out)
            qc = check_clip(out, L, out_words, sa)
    peak = clip.get("peak")
    cover_t = max(0.2, min(L - 0.2, (parse_t(peak) - s) if peak else L * 0.35))
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{cover_t:.2f}", "-i", str(out),
         "-frames:v", "1", "-q:v", "3", str(out.with_suffix(".jpg"))], check=False)
    kit = post_kit(info, clip, s, L, out)
    out.with_suffix(".txt").write_text(kit)
    meta = {"id": cid, "file": out.name, "stream": d.name, "streamer": who, "platform": info.get("platform"),
            "title": info.get("title"), "start": round(T0 + keep[0][0], 2), "end": round(T0 + keep[-1][1], 2),
            "length": round(L, 2), "hook": hook, "hook_duration": float(clip.get("hook_duration", 3.5)),
            "hook_start": float(clip.get("hook_start", 0.0)),
            "hook_y": hook_y, "cap_y": int(custom_cap_y) if custom_cap_y is not None else CAP_Y.get(lays[0], 1460),
            "gameplay_zoom": float(clip.get("gameplay_zoom", 1.0)),
            "gameplay_pan_x": float(clip.get("gameplay_pan_x", 0.0)),
            "preset": clip.get("preset", "pop_glow"),
            "type": clip.get("type", ""), "score": clip.get("score"),
            "why": clip.get("why", ""), "layout": lay, "layout_why": sa.get("why"), "cuts": len(keep) - 1,
            "words": len(out_words), "words_list": out_words, "qc": qc, "source": source_link(info, T0 + keep[0][0]),
            "made": time.strftime("%Y-%m-%d %H:%M"), "seconds_to_make": round(time.time() - t0, 1),
            "caption": clip.get("caption", ""), "hashtags": clip.get("hashtags", []), "music": clip.get("music")}
    (out.with_suffix(".json")).write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    try:
        (out.with_suffix(".words.json")).write_text(json.dumps(out_words, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    try:
        import shutil
        shutil.copy(work / "facecam.jpg", out_dir / f"{name}.facecam.jpg")
    except Exception:
        pass
    return meta


def _measure_keep(sec: Path, keep) -> float:
    """Integrated loudness of exactly the parts we keep, measured on the original (stereo) audio."""
    parts = "".join(f"[0:a]atrim=start={a:.3f}:end={b:.3f},asetpts=PTS-STARTPTS[m{i}];" for i, (a, b) in enumerate(keep))
    fc = parts + "".join(f"[m{i}]" for i in range(len(keep))) + f"concat=n={len(keep)}:v=0:a=1,ebur128[out]"
    p = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(sec), "-filter_complex", fc, "-map", "[out]",
                        "-f", "null", "-"], capture_output=True, text=True, errors="replace")
    vals = re.findall(r"I:\s+(-?[\d.]+) LUFS", p.stderr)
    v = float(vals[-1]) if vals else -99.0
    return v if v > -70 else -23.0


def check_clip(out: Path, L: float, words: list[dict], fa: dict) -> dict:
    pv = probe_video(out)
    lufs = loudness(out)
    p = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(out), "-vf", "blackdetect=d=0.4:pix_th=0.06",
                        "-an", "-f", "null", "-"], capture_output=True, text=True, errors="replace")
    blacks = len(re.findall(r"black_start", p.stderr))
    issues = []
    if not (12 <= pv["dur"] <= 61):
        issues.append(f"length {pv['dur']:.1f} s")
    if abs(lufs + 14) > 1.5:
        issues.append(f"loudness {lufs:.1f} LUFS")
    if blacks:
        issues.append(f"{blacks} black stretch(es)")
    if pv["w"] != 1080 or pv["h"] != 1920:
        issues.append(f"size {pv['w']}x{pv['h']}")
    if len(words) < max(4, L * 0.6):
        issues.append("very few words: check the captions")
    if words and words[0]["s"] > 1.2:
        issues.append(f"first word at {words[0]['s']:.1f} s")
    if any(sc["layout"] == "stack" for sc in fa.get("scenes", [])) and fa.get("cam_edges", 0) <= 1:
        issues.append("facecam box guessed (look at the top panel)")
    return {"ok": not issues, "issues": issues, "lufs": round(lufs, 1), "seconds": round(pv["dur"], 2),
            "fps": round(pv["fps"], 2)}


def post_kit(info: dict, clip: dict, s: float, L: float, out: Path) -> str:
    who = clip.get("streamer") or info.get("streamer", "")
    plat = (info.get("platform") or "").title()
    tags = clip.get("hashtags") or settings()["hashtags"]
    tags = [t if t.startswith("#") else f"#{t}" for t in tags][:6]
    cap = (clip.get("caption") or clip.get("hook") or "").strip()
    src = source_link(info, s)
    lines = [
        f"HOOK (on screen): {clip.get('hook', '')}",
        "",
        "— TikTok / Instagram Reels —",
        f"{cap}",
        f"🎥 {who} on {plat}",
        " ".join(tags),
        "",
        "— YouTube Shorts —",
        f"Title: {(clip.get('hook') or cap)[:95]}",
        f"Description: {cap}\n🎥 {who} on {plat}: {src}\n" + " ".join(tags[:3]),
        "",
        f"Source: {src}",
        f"Length: {L:.1f} s",
        "Before you post: watch it once, and make sure this streamer is fine with clips.",
    ]
    return "\n".join(lines) + "\n"


import shutil

def match_clip_id(a, b) -> bool:
    t = str(a).lower().replace("_", "").replace("-", "")
    o = str(b).lower().replace("_", "").replace("-", "")
    if t == o: return True
    tn, on = re.sub(r"[^\d]", "", t), re.sub(r"[^\d]", "", o)
    return bool(tn and on and int(tn) == int(on))


def render_stream(key: str, only: str | None = None, layout: str | None = None, on_progress = None) -> list[dict]:
    d = stream_dir(key)
    info = read_json(d / "info.json", {})
    final = read_json(d / "final.json")
    if not final:
        raise SystemExit(f"no final.json in {d}. Do the final round first (reference/final-round.md).")
    clips = final.get("clips") if isinstance(final, dict) else final
    out_dir = WORK / "clips" / today()
    out_dir.mkdir(parents=True, exist_ok=True)
    target_clips = d / "clips"
    target_clips.mkdir(parents=True, exist_ok=True)
    made = []
    
    total_clips = len(clips)
    for i, c in enumerate(clips, 1):
        cid = c.get("id") or f"c{i:02d}"
        c["id"] = cid
        if only:
            only_list = [o.strip() for o in only.split(",")]
            def _match(a, b):
                t = str(a).lower().replace("_", "").replace("-", "")
                o = str(b).lower().replace("_", "").replace("-", "")
                if t == o: return True
                tn, on = re.sub(r"[^\d]", "", t), re.sub(r"[^\d]", "", o)
                return bool(tn and on and int(tn) == int(on))
            if not any(_match(cid, o) for o in only_list):
                continue
        hook_name = c.get('hook') or c.get('title') or f"Clip {i}"
        say(f"→ clip {i}/{total_clips}: {hook_name}  ({c['start']}–{c['end']})")
        if on_progress:
            try:
                on_progress(i, total_clips, hook_name, "rendering", None)
            except Exception:
                pass
        try:
            m = render_one(info, d, c, _next_index(out_dir), out_dir, layout)
        except Exception as ex:
            warn(f"{cid} failed: {ex}")
            continue
        made.append(m)

        # IMMEDIATELY copy rendered clip to stream folder so the dashboard displays it LIVE!
        try:
            clip_file = m.get("file")
            if clip_file:
                c_stem = Path(clip_file).stem
                # Clean up any obsolete previous clip files for this cid in target_clips
                for old_f in list(target_clips.glob("*")):
                    if (match_clip_id(old_f.stem.split("_")[0], cid) or old_f.stem.startswith(f"{cid}_")) and not old_f.name.startswith(c_stem):
                        try: old_f.unlink()
                        except Exception: pass
                for item in out_dir.glob(f"{c_stem}*"):
                    shutil.copy2(item, target_clips / item.name)

            c["file"] = clip_file
            try:
                write_json(d / "final.json", final)
            except Exception:
                pass

            st = read_json(d / "state.json", {})
            st["rendered"] = time.time()
            existing_clips = st.get("clips", [])
            if clip_file and clip_file not in existing_clips:
                existing_clips.append(clip_file)
            st["clips"] = existing_clips
            write_json(d / "state.json", st)
        except Exception as cp_err:
            warn(f"Failed to copy live clip {cid}: {cp_err}")

        if on_progress:
            try:
                on_progress(i, total_clips, hook_name, "done_clip", m)
            except Exception:
                pass

        flag = "✓" if m["qc"]["ok"] else "⚠ " + "; ".join(m["qc"]["issues"])
        say(f"  {flag}  {m['file']}  {m['length']:.1f} s · {m['layout']} · {m['cuts']} dead-air cuts · "
            f"{m['qc']['lufs']} LUFS · {m['seconds_to_make']:.0f} s to make")
    _ledger_add(made)
    st = read_json(d / "state.json", {})
    st["rendered"] = time.time()
    st["clips"] = [m["file"] for m in made]
    write_json(d / "state.json", st)
    try:
        from . import review
        page = review.build(today())
        say(f"✓ {len(made)} clips in {out_dir}\n  review page: {page}")
    except Exception:
        pass
    return made


def _next_index(out_dir: Path) -> int:
    nums = [int(m.group(1)) for p in out_dir.glob("*.mp4") if (m := re.match(r"(\d+)_", p.name))]
    return (max(nums) + 1) if nums else 1


def _ledger_add(made: list[dict]) -> None:
    led = read_json(LEDGER, []) or []
    for m in made:
        led.append({k: m[k] for k in ("file", "stream", "streamer", "start", "end", "hook", "type", "made", "length")}
                   | {"date": today(), "posted": {}})
    write_json(LEDGER, led)
