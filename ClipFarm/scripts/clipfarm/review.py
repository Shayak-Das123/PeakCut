"""The review page: every clip of the day, playable, with its captions one tap from your clipboard."""
from __future__ import annotations

import html
import json
from pathlib import Path

from .util import LEDGER, WORK, hms, read_json, say, write_json


def build(date: str) -> Path:
    folder = WORK / "clips" / date
    folder.mkdir(parents=True, exist_ok=True)
    clips = []
    for meta in sorted(folder.glob("*.json")):
        if meta.name.endswith(".facecam.json"):
            continue
        m = read_json(meta, None)
        if not m or "file" not in m:
            continue
        kit = (folder / m["file"]).with_suffix(".txt")
        m["kit"] = kit.read_text() if kit.exists() else ""
        clips.append(m)
    streams = {}
    for m in clips:
        k = m.get("stream")
        if k and k not in streams:
            sd = WORK / "streams" / k
            st = read_json(sd / "state.json", {})
            streams[k] = {"spark": read_json(sd / "spark.json", []), "duration": st.get("duration", 0),
                          "streamer": st.get("streamer", ""), "title": st.get("title", ""),
                          "signals": st.get("signals", []), "scan_seconds": st.get("scan_seconds"),
                          "program": st.get("program", [])}
    page = folder / "index.html"
    page.write_text(TEMPLATE.replace("__DATA__", json.dumps({"date": date, "clips": clips, "streams": streams},
                                                              ensure_ascii=False).replace("</", "<\\/"))
                    .replace("__DATE__", html.escape(date)))
    return page


def mark_posted(clip: str, url: str) -> None:
    led = read_json(LEDGER, []) or []
    name = Path(clip).name
    hit = False
    for c in led:
        if c.get("file") == name or name in c.get("file", ""):
            plat = ("youtube" if "youtu" in url else "tiktok" if "tiktok" in url else
                    "instagram" if "instagram" in url else "other")
            c.setdefault("posted", {})[plat] = url
            hit = True
    if not hit:
        raise SystemExit(f"no clip called {name} in the ledger")
    write_json(LEDGER, led)
    say(f"✓ marked {name} as posted")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Clip Farm · __DATE__</title>
<style>
:root{--bg:#0e0f12;--card:#17191e;--line:#262a31;--ink:#f2f2f0;--dim:#9aa0a8;--acc:#ffd60a;--ok:#3ddc84;--warn:#ff9f43}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 -apple-system,BlinkMacSystemFont,"SF Pro Text",Inter,sans-serif}
header{padding:28px 24px 8px;max-width:1280px;margin:auto}h1{font-size:26px;margin:0 0 4px;letter-spacing:-.01em}
.sub{color:var(--dim)}main{max-width:1280px;margin:auto;padding:12px 24px 60px}
.stream{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px 16px;margin:14px 0}
.stream b{font-size:15px}.stream .meta{color:var(--dim);font-size:13px;margin:2px 0 8px}
.spark{width:100%;height:56px;display:block}.prog{font-size:13px;color:var(--acc);margin-top:6px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px;margin-top:16px}
.clip{background:var(--card);border:1px solid var(--line);border-radius:14px;overflow:hidden;display:flex;flex-direction:column}
.clip.posted{opacity:.55}.clip video{width:100%;aspect-ratio:9/16;background:#000;display:block}
.body{padding:12px 14px 14px;display:flex;flex-direction:column;gap:8px}.hook{font-weight:700;font-size:16px}
.tags{display:flex;flex-wrap:wrap;gap:6px}.tag{font-size:12px;padding:2px 8px;border-radius:999px;background:#22262d;color:var(--dim)}
.tag.ok{color:var(--ok)}.tag.warn{color:var(--warn)}.why{color:var(--dim);font-size:13px}
.btns{display:flex;flex-wrap:wrap;gap:6px}button,a.btn{all:unset;cursor:pointer;font-size:13px;padding:7px 10px;border-radius:9px;background:#22262d;color:var(--ink)}
button:hover,a.btn:hover{background:#2c313a}button.primary{background:var(--acc);color:#111;font-weight:600}
details{font-size:13px;color:var(--dim)}details img{width:100%;border-radius:8px;margin-top:6px}
pre{white-space:pre-wrap;font:12px/1.4 ui-monospace,Menlo,monospace;background:#111317;padding:8px;border-radius:8px;color:#cfd3d8}
.toast{position:fixed;left:50%;bottom:24px;transform:translateX(-50%);background:var(--acc);color:#111;padding:8px 14px;border-radius:10px;font-weight:600;opacity:0;transition:.2s}
.toast.on{opacity:1}
@media (max-width:520px){header,main{padding-left:16px;padding-right:16px}.grid{grid-template-columns:1fr}}
</style></head><body>
<header><h1>Clip Farm · __DATE__</h1><div class="sub" id="sum"></div></header>
<main><div id="streams"></div><div class="grid" id="grid"></div></main>
<div class="toast" id="toast">Copied</div>
<script>
const D=__DATA__;
const $=s=>document.querySelector(s);
const fmt=s=>{s=Math.max(0,Math.round(s));const h=Math.floor(s/3600),m=Math.floor(s%3600/60),x=s%60;return (h?h+':'+String(m).padStart(2,'0'):m)+':'+String(x).padStart(2,'0')};
function store(k,v){try{if(v===undefined)return localStorage.getItem(k);localStorage.setItem(k,v)}catch(e){return null}}
function copy(t){const done=()=>{const e=$('#toast');e.classList.add('on');setTimeout(()=>e.classList.remove('on'),900)};
 if(navigator.clipboard&&navigator.clipboard.writeText){navigator.clipboard.writeText(t).then(done,()=>fallback(t,done))}else fallback(t,done)}
function fallback(t,done){const a=document.createElement('textarea');a.value=t;document.body.appendChild(a);a.select();try{document.execCommand('copy');done()}catch(e){prompt('Copy:',t)}a.remove()}
function section(kit,name){const i=kit.indexOf('— '+name);if(i<0)return '';const rest=kit.slice(i).split('\n').slice(1);const out=[];for(const l of rest){if(l.startsWith('— ')||l.startsWith('Source:'))break;out.push(l)}return out.join('\n').trim()}
const total=D.clips.length, secs=D.clips.reduce((a,c)=>a+(c.length||0),0);
$('#sum').textContent=`${total} clips · ${Object.keys(D.streams).length} streams · ${Math.round(secs)} s of video. Watch each one before you post it.`;
for(const [k,s] of Object.entries(D.streams)){
 const el=document.createElement('div');el.className='stream';
 const w=1000,h=56,n=s.spark.length||1;
 const pts=s.spark.map((v,i)=>`${(i/(n-1||1))*w},${h-3-v*(h-8)}`).join(' ');
 const marks=D.clips.filter(c=>c.stream===k).map(c=>{const x=(c.start/(s.duration||1))*w;return `<line x1="${x}" x2="${x}" y1="0" y2="${h}" stroke="#ffd60a" stroke-width="3"/>`}).join('');
 el.innerHTML=`<b>${s.streamer}</b><div class="meta">${s.title||''} · ${fmt(s.duration)} · signals: ${(s.signals||[]).join(', ')||'transcript only'}${s.scan_seconds?` · scanned in ${fmt(s.scan_seconds)}`:''}</div>
 <svg class="spark" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none"><polyline points="${pts}" fill="none" stroke="#7d8590" stroke-width="2"/>${marks}</svg>
 ${(s.program&&s.program.length)?`<div class="prog">💰 clipping program mentioned: ${s.program.map(x=>x.replace(/</g,'&lt;')).join(' · ')}</div>`:''}`;
 $('#streams').appendChild(el)}
for(const c of D.clips){
 const el=document.createElement('div');el.className='clip';const pk='posted:'+c.file;if(store(pk)==='1')el.classList.add('posted');
 const qc=c.qc||{};const tags=[`<span class="tag">${c.streamer}</span>`,`<span class="tag">${(c.length||0).toFixed(1)} s</span>`,c.type?`<span class="tag">${c.type}</span>`:'',
 `<span class="tag">${c.layout}</span>`,c.cuts?`<span class="tag">${c.cuts} dead-air cuts</span>`:'',qc.ok?'<span class="tag ok">checks passed</span>':`<span class="tag warn">⚠ ${(qc.issues||[]).join('; ')}</span>`,
 c.music?'<span class="tag warn">music under it</span>':''].join('');
 const tt=section(c.kit,'TikTok'), yt=section(c.kit,'YouTube');
 el.innerHTML=`<video src="${encodeURI(c.file)}" controls preload="metadata" playsinline poster="${encodeURI(c.file.replace(/\.mp4$/,'.jpg'))}"></video>
 <div class="body"><div class="hook">${(c.hook||'').replace(/</g,'&lt;')}</div><div class="tags">${tags}</div>
 <div class="why">${(c.why||'').replace(/</g,'&lt;')}</div>
 <div class="btns"><button class="primary" data-t="tt">Copy TikTok / Reels caption</button><button data-t="yt">Copy Shorts text</button>
 <a class="btn" href="${c.source}" target="_blank" rel="noopener">Source at ${fmt(c.start)}</a><button data-t="posted">${store(pk)==='1'?'Posted ✓':'Mark posted'}</button></div>
 <details><summary>Post kit and facecam check</summary><pre>${c.kit.replace(/</g,'&lt;')}</pre><img src="${encodeURI(c.file.replace(/\.mp4$/,'.facecam.jpg'))}" alt="facecam found" onerror="this.remove()"></details></div>`;
 el.querySelector('[data-t=tt]').onclick=()=>copy(tt);el.querySelector('[data-t=yt]').onclick=()=>copy(yt);
 el.querySelector('[data-t=posted]').onclick=e=>{const on=store(pk)!=='1';store(pk,on?'1':'0');el.classList.toggle('posted',on);e.target.textContent=on?'Posted ✓':'Mark posted'};
 $('#grid').appendChild(el)}
</script></body></html>
"""
