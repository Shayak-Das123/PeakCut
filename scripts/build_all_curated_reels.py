"""Batch builder for elite curated reels across Kai Cenat, IShowSpeed, xQc, and CarryMinati."""
import json
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.render_elite_clips import render_curated_reel, load_stream_segments

def build_kai_cenat():
    stream_key = "youtube-VZqAPdUvqnQ"
    streamer = "Kai Cenat"
    s_dir = ROOT / "ClipFarm" / "work" / "streams" / stream_key
    words_file = s_dir / "words.json"
    if not words_file.exists():
        print(f"Skipping {streamer}: words.json not found")
        return
    words = json.loads(words_file.read_text("utf-8"))
    segments = load_stream_segments(s_dir)
    if not segments:
        print(f"Skipping {streamer}: buffer segments not found")
        return

    moments = [
        {
            "cid": "clip_01",
            "start": 70.0,
            "end": 93.0,
            "hook": "Kai Cenat breaks his diet in GTA 6",
            "caption": "Kai Cenat ordered fast food in Vice City and immediately gave up on his diet! Would you stick to your fitness goals in GTA 6 or feast on every restaurant?",
            "score": 96
        },
        {
            "cid": "clip_02",
            "start": 94.0,
            "end": 136.0,
            "hook": "Kai Cenat buys romantic gifts for Lucia",
            "caption": "Kai Cenat raided the store buying luxury wine, bath supplies, and snacks to impress Lucia! Is Kai the biggest simp in Vice City history?",
            "score": 94
        },
        {
            "cid": "clip_03",
            "start": 204.0,
            "end": 228.0,
            "hook": "Kai Cenat's first GTA 6 haircut",
            "caption": "Kai Cenat pulled up to the Vice City barber demanding the freshest lineup in gaming! Did Rockstar give Kai the best haircut in GTA history?",
            "score": 99
        },
        {
            "cid": "clip_04",
            "start": 239.0,
            "end": 270.0,
            "hook": "Kai Cenat judges his GTA 6 lineup",
            "caption": "Kai Cenat inspected his character in the barber mirror making sure the hairline was razor sharp! Would you trust a GTA 6 barber with your haircut?",
            "score": 97
        },
        {
            "cid": "clip_05",
            "start": 292.0,
            "end": 336.0,
            "hook": "Kai Cenat tries his first GTA 6 robbery",
            "caption": "Kai Cenat tried pulling off his first convenience store heist in Vice City before chaos erupted! Would you have escaped with the loot or panicked?",
            "score": 98
        },
        {
            "cid": "clip_06",
            "start": 340.0,
            "end": 368.0,
            "hook": "Kai Cenat triggers wild police chase",
            "caption": "Kai Cenat barely stepped out of the store before sirens flooded the street! Are GTA 6 Vice City cops way more aggressive than GTA 5?",
            "score": 95
        },
        {
            "cid": "clip_07",
            "start": 554.0,
            "end": 585.0,
            "hook": "Kai Cenat buys luxury roses for Lucia",
            "caption": "Kai Cenat walked into a Vice City flower boutique and bought the biggest bouquet for Lucia! Would you drop all your heist earnings on roses?",
            "score": 94
        },
        {
            "cid": "clip_08",
            "start": 585.0,
            "end": 605.0,
            "hook": "Vice City florist roasts Kai Cenat",
            "caption": "The florist NPC in GTA 6 immediately called Kai Cenat a desperate Romeo! Do GTA 6 NPCs have the most savage insults in gaming?",
            "score": 96
        },
        {
            "cid": "clip_09",
            "start": 753.0,
            "end": 792.0,
            "hook": "Lucia's best friend interrogates Kai Cenat",
            "caption": "Lucia's best friend warned Kai Cenat that he better treat Lucia right or catch hands in Vice City! Is Andy already the wildest supporting character?",
            "score": 95
        }
    ]

    rendered_clips = []
    for m in moments:
        w_in = [w for w in words if m["start"] <= w["s"] <= m["end"]]
        meta = render_curated_reel(
            stream_key=stream_key,
            streamer=streamer,
            cid=m["cid"],
            start_sec=m["start"],
            end_sec=m["end"],
            hook=m["hook"],
            caption=m["caption"],
            hashtags=["#GTA6", "#KaiCenat", "#GrandTheftAuto6", "#Shorts", "#Gaming"],
            score=m["score"],
            words_in_clip=w_in,
            segments=segments
        )
        if meta:
            rendered_clips.append(meta)

    # Save final.json & picks.json
    final_p = s_dir / "final.json"
    final_p.write_text(json.dumps({"key": stream_key, "clips": rendered_clips}, indent=2, ensure_ascii=False), encoding="utf-8")
    picks_p = s_dir / "picks.json"
    picks_p.write_text(json.dumps({"curated_picks": rendered_clips}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{streamer}] Finished {len(rendered_clips)} elite reels!")

def build_speed():
    stream_key = "youtube-qSyo5d-dVzA"
    streamer = "IShowSpeed"
    s_dir = ROOT / "ClipFarm" / "work" / "streams" / stream_key
    words_file = s_dir / "words.json"
    if not words_file.exists():
        print(f"Skipping {streamer}: words.json not found")
        return
    words = json.loads(words_file.read_text("utf-8"))
    segments = load_stream_segments(s_dir)
    if not segments:
        print(f"Skipping {streamer}: buffer segments not found")
        return

    moments = [
        {
            "cid": "clip_01",
            "start": 218.0,
            "end": 252.0,
            "hook": "Speed can't believe GTA 6 cutscene",
            "caption": "IShowSpeed watched the first Lucia story cutscene in GTA 6 and completely lost his composure! Is this the greatest story introduction Rockstar ever made?",
            "score": 96
        },
        {
            "cid": "clip_02",
            "start": 260.0,
            "end": 283.0,
            "hook": "Speed discovers wild GTA 6 detail",
            "caption": "IShowSpeed paused gameplay to inspect the character customization details in GTA 6 and couldn't stop yelling! Did Rockstar take realism too far?",
            "score": 95
        },
        {
            "cid": "clip_03",
            "start": 324.0,
            "end": 342.0,
            "hook": "Speed customizes his GTA 6 outfit",
            "caption": "IShowSpeed went full fashion mode picking boots and sneakers in Vice City! Is Speed officially the best dressed criminal in Florida?",
            "score": 93
        },
        {
            "cid": "clip_04",
            "start": 395.0,
            "end": 415.0,
            "hook": "Speed deposits his first heist money",
            "caption": "IShowSpeed deposited his hard-earned thousands into the bank ATM before getting ambushed! Would you store your money in the bank or carry cash?",
            "score": 94
        },
        {
            "cid": "clip_05",
            "start": 564.0,
            "end": 592.0,
            "hook": "Speed finds hilarious GTA 6 glitch",
            "caption": "IShowSpeed ran into a hilarious physics glitch in early access Vice City that sent him screaming! Are day-one GTA glitches the funniest part of gaming?",
            "score": 97
        },
        {
            "cid": "clip_06",
            "start": 638.0,
            "end": 666.0,
            "hook": "Speed roasts billionaire NPC in GTA 6",
            "caption": "IShowSpeed overheard a rich English NPC in Vice City talking down to locals and immediately started roasting him! Are GTA 6 NPCs funnier than GTA 5?",
            "score": 95
        },
        {
            "cid": "clip_07",
            "start": 699.0,
            "end": 718.0,
            "hook": "Speed plans his first GTA 6 heist",
            "caption": "IShowSpeed put on his heist mask and checked comms with his crew before the big score! Which crew member is definitely messing up the plan?",
            "score": 98
        },
        {
            "cid": "clip_08",
            "start": 805.0,
            "end": 825.0,
            "hook": "Florida storm threatens Speed's stream",
            "caption": "A real-life Florida hurricane storm hit outside Speed's room while playing GTA 6! Was God testing Speed's dedication to the grind?",
            "score": 96
        },
        {
            "cid": "clip_09",
            "start": 828.0,
            "end": 866.0,
            "hook": "Speed loses his mind over Miami graphics",
            "caption": "IShowSpeed took one look at the Miami skyline reflections and started yelling at his chat! Did Rockstar officially peak video game graphics forever?",
            "score": 99
        }
    ]

    rendered_clips = []
    for m in moments:
        w_in = [w for w in words if m["start"] <= w["s"] <= m["end"]]
        meta = render_curated_reel(
            stream_key=stream_key,
            streamer=streamer,
            cid=m["cid"],
            start_sec=m["start"],
            end_sec=m["end"],
            hook=m["hook"],
            caption=m["caption"],
            hashtags=["#GTA6", "#IShowSpeed", "#GrandTheftAuto6", "#Shorts", "#Gaming"],
            score=m["score"],
            words_in_clip=w_in,
            segments=segments
        )
        if meta:
            rendered_clips.append(meta)

    final_p = s_dir / "final.json"
    final_p.write_text(json.dumps({"key": stream_key, "clips": rendered_clips}, indent=2, ensure_ascii=False), encoding="utf-8")
    picks_p = s_dir / "picks.json"
    picks_p.write_text(json.dumps({"curated_picks": rendered_clips}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{streamer}] Finished {len(rendered_clips)} elite reels!")

def build_xqc():
    stream_key = "twitch-xqc"
    streamer = "xQc"
    s_dir = ROOT / "ClipFarm" / "work" / "streams" / stream_key
    words_file = s_dir / "words.json"
    if not words_file.exists():
        print(f"Skipping {streamer}: words.json not found")
        return
    words = json.loads(words_file.read_text("utf-8"))
    segments = load_stream_segments(s_dir)
    if not segments:
        print(f"Skipping {streamer}: buffer segments not found")
        return

    moments = [
        {
            "cid": "clip_01",
            "start": 309.0,
            "end": 329.0,
            "hook": "xQc explores luxury Vice City apartment",
            "caption": "xQc walked into a penthouse apartment in GTA 6 and couldn't believe the interior detail! Would you grind for the luxury crib first or the supercar?",
            "score": 95
        },
        {
            "cid": "clip_02",
            "start": 330.0,
            "end": 353.0,
            "hook": "xQc rizzes up Vice City NPC",
            "caption": "xQc tried using his best sushi pickup line on an NPC in GTA 6 and got instantly rejected! Does xQc have negative rizz in Vice City?",
            "score": 97
        },
        {
            "cid": "clip_03",
            "start": 468.0,
            "end": 489.0,
            "hook": "xQc gets lost in massive GTA 6 mansion",
            "caption": "xQc wandered through a sprawling Vice City estate and couldn't find the exit door! Did Rockstar make the map way too big for xQc's brain?",
            "score": 94
        },
        {
            "cid": "clip_04",
            "start": 513.0,
            "end": 548.0,
            "hook": "xQc recaps his wildest night in Vice City",
            "caption": "xQc woke up in GTA 6 after a wild night out and had to retrace his steps through the city! What was your character's craziest night in GTA?",
            "score": 96
        },
        {
            "cid": "clip_05",
            "start": 679.0,
            "end": 696.0,
            "hook": "xQc visits his old hood in GTA 6",
            "caption": "xQc took a drive down memory lane visiting his character's old neighborhood in Vice City! Which classic GTA location are you visiting first?",
            "score": 93
        },
        {
            "cid": "clip_06",
            "start": 891.0,
            "end": 911.0,
            "hook": "xQc reacts to GTA 6 social media",
            "caption": "xQc scrolled through Snapmatic in GTA 6 and found NPCs flexing millions on plastic surgery! Is Rockstar's parody of modern internet culture 100% accurate?",
            "score": 98
        }
    ]

    rendered_clips = []
    for m in moments:
        w_in = [w for w in words if m["start"] <= w["s"] <= m["end"]]
        meta = render_curated_reel(
            stream_key=stream_key,
            streamer=streamer,
            cid=m["cid"],
            start_sec=m["start"],
            end_sec=m["end"],
            hook=m["hook"],
            caption=m["caption"],
            hashtags=["#GTA6", "#xQc", "#GrandTheftAuto6", "#Shorts", "#Gaming"],
            score=m["score"],
            words_in_clip=w_in,
            segments=segments
        )
        if meta:
            rendered_clips.append(meta)

    final_p = s_dir / "final.json"
    final_p.write_text(json.dumps({"key": stream_key, "clips": rendered_clips}, indent=2, ensure_ascii=False), encoding="utf-8")
    picks_p = s_dir / "picks.json"
    picks_p.write_text(json.dumps({"curated_picks": rendered_clips}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{streamer}] Finished {len(rendered_clips)} elite reels!")

def build_carryminati():
    stream_key = "youtube-carryminati"
    streamer = "CarryMinati"
    s_dir = ROOT / "ClipFarm" / "work" / "streams" / stream_key
    words_file = s_dir / "words.json"
    if not words_file.exists():
        print(f"Skipping {streamer}: words.json not found")
        return
    words = json.loads(words_file.read_text("utf-8"))
    segments = load_stream_segments(s_dir)
    if not segments:
        print(f"Skipping {streamer}: buffer segments not found")
        return

    moments = [
        {
            "cid": "clip_01",
            "start": 110.0,
            "end": 142.0,
            "hook": "CarryMinati's wild Hindi reaction in GTA 6",
            "caption": "CarryMinati could not hold back his laughter reacting to Vice City NPCs in Hindi! Is CarryMinati's commentary the funniest GTA 6 experience on YouTube?",
            "score": 98
        },
        {
            "cid": "clip_02",
            "start": 155.0,
            "end": 184.0,
            "hook": "CarryMinati plans his first Vice City heist",
            "caption": "CarryMinati scouted the biggest money players in Vice City to pull off an insane job! Are you grinding missions solo or teaming up in GTA 6?",
            "score": 95
        },
        {
            "cid": "clip_03",
            "start": 213.0,
            "end": 244.0,
            "hook": "CarryMinati executes intense water landing",
            "caption": "CarryMinati had to keep the chopper steady right above ocean waves for a high-stakes drop! Could you nail this helicopter landing under pressure?",
            "score": 96
        },
        {
            "cid": "clip_04",
            "start": 262.0,
            "end": 298.0,
            "hook": "Parole officer catches Jason in GTA 6",
            "caption": "Jason's ankle monitor started flashing while the parole officer called 30 times in a row! Would you make a run for it or try to talk your way out?",
            "score": 99
        },
        {
            "cid": "clip_05",
            "start": 310.0,
            "end": 338.0,
            "hook": "Jason scrambles to hide contraband from officer",
            "caption": "Panic mode activated as the PO pulls into the driveway with seconds to hide illegal packages in the shed! Did your heart drop during this mission?",
            "score": 97
        },
        {
            "cid": "clip_06",
            "start": 376.0,
            "end": 412.0,
            "hook": "Jason lies to save Lucia from prison",
            "caption": "Jason looked the parole officer dead in the eyes and lied that Lucia was on the couch the whole time! Would you risk going to jail to protect your partner?",
            "score": 98
        }
    ]

    rendered_clips = []
    for m in moments:
        w_in = [w for w in words if m["start"] <= w["s"] <= m["end"]]
        meta = render_curated_reel(
            stream_key=stream_key,
            streamer=streamer,
            cid=m["cid"],
            start_sec=m["start"],
            end_sec=m["end"],
            hook=m["hook"],
            caption=m["caption"],
            hashtags=["#GTA6", "#CarryMinati", "#GrandTheftAuto6", "#Shorts", "#Gaming"],
            score=m["score"],
            words_in_clip=w_in,
            segments=segments
        )
        if meta:
            rendered_clips.append(meta)

    final_p = s_dir / "final.json"
    final_p.write_text(json.dumps({"key": stream_key, "clips": rendered_clips}, indent=2, ensure_ascii=False), encoding="utf-8")
    picks_p = s_dir / "picks.json"
    picks_p.write_text(json.dumps({"curated_picks": rendered_clips}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{streamer}] Finished {len(rendered_clips)} elite reels!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--streamer", default="all", choices=["all", "kai", "speed", "xqc", "carry"])
    args = parser.parse_args()

    t0 = time.time()
    print("=== STARTING ELITE REELS BUILD (RTX 4050 GPU NVENC) ===")
    if args.streamer in ("all", "kai"):
        build_kai_cenat()
    if args.streamer in ("all", "speed"):
        build_speed()
    if args.streamer in ("all", "xqc"):
        build_xqc()
    if args.streamer in ("all", "carry"):
        build_carryminati()
    print(f"=== ALL CURATED REELS COMPLETE IN {time.time() - t0:.1f}s ===")
