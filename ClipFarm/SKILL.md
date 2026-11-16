---
name: clip-farm
description: Turns long Twitch, YouTube or Kick streams into ready-to-post vertical clips (facecam on top, gameplay below, hook title, word-by-word captions, post captions), picking moments from the transcript plus chat speed, what chat typed, loudness, laughs, YouTube's most-replayed graph and the clips Twitch viewers already made. Also finds today's biggest streams of a game, follows a stream while it is live, and learns from your own Shorts' views. Use when someone wants to clip streams, set up the clipper, find streams to clip, clip a live stream, or check which clips work. Built for GTA 6 launch week, works on any stream.
---

# Clip Farm

You run a clipping pipeline for the user. The heavy work (downloads, listening, the facecam, the edit) runs on their
Mac through one command, `cf`. Your job is the part that needs taste: reading the stream and picking the moments people
share. Talk to the user in plain words; they are often not technical.

`CF="$HOME/.claude/skills/clip-farm/scripts/cf"` (if the skill lives elsewhere, use the `scripts/cf` next to this file).
Run commands in the foreground unless a step below says otherwise, and show the user the lines that matter, not the whole log.

## 1. "Set up the clipper" (first time, or when anything breaks)

1. Run `$CF setup`. It checks the Mac, installs what's missing with Homebrew (ffmpeg, whisper-cpp, deno, python@3.12),
   makes the clipper's own Python in `~/.clip-farm`, and downloads the Whisper model (about 480 MB, once).
2. If Homebrew itself is missing, it prints the one install command. Homebrew asks for the Mac password, so the user has
   to paste it in their own Terminal; tell them exactly that, then run `$CF setup` again.
3. Finish with `$CF doctor`. When it says Ready, tell them, and tell them where clips will land (`~/ClipFarm/clips`).
4. Age-restricted YouTube streams (common with GTA) and Kick need the user's browser login: `$CF settings cookies chrome`
   (or safari, firefox, brave). Only set this when a stream needs it, and say what it does: yt-dlp reads their own
   browser's YouTube login on this Mac to download, nothing is sent anywhere else.
5. Most YouTube errors are fixed by running `$CF setup` again (it updates yt-dlp).

## 2. "Clip this stream" (one or several links)

1. **Scan:** `$CF scan "<url>" ["<url>" …]` (one call for several streams; downloads overlap). Always quote the links:
   the `?` in them is a wildcard to the shell. Run it in the background and wait for it to finish. This takes a few minutes per
   hour of stream. It prints the stream key, the signals it found and the number of chunks. If it says a stream is
   live, go to section 4. Read `program.json` in the stream folder: if the streamer mentions a clipping program,
   tell the user (that's money on top).
2. **Pick, chunk by chunk.** Each `~/ClipFarm/streams/<key>/chunks/chunk_NN.md` is ~12 minutes of transcript with the
   signal peaks marked (🔥🔥 top 3 %, 🔥 top 8 % of that stream). Read `reference/picking.md` once, then for every chunk
   write `picks/chunk_NN.json` in the stream folder. Every chunk needs a file, even an empty one.
   - Fast way: with more than 4 chunks, hand them to helpers in parallel (Agent tool), about 4 chunks each. Give each
     helper the full path of `reference/picking.md`, its chunk paths, and the exact output paths. Tell them to read the
     whole chunk, follow the rules exactly, write only the JSON, and reply with the moments they wrote.
   - On a small local model (Ollama), read the chunks yourself one by one, and before scanning run
     `$CF settings chunk_minutes 6` so each chunk fits.
3. `$CF check <key>` must say all picks files are valid. Fix what it lists.
4. **Shortlist:** `$CF shortlist <key>` merges every pick with the signals and drops repeats and anything already clipped.
5. **Final round:** read `reference/final-round.md` and the stream's `shortlist.md`, then write `final.json` in the
   stream folder. This is where you are harshest. Default 3 clips per stream (`settings clips_per_stream`), never more
   than 5, fewer if they aren't good enough.
6. **Make them:** `$CF render <key>`. Each clip downloads only the minutes it needs, finds the facecam, cuts the dead air,
   adds the hook and captions, and checks itself (length, loudness, black frames, the facecam box). It prints ✓ or a ⚠
   with the reason. It picks the layout per scene by itself (full camera, facecam over gameplay, or gameplay alone,
   switching inside a clip when the stream switches). A ✓ clip needs nothing more from you; spend your time on the ⚠
   ones: open the `.facecam.jpg` next to the clip, look, and if needed re-make one clip with `--only <id>` and
   `--layout stack|fullcam|gameplay`.
7. **Hand over:** run `$CF review` (opens the day's review page: every clip playable, its captions one tap from the
   clipboard, a mark-as-posted button). Tell the user in two or three lines: how long the stream was, how many moments
   you found, how many clips you kept and how long it all took (from `state.json`: `scan_seconds`, plus the render
   times). Remind them to watch every clip before posting.

Several streams: do steps 2–6 per stream; the helpers for different streams can run at the same time.

## 3. "Find me today's biggest GTA 6 streams"

`$CF find` (Twitch + YouTube, English, ranked by viewers; `--game "<Twitch category>"` for another game, `--limit N`).
Each Twitch row has a "so far" link: the stream's own VOD, which already holds everything up to now. Scan the top ones
the user wants (default: the top 3 that aren't 24/7 or giveaway streams) with section 2, using the so-far links.

## 4. "Clip this live stream" (follow it while it's on)

1. `$CF live start "<live url>"` starts recording in the background (it keeps the last 45 minutes on disk) and listens
   every 20 seconds (10-second segments). It returns at once.
2. Loop: `$CF live next --wait 900`. It waits until something happens (chat, loudness, laughs) and prints the last
   minutes of transcript with the peak marked. Decide with the same rules as the final round:
   - not worth it → `$CF live skip <n>`;
   - worth it → write `moment_<n>.decision.json` where it tells you (start, end, hook, caption, hashtags, type), then
     `$CF live cut <n>`. The clip lands with the others, usually 2–4 minutes after the moment happened.
3. Keep looping until the user says stop, then `$CF live stop`. `$CF live status` shows what's running (and the
   current hype against the bar it has to pass). A quiet stream that never triggers: `$CF settings live_threshold 0.25`;
   one that triggers on nothing: 0.45. Twitch shows ads to logged-out viewers, and an ad break gets recorded like the
   stream; it rarely triggers a moment, but if a decision window shows an ad, skip it.
4. Tell the user each clip as it lands, with its hook. Live mode works on Twitch, YouTube and Kick (Kick needs cookies).

## 5. "What's working?" (after a few days of posting)

`$CF learn <their YouTube channel link>` matches their Shorts to the clips it made and writes `~/.clip-farm/taste.md`:
which moment types and streamers beat their channel's median. The next scans carry that line into every chunk, and
you should lean on it when you score. Read them the table and one plain takeaway.

## Settings worth knowing

`$CF settings` shows all. `clips_per_stream` (3), `max_clips_per_stream` (5), `max_clip_s` (59), `chunk_minutes` (12),
`hashtags` (comma list), `credit` (true: the streamer's name on every clip), `cookies` (browser name or "none").

## Be straight with the user

- Clip streamers who are fine with clipping, and say so when a stream title or description says otherwise.
- YouTube pays Shorts through the Partner Program (1,000 subscribers + 4,000 watch hours in 12 months, or 10 million
  Shorts views in 90 days), and it can refuse "reused content". The hook, the captions and the cut help; their own
  commentary or a reaction stack helps more. Never promise views or money.
- Clips with copyrighted music under them (the game radio, a song on stream) can get claimed or muted: the shortlist
  marks them, prefer others.
- Mac only for now.
