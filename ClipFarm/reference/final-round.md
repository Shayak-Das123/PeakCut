# The final round

`shortlist.md` holds the best candidates of one stream, ranked by your chunk scores plus the signals, each with its full
transcript (the lines with │ are inside the clip). Now be harsh: only the ones that win get made.

## 1. Put them against each other

For every candidate, answer three questions honestly:
1. **Would a stranger stop in the first two seconds?** (A face mid-scream, a shocking line, a clear question.)
2. **Does it make sense with zero context?**
3. **Does it pay off inside the clip?**

Any "no" and it's out. Then rank what's left against each other: if you could post only one today, which one? Then which
second? Stop at the number the shortlist asks for (default 3, never more than 5). Fewer is fine.

Mix the kinds: three clips of the same type from one stream only if they're truly the three best.
Prefer candidates without ⚠ music; take one with music only if it's clearly better than the rest.

## 2. Tighten the edges

- Start at most 2 seconds before the first line that matters. Cut the warm-up.
- End at most 1.5 seconds after the payoff or the reaction to it.
- Sweet spot 18–40 seconds. Hard maximum 59. The renderer also removes dead air inside the clip, so don't fear a pause.

## 3. Write the hook (the white title on screen for the first 3 seconds)

- 8 words max, present tense, says the tension or what's happening.
- Use the streamer's first name when people know their face: "Tim can't believe who's streaming inside GTA 6".
- Shapes that work: "<Name> can't believe…", "<Name> finds…", "<Name> loses it when…", "Nobody expected…",
  "<Name> has a hot take on…", "This GTA 6 detail broke <Name>".
- It must be true to the clip. No emojis, no ALL CAPS, no "WAIT FOR IT".

## 4. Write the post caption

One or two plain sentences that say what happens, plus one question that gets comments. Under 150 characters.
Example: "Tim found a working stream inside GTA 6 and lost it. Would you watch a streamer inside the game?"
Hashtags: 3 to 5. The game first, then the streamer, then one broad one: `#GTA6 #TimTheTatman #gaming`.

## 5. Write exactly this to `final.json` (in the stream's folder)

```json
{"clips": [
  {"id": "c03", "start": "1:12:41", "end": "1:13:16", "peak": "1:12:58", "type": "discovery", "score": 88,
   "hook": "Tim finds a working stream inside GTA 6",
   "caption": "Tim found a working stream inside GTA 6 and lost it. Would you watch a streamer inside the game?",
   "hashtags": ["#GTA6", "#TimTheTatman", "#gaming"],
   "why": "Beat c01 and c07: instant payoff, no context needed, chat ×4 and CLIP IT ×9."}
]}
```

Keep the shortlist's `id` for each clip you keep. Add `"streamer"` when the channel name isn't the name people know
(the clip credit and caption use it): `"CaseOh"`, not `"MoreCaseOh"`. Optional per clip: `"layout": "stack" | "fullcam" | "gameplay"` to
force a layout, `"trim_air": false` to keep every pause (for a slow-burn joke).
Then run `cf render <key>`.
