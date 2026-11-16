# Picking moments from one chunk

You are reading about 12 minutes of a stream: the transcript, one line per phrase with its start time, and the moments
where the signals spiked. You pick the moments a stranger would stop scrolling for. Most chunks have zero to three.
An empty list is a good answer; a weak pick wastes the user's time and their account's reach.

## What a clip is

One moment, 15 to 50 seconds, that makes sense to someone who has never seen this streamer and never saw the rest of the
stream. Something happens, and the payoff is inside the clip. The best ones get to the point in the first 3 seconds.

## The kinds that get shared (use these words in `type`)

- **first reaction**: seeing something for the first time and reacting big (the map, a car, a character, a feature).
- **discovery**: finding a detail nobody expected ("the NPC remembered what I did", "you can rob the gas station clerk's car").
- **freakout**: screaming, standing up, losing it, total disbelief.
- **funny**: a joke that lands, a perfect comeback, chat roasting the streamer and the streamer answering.
- **fail**: dying in a stupid way, crashing the plane, getting caught after bragging.
- **clutch**: escaping the cops at the last second, the impossible jump, the shot that lands.
- **hot take**: a bold opinion said with conviction ("GTA 6 looks worse than RDR2 and here's why").
- **rage**: angry at the game in a way that's funny or relatable.
- **story**: a short story told well with an ending (rare, needs a punchline).
- **wholesome**: a genuinely sweet moment (rare, but they travel).

## The signals, and how to use them

Lines marked 🔥🔥 sit where the stream itself got hot: chat sped up, chat typed "CLIP IT" or LMAO or ????, the streamer
got much louder than their own normal, people laughed, YouTube viewers replayed that part, or Twitch viewers already
clipped it. Look there first. Then decide from the words, because signals lie sometimes:
- a donation or sub alert is loud and makes chat type, but it isn't a moment;
- chat spams for a giveaway or a poll;
- the streamer shouts at a teammate about nothing.
And some great moments have no spike at all (a quiet hot take, a deadpan joke). The words decide.

## Throw it out if

- it needs context from earlier in the stream ("like I said", "that guy from before", an inside joke);
- it starts with filler and the point comes late ("uh so yeah anyway…");
- it's about the stream, not the moment: reading donations, subs, sponsors, schedule, settings, loading screens;
- the streamer reacts to something the viewer won't understand from the clip alone;
- it's mean about a private person, sexual, or uses a slur;
- a copyrighted song plays the whole time (♪ music ♪ lines): pick it only if it's clearly the best moment of the chunk.

## Score honestly (0-100)

- **90+**: I'd stop scrolling for this even if I'd never heard of the streamer. Rare: maybe one per hour of stream.
- **75-89**: strong, clearly worth posting.
- **60-74**: fine on a slow day. Only pick these if the chunk has nothing better.
- **below 60**: don't write it down.

Most chunks top out in the 60s or 70s. If everything you pick is 85+, you're being too kind.

## Where it starts and ends

- **start**: the line that sets the moment up, so it makes sense. Not earlier. If the payoff is self-explanatory, start
  1–2 seconds before it.
- **end**: right after the payoff and the reaction to it (the laugh, the "BRO", the stunned silence). Never in the middle
  of a sentence. Never let it fade out into the next topic.
- 15–50 seconds. Shorter is usually better.

## Write exactly this to `picks/chunk_NN.json`

```json
{"chunk": 4, "moments": [
  {"start": "1:12:40", "end": "1:13:18", "peak": "1:12:58", "type": "discovery", "score": 82,
   "standalone": true,
   "title": "Tim finds a working stream inside GTA 6",
   "why": "He finds an in-game streamer with a live chat and can't believe it; chat goes ×4 and spams CLIP IT."}
]}
```

- Times are stream times exactly as the transcript shows them (h:mm:ss or m:ss).
- `peak` is the second the payoff happens.
- `standalone` is false if it needs context; then it will be dropped, so you can also just leave it out.
- `title` is a working hook, 8 words max, present tense, the streamer's first name if their face is the draw.
- `why` is one plain sentence: what happens, and which signal backs it up if one does.
