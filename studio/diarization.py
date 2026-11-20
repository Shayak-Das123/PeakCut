"""Speaker Diarization: Distinguishes live streamer speech from in-game characters/NPCs in GTA 6."""
from __future__ import annotations

import json
from typing import Optional
from studio import ai


def group_words_into_utterances(words: list[dict], max_gap: float = 0.55) -> list[list[dict]]:
    """Groups consecutive words into cohesive dialogue utterances based on speech pauses."""
    if not words:
        return []
    utterances = []
    cur = [words[0]]
    for w in words[1:]:
        if w["s"] - cur[-1]["e"] > max_gap:
            utterances.append(cur)
            cur = [w]
        else:
            cur.append(w)
    if cur:
        utterances.append(cur)
    return utterances


def filter_streamer_words(streamer: str, words_in_clip: list[dict], game: str = "GTA 6") -> list[dict]:
    """
    Classifies utterance by utterance whether dialogue was spoken by the streamer or an in-game character.
    Returns only words spoken by the streamer so game characters are NOT captioned.
    """
    if not words_in_clip:
        return []

    utterances = group_words_into_utterances(words_in_clip)
    if len(utterances) <= 1:
        return words_in_clip

    formatted = []
    for idx, u in enumerate(utterances):
        text = " ".join(w["w"] for w in u).strip()
        formatted.append(f"[{idx}] ({u[0]['s']:.1f}s - {u[-1]['e']:.1f}s): \"{text}\"")

    prompt = f"""You are a broadcast audio diarization assistant for stream clips.
Streamer: {streamer} (the real human streamer with a facecam and microphone).
Game being played: {game}.

Here are the transcribed spoken lines from the clip audio track:
{chr(10).join(formatted)}

Task: Identify which lines were spoken LIVE by {streamer} into their microphone, and which lines were spoken by IN-GAME CHARACTERS/NPCs (e.g. cutscene actors, radio dispatch, mission NPCs, police, companion characters).

Rules:
1. Streamer dialogue includes: talking to chat, yelling, laughing, commentary, streamer slang (chat, bro, w, l, oh my god, what are they doing, look at this, no way, let's go, what the hell).
2. Game character dialogue includes: scripted story cutscenes, in-game radio, police chatter, NPC responses (affirmative, cargo, dispatch, mission instructions, character banter).
3. We MUST NOT display captions for in-game characters. Only {streamer} should receive captions.

Respond in JSON with this exact schema:
{{
  "streamer_line_indices": [0, 2],
  "character_line_indices": [1],
  "reasoning": "brief explanation"
}}
"""
    try:
        schema = {
            "type": "OBJECT",
            "properties": {
                "streamer_line_indices": {"type": "ARRAY", "items": {"type": "INTEGER"}},
                "character_line_indices": {"type": "ARRAY", "items": {"type": "INTEGER"}},
                "reasoning": {"type": "STRING"}
            },
            "required": ["streamer_line_indices", "character_line_indices"]
        }
        models = ai.models()
        model_name = models[0] if models else "gemini-2.5-flash"
        res, _, _ = ai.call(model_name, prompt, schema)
        streamer_indices = set(res.get("streamer_line_indices", []))
        char_indices = set(res.get("character_line_indices", []))

        if char_indices:
            kept_words = []
            for idx, u in enumerate(utterances):
                if idx in streamer_indices or idx not in char_indices:
                    kept_words.extend(u)
            print(f"[Diarization] Filtered out {len(char_indices)} character line(s). Kept {len(kept_words)}/{len(words_in_clip)} streamer words.")
            return kept_words
    except Exception as ex:
        print(f"[Diarization] AI speaker filter fallback ({ex})")

    # Heuristic fallback if AI unavailable
    kept_words = []
    npc_keywords = {"affirmative", "dispatch", "suspect", "perimeter", "10-4", "officer", "headquarters", "radio", "cargo"}
    for u in utterances:
        text_lower = " ".join(w["w"] for w in u).lower()
        if any(kw in text_lower for kw in npc_keywords):
            print(f"[Diarization Heuristic] Filtered out character line: '{text_lower}'")
            continue
        kept_words.extend(u)
    return kept_words
