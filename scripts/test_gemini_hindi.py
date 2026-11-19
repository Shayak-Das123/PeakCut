import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from studio import ai
import json

sample_hindi_dialogue = """
Carry: Arre bhai gaadi thok di! 3 star police lag gayi piche!
Carry: Jaldi bhaga gaadi, Lucia ko bachana hai bhai!
Carry: Arre ye NPC kya bol raha hai? 'Hey watch it buddy!' Arre tere baap ki road hai kya bsdk?
Carry: Bhai helicopter aa gaya! Helicopter se goli chal rahi hai!
Carry: Gadi ud gayi! Bhai kya physics banaya hai Rockstar ne, maza hi aa gaya!
"""

prompt = f"""You are the Lead Viral Clip Director scouting highlights for YouTube Shorts, Instagram Reels, and TikTok.
Streamer: CarryMinati
Game: Grand Theft Auto VI
Stream Language: Hindi / Hinglish

Spoken Dialogue in clip:
\"\"\"{sample_hindi_dialogue}\"\"\"

Generate high-converting, viral packaging for this clip:
1. hook: The big on-screen title card shown in the first 3 seconds (MAX 7-8 WORDS).
   - Use bold, punchy, present tense English or Hinglish that Indian gaming audiences love.
   - e.g. 'CarryMinati Gets 3 Star Police In GTA 6' or 'CarryMinati Roasts GTA 6 Vice City NPC'
   - NEVER use broken syllables or raw quotation marks!
2. caption: 1-2 punchy sentences in engaging Hinglish/English explaining what happened + 1 comment question.
3. hashtags: 4-6 viral hashtags (e.g. ['#carryminati', '#gta6', '#gaming', '#funny']).
"""

schema = {
    "type": "OBJECT",
    "properties": {
        "hook": {"type": "STRING"},
        "caption": {"type": "STRING"},
        "hashtags": {"type": "ARRAY", "items": {"type": "STRING"}}
    },
    "required": ["hook", "caption", "hashtags"]
}

res, tokens_in, tokens_out = ai.call("gemini-2.5-flash", prompt, schema)
print("Gemini Hindi Understanding & Curation Result:")
print(json.dumps(res, indent=2))
