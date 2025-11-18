
# coach.py
from __future__ import annotations
from datetime import date, timedelta
from typing import List, Dict, Any, Optional
import streamlit as st
import streamlit.components.v1 as components

# We will call the app's gemini_query via a thin wrapper you provide to us at import time.
# In app.py we will call: register_llm(gemini_query)
_LLM = None

def register_llm(llm_callable):
    """
    Register an LLM callable with signature: (prompt:str, history:list[tuple[str,str]], context:str)->str
    This lets us reuse your existing gemini_query (with OpenAI fallback) without circular imports.
    """
    global _LLM
    _LLM = llm_callable


PROMPT_TEMPLATE = """You are a 'Tough Love Financial Coach AI' named PRAKRITI AI. Your primary goal is to provide blunt, personalized feedback on wasteful spending (fizool karchi) and immediately pivot the user to an actionable saving commitment. You must use a conversational, slightly stern/humorous tone (Hinglish/Hindi-Latin script) suitable for a young Indian user.

Your output must be structured strictly in two separate sections, labeled 'SCOLD' and 'CHALLENGE'.

1) SCOLD Section (Fizool Karchi Warning):
- Tone: Conversational, slightly exaggerated, and stern, but NOT rude. Use emojis.
- Content: Directly reference the expense category and amount, and the reason for the warning. The warning must be 1-2 concise sentences.

2) CHALLENGE Section (Saving Quet/Question):
- Content: Generate a specific, actionable 7-day challenge related to the overspent category, and a clear question requiring a commitment to saving the potential amount.
- Use one of the provided SAVING_GOAL_CATEGORIES in the question.

---
INPUT DATA:
- USER_ID: {USER_ID}
- EXPENSE_CATEGORY: {EXPENSE_CATEGORY}
- EXPENSE_AMOUNT (Trigger): {EXPENSE_AMOUNT}
- TRIGGER_REASON: {TRIGGER_REASON}
- AVERAGE_DAILY_SPEND: {AVERAGE_DAILY_SPEND}
- SAVING_GOAL_CATEGORIES: {SAVING_GOAL_CATEGORIES}
---

GENERATE OUTPUT NOW:
SCOLD:

CHALLENGE:
"""

def _deterministic_fallback(
    USER_ID: str,
    EXPENSE_CATEGORY: str,
    EXPENSE_AMOUNT: str,
    TRIGGER_REASON: str,
    AVERAGE_DAILY_SPEND: str,
    SAVING_GOAL_CATEGORIES: List[str],
) -> str:
    goal = (SAVING_GOAL_CATEGORIES or ["Investment"])[0]
    # Safe parse for numbers
    def _num(s):
        try:
            return float(str(s).replace("₹","").replace(",",""))
        except:
            return 0.0
    daily = _num(AVERAGE_DAILY_SPEND)
    # conservative 7-day challenge saving
    potential = max(300, round(min(_num(EXPENSE_AMOUNT) * 0.35, daily * 1.5), -1))
    return (
f"SCOLD:\n"
f"Arey {USER_ID}, {EXPENSE_CATEGORY} mein {EXPENSE_AMOUNT}? 😬 {TRIGGER_REASON}. Bas karo fizool karchi!\n\n"
f"CHALLENGE:\n"
f"🔥 7-Day Challenge: **{EXPENSE_CATEGORY} detox!** Agle 7 din {EXPENSE_CATEGORY} par ₹0/low spend. "
f"Target bachat: **₹{int(potential)}**. Ab bolo — kya yeh **₹{int(potential)}** '{goal}' category mein seedhe allocate karein? COMMIT karo abhi! ✅"
    )

def generate_tough_love_coach_response(
    USER_ID: str,
    EXPENSE_CATEGORY: str,
    EXPENSE_AMOUNT: str,
    TRIGGER_REASON: str,
    AVERAGE_DAILY_SPEND: str,
    SAVING_GOAL_CATEGORIES: List[str],
    history: Optional[list] = None,
    context: str = "",
) -> str:
    """
    Returns strictly formatted text with `SCOLD:` and `CHALLENGE:` blocks.
    Uses the registered LLM; falls back to a deterministic template.
    """
    prompt = PROMPT_TEMPLATE.format(
        USER_ID=USER_ID,
        EXPENSE_CATEGORY=EXPENSE_CATEGORY,
        EXPENSE_AMOUNT=EXPENSE_AMOUNT,
        TRIGGER_REASON=TRIGGER_REASON,
        AVERAGE_DAILY_SPEND=AVERAGE_DAILY_SPEND,
        SAVING_GOAL_CATEGORIES=SAVING_GOAL_CATEGORIES,
    )
    if _LLM is None:
        # no LLM registered — deterministic fallback
        return _deterministic_fallback(USER_ID, EXPENSE_CATEGORY, EXPENSE_AMOUNT, TRIGGER_REASON, AVERAGE_DAILY_SPEND, SAVING_GOAL_CATEGORIES)

    try:
        raw = _LLM(prompt, history or [], context or f"Tough-love coach for {USER_ID}")
        # Strip any model preface like "🧠 *Gemini Smart AI:*"
        txt = str(raw).replace("🧠 *Gemini Smart AI:*", "").strip()
        # Ensure sections exist; if not, fallback
        if "SCOLD" not in txt or "CHALLENGE" not in txt:
            raise ValueError("Model did not respect format")
        return txt
    except Exception:
        return _deterministic_fallback(USER_ID, EXPENSE_CATEGORY, EXPENSE_AMOUNT, TRIGGER_REASON, AVERAGE_DAILY_SPEND, SAVING_GOAL_CATEGORIES)


def render_coach_bubble(message: str, voice: bool = True, bubble_id: str = "coach-bubble-1") -> None:
    """
    Pretty, glowing bubble + (optional) TTS using Web Speech API.
    """
    # HTML escape minimal (we expect plain text with ** possibly)
    msg_html = message.replace("\n", "<br>").replace("**", "<b>").replace("__", "<i>")
    components.html(f"""
    <style>
      .coach-bubble {{
        margin: 12px 0; padding: 14px 16px; border-radius: 16px;
        background: radial-gradient(120% 120% at 0% 0%, rgba(255,87,166,.10) 0%, rgba(255,255,255,.03) 70%);
        border: 1px solid rgba(255,255,255,.22);
        color: #ffeaf3;
        box-shadow: 0 10px 30px rgba(255,87,166,.25), inset 0 0 40px rgba(255,255,255,.04);
        position: relative;
      }}
      .coach-bubble:before {{
        content: "₹"; position: absolute; left: -10px; top: -10px;
        background: #ff57a6; color:#0f1117; width: 32px; height: 32px; border-radius: 50%;
        display:flex; align-items:center; justify-content:center; font-weight:900;
        box-shadow: 0 0 18px rgba(255,87,166,.6);
        animation: pop 1.8s ease-in-out infinite;
      }}
      @keyframes pop {{ 0% {{ transform: scale(1) }} 50% {{ transform: scale(1.08) }} 100% {{ transform: scale(1) }} }}
      .coach-title {{
        font-weight: 900; letter-spacing: .4px; margin-bottom: 6px; font-size: 14px; color: #ffd6ea;
      }}
      .coach-tts {{
        position:absolute; right:12px; top:10px; font-size:12px; opacity:.85; cursor:pointer;
        padding:4px 8px; border-radius:999px; background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.18);
      }}
      .glow {{
        animation: glow 2.2s ease-in-out infinite;
      }}
      @keyframes glow {{
        0% {{ box-shadow: 0 0 10px rgba(255,122,184,.25)}}
        50% {{ box-shadow: 0 0 26px rgba(255,122,184,.55)}}
        100% {{ box-shadow: 0 0 10px rgba(255,122,184,.25)}}
      }}
    </style>
    <div id="{bubble_id}" class="coach-bubble glow">
      <div class="coach-title">💬 PRAKRITI AI — Tough Love Coach</div>
      <div style="white-space: normal; line-height:1.5;">{msg_html}</div>
      <div class="coach-tts" onclick="(function(){{
          try {{
            window.speechSynthesis.cancel();
            var text = `{message.replace("`", "'")}`;
            var u = new SpeechSynthesisUtterance(text);
            u.lang = 'hi-IN';
            u.rate = 1.0; u.pitch = 1.0;
            window.speechSynthesis.speak(u);
          }} catch(e) {{ console.warn(e); }}
      }})()">🔊 बोलो</div>
    </div>
    """, height=160)
