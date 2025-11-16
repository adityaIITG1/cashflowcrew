
# ai_helpers.py
from __future__ import annotations

import json
from typing import Dict, Any, Tuple, List

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


# ----------------- Simple Utility (local money formatter) ----------------- #

def money(amount: float) -> str:
    """Local money formatter (independent of ui_patches.money)."""
    try:
        return f"₹{float(amount):,.0f}"
    except Exception:
        return f"₹{amount}"


# ----------------- Bilingual Advice Builder (Display Text) ----------------- #

def build_smart_advice_bilingual(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Smart Machine gives:
    - Short intro (who am I)
    - Tech + team + features in simple language
    - Then focused spending + saving advice

    NOTE:
    - This returns full paragraphs (good for display in UI).
    - TTS will internally convert this into small chunks to avoid 'tut tut' effect.
    """

    advice_hi = (
        "नमस्ते! मैं स्मार्ट मशीन हूँ — आपका अपना AI खर्चा और बचत सलाहकार। "
        "मुझे Python, Streamlit और Google Gemini Gen AI से बनाया गया है, "
        "और मेरी टीम लीड हैं प्राकृति जायसवाल, जो एक डेटा साइंस स्टूडेंट हैं। "

        "मेरा काम है: तुम्हारे खर्चों और आय के ग्राफ दिखाना, "
        "तुम्हारा Financial Health Index बताना, नो-स्पेंड डेज़ की स्ट्रीक ट्रैक करना, "
        "बिल की फोटो से ऑटोमेटिक अमाउंट पढ़ना, और चैटबॉट की तरह तुम्हारे सवालों के जवाब देना। "

        "\n\nअभी जो pattern दिख रहा है, उससे लगता है कि कुछ खर्च थोड़े बिखरे हुए हैं — "
        "जैसे देर रात के स्नैक्स, फूड डिलीवरी या छोटी-छोटी आउटिंग पर पैसे निकल रहे हैं। "
        "ये खर्च छोटे दिखते हैं, लेकिन मिलकर बड़ी राशि बन जाते हैं, इसलिए इन्हें थोड़ा कम करना ज़रूरी है। "

        "आज से एक simple rule पकड़ लो: जितना भी extra या बेवजह खर्च होता है, उसका कम से कम 30% "
        "तुरंत सेविंग या किसी simple SIP में डाल दो। "
        "यहीं से असली धन बनता है — वरना पैसा बस आता-जाता ही रहेगा, बचत नहीं बनेगी।"
    )

    advice_en = (
        "Hello! I am Smart Machine — your AI-powered spending and savings advisor. "
        "I am built using Python, Streamlit and Google Gemini Gen AI, "
        "and I'm guided by team lead Prakriti Jaiswal, a data science student. "

        "My job is to show you clear graphs of your income and expenses, "
        "compute your Financial Health Index, track your no-spend day streaks, "
        "read amounts from bill photos using OCR, and answer your queries like a smart chatbot. "

        "\n\nFrom your recent pattern, I can see some scattered spending — "
        "on late-night snacks, food deliveries and small random outings. "
        "These feel small in the moment but quietly add up to a big leak in your budget, "
        "so it's important to reduce them a bit. "

        "From today, follow one simple rule: take at least 30% of any non-essential or impulsive spending "
        "and immediately move it into savings or a basic SIP. "
        "That's how real wealth is built — otherwise money will keep flowing in and flowing out "
        "without creating solid savings."
    )

    return advice_hi, advice_en


# ----------------- Internal Helper: Split into Short TTS Chunks ----------- #

def _to_chunks(text: str, lang: str) -> List[str]:
    """
    Convert long paragraph into short, TTS-friendly chunks.
    - For Hindi: split on '।', newlines.
    - For English: split on '.', '!', '?', newlines.
    - Filters out empty small bits.
    """
    if not text:
        return []

    # Normalize newlines
    txt = text.replace("\r\n", "\n").replace("\r", "\n")

    # First, split by newline to respect paragraph breaks
    raw_parts: List[str] = []
    for block in txt.split("\n"):
        block = block.strip()
        if not block:
            continue
        if lang == "hi":
            # Hindi sentence boundaries
            sentences = [s.strip() for s in block.split("।") if s.strip()]
            raw_parts.extend(sentences)
        else:
            # English sentence boundaries
            temp = block
            for sep in [".", "?", "!"]:
                temp = temp.replace(sep, ".")
            sentences = [s.strip() for s in temp.split(".") if s.strip()]
            raw_parts.extend(sentences)

    # Final clean + small merges if needed
    chunks: List[str] = []
    for part in raw_parts:
        if not part:
            continue
        # Re-add punctuation lightly so voice doesn't sound flat
        if lang == "hi":
            if not part.endswith("।"):
                part = part + "।"
        else:
            if not part.endswith("."):
                part = part + "."
        chunks.append(part)

    return chunks


# ----------------- Button-based TTS (Speak Now, Chunked) ------------------ #

def speak_bilingual_js(advice_hi: str, advice_en: str, order: str = "hi-en") -> None:
    """
    'Speak now' button ke liye: sirf TTS chalata hai, bina wake-word ke.
    - Internally advice ko chhote-chhote chunks me tod deta hai.
    - order = "hi-en" ya "en-hi"
    """

    hi_chunks = _to_chunks(advice_hi, lang="hi")
    en_chunks = _to_chunks(advice_en, lang="en")

    cfg = {
        "hi_chunks": hi_chunks,
        "en_chunks": en_chunks,
        "order": order or "hi-en",
    }
    payload = json.dumps(cfg, ensure_ascii=False)

    html = f"""
    <script>
    (function() {{
        const cfg = {payload};

        if (!("speechSynthesis" in window)) {{
            console.log("SpeechSynthesis not available in this browser.");
            return;
        }}

        function speakSequence() {{
            const parts = cfg.order === "hi-en" ? ["hi", "en"] : ["en", "hi"];
            const queue = [];

            parts.forEach((key) => {{
                const chunkKey = key === "hi" ? "hi_chunks" : "en_chunks";
                const chunks = cfg[chunkKey] || [];
                chunks.forEach((txt) => {{
                    if (txt && txt.trim().length > 0) {{
                        queue.push({{
                            text: txt.trim(),
                            lang: key === "hi" ? "hi-IN" : "en-IN"
                        }});
                    }}
                }});
            }});

            if (!queue.length) return;

            window.speechSynthesis.cancel();

            const speakNext = () => {{
                if (!queue.length) return;
                const item = queue.shift();
                const u = new SpeechSynthesisUtterance(item.text);
                u.lang = item.lang;

                // ⚡ Energetic Female Coach Mode (button-based)
                try {{
                    const voices = window.speechSynthesis.getVoices() || [];
                    const targetLang = item.lang === "hi-IN" ? "hi" : "en";
                    const preferred = voices.find(v =>
                        (
                            v.name.toLowerCase().includes("female") ||
                            v.name.toLowerCase().includes("woman") ||
                            v.name.toLowerCase().includes("girl") ||
                            v.name.toLowerCase().includes("coach") ||
                            v.name.toLowerCase().includes("strong") ||
                            v.name.toLowerCase().includes("premium") ||
                            v.name.toLowerCase().includes("neural") ||
                            v.name.toLowerCase().includes("natural") ||
                            v.name.toLowerCase().includes("wavenet")
                        ) &&
                        v.lang.toLowerCase().includes(targetLang)
                    );
                    if (preferred) {{
                        u.voice = preferred;
                    }}
                }} catch (e) {{
                    console.log("Voice selection error:", e);
                }}

                // Energetic but still clear
                u.rate = 1.20;   // thoda fast, smooth
                u.pitch = 1.15;  // confident female tone
                u.volume = 1.0;

                u.onend = speakNext;
                try {{
                    window.speechSynthesis.speak(u);
                }} catch (e) {{
                    console.error("Smart Machine TTS error:", e);
                }}
            }};

            speakNext();
        }}

        speakSequence();
    }})();
    </script>
    """

    components.html(html, height=0, scrolling=False)
    st.info(f"🔊 Speaking advice in order: **{order}** (allow sound in browser).")


# ----------------- Always-on Wake Word Listener (SMART MACHINE) ----------- #

def smart_machine_listener(
    advice_hi: str,
    advice_en: str,
    wake_word: str = "smart machine",
    order: str = "hi-en",
) -> None:
    """
    Browser ke Web Speech API se listening:
    - Continuous listening (jab toggle ON ho)
    - Jaise hi user bole 'smart machine' (thoda sa variation bhi chalega),
      turant laptop se Hindi + English advice awaaz me.
    - Koi speech_recognition / PyAudio nahi, sirf JS + mic permission browser me.

    NOTE:
    - App side se typical use:
        hi, en = build_smart_advice_bilingual(df)
        if enable_listener:
            smart_machine_listener(hi, en)
    """

    hi_chunks = _to_chunks(advice_hi, lang="hi")
    en_chunks = _to_chunks(advice_en, lang="en")

    cfg = {
        "hi_chunks": hi_chunks,
        "en_chunks": en_chunks,
        "order": order or "hi-en",
        "wake_word": (wake_word or "smart machine").lower(),
    }
    payload = json.dumps(cfg, ensure_ascii=False)

    html = f"""
    <script>
    (function() {{
        const cfg = {payload};

        // Check for SpeechRecognition support
        const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRec) {{
            console.log("Smart Machine: SpeechRecognition not available in this browser.");
            return;
        }}

        if (!("speechSynthesis" in window)) {{
            console.log("Smart Machine: speechSynthesis not available.");
            return;
        }}

        const rec = new SpeechRec();
        rec.lang = "en-IN";  // Hindi + English mix ke liye kaafi acha
        rec.continuous = true;
        rec.interimResults = true;

        let lastTrigger = 0;
        const COOLDOWN_MS = 6000; // 6 sec gap between two replies

        function speakSequence() {{
            const parts = cfg.order === "hi-en" ? ["hi", "en"] : ["en", "hi"];
            const queue = [];

            parts.forEach((key) => {{
                const chunkKey = key === "hi" ? "hi_chunks" : "en_chunks";
                const chunks = cfg[chunkKey] || [];
                chunks.forEach((txt) => {{
                    if (txt && txt.trim().length > 0) {{
                        queue.push({{
                            text: txt.trim(),
                            lang: key === "hi" ? "hi-IN" : "en-IN"
                        }});
                    }}
                }});
            }});

            if (!queue.length) return;

            window.speechSynthesis.cancel();

            const speakNext = () => {{
                if (!queue.length) return;
                const item = queue.shift();
                const u = new SpeechSynthesisUtterance(item.text);
                u.lang = item.lang;

                // ⚡ Energetic Female Coach Mode (wake-word based)
                try {{
                    const voices = window.speechSynthesis.getVoices() || [];
                    const targetLang = item.lang === "hi-IN" ? "hi" : "en";
                    const preferred = voices.find(v =>
                        (
                            v.name.toLowerCase().includes("female") ||
                            v.name.toLowerCase().includes("woman") ||
                            v.name.toLowerCase().includes("girl") ||
                            v.name.toLowerCase().includes("coach") ||
                            v.name.toLowerCase().includes("strong") ||
                            v.name.toLowerCase().includes("premium") ||
                            v.name.toLowerCase().includes("neural") ||
                            v.name.toLowerCase().includes("natural") ||
                            v.name.toLowerCase().includes("wavenet")
                        ) &&
                        v.lang.toLowerCase().includes(targetLang)
                    );
                    if (preferred) {{
                        u.voice = preferred;
                    }}
                }} catch (e) {{
                    console.log("Voice selection error:", e);
                }}

                u.rate = 1.15;   // energetic coach feel
                u.pitch = 1.15;
                u.volume = 3.0;

                u.onend = speakNext;
                try {{
                    window.speechSynthesis.speak(u);
                }} catch (e) {{
                    console.error("Smart Machine TTS error:", e);
                }}
            }};

            speakNext();
        }}

        function handleTranscript(text) {{
            if (!text) return;
            const now = Date.now();
            const normalized = text.toLowerCase();

            // Approx wake-word matching
            if (
                normalized.includes(cfg.wake_word) ||
                normalized.includes("smart machine") ||
                normalized.includes("smar machine") ||
                normalized.includes("smart masheen") ||
                normalized.includes("स्मार्ट मशीन")
            ) {{
                if (now - lastTrigger > COOLDOWN_MS) {{
                    lastTrigger = now;
                    console.log("Smart Machine: wake word detected -> speaking advice");
                    speakSequence();
                }} else {{
                    console.log("Smart Machine: wake word ignored (cooldown)");
                }}
            }}
        }}

        rec.onresult = (event) => {{
            let full = "";
            for (let i = event.resultIndex; i < event.results.length; i++) {{
                full += event.results[i][0].transcript;
            }}
            handleTranscript(full);
        }};

        rec.onerror = (e) => {{
            console.log("Smart Machine recognition error:", e);
        }};

        rec.onend = () => {{
            // Restart for always-on feel (jab tak toggle se component re-render ho raha hai)
            try {{
                rec.start();
            }} catch (e) {{
                console.log("Smart Machine: restart failed", e);
            }}
        }};

        try {{
            rec.start();
            console.log("Smart Machine listener: started. Say 'smart machine' near mic.");
        }} catch (e) {{
            console.log("Smart Machine: start failed", e);
        }}
    }})();
    </script>
    """

    components.html(html, height=0, scrolling=False)
    st.caption("🎧 **Smart Machine listening…** Browser se mic permission allow karo, phir bolo: **“smart machine”**.")


# ----------------- Placeholders for other imports ------------------------- #

def gen_viz_spec(*args, **kwargs) -> Dict[str, Any]:
    """Placeholder so import na toote; app abhi main gen_viz module se use kar raha hai."""
    return {
        "description": "Mock viz spec from ai_helpers.gen_viz_spec (not used in main flow).",
        "spec": {
            "data": [
                {"type": "bar", "x": ["Income", "Expense"], "y": [1, 1]}
            ],
            "layout": {"title": "Mock Chart"},
        },
    }


def chat_reply(prompt: str, history: List[Tuple[str, str]] | None = None, context: str = "") -> str:
    """Simple placeholder chat reply (in case kahin use ho jaaye)."""
    return (
        "PRAKRITI AI (mock reply): I am a lightweight helper from ai_helpers.py. "
        "Main detailed logic ke liye app.py ke main chatbot ko use karta hoon. 😊"
    )


def gemini_enabled() -> bool:
    """Simple flag; yahan se always True de sakte hain ya env ke hisaab se decide kar sakte ho."""
    return True
