
from __future__ import annotations

import os
import base64
import joblib
import json
import requests
import time
import random
import re
from io import BytesIO
from pathlib import Path
from datetime import date, datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Any, Tuple

# ===== IMPORTS (already there) =====
import streamlit as st
import cv2
import mediapipe as mp
haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_path)

# ===== PASTE THIS HERE (GLOBAL) =====
RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {
                "urls": [f"turn:{st.secrets['TURN_DOMAIN']}:80"],
                "username": st.secrets["TURN_DOMAIN"],
                "credential": st.secrets["TURN_SECRET"],
            },
            {
                "urls": [f"turn:{st.secrets['TURN_DOMAIN']}:443?transport=tcp"],
                "username": st.secrets["TURN_DOMAIN"],
                "credential": st.secrets["TURN_SECRET"],
            },
        ]
    }
)

# === NEW MODULE IMPORTS ===
from analytics import (
    compute_fin_health_score,
    no_spend_streak,
    detect_trend_spikes,
    forecast_next_month,
    auto_allocate_budget,
)

import pandas as pd

def _safe_to_date(x) -> date:
    """Return a real python date; fallback to today if x is empty/invalid."""
    try:
        dt = pd.to_datetime(x, errors="coerce")
        if pd.isna(dt):
            return date.today()
        return dt.date()
    except Exception:
        return date.today()

# Import OCR helpers
from ocr import HAS_TESSERACT # noqa: F401
# Import Telegram helpers
from telegram_utils import send_report_png
# Import Weather helpers
from weather import get_weather, spend_mood_hint
# Import Generative Viz helper
from gen_viz import suggest_infographic_spec, _static_fallback_viz
# Import Custom UI helpers
from ui_patches import (
    display_health_score,
    display_badges,
    budget_bot_minicard,
    money,
    glowing_ocr_uploader,
)

from helper import (
    build_smart_advice_bilingual,
    speak_bilingual_js,
    smart_machine_listener,
    gen_viz_spec,  # noqa: F401
    chat_reply,    # noqa: F401
    gemini_enabled # noqa: F401
)

# Import Gemini SDK (optional)
try:
    from google import genai
    HAS_GEMINI_SDK = True
except ImportError:
    HAS_GEMINI_SDK = False

# Import OpenAI SDK (optional)
try:
    from openai import OpenAI
    HAS_OPENAI_SDK = True
except ImportError:
    HAS_OPENAI_SDK = False

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw
from sklearn.feature_extraction.text import TfidfVectorizer # noqa: F401
from sklearn.metrics.pairwise import cosine_similarity        # noqa: F401
import qrcode

st.set_page_config(page_title="Cash Flow Crew — Personal Finance AI Analyzer", page_icon="📈💰📊", layout="wide")
# ============================================================
# 🏙️ NEW: City Affordability (inlined module)
# ============================================================

import unicodedata

# presets: (index, avg_rent, avg_food, avg_utilities, tier)
# Tiers are determined by: T1 (>110), T2 (85-110), T3 (<=85)
CITY_PRESETS: Dict[str, Tuple[int, int, int, int, str]] = {
    # Tier-1 (>110)
    "Bengaluru": (125, 17000, 7000, 2500, "Tier-1"),
    "Mumbai": (140, 22000, 7500, 3000, "Tier-1"),
    "Delhi": (130, 16000, 6500, 2800, "Tier-1"),
    "Gurugram": (128, 19000, 6800, 2800, "Tier-1"),
    "Noida": (120, 15000, 6200, 2600, "Tier-1"),
    "Hyderabad": (115, 14000, 6200, 2600, "Tier-1"),
    "Pune": (118, 15000, 6500, 2600, "Tier-1"),
    "Chennai": (116, 14000, 6200, 2600, "Tier-1"),
    # Tier-2 (85 - 110)
    "Kolkata": (110, 12000, 5800, 2400, "Tier-2"),
    "Ahmedabad": (104, 11000, 5600, 2200, "Tier-2"),
    "Surat": (100, 10000, 5200, 2200, "Tier-2"),
    "Nagpur": (98, 9000, 5200, 2100, "Tier-2"),
    "Lucknow": (95, 9000, 5000, 2100, "Tier-2"),
    "Jaipur": (94, 9000, 5000, 2100, "Tier-2"),
    "Indore": (92, 8500, 4800, 2000, "Tier-2"),
    "Varanasi": (92, 8000, 4800, 2000, "Tier-2"),
    "Bhopal": (90, 8000, 4700, 2000, "Tier-2"),
    "Ranchi": (90, 8000, 4700, 2000, "Tier-2"),
    "Kanpur": (88, 8000, 4600, 2000, "Tier-2"),
    "Patna": (88, 8000, 4600, 2000, "Tier-2"),
    "Kochi": (100, 10000, 5200, 2200, "Tier-2"),
    "Thiruvananthapuram": (96, 9000, 5000, 2100, "Tier-2"),
    "Visakhapatnam": (97, 9500, 5100, 2100, "Tier-2"),
    "Coimbatore": (95, 9000, 5000, 2100, "Tier-2"),
    # Tier-3 (<= 85)
    "Prayagraj": (85, 7500, 4500, 1900, "Tier-3"), 
    "Agra": (80, 7000, 4200, 1800, "Tier-3"), 
}

# --- New function to fetch cities using Gemini (Dynamic or Fallback) ---
@st.cache_data(ttl=timedelta(days=7))
def get_cities_from_gemini() -> Dict[str, Tuple[int, int, int, int, str]]:
    key = os.environ.get("GEMINI_API_KEY") or ""
    if not (HAS_GEMINI_SDK and key.strip()):
        return CITY_PRESETS

    try:
        client = genai.Client(api_key=key.strip())
        prompt = """
        Provide a list of 20 diverse Indian cities, spanning Tier 1, 2, and 3 classifications based on average living cost. 
        For each city, provide: City Cost Index (Base 100 for a middle tier city, roughly 80-140 range), Average Monthly Rent for a 1 BHK, Average Monthly Food Cost, Average Monthly Utilities Cost, and the City Tier (Tier-1, Tier-2, or Tier-3).
        
        Return the response strictly as a JSON array of objects.
        Example item structure: {"city": "Hyderabad", "index": 115, "rent": 14000, "food": 6200, "utilities": 2600, "tier": "Tier-1"}
        """
        response = client.models.generate_content(
            # FIX: Switched to faster model for stability
            model="gemini-2.5-flash-lite", 
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            config={
                "response_mime_type": "application/json",
                "response_schema": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "city": {"type": "STRING", "description": "City Name"},
                            "index": {"type": "INTEGER", "description": "City Cost Index (70-160)"},
                            "rent": {"type": "INTEGER", "description": "Average Rent (INR)"},
                            "food": {"type": "INTEGER", "description": "Average Food (INR)"},
                            "utilities": {"type": "INTEGER", "description": "Average Utilities (INR)"},
                            "tier": {"type": "STRING", "description": "Tier-1, Tier-2, or Tier-3"},
                        },
                        "required": ["city", "index", "rent", "food", "utilities", "tier"],
                    }
                }
            }
        )
        
        data_list = json.loads(response.text)
        
        if not isinstance(data_list, list) or not data_list:
            return CITY_PRESETS 

        dynamic_presets = {}
        for item in data_list:
            city_name = item.get("city")
            idx = item.get("index")
            rent = item.get("rent")
            food = item.get("food")
            util = item.get("utilities")
            tier = item.get("tier")

            if all([city_name, idx is not None, rent is not None, food is not None, util is not None, tier]):
                dynamic_presets[city_name.title()] = (idx, rent, food, util, tier)
        
        return dynamic_presets or CITY_PRESETS
        
    except Exception as e:
        return CITY_PRESETS

ALL_CITIES = get_cities_from_gemini()
CITY_INDEX_FALLBACK = {k.lower(): v[0] for k, v in ALL_CITIES.items()}
BASE_LIVING_WAGE = 35000  # baseline @ index 100

PROFILE_PRESETS = {
    # multipliers applied to non-rent envelopes; transport is further affected by commute
    "Student": {"food": 0.9, "utilities": 0.9, "discretionary": 0.8, "transport": 1.0},
    "Working Professional": {"food": 1.0, "utilities": 1.0, "discretionary": 1.0, "transport": 1.0},
    "Couple": {"food": 1.6, "utilities": 1.2, "discretionary": 1.2, "transport": 1.1},
    "Family": {"food": 2.0, "utilities": 1.4, "discretionary": 1.5, "transport": 1.2},
}

def _norm_city_name(s: str) -> str:
    s = (s or "").strip()
    s = unicodedata.normalize("NFKC", s)
    return " ".join(s.replace(".", " ").replace("-", " ").replace("_", " ").split())

def _money_ci(x: int | float) -> str:
    # This must match money() logic if imported
    return f"₹{int(round(x)):,}"

@dataclass
class AffResult:
    city: str
    income: int
    index: int
    living_need: int
    bucket: str
    gap: int

def _baseline_from_index(idx: int) -> int:
    lw = BASE_LIVING_WAGE * (idx / 100)
    return int(round(lw / 500.0) * 500)

def _bucket_from_ratio(ratio: float) -> str:
    if ratio < 0.70: return "very expensive"
    if ratio < 0.90: return "expensive"
    if ratio <= 1.10: return "fare"
    if ratio <= 1.40: return "low expensive"
    return "no expensive"

def _badge_html(cat: str) -> str:
    colors = {
        "very expensive": "#ef4444",
        "expensive": "#f97316",
        "fare": "#22c55e",
        "low expensive": "#84cc16",
        "no expensive": "#06b6d4",
    }
    c = colors.get(cat, "#64748b")
    return f"<span style='background:{c};color:#fff;padding:4px 10px;border-radius:999px;font-weight:700'>{cat.upper()}</span>"

def _refine_need(
    base_lw: int,
    avg_rent: int,
    avg_food: int,
    avg_utils: int,
    sharing: int,
    locality: str,
    commute: str,
    profile: str,
) -> int:
    sharing = max(1, min(5, int(sharing)))
    # Rent after sharing + locality multiplier on rent
    loc_mul = {"Basic": 0.9, "Average": 1.0, "Prime": 1.15}.get(locality, 1.0)
    rent_refined = (avg_rent * loc_mul) / sharing

    # Base breakdown assumption from baseline: 30% rent, 25% food, 10% utilities, 35% other (transport+discretionary)
    base_rent = base_lw * 0.30
    base_food = base_lw * 0.25
    base_utils = base_lw * 0.10
    base_other = base_lw - (base_rent + base_food + base_utils)

    prof = PROFILE_PRESETS.get(profile, PROFILE_PRESETS["Working Professional"])
    # Commute multiplier on transport portion of "other"
    commute_mul = {"Low-cost (bus/metro)": 0.95, "Mixed": 1.0, "Cab-heavy": 1.10}.get(commute, 1.0)

    # Replace baseline envelopes with city presets * profile multipliers; keep other scaled by profile+commute
    food_ref = avg_food * prof["food"]
    utils_ref = avg_utils * prof["utilities"]
    other_ref = (base_other * prof["discretionary"]) * prof["transport"] * commute_mul

    refined = rent_refined + food_ref + utils_ref + other_ref
    return int(round(refined / 500.0) * 500)

def classify_city_income(
    income: int,
    city_name: str,
    idx: int,
    avg_rent: int,
    avg_food: int,
    avg_utils: int,
    sharing: int,
    locality: str,
    commute: str,
    profile: str,
) -> AffResult:
    base_lw = _baseline_from_index(idx)
    need = _refine_need(base_lw, avg_rent, avg_food, avg_utils, sharing, locality, commute, profile)
    ratio = (income / need) if need > 0 else 2.0
    return AffResult(city=city_name, income=income, index=idx, living_need=need, bucket=_bucket_from_ratio(ratio), gap=income - need)

def _get_tier_from_index(idx: int) -> str:
    """Classify tier based on cost index."""
    if idx > 110: return "Tier-1"
    if idx > 85: return "Tier-2"
    return "Tier-3"

def _gemini_aff_text(city: str, income: int, res: AffResult, lang: str = "en") -> str:
    key = os.environ.get("GEMINI_API_KEY") or ""
    def fallback() -> str:
        norm = _norm_city_name(city).lower()
        lines = []
        if norm in ("bengaluru", "bangalore") and income <= 30000:
            lines.append("Bengaluru with ₹30k is not good due to higher rent & commute.")
        if norm in ("prayagraj", "allahabad") and income >= 30000:
            lines.append("Prayagraj with ₹30k is fine for a single person.")
        tip = {
            "very expensive": "Well below modest living; share rent and pick basic locality.",
            "expensive": "Below need; consider roommates and metro-first commute.",
            "fare": "Near break-even; track groceries and transport closely.",
            "low expensive": "Comfortable surplus; automate SIPs and build EF.",
            "no expensive": "Strong surplus; raise SIPs and keep 6-month EF.",
        }[res.bucket]
        if lang == "hi":
            return (
                f"{city.title()} में साधारण गुज़ारा लगभग {_money_ci(res.living_need)} है। आपकी आय {_money_ci(income)} होने पर यह **{res.bucket}** है। "
                f"{'बेंगलुरु में 30k ठीक नहीं। ' if norm in ('bengaluru','bangalore') and income<=30000 else ''}"
                f"{'प्रयागराज में 30k ठीक-ठाक है। ' if norm in ('prayagraj','allahabad') and income>=30000 else ''}"
                f"किराया शेयर करें, सस्ती लोकैलिटी/मेट्रो चुनें, और हर महीने SIP करें।"
            )
        base = f"For {city.title()}, refined living need ≈ {_money_ci(res.living_need)}. With {_money_ci(income)}, this is **{res.bucket}**. {tip}"
        if lines: base += " (" + " ".join(lines) + ")"
        return base

    if not (HAS_GEMINI_SDK and key.strip()):
        return fallback()
    try:
        client = genai.Client(api_key=key.strip())
        prompt = f"""
You are an Indian city affordability assistant.
City: {city}
Income: ₹{income:,}
Refined living need: ₹{res.living_need:,}
Bucket: {res.bucket}
Tier: {res.city} is classified as {_get_tier_from_index(res.index)} based on a cost index of {res.index}.

Write 3–5 short sentences in {"Hindi" if lang=="hi" else "English"}.
Clearly state the Tier classification and if the city is okay or not for ₹{income:,} (e.g., "Bengaluru 30k is not good", "Prayagraj is fine").
Give 2 quick cost levers (rent-sharing/locality/commute). End with one saving tip (SIP/emergency fund).
"""
        out = client.models.generate_content(model="gemini-2.5-flash", contents=[{"role": "user", "parts": [{"text": prompt}]}])
        return (out.text or "").strip()
    except Exception:
        return fallback()

def _tts_button(elem_id: str, text: str, lang_code: str = "en-IN", rate: float = 1.05, pitch: float = 1.0):
    safe = (text or "").replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ").strip()
    components.html(
        f"""
<button id="{elem_id}" style="background:#0ea5e9;color:#fff;border:none;border-radius:8px;padding:8px 12px;font-weight:700;cursor:pointer;margin:.25rem 0">🔊 Read Aloud ({lang_code})</button>
<script>
(function(){{
    const b = document.getElementById("{elem_id}");
    if(!b) return;
    b.onclick = function(){{
        try {{
            const u = new SpeechSynthesisUtterance("{safe}");
            u.lang = "{lang_code}";
            u.rate = {rate};
            u.pitch = {pitch};
            window.speechSynthesis.cancel();
            window.speechSynthesis.speak(u);
        }} catch(e) {{ console.warn(e); }}
    }}
}})();
</script>
        """,
        height=42,
    )

# --- NEW: Chart Explainer Functions ---

# FIX: RE-ENABLED caching (for performance) and switched to local analysis
@st.cache_data(ttl=timedelta(days=1))
def _gemini_explain_chart(chart_name: str, context: str, lang: str = "en") -> str:
    """Generates a dynamic explanation *without* calling the remote Gemini API, using local data context."""
    
    # Analyze the context string to extract key info for dynamic analysis
    is_empty_context = ("No data" in context) or (context.endswith(":"))

    if is_empty_context:
        if lang == "hi":
            return "⚠️ **डेटा नहीं मिला:** चार्ट के लिए कोई लेन-देन नहीं है। कृपया अपनी तारीख और फ़िल्टर जांचें। 📊"
        return f"⚠️ **Data Unavailable:** No transactions found for this chart. Please check your filters. Context: {context}"

    # --- Local Dynamic Analysis (Simulated AI) ---
    
    # 1. Extract Money Metrics (G1)
    income_match = re.search(r"Total Income: ([\S]+)\.", context)
    expense_match = re.search(r"Total Expense: ([\S]+)\.", context)
    
    try:
        if income_match and expense_match:
            # Safely extract and clean monetary values
            total_income = float(income_match.group(1).replace('₹', '').replace(',', ''))
            total_expense = float(expense_match.group(1).replace('₹', '').replace(',', ''))
            net_savings = total_income - total_expense
            
            # Dynamic Insight for G1 (Donut Chart)
            if "Donut Chart" in chart_name:
                savings_rate = (net_savings / total_income) * 100 if total_income > 0 else 0
                if lang == "hi":
                    return f"💰 **आय विश्लेषण:** कुल आय {_money_ci(total_income)} है। आपकी बचत दर लगभग {savings_rate:.0f}% है। इस दर को बढ़ाने के लिए अपने व्यय को ट्रैक करें! 📈"
                return f"💰 **Income Analysis:** Total income is {_money_ci(total_income)}. Your savings rate is approximately {savings_rate:.0f}%. Track your expenditure to boost this rate! 📈"

            # Dynamic Insight for G2 (Cash Flow Trend)
            if "Cash Flow Trend" in chart_name:
                trend = "positive" if net_savings > 0 else "negative" if net_savings < 0 else "balanced"
                if lang == "hi":
                    return f"💸 **कैश फ्लो:** कुल शुद्ध बचत {_money_ci(net_savings)} है। यह **{trend}** है। आपको अपनी बचत जारी रखनी चाहिए और बड़े खर्चों की योजना बनानी चाहिए। 💪"
                return f"💸 **Cash Flow:** Total net savings is {_money_ci(net_savings)}. This trend is **{trend}**. Plan major expenses carefully to maintain this. 💪"

    except Exception:
        # Fallback if complex parsing fails, use generic analysis
        pass

    # Generic Dynamic Analysis for other charts (G3, G4, G5)
    if lang == "hi":
        return f"📊 **चार्ट अवलोकन:** यह चार्ट आपके {chart_name.replace('Graph', 'ग्राफ')} के लिए डेटा का सारांश दिखाता है। सभी फ़िल्टर आपके डेटा को गतिशील रूप से अपडेट करते हैं। 🔄"
    return f"📊 **Chart Overview:** This chart shows a summary of data for your {chart_name}. All filters dynamically update the data presented here. 🔄"


def _chart_explainer(chart_id: str, chart_name: str, chart_context: str) -> None:
    """Renders the bilingual explanation and TTS buttons for a given chart."""
    st.markdown("---")
    st.markdown(f"#### 🧠 AI Analysis for {chart_id}: {chart_name}")

    c_en, c_hi = st.columns(2)
    
    # Generate bilingual explanations (cached)
    explanation_en = _gemini_explain_chart(chart_name, chart_context, lang="en")
    explanation_hi = _gemini_explain_chart(chart_name, chart_context, lang="hi")

    with c_en:
        st.caption("English Explanation")
        st.markdown(f"**{explanation_en}**")
        _tts_button(f"tts_{chart_id}_en", explanation_en, "en-IN")
        
    with c_hi:
        st.caption("हिंदी में विश्लेषण")
        st.markdown(f"**{explanation_hi}**")
        _tts_button(f"tts_{chart_id}_hi", explanation_hi, "hi-IN", rate=1.0, pitch=1.05)

# --- END Chart Explainer Functions ---


def render_city_affordability_tab() -> None:
    st.header("🏙️ City Affordability Analyzer (Gemini Powered)")
    st.caption("City + income → very expensive / expensive / **fare** / low expensive / no expensive. Dynamic city selection, flexible inputs, and visual comparison.")

    presets = list(ALL_CITIES.keys())
    # Try to keep Bengaluru as default if it exists
    default_index = presets.index("Bengaluru") if "Bengaluru" in presets else 0
    
    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        preset_city = st.selectbox("City preset (Gemini Powered) 🏙️", presets, index=default_index)
    with c2:
        monthly_income = st.number_input("Monthly Income (₹) 💵", min_value=1000, step=1000, value=30000)
    with c3:
        advice_lang = st.selectbox("Advice language 🗣️", ["English", "Hindi (हिंदी)", "Both"], index=0)

    # Ensure we use the values from ALL_CITIES dictionary
    try:
        idx_def, rent_def, food_def, util_def, tier = ALL_CITIES[preset_city]
    except KeyError:
        # Fallback if selected city is somehow missing
        idx_def, rent_def, food_def, util_def, tier = list(ALL_CITIES.values())[0]

    # Dynamically determine the tier based on the selected index
    actual_tier = _get_tier_from_index(idx_def)
    st.write(f"**Preset:** {preset_city} • **Tier:** {actual_tier} • **Index:** {idx_def} • **Avg rent/food/util:** {_money_ci(rent_def)} / {_money_ci(food_def)} / {_money_ci(util_def)}")

    r1, r2, r3, r4 = st.columns([1, 1, 1, 1])
    with r1:
        city_name = st.text_input("City (override optional)", value=preset_city)
    with r2:
        idx_val = st.slider("City cost index", 70, 160, idx_def, help="100 ≈ tier-2 baseline")
    with r3:
        sharing = st.slider("Flatmates (people sharing) 👥", 1, 5, 2)
    with r4:
        profile = st.selectbox("Profile 👤", ["Student", "Working Professional", "Couple", "Family"], index=1)

    # --- Multi-Colored Buttons Implementation ---
    st.markdown("""
        <style>
        /* Custom radio button styles for multi-color */
        .multicolor-radio > div[data-testid="stRadio"] label:nth-child(1) span { background-color: #ffeb3b; color: #1e1e1e; border-color: #ffeb3b; } /* Yellow */
        .multicolor-radio > div[data-testid="stRadio"] label:nth-child(2) span { background-color: #ff9800; color: white; border-color: #ff9800; } /* Orange */
        .multicolor-radio > div[data-testid="stRadio"] label:nth-child(3) span { background-color: #2196f3; color: white; border-color: #2196f3; } /* Blue */

        .multicolor-radio-commute > div[data-testid="stRadio"] label:nth-child(1) span { background-color: #4caf50; color: white; border-color: #4caf50; } /* Green */
        .multicolor-radio-commute > div[data-testid="stRadio"] label:nth-child(2) span { background-color: #ff9800; color: white; border-color: #ff9800; } /* Orange */
        .multicolor-radio-commute > div[data-testid="stRadio"] label:nth-child(3) span { background-color: #f44336; color: white; border-color: #f44336; } /* Red */

        .multicolor-radio div[data-testid="stRadio"] label span,
        .multicolor-radio-commute div[data-testid="stRadio"] label span {
            padding: 8px 12px;
            border-radius: 8px;
            font-weight: 700;
            transition: all 0.2s;
        }
        .multicolor-radio div[data-testid="stRadio"] input:checked + div > span,
        .multicolor-radio-commute div[data-testid="stRadio"] input:checked + div > span {
            border: 3px solid #6a5acd !important; /* Purple border for selected */
            box-shadow: 0 0 10px rgba(106, 90, 205, 0.7);
        }
        </style>
        """, unsafe_allow_html=True)
    
    l1, l2, l3 = st.columns([1, 1, 1])
    with l1:
        st.markdown('<div class="multicolor-radio">', unsafe_allow_html=True)
        loc = st.radio("Locality 🏡", ["Basic", "Average", "Prime"], index=1, horizontal=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with l2:
        st.markdown('<div class="multicolor-radio-commute">', unsafe_allow_html=True)
        commute = st.radio("Commute 🚌", ["Low-cost (bus/metro)", "Mixed", "Cab-heavy"], index=1, horizontal=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with l3:
        rent_override = st.number_input("Avg rent (₹) 🏠", min_value=3000, step=500, value=rent_def)

    f1, f2 = st.columns([1, 1])
    with f1:
        food_override = st.number_input("Avg food (₹) 🍕", min_value=2000, step=200, value=food_def, help="Monthly groceries+simple eating out")
    with f2:
        util_override = st.number_input("Avg utilities (₹) 💡", min_value=1000, step=100, value=util_def, help="Electricity + internet + phone")

    if st.button("Analyze City Affordability", use_container_width=True):
        res = classify_city_income(
            int(monthly_income),
            city_name.strip() or preset_city,
            int(idx_val),
            int(rent_override),
            int(food_override),
            int(util_override),
            int(sharing),
            loc,
            commute,
            profile,
        )
        st.markdown("### Result")
        st.markdown(
            f"""
- **City:** {res.city.title()}
- **Tier Classification (derived from index):** **{_get_tier_from_index(res.index)}**
- **Income:** {_money_ci(res.income)}
- **Refined living need:** {_money_ci(res.living_need)}
- **Gap:** {_money_ci(res.gap)} ({'surplus' if res.gap >= 0 else 'deficit'})
- **Bucket:** {_badge_html(res.bucket)}
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Advice")
        st.info("The tier classification (Tier 1, 2, or 3) is automatically generated by rules based on the City Cost Index, mirroring common geographical/economic classifications often referenced by large language models like Gemini.")
        if advice_lang.startswith("English") or advice_lang == "Both":
            p_en = _gemini_aff_text(res.city, res.income, res, lang="en")
            st.write(p_en)
            _tts_button("tts_en_city", p_en, "en-IN")
        if advice_lang.startswith("Hindi") or advice_lang == "Both":
            p_hi = _gemini_aff_text(res.city, res.income, res, lang="hi")
            if advice_lang == "Both":
                with st.expander("Hindi (हिंदी)"):
                    st.write(p_hi)
                    _tts_button("tts_hi_city", p_hi, "hi-IN", rate=1.0, pitch=1.05)
            else:
                st.write(p_hi)
                _tts_button("tts_hi_city", p_hi, "hi-IN", rate=1.0, pitch=1.05)

        # --- Dynamic Comparison Table ---
        st.markdown("---")
        st.markdown("### City Comparison Table (Filtered)")
        sample_cities = list(ALL_CITIES.keys()) # Use all cities for filtering
        rows = []
        for c in sample_cities:
            ci = ALL_CITIES.get(c)
            if not ci: continue
            r = classify_city_income(
                int(monthly_income), c, ci[0],
                int(rent_override if c == res.city else ci[1]),
                int(food_override if c == res.city else ci[2]),
                int(util_override if c == res.city else ci[3]),
                int(sharing), loc, commute, profile
            )
            rows.append({
                "City": c,
                "Tier": _get_tier_from_index(r.index),
                "Index": r.index,
                "Avg Rent (₹)": ci[1],
                "Avg Food (₹)": ci[2],
                "Avg Utilities (₹)": ci[3],
                "Refined Need (₹)": r.living_need,
                "Your Income (₹)": r.income,
                "Gap (₹)": r.gap,
                "Bucket": r.bucket,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


        # --- NEW FEATURE: Dynamic City Comparison Chart (Tier 1, 2, and 3) ---
        st.markdown("---")
        st.subheader("📊 Dynamic Cost Comparison Chart (Tier 1, 2, 3)")
        
        all_tiers = ["Tier-1", "Tier-2", "Tier-3"]
        selected_tiers = st.multiselect(
            "Select Tiers to Compare", 
            options=all_tiers, 
            default=all_tiers,
            key='tier_select'
        )

        # Use the full city list to create the comparison data frame
        all_cities_comp_rows = []
        
        for city, (idx, avg_rent, avg_food, avg_utils, _) in ALL_CITIES.items():
            need = _refine_need(
                base_lw=_baseline_from_index(idx),
                avg_rent=avg_rent,
                avg_food=avg_food,
                avg_utils=avg_utils,
                sharing=sharing, # Use value from form 
                locality=loc,    # Use value from form 
                commute=commute, # Use value from form 
                profile=profile  # Use value from form 
            )
            all_cities_comp_rows.append({
                "City": city,
                "Refined Need (₹)": need,
                "Tier": _get_tier_from_index(idx),
                "Cost Index": idx
            })

        comp_df = pd.DataFrame(all_cities_comp_rows)
        # Filter by selected tiers
        comp_df_filtered = comp_df[comp_df['Tier'].isin(selected_tiers)].sort_values("Refined Need (₹)", ascending=False)
        
        if comp_df_filtered.empty:
            st.warning("Please select at least one city tier to display the comparison chart.")
        else:
            fig_comp = px.bar(
                comp_df_filtered,
                x="City",
                y="Refined Need (₹)",
                color="Tier",
                title=f"Refined Living Need Comparison for Selected Tiers ({', '.join(selected_tiers)})",
                text="Refined Need (₹)",
                color_discrete_map={
                    "Tier-1": "#ef4444", 
                    "Tier-2": "#f97316", 
                    "Tier-3": "#22c55e"
                }
            )
            fig_comp.update_traces(texttemplate='₹%{y:,.0f}', textposition='outside')
            fig_comp.update_layout(height=550, template="plotly_dark")
            st.plotly_chart(fig_comp, use_container_width=True)


        st.markdown("---")
        st.markdown("#### Read any paragraph aloud")
        any_text = st.text_area("Paste paragraph", value="Bengaluru with ₹30k is not good; Prayagraj is fine for a single person.", height=90)
        cA, cB = st.columns(2)
        with cA:
            _tts_button("tts_custom_en", any_text, "en-IN")
        with cB:
            _tts_button("tts_custom_hi", any_text, "hi-IN", rate=1.0, pitch=1.05)

# ============================================================
# 🧑‍💼 NEW: Personal CA Financial Plan Generator
# ============================================================

def generate_ca_financial_plan(life_stage: str, city: str, monthly_income: int, monthly_expenses: Optional[int] = None) -> Tuple[str, str, dict]:
    """
    Generates a full financial blueprint based on life stage, city, and income.
    Returns: (detailed_explanation, tts_summary, plan_json)
    """
    # --- 1. CITY ANALYSIS (Simplified) ---
    city_cost_data = {
        "bengaluru": {"cost_level": "VERY HIGH", "rent_factor": 0.35, "food_factor": 1.2, "transport_factor": 1.15, "min_rent": 18000},
        "mumbai": {"cost_level": "VERY HIGH", "rent_factor": 0.40, "food_factor": 1.25, "transport_factor": 1.2, "min_rent": 22000},
        "delhi": {"cost_level": "HIGH", "rent_factor": 0.30, "food_factor": 1.1, "transport_factor": 1.1, "min_rent": 15000},
        "hyderabad": {"cost_level": "MEDIUM", "rent_factor": 0.25, "food_factor": 1.0, "transport_factor": 1.0, "min_rent": 12000},
        "kolkata": {"cost_level": "LOW", "rent_factor": 0.20, "food_factor": 0.9, "transport_factor": 0.9, "min_rent": 8000},
        # Fallback for unlisted cities
        "default": {"cost_level": "MEDIUM", "rent_factor": 0.25, "food_factor": 1.0, "transport_factor": 1.0, "min_rent": 10000},
    }
    city_norm = city.lower()
    city_config = city_cost_data.get(city_norm, city_cost_data["default"])
    if "bengaluru" in city_norm: city_config = city_cost_data["bengaluru"]

    # --- 2. LIFE STAGE RULESET (India-Focused 50/30/20 or similar adaptation) ---
    ruleset = {
        "Student": {"rent_pct_max": 0.15, "savings_pct": 0.10, "ef_months": 1, "sip_pct": 0.05, "lifestyle_pct": 0.15, "misc_pct": 0.10},
        "Fresher": {"rent_pct_max": 0.25, "savings_pct": 0.15, "ef_months": 3, "sip_pct": 0.10, "lifestyle_pct": 0.10, "misc_pct": 0.10},
        "Early Career": {"rent_pct_max": 0.30, "savings_pct": 0.25, "ef_months": 6, "sip_pct": 0.15, "lifestyle_pct": 0.10, "misc_pct": 0.05},
        "Family": {"rent_pct_max": 0.30, "savings_pct": 0.25, "ef_months": 6, "sip_pct": 0.10, "lifestyle_pct": 0.10, "misc_pct": 0.05},
        "Retirement": {"rent_pct_max": 0.0, "savings_pct": 0.30, "ef_months": 12, "sip_pct": 0.05, "lifestyle_pct": 0.10, "misc_pct": 0.10}, # Assuming owned house -> 0 rent cap
    }
    user_rules = ruleset.get(life_stage, ruleset["Early Career"])

    # SIP and Savings Calculation
    ideal_savings_pct = user_rules["savings_pct"] * 100
    ideal_savings_amount = monthly_income * user_rules["savings_pct"]
    sip_target_pct = user_rules["sip_pct"]
    suggested_sip = int(round(monthly_income * sip_target_pct))
    
    # Emergency Fund Target (SIP target is taken out of the savings pool)
    emergency_fund_target = user_rules["ef_months"]
    
    # Rent Range (City + Life Stage Adjustment)
    rent_cap_max_income = monthly_income * user_rules["rent_pct_max"]
    rent_min = city_config["min_rent"]
    # Final rent cap is the lower of the income-based cap and the city-adjusted cap
    rent_max = int(round(min(rent_cap_max_income * city_config["rent_factor"], rent_cap_max_income)))

    # Expense Caps (Remaining amount for other categories)
    # Remaining for expenses (excluding savings)
    expense_pool = monthly_income - ideal_savings_amount
    
    # Initial allocation based on standard percentages (adjusted for rent)
    rent_used = min(rent_max, expense_pool * 0.30) # Use max rent cap for planning purposes
    
    remaining_budget = expense_pool - rent_used

    # Distribution of remaining budget (using city cost factors as multipliers on a base distribution)
    # Base split of remaining: Food 40%, Transport 15%, Lifestyle 20%, Other 25% (Adjusted for Indian context)
    food_cap_base = remaining_budget * 0.40
    transport_cap_base = remaining_budget * 0.15
    # Lifestyle is kept simple, capped by the life_stage rule, ensuring it doesn't break the budget
    lifestyle_cap = int(round(monthly_income * user_rules["lifestyle_pct"]))
    
    food_cap = int(round(food_cap_base * city_config["food_factor"]))
    transport_cap = int(round(transport_cap_base * city_config["transport_factor"]))
    
    # The 'Other' category absorbs any rounding/re-adjustment necessary to meet the total expense pool
    calculated_expenses = rent_used + food_cap + transport_cap + lifestyle_cap
    other_cap = max(0, expense_pool - calculated_expenses)

    # Final set of caps (rounding to nearest 100 for readability)
    final_caps = {
        "rent": int(round(rent_used / 100) * 100),
        "food": int(round(food_cap / 100) * 100),
        "transport": int(round(transport_cap / 100) * 100),
        "lifestyle": int(round(lifestyle_cap / 100) * 100),
        "other": int(round(other_cap / 100) * 100),
    }

    # Final SIP adjustment: Must be positive, taken out of the Ideal Savings
    suggested_sip = min(suggested_sip, ideal_savings_amount)
    suggested_sip = max(1000, int(round(suggested_sip / 100) * 100)) # Minimum SIP is 1000 INR
    ideal_savings_amount = int(round(ideal_savings_amount / 100) * 100)

    # --- 3. FULL FINANCIAL BLUEPRINT (BILINGUAL) ---
    bengaluru_info = ""
    if city_config["cost_level"] in ["VERY HIGH", "HIGH"] and monthly_income < 60000:
        bengaluru_info = "\n\n**Note:** The cost of living is **Very High** here. You must be extremely disciplined."
    
    def money(x):
        return f"₹{int(round(x)):,}"
        
    explanation = f"""
## 🎯 Your Personalized Financial Blueprint by PRAKRITI AI 👩‍💻

Dear client, based on your **{life_stage}** stage and **{city.title()}** being a **{city_config['cost_level']}** city, here is your customized financial plan.

### Financial Summary (₹{monthly_income:,} Monthly Income)

| Metric | Recommendation (English) | सलाह (Hindi) |
| :--- | :--- | :--- |
| **Ideal Savings Rate** | **{ideal_savings_pct:.0f}%** of your income. | अपनी आय का **{ideal_savings_pct:.0f}%** बचाएँ। |
| **SIP Target** | **{money(suggested_sip)}** per month. | हर महीने **{money(suggested_sip)}** का SIP शुरू करें। |
| **Emergency Fund** | **{emergency_fund_target} months** of expenses. | **{emergency_fund_target} महीने** के ख़र्चों के बराबर। |
| **Rent Range** | **{money(rent_min)} – {money(rent_max)}** (Max {user_rules['rent_pct_max']*100:.0f}%) | किराए की सीमा **{money(rent_min)} – {money(rent_max)}** (अधिकतम {user_rules['rent_pct_max']*100:.0f}%)। |

{bengaluru_info}

### 💸 Monthly Expense Caps (Budget Allocation)

Your budget is broken down using an adapted **50:30:20 Rule** (Needs:Wants:Savings), adjusted for your stage.

| Category | Recommended Cap (INR) | Percent of Income |
| :--- | :--- | :--- |
| **Rent/EMI** | **{money(final_caps["rent"])}** | {final_caps["rent"] / monthly_income * 100:.1f}% |
| **Food (Groceries/Dining)** | **{money(final_caps["food"])}** | {final_caps["food"] / monthly_income * 100:.1f}% |
| **Transport/Fuel** | **{money(final_caps["transport"])}** | {final_caps["transport"] / monthly_income * 100:.1f}% |
| **Lifestyle/Wants** | **{money(final_caps["lifestyle"])}** | {final_caps["lifestyle"] / monthly_income * 100:.1f}% |
| **Other (Utilities, Misc)** | **{money(final_caps["other"])}** | {final_caps["other"] / monthly_income * 100:.1f}% |
| **Savings/SIP** | **{money(ideal_savings_amount)}** | {ideal_savings_pct:.0f}% |
| **Total** | **{money(sum(final_caps.values()) + ideal_savings_amount)}** | 100.0% |

### ✅ DOs and ❌ DON'Ts for an {life_stage} Professional

| DOs (करें) | DON'Ts (न करें) |
| :--- | :--- |
| ✅ **Start SIP NOW** with {money(suggested_sip)}. Consistency is key! | ❌ **Avoid Credit Card Debt.** Only use credit cards if you can pay the full bill every month. |
| ✅ **Automate Savings.** The {ideal_savings_pct:.0f}% savings and SIP should be debited automatically on the 1st of the month. | ❌ **Don't Forget Health Insurance.** Medical emergencies can ruin your finances. Get a basic health cover. |
| ✅ **Negotiate Rent.** In a city like {city.title()}, finding roommates and splitting rent is vital to stay within the {money(final_caps["rent"])} cap. | ❌ **Don't Overspend on Lifestyle.** Your lifestyle cap is {money(final_caps["lifestyle"])}. Track dining out and subscriptions strictly. |

### 💡 5 Actionable Tips (5 सरल सुझाव)

1.  **Set Up SIP:** Immediately start a monthly SIP of **{money(suggested_sip)}** in a diversified equity or index fund. (तुरंत एक SIP शुरू करें)
2.  **Rent Share:** For {city.title()}, consider sharing your 1 BHK or moving to a 2 BHK with roommates to reduce your rent burden. (किराया शेयर करके अपना खर्च कम करें)
3.  **Track Everything:** Use a tracking app (like this dashboard!) for 90 days to find where you can save an extra 5%. (हर खर्च को 90 दिनों तक ट्रैक करें)
4.  **Term Insurance:** Buy a simple term life insurance plan *now* while you are young and premiums are low. (कम प्रीमियम वाला टर्म इंश्योरेंस लें)
5.  **Build EF:** Focus on rapidly building the {emergency_fund_target}-month Emergency Fund; keep it in a Liquid Fund/FD. (आपातकालीन फंड जल्दी बनाएँ)
"""

    # --- 4. SHORT TTS-FRIENDLY SUMMARY (FOR READ ALOUD) ---
    tts_summary = f"TTS_SUMMARY:\nAap ek {life_stage} professional hain aur {city.title()} mein rehte hain. Aapki monthly income {money(monthly_income)} hai. Humari salah hai ki aap {ideal_savings_pct:.0f} percent yani {money(ideal_savings_amount)} har mahine save karein, jismein se {money(suggested_sip)} ka SIP zaroor shuru karein. Aapka rent {money(final_caps['rent'])} se zyada nahi hona chahiye. Emergency fund ke liye {emergency_fund_target} mahine ke kharche alag se rakhein. Kiraya share karein aur credit card ke debt se bachein. Apne savings aur SIP ko automatic kar dein. All the best!"
    
    # --- 5 & 6. CHART BLUEPRINTS + JSON OUTPUT ---
    final_json = {
        "rent_recommendation": [rent_min, rent_max],
        "ideal_savings_pct": ideal_savings_pct,
        "emergency_fund_months": emergency_fund_target,
        "suggested_sip": suggested_sip,
        "expense_caps": {
            "rent": final_caps["rent"],
            "food": final_caps["food"],
            "transport": final_caps["transport"],
            "lifestyle": final_caps["lifestyle"],
            "other": final_caps["other"]
        },
        "chart_blueprints": [
            {
                "id": "expense_caps_bar",
                "title": "Recommended Monthly Expense Caps (INR)",
                "chart_type": "bar",
                "description": "Visual representation of the suggested budget limits for all expense categories.",
                "data_source": "expense_caps",
                "recommended_x": "category",
                "recommended_y": "amount"
            },
            {
                "id": "savings_allocation_donut",
                "title": f"Monthly Income Split ({ideal_savings_pct:.0f}% Savings)",
                "chart_type": "donut",
                "description": "Shows how your total income is distributed across expenses and the target savings percentage.",
                "data_source": "income_vs_caps",
                "recommended_x": "type",
                "recommended_y": "amount"
            },
            {
                "id": "sip_vs_savings_pie",
                "title": "Savings and Investment Split",
                "chart_type": "pie",
                "description": "Breakdown of the total target savings into SIP, Emergency Fund contribution, and remaining savings.",
                "data_source": "savings_split",
                "recommended_x": "label",
                "recommended_y": "value"
            },
            {
                "id": "emergency_fund_gauge",
                "title": "Emergency Fund Target (Months)",
                "chart_type": "gauge",
                "description": f"Target gauge for your {emergency_fund_target}-month emergency fund, a key priority for your stage.",
                "data_source": "emergency_fund",
                "recommended_x": "label",
                "recommended_y": "value"
            },
            {
                "id": "projected_sip_line",
                "title": "1-Year Projected SIP Growth (Simulated)",
                "chart_type": "line",
                "description": "A simple linear projection of your wealth if you consistently maintain the suggested SIP amount for 12 months.",
                "data_source": "projected_savings",
                "recommended_x": "month",
                "recommended_y": "amount"
            },
            {
                "id": "rent_recommendation_bar",
                "title": f"Recommended Rent Band in {city.title()}",
                "chart_type": "bar",
                "description": "The ideal minimum and maximum rent you should aim for in your city based on income.",
                "data_source": "rent_band",
                "recommended_x": "type",
                "recommended_y": "amount"
            }
        ]
    }
    
    return explanation, tts_summary, final_json


def render_ca_plan_tab(df: pd.DataFrame):
    """Renders the Personal CA Plan Generator tab."""
    st.header("🧑‍💼 Personal CA Financial Blueprint Generator")
    st.caption("Get a detailed, stage-specific financial plan including SIP and expense caps.")

    life_stages = ["Student", "Fresher", "Early Career", "Family", "Retirement"]
    # Get all cities from the main preset dictionary for the dropdown
    cities_list = sorted(list(set([k.title() for k in ALL_CITIES.keys()]))) 
    
    col_input_1, col_input_2, col_input_3 = st.columns(3)
    
    with col_input_1:
        life_stage = st.selectbox("Life Stage 👤", options=life_stages, index=life_stages.index("Early Career"), key="ca_life_stage")
    with col_input_2:
        city = st.selectbox("City of Residence 🏙️", options=cities_list, index=cities_list.index("Bengaluru") if "Bengaluru" in cities_list else 0, key="ca_city")
    with col_input_3:
        monthly_income = st.number_input("Monthly Income (₹) 💵", min_value=10000, step=5000, value=75000, key="ca_income")
    
    if st.button("Generate My Financial Blueprint", use_container_width=True, key="generate_blueprint_btn"):
        if monthly_income < 10000:
            st.error("Please enter a valid monthly income (minimum ₹10,000).")
            return

        with st.spinner("🧠 Analyzing profile and generating plan..."):
            # Calculate a proxy for average monthly expenses if user has data
            recent_expenses = df[df['type'] == 'expense'].tail(90).groupby(pd.to_datetime(df['date']).dt.to_period('M'))['amount'].sum().mean()
            
            explanation, tts_summary, plan_json = generate_ca_financial_plan(
                life_stage=life_stage,
                city=city,
                monthly_income=int(monthly_income),
                monthly_expenses=recent_expenses if not pd.isna(recent_expenses) else None
            )

            st.session_state["ca_plan_explanation"] = explanation
            st.session_state["ca_plan_tts_summary"] = tts_summary
            st.session_state["ca_plan_json"] = plan_json
            
            st.rerun()

    # --- Display Results ---
    if "ca_plan_json" in st.session_state and st.session_state["ca_plan_json"]:
        plan_json = st.session_state["ca_plan_json"]
        explanation = st.session_state["ca_plan_explanation"]
        tts_summary = st.session_state["ca_plan_tts_summary"]
        
        # Determine income used for visualization (match what was sent to generator)
        monthly_income_used = int(st.session_state.get("ca_income", 75000))

        st.markdown(explanation, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # TTS Summary Section
        st.subheader("🗣️ Short Summary for Read Aloud (TTS)")
        tts_text = tts_summary.replace("TTS_SUMMARY:", "").strip()
        st.info(tts_text)
        _tts_button("tts_ca_plan_final", tts_text, "hi-IN", rate=1.0, pitch=1.05)
        
        # Chart Blueprints and Visualization
        st.markdown("---")
        st.subheader("📊 Visual Insights from Personal CA (6 Charts)")
        
        chart_blueprints = plan_json.get("chart_blueprints", [])
        expense_caps = plan_json.get("expense_caps", {})
        rent_min, rent_max = plan_json.get("rent_recommendation", [0, 0])
        ideal_savings_pct = plan_json.get("ideal_savings_pct", 0)
        suggested_sip = plan_json.get("suggested_sip", 0)
        emergency_months = plan_json.get("emergency_fund_months", 0)

        # Derived data sources for plotting
        ideal_savings_amount = monthly_income_used * (ideal_savings_pct / 100.0)
        
        # Simple cumulative savings projection for 12 months (SIP only)
        projected_savings_data = [{"month": 0, "amount": 0, "Date": date.today()}]
        for i in range(1, 13):
             projected_savings_data.append({
                 "month": i, 
                 "amount": suggested_sip * i, 
                 "Date": date.today() + timedelta(days=30*i)
             })

        data_sources = {
            "expense_caps": [
                {"category": k.capitalize(), "amount": v}
                for k, v in expense_caps.items()
            ],
            "income_vs_caps": [
                {"type": "Savings/Investment", "amount": ideal_savings_amount},
                {"type": "Total Expenses (Caps)", "amount": sum(expense_caps.values())},
            ],
            "projected_savings": projected_savings_data,
            "rent_band": [
                {"type": "Minimum Rent", "amount": rent_min},
                {"type": "Maximum Rent", "amount": rent_max},
            ],
            "emergency_fund": [
                {"label": "Emergency Fund Target", "value": emergency_months}
            ],
            "savings_split": [
                {"label": "SIP Target", "value": suggested_sip},
                {"label": "SIP Contribution", "value": suggested_sip},
                {"label": "Other Savings (EF, Buffer)", "value": max(0, ideal_savings_amount - suggested_sip)},
            ]
        }
        
        cols_viz_1, cols_viz_2 = st.columns(2)
        cols_viz_3, cols_viz_4 = st.columns(2)
        cols_viz_5, cols_viz_6 = st.columns(2)

        chart_cols = [cols_viz_1, cols_viz_2, cols_viz_3, cols_viz_4, cols_viz_5, cols_viz_6]
        
        # Dynamic Chart Rendering Loop
        for idx, bp in enumerate(chart_blueprints[:6]):
            with chart_cols[idx]:
                st.markdown(f"**{idx+1}. {bp['title']}**")
                chart_type = bp["chart_type"]
                source_key = bp["data_source"]
                x_key = bp.get("recommended_x")
                y_key = bp.get("recommended_y")

                data = data_sources.get(source_key)
                if not data:
                    st.warning(f"No data for {source_key}.")
                    continue

                df_chart = pd.DataFrame(data)

                # --- Chart Visualization Logic ---
                if chart_type in ["bar", "pie", "donut", "area", "line"]:
                    
                    if chart_type == "bar":
                        fig = px.bar(df_chart, x=x_key, y=y_key, color_discrete_sequence=['#6a5acd'])
                    elif chart_type == "pie":
                        fig = px.pie(df_chart, names=x_key, values=y_key, hole=0.3, color_discrete_sequence=px.colors.qualitative.Pastel)
                    elif chart_type == "donut":
                        fig = px.pie(df_chart, names=x_key, values=y_key, hole=0.6, color_discrete_sequence=px.colors.qualitative.Pastel)
                    elif chart_type == "line":
                        fig = px.line(df_chart, x="Date" if x_key == "month" else x_key, y=y_key, markers=True, color_discrete_sequence=['#22c55e'])
                        fig.update_xaxes(title_text='Month')
                        
                    elif chart_type == "area":
                        fig = px.area(df_chart, x=x_key, y=y_key, color_discrete_sequence=['#8a2be2'])
                    
                    fig.update_layout(template="plotly_dark", height=300, showlegend=True, margin=dict(t=30, b=30, l=20, r=20))
                    if chart_type in ["bar"]:
                        fig.update_traces(texttemplate='₹%{y:,.0f}', textposition='outside')
                        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

                    st.plotly_chart(fig, use_container_width=True)

                elif chart_type == "gauge":
                    value = df_chart["value"].iloc[0]
                    # Gauge max is 1.5x the target or max 12
                    max_val = max(12, emergency_months + 3) if bp['id'] == "emergency_fund_gauge" else monthly_income_used
                    
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=value,
                        gauge={"axis": {"range": [0, max_val], "tickwidth": 1, "tickcolor": "darkblue"},
                               "bar": {"color": "#6a5acd"},
                               "bgcolor": "white",
                               "steps": [
                                   {"range": [0, max_val * 0.5], "color": "lightgray"},
                                   {"range": [max_val * 0.5, max_val], "color": "gray"}
                               ],
                               "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": emergency_months}
                               },
                        number={"valueformat": ".1f"}
                    ))
                    fig.update_layout(template="plotly_dark", height=300, margin=dict(t=50, b=50, l=20, r=20))
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown(f"<p style='color:#888;font-size:12px;'>💡 {bp['description']}</p>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("Raw JSON Blueprint")
        st.json(plan_json)

# ============================================================
# (Rest of your original app continues unchanged)
# ============================================================

# REPLACE your _inject_global_particles with this FPS-aware, auto-throttling version
def _inject_global_particles(enabled: bool = True) -> None:
    """Global particles with parallax + hover + scroll-velocity boost + FPS-based autothrottle."""
    if not enabled:
        components.html(
            """
            <script>
              try { const old = document.getElementById('cc-particles'); if (old) old.remove(); } catch(e){}
            </script>
            """,
            height=0,
        )
        return
    components.html(
        """
<style>
/* ... (CSS for particle animation is omitted for brevity but remains in your local file) ... */
</style>
<canvas id="cc-particles"></canvas>
<script>
// ... (JavaScript for particle animation is omitted for brevity but remains in your local file) ...
</script>
        """,
        height=0,
    )

# ============================== Mini In-Memory DB (Multi-User) ==============================

@dataclass
class Order:
    id: int
    amount: float
    currency: str
    status: str
    note: str = ""
    created_at: str = datetime.utcnow().isoformat(timespec="seconds")


@dataclass
class Transaction:
    id: int
    user_id: str
    date: str
    amount: float
    category: str
    description: str
    type: str
    created_at: str = datetime.utcnow().isoformat(timespec="seconds")


class MiniDB:
    """In-memory orders + transactions with optional JSON persistence."""
    DB_PATH = Path("mini_db.json")

    def __init__(self) -> None:
        self._orders: Dict[int, Order] = {}
        self._tx: Dict[int, Transaction] = {}
        self._order_seq: int = 0
        self._tx_seq: int = 0

    def create_order(self, amount: float, currency: str = "INR", note: str = "") -> Order:
        self._order_seq += 1
        o = Order(
            id=self._order_seq,
            amount=float(amount),
            currency=currency,
            status="pending",
            note=note,
        )
        self._orders[o.id] = o
        return o

    def list_orders(self, status: Optional[str] = None) -> List[Order]:
        vals = list(self._orders.values())
        return [o for o in vals if (status is None or o.status == status)]

    def _filter_txns(self, user_id: str) -> List[Transaction]:
        return [t for t in self._tx.values() if t.user_id == user_id]

    def add_txn(
        self,
        *,
        user_id: str,
        dt: date,
        amount: float,
        category: str,
        description: str,
        typ: str,
    ) -> Transaction:
        if typ not in ("income", "expense"):
            raise ValueError("typ must be 'income' or 'expense'")
        self._tx_seq += 1
        t = Transaction(
            id=self._tx_seq,
            user_id=user_id,
            date=_safe_to_date(dt).isoformat(),
            amount=float(amount),
            category=(category or "uncategorized"),
            description=(description or ""),
            type=typ,
        )
        self._tx[t.id] = t
        return t

    def list_txns(
        self,
        user_id: str,
        *,
        start: Optional[date] = None,
        end: Optional[date] = None,
        categories: Optional[Sequence[str]] = None,
        types: Optional[Sequence[str]] = None,
    ) -> List[Transaction]:
        rows = self._filter_txns(user_id)
        if start:
            rows = [r for r in rows if r.date >= _safe_to_date(start).isoformat()]
        if end:
            rows = [r for r in rows if r.date <= _safe_to_date(end).isoformat()]
        if categories:
            cs = set(categories)
            rows = [r for r in rows if r.category in cs]
        if types:
            ts = set(types)
            rows = [r for r in rows if r.type in ts]
        return sorted(rows, key=lambda r: (r.date, r.id))

    def totals(self, user_id: str) -> dict:
        user_txns = self._filter_txns(user_id)
        inc = sum(t.amount for t in user_txns if t.type == "income")
        exp = sum(t.amount for t in user_txns if t.type == "expense")
        return {"income": inc, "expense": exp, "net": inc - exp}

    def piggy_balance(self, user_id: str, category: str = "collection") -> float:
        user_txns = self._filter_txns(user_id)
        return sum(t.amount for t in user_txns if t.type == "income" and t.category == category)

    def update_txn(self, txn_id: int, **fields) -> bool:
        """Update a single transaction by id."""
        t = self._tx.get(txn_id)
        if not t:
            return False
        safe = {"date", "amount", "category", "description", "type", "user_id"}
        for k, v in fields.items():
            if k in safe:
                if k == "date":
                    v = _safe_to_date(v).isoformat()
                if k == "amount":
                    v = float(v)
                setattr(t, k, v)
        self._tx[txn_id] = t
        return True

    def delete_txn(self, txn_id: int) -> bool:
        if txn_id in self._tx:
            del self._tx[txn_id]
            return True
        return False
    
    # --- NEW FEATURE: Delete All Transactions ---
    def delete_all_txns(self, user_id: str) -> int:
        """Deletes all transactions for a specific user ID."""
        txns_to_delete = [tid for tid, txn in self._tx.items() if txn.user_id == user_id]
        for tid in txns_to_delete:
            del self._tx[tid]
        return len(txns_to_delete)
    # -------------------------------------------

    def rename_or_merge_category(self, user_id: str, old_cat: str, new_cat: str) -> int:
        count = 0
        for t in self._tx.values():
            if t.user_id == user_id and t.category == old_cat:
                t.category = new_cat
                count += 1
        return count

    def find_duplicates(self, user_id: str) -> list[list[int]]:
        from collections import defaultdict
        buckets = defaultdict(list)
        for t in self._tx.values():
            if t.user_id != user_id:
                continue
            key = (
                t.user_id,
                t.date,
                round(float(t.amount), 2),
                t.category.strip().lower(),
                t.description.strip().lower(),
                t.type,
            )
            buckets[key].append(t.id)
        return [ids for ids in buckets.values() if len(ids) > 1]

    def delete_duplicates_keep_smallest_id(self, user_id: str) -> int:
        removed = 0
        for group in self.find_duplicates(user_id):
            group_sorted = sorted(group)
            for tid in group_sorted[1:]:
                if self.delete_txn(tid):
                    removed += 1
        return removed

    def to_json(self) -> dict:
        return {
            "order_seq": self._order_seq,
            "tx_seq": self._tx_seq,
            "orders": [asdict(o) for o in self._orders.values()],
            "transactions": [asdict(t) for t in self._tx.values()],
        }

    @classmethod
    def from_json(cls, data: dict) -> "MiniDB":
        db = cls()
        db._order_seq = int(data.get("order_seq", 0))
        db._tx_seq = int(data.get("tx_seq", 0))
        for o in data.get("orders", []):
            try:
                obj = Order(**o)
                db._orders[obj.id] = obj
            except TypeError:
                pass
        for t in data.get("transactions", []):
            try:
                if "user_id" not in t:
                    t["user_id"] = "prakriti11"
                obj = Transaction(**t)
                db._tx[obj.id] = obj
            except TypeError:
                pass
        return db

    def save(self) -> None:
        self.DB_PATH.write_text(
            json.dumps(self.to_json(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls) -> "MiniDB":
        if not cls.DB_PATH.exists():
            return cls()
        try:
            return cls.from_json(json.loads(cls.DB_PATH.read_text(encoding="utf-8")))
        except Exception:
            return cls()

# ============================== API Keys and Configuration ==============================

# FIX: USING LATEST VALID KEY AND ENSURING DEFINITIVE ASSIGNMENT.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or "AIzaSyDEYIm09tc6EvmKwD3JwYIIQSfpAELjZ-Q"
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN") or "8553931141:AAETBKCN1jCYub3Hf7BZ1ylS3izMB5EDzII"
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID") or "6242960424"
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

ALPHAVANTAGE_API_KEY = "F6DEPCVL8IU9ZKAO"
OPENAQ_API_KEY = "0e673c3f15e1c0733ac022d51e0966fc3e721fc35b52e42082e18815ae49084f"
WAQI_TOKEN = "efc26bc8e169b40bb7d85dba79e0b96aaf84229a"
NEWSAPI_KEY = "pub_a3c1025e77b84894b2cd7c545677906d"

# ---------- Constants ----------
KB_FILE = Path("knowledge_base.txt")
KB_VECT = Path("finance_tfidf.joblib")
STREAK_FILE = Path("streak_store.json")
UPI_QR_IMG = Path("upi_qr.png")
UPI_QR_IMG_JPG = Path("upi_qr.jpg")
PROFILE_IMG = Path("profile_money.jpg")
FORM_URL = "https://docs.google.com/forms/d/e/1FAIpQLSdpc-tUatBPZodydM8viqM8fuZgoXC_IPfJiSvx0KdMYifBEw/viewform?usp=header"
APP_BASE_URL = "http://localhost:8501"
RAIN_DURATION_SEC = 5.0
SOUND_EFFECT_URL = "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"

# --- UPI Details ---
UPI_ID = "jaiswalprakriti26@okaxis"
UPI_PAYMENT_STRING = f"upi://pay?pa={UPI_ID}&pn=PRAKRITI&cu=INR"

# --- Personalized Information ---
TEAM_INFO = {
    "Team Name": "Cashflow Crew",
    "Team Leader": "Prakriti Jaiswal",
    "Leader Expertise": "B.Com student at Allahabad University, expert in commerce.",
    "Frontend": "Ujjwal Singh",
    "Guidance": "Akash Pandey Sir (Technosavvys)",
    "Contact": "9170397988",
    "Email": "jaiswalprakriti26@gmail.com",
    "Donate UPI": UPI_ID,
}

HAS_QR = True

# ============================== Utilities / FX / Sound ==============================

def generate_placeholder_image(path: Path, size: int = 300, color: str = "pink", text: str = "Placeholder") -> None:
    """Generate a placeholder if the asset is missing."""
    if path.exists():
        return
    try:
        img = Image.new("RGB", (size, size), color=color)
        d = ImageDraw.Draw(img)
        d.text((size // 4, size // 2), text, fill=(0, 0, 0))
        img.save(path)
    except Exception:
        pass


def _img64(path: Path | None) -> str:
    try:
        if not path or not path.exists():
            return ""
        with open(path, "rb") as fh:
            return base64.b64encode(fh.read()).decode("utf-8")
    except Exception:
        return ""


def _pick_qr_path() -> Path | None:
    if UPI_QR_IMG.exists():
        return UPI_QR_IMG
    if UPI_QR_IMG_JPG.exists():
        return UPI_QR_IMG_JPG
    return None


def _generate_default_upi_qr(upi_string: str, path: Path):
    if not HAS_QR:
        return False
    try:
        qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=10, border=2)
        qr.add_data(upi_string)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        img.save(path)
        return True
    except Exception:
        return False


def _save_uploaded_qr(file) -> str:
    try:
        img = Image.open(file).convert("RGB")
        img.save(UPI_QR_IMG)
        return "QR updated. If not visible, press 'Rerun' or refresh."
    except Exception as e:
        return f"Failed to save QR: {e}"


def _b64_audio_from_file(path: Path) -> str | None:
    try:
        if path.exists():
            return base64.b64encode(path.read_bytes()).decode("utf-8")
    except Exception:
        pass
    return None


_FALLBACK_WAV_B64 = (
    "UklGRiQAAABXQVZFZm10IBAAAAABAAEAESsAACJWAAACABYAAAACABYAAABkYXRhAAAAAA"
    "AAAAAAgP8AgP8AgP8AgP8AgP8AgP8AgP8AgP8AgP8AgP8AgP8AgP8AgP8AgP8AgP8AgP8A"
)

def play_paid_sound(name: str, amount: float) -> None:
    """Play cash sound and speak Hindi line in browser."""
    audio_src = SOUND_EFFECT_URL
    if st.session_state.get("sound_muted", False):
        return
    spoken = f"₹{int(round(amount))} का भुगतान सफल — {name} प्राप्त हुए हैं।"
    rand_id = random.randint(1000, 9999)
    html = f"""
      <audio id="payment-sound-{rand_id}" src="{audio_src}" preload="auto" autoplay></audio>
      <script>
        document.getElementById('payment-sound-{rand_id}').play().catch(e => console.log('Audio play blocked or failed:', e));
        try {{
          const u = new SpeechSynthesisUtterance("{spoken}");
          u.lang = "hi-IN";
          u.rate = 1.0; u.pitch = 1.0;
          window.speechSynthesis.cancel();
          window.speechSynthesis.speak(u);
        }} catch(e) {{ console.warn(e); }}
      </script>
    """
    components.html(html, height=0, scrolling=False)


def show_coin_rain(seconds: float = 5.0) -> None:
    """Displays the coin rain animation."""
    coin_spans = "".join(
        [
            f"<span style='left:{random.randint(5, 95)}%; animation-delay:{random.uniform(0, RAIN_DURATION_SEC/2):.2f}s;'>🪙</span>"
            for _ in range(20)
        ]
    )
    st.markdown(
        f"""
<style>
/* NEW: Enhanced Coin Animation and Visibility */
@keyframes coin-pulse {{
    0%, 100% {{
        transform: scale(1.0) translateY(0px);
        filter: drop-shadow(0 0 8px gold) drop-shadow(0 0 3px orange);
    }}
    50% {{
        transform: scale(1.1) translateY(-2px);
        filter: drop-shadow(0 0 12px gold) drop-shadow(0 0 6px orange);
    }}
}}
.coin-rain {{
  position: fixed; inset: 0; pointer-events: none; z-index: 9999;
}}
.coin-rain span {{
  position:absolute; top:-50px; font-size:22px; filter:drop-shadow(0 6px 8px rgba(0,0,0,.35));
  animation: rain 2.2s linear infinite, coin-pulse 2s ease-in-out infinite;
}}
@keyframes rain{{0%{{transform:translateY(-60px) rotate(0deg);opacity:0}}
15%{{opacity:1}}100%{{transform:translateY(120vh) rotate(360deg);opacity:0}}}}
</style>
<div class="coin-rain">
    {coin_spans}
</div>
        """,
        unsafe_allow_html=True,
    )


def green_tick(msg: str) -> None:
    """Displays a large, noticeable green tick message."""
    st.markdown(
        f"""<div style="padding: 10px; border-radius: 8px; background-color: rgba(34, 197, 94, 0.2); color: #22c55e; margin-top: 15px;">
    <span style="font-size: 24px;">✅</span><span style="margin-left: 10px; font-weight: bold;">{msg}</span>
    </div>""",
        unsafe_allow_html=True,
    )

# --- NEW: openai_query function (REQUIRED FOR FALLBACK) ---
def openai_query(prompt: str, history: list[tuple[str, str]], context: str) -> str:
    """Handles the intelligent response using the OpenAI API."""
    if not HAS_OPENAI_SDK or not OPENAI_API_KEY:
        return "❌ **OPENAI KEY MISSING:** Please set the `OPENAI_API_KEY` environment variable."
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        system_instruction = (
            "You are a friendly, professional AI financial advisor named PRAKRITI AI. "
            "You are acting as a fallback because the main AI failed. "
            "Be concise (3-5 sentences) and polite. Use emojis."
        )
        messages = [{"role": "system", "content": system_instruction}]
        messages.append({"role": "user", "content": context})
        for speaker, msg in history:
            messages.append({"role": "user", "content": f"{speaker}: {msg}"})
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=200
        )
        return f"🤖 *OpenAI Fallback AI:* {response.choices[0].message.content}"
    except Exception as e:
        return f"❌ **OPENAI API Error:** Failed to generate response. Error: {e}"

# --- ORIGINAL: gemini_query (MODIFIED) ---
def gemini_query(prompt: str, history: list[tuple[str, str]], context: str) -> str:
    """Handles the intelligent response using the Gemini API, with OpenAI fallback."""
    if not GEMINI_API_KEY:
        if HAS_OPENAI_SDK and OPENAI_API_KEY:
            return openai_query(prompt, history, context)
        return "❌ **GEMINI KEY MISSING:** Please set the `GEMINI_API_KEY` environment variable."
    if not HAS_GEMINI_SDK:
        if HAS_OPENAI_SDK and OPENAI_API_KEY:
            return openai_query(prompt, history, context)
        return "⚠️ **GEMINI SDK Missing:** Cannot connect to the intelligent chatbot. Please run `pip install google-genai`."
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        system_instruction = (
            "You are a versatile, professional AI financial advisor named PRAKRITI AI, part of the Cashflow Crew. "
            "Your persona is based on the following: " + context +
            "You must be able to answer finance questions, but also handle casual conversation, greetings, and nonsense questions gracefully. "
            "For finance queries, be concise (3-5 sentences) and proactive in suggesting ideas. "
            "For casual queries, respond like a friendly assistant. "
            "If the user asks a casual question (like 'hi' or 'how are you' or a simple greeting), use a simple, friendly response (e.g., 'I am fine, how are you?')."
            "Always include emojis in your responses to make them more engaging."
        )
        final_prompt = system_instruction + "\n\n" + prompt
        contents = [{"role": "user", "parts": [{"text": final_prompt}]}]
        # FIX: Using the faster, more stable lite model for better performance/reliability
        response = client.models.generate_content(model="gemini-2.5-flash-lite", contents=contents)
        return f"🧠 *Gemini Smart AI:* {response.text}"
    except Exception as e:
        if HAS_OPENAI_SDK and OPENAI_API_KEY:
            st.warning(f"Gemini API failed with error: {e}. Falling back to OpenAI.")
            return openai_query(prompt, history, context)
        return f"❌ **GEMINI API Error:** Failed to generate response. Check your API key and network connection. Error: {e}"

# AlphaVantage API Utility (Simulated)
def fetch_stock_quote(symbol: str) -> dict | str:
    symbol_upper = symbol.upper()
    np.random.seed(len(symbol_upper) + datetime.now().day)
    if symbol_upper == "TCS.BSE":
        base_price = 4000
    elif symbol_upper == "RELIANCE.NSE":
        base_price = 2800
    elif "ITC" in symbol_upper:
        base_price = 420
    else:
        base_price = 450 + len(symbol_upper) * 10
    change_pct = np.random.uniform(-1.5, 1.5)
    volume_base = 500000 + len(symbol_upper) * 100000
    return {
        "symbol": symbol_upper,
        "price": f"{base_price:,.2f}",
        "change": f"{change_pct:+.2f}%",
        "volume": f"{int(volume_base/1000)}K",
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

@st.cache_data
def generate_simulated_daily_data(symbol: str, days: int = 60) -> pd.DataFrame:
    symbol_upper = symbol.upper()
    if "TCS" in symbol_upper:
        base_price = 4000
    elif "RELIANCE" in symbol_upper:
        base_price = 2800
    elif "ITC" in symbol_upper:
        base_price = 420
    else:
        base_price = 450 + len(symbol_upper) * 10
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days, freq="D")
    np.random.seed(len(symbol_upper))
    prices = [base_price]
    for i in range(1, days):
        change = np.random.normal(0, 15) * (1 + np.sin(i / 20))
        new_price = prices[-1] * (1 + change / 1000)
        prices.append(new_price)
    volumes = np.random.randint(100000, 3000000, size=days)
    df = pd.DataFrame(
        {"Date": dates, "Close Price (₹)": [round(p, 2) for p in prices], "Volume": volumes}
    )
    df = df.set_index("Date").sort_index()
    
    # Calculate Simple Moving Averages (SMA)
    df['SMA_Short'] = df['Close Price (₹)'].rolling(window=10).mean()
    df['SMA_Long'] = df['Close Price (₹)'].rolling(window=30).mean()
    
    return df

# --- KB/TFIDF Helpers ---
def ensure_kb_exists(default_kb: list[str] | None = None) -> None:
    default_kb = default_kb or [
        "help - Type questions about expenses, income, trends (e.g., 'total expense', 'top categories')",
        "overview - Show project overview and advantages",
        "trend groceries - Show spending trend for groceries",
        "plot - Explain the current plot and data",
        "streak - Show current and longest saving streak",
        "invest advice - Ask for general saving and investment advice",
    ]
    if not KB_FILE.exists():
        try:
            KB_FILE.write_text("\n".join(default_kb), encoding="utf-8")
        except Exception:
            pass

# --- Data/Plot Helpers ---
def to_excel_bytes(df: pd.DataFrame) -> bytes:
    out = BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="data")
    return out.getvalue()

def generate_sample(months: int = 6) -> pd.DataFrame:
    rng = pd.date_range(end=pd.Timestamp.today(), periods=months * 30)
    cats = ["groceries", "rent", "salary", "investment", "subscriptions", "dining"]
    rows = []
    for d in rng:
        for _ in range(np.random.poisson(1)):
            cat = np.random.choice(cats, p=[0.2, 0.1, 0.15, 0.15, 0.2, 0.2])
            t = "income" if cat in ("salary", "investment") else "expense"
            amt = abs(round(np.random.normal(1200 if t == "income" else 50, 35), 2))
            rows.append(
                {"date": d.date(), "amount": amt, "category": cat, "description": f"{cat}", "type": t}
            )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

def read_file(file):
    if isinstance(file, (str, Path)):
        if str(file).endswith(".csv"):
            return pd.read_csv(file)
    return pd.read_excel(file)

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    df = df.copy()
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    if "category" not in df.columns:
        df["category"] = "uncategorized"
    if "description" not in df.columns:
        df["description"] = ""
    if "type" not in df.columns:
        df["type"] = "expense"
    date_cols = [c for c in df.columns if "date" in c.lower()]
    if date_cols:
        df["date"] = pd.to_datetime(df[date_cols[0]], errors="coerce").dt.date
    else:
        df["date"] = pd.Timestamp.today().date()
    return df

def add_period(df: pd.DataFrame, group_period: str) -> pd.DataFrame:
    t = df.copy()
    t["date"] = pd.to_datetime(t["date"])
    if group_period == "Monthly":
        t["period"] = t["date"].dt.to_period("M").astype(str)
    elif group_period == "Weekly":
        t["period"] = t["date"].dt.strftime("%G-") + t["date"].dt.isocalendar().week.astype(str).str.zfill(2)
    else:
        t["period"] = t["date"].dt.date.astype(str)
    return t

def daily_net_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.shape[0] == 0:
        return pd.DataFrame(columns=["day", "income", "expense", "net_saving"])
    tmp = df.copy()
    tmp["day"] = pd.to_datetime(tmp["date"]).dt.date
    g = tmp.groupby(["day", "type"])["amount"].sum().unstack(fill_value=0)
    if "income" not in g:
        g["income"] = 0.0
    if "expense" not in g:
        g["expense"] = 0.0
    g["net_saving"] = g["income"] - g["expense"]
    if not g.empty:
        min_date = g.index.min()
        max_date = g.index.max()
        full = pd.date_range(min_date, max_date, freq="D").date
        g = g.reindex(full, fill_value=0.0)
        g.index.name = "day"
    return g.reset_index()

def compute_streak(series_bool: pd.Series) -> tuple[int, int]:
    if series_bool.empty:
        return 0, 0
    s = series_bool.copy()
    s = s.reindex(sorted(s.index))
    longest = run = 0
    for v in s.values:
        run = run + 1 if v else 0
        longest = max(longest, run)
    curr = 0
    for v in reversed(s.values):
        if v:
            curr += 1
        else:
            break
    return int(curr), int(longest)

def explain_plot_and_data(user_q: str, view: pd.DataFrame, tmp: pd.DataFrame, plot_type: str, group_period: str) -> str:
    if view is None or view.shape[0] == 0:
        return "There is no data in the current selection. Adjust date range and filters to include transactions before asking about the plot."
    lines = []
    n = int(view.shape[0])
    total_income = float(view[view["type"] == "income"]["amount"].sum())
    total_expense = float(view[view["type"] == "expense"]["amount"].sum())
    net = total_income - total_expense
    lines.append(
        f"Current selection contains *{n} transactions. Total income **{money(total_income)}**, total expense **{money(total_expense)}**, net **{money(net)}**.*"
    )
    try:
        top_exp = (
            view[view["type"] == "expense"]
            .groupby("category")["amount"]
            .sum()
            .sort_values(ascending=False)
            .head(3)
        )
        if not top_exp.empty:
            items = ", ".join([f"{k} ({money(v)})" for k, v in top_exp.items()])
            lines.append(f"Top expense categories: *{items}*.")
    except Exception:
        pass
    if "line" in plot_type.lower() or "trend" in plot_type.lower():
        lines.append(f"This is a *trend (line/area) plot* grouped by {group_period}.")
    elif "bar" in plot_type.lower():
        lines.append(f"This is a *bar plot* over the {group_period.lower()}.")
    elif "scatter" in plot_type.lower():
        lines.append("This *scatter plot* shows individual transactions — useful to spot outliers.")
    elif "distribution" in plot_type.lower() or "hist" in plot_type.lower():
        lines.append("This shows the *distribution of amounts*.")
    try:
        per = tmp.groupby(["period", "type"])["amount"].sum().unstack(fill_value=0)
        per["net"] = per.get("income", 0) - per.get("expense", 0)
        if per.shape[0] >= 2:
            last = float(per["net"].iloc[-1])
            prev = float(per["net"].iloc[-2])
            diff = last - prev
            pct = (diff / prev * 100) if prev != 0 else float("nan")
            trend = "increasing" if diff > 0 else "decreasing" if diff < 0 else "flat"
            lines.append(
                f"Net change from previous {group_period.lower()}: *{money(diff)}* ({pct:.1f}%). Recent trend: *{trend}*."
            )
    except Exception:
        pass
    lines.append("Tip: Use the Group period and date filters to zoom.")
    return "\n".join(lines)

def project_overview_and_advantages() -> str:
    return (
        "Project overview:\n"
        "This app is an interactive *Personal Finance AI Dashboard* that visualizes expenses and income, computes saving streaks, and provides quick actionable insights.\n\n"
        "- *Interactive visualizations* help you spot trends and top spending categories quickly. 📊\n"
        "- *Smart chatbot (powered by Gemini) and KB* allow generative financial advice and semantic lookups without exposing data externally. 🤖\n"
        "- Built-in *UPI/QR* and form workflow for easy logging. 📲\n"
        "- *Lightweight* and runs locally — your data stays with you. 🔒\n"
    )

# --- VFA Plan Generation ---
def generate_financial_plan_file(df: pd.DataFrame) -> bytes:
    """Generates a sample CSV financial plan based on current data."""
    if not df.empty:
        df_copy = df.copy()
        df_copy["date"] = pd.to_datetime(df_copy["date"])
    else:
        df_copy = pd.DataFrame(
            {"date": [date.today()], "amount": [0], "category": ["Initial"], "type": ["income"]}
        )

    plan_data: list[str] = []

    monthly_summary = df_copy.copy()
    monthly_summary["Month"] = monthly_summary["date"].dt.to_period("M").astype(str)

    if not monthly_summary.empty:
        net_summary = (
            monthly_summary.groupby("Month")
            .agg(
                Total_Income=(
                    "amount",
                    lambda x: x[monthly_summary.loc[x.index, "type"] == "income"].sum(),
                ),
                Total_Expense=(
                    "amount",
                    lambda x: x[monthly_summary.loc[x.index, "type"] == "expense"].sum(),
                ),
                Net_Savings=(
                    "amount",
                    lambda x: x[monthly_summary.loc[x.index, "type"] == "income"].sum()
                    - x[monthly_summary.loc[x.index, "type"] == "expense"].sum(),
                ),
            )
            .reset_index()
        )
    else:
        net_summary = pd.DataFrame(
            {"Month": ["N/A"], "Total_Income": [0], "Total_Expense": [0], "Net_Savings": [0]}
        )

    plan_data.append("--- Monthly Performance Summary ---")
    plan_data.append(net_summary.to_csv(index=False))

    avg_expense = (
        df_copy[df_copy["type"] == "expense"]["amount"].mean()
        if not df_copy[df_copy["type"] == "expense"].empty
        else 500.0
    )
    saving_recommendation = max(50, round(avg_expense * 0.1, 0))

    plan_data.append("\n--- Actionable Plan ---")
    plan_data.append("Action,Target,Category,Recommendation")
    plan_data.append(
        f"Reduce Expense,Monthly,Dining,Reduce dining out by {money(saving_recommendation)} (10% of avg expense)."
    )
    plan_data.append(
        f"Increase Saving,Weekly,Investment,Invest {money(100)} weekly into low-risk funds."
    )

    plan_content = "\n".join(plan_data)
    return plan_content.encode("utf-8")

def save_transactions(user_id: str, df: pd.DataFrame):
    """Adds rows from a normalized DataFrame to the MiniDB."""
    global DB
    for _, row in df.iterrows():
        DB.add_txn(
            user_id=user_id,
            dt=row["date"],
            amount=row["amount"],
            category=row["category"],
            description=row["description"],
            typ=row["type"],
        )

# --- NEW: AI Financial Plan Logic ---
def _get_average_monthly_income(df: pd.DataFrame) -> float:
    """Calculates the average monthly income from the DataFrame."""
    if df.empty:
        return 0.0
    income_df = df[df['type'] == 'income'].copy()
    if income_df.empty:
        return 0.0
    income_df['date'] = pd.to_datetime(income_df['date'])
    income_df['month'] = income_df['date'].dt.to_period('M')
    monthly_income = income_df.groupby('month')['amount'].sum()
    return monthly_income.mean() if not monthly_income.empty else 0.0

def _ai_financial_plan_view(df: pd.DataFrame) -> None:
    """Renders the AI Financial Plan Tab content (The older 50/30/20 view)."""
    st.markdown("""
    <style>
    .fade-line { opacity: 0; background: rgba(255,255,255,0.07); border-left: 4px solid #00f5d4; margin: 6px 0;
                 padding: 10px 12px; border-radius: 10px; color: #ffffff; font-size: 16px; font-weight: 500;
                 animation: fadeIn 1.3s ease-in-out forwards; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); box-shadow: 0 0 5px #00f5d4; }
                               to { opacity: 1; transform: translateY(0); box-shadow: 0 0 20px #00f5d4; } }
    .plan-title { color: #8e2de2; text-align: center; margin-bottom: 20px; }
    .speak-button { background:#8e2de2;color:white;border:none;padding:10px 16px;border-radius:8px;cursor:pointer;font-weight:600; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h2 class='plan-title'>🎯 Personalized Gemini 2.5 Savings Strategy (Legacy View)</h2>", unsafe_allow_html=True)
    st.info("This is the original AI Savings Strategy tab. For the detailed CA plan, use the 'Personal CA Blueprint' tab.")

    avg_income = _get_average_monthly_income(df)
    default_salary = 60000.0 if avg_income == 0.0 else round(avg_income)

    # --- FIX 1: Initialize session state key only if it doesn't exist ---
    if "ai_plan_salary" not in st.session_state:
        st.session_state["ai_plan_salary"] = int(default_salary)
    
    # --- FIX 2: Use key to manage state and remove the conflicting 'value' parameter ---
    st.number_input(
        "💰 Enter/Confirm your Monthly Income (₹):",
        min_value=5000,
        # value=int(default_salary),  <--- REMOVED TO PREVENT StreamlitAPIException
        step=1000,
        key="ai_plan_salary"
    )
    # Read the current salary from session state, regardless of source (default or user input)
    salary = st.session_state["ai_plan_salary"] 

    goal = st.text_input(
        "🎯 Your Current Financial Goal (optional):",
        placeholder="e.g., Save for laptop, trip, or emergency fund",
        key="ai_plan_goal"
    )

    if st.button("🚀 Generate My AI Savings Strategy", use_container_width=True):
        if salary < 5000:
            st.error("Monthly income must be at least ₹5000 to generate a plan.")
            return

        with st.spinner("🤖 Gemini 2.5 is analyzing your profile and creating a strategy..."):
            prompt = f"""
            You are a professional financial advisor named PRAKRITI AI.
            The user earns ₹{salary:,.0f} per month and has a goal: '{goal if goal else 'None'}'.
            Provide a real-life savings strategy.
            Suggest:
            1. Monthly breakdown and ideal percentages for four categories: **Essentials (50%)**, **Savings (25%)**, **Investments (20%)**, and **Lifestyle/Flex (5%)**.
            2. 3-4 quick, actionable financial tips related to their goal (if specified) or their income level.
            3. A summary of the breakdown in a bulleted list format.
            Be concise (max 300 words), practical, realistic, and easy to follow. Include emojis.
            """
            context_str = (
                f"You are a financial coach. The user is {CURRENT_USER_ID} and their average monthly income is {money(avg_income)}."
            )
            response = gemini_query(prompt, [], context_str)
            advice = response.replace("🧠 *Gemini Smart AI:*", "").replace("🤖 *OpenAI Fallback AI:*", "").strip()

            st.markdown("### 🌟 Your Personalized Financial Plan")
            st.markdown(
                f"""
                <div style='background: rgba(142, 45, 226, 0.1); border-left: 5px solid #8e2de2; padding: 15px; border-radius: 10px; margin-top: 15px; color: #1e1e1e;'>
                {advice}
                </div>
                """,
                unsafe_allow_html=True
            )
            st.subheader("📊 Proposed 50/25/20/5 Rule Distribution")
            try:
                labels = ['🏠 Essentials (50%)', '💰 Savings (25%)', '📈 Investments (20%)', '🎉 Lifestyle (5%)']
                values = [0.5 * salary, 0.25 * salary, 0.2 * salary, 0.05 * salary]
                colors = ['#FFB6C1', '#A9FFCB', '#90E0EF', '#FFD6A5']
                fig_pie = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=values,
                    hole=.5,
                    marker_colors=colors,
                    textinfo='label+percent'
                )])
                fig_pie.update_layout(
                    title_text=f"Monthly Distribution of ₹{salary:,.0f}",
                    template="plotly_dark",
                    height=450
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            except Exception as e:
                st.error(f"Failed to generate visualization: {e}")

            st.markdown("---")
            safe_advice_js = advice.replace('"', '\\"').replace("\n", " ")
            st.markdown(
                f"""
                <button onclick="speak_advice()" class='speak-button' id='speak-advice-btn'>🔊 Speak Advice</button>
                <script>
                function speak_advice() {{
                    const text = "{safe_advice_js}";
                    const utterance = new SpeechSynthesisUtterance(text);
                    utterance.lang = "en-IN";
                    utterance.rate = 1.05;
                    utterance.pitch = 1.05;
                    utterance.volume = 1.0;
                    window.speechSynthesis.cancel();
                    window.speechSynthesis.speak(utterance);
                }}
                </script>
                """,
                unsafe_allow_html=True
            )
            st.caption("Click to have the advice read out loud.")

    st.markdown("<hr><p style='text-align:center;color:gray;'>✨ AI Financial Plan powered by Gemini 2.5 Flash ✨</p>", unsafe_allow_html=True)


# ---------- Initial Setup ----------
if not UPI_QR_IMG.exists():
    if not _generate_default_upi_qr(UPI_PAYMENT_STRING, UPI_QR_IMG):
        generate_placeholder_image(UPI_QR_IMG, text="UPI QR (Error)")

generate_placeholder_image(PROFILE_IMG, size=70, color="#25D366", text="Money Icon")
ensure_kb_exists()

PROFILE64 = _img64(PROFILE_IMG)

# ============================== App Initialization ==============================


# === NEW: Keyboard shortcuts (E: Tools tab, B: backup, F: focus chat) ===
components.html("""
<script>
document.addEventListener('keydown', (e)=>{
    const t = e.target.tagName.toLowerCase();
    if (t === 'input' || t === 'textarea') return;
    if (e.key.toLowerCase() === 'e') {
        [...document.querySelectorAll('button, div[role=tab]')].forEach(el=>{
            if (el.innerText && el.innerText.includes('Tools (Edit • Backup • Dedupe)')) el.click();
        });
    }
    if (e.key.toLowerCase() === 'b') {
        [...document.querySelectorAll('button')].forEach(el=>{
            if (el.innerText && el.innerText.includes('Export MiniDB')) el.click();
        });
    }
    if (e.key.toLowerCase() === 'f') {
        const chat = [...document.querySelectorAll('input')].find(x => x.placeholder && x.placeholder.toLowerCase().includes('top categories'));
        if (chat) chat.focus();
    }
});
</script>
""", height=0)

if "DB" not in st.session_state:
    st.session_state["DB"] = MiniDB.load()
DB: MiniDB = st.session_state["DB"]

if "paid_orders_applied" not in st.session_state:
    st.session_state["paid_orders_applied"] = set()
if "thinking" not in st.session_state:
    st.session_state["thinking"] = False
if "longest_streak_ever" not in st.session_state:
    st.session_state["longest_streak_ever"] = 0
if "sound_muted" not in st.session_state:
    st.session_state["sound_muted"] = False
if "user_budgets" not in st.session_state:
    st.session_state["user_budgets"] = {}
if "weather_city" not in st.session_state:
    st.session_state["weather_city"] = "Prayagraj"
if "weather_data" not in st.session_state:
    st.session_state["weather_data"] = get_weather(st.session_state["weather_city"])
if "global_budgets" not in st.session_state:
    st.session_state["global_budgets"] = {}
# --- Goal Tracking State ---
if "goal_target" not in st.session_state:
    st.session_state["goal_target"] = 100000
if "goal_date" not in st.session_state:
    st.session_state["goal_date"] = date.today() + timedelta(days=365)
if "goal_current" not in st.session_state:
    st.session_state["goal_current"] = 0
if "ca_plan_json" not in st.session_state: # NEW STATE FOR CA PLAN
    st.session_state["ca_plan_json"] = None
if "ca_plan_explanation" not in st.session_state:
    st.session_state["ca_plan_explanation"] = None
if "ca_plan_tts_summary" not in st.session_state:
    st.session_state["ca_plan_tts_summary"] = None


# ============================== Login & User Management (FACE DETECTOR) ==============================

VALID_USERS = {
    "prakriti11": "face_biometric_code_A",
    "ujjwal11": "face_biometric_code_B",
    "aditya11": "face_biometric_code_C",
    "guest1": "face_biometric_code_D",
    "guest2": "face_biometric_code_E",
    "guest3": "face_biometric_code_F",
    "guest4": "face_biometric_code_G",
    "guest5": "face_biometric_code_H",
    "guest6": "face_biometric_code_I",
}

def _get_all_users(db: MiniDB) -> set:
    db_users = {t.user_id for t in db._tx.values()}
    return set(VALID_USERS.keys()) | db_users

if "ml_login_step" not in st.session_state:
    st.session_state["ml_login_step"] = 1
if "ml_face_code_live" not in st.session_state:
    st.session_state["ml_face_code_live"] = ""

def _login_view() -> None:
    # UPDATED CSS FOR LOGIN VIEW
    st.markdown(
        """
    <style>
    /* Light Purple/Blue Background */
    [data-testid="stAppViewContainer"] > .main {background-color: #f0f2f6; background: linear-gradient(145deg, #e4e7ff, #f0f2f6); color: #1e1e1e;}
    .login-card-container { display: grid; grid-template-columns: 1fr 1fr; gap: 40px; }
    .login-card-inner {
        padding: 30px; border-radius: 16px;
        background: #ffffff; /* White card */
        border: 1px solid #d4c1f5;
        box-shadow: 0 4px 15px rgba(106, 90, 205, 0.2), 0 0 10px rgba(138, 43, 226, 0.1);
        transition: all 0.3s;
        color: #1e1e1e;
    }
    .login-card-inner:hover { box-shadow: 0 6px 20px rgba(106, 90, 205, 0.3); }
    .stSelectbox div { background-color: #f0f2f6 !important; color: #1e1e1e !important; }
    .stSelectbox button { color: #6a5acd !important; }
    .navbar {
        position: sticky; top: 0; z-index: 1000; padding: 12px 18px; margin: 0 0 18px 0;
        background: linear-gradient(90deg, #6a5acd 0%, #8a2be2 100%); /* Purple gradient */
        box-shadow: 0 8px 20px rgba(0,0,0,0.1); 
        border: 1px solid rgba(255,255,255,0.35);
        display: flex; justify-content: space-between; align-items: center;
    }
    .nav-title { font-weight: 800; font-size: 24px; color:#ffffff; letter-spacing: .5px; }
    .nav-sub { color:#ddddff; font-size:13px; margin-top:-2px; }
    .stSuccess { background-color: #e6f7e9 !important; border-left: 5px solid #22c55e !important; color: #1e1e1e !important; }
    .stError { background-color: #f7e6e6 !important; border-left: 5px solid #ef4444 !important; color: #1e1e1e !important; }
    </style>
    <div class="navbar">
      <div style="display:flex;justify-content:space-between;align-items:center;width:100%">
        <div>
          <div class="nav-title">🔐 Finance Analyzer — BIOMETRIC FACE LOGIN</div>
          <div class="nav-sub">Face Detection for Simplified Biometric Authentication</div>
        </div>
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    all_users = sorted(list(_get_all_users(DB)))

    st.subheader("Live Face Recognition")
    st.warning("Grant camera access. Look directly at the camera to allow face detection and capture.")

    st.subheader("📸 Face Login (Capture Photo)")

img_file = st.camera_input("Take a photo to login")

if img_file is not None:
    import numpy as np
    import cv2

    bytes_data = img_file.getvalue()
    img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    st.image(img, caption="Captured Photo", use_container_width=True)

    # ✅ yaha tumhara face detect / match logic chalega
    # Example:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    if len(faces) > 0:
        st.success("Face detected ✅ Now verify user...")
        # TODO: embedding compare + login set session_state
    else:
        st.error("No face detected ❌ Please try again clearly.")



    if ctx.video_processor:
        face_detected_count = ctx.video_processor.face_detected_count
        current_face_vector = ctx.video_processor.current_face_vector
    else:
        face_detected_count = 0
        current_face_vector = ""

    st.markdown('<div class="login-card-container">', unsafe_allow_html=True)

    # 1. EXISTING USER LOGIN
    with st.container():
        st.markdown('<div class="login-card-inner">', unsafe_allow_html=True)
        st.subheader("Existing User Login (Face Scan)")

        u_select = st.selectbox("Select Username", options=all_users, key="user_select")
        target_code = VALID_USERS.get(u_select, "")

        if face_detected_count > 0:
            st.success("Face detected! Click 'Scan and Login'.")
        else:
            st.error("No face detected. Please look at the camera.")

        if st.button("Scan and Login", use_container_width=True):
            if face_detected_count == 0:
                st.error("❌ **Detection Error:** No face detected to scan.")
            elif face_detected_count > 1:
                st.error("❌ **Detection Error:** Too many faces detected. Please ensure only one person is visible.")
            elif not target_code:
                st.error("❌ **Registration Error:** User not fully registered.")
            else:
                st.session_state["auth_ok"] = True
                st.session_state["auth_user"] = u_select
                st.session_state["chat_history"] = []
                st.success(f"🎉 **Face Biometric Login Success!** Welcome, **{u_select}**.")
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    # 2. NEW USER REGISTRATION
    with st.container():
        st.markdown('<div class="login-card-inner">', unsafe_allow_html=True)
        st.subheader("New User Registration (Enroll Face)")

        u_new = st.text_input("New Username", key="user_new")

        if u_new in VALID_USERS:
            st.info(f"User **{u_new}** is already enrolled.")
        elif face_detected_count > 0:
            st.info("Face detected! Click 'Enroll Face' to save your biometric data.")
        else:
            st.error("Enrollment requires a face detected in the camera.")

        if st.button("Enroll Face & Create User", key="enroll_button", use_container_width=True):
            new_user_id = u_new.strip()
            if not new_user_id:
                st.error("New Username cannot be empty.")
            elif new_user_id in _get_all_users(DB):
                st.error(f"User **{new_user_id}** already exists.")
            elif face_detected_count == 0:
                st.error("❌ **Enrollment Error:** No face detected to enroll.")
            else:
                VALID_USERS[new_user_id] = f"face_biometric_code_{new_user_id}"
                st.session_state["auth_ok"] = True
                st.session_state["auth_user"] = new_user_id
                st.session_state["chat_history"] = []
                st.success(f"🎉 **New Face Biometric User Registered!** Welcome, **{new_user_id}**.")
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # End login-card-container
    st.markdown("---")
    st.caption(f"Current Biometric Code (Face Position): {current_face_vector or 'N/A'}")


if "auth_ok" not in st.session_state:
    st.session_state["auth_ok"] = False
    st.session_state["auth_user"] = None

if not st.session_state["auth_ok"]:
    _login_view()
    st.stop()

CURRENT_USER_ID = st.session_state["auth_user"]

# ---------- Post-Login Setup ----------
if "coin_rain_show" not in st.session_state:
    st.session_state["coin_rain_show"] = True
st.markdown(
    """
<style>
/* Reset basic page styles for the wide view */
html, body, [data-testid="stAppViewContainer"] {
  background: #f0f2f6;
  background: linear-gradient(145deg, #e4e7ff, #f0f2f6);
  color: #1e1e1e;
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
}

/* 🚀 AGGRESSIVE FULL-WIDTH AND CENTER ALIGNMENT FIXES */
[data-testid="stAppViewContainer"] > .main {
    padding: 0px !important;
    margin: 0 auto !important;
}
.main > div {
    max-width: 100% !important; 
    margin: 0 auto !important;
    padding: 1rem 1rem !important; 
}
[data-testid="stBlockContainer"], .block-container {
    max-width: 100% !important;
    padding: 0 !important;
}
[data-testid="stVerticalBlock"], [data-testid="stHorizontalBlock"] {
    max-width: 100% !important;
    padding: 0 !important;
}
[data-testid="stBlock"] {
    max-width: 100% !important;
    padding: 0 !important;
}
div.block-container {
    max-width: 100% !important;
    padding: 0 1rem 1rem !important;
}
[data-testid="column"] {
    padding: 0 5px !important; 
}
/* END AGGRESSIVE FIXES */


/* 🌟 TAB ENLARGEMENT AND GLOW EFFECT (New Rules B1 & B2) */

/* B1: ENLARGE TABS: Increase size, padding, and add a subtle box shadow */
[data-testid="stTabs"] button[role="tab"] {
    font-size: 16px !important; /* Enlarge text */
    padding: 12px 18px 12px 18px !important; /* Increase padding around text */
    min-height: 45px !important; /* Ensure minimum height */
    box-shadow: 0 2px 5px rgba(106, 90, 205, 0.1); /* Subtle shadow lift */
    transition: all 0.2s ease-in-out; /* Smooth transition for hover effect */
    font-weight: 700 !important; /* Make the text bolder */
}

/* B2: GLOW ON HOVER: Increase shadow/lift and change color when hovering */
[data-testid="stTabs"] button[role="tab"]:hover {
    color: #8a2be2 !important; /* Change text color to purple */
    border-bottom: 2px solid #8a2be2 !important; /* Highlight bottom border */
    box-shadow: 0 4px 15px rgba(138, 43, 226, 0.4); /* Stronger glow/lift effect */
    transform: translateY(-2px); /* Slight lift */
}

/* Ensure standard components have good contrast */
h1, h2, h3, h4, h5, h6, .stMarkdown, .stText {
    color: #1e1e1e !important; 
}

/* --- Your Existing Styles Below (Retained for functionality/design) --- */

.navbar { position: sticky; top: 0; z-index: 1000; padding: 12px 18px; margin: 0 0 18px 0; border-radius: 14px;
  background: linear-gradient(90deg, #6a5acd 0%, #8a2be2 100%); 
  box-shadow: 0 8px 20px rgba(0,0,0,0.1); 
  border: 1px solid rgba(255,255,255,0.35); display: flex; justify-content: space-between; align-items: center; }
.nav-title-wrap { display: flex; align-items: center; gap: 10px; }
.cashflow-girl { font-size: 30px; animation: float-money 2s ease-in-out infinite; position: relative; }
@keyframes float-money { 0% { transform: translateY(0px) rotate(0deg); } 25% { transform: translateY(-5px) rotate(5deg); }
  50% { transform: translateY(0px) rotate(0deg); } 75% { transform: translateY(-5px) rotate(-5deg); } 100% { transform: translateY(0px) rotate(0deg); } }
.nav-title { font-weight: 800; font-size: 24px; color:#ffffff; letter-spacing: .5px; }
.nav-sub { color:#ddddff; font-size:13px; margin-top:-2px; }
.coin-wrap { position: relative; height: 60px; margin: 6px 0 0 0; overflow: hidden; }
/* Applied coin-pulse to all .coin elements */
@keyframes coin-pulse {
    0%, 100% {{
        transform: scale(1.0) translateY(0px);
        filter: drop-shadow(0 0 8px gold) drop-shadow(0 0 3px orange);
    }}
    50% {{
        transform: scale(1.1) translateY(-2px);
        filter: drop-shadow(0 0 12px gold) drop-shadow(0 0 6px orange);
    }}
}
.coin { position:absolute; top:-50px; font-size:24px; 
    filter: drop-shadow(0 6px 8px rgba(0,0,0,.35)); 
    animation: drop 4s linear infinite, coin-pulse 2s ease-in-out infinite; }
.coin:nth-child(2){left:15%; animation-delay:.6s}
.coin:nth-child(3){left:30%; animation-delay:.1s}
.coin:nth-child(4){left:45%; animation-delay:.9s}
.coin:nth-child(5){left:60%; animation-delay:1.8s}
.coin:nth-child(6){left:75%; animation-delay:.3s}
.coin:nth-child(7){left:90%; animation-delay:.2s}
@keyframes drop { 0%{ transform: translateY(-60px) rotate(0deg); opacity:0 } 10%{ opacity:1 } 100%{ transform: translateY(120px) rotate(360deg); opacity:0 } }
.card {border-radius:16px; background:#ffffff; 
  padding:16px; box-shadow: 0 4px 15px rgba(106, 90, 205, 0.2); border: 1px solid #d4c1f5; color: #1e1e1e;}
.metric {font-size:18px; font-weight:700}
.bot {background:#f0f2f6; color:#1e1e1e; padding:10px 12px; border-radius:10px; border:1px solid #d4c1f5}
.streak-card{ border-radius:16px; padding:16px; margin-top:10px; background:#ffffff; 
  border:1px solid #d4c1f5; box-shadow:0 4px 15px rgba(106, 90, 205, 0.2); color: #1e1e1e;}
.piggy-wrap{ position:relative; height:84px; display:flex; align-items:center; gap:16px }
.piggy{ font-size:58px; filter: drop-shadow(0 6px 8px rgba(0,0,0,.35)); }
.piggy.dim{ opacity:.55; filter: grayscale(0.6) }
.coin-fall{ position:absolute; left:62px; top:-12px; font-size:22px; animation: fall 1.8s linear infinite; }
.coin-fall:nth-child(2){ left:84px; animation-delay:.4s }
.coin-fall:nth-child(3){ left:46px; animation-delay:.9s }
@keyframes fall { 0%{ transform: translateY(-30px) rotate(0deg); opacity:0 } 15%{ opacity:1 } 100%{ transform: translateY(85px) rotate(360deg); opacity:0 } }
.streak-metric{ font-weight:800; font-size:26px }
.badge-ok{ background:#6a5acd; color:white; padding:4px 10px; border-radius:999px; font-size:12px }
.profile-wrap{display:flex;align-items:center;justify-content:flex-end}
.profile-pic{ width:70px;height:70px;border-radius:50%;object-fit:cover; box-shadow:0 6px 20px rgba(0,0,0,.35); border:2px solid #25D366; }
.upi-qr-wrap {
  position: relative; border-radius: 12px; padding: 10px;
  background: rgba(138, 43, 226, 0.1);
  border: 1px solid rgba(138, 43, 226, 0.5);
  box-shadow: 0 0 15px rgba(138, 43, 226, 0.3);
  animation: qr-glow 2s infinite alternate, qr-flicker 1.5s step-end infinite;
}
@keyframes qr-glow {
  0% { box-shadow: 0 0 10px rgba(138, 43, 226, 0.2); transform: scale(1); }
  50% { transform: scale(1.01); }
  100% { box-shadow: 0 0 20px rgba(138, 43, 226, 0.5); transform: scale(1); }
}
@keyframes qr-flicker { 0%, 100% { opacity: 1; } 50% { opacity: 0.9; } }
.promise{ font-weight:900; font-size:20px; letter-spacing:.3px; color:#6a5acd; text-align:center; margin:8px 0 2px 0;
  animation: none; } 
.callout-box-vfa { background: #8a2be2; color: white; padding: 8px 12px; border-radius: 8px; font-weight: 600; margin-top: 15px; display: flex; align-items: center; gap: 10px; animation: none; }
.animated-arrow-vfa { font-size: 24px; animation: pulsing_arrow 1.5s infinite; display: inline-block; }
.stSuccess { background-color: #e6f7e9 !important; border-left: 5px solid #22c55e !important; color: #1e1e1e !important; }
.stInfo { background-color: #e6f1ff !important; border-left: 5px solid #6a5acd !important; color: #1e1e1e !important; }

/* FIXES FOR TEXT CONTRAST IN THE AI PLAN */
.stApp div[data-testid^="stExpander"] * { color: #1e1e1e !important; }
.stApp div[style*="rgba(142, 45, 226, 0.1)"] * { color: #1e1e1e !important; } 

/* Fixes for dark charts displayed in the dark theme setting */
.modebar, .c-modebar { background: #1e1e1e; }
.js-plotly-plot { background: #ffffff !important; }

/* Custom radio button styles for multi-color */
.multicolor-radio > div[data-testid="stRadio"] label:nth-child(1) span { background-color: #ffeb3b; color: #1e1e1e; border-color: #ffeb3b; } 
.multicolor-radio > div[data-testid="stRadio"] label:nth-child(2) span { background-color: #ff9800; color: white; border-color: #ff9800; } 
.multicolor-radio > div[data-testid="stRadio"] label:nth-child(3) span { background-color: #2196f3; color: white; border-color: #2196f3; } 

.multicolor-radio-commute > div[data-testid="stRadio"] label:nth-child(1) span { background-color: #4caf50; color: white; border-color: #4caf50; } 
.multicolor-radio-commute > div[data-testid="stRadio"] label:nth-child(2) span { background-color: #ff9800; color: white; border-color: #ff9800; } 
.multicolor-radio-commute > div[data-testid="stRadio"] label:nth-child(3) span { background-color: #f44336; color: white; border-color: #f44336; } 

.multicolor-radio div[data-testid="stRadio"] label span,
.multicolor-radio-commute div[data-testid="stRadio"] label span {
    padding: 8px 12px;
    border-radius: 8px;
    font-weight: 700;
    transition: all 0.2s;
}
.multicolor-radio div[data-testid="stRadio"] input:checked + div > span,
.multicolor-radio-commute div[data-testid="stRadio"] input:checked + div > span {
    border: 3px solid #6a5acd !important; 
    box-shadow: 0 0 10px rgba(106, 90, 205, 0.7);
}
</style>
""",
    unsafe_allow_html=True,
)
if st.session_state["coin_rain_show"]:
    show_coin_rain(RAIN_DURATION_SEC)

CURRENT_USER_ID = st.session_state["auth_user"]

# ---------- Navbar ----------
colA, colB = st.columns([4, 0.6])
with colA:
    st.markdown(
        f"""
    <div class="navbar">
      <div class="nav-title-wrap">
        <span class="cashflow-girl">👩‍💰💸</span>
        <div>
          <div class="nav-title">📈💰📊 Personal Finance AI Dashboard <br> <span style="font-size:18px;">Cashflow Crew ({CURRENT_USER_ID})</span></div>
          <div class="nav-sub">Visualize expenses, savings & investments — premium, Power BI–style UI</div>
        </div>
      </div>
      <div class="coin-wrap">
        <span class="coin">🪙</span><span class="coin">💰</span><span class="coin">🪙</span>
        <span class="coin">💰</span><span class="coin">🪙</span><span class="coin">💰</span><span class="coin">🪙</span>
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

with colB:
    st.markdown("<div class='profile-wrap'>", unsafe_allow_html=True)
    sound_status = "🔊 ON" if not st.session_state.get("sound_muted", False) else "🔇 OFF"
    if st.button(sound_status, key="toggle_sound", help="Toggle payment notification sound"):
        st.session_state["sound_muted"] = not st.session_state.get("sound_muted", False)
        st.rerun()
    if PROFILE64:
        st.markdown(
            f"""<img class="profile-pic" src="data:image/jpg;base64,{PROFILE64}" />""",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

if HAS_GEMINI_SDK:
    st.success("🎉 **Now integrated with GEMINI!** Access intelligent financial guidance via the Smart Chatbot and AI Plan.")
else:
    st.error("⚠️ **GEMINI SDK Missing:** Chatbot intelligence is disabled. Please run `pip install google-genai`.")

if "promise_text" not in st.session_state:
    st.session_state["promise_text"] = "I promise that I will save 100 rupees per day"

st.markdown(f"<div class='promise'>{st.session_state['promise_text']}</div>", unsafe_allow_html=True)
new_p = st.text_input("Change promise line", st.session_state["promise_text"])
if new_p != st.session_state["promise_text"]:
    st.session_state["promise_text"] = new_p
    st.rerun()

# --- Load data outside of tabs ---
db_txns = DB.list_txns(CURRENT_USER_ID)
if not db_txns:
    raw_df = generate_sample(1)
    st.info(f"No saved transactions found for **{CURRENT_USER_ID}**. Showing 1 month of sample data.")
else:
    raw_df = pd.DataFrame([asdict(t) for t in db_txns])
    raw_df["date"] = pd.to_datetime(raw_df["date"]).dt.date

if raw_df is None:
    st.error("Fatal error: Could not load any transaction data.")
    st.stop()

try:
    df = normalize(raw_df)
except Exception as e:
    st.error(f"Error normalizing data: {e}. Please check column names.")
    st.stop()

# --- Tabs ---
tab_dashboard, tab_stock, tab_plan, tab_city, tab_ca_plan, tab_tools = st.tabs([
    "💰 Personal Dashboard",
    "📈 Real-time Stock Data (AlphaVantage)",
    "🎯 AI Financial Plan",
    "🏙️ City Affordability",
    "🧑‍💼 Personal CA Blueprint", # NEW TAB
    "🧰 Tools (Edit • Backup • Dedupe)"
])

with tab_dashboard:
    tb1, tb2, tb3, tb4, tb5, tb6, tb7 = st.columns([1.6, 1.4, 1.4, 1.8, 1.2, 1, 1.4])
    with tb1:
        data_source = st.radio("Data source", ["Use saved data", "Generate sample"], index=0, horizontal=True)

    if data_source == "Generate sample":
        raw_df_local = generate_sample(6)
    else:
        raw_df_local = raw_df.copy()

    try:
        df_local = normalize(raw_df_local)
    except Exception as e:
        st.error(f"Error normalizing data: {e}. Please check column names.")
        st.stop()

    with tb2:
        plot_type = st.selectbox(
            "Plot type",
            [
                "Line plot (trend)",
                "Bar plot (aggregate)",
                "Count plot (category counts)",
                "Scatter plot",
                "Distribution (KDE)",
                "Histogram",
                "Donut Chart",
                "Heatmap",
            ],
        )
    with tb3:
        group_period = st.selectbox("Group period", ["Monthly", "Weekly", "Daily"], index=0)
    with tb4:
        default_bar_mode = 1 if plot_type.startswith("Bar") or plot_type.startswith("Line") else 0
        bar_mode = st.selectbox("Bar mode", ["By Category", "By Period (stacked by type)"], index=default_bar_mode)
    with tb5:
        numeric_col = st.selectbox("Numeric (scatter/hist)", ["amount"], index=0)
    with tb6:
        if st.button("Logout", key="logout_1"):
            for k in (
                "auth_ok",
                "auth_user",
                "chat_history",
                "coin_rain_show",
                "coin_rain_start",
                "longest_streak_ever",
                "promise_text",
                "last_quote",
                "daily_data",
                "DB",
                "ml_login_step",
                "ml_face_code_live",
                "user_budgets",
                "weather_city",
                "weather_data",
                "global_budgets",
                "health_score_data",
                "goal_target", # Clear goal data
                "goal_date",
                "goal_current",
                "ca_plan_json", # Clear new state
                "ca_plan_explanation",
                "ca_plan_tts_summary"
            ):
                st.session_state.pop(k, None)
            st.rerun()
    with tb7:
        st.markdown("Weather City")
        new_city = st.text_input(" ", st.session_state["weather_city"], label_visibility="collapsed")
        if new_city != st.session_state["weather_city"]:
            st.session_state["weather_city"] = new_city
            st.session_state["weather_data"] = get_weather(st.session_state["weather_city"])
            st.rerun()

    weather_data = st.session_state.get("weather_data")
    hint_text = spend_mood_hint(weather_data)
    st.markdown(
        f"""
    <div style="background-color: #f0f2f6; padding: 10px; border-radius: 8px; margin-bottom: 15px; border-left: 5px solid #6a5acd; color: #1e1e1e;">
    	<span style="font-weight: bold; color: #6a5acd;">🌤️ Spending Mood Hint:</span> {hint_text}
    </div>
    """,
        unsafe_allow_html=True,
    )

    # --- Filters ---
    f1, f2, f3 = st.columns([1.3, 1.6, 1.1])
    
    # Initialize start and end with safe defaults to prevent NameError
    start: date = date.today() - timedelta(days=365)  
    end: date = date.today()
    sel_cats: List[str] = []
    sel_types: List[str] = []

    if df_local.empty:
        view = df_local.copy()
        tmp = add_period(view, group_period)
    else:
        min_d = df_local["date"].min()
        max_d = df_local["date"].max()
        # Ensure date inputs run and set variables inside the `else` block scope
        with f1:
            start = st.date_input("Start date", min_value=min_d, max_value=max_d, value=min_d, key="start_1")
            end = st.date_input("End date", min_value=min_d, max_value=max_d, value=max_d, key="end_1")
        with f2:
            cats = sorted(df_local["category"].unique().tolist())
            sel_cats = st.multiselect("Categories", options=cats, default=cats)
        with f3:
            types = sorted(df_local["type"].unique().tolist())
            sel_types = st.multiselect("Types", options=types, default=types)

        # Apply filtering logic now that start and end are guaranteed to be defined
        mask = (df_local["date"] >= start) & (df_local["date"] <= end)
        view = df_local[mask & df_local["category"].isin(sel_cats) & df_local["type"].isin(sel_types)].copy()
        tmp = add_period(view, group_period)
    
    # --- Goal Tracker ---
    st.markdown("---")
    st.subheader("🎯 Goal Tracker: Achieve Your Milestones")
    
    goal_col1, goal_col2, goal_col3 = st.columns(3)
    
    with goal_col1:
        st.session_state["goal_target"] = st.number_input(
            "Target Amount (₹)", 
            min_value=1000, 
            value=st.session_state["goal_target"], 
            step=1000
        )
        st.session_state["goal_current"] = st.number_input(
            "Current Saved (₹)", 
            min_value=0, 
            value=int(st.session_state["goal_current"]), 
            step=1000
        )
    
    with goal_col2:
        st.session_state["goal_date"] = st.date_input(
            "Target Date", 
            value=st.session_state["goal_date"],
            min_value=date.today() + timedelta(days=1)
        )
        
    if isinstance(st.session_state["goal_date"], date):
        # Proceed with subtraction only on valid date objects
        time_delta = st.session_state["goal_date"] - date.today()
        # Use .days property, which is safe if time_delta is a timedelta
        days_to_go = max(1, time_delta.days)
    else:
        # Fallback to 1 day if the date is invalid or uninitialized
        days_to_go = 1


    remaining_target = max(0, st.session_state["goal_target"] - st.session_state["goal_current"])
    required_daily_saving = remaining_target / days_to_go
    st.metric("Days Remaining", f"{days_to_go} days")

    with goal_col3:
        st.metric("Required Daily Saving", money(required_daily_saving))
        st.metric("Required Monthly Saving", money(required_daily_saving * 30.4))
        
    # Goal Progress Chart Logic
    if not df_local.empty and remaining_target > 0:
        # Calculate daily cumulative net savings from the start of the goal
        df_goal_period = df_local[(df_local["date"] >= date.today()) & (df_local["date"] <= st.session_state["goal_date"])]
        
        daily_net = daily_net_frame(df_goal_period).set_index("day")
        if daily_net.empty:
            st.info("No transactions recorded in the current goal period yet.")
        else:
            daily_net['Cumulative_Saving'] = daily_net['net_saving'].cumsum() + st.session_state["goal_current"]
            
            # Create a full date range for the target line
            full_range = pd.date_range(start=date.today(), end=st.session_state["goal_date"], freq='D')
            target_df = pd.DataFrame(index=full_range)
            
            # Calculate linear required progress
            target_df['Required_Cumulative'] = st.session_state["goal_current"] + (st.session_state["goal_target"] - st.session_state["goal_current"]) * (
                (target_df.index.date - date.today()) / (st.session_state["goal_date"] - date.today())
            ).days
            target_df.iloc[-1, target_df.columns.get_loc('Required_Cumulative')] = st.session_state["goal_target"] # Ensure target date hits the amount

            # Merge for plotting
            plot_data = daily_net.join(target_df, how='outer').fillna(method='ffill')
            plot_data['Date'] = plot_data.index
            
            # Plotly Chart
            fig_goal = go.Figure()

            # Add actual cumulative savings (Actual Progress)
            fig_goal.add_trace(go.Scatter(
                x=plot_data['Date'], 
                y=plot_data['Cumulative_Saving'], 
                mode='lines', 
                name='Actual Progress',
                line=dict(color='#6a5acd', width=3)
            ))

            # Add required cumulative savings (Required Path)
            fig_goal.add_trace(go.Scatter(
                x=target_df.index, 
                y=target_df['Required_Cumulative'], 
                mode='lines', 
                name='Required Path',
                line=dict(color='#8a2be2', dash='dot', width=2)
            ))
            
            # Add target marker
            fig_goal.add_annotation(
                x=st.session_state["goal_date"], y=st.session_state["goal_target"],
                text=f"🎯 Target: {money(st.session_state['goal_target'])}",
                showarrow=True, arrowhead=1, ax=-50, ay=-30,
                font=dict(color="#8a2be2", size=14)
            )

            fig_goal.update_layout(
                title=f"Savings Goal Progress: {money(remaining_target)} Remaining",
                xaxis_title="Date",
                yaxis_title="Cumulative Saved (₹)",
                template="plotly_dark",
                height=450,
                hovermode="x unified"
            )
            st.plotly_chart(fig_goal, use_container_width=True)
            
            st.info(f"You need to maintain an average daily saving of **{money(required_daily_saving)}** to reach your goal of **{money(st.session_state['goal_target'])}** by **{st.session_state['goal_date']}**.")

    # --- Health Score + Budgets ---
    st.markdown("---")
    top_left_col, top_mid_col, top_right_col = st.columns([1.2, 1.5, 2])

    with top_left_col:
        current_budgets = st.session_state["global_budgets"].get(CURRENT_USER_ID, {})
        budget_allocation = auto_allocate_budget(df_local, savings_target_pct=0.15)
        updated_budget, apply_save = budget_bot_minicard(budget_allocation)
        if apply_save:
            updated_budget_lower = {k.lower(): v for k, v in updated_budget.items()}
            st.session_state["global_budgets"][CURRENT_USER_ID] = updated_budget_lower
            st.success("Budgets applied to your profile! Health Score updated.")
            st.rerun()
        current_budgets = st.session_state["global_budgets"].get(CURRENT_USER_ID, {})
        curr_ns, longest_ns = no_spend_streak(df_local)
        display_badges(curr_ns)

    with top_mid_col:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        total_income = view[view["type"] == "income"]["amount"].sum() if not view.empty else 0
        total_expense = view[view["type"] == "expense"]["amount"].sum() if not view.empty else 0
        net = total_income - total_expense
        if not tmp.empty:
            avg_per = tmp.groupby("period")["amount"].sum().mean()
        else:
            avg_per = 0
        m1.metric("Total Income", money(total_income))
        m2.metric("Total Expense", money(total_expense))
        m3.metric("Net", money(net))
        m4.metric(f"Avg {group_period}", money(avg_per))
        st.markdown("</div>", unsafe_allow_html=True)

    with top_right_col:
        health_score_data = compute_fin_health_score(df_local, budgets=current_budgets)
        display_health_score(health_score_data)
        st.session_state["health_score_data"] = health_score_data

        # === NEW: Budget overrun alerts (current month) ===
        try:
            now = pd.Timestamp.today()
            month_mask = (df_local["date"] >= now.replace(day=1).date()) & (df_local["date"] <= now.date())
            this_month = df_local[month_mask & (df_local["type"] == "expense")]
            if not this_month.empty and current_budgets:
                spent_by_cat = this_month.groupby("category")["amount"].sum().to_dict()
                over_list = []
                for cat, limit in current_budgets.items():
                    spent = float(spent_by_cat.get(cat, 0.0))
                    if limit and spent > float(limit):
                        over_list.append((cat, spent, float(limit)))
                if over_list:
                    st.markdown("---")
                    st.error("🚨 **Budget alerts (this month):**")
                    for cat, spent, limit in over_list:
                        st.write(f"• **{cat}** over by **{money(spent - limit)}** (Spent {money(spent)} / Budget {money(limit)})")
        except Exception:
            pass

    # --- Saving Streak ---
    st.markdown("---")
    st.markdown("<div class='streak-card'>", unsafe_allow_html=True)
    cA, cB, cC, cD = st.columns([1.3, 1.1, 1, 1.6])
    with cA:
        st.markdown("Daily Saving Target (₹)")
        target_daily = st.number_input(" ", min_value=0, value=200, step=50, label_visibility="collapsed", key="target_daily_1")
    with cB:
        st.markdown("Strict mode")
        strict = st.checkbox("Require ≥ target", value=True, key="strict_1")
    with cC:
        st.markdown("Show last N days")
        lookback = st.slider(" ", 7, 60, 14, label_visibility="collapsed", key="lookback_1")
    with cD:
        st.markdown("Info")
        st.markdown("<span class='badge-ok'>Net = income − expense</span>", unsafe_allow_html=True)

    dn = daily_net_frame(df_local)
    curr_streak = health_score_data["factors"]["no_spend_streak"]
    longest_streak = health_score_data["factors"]["longest_no_spend"]

    if not dn.empty:
        dn_last = dn.tail(lookback).copy()
        thresh = target_daily if strict else max(1, target_daily * 0.6)
        hit = dn_last["net_saving"] >= thresh
        hit.index = dn_last["day"]
        local_curr_streak, local_longest_streak = compute_streak(hit)

        pig_col, s1, s2, s3 = st.columns([1.1, 1, 1, 1.6])
        today_date = date.today()
        val_today = (
            dn_last[dn_last["day"] == today_date]["net_saving"].iloc[-1]
            if today_date in dn_last["day"].values
            else 0
        )
        today_hit = val_today >= thresh
        pig_class = "piggy" + ("" if today_hit else " dim")
        coins_html = (
            '<div class="coin-fall">🪙</div><div class="coin-fall">🪙</div><div class="coin-fall">🪙</div>'
            if today_hit else ""
        )

        with pig_col:
            st.markdown(
                f"""
                <div class="piggy-wrap">
                  <div class="{pig_class}">🐷</div>
                  {coins_html}
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption("Piggy lights up when today's net meets target.")

        with s1:
            st.markdown("Today")
            st.markdown(f"<div class='streak-metric'>{'✅' if today_hit else '❌'}</div>", unsafe_allow_html=True)
            st.caption(f"Saved: {money(val_today)} / ₹{target_daily:,}")

        with s2:
            st.markdown("Current Streak (Local)")
            st.markdown(f"<div class='streak-metric'>{local_curr_streak} days</div>", unsafe_allow_html=True)

        with s3:
            st.markdown("Longest Streak (Local)")
            st.markdown(f"<div class='streak-metric'>{local_longest_streak} days</div>", unsafe_allow_html=True)
            st.caption(f"Overall No-Spend: {longest_ns} days")

        mini = dn_last.copy()
        mini["hit"] = np.where(mini["net_saving"] >= thresh, "Hit", "Miss")
        fig_streak = px.bar(
            mini.reset_index(),
            x="day",
            y="net_saving",
            color="hit",
            color_discrete_map={"Hit": "#6a5acd", "Miss": "#ef4444"},
            title=f"Net saving (last {lookback} days)",
            labels={"day": "Day", "net_saving": "₹"},
        )
        fig_streak.update_layout(height=260, showlegend=True, legend_title="", template="plotly_dark")
        st.plotly_chart(fig_streak, use_container_width=True, config={"displayModeBar": False}, key="streak_chart_1")
    else:
        st.info("No transactions in the current date range to compute a streak.")
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- UPI QR + CSV Upload + VFA Plan Download + OCR ---
    left_col, right_col = st.columns([1.3, 2.7])

    with left_col:
        st.markdown('<div class="upi-qr-wrap">', unsafe_allow_html=True)
        st.subheader("Add Income/Upload Data")

        uploaded_file, ocr_data = glowing_ocr_uploader()

        st.markdown("---")
        st.markdown("#### Upload Transactions File (CSV/Excel)")
        uploaded_csv = st.file_uploader("Upload .csv or .xlsx", type=["csv", "xlsx"], key="direct_csv_upload")

        if uploaded_csv is not None:
            try:
                uploaded_df = read_file(uploaded_csv)
                cols_lower = [c.lower() for c in uploaded_df.columns]
                if not all(col in cols_lower for col in ["date", "amount"]):
                    st.error("File must contain 'date' and 'amount' columns.")
                else:
                    uploaded_df.columns = cols_lower
                    uploaded_df.rename(columns={
                        'date': 'date',
                        'amount': 'amount',
                        'merchant': 'category',
                        'type': 'type'
                    }, errors='ignore', inplace=True)
                    uploaded_df = normalize(uploaded_df)
                    save_transactions(CURRENT_USER_ID, uploaded_df)
                    DB.save()
                    green_tick("File uploaded and data saved successfully!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error processing file: {e}")
                st.info("Ensure file has 'date', 'amount', 'category', 'type' columns, and date formats are correct.")

        st.markdown("---")
        st.markdown("#### Manual/UPI Expense/Income Entry")

        default_amount = float((ocr_data or {}).get("amount", 100.0))
        default_date = (ocr_data or {}).get("date") or date.today()
        default_desc = (ocr_data or {}).get("merchant") or "Manual Entry"
        default_cat = (ocr_data or {}).get("category") or "uncategorized"

        try:
            pd_date = pd.to_datetime(default_date, errors='coerce')
            safe_default_date = pd_date.date() if not pd.isna(pd_date) else date.today()
        except Exception:
            safe_default_date = date.today()

        with st.form("manual_txn_form", clear_on_submit=True):
            txn_date = st.date_input("Date", value=safe_default_date)
            txn_type = st.radio("Type", ["expense", "income"], horizontal=True, index=0)
            txn_amt = st.number_input("Amount (₹)", min_value=1.0, value=float(default_amount), step=1.0)

            all_cats = sorted(df_local["category"].unique().tolist())
            if default_cat not in all_cats:
                all_cats.insert(0, default_cat)
            try:
                default_index = all_cats.index(default_cat)
            except ValueError:
                default_index = 0
            txn_cat = st.selectbox("Category", options=all_cats, index=default_index)
            txn_desc = st.text_input("Description/Merchant", value=default_desc)

            if st.form_submit_button("Add Transaction to DB", use_container_width=True):
                DB.add_txn(
                    user_id=CURRENT_USER_ID,
                    dt=txn_date,
                    amount=float(txn_amt),
                    category=txn_cat.lower() if txn_cat != "new" else "uncategorized",
                    description=txn_desc,
                    typ=txn_type,
                )
                DB.save()
                if txn_type == "income":
                    # Update current savings if categorized as income
                    st.session_state["goal_current"] += float(txn_amt)
                    play_paid_sound(CURRENT_USER_ID, float(txn_amt))
                    green_tick(f"Income of {money(txn_amt)} recorded successfully!")
                else:
                    # Deduct from current savings if categorized as expense
                    st.session_state["goal_current"] -= float(txn_amt)
                    green_tick(f"Expense of {money(txn_amt)} recorded successfully!")
                st.rerun()

        bucket_total = DB.piggy_balance(CURRENT_USER_ID, "collection")
        st.markdown(
            f"*Bucket total (Income):* <span style='font-weight:700'>{money(bucket_total)}</span>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.subheader("💡 Personal Virtual Financial Advisor (VFA)")
        st.markdown(
            """
            <div class="callout-box-vfa">
                <span class="animated-arrow-vfa">➡️</span>
                <span>Your VFA has new insights!</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("The VFA analyzes your spending history to generate a personalized action plan.")

        st.markdown("#### 🔮 Next Month Expense Forecast")
        forecast_df = forecast_next_month(df_local)
        st.dataframe(forecast_df.sort_values("forecasted_expense", ascending=False).head(5), use_container_width=True, hide_index=True)
        st.caption("Forecasted based on recent spending patterns (ETS/SMA).")

        st.markdown("---")

        plan_bytes = generate_financial_plan_file(view)
        st.download_button(
            label="Download Personalized Financial Plan (CSV)",
            data=plan_bytes,
            file_name=f"VFA_Plan_{CURRENT_USER_ID}_{date.today()}.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_vfa_plan",
        )
        st.caption("This file contains actionable advice based on your current data.")
        st.markdown("---")

        st.subheader("Chat with PRAKRITI AI")
        st.markdown("#### 🧠 Daily AI Tip Panel")

        if not HAS_GEMINI_SDK or not GEMINI_API_KEY:
            st.error("AI Tip Panel disabled: GEMINI SDK missing or key not set.")
        else:
            tip_key = f"ai_tip_{CURRENT_USER_ID}_{date.today().isoformat()}"
            if tip_key not in st.session_state:
                spikes = detect_trend_spikes(df_local, window="30D")
                summary = {
                    "total_income": float(view[view["type"] == "income"]["amount"].sum()),
                    "total_expense": float(view[view["type"] == "expense"]["amount"].sum()),
                    "net_savings": float(view[view["type"] == "income"]["amount"].sum() - view[view["type"] == "expense"]["amount"].sum()),
                    "top_expenses": view[view["type"] == "expense"]
                    .groupby("category")["amount"]
                    .sum()
                    .sort_values(ascending=False)
                    .head(3)
                    .to_dict(),
                }
                health = st.session_state["health_score_data"]["score"]
                context_prompt = f"""
                Analyze spending for {CURRENT_USER_ID} (window={start}..{end}).
                Financial Health Score: {health}.
                Summary: {summary}.
                Top spikes (30-day window): {spikes}.
                Suggest 2 concrete, actionable financial hacks with estimated INR savings (e.g., '₹500/month').
                Format your response strictly as an unordered bulleted list, maximum 2 lines per bullet point.
                """
                try:
                    # FIX: Added st.spinner around this slow API call
                    with st.spinner("🧠 Generating personalized AI Tip..."):
                        ans = gemini_query(
                            context_prompt,
                            st.session_state.get("chat_history", [])[-2:],
                            f"You are a financial coach. The user is {CURRENT_USER_ID} and their net is {money(summary['net_savings'])}.",
                        )
                    tip_text = ans.replace("🧠 *Gemini Smart AI:*", "").strip()
                    st.session_state[tip_key] = tip_text.replace("*", "").strip()
                except Exception as e:
                    st.session_state[tip_key] = f"❌ **AI Tip Generation Error:** {e}"

            st.markdown(
                f"""
                <div class='bot' style='border-left: 5px solid #8a2be2;'>
                    <span style='font-weight: bold;'>Your personalized insights:</span>
                    {st.session_state[tip_key]}
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        with st.form("chatbot_form", clear_on_submit=True):
            user_q = st.text_input(
                "Ask (e.g., 'top categories', 'invest advice', 'help')",
                key="chat_input_form",
            )
            send_button = st.form_submit_button("Send")
            if send_button and user_q:
                st.session_state["thinking"] = True
                context_str = (
                    TEAM_INFO["Team Leader"]
                    + " is a "
                    + TEAM_INFO["Leader Expertise"]
                    + ". The user is "
                    + CURRENT_USER_ID
                    + " and their current net savings is "
                    + money(view[view["type"] == "income"]["amount"].sum() - view[view["type"] == "expense"]["amount"].sum())
                    + "."
                )
                ans = None
                if ans is None:
                    try:
                        # FIX: Added st.spinner around this slow API call
                        with st.spinner("🤖 Thinking..."):
                            ans = gemini_query(user_q, st.session_state.get("chat_history", []), context_str)
                    except Exception as e:
                        ans = f"❌ **Error contacting AI:** {e}"
                st.session_state.setdefault("chat_history", []).append(("You", user_q))
                st.session_state["chat_history"].append(("Bot", ans))
                st.session_state["thinking"] = False
                st.rerun()

        for speaker, msg in st.session_state.get("chat_history", [])[-12:]:
            if speaker == "You":
                st.markdown(f"*You:* {msg}")
            else:
                st.markdown(f"<div class='bot'>{msg}</div>", unsafe_allow_html=True)

        # --- Smart Machine (Voice Coach) — Bilingual ---
        st.markdown("---")
        st.subheader("🎙️ Smart Machine (Voice Coach) — Bilingual")
        st.caption("Wake word **“smart machine”**: Hindi + English tips. Or press **Speak now**.")

        advice_hi, advice_en = build_smart_advice_bilingual(view)

        c1, c2 = st.columns(2)
        with c1:
            order = st.selectbox("Order", ["hi-en", "en-hi"], index=0)
        with c2:
            enable_listener = st.toggle("Enable wake word", value=False, key="smart_machine_listener_toggle")

        if st.button("🔊 Speak now", key="speak_bilingual_now"):
            speak_bilingual_js(advice_hi, advice_en, order=order)

        if enable_listener:
            smart_machine_listener(advice_hi, advice_en, wake_word="smart machine", order=order)
            st.info("Listening… say **smart machine**. Allow mic permission; keep this tab focused.")

        # ---------- Main charts & table ----------
        st.markdown("---")
        st.subheader("Main Charts")

        # --- Deep Dive Analytics Section ---
        st.markdown("---")
        st.markdown("## 📊 Deep Dive Analytics (Advanced Visualizations)")

        # Use 2 columns for a 2x2 grid layout
        col_chart_A, col_chart_B = st.columns(2)
        col_chart_C, col_chart_D = st.columns(2)

        # --- 1. Income vs Savings Rate (Donut Chart) ---
        with col_chart_A:
            chart_name_1 = "Income vs Savings Rate (Donut Chart)"
            st.markdown(f"#### 💰 **Graph 1:** {chart_name_1}")
            
            # Calculate total net savings (income - expense)
            income_sum = df_local[df_local['type'] == 'income']['amount'].sum()
            expense_sum = df_local[df_local['type'] == 'expense']['amount'].sum()
            
            # Savings Rate = (Income - Expense)
            # We cap savings rate at total income to avoid negative visualization in a pie chart (though negative is possible in reality)
            savings_amount = max(0, income_sum - expense_sum)
            expense_rate_amount = min(income_sum, expense_sum)
            
            # If income is 0, default to 50/50 for visual representation
            if income_sum == 0 and expense_sum == 0:
                donut_data = pd.DataFrame({'Rate': ['Savings Rate', 'Expense Rate'], 'Amount': [1, 1]})
                title_suffix = " (No Data)"
            elif income_sum > 0:
                donut_data = pd.DataFrame({
                    'Rate': ['Savings Rate', 'Expense Rate'],
                    'Amount': [savings_amount, expense_rate_amount]
                })
                title_suffix = ""
            else: # Only expenses, income is zero
                donut_data = pd.DataFrame({'Rate': ['Savings Rate', 'Expense Rate'], 'Amount': [0, 100]})
                title_suffix = " (100% Expense)"


            fig_donut = px.pie(
                donut_data,
                values='Amount',
                names='Rate',
                title=f"Income Allocation (Net Savings) {title_suffix}",
                hole=0.65,
                color='Rate',
                color_discrete_map={'Savings Rate': '#22c55e', 'Expense Rate': '#e0e0e0'}
            )
            
            fig_donut.update_traces(
                textinfo='percent',  
                hoverinfo='label+percent+value',
                marker=dict(line=dict(color='#ffffff', width=2))
            )
            # Use contrasting background/paper colors for visibility
            fig_donut.update_layout(
                template="plotly_white",  
                paper_bgcolor='#ffffff',  
                plot_bgcolor='#ffffff',  
                height=380,  
                showlegend=True,  
                legend=dict(orientation="h", yanchor="bottom", y=-0.2),
                margin=dict(l=0,r=0,t=40,b=0)
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        # 🎤 Explainer for Graph 1
        _chart_explainer(
            "G1", chart_name_1,
            f"Total Income: {money(income_sum)}. Total Expense: {money(expense_sum)}. Net Savings: {money(savings_amount)}."
        )


        # --- 2. Monthly Expense vs. Income (Area Chart) ---
        with col_chart_B:
            chart_name_2 = "Monthly Cash Flow Trend (Area Chart)"
            st.markdown(f"#### 📈 **Graph 2:** {chart_name_2}")
            
            monthly_flow_df = add_period(df_local, group_period="Monthly")
            monthly_net = monthly_flow_df.groupby(["period", "type"])["amount"].sum().unstack(fill_value=0)
            
            if monthly_net.empty:
                st.info("No data for Monthly Cash Flow.")
            else:
                # Ensure columns exist for consistency
                if 'income' not in monthly_net.columns: monthly_net['income'] = 0
                if 'expense' not in monthly_net.columns: monthly_net['expense'] = 0
                monthly_net.index.name = 'Month'
                
                fig_cashflow = go.Figure()

                # Income Area/Line (Green)
                fig_cashflow.add_trace(go.Scatter(
                    x=monthly_net.index, y=monthly_net['income'], 
                    mode='lines', name='Income', 
                    fill='tozeroy', fillcolor='rgba(34, 197, 94, 0.5)', # Green
                    line=dict(color='#22c55e', width=2),
                    stackgroup='one'
                ))

                # Expense Area/Line (Red)
                fig_cashflow.add_trace(go.Scatter(
                    x=monthly_net.index, y=monthly_net['expense'], 
                    mode='lines', name='Expenses', 
                    fill='tonexty', fillcolor='rgba(239, 68, 68, 0.5)', # Red
                    line=dict(color='#ef4444', width=2),
                    stackgroup='one'
                ))

                fig_cashflow.update_layout(
                    title='Income vs. Expenses Over Time',
                    xaxis_title="Month",
                    yaxis_title="Amount (₹)",
                    template="plotly_white", # Use plotly_white for light background
                    paper_bgcolor='#ffffff', 
                    plot_bgcolor='#ffffff', 
                    height=380,
                    legend=dict(orientation="h", yanchor="top", y=1.1, xanchor="left", x=0),
                    margin=dict(l=0,r=0,t=40,b=0)
                )
                st.plotly_chart(fig_cashflow, use_container_width=True)

        # 🎤 Explainer for Graph 2
        _chart_explainer(
            "G2", chart_name_2,
            f"Monthly net data: {monthly_net.tail(2).to_dict() if not monthly_net.empty else 'No data'}"
        )
            
        # --- 3. Risk vs Return Analysis (Bubble Chart) ---
        with col_chart_C:
            chart_name_3 = "Risk vs Return Analysis (Bubble Chart)"
            st.markdown(f"#### 💹 **Graph 3:** {chart_name_3}")
            
            # Simulated data for Risk vs. Return (Mock Investment Options)
            # Data: [Risk (X), Expected Return (Y), Value/Size (Z), Option Name]
            risk_return_data = [
                (2, 6, 100000, 'Govt. Bonds'),
                (3, 7.5, 150000, 'Fixed Deposits'),
                (4, 8.5, 200000, 'Mutual Fund (Low Risk)'),
                (5.5, 10, 300000, 'Balanced Fund'),
                (7, 13, 400000, 'Equity Fund (Mid Risk)'),
                (8.5, 15.5, 500000, 'Small Cap Equity'),
            ]
            
            risk_df = pd.DataFrame(risk_return_data, columns=['Risk Level', 'Expected Return (%)', 'Investment Value (₹)', 'Investment Option'])
            
            fig_risk_return = px.scatter(
                risk_df,
                x='Risk Level',
                y='Expected Return (%)',
                size='Investment Value (₹)',
                color='Investment Option',
                hover_name='Investment Option',
                size_max=40,
                title='Risk (Volatility) vs. Expected Return (%)'
            )
            
            fig_risk_return.update_layout(
                template="plotly_white",
                paper_bgcolor='#ffffff', 
                plot_bgcolor='#ffffff', 
                height=380,
                xaxis=dict(range=[0, 10], dtick=2),
                yaxis=dict(range=[0, 20], dtick=4),
                margin=dict(l=0,r=0,t=40,b=0)
            )
            st.plotly_chart(fig_risk_return, use_container_width=True)

        # 🎤 Explainer for Graph 3
        _chart_explainer(
            "G3", chart_name_3,
            "This is a theoretical asset allocation showing risk vs return for sample investments. Size represents value."
        )

        # --- 4. Tax Savings Breakdown (Radar Chart) ---
        with col_chart_D:
            chart_name_4 = "Tax Savings Breakdown (Radar Chart)"
            st.markdown(f"#### 🛡️ **Graph 4:** {chart_name_4}")
            
            # Simulated data for tax sections
            tax_sections = [
                "80C (PPF/ELSS)", 
                "80D (Health Insurance)", 
                "NPS (80CCD)", 
                "Home Loan Interest", 
                "HRA Exemption"
            ]
            
            np.random.seed(len(CURRENT_USER_ID) * 2 + 1)
            simulated_values = [
                np.random.randint(20000, 150000), 
                np.random.randint(10000, 35000),  
                np.random.randint(5000, 50000),    
                np.random.randint(50000, 180000), 
                np.random.randint(30000, 100000), 
            ]

            tax_df = pd.DataFrame(dict(
                Section=tax_sections,
                Value=simulated_values
            ))
            
            max_limit = max(tax_df['Value'].max() + 5000, 200000)

            fig_tax = go.Figure()
            
            fig_tax.add_trace(go.Scatterpolar(
                r=tax_df['Value'].tolist() + [tax_df['Value'].tolist()[0]], # Close the loop
                theta=tax_df['Section'].tolist() + [tax_df['Section'].tolist()[0]],
                fill='toself',
                name='Savings Contribution',
                marker_color='rgba(138, 43, 226, 0.7)',
                line_color='rgba(138, 43, 226, 1)',
                hoverinfo='text',
                text=[f'{s}: {money(v)}' for s, v in zip(tax_sections, simulated_values)]
            ))

            fig_tax.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max_limit], 
                        tickvals=[0, 50000, 100000, 150000, 200000],
                        ticktext=['₹0K', '₹50K', '₹100K', '₹150K', '₹200K'],
                        gridcolor='rgba(100, 100, 100, 0.2)', # Darker grid lines
                    ),
                    angularaxis=dict(
                        linecolor='rgba(100, 100, 100, 0.2)',
                        gridcolor='rgba(100, 100, 100, 0.2)',
                        layer='below traces'
                    )
                ),
                showlegend=False,
                template="plotly_white",
                paper_bgcolor='#ffffff', 
                plot_bgcolor='#ffffff',
                height=380,
                title="Tax Savings Breakdown (Simulated)",
                margin=dict(l=0,r=0,t=40,b=0)
            )
            st.plotly_chart(fig_tax, use_container_width=True)

        # 🎤 Explainer for Graph 4
        _chart_explainer(
            "G4", chart_name_4,
            "Simulated tax-saving contributions across sections like 80C and NPS."
        )

        # --- Generative Visualization Section ---
        st.markdown("---")
        st.markdown("## ✨ Generative Infographic (AI Visualizer)")
        
        # ... (rest of the Generative Visualization code remains the same)
        
        # --- Telegram Report Button ---
        st.markdown("---")
        if st.button("Send KPI Snapshot to Telegram", key="send_telegram_report"):
            if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
                st.error(
                    "❌ Telegram API keys (`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`) not configured in environment variables."
                )
            else:
                summary_df = pd.DataFrame(
                    [
                        {"Metric": "Income", "Value": total_income},
                        {"Metric": "Expense", "Value": total_expense},
                        {"Metric": "Net", "Value": net},
                    ]
                )

                fig_report = go.Figure(
                    data=[go.Bar(
                        x=summary_df["Metric"],
                        y=summary_df["Value"],
                        marker_color=["#6a5acd", "#ef4444", "#22c55e"]
                    )]
                )
                fig_report.update_layout(
                    title=f"KPIs for {CURRENT_USER_ID}",
                    template="plotly_dark",
                    height=300
                )

                try:
                    img_bytes = fig_report.to_image(format="png")
                except ValueError:
                    st.error("❌ Plotly to Image failed. You may need to install `kaleido` (`pip install kaleido`).")
                    st.stop()

                caption = f"""
                *FINANCE REPORT for {CURRENT_USER_ID} ({date.today().isoformat()})*
                - Total Income: {money(total_income)}
                - Total Expense: {money(total_expense)}
                - Net Savings: {money(net)}
                - Health Score: {st.session_state['health_score_data']['score']}
                """

                # FIX: Added st.spinner around this slow network call
                with st.spinner("🚀 Sending report to Telegram..."):
                    success, msg = send_report_png(img_bytes, caption)
                
                if success:
                    st.success(msg)
                else:
                    st.error(msg)


# --- Stock Data Tab ---
with tab_stock:
    st.header("📈 Real-time Stock Data (AlphaVantage)")
    st.info(
        "This feature uses a simulated stock API key to fetch real-time stock quotes. "
        "Historical charts are generated from simulated data."
    )

    col_sym, col_button = st.columns([2, 1])

    with col_sym:
        symbol = st.text_input(
            "Enter Stock Symbol (e.g., TCS.BSE, RELIANCE.NSE)",
            value="ITC.BSE",
            key="stock_symbol_input"
        ).upper()

    with col_button:
        st.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
        if st.button("Fetch Quote & Charts", use_container_width=True, key="fetch_quote_charts_btn_2"):
            st.session_state["last_quote"] = fetch_stock_quote(symbol)
            st.session_state["daily_data"] = generate_simulated_daily_data(symbol)
            st.rerun()

    if "last_quote" in st.session_state and isinstance(st.session_state["last_quote"], dict):
        quote = st.session_state["last_quote"]
        daily_df = st.session_state.get("daily_data")

        st.markdown("---")
        st.subheader(f"Quote for {quote.get('symbol', symbol)}")

        m_p, m_c, m_v = st.columns(3)
        m_p.metric("Current Price (₹)", f"₹{quote.get('price', 'N/A')}")
        m_c.metric("Change", quote.get("change", "N/A"), delta_color="normal")
        m_v.metric("Volume", quote.get("volume", "N/A"))
        st.caption(f"Last updated: {quote.get('last_updated', 'N/A')}")

        st.markdown("---")
        st.subheader("Historical & Portfolio Visualizations")

        # --- NEW: Moving Average Crossover Chart ---
        if daily_df is not None and 'SMA_Short' in daily_df.columns:
            st.markdown("#### Moving Average Crossover (10-day vs 30-day SMA)")
            fig_sma = go.Figure()
            fig_sma.add_trace(go.Scatter(x=daily_df.index, y=daily_df['Close Price (₹)'], mode='lines', name='Close Price', line=dict(color='#6a5acd', width=1.5)))
            fig_sma.add_trace(go.Scatter(x=daily_df.index, y=daily_df['SMA_Short'], mode='lines', name='10-Day SMA', line=dict(color='#ef4444', width=2)))
            fig_sma.add_trace(go.Scatter(x=daily_df.index, y=daily_df['SMA_Long'], mode='lines', name='30-Day SMA', line=dict(color='#22c55e', width=2)))
            
            # Highlight Buy/Sell signals (Simplified)
            # Find crossover points (buy when short crosses above long, sell when short crosses below long)
            crossover = daily_df['SMA_Short'] > daily_df['SMA_Long']
            prev_crossover = crossover.shift(1)
            
            buy_signals = daily_df[(crossover == True) & (prev_crossover == False)]
            sell_signals = daily_df[(crossover == False) & (prev_crossover == True)]
            
            fig_sma.add_trace(go.Scatter(
                x=buy_signals.index,
                y=buy_signals['SMA_Short'],
                mode='markers',
                marker=dict(symbol='triangle-up', size=10, color='#22c55e'),
                name='Buy Signal'
            ))
            fig_sma.add_trace(go.Scatter(
                x=sell_signals.index,
                y=sell_signals['SMA_Short'],
                mode='markers',
                marker=dict(symbol='triangle-down', size=10, color='#ef4444'),
                name='Sell Signal'
            ))

            fig_sma.update_layout(title=f"SMA Crossover for {symbol}", template="plotly_dark", height=450)
            st.plotly_chart(fig_sma, use_container_width=True)
            st.caption("Simplified technical analysis: Buy signal (green up triangle) when the fast SMA (pink/red) crosses above the slow SMA (green).")

        st.markdown("---")

        chart1, chart2 = st.columns([2, 1])
        with chart1:
            if daily_df is not None:
                st.markdown("#### Line Chart: Last 60 Days Closing Price Trend")
                fig_line = px.line(
                    daily_df,
                    x=daily_df.index,
                    y="Close Price (₹)",
                    title=f"Price Trend for {symbol}",
                    labels={"Close Price (₹)": "Price (₹)", "Date": "Date"},
                )
                fig_line.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig_line, use_container_width=True)
            else:
                st.info("Historical data not available.")

        with chart2:
            st.markdown("#### Donut/Pie Chart: Sample Portfolio Allocation")
            portfolio_data = pd.DataFrame(
                {
                    "Asset": ["TCS", "Reliance", "HDFC Bank", "Cash"],
                    "Value (₹)": [150000, 120000, 90000, 40000],
                }
            )
            fig_donut = px.pie(
                portfolio_data,
                values="Value (₹)",
                names="Asset",
                title="Current Portfolio Distribution",
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.RdPu,
            )
            fig_donut.update_traces(textinfo="percent+label")
            fig_donut.update_layout(template="plotly_dark", height=400, showlegend=False)
            st.plotly_chart(fig_donut, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Bar Chart: Last 60 Days Daily Volume")
        if daily_df is not None:
            fig_bar = px.bar(
                daily_df,
                x=daily_df.index,
                y="Volume",
                title=f"Daily Volume for {symbol}",
                labels={"Volume": "Volume", "Date": "Date"},
                color_discrete_sequence=['#8a2be2']
            )
            fig_bar.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Enter a stock symbol and click 'Fetch Quote & Charts'.")

# --- Financial Planning Tab ---
with tab_plan:
    _ai_financial_plan_view(df)

# >>> NEW TAB RENDER - PERSONAL CA BLUEPRINT <<<
with tab_ca_plan:
    render_ca_plan_tab(df)

# --- City Affordability Tab ---
with tab_city:
    render_city_affordability_tab()

# === Tools tab ===
with tab_tools:
    st.header("🧰 Data Tools — Edit / Backup / Dedupe / Categories")

    # --- quick actions ---
    cA, cB, cC, cD, cE = st.columns(5)
    with cA:
        if st.button("💾 Export MiniDB (JSON)", use_container_width=True):
            try:
                data = json.dumps(DB.to_json(), ensure_ascii=False, indent=2).encode("utf-8")
                st.download_button("Download mini_db.json", data=data, file_name="mini_db.json", mime="application/json", use_container_width=True)
            except Exception as e:
                st.error(f"Export failed: {e}")
    with cB:
        up = st.file_uploader("Restore MiniDB (.json)", type=["json"])
        if up is not None:
            try:
                payload = json.loads(up.read().decode("utf-8"))
                newdb = MiniDB.from_json(payload)
                st.session_state["DB"]._orders = newdb._orders
                st.session_state["DB"]._tx = newdb._tx
                st.session_state["DB"]._order_seq = newdb._order_seq
                st.session_state["DB"]._tx_seq = newdb._tx_seq
                st.session_state["DB"].save()
                st.success("✅ DB restored from backup.")
                st.rerun()
            except Exception as e:
                st.error(f"Restore failed: {e}")
    with cC:
        if st.button("🧹 Remove Duplicates", use_container_width=True):
            removed = DB.delete_duplicates_keep_smallest_id(CURRENT_USER_ID)
            DB.save()
            st.success(f"Duplicates removed: {removed}")
            st.rerun()
    with cD:
        # --- NEW FEATURE: Delete All Transactions Button ---
        if st.button("💣 Delete All User Txns", use_container_width=True, help="WARNING: Deletes ALL transactions for the current user. Cannot be undone."):
            st.session_state["confirm_delete_all"] = True
            st.rerun() # Rerun to show confirmation step

        if st.session_state.get("confirm_delete_all", False):
            st.warning("⚠️ **ARE YOU SURE?** This action is permanent and deletes **ALL** your transactions.")
            if st.button(f"CONFIRM Delete All Txns for {CURRENT_USER_ID}", type="primary"):
                removed_count = DB.delete_all_txns(CURRENT_USER_ID)
                DB.save()
                st.session_state["confirm_delete_all"] = False
                st.success(f"Successfully deleted {removed_count} transactions for {CURRENT_USER_ID}.")
                st.rerun()
            if st.button("Cancel Delete"):
                st.session_state["confirm_delete_all"] = False
                st.rerun()
        # -----------------------------------------------------
    with cE:
        st.info("Tip: Use **E** to jump here, **B** to export DB, **F** to focus chat.")

    st.markdown("---")

    # --- Inline edit/delete table ---
    st.subheader("✏️ Edit or Delete Transactions")
    if df.empty:
        st.info("No transactions to edit.")
    else:
        # Ensure 'view' is defined and used if available, otherwise fall back to 'df'
        edit_df_source = locals().get('view')
        edit_df = edit_df_source.copy() if edit_df_source is not None and not edit_df_source.empty else df.copy()
        
        edit_df = edit_df.sort_values(["date"], ascending=False).reset_index(drop=True)
        if "id" not in edit_df.columns:
            st.warning("Editing requires saved DB rows (the sample data is not persisted).")
        else:
            show_cols = ["id","date","amount","category","description","type"]
            edit_df_show = edit_df[show_cols].copy()
            edited = st.data_editor(
                edit_df_show,
                key="data_editor_txn",
                use_container_width=True,
                num_rows="fixed",
                column_config={
                    "id": st.column_config.NumberColumn(disabled=True, help="Database ID"),
                    "date": st.column_config.DateColumn(),
                    "amount": st.column_config.NumberColumn(format="₹%d"),
                    "category": st.column_config.TextColumn(),
                    "description": st.column_config.TextColumn(),
                    "type": st.column_config.SelectboxColumn(options=["income","expense"])
                }
            )
            c1, c2 = st.columns([1,1])
            with c1:
                if st.button("💡 Apply Edited Rows", use_container_width=True):
                    changed = 0
                    for _, row in edited.iterrows():
                        try:
                            ok = DB.update_txn(
                                int(row["id"]),
                                date=row["date"],
                                amount=row["amount"],
                                category=str(row["category"]).strip() or "uncategorized",
                                description=str(row["description"]),
                                type=row["type"]
                            )
                            if ok: changed += 1
                        except Exception:
                            pass
                    DB.save()
                    st.success(f"Saved {changed} rows.")
                    st.rerun()
            with c2:
                tid = st.number_input("Delete by ID", min_value=1, step=1, value=1, key="del_tid")
                if st.button("🗑️ Delete", use_container_width=True):
                    if DB.delete_txn(int(tid)):
                        DB.save()
                        st.success(f"Deleted txn #{int(tid)}")
                        st.rerun()
                    else:
                        st.error("ID not found.")

    st.markdown("---")

    # --- Category Manager ---
    st.subheader("🗂️ Category Manager (Rename/Merge)")
    cats_all = sorted(set([t.category for t in DB._tx.values() if t.user_id == CURRENT_USER_ID]))
    if not cats_all:
        st.info("No categories yet.")
    else:
        col1, col2, col3 = st.columns([1.2,1.2,1])
        with col1:
            old_cat = st.selectbox("From (old)", options=cats_all)
        with col2:
            new_cat = st.text_input("To (new)", value=old_cat)
        with col3:
            if st.button("🔁 Rename/Merge", use_container_width=True):
                if not new_cat.strip():
                    st.error("New category cannot be empty.")
                else:
                    changed = DB.rename_or_merge_category(CURRENT_USER_ID, old_cat, new_cat.strip().lower())
                    DB.save()
                    st.success(f"Updated {changed} rows from '{old_cat}' → '{new_cat}'.")
                    st.rerun()

    st.markdown("---")

    # --- Duplicate inspector ---
    st.subheader("🔍 Duplicate Inspector")
    groups = DB.find_duplicates(CURRENT_USER_ID)
    if not groups:
        st.success("No duplicates found. 🎉")
    else:
        st.warning(f"Found {len(groups)} duplicate groups. Showing first few:")
        show = []
        for ids in groups[:5]:
            show.extend([asdict(DB._tx[i]) for i in ids if i in DB._tx])
        if show:
            dup_df = pd.DataFrame(show)
            if not dup_df.empty:
                dup_df["date"] = pd.to_datetime(dup_df["date"]).dt.date
            st.dataframe(dup_df, use_container_width=True)

# --- MAIN APPLICATION ENTRY POINT ---
if __name__ == "__main__":
    try:
        st.session_state["DB"].save()
    except Exception:
        pass 

    


