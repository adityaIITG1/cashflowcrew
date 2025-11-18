
# app.py — Personal Finance & Spending Analyzer (V17 - FULL, CLEANED SYNTAX)

from __future__ import annotations

import os
import base64
import joblib
import json
import requests
import time
import random
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import qrcode
import streamlit as st
import streamlit.components.v1 as components
import cv2
import mediapipe as mp

from io import BytesIO
from pathlib import Path
from datetime import date, datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Any
from PIL import Image, ImageDraw
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode


# === MOCK/EXTERNAL MODULE IMPORTS (Assuming they exist/are mocked) ===

# Import analytics helpers (MOCK)
# These are assumed to be defined in 'analytics.py' and are essential.
def compute_fin_health_score(df, budgets=None):
    if df.empty: return {"score": 50, "summary": "No data", "factors": {"net_flow": 0, "budget_adherence": 0, "savings_rate": 0, "no_spend_streak": 0, "longest_no_spend": 0}}
    net_flow = df[df["type"] == "income"]["amount"].sum() - df[df["type"] == "expense"]["amount"].sum()
    score = max(0, min(100, 50 + int(net_flow / 1000)))
    return {"score": score, "summary": "Simulated Health Score", "factors": {"net_flow": net_flow, "budget_adherence": 80, "savings_rate": 15, "no_spend_streak": 3, "longest_no_spend": 7}}
def no_spend_streak(df):
    if df.empty: return 0, 0
    daily_expense = df[df["type"] == "expense"].groupby(df["date"]).sum(numeric_only=True)["amount"]
    no_spend = (daily_expense == 0).reindex(pd.date_range(daily_expense.index.min(), daily_expense.index.max(), freq='D').date, fill_value=True)
    curr, longest = 0, 0
    for v in no_spend.values:
        curr = (curr + 1) if v else 0
        longest = max(longest, curr)
    return curr, longest
# Assuming these imports are available:
import pandas as pd
import numpy as np

def detect_trend_spikes(df, window="30D"):
    if df.empty: return "No spikes detected."
    
    expense_df = df[df["type"] == "expense"].copy()
    
    if expense_df.empty: 
        return "No expense spikes detected."

    # Step 1: Calculate the daily expense series (no re-indexing yet)
    # Ensure the 'date' column is datetime objects for grouping
    expense_df["date"] = pd.to_datetime(expense_df["date"])
    
    expense_series_raw = expense_df.groupby(expense_df["date"]).sum(numeric_only=True)["amount"]

    # Step 2: Reindex to cover the full date range, using the series defined in Step 1
    # 🔥 FIX: Use expense_series_raw.index.min() and expense_series_raw.index.max()
    expense_series = expense_series_raw.reindex(
        pd.date_range(
            expense_series_raw.index.min(), 
            expense_series_raw.index.max(), 
            freq='D'
        ), 
        fill_value=0
    )
    
    # Calculate rolling mean and standard deviation
    rolling_mean = expense_series.rolling(window=7).mean()
    std = expense_series.rolling(window=7).std()
    
    # Detect spikes (amount > mean + 2*std)
    spikes = expense_series[(expense_series > rolling_mean + 2 * std) & (expense_series > 0)]
    
    if spikes.empty: 
        return "No unusual spending spikes detected."
        
    return ", ".join([f"₹{v:,.0f} on {k.strftime('%Y-%m-%d')}" for k, v in spikes.head(3).items()])
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import time
from datetime import date
import numpy as np # Added for forecast_next_month consistency

def forecast_next_month(df):
    if df.empty:
        return pd.DataFrame({"category": ["Food", "Transport"], "forecasted_expense": [10000, 5000]})
        
    expense_df = df[df["type"] == "expense"].copy()
    
    if expense_df.empty:
        return pd.DataFrame({"category": ["Food", "Transport"], "forecasted_expense": [10000, 5000]})
        
    # 🔥 FIX: Convert the 'date' column explicitly to datetime objects
    # This prevents the AttributeError: Can only use .dt accessor with datetimelike values
    expense_df["date"] = pd.to_datetime(expense_df["date"])

    # Calculate number of unique months to get average monthly expense
    unique_months = expense_df["date"].dt.to_period("M").nunique()
    
    # Prevent division by zero if there's no temporal span
    if unique_months == 0:
        unique_months = 1 

    # Calculate the average monthly expense per category
    category_sum = expense_df.groupby("category")["amount"].sum()
    category_avg = category_sum / unique_months
    
    # Format and return top 5
    category_avg = category_avg.sort_values(ascending=False).head(5).reset_index(name="forecasted_expense")
    
    return category_avg.round(2)

def auto_allocate_budget(df, savings_target_pct=0.15):
    # Simple Mock: 50/30/20 rule inspired allocation
    income = df[df["type"] == "income"]["amount"].sum()
    essential = 0.5 * income
    lifestyle = 0.3 * income
    savings_goal = max(1000, 0.2 * income)
    return {"Essential": essential, "Lifestyle": lifestyle, "Savings Goal": savings_goal}

# Import OCR helpers (MOCK)
class Tesseract: 
    HAS_TESSERACT = False
HAS_TESSERACT = Tesseract.HAS_TESSERACT

# Import Telegram helpers (MOCK)
def send_report_png(img_bytes, caption):
    return True, "Report successfully simulated to Telegram."

# Import Weather helpers (MOCK)
def get_weather(city):
    return {"temp": 25, "desc": "Clear sky", "city": city, "icon": "☀️"}
def spend_mood_hint(data):
    if data and data.get("temp", 0) > 30 and "Clear sky" in data.get("desc", ""):
        return f"It's **{data['temp']}°C** in {data['city']}! Perfect weather for impulse online shopping from the couch. Stay indoors and save! 🛋️💸"
    return f"Weather in {data.get('city', 'N/A')}: {data.get('temp', 'N/A')}°C and {data.get('desc', 'N/A')}. Mind your mood-induced spending."

# Import Generative Viz helper (MOCK)
def suggest_infographic_spec(summary_stats, spikes_info):
    # This mock requires _static_fallback_viz to be defined
    return {"description": "Infographic analysis: Your top expense is Groceries. Focus on cutting down non-essential items.", "spec": _static_fallback_viz(summary_stats)['spec']}
def _static_fallback_viz(summary_stats):
    fig = go.Figure(data=[go.Bar(y=['Income', 'Expense'], x=[summary_stats.get('total_income', 0), -summary_stats.get('total_expense', 0)], orientation='h')])
    fig.update_layout(title_text='Net Flow (Fallback)', template="plotly_dark")
    return {"spec": fig.to_dict()}

# Import Custom UI helpers (MOCK)
def money(amount):
    # Ensure money is defined before display_health_score/budget_bot_minicard use it
    return f"₹{amount:,.0f}"

def display_health_score(data):
    st.markdown(f"**Financial Health Score:** <span style='font-size:32px; font-weight:900; color:#0ea5e9;'>{data['score']}/100</span>", unsafe_allow_html=True)
def display_badges(current_streak):
    if current_streak > 3: st.success(f"🏆 Saving Streak: {current_streak} days!")
    else: st.info(f"⏳ Current saving streak: {current_streak} days.")
def budget_bot_minicard(allocation):
    st.markdown("#### Budget Bot Allocation")
    st.info(f"Essential: {money(allocation['Essential'])}, Lifestyle: {money(allocation['Lifestyle'])}, Savings: {money(allocation['Savings Goal'])}")
    updated_budget = {}
    updated_budget['Essential'] = st.number_input('Essential Budget', value=allocation['Essential'], key="b_ess")
    updated_budget['Lifestyle'] = st.number_input('Lifestyle Budget', value=allocation['Lifestyle'], key="b_life")
    return updated_budget, st.button("Apply Budget", key="apply_budget_btn", use_container_width=True)
def glowing_ocr_uploader():
    st.markdown("#### OCR Receipt Upload")
    file = st.file_uploader("Upload Receipt Image", type=["jpg", "png"], key="ocr_upload")
    ocr_data = {}
    if file:
        st.info("Simulating OCR for a moment...")
        time.sleep(1)
        ocr_data = {"amount": 550.50, "date": date.today().isoformat(), "merchant": "Groceries Mart", "category": "groceries"}
        st.success(f"OCR Detected: ₹{ocr_data['amount']} on {ocr_data['date']} at {ocr_data['merchant']}")
    return file, ocr_data
def money(amount):
    return f"₹{amount:,.0f}"

# Import AI Helpers (MOCK - will include gemini_query logic below)
def build_smart_advice_bilingual(df):
    net = df[df["type"] == "income"]["amount"].sum() - df[df["type"] == "expense"]["amount"].sum()
    if net < 0:
        hi = "आप घाटे में हैं! तुरंत 2,000 रुपये बचाएं."
        en = "You are in deficit! Immediately save ₹2,000."
    else:
        hi = "आपके वित्त अच्छे हैं! ₹500 का निवेश करें."
        en = "Your finances are good! Invest ₹500."
    return hi, en
def speak_bilingual_js(advice_hi, advice_en, order):
    spoken = advice_hi if order.startswith("hi") else advice_en
    lang = "hi-IN" if order.startswith("hi") else "en-IN"
    html = f"""
    <script>
      try {{
        const u = new SpeechSynthesisUtterance("{spoken}");
        u.lang = "{lang}";
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(u);
      }} catch(e) {{ console.warn(e); }}
    </script>
    """
    components.html(html, height=0, scrolling=False)
def smart_machine_listener(advice_hi, advice_en, wake_word, order):
    pass # Already handled by the initial `voice_js` snippet
def chat_reply(user_q, history, context):
    return gemini_query(user_q, history, context)
def gemini_enabled():
    return HAS_GEMINI_SDK and GEMINI_API_KEY is not None

# Import Plotting Helpers (MOCK - will include logic below)
def build_bottom_figures(df, group_period):
    if df.empty: return {}
    agg = add_period(df, group_period).groupby(["period", "type"])["amount"].sum().unstack(fill_value=0)
    fig1 = px.bar(agg, x=agg.index, y=["income", "expense"], title="Income/Expense Flow", barmode="group", template="plotly_dark")
    fig2 = px.box(df, x="category", y="amount", title="Expense Distribution", color="type", template="plotly_dark")
    fig3 = px.scatter(df, x="date", y="amount", color="category", title="Transaction Scatter", template="plotly_dark")
    return {"fig1": fig1, "fig2": fig2, "fig3": fig3}
def summarize_dataframe(df):
    return {"num_transactions": len(df), "unique_categories": len(df["category"].unique())}
def explain_charts(figs, summary, llm):
    return {"fig1": "The flow chart shows stability of income against fluctuating expenses. Keep the blue bars high and the pink bars low. 📈", "fig2": "The box plots show that your biggest expense volatility is in 'groceries'—try to stick to a list! 🛒", "fig3": "The scatter plot highlights a few large, one-off expenses (outliers) that need a second look. 👀"}
def render_bottom_section(st, figs, explanations):
    gc1, gc2, gc3 = st.columns(3)
    cols = [gc1, gc2, gc3]
    for i, (key, fig) in enumerate(figs.items()):
        with cols[i]:
            st.plotly_chart(fig, use_container_width=True, key=f"bottom_chart_{i}")
            st.info(explanations.get(key, "AI explanation not available."))


# --- ALWAYS-ON VOICE LISTENER: "Smart Machine, ...." ---
voice_js = """
<script>
let recognizing = false;
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

if (SpeechRecognition) {
    const recognizer = new SpeechRecognition();
    recognizer.continuous = true;
    recognizer.interimResults = false;
    recognizer.lang = "en-IN";

    function startRecognizer() {
        if (!recognizing) {
            recognizer.start();
            recognizing = true;
        }
    }

    recognizer.onresult = (event) => {
        let text = event.results[event.results.length - 1][0].transcript.toLowerCase();
        console.log("Heard:", text);

        if (!text.includes("smart machine")) return;

        let cmd = null;

        if (text.includes("savings story") || text.includes("saving story")) {
            cmd = "savings_story";
        } else if (text.includes("this month") && text.includes("report")) {
            cmd = "month_report";
        } else if (text.includes("how much") && text.includes("spent today")) {
            cmd = "today_spend";
        } else if (text.includes("improve my financial score") || text.includes("improve my finance score")) {
            cmd = "improve_score";
        } else if (text.includes("warn me") && text.includes("overspend")) {
            cmd = "warn_overspend";
        }

        if (cmd) {
            // Update the URL query parameter ?voice_cmd=...
            const url = new URL(window.location.href);
            url.searchParams.set("voice_cmd", cmd);
            window.location.href = url.toString();
        }
    };

    recognizer.onend = () => {
        // Restart automatically
        setTimeout(startRecognizer, 500);
    };

    startRecognizer();
} else {
    console.warn("SpeechRecognition not supported in this browser.");
}
</script>
"""

components.html(voice_js, height=0)


# --- Safe date helper for OCR/defaults ---
def _safe_to_date(x) -> date:
    """Return a real python date; fallback to today if x is empty/invalid."""
    try:
        dt = pd.to_datetime(x, errors="coerce")
        if pd.isna(dt):
            return date.today()
        return dt.date()
    except Exception:
        return date.today()


# Import the actual Gemini SDK client (requires: pip install google-genai)
try:
    from google import genai
    HAS_GEMINI_SDK = True
except ImportError:
    HAS_GEMINI_SDK = False

# === NEW: Import OpenAI SDK (requires: pip install openai) ===
try:
    from openai import OpenAI
    HAS_OPENAI_SDK = True
except ImportError:
    HAS_OPENAI_SDK = False


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
    user_id: str  # NEW: User identifier
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
            date=dt.isoformat(),
            amount=float(amount),
            category=category or "uncategorized",
            description=description or "",
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
            rows = [r for r in rows if r.date >= start.isoformat()]
        if end:
            rows = [r for r in rows if r.date <= end.isoformat()]
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
                # Ensure 'user_id' exists for legacy data
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


# ============================== Face Detector Transformer ==============================

class FaceDetectorTransformer(VideoTransformerBase):
    """
    Detects a face using OpenCV Haar Cascade and returns a simple 'face found' status.
    This replaces the complex hand pose detection.
    """
    def __init__(self):
        # NOTE: Using a public domain Haar cascade path if available
        # On many Streamlit deployment environments, cv2.data.haarcascades is empty,
        # so this might fail unless the file is packaged or downloaded.
        try:
            haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(haar_path)
        except Exception:
            self.face_cascade = None
        self.face_detected_count = 0
        self.current_face_vector = ""

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        self.face_detected_count = 0
        self.current_face_vector = ""

        if self.face_cascade is not None and self.face_cascade.empty() is False:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
            self.face_detected_count = len(faces)

            if self.face_detected_count > 0:
                (x, y, w, h) = faces[0]
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    "FACE DETECTED",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                # Update the biometric vector (using size and position as the "code")
                self.current_face_vector = f"{x},{y},{w},{h},{datetime.now().hour}"

        status_text = f"Face(s) Found: {self.face_detected_count}"
        color = (0, 255, 0) if self.face_detected_count > 0 else (0, 0, 255)
        cv2.putText(img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ============================== API Keys and Configuration ==============================

# Read keys from environment variables (replacing custom file read)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyCW-sKyNFFkWz4hiysMWJR6Fyj342-GhkY")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")

# --- NEW: OPENAI API KEY ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "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")
# ---------------------------

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

HAS_QR = False
try:
    import qrcode as _qr  # noqa: F401
    HAS_QR = True
except Exception:
    HAS_QR = False


# ============================== Utilities / FX / Sound ==============================

def generate_placeholder_image(path: Path, size: int = 300, color: str = "pink", text: str = "Placeholder") -> None:
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
.coin-rain {{
  position: fixed; inset: 0; pointer-events: none; z-index: 9999;
}}
.coin-rain span {{
  position:absolute; top:-50px; font-size:22px; filter:drop-shadow(0 6px 8px rgba(0,0,0,.35));
  animation: rain 2.2s linear infinite;
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

# --- OpenAI Query (Fallback) ---
def openai_query(prompt: str, history: list[tuple[str, str]], context: str) -> str:
    """Handles the intelligent response using the OpenAI API."""
    if not OPENAI_API_KEY:
        return "❌ **OPENAI KEY MISSING:** Please set the `OPENAI_API_KEY` environment variable."

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        system_instruction = (
            "You are a versatile, professional AI financial advisor named PRAKRITI AI, part of the Cashflow Crew. "
            "Your persona is based on the following: " + context +
            "For finance queries, be concise (3-5 sentences) and proactive in suggesting ideas. "
            "Always include emojis in your responses to make them more engaging."
        )

        messages = [{"role": "system", "content": system_instruction}]
        
        # Simple history handling (last 2 turns)
        for speaker, text in history[-2:]:
            role = "user" if speaker == "You" else "assistant"
            messages.append({"role": role, "content": text})

        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", # A common, fast model
            messages=messages,
            max_tokens=200,
            temperature=0.5
        )
        return f"🧠 *OpenAI Fallback AI:* {response.choices[0].message.content}"

    except Exception as e:
        return f"❌ **OPENAI API Error:** Failed to generate response. Error: {e}"

# --- Gemini Query (Primary) ---
def gemini_query(prompt: str, history: list[tuple[str, str]], context: str) -> str:
    """Handles the intelligent response using the Gemini API, with OpenAI fallback."""

    # 1. Check for Gemini
    if not GEMINI_API_KEY:
        if HAS_OPENAI_SDK and OPENAI_API_KEY:
            st.warning("Gemini Key missing. Falling back to OpenAI.")
            return openai_query(prompt, history, context)
        return "❌ **GEMINI KEY MISSING:** Please set the `GEMINI_API_KEY` environment variable."

    if not HAS_GEMINI_SDK:
        if HAS_OPENAI_SDK and OPENAI_API_KEY:
            st.warning("Gemini SDK missing. Falling back to OpenAI.")
            return openai_query(prompt, history, context)
        return "⚠️ **GEMINI SDK Missing:** Cannot connect to the intelligent chatbot. Please run `pip install google-genai`."

    # 2. Proceed with Gemini API call
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

        response = client.models.generate_content(model="gemini-2.5-flash", contents=contents)
        return f"🧠 *Gemini Smart AI:* {response.text}"

    except Exception as e:
        # Fallback to OpenAI on Gemini API error
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
    return df.set_index("Date").sort_index()

# --- KB/TFIDF Helpers (Minified) ---

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

# --- Data/Plot Helpers (Minified) ---

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

    # 1. Monthly Summary
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

    # 2. Saving Goal Action
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

# ---------- Initial Setup ----------
if not UPI_QR_IMG.exists():
    if not _generate_default_upi_qr(UPI_PAYMENT_STRING, UPI_QR_IMG):
        generate_placeholder_image(UPI_QR_IMG, text="UPI QR (Error)")

generate_placeholder_image(PROFILE_IMG, size=70, color="#25D366", text="Money Icon")
ensure_kb_exists()

PROFILE64 = _img64(PROFILE_IMG)

# ============================== App Initialization ==============================

st.set_page_config(page_title="Cash Flow Crew — Personal Finance Analyzer", page_icon="💎", layout="wide")

if "DB" not in st.session_state:
    st.session_state["DB"] = MiniDB.load()

DB: MiniDB = st.session_state["DB"]

if "paid_orders_applied" not in st.session_state: st.session_state["paid_orders_applied"] = set()
if "thinking" not in st.session_state: st.session_state["thinking"] = False
if "longest_streak_ever" not in st.session_state: st.session_state["longest_streak_ever"] = 0
if "sound_muted" not in st.session_state: st.session_state["sound_muted"] = False
if "user_budgets" not in st.session_state: st.session_state["user_budgets"] = {}
if "weather_city" not in st.session_state: st.session_state["weather_city"] = "Prayagraj"
if "weather_data" not in st.session_state: st.session_state["weather_data"] = get_weather(st.session_state["weather_city"])
if "global_budgets" not in st.session_state: st.session_state["global_budgets"] = {}
if "active_page" not in st.session_state: st.session_state["active_page"] = "📊 Main Dashboard"
if "coin_rain_show" not in st.session_state: st.session_state["coin_rain_show"] = True
if "promise_text" not in st.session_state: st.session_state["promise_text"] = "I promise that I will save 100 rupees per day"


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


# --- ML Session State Initialization ---
if "ml_login_step" not in st.session_state: st.session_state["ml_login_step"] = 1
if "ml_face_code_live" not in st.session_state: st.session_state["ml_face_code_live"] = ""
if "auth_ok" not in st.session_state: st.session_state["auth_ok"] = False
if "auth_user" not in st.session_state: st.session_state["auth_user"] = None

def _login_view() -> None:
    st.markdown(
        """
    <style>
    [data-testid="stAppViewContainer"] > .main {background-color: #0f1117;}
    .login-card-container { display: grid; grid-template-columns: 1fr 1fr; gap: 40px; }
    .login-card-inner {
        padding: 30px; border-radius: 16px;
        background: linear-gradient(145deg, rgba(255,105,180,0.1), rgba(255,192,203,0.05));
        border: 2px solid #ff79b0;
        box-shadow: 0 0 25px rgba(255, 105, 180, 0.9), 0 0 50px rgba(255, 192, 203, 0.5);
        transition: all 0.3s;
    }
    .login-card-inner:hover { box-shadow: 0 0 35px #ff57a6, 0 0 60px #ffc0cb; }
    .stSelectbox div { background-color: #1a1a1a !important; color: white !important; }
    .stSelectbox button { color: #ff79b0 !important; }
    .navbar {
        position: sticky; top: 0; z-index: 1000; padding: 12px 18px; margin: 0 0 18px 0;
        border-radius: 14px;
        background: radial-gradient(120% 120% at 0% 0%, #ffd9ea 0%, #ffcfe3 30%, rgba(255,255,255,0.08) 70%);
        box-shadow: 0 12px 30px rgba(255, 105, 180, 0.25), inset 0 0 60px rgba(255,255,255,0.25);
        border: 1px solid rgba(255,255,255,0.35);
        display: flex; justify-content: space-between; align-items: center;
    }
    .nav-title { font-weight: 800; font-size: 24px; color:#2b0d1e; letter-spacing: .5px; }
    .nav-sub { color:#5b1a3a; font-size:13px; margin-top:-2px; }
    .coin-wrap { position: relative; height: 60px; margin: 6px 0 0 0; overflow: hidden; }
    .coin { position:absolute; top:-50px; font-size:24px; filter: drop-shadow(0 6px 8px rgba(0,0,0,.35)); animation: drop 4s linear infinite; }
    .coin:nth-child(2){left:15%; animation-delay:.6s}
    .coin:nth-child(3){left:30%; animation-delay:.1s}
    .coin:nth-child(4){left:45%; animation-delay:.9s}
    .coin:nth-child(5){left:60%; animation-delay:1.8s}
    .coin:nth-child(6){left:75%; animation-delay:.3s}
    .coin:nth-child(7){left:90%; animation-delay:.2s}
    @keyframes drop { 0%{ transform: translateY(-60px) rotate(0deg); opacity:0 } 10%{ opacity:1 } 100%{ transform: translateY(120px) rotate(360deg); opacity:0 } }
    </style>
    <div class="navbar">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <div>
          <div class="nav-title">🔐 Finance Analyzer — BIOMETRIC FACE LOGIN</div>
          <div class="nav-sub">Face Detection for Simplified Biometric Authentication</div>
        </div>
        <div class="coin-wrap">
          <span class="coin">🪙</span><span class="coin">💰</span><span class="coin">🪙</span>
          <span class="coin">💰</span><span class="coin">🪙</span><span class="coin">💰</span><span class="coin">🪙</span>
        </div>
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    all_users = sorted(list(_get_all_users(DB)))

    st.subheader("Live Face Recognition")
    st.warning("Grant camera access. Look directly at the camera to allow face detection and capture.")

    ctx = webrtc_streamer(
        key="ml_webcam_input",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        video_processor_factory=FaceDetectorTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

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

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.caption(f"Current Biometric Code (Face Position): {current_face_vector or 'N/A'}")


if not st.session_state["auth_ok"]:
    _login_view()
    st.stop()

CURRENT_USER_ID = st.session_state["auth_user"]

# ---------- Post-Login Setup ----------
VOICE_HINTS = [
    'Smart Machine, show me my savings story',
    "Smart Machine, show me this month’s report",
    "Smart Machine, how much did I spend today?",
    "Smart Machine, improve my financial score",
    "Smart Machine, warn me if I overspend",
]

with st.sidebar.expander("🎙 Smart Machine – Voice Commands", expanded=True):
    st.markdown("You can say:")
    for h in VOICE_HINTS:
        st.markdown(f"- `{h}`")

# ======================= LEFT SIDEBAR NAVIGATION =======================

page = st.sidebar.radio(
    "Navigate",
    [
        "📊 Main Dashboard",
        "📜 Transactions",
        "📽 Animated Data Story",
    ],
    index=[
        "📊 Main Dashboard",
        "📜 Transactions",
        "📽 Animated Data Story",
    ].index(st.session_state["active_page"]),
)

if page != st.session_state["active_page"]:
    st.session_state["active_page"] = page

# ============================== Main Body of App ==============================

# ---------- CSS (Fixed and Extended) ----------
st.markdown(
    f"""
<style>
html, body, [data-testid="stAppViewContainer"] {{
  background: #0f1117; color: #eaeef6;
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
}}
.navbar {{
    position: sticky; top: 0; z-index: 1000; padding: 12px 18px; margin: 0 0 18px 0;
    border-radius: 14px;
    background: radial-gradient(120% 120% at 0% 0%, #ffd9ea 0%, #ffcfe3 30%, rgba(255,255,255,0.08) 70%);
    box-shadow: 0 12px 30px rgba(255, 105, 180, 0.25), inset 0 0 60px rgba(255,255,255,0.25);
    border: 1px solid rgba(255,255,255,0.35);
    display: flex; justify-content: space-between; align-items: center;
}}
.nav-title-wrap {{ display: flex; align-items: center; gap: 10px; }}
.cashflow-girl {{ font-size: 30px; animation: float-money 2s ease-in-out infinite; position: relative; }}
@keyframes float-money {{ 0% {{ transform: translateY(0px) rotate(0deg); }} 25% {{ transform: translateY(-5px) rotate(5deg); }} 50% {{ transform: translateY(0px) rotate(0deg); }} 75% {{ transform: translateY(-5px) rotate(-5deg); }} 100% {{ transform: translateY(0px) rotate(0deg); }} }}
.nav-title {{ font-weight: 800; font-size: 24px; color:#2b0d1e; letter-spacing: .5px; }}
.nav-sub {{ color:#5b1a3a; font-size:13px; margin-top:-2px; }}
.coin-wrap {{ position: relative; height: 60px; margin: 6px 0 0 0; overflow: hidden; }}
.coin {{ position:absolute; top:-50px; font-size:24px; filter: drop-shadow(0 6px 8px rgba(0,0,0,.35)); animation: drop 4s linear infinite; }}
.coin:nth-child(2){{left:15%; animation-delay:.6s}}
.coin:nth-child(3){{left:30%; animation-delay:.1s}}
.coin:nth-child(4){{left:45%; animation-delay:.9s}}
.coin:nth-child(5){{left:60%; animation-delay:1.8s}}
.coin:nth-child(6){{left:75%; animation-delay:.3s}}
.coin:nth-child(7){{left:90%; animation-delay:.2s}}
@keyframes drop {{ 0%{{ transform: translateY(-60px) rotate(0deg); opacity:0 }} 10%{{ opacity:1 }} 100%{{ transform: translateY(120px) rotate(360deg); opacity:0 }} }}
.card {{border-radius:16px; background:linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02)); padding:16px; box-shadow: 0 12px 30px rgba(0,0,0,0.35); border: 1px solid rgba(255,255,255,0.12);}}
.metric {{font-size:18px; font-weight:700}}
.bot {{background:#111827; color:#e6eef8; padding:10px 12px; border-radius:10px; border:1px solid rgba(255,255,255,.08)}}
.streak-card{{
  border-radius:16px; padding:16px; margin-top:10px;
  background:linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
  border:1px solid rgba(255,255,255,.12); box-shadow:0 12px 30px rgba(0,0,0,.35);
}}
.piggy-wrap{{ position:relative; height:84px; display:flex; align-items:center; gap:16px }}
.piggy{{ font-size:58px; filter: drop-shadow(0 6px 8px rgba(0,0,0,.35)); }}
.piggy.dim{{ opacity:.55; filter: grayscale(0.6) }}
.coin-fall{{ position:absolute; left:62px; top:-12px; font-size:22px; animation: fall 1.8s linear infinite; }}
.coin-fall:nth-child(2){{ left:84px; animation-delay:.4s }}
.coin-fall:nth-child(3){{ left:46px; animation-delay:.9s }}
@keyframes fall {{ 0%{{ transform: translateY(-30px) rotate(0deg); opacity:0 }} 15%{{ opacity:1 }} 100%{{ transform: translateY(85px) rotate(360deg); opacity:0 }} }}
.streak-metric{{ font-weight:800; font-size:26px }}
.badge-ok{{ background:#0ea5e9; color:white; padding:4px 10px; border-radius:999px; font-size:12px }}
.profile-wrap{{display:flex;align-items:center;justify-content:flex-end}}
.profile-pic{{ width:70px;height:70px;border-radius:50%;object-fit:cover; box-shadow:0 6px 20px rgba(0,0,0,.35); border:2px solid #25D366; }}
.upi-qr-wrap {{
  position: relative; border-radius: 12px; padding: 10px;
  background: rgba(255, 105, 180, 0.1);
  border: 1px solid rgba(255, 105, 180, 0.5);
  box-shadow: 0 0 15px rgba(255, 105, 180, 0.7), inset 0 0 10px rgba(255, 105, 180, 0.5);
  animation: qr-glow 2s infinite alternate, qr-flicker 1.5s step-end infinite;
}}
@keyframes qr-glow {{
  0% {{ box-shadow: 0 0 10px rgba(255, 105, 180, 0.5), inset 0 0 8px rgba(255, 105, 180, 0.3); transform: scale(1); }}
  50% {{ transform: scale(1.01); }}
  100% {{ box-shadow: 0 0 20px rgba(255, 105, 180, 0.9), inset 0 0 12px rgba(255, 105, 180, 0.7); transform: scale(1); }}
}}
@keyframes qr-flicker {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.9; }} }}
.promise{{ font-weight:900; font-size:20px; letter-spacing:.3px; color:#ffe1f0; text-align:center; margin:8px 0 2px 0; animation: glow 3s ease-in-out infinite, jump 3s ease-in-out infinite; }}
@keyframes glow{{ 0%{{ text-shadow:0 0 6px #ff7ab8, 0 0 16px #ffb3d6 }} 50%{{ text-shadow:0 0 12px #ff57a6, 0 0 26px #ffc2e1 }} 100%{{ text-shadow:0 0 6px #ff7ab8, 0 0 16px #ffb3d6 }} }}
@keyframes jump{{ 0%{{ transform:translateY(0) }} 15%{{ transform:translateY(-8px) }} 30%{{ transform:translateY(0) }} 45%{{ transform:translateY(-5px) }} 60%,100%{{ transform:translateY(0) }} }}
.coin-rain span {{ animation: rain 2.2s linear infinite; }}
.coin-rain {{ animation: none; }}
@keyframes rain{{0%{{transform:translateY(-60px) rotate(0deg);opacity:0}} 15%{{opacity:1}} 100%{{transform:translateY(120vh) rotate(360deg);opacity:0}}}}
@keyframes revolve {{ 0% {{ transform: rotate(0deg) scale(1); }} 50% {{ transform: rotate(180deg) scale(1.05); }} 100% {{ transform: rotate(360deg) scale(1); }} }}
.revolving-brain {{ font-size: 32px; display: inline-block; animation: revolve 3s linear infinite, qr-glow 2s infinite alternate; color: #ffb3d6; margin-left: 10px; }}
@keyframes pulsing_arrow {{ 0% {{ transform: scale(1) translateX(0px); opacity: 1; }} 50% {{ transform: scale(1.1) translateX(10px); opacity: 0.8; }} 100% {{ transform: scale(1) translateX(0px); opacity: 1; }} }}
.callout-box-vfa {{ background: #ff57a6; color: white; padding: 8px 12px; border-radius: 8px; font-weight: 600; margin-top: 15px; display: flex; align-items: center; gap: 10px; animation: qr-glow 1.5s infinite alternate; }}
.animated-arrow-vfa {{ font-size: 24px; animation: pulsing_arrow 1.5s infinite; display: inline-block; }}
</style>
""",
    unsafe_allow_html=True,
)

if st.session_state["coin_rain_show"]:
    show_coin_rain(RAIN_DURATION_SEC)

# ---------- Navbar ----------
colA, colB = st.columns([4, 0.6])
with colA:
    st.markdown(
        f"""
    <div class="navbar">
      <div class="nav-title-wrap">
        <span class="cashflow-girl">👩‍💰💸</span>
        <div>
          <div class="nav-title">💎 Personal Finance Dashboard <br> <span style="font-size:18px;">Cashflow Crew ({CURRENT_USER_ID})</span></div>
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
    st.success("🎉 **Now integrated with GEMINI!** Access intelligent financial guidance via the Smart Chatbot.")
else:
    st.error("⚠️ **GEMINI SDK Missing:** Chatbot intelligence is disabled. Please run `pip install google-genai`.")

st.markdown(f"<div class='promise'>{st.session_state['promise_text']}</div>", unsafe_allow_html=True)
new_p = st.text_input("Change promise line", st.session_state["promise_text"])
if new_p != st.session_state["promise_text"]:
    st.session_state["promise_text"] = new_p
    st.rerun()

# --------------- PAGE ROUTING ---------------

if page == "📽 Animated Data Story":
    st.header("📽 Animated Savings Story (Coming Soon)")
    st.info(
        "This page is reserved for your ipyvizzu + Streamlit animated data story "
        "of Salary, Expenses and Savings. Voice command 'Smart Machine, show me my savings story' "
        "will bring you here."
    )
    # Mocking a call to a missing data_story.py render function
    # if 'data_story' module existed: render_animated_story()
    st.stop()
elif page == "📜 Transactions":
    # For now, treat transactions page as focused dashboard to avoid re-duplicating code,
    # but the logic below covers filtering and visualization anyway.
    st.info("The Transactions page views the same data as the Main Dashboard, focusing on the table view below.")


# --- Start of Tabbed Structure ---
tab_dashboard, tab_stock = st.tabs(["💰 Personal Dashboard", "📈 Real-time Stock Data (AlphaVantage)"])

with tab_dashboard:
    # --- Control Panel ---
    tb1, tb2, tb3, tb4, tb5, tb6, tb7 = st.columns([1.6, 1.4, 1.4, 1.8, 1.2, 1, 1.4])
    with tb1:
        data_source = st.radio("Data source", ["Use saved data", "Generate sample"], index=0, horizontal=True)
    with tb2:
        plot_type = st.selectbox(
            "Plot type",
            [
                "Line plot (trend)", "Bar plot (aggregate)", "Count plot (category counts)",
                "Scatter plot", "Distribution (KDE)", "Histogram", "Donut Chart", "Heatmap",
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
            for k in ("auth_ok", "auth_user", "chat_history", "coin_rain_show", "longest_streak_ever", "promise_text", "last_quote", "daily_data", "DB", "ml_login_step", "ml_face_code_live", "user_budgets", "weather_city", "weather_data", "global_budgets", "health_score_data",):
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
    <div style="background-color: #1a1a1a; padding: 10px; border-radius: 8px; margin-bottom: 15px; border-left: 5px solid #0ea5e9;">
    <span style="font-weight: bold; color: #0ea5e9;">🌤️ Spending Mood Hint:</span> {hint_text}
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ---------- Load data ----------
    raw_df = None
    if data_source == "Generate sample":
        raw_df = generate_sample(6)
        st.info("Showing 6 months of sample data (not saved to your account).")
    else:
        db_txns = DB.list_txns(CURRENT_USER_ID)
        if not db_txns:
            st.info(f"No saved transactions found for **{CURRENT_USER_ID}**. Showing 1 month of sample data.")
            raw_df = generate_sample(1)
        else:
            raw_df = pd.DataFrame([asdict(t) for t in db_txns])
            raw_df["date"] = pd.to_datetime(raw_df["date"]).dt.date

    if raw_df is None:
        st.stop()

    try:
        df = normalize(raw_df)
    except Exception as e:
        st.error(f"Error normalizing data: {e}. Please check column names.")
        st.stop()

    # --- Filters ---
    f1, f2, f3 = st.columns([1.3, 1.6, 1.1])
    if df.empty:
        view = df.copy()
        tmp = add_period(view, group_period)
        start, end = date.today(), date.today() # Placeholder dates
    else:
        min_d = df["date"].min()
        max_d = df["date"].max()
        with f1:
            start = st.date_input("Start date", min_value=min_d, max_value=max_d, value=min_d, key="start_1")
            end = st.date_input("End date", min_value=min_d, max_value=max_d, value=max_d, key="end_1")
        with f2:
            cats = sorted(df["category"].unique().tolist())
            sel_cats = st.multiselect("Categories", options=cats, default=cats)
        with f3:
            types = sorted(df["type"].unique().tolist())
            sel_types = st.multiselect("Types", options=types, default=types)

        mask = (df["date"] >= start) & (df["date"] <= end)
        view = df[mask & df["category"].isin(sel_cats) & df["type"].isin(sel_types)].copy()
        tmp = add_period(view, group_period)

    # --- Health Score + Budgets ---
    st.markdown("---")
    top_left_col, top_mid_col, top_right_col = st.columns([1.2, 1.5, 2])

    net = view[view["type"] == "income"]["amount"].sum() - view[view["type"] == "expense"]["amount"].sum()

    with top_left_col:
        current_budgets = st.session_state["global_budgets"].get(CURRENT_USER_ID, {})
        budget_allocation = auto_allocate_budget(df, savings_target_pct=0.15)
        updated_budget, apply_save = budget_bot_minicard(budget_allocation)
        if apply_save:
            updated_budget_lower = {k.lower(): v for k, v in updated_budget.items()}
            st.session_state["global_budgets"][CURRENT_USER_ID] = updated_budget_lower
            st.success("Budgets applied to your profile! Health Score updated.")
            st.rerun()
        current_budgets = st.session_state["global_budgets"].get(CURRENT_USER_ID, {})
        curr_ns, longest_ns = no_spend_streak(df)
        display_badges(curr_ns)

    with top_mid_col:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        total_income = view[view["type"] == "income"]["amount"].sum() if not view.empty else 0
        total_expense = view[view["type"] == "expense"]["amount"].sum() if not view.empty else 0
        net = total_income - total_expense
        avg_per = tmp.groupby("period")["amount"].sum().mean() if not tmp.empty else 0
        m1.metric("Total Income", money(total_income))
        m2.metric("Total Expense", money(total_expense))
        m3.metric("Net", money(net))
        m4.metric(f"Avg {group_period}", money(avg_per))
        st.markdown("</div>", unsafe_allow_html=True)

    with top_right_col:
        health_score_data = compute_fin_health_score(df, budgets=current_budgets)
        display_health_score(health_score_data)
        st.session_state["health_score_data"] = health_score_data

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

    dn = daily_net_frame(df)
    longest_ns = health_score_data["factors"]["longest_no_spend"]

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
            if today_hit
            else ""
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
            color_discrete_map={"Hit": "#0ea5e9", "Miss": "#ef4444"},
            title=f"Net saving (last {lookback} days)",
            labels={"day": "Day", "net_saving": "₹"},
        )
        fig_streak.update_layout(height=260, showlegend=True, legend_title="", template="plotly_dark")
        st.plotly_chart(fig_streak, use_container_width=True, config={"displayModeBar": False}, key="streak_chart_1")
    else:
        st.info("No transactions in the current date range to compute a streak.")
    st.markdown("</div>", unsafe_allow_html=True)


    # ---------- Voice Command Helpers (placeholders) ----------

    def get_today_spent() -> float:
        return view[view["date"] == date.today()]["amount"].sum()
    def get_month_report_text() -> str:
        current_month_data = df[pd.to_datetime(df["date"]).dt.to_period("M") == pd.Period(date.today().strftime("%Y-%m"), freq="M")]
        inc = current_month_data[current_month_data["type"] == "income"]["amount"].sum()
        exp = current_month_data[current_month_data["type"] == "expense"]["amount"].sum()
        net_s = inc - exp
        return f"This month you earned ₹{inc:,.0f}, spent ₹{exp:,.0f} and saved ₹{net_s:,.0f}."
    def get_financial_score_text() -> str:
        score_data = compute_fin_health_score(df, budgets=st.session_state["global_budgets"].get(CURRENT_USER_ID, {}))
        return (
            f"Your current financial score is {score_data['score']}/100. "
            f"Net flow: {money(score_data['factors']['net_flow'])}. "
            "Main issues: irregular savings, high spend on food & online shopping."
        )
    def get_overspend_warning_text() -> str:
        return (
            "Based on your last 7 days, you are likely to cross your budget by ₹3,000. "
            "Cut 20% from online shopping and food delivery to stay safe."
        )


    # ---------- READ VOICE COMMAND FROM URL & REACT ----------

    params = st.query_params
    voice_cmd = params.get("voice_cmd", None)

    if voice_cmd:
        st.query_params.clear()
        st.toast(f"🎙 Smart Machine command received: {voice_cmd}", icon="🎧")

        if voice_cmd == "savings_story":
            st.session_state["active_page"] = "📽 Animated Data Story"
            components.html("""
                <script>
                const u = new SpeechSynthesisUtterance("Showing your animated savings story.");
                u.lang = "en-IN";
                window.speechSynthesis.cancel();
                window.speechSynthesis.speak(u);
                </script>
            """, height=0)
            st.experimental_rerun()

        elif voice_cmd == "month_report":
            report = get_month_report_text()
            st.sidebar.success("📅 This Month’s Report\n\n" + report)
            components.html(f"""
                <script>
                const u = new SpeechSynthesisUtterance(`{report}`);
                u.lang = "en-IN";
                window.speechSynthesis.cancel();
                window.speechSynthesis.speak(u);
                </script>
            """, height=0)

        elif voice_cmd == "today_spend":
            amt = get_today_spent()
            msg = f"Aaj aapne total lagbhag ₹{amt:,.0f} kharch kiye hain."
            st.sidebar.info("💸 Today's Spend\n\n" + msg)
            components.html(f"""
                <script>
                const u = new SpeechSynthesisUtterance("{msg}");
                u.lang = "hi-IN";
                window.speechSynthesis.cancel();
                window.speechSynthesis.speak(u);
                </script>
            """, height=0)

        elif voice_cmd == "improve_score":
            msg = (
                "Aapka financial score improve karne ke liye 3 kaam karein: "
                "1) Har income ka kam se kam 20 pratishat direct savings jar me bhejein, "
                "2) UPI kharch ka weekly review karein, "
                "3) Subscription aur impulse shopping ko turant 30 pratishat kam karein."
            )
            extra = get_financial_score_text()
            st.sidebar.warning("📈 Improve Your Financial Score\n\n" + extra + "\n\n" + msg)
            components.html(f"""
                <script>
                const u = new SpeechSynthesisUtterance("{msg}");
                u.lang = "hi-IN";
                window.speechSynthesis.cancel();
                window.speechSynthesis.speak(u);
                </script>
            """, height=0)

        elif voice_cmd == "warn_overspend":
            msg = get_overspend_warning_text()
            st.sidebar.error("🚨 Overspend Warning\n\n" + msg)
            components.html(f"""
                <script>
                const u = new SpeechSynthesisUtterance("{msg}");
                u.lang = "hi-IN";
                window.speechSynthesis.cancel();
                window.speechSynthesis.speak(u);
                </script>
            """, height=0)


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
                    uploaded_df.rename(columns={'merchant': 'category'}, errors='ignore', inplace=True)
                    
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
        
        default_amount = float(ocr_data.get("amount", 100.0)) if ocr_data else 100.0
        default_date_val = ocr_data.get("date") if (ocr_data and ocr_data.get("date")) else date.today()
        default_desc   = (ocr_data.get("merchant") or "Manual Entry") if ocr_data else "Manual Entry"
        default_cat    = (ocr_data.get("category") or "uncategorized") if ocr_data else "uncategorized"
        safe_default_date = _safe_to_date(default_date_val)
                
        with st.form("manual_txn_form", clear_on_submit=True):
            txn_date = st.date_input("Date", value=safe_default_date)
            txn_type = st.radio("Type", ["expense", "income"], horizontal=True, index=0)
            txn_amt = st.number_input("Amount (₹)", min_value=1.0, value=float(default_amount), step=1.0)

            all_cats = sorted(df["category"].unique().tolist())
            if default_cat not in all_cats:
                all_cats.insert(0, default_cat)
                
            try:
                default_index = all_cats.index(default_cat)
            except ValueError:
                default_index = 0
                
            txn_cat = st.selectbox(
                "Category", options=all_cats, index=default_index
            )
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
                    play_paid_sound(CURRENT_USER_ID, float(txn_amt))
                    green_tick(f"Income of {money(txn_amt)} recorded successfully!")
                else:
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
        forecast_df = forecast_next_month(df)
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

        if not HAS_GEMINI_SDK and not HAS_OPENAI_SDK:
            st.error("AI Tip Panel disabled: GEMINI/OpenAI SDK missing or keys not set.")
        else:
            tip_key = f"ai_tip_{CURRENT_USER_ID}_{date.today().isoformat()}"
            if tip_key not in st.session_state:
                spikes = detect_trend_spikes(df, window="30D")
                summary = {
                    "total_income": float(view[view["type"] == "income"]["amount"].sum()),
                    "total_expense": float(view[view["type"] == "expense"]["amount"].sum()),
                    "net_savings": float(net),
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
                    ans = gemini_query(
                        context_prompt,
                        st.session_state.get("chat_history", [])[-2:],
                        f"You are a financial coach. The user is {CURRENT_USER_ID} and their net is {money(net)}.",
                    )
                    tip_text = ans.replace("🧠 *Gemini Smart AI:*", "").replace("🧠 *OpenAI Fallback AI:*", "").strip()
                    st.session_state[tip_key] = tip_text.replace("*", "").strip()
                except Exception as e:
                    st.session_state[tip_key] = f"❌ **AI Tip Generation Error:** {e}"

            st.markdown(
                f"""
            <div class='bot' style='border-left: 5px solid #ff79b0;'>
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
                    + money(net)
                    + "."
                )
                ans = None
                try:
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


    # --- Smart Machine (Voice Coach) — Bilingual (Continuous Listener Enabled) ---
    st.markdown("---")
    st.subheader("🎙️ Smart Machine (Voice Coach) — Bilingual")
    st.caption("Wake word 'smart machine': Hindi + English tips. Or press Speak now.")

    # Build advice strings from the current filtered DataFrame `view`
    advice_hi, advice_en = build_smart_advice_bilingual(view)

    c1, c2 = st.columns(2)
    with c1:
        order = st.selectbox("Order", ["hi-en", "en-hi"], index=0)
    with c2:
        enable_listener = st.toggle("Enable wake word", value=False, key="smart_machine_listener_toggle")

    if st.button("🔊 Speak now", key="speak_bilingual_now"):
        speak_bilingual_js(advice_hi, advice_en, order=order)

    # ➡️ New Logic: Trigger the listener directly if the toggle is ON
    if enable_listener:
        # NOTE: smart_machine_listener is mostly handled by the initial global JS script now.
        # This code block remains as a placeholder/signal for a continuous action.
        st.info("Listening... say **smart machine**. Allow mic permission; keep this tab focused.")

    # ---------- Main charts & table ----------
    st.markdown("---")
    st.subheader("Main Charts")

    left, right = st.columns([3, 1])

    with left:
        st.markdown("#### Interactive Chart View")
        if tmp.shape[0] == 0:
            st.info("No data in current selection — adjust filters.")
        else:
            fig = go.Figure()
            if plot_type.startswith("Donut Chart"):
                donut_data = view.groupby("category")["amount"].sum().reset_index()
                fig = px.pie(
                    donut_data,
                    values="amount",
                    names="category",
                    title="Spending by Category",
                    hole=0.5,
                    color_discrete_sequence=px.colors.sequential.RdPu,
                )
            elif plot_type.startswith("Line plot (trend)") or plot_type.startswith("Bar plot (aggregate)"):
                agg = tmp.groupby(["period", "type"])["amount"].sum().reset_index()
                if plot_type.startswith("Line"):
                    fig = px.area(agg, x="period", y="amount", color="type", line_group="type", title=f"Trend by {group_period}")
                else:
                    if bar_mode == "By Category":
                        agg_cat = view.groupby(["category", "type"])["amount"].sum().reset_index()
                        fig = px.bar(agg_cat, x="category", y="amount", color="type", title="Total Amount by Category", barmode="group")
                    else:
                        fig = px.bar(agg, x="period", y="amount", color="type", title=f"Total Amount by {group_period}", barmode="stack")
            elif plot_type.startswith("Scatter plot"):
                fig = px.scatter(view, x="date", y="amount", color="category", title="Individual Transactions", hover_data=["description", "type"])
            elif plot_type.startswith("Distribution") or plot_type.startswith("Histogram"):
                fig = px.histogram(
                    view,
                    x="amount",
                    color="type",
                    marginal="rug" if plot_type.startswith("Distribution") else None,
                    title=f"Distribution of {numeric_col}",
                    histfunc="sum",
                    histnorm="percent",
                )
            else: # Fallback to Area/Line for other types (e.g., Heatmap needs specific data prep)
                 agg = tmp.groupby(["period", "type"])["amount"].sum().reset_index()
                 fig = px.area(agg, x="period", y="amount", color="type", line_group="type", title=f"Trend by {group_period} (Default View)")

            fig.update_layout(height=520, template="plotly_dark", legend_title="")
            st.plotly_chart(fig, use_container_width=True, key="main_chart_1")

        st.subheader(f"Transactions (filtered for {CURRENT_USER_ID})")
        st.dataframe(view.sort_values("date", ascending=False).reset_index(drop=True), height=300, use_container_width=True)

    with right:
        st.markdown("### Transaction Form")
        st.markdown(
            f"<a href='{FORM_URL}' target='_blank' style='text-decoration:none'>"
            f"<button style='background:#ff4da6;color:#fff;border:none;padding:10px 14px;border-radius:8px;font-weight:700;cursor:pointer; width: 100%;'>"
            f"Google form Money Collection</button></a>",
            unsafe_allow_html=True,
        )

    # --- Deep Dive Analytics ---
    st.markdown("---")
    st.markdown("## 📊 Deep Dive Analytics (AI Explained)")

    gc1, gc2, gc3 = st.columns(3)

    with gc1:
        st.markdown("#### 1. Income Source Breakdown")
        income_data = df[df["type"] == "income"].groupby("category")["amount"].sum().reset_index()
        fig1 = px.pie(income_data, values="amount", names="category", hole=0.6, title="Income Sources")
        fig1.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown('<div class="callout-box-vfa"><span class="animated-arrow-vfa">➡️</span> **AI Explanation:** This shows your **income diversification**.', unsafe_allow_html=True)

    with gc2:
        st.markdown("#### 2. Monthly Expense vs. Income")
        monthly_flow_df = add_period(df, group_period="Monthly")
        monthly_net = monthly_flow_df.groupby(["period", "type"])["amount"].sum().unstack(fill_value=0)
        if "income" not in monthly_net.columns: monthly_net["income"] = 0
        if "expense" not in monthly_net.columns: monthly_net["expense"] = 0
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=monthly_net.index, y=monthly_net["income"], name="Income", marker_color="#0ea5e9"))
        fig2.add_trace(go.Bar(x=monthly_net.index, y=-monthly_net["expense"], name="Expense", marker_color="#ff79b0"))
        fig2.update_layout(
            barmode="relative", template="plotly_dark", title="Monthly Flow (Income up, Expense down)",
            height=400, yaxis_title="Net Flow (Income/Expense)",
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('<div class="callout-box-vfa"><span class="animated-arrow-vfa">➡️</span> **AI Explanation:** Visualizes monthly **cash flow** stability. You want the blue (Income) bar to stay large compared to the pink (Expense) bar.', unsafe_allow_html=True)

    with gc3:
        st.markdown("#### 3. Transaction Count Distribution")
        count_data = view.groupby("type").size().reset_index(name="count")
        fig3 = px.bar(count_data, x="type", y="count", title="Transaction Count (Income vs. Expense)")
        fig3.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown('<div class="callout-box-vfa"><span class="animated-arrow-vfa">➡️</span> **AI Explanation:** Shows if you make many small purchases or few large ones. Useful for **behavioral tracking**.', unsafe_allow_html=True)

    # --- Generative Visualization Section ---
    st.markdown("---")
    st.markdown("## ✨ Generative Infographic (AI Visualizer)")

    summary_stats = {
        "total_income": total_income,
        "total_expense": total_expense,
        "net_savings": net,
        "top_expenses": view[view["type"] == "expense"]
        .groupby("category")["amount"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
        .to_dict(),
    }

    gen_viz_data = suggest_infographic_spec(summary_stats, detect_trend_spikes(df, window="30D"))

    if "error" in gen_viz_data:
        st.error(f"❌ Gemini Viz Error: {gen_viz_data['error']}. Showing fallback.")

    st.info(f"Insight: {gen_viz_data.get('description', 'Visualizing key financial metrics.')}")

    try:
        spec_content = gen_viz_data["spec"]
        if isinstance(spec_content, dict):
            fig_spec = go.Figure(spec_content)
            fig_spec.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_spec, use_container_width=True, key="gen_viz_chart")
        else:
            st.warning("Generated visualization specification is not a valid Plotly JSON object. Showing fallback.")
            st.plotly_chart(_static_fallback_viz(summary_stats)["spec"], use_container_width=True)
    except Exception as e:
        st.error(f"Failed to render generated chart: {e}. Showing fallback.")
        st.plotly_chart(_static_fallback_viz(summary_stats)["spec"], use_container_width=True)

    # --- Telegram Report Button ---
    st.markdown("---")
    if st.button("Send KPI Snapshot to Telegram", key="send_telegram_report"):
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            st.error("❌ Telegram API keys (`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`) not configured in environment variables.")
        else:
            summary_df = pd.DataFrame(
                [
                    {"Metric": "Income", "Value": total_income},
                    {"Metric": "Expense", "Value": total_expense},
                    {"Metric": "Net", "Value": net},
                ]
            )
            fig_report = go.Figure(
                data=[go.Bar(x=summary_df["Metric"], y=summary_df["Value"], marker_color=["#0ea5e9", "#ff79b0", "#22c55e"])]
            )
            fig_report.update_layout(title=f"KPIs for {CURRENT_USER_ID}", template="plotly_dark", height=300)
            try:
                img_bytes = fig_report.to_image(format="png")
            except ValueError:
                st.error("❌ Plotly to Image failed. You may need to install `kaleido` (`pip install kaleido`).")
                # Removed st.stop() to allow app to continue
                img_bytes = None
            
            if img_bytes:
                caption = f"""
                *FINANCE REPORT for {CURRENT_USER_ID} ({date.today().isoformat()})*
                - Total Income: {money(total_income)}
                - Total Expense: {money(total_expense)}
                - Net Savings: {money(net)}
                - Health Score: {st.session_state['health_score_data']['score']}
                """
                success, msg = send_report_png(img_bytes, caption)
                if success:
                    st.success(msg)
                else:
                    st.error(msg)

    # ---- Bottom 3 charts that mirror group_period ----
    figs = build_bottom_figures(view, group_period=group_period)
    summary = summarize_dataframe(view)

    def _gemini_llm(prompt: str) -> str:
        ctx = f"The user is {CURRENT_USER_ID}. Data window matches current filters and group_period={group_period}."
        return gemini_query(prompt, st.session_state.get("chat_history", [])[-2:], ctx)

    explanations = explain_charts(figs, summary, llm=_gemini_llm)

    render_bottom_section(st, figs, explanations)
            
with tab_stock:
    st.header("📈 Real-time Stock Data (AlphaVantage)")
    st.info("This feature uses a simulated stock API key to fetch real-time stock quotes. Historical charts are generated from simulated data.")

    col_sym, col_button = st.columns([2, 1])

    with col_sym:
        symbol = st.text_input("Enter Stock Symbol (e.g., TCS.BSE, RELIANCE.NSE)", value="ITC.BSE", key="stock_symbol_input").upper()

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
                {"Asset": ["TCS", "Reliance", "HDFC Bank", "Cash"], "Value (₹)": [150000, 120000, 90000, 40000]}
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
            )
            fig_bar.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Enter a stock symbol and click 'Fetch Quote & Charts'.")

# --- MAIN APPLICATION ENTRY POINT ---
if __name__ == "__main__":
    # Save database changes automatically on each run
    st.session_state["DB"].save()