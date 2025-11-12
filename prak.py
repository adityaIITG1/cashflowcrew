
# app.py — Personal Finance & Spending Analyzer (V15 - Fully Complete, All Logic Restored)

from __future__ import annotations

import os
import base64
import joblib
import json
import requests
import time
import random
from io import BytesIO
from pathlib import Path
from datetime import date, datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence

# Import the actual Gemini SDK client (requires: pip install google-genai)
try:
    from google import genai
    HAS_GEMINI_SDK = True
except ImportError:
    HAS_GEMINI_SDK = False

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go # Added for custom chart control
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import qrcode


# ============================== Mini In-Memory DB (Multi-User) ==============================

@dataclass
class Order:
    id: int
    amount: float
    currency: str
    status: str 					# 'pending' | 'paid' | 'failed'
    note: str = ""
    created_at: str = datetime.utcnow().isoformat(timespec="seconds")

@dataclass
class Transaction:
    id: int
    user_id: str # NEW: User identifier
    date: str 					# 'YYYY-MM-DD'
    amount: float
    category: str
    description: str
    type: str 					# 'income' | 'expense'
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
        o = Order(id=self._order_seq, amount=float(amount), currency=currency, status="pending", note=note)
        self._orders[o.id] = o
        return o

    def list_orders(self, status: Optional[str] = None) -> List[Order]:
        vals = list(self._orders.values())
        return [o for o in vals if (status is None or o.status == status)]

    def _filter_txns(self, user_id: str) -> List[Transaction]:
        return [t for t in self._tx.values() if t.user_id == user_id]

    def add_txn(self, *, user_id: str, dt: date, amount: float, category: str, description: str, typ: str) -> Transaction:
        if typ not in ("income", "expense"):
            raise ValueError("typ must be 'income' or 'expense'")
        self._tx_seq += 1
        t = Transaction(
            id=self._tx_seq, user_id=user_id, date=dt.isoformat(), amount=float(amount),
            category=category or "uncategorized", description=description or "", type=typ,
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
            try: obj = Order(**o); db._orders[obj.id] = obj
            except TypeError: pass
        for t in data.get("transactions", []):
            try: 
                # Ensure 'user_id' exists, defaulting to 'prakriti11' for old data
                if 'user_id' not in t: t['user_id'] = 'prakriti11' 
                obj = Transaction(**t); db._tx[obj.id] = obj
            except TypeError: pass 
        return db

    def save(self) -> None:
        self.DB_PATH.write_text(json.dumps(self.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls) -> "MiniDB":
        if not cls.DB_PATH.exists():
            return cls()
        try:
             return cls.from_json(json.loads(cls.DB_PATH.read_text(encoding="utf-8")))
        except Exception:
             # If loading fails (e.g., corrupted file), start fresh
             return cls()

# ============================== API Keys and Configuration ==============================

def _read_key_from_file(filename: str) -> str:
    """Reads a key directly from a text file, bypassing secrets.toml"""
    key_file = Path(filename)
    if key_file.exists():
        return key_file.read_text(encoding="utf-8").strip()
    return "KEY_FILE_NOT_FOUND"

# Read Gemini key using the custom function
GEMINI_API_KEY = _read_key_from_file("gemini_key.txt") 
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

# --- UPI Details (Your Real QR Data) ---
UPI_ID = 'jaiswalprakriti26@okaxis'
UPI_PAYMENT_STRING = f'upi://pay?pa={UPI_ID}&pn=PRAKRITI&cu=INR'

# --- Personalized Information (Updated Chatbot Context) ---
TEAM_INFO = {
    "Team Name": "Cashflow Crew",
    "Team Leader": "Prakriti Jaiswal",
    "Leader Expertise": "B.Com student at Allahabad University, expert in commerce.",
    "Frontend": "Ujjwal Singh",
    "Guidance": "Akash Pandey Sir (Technosavvys)",
    "Contact": "9170397988",
    "Email": "jaiswalprakriti26@gmail.com",
    "Donate UPI": UPI_ID
}


HAS_QR = False
try:
    import qrcode # noqa: F401
    HAS_QR = True
except Exception:
    HAS_QR = False

# ============================== Utilities / FX / Sound ==============================

def generate_placeholder_image(path: Path, size: int = 300, color: str = "pink", text: str = "Placeholder") -> None:
    """Generates a simple PIL image and saves it if the path doesn't exist."""
    if path.exists(): return
    try:
        img = Image.new("RGB", (size, size), color=color)
        d = ImageDraw.Draw(img)
        d.text((size // 4, size // 2), text, fill=(0, 0, 0))
        img.save(path)
    except Exception:
        pass

def _img64(path: Path | None) -> str:
    """Return base64 string for an image file, or empty string if not available."""
    try:
        if not path or not path.exists(): return ""
        with open(path, "rb") as fh: return base64.b64encode(fh.read()).decode("utf-8")
    except Exception: return ""

def _pick_qr_path() -> Path | None:
    """Prefer user-updated PNG, else JPG, else None."""
    if UPI_QR_IMG.exists(): return UPI_QR_IMG
    if UPI_QR_IMG_JPG.exists(): return UPI_QR_IMG_JPG
    return None

def _generate_default_upi_qr(upi_string: str, path: Path):
    """Generates the QR image for the user's UPI ID."""
    if not HAS_QR: return False
    try:
        qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=10, border=2)
        qr.add_data(upi_string)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        img.save(path)
        return True
    except Exception as e: return False
    
def _save_uploaded_qr(file) -> str:
    """Save uploaded QR as PNG (more portable). This overrides the default QR."""
    try:
        img = Image.open(file).convert("RGB")
        img.save(UPI_QR_IMG) 
        return "QR updated. If not visible, press 'Rerun' or refresh."
    except Exception as e: return f"Failed to save QR: {e}"

def money(v: float) -> str:
    return f"₹{v:,.0f}" if abs(v) >= 1000 and v.is_integer() else f"₹{v:,.2f}"

# tiny fallback WAV beep
_FALLBACK_WAV_B64 = (
    "UklGRiQAAABXQVZFZm10IBAAAAABAAEAESsAACJWAAACABYAAAACABYAAABkYXRhAAAAAA"
    "AAAAAAgP8AgP8AgP8AgP8AgP8AgP8AgP8AgP8AgP8AgP8AgP8AgP8AgP8AgP8AgP8AgP8A"
)

def play_paid_sound(name: str, amount: float) -> None:
    """Play cash sound and speak Hindi line in browser."""
    audio_src = SOUND_EFFECT_URL

    if st.session_state.get('sound_muted', False):
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
    coin_spans = "".join([
        f"<span style='left:{random.randint(5, 95)}%; animation-delay:{random.uniform(0, RAIN_DURATION_SEC/2):.2f}s;'>🪙</span>" 
        for _ in range(20)
    ])
    st.markdown(
        f"""
<style>
.coin-rain {{
  position: fixed; inset: 0; pointer-events: none; z-index: 9999;
  animation: fade-out {seconds+1}s forwards;
}}
.coin-rain span {{
  position:absolute; top:-50px; font-size:22px; filter:drop-shadow(0 6px 8px rgba(0,0,0,.35));
  animation: rain 2.2s linear infinite;
}}
@keyframes rain{{0%{{transform:translateY(-60px) rotate(0deg);opacity:0}}
15%{{opacity:1}}100%{{transform:translateY(120vh) rotate(360deg);opacity:0}}}}
@keyframes fade-out{{0%{{opacity:1}}95%{{opacity:1}}100%{{opacity:0; visibility: hidden;}}}}
</style>
<div class="coin-rain">
    {coin_spans}
</div>
        """,
        unsafe_allow_html=True,
    )

def green_tick(msg: str) -> None:
    """Displays a large, noticeable green tick message."""
    st.markdown(f"""<div style="padding: 10px; border-radius: 8px; background-color: rgba(34, 197, 94, 0.2); color: #22c55e; margin-top: 15px;">
    <span style="font-size: 24px;">✅</span><span style="margin-left: 10px; font-weight: bold;">{msg}</span>
    </div>""", unsafe_allow_html=True)

# --- Chatbot Helpers ---

def gemini_query(prompt: str, history: list[tuple[str, str]], context: str) -> str:
    """Handles the intelligent response using the Gemini API."""
    
    if GEMINI_API_KEY == "KEY_FILE_NOT_FOUND":
          return "❌ **GEMINI KEY MISSING:** Please create 'gemini_key.txt' in your root folder and paste your key inside."

    if not HAS_GEMINI_SDK:
        return "⚠️ **GEMINI SDK Missing:** Cannot connect to the intelligent chatbot. Please run `pip install google-genai`."
          
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
          
        system_instruction = (
            "You are a versatile, professional AI financial advisor named **Cash Crew AI**. "
            "Your persona is based on the following: " + context +
            "You must be able to answer finance questions, but also handle casual conversation, greetings, and nonsense questions gracefully. "
            "For finance queries, be concise (3-5 sentences) and proactive in suggesting ideas. "
            "For casual queries, respond like a friendly assistant. "
            "Always include emojis in your responses to make them more engaging, using a professional yet enthusiastic tone."
        )
          
        final_prompt = system_instruction + "\n\n" + prompt

        contents = [{"role": "user", "parts": [{"text": final_prompt}]}]

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=contents
        )
          
        return f"🧠 *Cash Crew AI:* {response.text}"

    except Exception as e:
        return f"❌ **GEMINI API Error:** Failed to generate response. Check your API key and network connection. Error: {e}"

# AlphaVantage API Utility (Simulated) 
def fetch_stock_quote(symbol: str) -> dict | str:
    symbol_upper = symbol.upper()
    np.random.seed(len(symbol_upper) + datetime.now().day)
    if symbol_upper == "TCS.BSE": base_price = 4000
    elif symbol_upper == "RELIANCE.NSE": base_price = 2800
    elif "ITC" in symbol_upper: base_price = 420
    else: base_price = 450 + len(symbol_upper) * 10
    change_pct = np.random.uniform(-1.5, 1.5)
    volume_base = 500000 + len(symbol_upper) * 100000
    return {
        "symbol": symbol_upper,
        "price": f"{base_price:,.2f}",
        "change": f"{change_pct:+.2f}%",
        "volume": f"{int(volume_base/1000)}K",
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

# Simulation for Daily Stock Data (for charts) 
@st.cache_data
def generate_simulated_daily_data(symbol: str, days: int = 60) -> pd.DataFrame:
    symbol_upper = symbol.upper()
    if "TCS" in symbol_upper: base_price = 4000
    elif "RELIANCE" in symbol_upper: base_price = 2800
    elif "ITC" in symbol_upper: base_price = 420
    else: base_price = 450 + len(symbol_upper) * 10
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days, freq='D')
    np.random.seed(len(symbol_upper)) 
    prices = [base_price]
    for _ in range(1, days):
        change = np.random.normal(0, 15) * (1 + np.sin(_ / 20)) 
        new_price = prices[-1] * (1 + change / 1000)
        prices.append(new_price)
    volumes = np.random.randint(100000, 3000000, size=days)
    df = pd.DataFrame({
        'Date': dates,
        'Close Price (₹)': [round(p, 2) for p in prices],
        'Volume': volumes
    })
    return df.set_index('Date').sort_index()

# --- KB/TFIDF Helpers (Minified, for local knowledge lookup) ---

def ensure_kb_exists(default_kb: list[str] | None = None) -> None:
    default_kb = default_kb or ["help - Type questions about expenses, income, trends (e.g., 'total expense', 'top categories')", "overview - Show project overview and advantages", "trend groceries - Show spending trend for groceries", "plot - Explain the current plot and data", "streak - Show current and longest saving streak", "invest advice - Ask for general saving and investment advice"]
    if not KB_FILE.exists():
        try: KB_FILE.write_text("\n".join(default_kb), encoding="utf-8")
        except Exception: pass

def kb_texts_from_file() -> list[str]:
    try:
        if not KB_FILE.exists(): return []
        return [line.strip() for line in KB_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception: return []

def build_vectorizer(texts: list[str]):
    vect = TfidfVectorizer(strip_accents="ascii", lowercase=True, stop_words="english", token_pattern=r"\b\w+\b")
    try:
        if not texts: return None, None
        kb_mat = vect.fit_transform(texts)
        joblib.dump(vect, KB_VECT)
        return vect, kb_mat
    except Exception: return None, None

def load_vectorizer():
    try:
        if KB_VECT.exists(): return joblib.load(KB_VECT)
    except Exception: pass
    return None

def tfidf_answer(query: str, vect, kb_texts: list[str], kb_mat, threshold: float = 0.1) -> str | None:
    try:
        q = vect.transform([query])
        sims = cosine_similarity(q, kb_mat)[0]
        best = int(np.argmax(sims))
        if sims[best] >= threshold: return kb_texts[best]
    except Exception: pass
    return None

# --- Data/Plot Helpers (Minified) ---

def read_file(file):
    if isinstance(file, (str, Path)):
        if str(file).endswith(".csv"): return pd.read_csv(file)
    return pd.read_excel(file) # Handles .xlsx

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df is None: return pd.DataFrame()
    df = df.copy()
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    if "category" not in df.columns: df["category"] = "uncategorized"
    if "description" not in df.columns: df["description"] = ""
    if "type" not in df.columns: df["type"] = "expense"
    date_cols = [c for c in df.columns if "date" in c.lower()]
    if date_cols: df["date"] = pd.to_datetime(df[date_cols[0]], errors="coerce").dt.date
    else: df["date"] = pd.Timestamp.today().date()
    return df

def generate_sample(months: int = 6) -> pd.DataFrame:
    rng = pd.date_range(end=pd.Timestamp.today(), periods=months * 30)
    cats = ["groceries","rent","salary","investment","subscriptions","dining"]
    rows=[]
    for d in rng:
        for _ in range(np.random.poisson(1)):
            cat = np.random.choice(cats, p=[0.2,0.1,0.15,0.15,0.2,0.2])
            t="income" if cat in ("salary","investment") else "expense"
            amt = abs(round(np.random.normal(1200 if t=="income" else 50, 35),2))
            rows.append({"date": d.date(), "amount": amt, "category": cat, "description": f"{cat}", "type": t})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

def add_period(df: pd.DataFrame, group_period: str) -> pd.DataFrame:
    t = df.copy()
    t["date"] = pd.to_datetime(t["date"])
    if group_period == "Monthly": t["period"] = t["date"].dt.to_period("M").astype(str)
    elif group_period == "Weekly": t["period"] = t["date"].dt.strftime("%G-") + t["date"].dt.isocalendar().week.astype(str).str.zfill(2)
    else: t["period"] = t["date"].dt.date.astype(str)
    return t

def daily_net_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.shape[0] == 0: return pd.DataFrame(columns=["day", "income", "expense", "net_saving"])
    tmp = df.copy()
    tmp["day"] = pd.to_datetime(tmp["date"]).dt.date
    g = tmp.groupby(["day", "type"])["amount"].sum().unstack(fill_value=0)
    if "income" not in g: g["income"] = 0.0
    if "expense" not in g: g["expense"] = 0.0
    g["net_saving"] = g["income"] - g["expense"]
    if not g.empty:
        min_date = g.index.min()
        max_date = g.index.max()
        full = pd.date_range(min_date, max_date, freq="D").date
        g = g.reindex(full, fill_value=0.0); g.index.name = "day"
    return g.reset_index()

def compute_streak(series_bool: pd.Series) -> tuple[int, int]:
    if series_bool.empty: return 0, 0
    s = series_bool.copy()
    s = s.reindex(sorted(s.index))
    longest = run = 0
    for v in s.values:
        run = run + 1 if v else 0
        longest = max(longest, run)
    curr = 0
    for v in reversed(s.values):
        if v: curr += 1
        else: break
    return int(curr), int(longest)

def explain_plot_and_data(user_q: str, view: pd.DataFrame, tmp: pd.DataFrame, plot_type: str, group_period: str) -> str:
    if view is None or view.shape[0] == 0: return "There is no data in the current selection. Adjust date range and filters to include transactions before asking about the plot."
    lines = []
    n = int(view.shape[0])
    total_income = float(view[view["type"] == "income"]["amount"].sum())
    total_expense = float(view[view["type"] == "expense"]["amount"].sum())
    net = total_income - total_expense
    lines.append(f"Current selection contains *{n} transactions. Total income **{money(total_income)}**, total expense **{money(total_expense)}**, net **{money(net)}**.*")
    try:
        top_exp = (view[view["type"] == "expense"].groupby("category")["amount"].sum().sort_values(ascending=False).head(3))
        if not top_exp.empty:
            items = ", ".join([f"{k} ({money(v)})" for k, v in top_exp.items()])
            lines.append(f"Top expense categories: *{items}*.")
    except Exception: pass
    if "line" in plot_type.lower() or "trend" in plot_type.lower(): lines.append(f"This is a *trend (line/area) plot* grouped by {group_period}.")
    elif "bar" in plot_type.lower(): lines.append(f"This is a *bar plot* over the {group_period.lower()}.")
    elif "scatter" in plot_type.lower(): lines.append("This *scatter plot* shows individual transactions — useful to spot outliers.")
    elif "distribution" in plot_type.lower() or "hist" in plot_type.lower(): lines.append("This shows the *distribution of amounts*.")
    try:
        per = tmp.groupby(["period", "type"])["amount"].sum().unstack(fill_value=0)
        per["net"] = per.get("income", 0) - per.get("expense", 0)
        if per.shape[0] >= 2:
            last = float(per["net"].iloc[-1])
            prev = float(per["net"].iloc[-2])
            diff = last - prev
            pct = (diff / prev * 100) if prev != 0 else float("nan")
            trend = "increasing" if diff > 0 else "decreasing" if diff < 0 else "flat"
            lines.append(f"Net change from previous {group_period.lower()}: *{money(diff)}* ({pct:.1f}%). Recent trend: *{trend}*.")
    except Exception: pass
    lines.append("Tip: Use the Group period and date filters to zoom.")
    return "\n".join(lines)

def project_overview_and_advantages() -> str:
    return (
        "Project overview:\n"
        "This app is an interactive *Personal Finance Dashboard* that visualizes expenses and income, computes saving streaks, and provides quick actionable insights.\n\n"
        "- *Interactive visualizations* help you spot trends and top spending categories quickly. 📊\n"
        "- *Smart chatbot (powered by Gemini) and KB* allow generative financial advice and semantic lookups without exposing data externally. 🤖\n"
        "- Built-in *UPI/QR* and form workflow for easy logging. 📲\n"
        "- *Lightweight* and runs locally — your data stays with you. 🔒\n"
    )

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


# ---------- Initial Setup (QR generation added) ----------
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

if "paid_orders_applied" not in st.session_state:
    st.session_state["paid_orders_applied"] = set()

if "thinking" not in st.session_state:
    st.session_state["thinking"] = False
    
if "longest_streak_ever" not in st.session_state:
    # Load longest streak from disk if available, otherwise 0
    try:
        with open(Path("streak_store.json"), "r") as f:
            data = json.load(f)
            st.session_state["longest_streak_ever"] = int(data.get("longest_streak", 0))
    except Exception:
        st.session_state["longest_streak_ever"] = 0

if "sound_muted" not in st.session_state:
    st.session_state["sound_muted"] = False

# ============================== Login & User Management (FIXED) ==============================

# Simulated User database (Primary users must use password)
VALID_USERS = {
    "prakriti11": "ujjwal11",
    "ujjwal11": "prakriti11",
}

def _get_all_users(db: MiniDB) -> set:
    """Gets all known users from VALID_USERS plus any user ID found in the database."""
    db_users = {t.user_id for t in db._tx.values()}
    return set(VALID_USERS.keys()) | db_users

def _login_view() -> None:
    # Use the original aesthetic CSS structure
    st.markdown("""
    <style>
    /* Ensure the background is dark for the glow to show */
    [data-testid="stAppViewContainer"] > .main {background-color: #0f1117;}
    .login-card-container {
        padding: 30px; border-radius: 16px; 
        background: linear-gradient(145deg, rgba(255,105,180,0.1), rgba(255,192,203,0.05));
        border: 2px solid #ff79b0; /* Light Pink Border */
        box-shadow: 0 0 25px rgba(255, 105, 180, 0.9), 0 0 50px rgba(255, 192, 203, 0.5);
        transition: all 0.3s;
    }
    .login-card-container:hover {
        box-shadow: 0 0 35px #ff57a6, 0 0 60px #ffc0cb;
    }
    </style>
    <div class="navbar">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <div>
          <div class="nav-title">🔐 Finance Analyzer — Login/Register</div>
          <div class="nav-sub">New or Existing User: Select or Create Account</div>
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
    
    # Combined Login/Sign Up Form
    with st.form("auth_form", clear_on_submit=False):
        st.markdown('<div class="login-card-container">', unsafe_allow_html=True)
        st.subheader("Sign In or Create Account")

        # Use a combination of select/input for better UX
        user_input_mode = st.radio("Mode", ["Select Existing User", "Type New Username"], index=0, horizontal=True, key="auth_mode")
        
        user_to_auth = None
        
        if user_input_mode == "Select Existing User":
            user_options = [u for u in all_users]
            # Use 'prakriti11' as a default hint if users exist
            default_index = user_options.index("prakriti11") if "prakriti11" in user_options else 0
            user_to_auth = st.selectbox("Username", options=user_options, index=default_index, key="user_select_auth")
        else:
            user_to_auth = st.text_input("New Username", key="user_new_auth").strip()
            
        password_hint = "Password (Required for primary users: 'ujjwal11' or 'prakriti11')"
        password = st.text_input(password_hint, type="password", key="password_auth")
        
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("---")
        
        auth_button = st.form_submit_button("Submit")

        if auth_button:
            if not user_to_auth:
                st.error("Please enter or select a valid username.")
            else:
                is_existing = user_to_auth in _get_all_users(DB)
                is_primary = user_to_auth in VALID_USERS
                
                if is_existing:
                    # --- LOGIN LOGIC ---
                    if is_primary and VALID_USERS[user_to_auth] != password:
                        st.error("Invalid password for primary user.")
                    else:
                        # Successful login (either primary user with correct password or data-only user)
                        st.session_state["auth_ok"] = True
                        st.session_state["auth_user"] = user_to_auth
                        st.session_state["chat_history"] = []
                        st.success(f"Login successful for **{user_to_auth}**.")
                        st.rerun()
                else:
                    # --- REGISTER LOGIC ---
                    if is_primary:
                         st.error("Cannot create primary user without system registration.")
                    else:
                        # Register as a new data-only user
                        if password:
                            VALID_USERS[user_to_auth] = password # Simulate saving the new user
                        st.session_state["auth_ok"] = True
                        st.session_state["auth_user"] = user_to_auth
                        st.session_state["chat_history"] = []
                        st.success(f"New user **{user_to_auth}** created and logged in! Start adding transactions.")
                        st.rerun()


if "auth_ok" not in st.session_state:
    st.session_state["auth_ok"] = False
    st.session_state["auth_user"] = None

if not st.session_state["auth_ok"]:
    _login_view()
    st.stop()


# Get the current logged-in user ID
CURRENT_USER_ID = st.session_state["auth_user"]


# ---------- Post-Login Setup (Coin Rain Control) ----------
if "coin_rain_start" not in st.session_state:
    st.session_state["coin_rain_start"] = None
    st.session_state["coin_rain_show"] = False

if st.session_state["coin_rain_show"]:
    # Check if coin rain duration has passed
    if st.session_state["coin_rain_start"] and datetime.now() > st.session_state["coin_rain_start"] + timedelta(seconds=RAIN_DURATION_SEC + 0.5):
        st.session_state["coin_rain_show"] = False
        st.session_state["coin_rain_start"] = None
        st.rerun() # Stop the continuous rerun

# ============================== Main Body of App ==============================

# ---------- CSS (Full CSS included, omitting most lines for readability in this output block) ----------
st.markdown(
    f"""
<style>
html, body, [data-testid="stAppViewContainer"] {{ background: #0f1117; color: #eaeef6; font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }}
section[data-testid="stSidebar"] {{ display: none !important; }}
.navbar {{ position: sticky; top: 0; z-index: 1000; padding: 12px 18px; margin: 0 0 18px 0; border-radius: 14px; background: radial-gradient(120% 120% at 0% 0%, #ffd9ea 0%, #ffcfe3 30%, rgba(255,255,255,0.08) 70%); box-shadow: 0 12px 30px rgba(255, 105, 180, 0.25), inset 0 0 60px rgba(255,255,255,0.25); border: 1px solid rgba(255,255,255,0.35); display: flex; justify-content: space-between; align-items: center; }}
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
.streak-card{{ border-radius:16px; padding:16px; margin-top:10px; background:linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02)); border:1px solid rgba(255,255,255,.12); box-shadow:0 12px 30px rgba(0,0,0,.35);}}
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
.profile-pic{{ width:70px;height:70px;border-radius:50%;object-fit:cover; box-shadow:0 6px 20px rgba(0,0,0,.35); border:2px solid #25D366;}}
.insights-wrap, .upi-qr-wrap {{
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
.promise{{ font-weight:900; font-size:20px; letter-spacing:.3px; color:#ffe1f0; text-align:center; margin:8px 0 2px 0; animation: glow 3s ease-in-out infinite, jump 3s ease-in-out infinite;}}
@keyframes glow{{ 0%{{ text-shadow:0 0 6px #ff7ab8, 0 0 16px #ffb3d6 }} 50%{{ text-shadow:0 0 12px #ff57a6, 0 0 26px #ffc2e1 }} 100%{{ text-shadow:0 0 6px #ff7ab8, 0 0 16px #ffb3d6 }} }}
@keyframes jump{{ 0%{{ transform:translateY(0) }} 15%{{ transform:translateY(-8px) }} 30%{{ transform:translateY(0) }} 45%{{ transform:translateY(-5px) }} 60%,100%{{ transform:translateY(0) }} }}
.coin-rain {{ position: fixed; inset: 0; pointer-events: none; z-index: 9999; animation: fade-out {RAIN_DURATION_SEC + 1}s forwards; }}
.coin-rain span{{ position:absolute; top:-50px; font-size:22px; filter:drop-shadow(0 6px 8px rgba(0,0,0,.35)); animation: rain 2.2s linear infinite; }}
@keyframes rain{{ 0%{{ transform:translateY(-60px) rotate(0deg); opacity:0 }} 15%{{ opacity:1 }} 100%{{ transform: translateY(120vh) rotate(360deg); opacity:0 }} }}
@keyframes fade-out {{ 0% {{ opacity: 1; visibility: visible; }} 90% {{ opacity: 1; visibility: visible; }} 100% {{ opacity: 0; visibility: hidden; }} }}
@keyframes revolve {{ 0% {{ transform: rotate(0deg) scale(1); }} 50% {{ transform: rotate(180deg) scale(1.05); }} 100% {{ transform: rotate(360deg) scale(1); }} }}
.revolving-brain {{ font-size: 32px; display: inline-block; animation: revolve 3s linear infinite, qr-glow 2s infinite alternate; color: #ffb3d6; margin-left: 10px; }}
@keyframes pulsing_arrow {{ 0% {{ transform: scale(1) translateX(0px); opacity: 1; }} 50% {{ transform: scale(1.1) translateX(10px); opacity: 0.8; }} 100% {{ transform: scale(1) translateX(0px); opacity: 1; }} }}
.callout-box-vfa {{ background: #ff57a6; color: white; padding: 8px 12px; border-radius: 8px; font-weight: 600; margin-top: 15px; display: flex; align-items: center; gap: 10px; animation: qr-glow 1.5s infinite alternate; }}
.animated-arrow-vfa {{ font-size: 24px; animation: pulsing_arrow 1.5s infinite; display: inline-block; }}
</style>
""",
    unsafe_allow_html=True,
)


# Get the current logged-in user ID
CURRENT_USER_ID = st.session_state["auth_user"]


# ---------- Navbar (with Cash Flow Crew animation) ----------
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
    sound_status = "🔊 ON" if not st.session_state.get('sound_muted', False) else "🔇 OFF"
    if st.button(sound_status, key='toggle_sound', help="Toggle payment notification sound"):
        st.session_state['sound_muted'] = not st.session_state.get('sound_muted', False)
        st.rerun()

    if PROFILE64:
        st.markdown(
            f"""<img class="profile-pic" src="data:image/jpg;base64,{PROFILE64}" />""",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

# Check and warn if Gemini SDK is missing
if HAS_GEMINI_SDK:
    st.success("🎉 **Now integrated with GEMINI!** Access intelligent financial guidance via the Smart Chatbot.")
else:
    st.error("⚠️ **GEMINI SDK Missing:** Chatbot intelligence is disabled. Please run `pip install google-genai`.")

# ---------- Promise ----------
if "promise_text" not in st.session_state:
    st.session_state["promise_text"] = "I promise that I will save 100 rupees per day"

st.markdown(f"<div class='promise'>{st.session_state['promise_text']}</div>", unsafe_allow_html=True)
new_p = st.text_input("Change promise line", st.session_state["promise_text"])
if new_p != st.session_state["promise_text"]:
    st.session_state["promise_text"] = new_p
    st.rerun()

# --- Start of Tabbed Structure ---
tab_dashboard, tab_stock = st.tabs(["💰 Personal Dashboard", "📈 Real-time Stock Data (AlphaVantage)"])

with tab_dashboard:
    # ---------- Toolbar ----------
    tb1, tb2, tb3, tb4, tb5, tb6 = st.columns([1.6, 1.4, 1.4, 1.8, 1.2, 1])
    with tb1:
        data_source = st.radio("Data source", ["Use saved data", "Generate sample"], index=0, horizontal=True)
    with tb2:
        plot_type = st.selectbox(
            "Plot type",
            ["Line plot (trend)", "Bar plot (aggregate)", "Count plot (category counts)", "Scatter plot", "Distribution (KDE)", "Histogram", "Donut Chart", "Heatmap"],
        )
    with tb3:
        group_period = st.selectbox("Group period", ["Monthly", "Weekly", "Daily"], index=0)
    with tb4:
        bar_mode = st.selectbox("Bar mode", ["By Category", "By Period (stacked by type)"], index=1 if plot_type.startswith("Bar") else 0)
    with tb5:
        numeric_col = st.selectbox("Numeric (scatter/hist)", ["amount"], index=0)
    with tb6:
        if st.button("Logout", key="logout_1"):
            for k in ("auth_ok", "auth_user", "chat_history", "coin_rain_show", "coin_rain_start", "longest_streak_ever"):
                st.session_state.pop(k, None)
            st.rerun()

    # ---------- Load data (Using DB) ----------
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
            raw_df['date'] = pd.to_datetime(raw_df['date']).dt.date

    if raw_df is None:
        st.stop()

    try:
        df = normalize(raw_df)
    except Exception as e:
        st.error(f"Error normalizing data: {e}. Please check column names.")
        st.stop()

    # ---------- Filters ----------
    f1, f2, f3 = st.columns([1.3, 1.6, 1.1])
    if df.empty:
        st.info("No data available after loading/generation.")
        view = df.copy() 
        tmp = add_period(view, group_period) 
        # Default empty values needed for KPIs/Chatbot
        total_income = 0
        total_expense = 0
        net = 0
        avg_per = 0
        curr_streak, longest_streak = 0, 0
        target_daily = 200
        today_hit = False
        val_today = 0
        thresh = 1
    else:
        min_d = pd.to_datetime(df["date"]).min()
        max_d = pd.to_datetime(df["date"]).max()

        with f1:
            start = st.date_input("Start date", min_value=min_d, max_value=max_d, value=min_d, key="start_1")
            end = st.date_input("End date", min_value=min_d, max_value=max_d, value=max_d, key="end_1")
        with f2:
            cats = sorted(df["category"].unique().tolist())
            sel_cats = st.multiselect("Categories", options=cats, default=cats)
        with f3:
            types = sorted(df["type"].unique().tolist())
            sel_types = st.multiselect("Types", options=types, default=types)

        mask = (pd.to_datetime(df["date"]) >= pd.to_datetime(start)) & (pd.to_datetime(df["date"]) <= pd.to_datetime(end))
        view = df[mask & df["category"].isin(sel_cats) & df["type"].isin(sel_types)].copy()
        tmp = add_period(view, group_period)

        # Recalculate KPI values
        total_income = view[view["type"] == "income"]["amount"].sum()
        total_expense = view[view["type"] == "expense"]["amount"].sum()
        net = total_income - total_expense
        avg_per = tmp.groupby("period")["amount"].sum().mean() if not tmp.empty else 0


    # ---------- KPIs ----------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Income", money(total_income))
    m2.metric("Total Expense", money(total_expense))
    m3.metric("Net", money(net))
    m4.metric(f"Avg {group_period}", money(avg_per))
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Saving Streak ----------
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
    curr_streak, longest_streak = 0, 0
    if not dn.empty:
        dn_last = dn.tail(lookback).copy()
        thresh = target_daily if strict else max(1, target_daily * 0.6)
        hit = dn_last["net_saving"] >= thresh
        hit.index = dn_last["day"]
        curr_streak, longest_streak = compute_streak(hit)

        # Update longest streak
        if longest_streak > st.session_state.get("longest_streak_ever", 0):
             st.session_state["longest_streak_ever"] = int(longest_streak)
        
        today_date = date.today()
        val_today = dn_last[dn_last["day"] == today_date]["net_saving"].iloc[-1] if today_date in dn_last["day"].values else 0
        today_hit = val_today >= thresh

        pig_col, s1, s2, s3 = st.columns([1.1, 1, 1, 1.6])
        pig_class = "piggy" + ("" if today_hit else " dim")
        coins_html = '<div class="coin-fall">🪙</div><div class="coin-fall">🪙</div><div class="coin-fall">🪙</div>' if today_hit else ""

        with pig_col:
            st.markdown(f"""<div class="piggy-wrap"><div class="{pig_class}">🐷</div>{coins_html}</div>""", unsafe_allow_html=True)
            st.caption("Piggy lights up when today's net meets target.")
        with s1:
            st.markdown("Today")
            st.markdown(f"<div class='streak-metric'>{'✅' if today_hit else '❌'}</div>", unsafe_allow_html=True)
            st.caption(f"Saved: {money(val_today)} / ₹{target_daily:,}")
        with s2:
            st.markdown("Current Streak")
            st.markdown(f"<div class='streak-metric'>{curr_streak} days</div>", unsafe_allow_html=True)
        with s3:
            st.markdown("Longest Streak")
            st.markdown(f"<div class='streak-metric'>{longest_streak} days</div>", unsafe_allow_html=True)
            st.caption(f"All-time longest: {st.session_state.get('longest_streak_ever', 0)} days")

        mini = dn_last.copy()
        mini["hit"] = np.where(mini["net_saving"] >= thresh, "Hit", "Miss")
        fig_streak = px.bar(
            mini.reset_index(), x="day", y="net_saving", color="hit",
            color_discrete_map={"Hit": "#0ea5e9", "Miss": "#ef4444"},
            title=f"Net saving (last {lookback} days)", labels={"day": "Day", "net_saving": "₹"},
        )
        fig_streak.update_layout(height=260, showlegend=True, legend_title="", template="plotly_dark")
        st.plotly_chart(fig_streak, use_container_width=True, config={"displayModeBar": False}, key="streak_chart_1")
    else:
        st.info("No transactions in the current date range to compute a streak.")
    st.markdown("</div>", unsafe_allow_html=True)
    # ---------- END of Saving Streak ----------


    # ---------- UPI QR + CSV Upload (FIXED) ----------
    st.markdown("---")
    qr1, qr2 = st.columns([1.2, 1.8])

    with qr1:
        # UPDATED QR Logic with Flicker CSS
        st.markdown('<div class="upi-qr-wrap">', unsafe_allow_html=True)
        st.subheader("Scan & Add Income")

        # Display the QR code image
        qr_path = _pick_qr_path()
        if qr_path:
            st.image(str(qr_path), caption=f"Scan & add ₹100 per week to make you smart! 🧠\nUPI ID: {UPI_ID}", use_container_width=True)
        else:
            st.warning(f"QR not found. Using UPI ID: {UPI_ID}")
            
        qr_upload = st.file_uploader("Replace QR (optional)", type=["png", "jpg", "jpeg"], key="qr_up")
        if qr_upload is not None:
            msg = _save_uploaded_qr(qr_upload)
            st.success(msg)
            st.rerun() 

        st.markdown("</div>", unsafe_allow_html=True)
        
        # --- Transaction Input ---
        st.markdown("#### Manual/Simulated Payment")
        scan_amt = st.number_input("Amount scanned (₹)", min_value=1, value=100, step=1, key="scan_amount")
        
        if st.button(f"I scanned ₹{scan_amt} — Add to bucket", key="add_bucket_1", use_container_width=True):
            DB.add_txn(user_id=CURRENT_USER_ID, dt=date.today(), amount=float(scan_amt), category="collection", description="Scanned UPI payment", typ="income")
            DB.save()
            
            # --- FIX: Play Sound and Green Tick ---
            play_paid_sound("Expert Cash dasboard", float(scan_amt)) 
            green_tick(f"Payment of {money(scan_amt)} recorded successfully!")
            
            st.session_state["coin_rain_show"] = True
            st.session_state["coin_rain_start"] = datetime.now()
            st.rerun()

        bucket_total = DB.piggy_balance(CURRENT_USER_ID, "collection") 
        st.markdown(f"*Current Bucket total:* <span style='font-weight:700'>{money(bucket_total)}</span>", unsafe_allow_html=True)


    with qr2:
        # --- CSV Upload (FIXED) ---
        st.subheader("Upload Transactions File (CSV/Excel)")
        uploaded_csv = st.file_uploader("Upload .csv or .xlsx", type=["csv", "xlsx"], key="direct_csv_upload")
          
        if uploaded_csv is not None:
            try:
                uploaded_df = read_file(uploaded_csv)
                if not all(col in uploaded_df.columns.str.lower() for col in ['date', 'amount']):
                    st.error("File must contain 'date' and 'amount' columns.")
                else:
                    uploaded_df = normalize(uploaded_df)
                    save_transactions(CURRENT_USER_ID, uploaded_df)
                    DB.save()
                    green_tick("File uploaded and data saved successfully!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error processing file: {e}")
                st.info("Ensure file has 'date', 'amount', 'category', 'type' columns.")

        st.markdown("---")
        
        # --- VFA Section ---
        st.subheader("💡 Personal Virtual Financial Advisor (VFA)")

        st.markdown("""
            <div class="insights-wrap">
                <span class="animated-arrow-vfa">➡️</span>
                <span>Your VFA has new insights!</span>
            </div>
        """, unsafe_allow_html=True)
          
        st.markdown("The VFA analyzes your spending history to generate a personalized action plan. **Check the chatbot below for advice!**")
        
        # --- Google Form ---
        st.markdown("### Daily Savings Form")
        st.markdown(
            f"<a href='{FORM_URL}' target='_blank' style='text-decoration:none'>"
            f"<button style='background:#ff4da6;color:#fff;border:none;padding:10px 14px;border-radius:8px;font-weight:700;cursor:pointer; width: 100%;'>"
            f"Google form Money Collection</button></a>",
            unsafe_allow_html=True,
        )

    # Show coin rain overlay (and trigger rerun if active)
    if st.session_state["coin_rain_show"]:
        show_coin_rain(RAIN_DURATION_SEC)
        # Continuous rerun for smooth animation until duration expires (handled in Post-Login Setup)
        if st.session_state["coin_rain_start"] and datetime.now() < st.session_state["coin_rain_start"] + timedelta(seconds=RAIN_DURATION_SEC + 0.5):
            st.rerun() 

    # ---------- Main charts & table (FULL LOGIC RESTORED) ----------
    st.markdown("---")
    left, right = st.columns([3, 1])

    with left:
        st.subheader("Interactive chart")
        if tmp.shape[0] == 0:
            st.info("No data in current selection — adjust filters.")
        else:
            # Chart plotting logic (fully restored)
            if plot_type.startswith("Line"):
                agg = tmp.groupby(["period", "type"])["amount"].sum().reset_index()
                fig = px.area(agg, x="period", y="amount", color="type", line_group="type", title=f"Trend by {group_period}")
            elif plot_type.startswith("Bar plot"):
                if bar_mode == "By Category":
                    bar = view.groupby("category")["amount"].sum().reset_index().sort_values("amount", ascending=False)
                    fig = px.bar(bar, x="category", y="amount", color="category", title=f"Spending by category ({group_period} selection)")
                else:
                    bar = tmp.groupby(["period", "type"])["amount"].sum().reset_index()
                    fig = px.bar(bar, x="period", y="amount", color="type", barmode="stack", title=f"Amount by {group_period} (stacked by type)")
            elif plot_type.startswith("Count plot"):
                cnt = view.groupby("category").size().reset_index(name="count").sort_values("count", ascending=False)
                fig = px.bar(cnt, x="category", y="count", color="category", title="Transaction counts by category")
            elif plot_type.startswith("Scatter"):
                fig = px.scatter(view, x="date", y="amount", color="category", hover_data=["description", "type"], title="Amount scatter over time")
            elif plot_type.startswith("Distribution"):
                data_kde = view[view["type"] == "expense"]["amount"]
                fig = px.histogram(data_kde, x="amount", nbins=40, histnorm="density", marginal="rug", title="Expense distribution (KDE approximation)")
            elif plot_type.startswith("Histogram"):
                fig = px.histogram(view, x="amount", nbins=40, color="type", title="Amount histogram")
            elif plot_type.startswith("Donut Chart"):
                donut_data = view.groupby('category')['amount'].sum().reset_index()
                fig = px.pie(donut_data, values='amount', names='category', title='Spending by Category', hole=0.5, color_discrete_sequence=px.colors.sequential.RdPu)
            elif plot_type.startswith("Heatmap"):
                view_copy = view.copy()
                view_copy['day_of_week'] = pd.to_datetime(view_copy['date']).dt.day_name()
                
                heatmap_data = view_copy.groupby(['category', 'day_of_week'])['amount'].sum().unstack(fill_value=0)
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                heatmap_data = heatmap_data.reindex(columns=[d for d in day_order if d in heatmap_data.columns], fill_value=0)
                
                fig = px.heatmap(heatmap_data, 
                                 x=heatmap_data.columns, 
                                 y=heatmap_data.index, 
                                 color_continuous_scale="RdPu",
                                 title="Spending Heatmap: Category vs. Day of Week")
            else:
                 # Default fallback
                 agg = tmp.groupby(["period", "type"])["amount"].sum().reset_index()
                 fig = px.area(agg, x="period", y="amount", color="type", line_group="type", title=f"Trend by {group_period}")

            fig.update_layout(height=520, template="plotly_dark", legend_title="")
            st.plotly_chart(fig, use_container_width=True, key="main_chart_1")

        st.subheader(f"Transactions (filtered for {CURRENT_USER_ID})")
        st.dataframe(view.sort_values("date", ascending=False).reset_index(drop=True), height=300)

    with right:
        # --- Smart Chatbot (FULL LOGIC RESTORED) ---
        st.subheader("Expert Finance Chat bot (Cash Crew AI)")
        
        if "thinking" in st.session_state and st.session_state["thinking"]:
            st.markdown('<div style="display:flex; align-items:center;">Thinking... <span class="revolving-brain">🧠</span></div>', unsafe_allow_html=True)

        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        # Load KB resources only if needed
        kb_texts = kb_texts_from_file()
        vect = load_vectorizer()
        kb_mat = None
        if vect is None and len(kb_texts) > 0:
            vect, kb_mat = build_vectorizer(kb_texts)
        elif vect is not None and len(kb_texts) > 0:
            kb_mat = vect.transform(kb_texts)

        with st.form("chatbot_form", clear_on_submit=True):
            user_q = st.text_input("Ask (e.g., 'top categories', 'invest advice', 'help')", key="chat_input_form")
            send_button = st.form_submit_button("Send")
            
            if send_button and user_q:
                st.session_state["thinking"] = True
                
                ql = user_q.lower()
                ans = None
                personal_context = (
                    f"Team Name: {TEAM_INFO['Team Name']}. Leader: {TEAM_INFO['Team Leader']} (Idea 💡 behind this project). "
                    f"Leader's Expertise: {TEAM_INFO['Leader Expertise']}. Frontend Developer: {TEAM_INFO['Frontend']}. "
                    f"Guided by: {TEAM_INFO['Guidance']}. Contact: {TEAM_INFO['Contact']}. Email: {TEAM_INFO['Email']}. "
                    f"Financial support UPI: {TEAM_INFO['Donate UPI']}."
                )
                
                top5 = view[view["type"] == "expense"].groupby("category")["amount"].sum().sort_values(ascending=False).head(5)
                top_expenses_str = ", ".join([f"{k}: {money(v)}" for k, v in top5.items()])
                net_saving_proxy = float(net) # Use calculated net
                
                # 1. Special handlers (plot, overview, streak)
                if any(k in ql for k in ["plot", "graph", "explain", "describe", "visual", "chart"]):
                    ans = explain_plot_and_data(user_q, view, tmp, plot_type, group_period)
                elif any(k in ql for k in ["overview", "project overview", "explain project", "advantage", "why use", "what can you do", "expert finance"]):
                    ans = project_overview_and_advantages()
                    if "expert finance" in ql:
                         ans = "**I am the Expert Finance Chat bot for Cashflow Crew, powered by Gemini!** I provide insights and financial advice tailored to your spending. 🚀\n\n" + ans
                elif "streak" in ql:
                    ans = f"✅ Your **Current Saving Streak** is **{curr_streak} days**! Your **Longest Streak** is **{longest_streak} days**. Keep it up! To continue the streak, your daily net saving must be at least {money(thresh)}."

                # 2. Local Data Insights
                if ans is None:
                    # Simplified local data check
                    if ("total" in ql or "what is" in ql) and "expense" in ql:
                        s = view[view["type"] == "expense"]["amount"].sum()
                        ans = f"💸 *Total expense in the current view:* **{money(s)}**."
                    elif ("total" in ql or "what is" in ql) and "income" in ql:
                        s = view[view["type"] == "income"]["amount"].sum()
                        ans = f"💰 *Total income in the current view:* **{money(s)}**."
                
                # 3. KB TF-IDF semantic match
                if ans is None and kb_mat is not None and vect is not None:
                    tfidf_match = tfidf_answer(user_q, vect, kb_texts, kb_mat)
                    if tfidf_match and " - " in tfidf_match: # Found a KB match (e.g., 'help - Type questions...')
                        ans = "📚 *Knowledge Base Match:* " + tfidf_match.split(" - ", 1)[1]

                # 4. Smart API / Gemini Catch-all
                if ans is None or any(k in ql for k in ["invest", "advice", "market", "recommend", "gemini"]):
                    if HAS_GEMINI_SDK:
                        gemini_prompt = (
                            f"User Query: {user_q}\n"
                            f"Current Filtered Net Savings: {money(net_saving_proxy)}\n"
                            f"Top 5 Expenses in view: {top_expenses_str}\n"
                            "Based on the user's question, the current financial context, and your persona, provide concise, generative financial advice or a smart response. If the question is purely casual, prioritize a friendly, casual, and brief response."
                        )
                        ans = gemini_query(gemini_prompt, st.session_state.get("chat_history", []), personal_context)
                    elif ans is None:
                        ans = "I couldn't find a direct answer. Try rephrasing or check the KB with *'help'*."
                
                st.session_state["thinking"] = False

                if ans:
                    st.session_state.chat_history.append(("You", user_q))
                    st.session_state.chat_history.append(("Bot", ans))
                    st.rerun()

        for speaker, msg in st.session_state.get("chat_history", [])[-12:]:
            if speaker == "You":
                st.markdown(f"*You:* {msg}")
            else:
                st.markdown(f"<div class='bot'>{msg}</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        # Google Form Embed (secondary location)
        st.markdown("### Google Form Embedded View")
        components.html(f"""
             <iframe src="{FORM_URL}" width="100%" height="400px" style="border:0; border-radius: 8px;"></iframe>
             """, height=420)


with tab_stock:
    # --- Stock tab logic (FULL LOGIC RESTORED) ---
    st.header("📈 Real-time Stock Data (AlphaVantage)")
    st.info("This feature uses a simulated stock API key to fetch real-time stock quotes. Historical charts are generated from simulated data.")
      
    col_sym, col_button = st.columns([2, 1])
      
    with col_sym:
        symbol = st.text_input("Enter Stock Symbol (e.g., TCS.BSE, RELIANCE.NSE)", value="ITC.BSE", key="stock_symbol_input").upper()
      
    with col_button:
        st.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
        if st.button("Fetch Quote & Charts", use_container_width=True, key="fetch_quote_charts_btn_2"):
            st.session_state['last_quote'] = fetch_stock_quote(symbol)
            st.session_state['daily_data'] = generate_simulated_daily_data(symbol)
            st.rerun() 

    if 'last_quote' in st.session_state and isinstance(st.session_state['last_quote'], dict):
        quote = st.session_state['last_quote']
        daily_df = st.session_state.get('daily_data')
          
        price = quote.get("price", "N/A")
        change = quote.get("change", "N/A")
        last_update = quote.get("last_updated", "N/A")
          
        st.markdown("---")
        st.subheader(f"Quote for {quote.get('symbol', symbol)}")
          
        m_p, m_c, m_v = st.columns(3)
          
        m_p.metric("Current Price (₹)", f"₹{price}")
        m_c.metric("Change", change, delta_color="normal")
        m_v.metric("Volume", quote.get("volume", "N/A"))
          
        st.caption(f"Last updated: {last_update}")
          
        st.markdown("---")
        st.subheader("Historical & Portfolio Visualizations")

        chart1, chart2 = st.columns([2, 1])

        with chart1:
            if daily_df is not None:
                st.markdown("#### Line Chart: Last 60 Days Closing Price Trend")
                fig_line = px.line(daily_df, x=daily_df.index, y='Close Price (₹)', 
                                    title=f"Price Trend for {symbol}",
                                    labels={'Close Price (₹)': 'Price (₹)', 'Date': 'Date'})
                fig_line.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig_line, use_container_width=True)
            else:
                st.info("Historical data not available.")

        with chart2:
            st.markdown("#### Donut/Pie Chart: Sample Portfolio Allocation")
            portfolio_data = pd.DataFrame({
                'Asset': ['TCS', 'Reliance', 'HDFC Bank', 'Cash'],
                'Value (₹)': [150000, 120000, 90000, 40000]
            })
              
            fig_donut = px.pie(portfolio_data, values='Value (₹)', names='Asset',
                                 title='Current Portfolio Distribution', 
                                 hole=0.4, 
                                 color_discrete_sequence=px.colors.sequential.RdPu)
            fig_donut.update_traces(textinfo='percent+label')
            fig_donut.update_layout(template="plotly_dark", height=400, showlegend=False)
            st.plotly_chart(fig_donut, use_container_width=True)
              
        st.markdown("---")
        st.markdown("#### Bar Chart: Last 60 Days Daily Volume")
        if daily_df is not None:
            fig_bar = px.bar(daily_df, x=daily_df.index, y='Volume',
                              title=f"Daily Volume for {symbol}",
                              labels={'Volume': 'Volume', 'Date': 'Date'})
            fig_bar.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
          
    else:
        st.info("Enter a stock symbol and click 'Fetch Quote & Charts'.")
        st.markdown("#### Sample Portfolio Allocation (Placeholder)")
        portfolio_data = pd.DataFrame({
            'Asset': ['Equity', 'Debt', 'Commodities', 'Cash'],
            'Value (₹)': [40, 30, 15, 15]
        })
        fig_donut_placeholder = px.pie(portfolio_data, values='Value (₹)', names='Asset',
                                 title='Portfolio Distribution', 
                                 hole=0.4, 
                                 color_discrete_sequence=px.colors.sequential.RdPu)
        fig_donut_placeholder.update_traces(textinfo='percent')
        fig_donut_placeholder.update_layout(template="plotly_dark", height=300, showlegend=True)
        st.plotly_chart(fig_donut_placeholder, use_container_width=True)


# --- MAIN APPLICATION ENTRY POINT ---
if __name__ == '__main__':
    # Save database changes automatically
    st.session_state["DB"].save()