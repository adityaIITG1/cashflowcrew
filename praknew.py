

# app.py — Personal Finance & Spending Analyzer (V14 - FINAL UX FIXES)

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
    status: str 
    note: str = ""
    created_at: str = datetime.utcnow().isoformat(timespec="seconds")

@dataclass
class Transaction:
    id: int
    user_id: str # NEW: User identifier
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
                # FIX: Ensure 'user_id' exists, defaulting to 'prakriti11' for old data
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

# --- UPI Details ---
UPI_ID = 'jaiswalprakriti26@okaxis'
UPI_PAYMENT_STRING = f'upi://pay?pa={UPI_ID}&pn=PRAKRITI&cu=INR'

# --- Personalized Information ---
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
    if path.exists(): return
    try:
        img = Image.new("RGB", (size, size), color=color)
        d = ImageDraw.Draw(img)
        d.text((size // 4, size // 2), text, fill=(0, 0, 0))
        img.save(path)
    except Exception:
        pass

def _img64(path: Path | None) -> str:
    try:
        if not path or not path.exists(): return ""
        with open(path, "rb") as fh: return base64.b64encode(fh.read()).decode("utf-8")
    except Exception: return ""

def _pick_qr_path() -> Path | None:
    if UPI_QR_IMG.exists(): return UPI_QR_IMG
    if UPI_QR_IMG_JPG.exists(): return UPI_QR_IMG_JPG
    return None

def _generate_default_upi_qr(upi_string: str, path: Path):
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
    try:
        img = Image.open(file).convert("RGB")
        img.save(UPI_QR_IMG) 
        return "QR updated. If not visible, press 'Rerun' or refresh."
    except Exception as e: return f"Failed to save QR: {e}"

def money(v: float) -> str:
    return f"₹{v:,.0f}" if abs(v) >= 1000 and v.is_integer() else f"₹{v:,.2f}"

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
    # Use fallback if actual sound file is not available
    audio_src = SOUND_EFFECT_URL 
    
    # Check if sound is muted
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
    # This must be called continuously for the effect to be non-stop. 
    # The CSS animation handles the unstoppable part.
    coin_spans = "".join([
        f"<span style='left:{random.randint(5, 95)}%; animation-delay:{random.uniform(0, RAIN_DURATION_SEC/2):.2f}s;'>🪙</span>" 
        for _ in range(20)
    ])
    st.markdown(
        f"""
<style>
.coin-rain {{
  position: fixed; inset: 0; pointer-events: none; z-index: 9999;
  /* Use a persistent animation block */
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

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=contents
        )
          
        return f"🧠 *Gemini Smart AI:* {response.text}"

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

# --- KB/TFIDF Helpers (Minified) ---

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

def to_excel_bytes(df: pd.DataFrame) -> bytes:
    out = BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="data")
    return out.getvalue()

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

def read_file(file):
    if isinstance(file, (str, Path)):
        if str(file).endswith(".csv"): return pd.read_csv(file)
    return pd.read_excel(file)

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
        # FIX: Ensure full date range works with date objects
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

# --- VFA Plan Generation ---
def generate_financial_plan_file(df: pd.DataFrame) -> bytes:
    """Generates a sample CSV financial plan based on current data."""
    # Ensure date column is datetime for manipulation
    if not df.empty:
        df_copy = df.copy()
        df_copy['date'] = pd.to_datetime(df_copy['date'])
    else:
        df_copy = pd.DataFrame({'date': [date.today()], 'amount': [0], 'category': ['Initial'], 'type': ['income']})
    
    # Simple rule-based plan simulation
    plan_data = []
    
    # 1. Monthly Summary
    monthly_summary = df_copy.copy()
    monthly_summary['Month'] = monthly_summary['date'].dt.to_period('M').astype(str)
    
    if not monthly_summary.empty:
        net_summary = monthly_summary.groupby('Month').agg(
            Total_Income=('amount', lambda x: x[monthly_summary.loc[x.index, 'type'] == 'income'].sum()),
            Total_Expense=('amount', lambda x: x[monthly_summary.loc[x.index, 'type'] == 'expense'].sum()),
            Net_Savings=('amount', lambda x: x[monthly_summary.loc[x.index, 'type'] == 'income'].sum() - x[monthly_summary.loc[x.index, 'type'] == 'expense'].sum())
        ).reset_index()
    else:
        net_summary = pd.DataFrame({'Month': ['N/A'], 'Total_Income': [0], 'Total_Expense': [0], 'Net_Savings': [0]})

    plan_data.append("--- Monthly Performance Summary ---")
    plan_data.append(net_summary.to_csv(index=False))
    
    # 2. Saving Goal Action
    avg_expense = df_copy[df_copy['type'] == 'expense']['amount'].mean() if not df_copy[df_copy['type'] == 'expense'].empty else 500.0
    saving_recommendation = max(50, round(avg_expense * 0.1, 0))
    
    plan_data.append("\n--- Actionable Plan ---")
    plan_data.append("Action,Target,Category,Recommendation")
    plan_data.append(f"Reduce Expense,Monthly,Dining,Reduce dining out by {money(saving_recommendation)} (10% of avg expense).")
    plan_data.append(f"Increase Saving,Weekly,Investment,Invest {money(100)} weekly into low-risk funds.")
    
    # Concatenate all parts into a single string for download
    plan_content = '\n'.join(plan_data)
    
    # Convert to bytes for download
    return plan_content.encode('utf-8')

# --- Helper to save transactions from a DataFrame to MiniDB ---
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

if "paid_orders_applied" not in st.session_state:
    st.session_state["paid_orders_applied"] = set()

if "thinking" not in st.session_state:
    st.session_state["thinking"] = False
    
# FIX: Initialize longest_streak_ever for the streak info to work
if "longest_streak_ever" not in st.session_state:
     st.session_state["longest_streak_ever"] = 0

if "sound_muted" not in st.session_state:
    st.session_state["sound_muted"] = False

# ============================== Login & User Management ==============================

# Simulated User database
VALID_USERS = {
    "prakriti11": "ujjwal11",
    "ujjwal11": "prakriti11",
}

def _get_all_users(db: MiniDB) -> set:
    db_users = {t.user_id for t in db._tx.values()}
    return set(VALID_USERS.keys()) | db_users

def _login_view() -> None:
    # --- Login CSS: Dark Pink Glow ---
    st.markdown("""
    <style>
    /* Ensure the background is dark for the glow to show */
    [data-testid="stAppViewContainer"] > .main {background-color: #0f1117;}
    .login-card-container {
        display: grid; grid-template-columns: 1fr 1fr; gap: 40px;
    }
    .login-card-inner {
        padding: 30px; border-radius: 16px; 
        background: linear-gradient(145deg, rgba(255,105,180,0.1), rgba(255,192,203,0.05));
        border: 2px solid #ff79b0; /* Light Pink Border */
        box-shadow: 0 0 25px rgba(255, 105, 180, 0.9), 0 0 50px rgba(255, 192, 203, 0.5);
        transition: all 0.3s;
    }
    .login-card-inner:hover {
        box-shadow: 0 0 35px #ff57a6, 0 0 60px #ffc0cb;
    }
    .stSelectbox div { background-color: #1a1a1a !important; color: white !important; }
    .stSelectbox button { color: #ff79b0 !important; }
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
      
    with st.form("login_form", clear_on_submit=False):
          
        st.markdown('<div class="login-card-container">', unsafe_allow_html=True)
          
        # 1. Existing User Login Column
        with st.container():
            st.markdown('<div class="login-card-inner">', unsafe_allow_html=True)
            st.subheader("Existing User Login")
              
            # --- FIX: Removed "-- New User --" from selection list
            user_options = [u for u in all_users]
            u_select = st.selectbox("Select Existing Username", options=user_options, key="user_select")
              
            p_label = "Password (Required for primary users 'ujjwal11' or 'prakriti11')" # Provide hint
            p_select = st.text_input(p_label, type="password", key="password_select")
            st.markdown("</div>", unsafe_allow_html=True)
          
        # 2. New User Creation Column
        with st.container():
            st.markdown('<div class="login-card-inner">', unsafe_allow_html=True)
            st.subheader("New User Creation")
            u_new = st.text_input("New Username", key="user_new")
            p_new = st.text_input("New Password (Optional)", type="password", key="password_new")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True) # End login-card-container
          
        st.markdown("---")
        c_login, c_register = st.columns([1, 1])
          
        # --- LOGIN BUTTON ---
        if c_login.form_submit_button("Login to Selected User"):
            user_to_login = u_select
            password_check = p_select
              
            if not user_to_login:
                 st.error("Please select a username.")
            elif user_to_login in VALID_USERS and VALID_USERS[user_to_login] != password_check:
                st.error("Invalid password for existing primary user.")
            else:
                st.session_state["auth_ok"] = True
                st.session_state["auth_user"] = user_to_login
                st.session_state["chat_history"] = []
                st.success(f"Login successful for **{user_to_login}**. Loading data...")
                st.rerun()

        # --- REGISTER BUTTON ---
        if c_register.form_submit_button("Create New User"):
            new_user_id = u_new.strip()
            if not new_user_id:
                st.error("New Username cannot be empty.")
            elif new_user_id in _get_all_users(DB):
                st.error(f"User **{new_user_id}** already exists. Please log in.")
            else:
                if p_new:
                    VALID_USERS[new_user_id] = p_new # Simple in-memory save for non-primary users
                st.session_state["auth_ok"] = True
                st.session_state["auth_user"] = new_user_id
                st.session_state["chat_history"] = []
                st.success(f"New user **{new_user_id}** created and logged in! Start adding transactions.")
                st.rerun()


if "auth_ok" not in st.session_state:
    st.session_state["auth_ok"] = False
    st.session_state["auth_user"] = None

if not st.session_state["auth_ok"]:
    _login_view()
    st.stop()


CURRENT_USER_ID = st.session_state["auth_user"]


# ---------- Post-Login Setup ----------
# Ensure coin animation runs continuously when logged in
if "coin_rain_show" not in st.session_state:
      st.session_state["coin_rain_show"] = True # Start immediately

# ============================== Main Body of App ==============================

# ---------- CSS (Fixed and Extended) ----------
st.markdown(
    f"""
<style>
/* Base Streamlit Styles (Keep dark theme for pink glow) */
html, body, [data-testid="stAppViewContainer"] {{
  background: #0f1117; color: #eaeef6;
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
}}

/* Navbar Styling */
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

/* Coin Drop Animation (in navbar) */
.coin-wrap {{ position: relative; height: 60px; margin: 6px 0 0 0; overflow: hidden; }}
.coin {{ position:absolute; top:-50px; font-size:24px; filter: drop-shadow(0 6px 8px rgba(0,0,0,.35)); animation: drop 4s linear infinite; }}
.coin:nth-child(2){{left:15%; animation-delay:.6s}}
.coin:nth-child(3){{left:30%; animation-delay:.1s}}
.coin:nth-child(4){{left:45%; animation-delay:.9s}}
.coin:nth-child(5){{left:60%; animation-delay:1.8s}}
.coin:nth-child(6){{left:75%; animation-delay:.3s}}
.coin:nth-child(7){{left:90%; animation-delay:.2s}}
@keyframes drop {{ 0%{{ transform: translateY(-60px) rotate(0deg); opacity:0 }} 10%{{ opacity:1 }} 100%{{ transform: translateY(120px) rotate(360deg); opacity:0 }} }}

/* Card Styles */
.card {{border-radius:16px; background:linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02)); padding:16px; box-shadow: 0 12px 30px rgba(0,0,0,0.35); border: 1px solid rgba(255,255,255,0.12);}}
.metric {{font-size:18px; font-weight:700}}
.bot {{background:#111827; color:#e6eef8; padding:10px 12px; border-radius:10px; border:1px solid rgba(255,255,255,.08)}}

/* Streak Card/Piggy */
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
.profile-pic{{
  width:70px;height:70px;border-radius:50%;object-fit:cover;
  box-shadow:0 6px 20px rgba(0,0,0,.35); border:2px solid #25D366;
}}
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

/* Savings Promise Line */
.promise{{
  font-weight:900; font-size:20px; letter-spacing:.3px;
  color:#ffe1f0; text-align:center; margin:8px 0 2px 0;
  animation: glow 3s ease-in-out infinite, jump 3s ease-in-out infinite;
}}
@keyframes glow{{
  0%{{ text-shadow:0 0 6px #ff7ab8, 0 0 16px #ffb3d6 }}
  50%{{ text-shadow:0 0 12px #ff57a6, 0 0 26px #ffc2e1 }}
  100%{{ text-shadow:0 0 6px #ff7ab8, 0 0 16px #ffb3d6 }}
}}
@keyframes jump{{
  0%{{ transform:translateY(0) }} 15%{{ transform:translateY(-8px) }}
  30%{{ transform:translateY(0) }} 45%{{ transform:translateY(-5px) }}
  60%,100%{{ transform:translateY(0) }}
}}

/* Coin rain animation styling (dynamic positioning) */
/* FIX: Ensure coin rain runs non-stop by removing fade-out for this file */
.coin-rain span {{ animation: rain 2.2s linear infinite; }}
.coin-rain {{ animation: none; }} /* Remove fade-out animation */

@keyframes rain{{0%{{transform:translateY(-60px) rotate(0deg);opacity:0}}
15%{{opacity:1}}100%{{transform:translateY(120vh) rotate(360deg);opacity:0}}}}

/* Revolving Brain for thinking state */
@keyframes revolve {{
    0% {{ transform: rotate(0deg) scale(1); }}
    50% {{ transform: rotate(180deg) scale(1.05); }}
    100% {{ transform: rotate(360deg) scale(1); }}
}}
.revolving-brain {{
    font-size: 32px;
    display: inline-block;
    animation: revolve 3s linear infinite, qr-glow 2s infinite alternate; /* Use existing glow for flicker/pulse */
    color: #ffb3d6;
    margin-left: 10px;
}}


/* Custom Pop-up message for VFA */
@keyframes pulsing_arrow {{ 0% {{ transform: scale(1) translateX(0px); opacity: 1; }} 50% {{ transform: scale(1.1) translateX(10px); opacity: 0.8; }} 100% {{ transform: scale(1) translateX(0px); opacity: 1; }} }}
.callout-box-vfa {{
    background: #ff57a6; /* Pink */ color: white; padding: 8px 12px; border-radius: 8px; font-weight: 600; margin-top: 15px;
    display: flex; align-items: center; gap: 10px; animation: qr-glow 1.5s infinite alternate;
}}
.animated-arrow-vfa {{ font-size: 24px; animation: pulsing_arrow 1.5s infinite; display: inline-block; }}

</style>
""",
    unsafe_allow_html=True,
)


# Show coin rain overlay non-stop
if st.session_state["coin_rain_show"]:
    show_coin_rain(RAIN_DURATION_SEC)


# Get the current logged-in user ID
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

# --- Sound Toggle Button ---
with colB:
    st.markdown("<div class='profile-wrap'>", unsafe_allow_html=True)
      
    # Sound Button (Toggle)
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
    # --- Simplified Toolbar ---
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
        # Keep consistent with plot_type for better UX, default to By Category
        default_bar_mode = 1 if plot_type.startswith("Bar") or plot_type.startswith("Line") else 0
        bar_mode = st.selectbox("Bar mode", ["By Category", "By Period (stacked by type)"], index=default_bar_mode)
    with tb5:
        numeric_col = st.selectbox("Numeric (scatter/hist)", ["amount"], index=0)
    with tb6:
        if st.button("Logout", key="logout_1"):
            # Clear all relevant session state for a clean log out
            for k in ("auth_ok", "auth_user", "chat_history", "coin_rain_show", "coin_rain_start", "longest_streak_ever", "promise_text", "last_quote", "daily_data", "DB"):
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
            # Ensure the date column is properly converted to date objects
            raw_df['date'] = pd.to_datetime(raw_df['date']).dt.date

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
    else:
        min_d = df["date"].min()
        max_d = df["date"].max()
        
        # Ensure min_value/max_value/value are date objects
        with f1:
            start = st.date_input("Start date", min_value=min_d, max_value=max_d, value=min_d, key="start_1")
            end = st.date_input("End date", min_value=min_d, max_value=max_d, value=max_d, key="end_1")
        with f2:
            cats = sorted(df["category"].unique().tolist())
            sel_cats = st.multiselect("Categories", options=cats, default=cats)
        with f3:
            types = sorted(df["type"].unique().tolist())
            sel_types = st.multiselect("Types", options=types, default=types)

        # Apply filtering on date objects
        mask = (df["date"] >= start) & (df["date"] <= end)
        view = df[mask & df["category"].isin(sel_cats) & df["type"].isin(sel_types)].copy()
        tmp = add_period(view, group_period)

    # --- KPIs ---
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    total_income = view[view["type"] == "income"]["amount"].sum() if not view.empty else 0
    total_expense = view[view["type"] == "expense"]["amount"].sum() if not view.empty else 0
    net = total_income - total_expense
    
    # Calculate average on the aggregated frame
    avg_per = tmp.groupby("period")["amount"].sum().mean() if not tmp.empty else 0
    
    m1.metric("Total Income", money(total_income))
    m2.metric("Total Expense", money(total_expense))
    m3.metric("Net", money(net))
    m4.metric(f"Avg {group_period}", money(avg_per))
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Saving Streak ---
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
        # Use the day column (date object) as index for streak calculation
        hit.index = dn_last["day"]
        curr_streak, longest_streak = compute_streak(hit)
        
        # Update overall longest streak
        if longest_streak > st.session_state.get('longest_streak_ever', 0):
             st.session_state['longest_streak_ever'] = longest_streak

        pig_col, s1, s2, s3 = st.columns([1.1, 1, 1, 1.6])
        # Check if today's date is in the index and if it hit the target
        today_date = date.today()
        val_today = dn_last[dn_last["day"] == today_date]["net_saving"].iloc[-1] if today_date in dn_last["day"].values else 0
        today_hit = val_today >= thresh

        pig_class = "piggy" + ("" if today_hit else " dim")
        coins_html = '<div class="coin-fall">🪙</div><div class="coin-fall">🪙</div><div class="coin-fall">🪙</div>' if today_hit else ""

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

    # ---------- UPI QR (FIXED) + CSV Upload + VFA Plan Download ----------
    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.markdown('<div class="upi-qr-wrap">', unsafe_allow_html=True)
        st.subheader("Add Income/Upload Data")
          
        # --- FIX: Direct CSV Upload Option ---
        st.markdown("#### Upload Transactions File (CSV/Excel)")
        uploaded_csv = st.file_uploader("Upload .csv or .xlsx", type=["csv", "xlsx"], key="direct_csv_upload")
          
        if uploaded_csv is not None:
            try:
                uploaded_df = read_file(uploaded_csv)
                # Quick check for minimum columns
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
          
        # --- Transaction Add (Manual/UPI) ---
        st.markdown("#### Manual/UPI Income Entry")
        scan_amt = st.number_input("Amount scanned (₹)", min_value=1, value=100, step=1, key="scan_amount")
          
        # FIX: The button should be tied to payment action and sound
        if st.button(f"Add ₹{scan_amt} to bucket (Simulate Payment)", key="add_bucket_1", use_container_width=True):
            DB.add_txn(
                user_id=CURRENT_USER_ID,
                dt=date.today(),
                amount=float(scan_amt),
                category="collection",
                description="Manual/Simulated Payment",
                typ="income",
            )
            DB.save()
              
            # Play Sound and Green Tick
            play_paid_sound(CURRENT_USER_ID, float(scan_amt))
            green_tick(f"Payment of {money(scan_amt)} recorded successfully!")
              
            st.session_state["coin_rain_show"] = True
            st.rerun()
              
        bucket_total = DB.piggy_balance(CURRENT_USER_ID, "collection")
        st.markdown(f"*Bucket total:* <span style='font-weight:700'>{money(bucket_total)}</span>", unsafe_allow_html=True)
          
        # FIX: Removed the confusing QR code display and its related upload/controls
        st.markdown("</div>", unsafe_allow_html=True)


    with right_col:
        # --- VFA Section (Updated Logic) ---
        st.subheader("💡 Personal Virtual Financial Advisor (VFA)")

        st.markdown("""
            <div class="callout-box-vfa">
                <span class="animated-arrow-vfa">➡️</span>
                <span>Your VFA has new insights!</span>
            </div>
        """, unsafe_allow_html=True)
          
        st.markdown("The VFA analyzes your spending history to generate a personalized action plan.")
          
        # --- VFA Plan Generation and Download ---
        plan_bytes = generate_financial_plan_file(view)

        st.download_button(
            label="Download Personalized Financial Plan (CSV)",
            data=plan_bytes,
            file_name=f"VFA_Plan_{CURRENT_USER_ID}_{date.today()}.csv",
            mime="text/csv",
            use_container_width=True,
            key='download_vfa_plan'
        )
        st.caption("This file contains actionable advice based on your current data.")
        st.markdown("---")
          
        # --- Smart Chatbot (Gemini Integrated) ---
        st.subheader("Chat with PRAKRITI AI")
          
        # FIX: Always show thinking if the button was just pressed (handled via session state)
        if "thinking" in st.session_state and st.session_state["thinking"]:
            st.markdown('<div style="display:flex; align-items:center;">Thinking... <span class="revolving-brain">🧠</span></div>', unsafe_allow_html=True)

        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []
              
        # Use a form structure to capture user input and trigger logic
        with st.form("chatbot_form", clear_on_submit=True):
            user_q = st.text_input(
                "Ask (e.g., 'top categories', 'invest advice', 'help')",
                key="chat_input_form", # New key to avoid conflict
            )
            send_button = st.form_submit_button("Send")
            
            if send_button and user_q:
                st.session_state["thinking"] = True
                
                # --- Advanced Chatbot Logic (Placeholder for full integration) ---
                # NOTE: In a real app, this is where the KB lookup and Gemini call would occur.
                
                # Context for Gemini
                context_str = TEAM_INFO["Team Leader"] + " is a " + TEAM_INFO["Leader Expertise"] + ". The user is " + CURRENT_USER_ID + " and their current net savings is " + money(net) + "."
                
                # 1. Attempt internal command or KB lookup (if implemented)
                ans = None
                
                # 2. Fallback to Gemini API if no internal answer
                if ans is None:
                    try:
                        # The actual gemini_query function will handle the API call
                        ans = gemini_query(user_q, st.session_state["chat_history"], context_str)
                    except Exception as e:
                        ans = f"❌ **Error contacting AI:** {e}"
                        
                # Ensure the thinking state is reset and chat history updated
                st.session_state.chat_history.append(("You", user_q))
                st.session_state.chat_history.append(("Bot", ans))
                st.session_state["thinking"] = False
                st.rerun() # Rerun to display the new chat message outside the form

        # Display chat history
        for speaker, msg in st.session_state.get("chat_history", [])[-12:]:
            if speaker == "You":
                st.markdown(f"*You:* {msg}")
            else:
                st.markdown(f"<div class='bot'>{msg}</div>", unsafe_allow_html=True)

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
                donut_data = view.groupby('category')['amount'].sum().reset_index()
                fig = px.pie(donut_data, values='amount', names='category', title='Spending by Category', hole=0.5, color_discrete_sequence=px.colors.sequential.RdPu)
            elif plot_type.startswith("Line plot (trend)") or plot_type.startswith("Bar plot (aggregate)"):
                 agg = tmp.groupby(["period", "type"])["amount"].sum().reset_index()
                 if plot_type.startswith("Line"):
                     fig = px.area(agg, x="period", y="amount", color="type", line_group="type", title=f"Trend by {group_period}")
                 else: # Bar plot
                     if bar_mode == "By Category":
                        agg_cat = view.groupby(["category", "type"])["amount"].sum().reset_index()
                        fig = px.bar(agg_cat, x="category", y="amount", color="type", title=f"Total Amount by Category", barmode='group')
                     else: # By Period (stacked by type)
                        fig = px.bar(agg, x="period", y="amount", color="type", title=f"Total Amount by {group_period}", barmode='stack')
            elif plot_type.startswith("Scatter plot"):
                fig = px.scatter(view, x='date', y='amount', color='category', title="Individual Transactions", hover_data=['description', 'type'])
            elif plot_type.startswith("Distribution") or plot_type.startswith("Histogram"):
                 fig = px.histogram(view, x="amount", color="type", marginal="rug" if plot_type.startswith("Distribution") else None, title=f"Distribution of {numeric_col}", histfunc="sum", histnorm='percent')
            else:
                 # Default fallback if a type is not fully implemented
                 agg = tmp.groupby(["period", "type"])["amount"].sum().reset_index()
                 fig = px.area(agg, x="period", y="amount", color="type", line_group="type", title=f"Trend by {group_period} (Default View)")

            fig.update_layout(height=520, template="plotly_dark", legend_title="")
            st.plotly_chart(fig, use_container_width=True, key="main_chart_1")

        st.subheader(f"Transactions (filtered for {CURRENT_USER_ID})")
        st.dataframe(view.sort_values("date", ascending=False).reset_index(drop=True), height=300)

    with right:
        # --- Right Column VFA content ---
        st.markdown("### Transaction Form")
        st.markdown(
            f"<a href='{FORM_URL}' target='_blank' style='text-decoration:none'>"
            f"<button style='background:#ff4da6;color:#fff;border:none;padding:10px 14px;border-radius:8px;font-weight:700;cursor:pointer; width: 100%;'>"
            f"Google form Money Collection</button></a>",
            unsafe_allow_html=True,
        )

    # --- FIX: Three Extra Graphs at Bottom ---
    st.markdown("---")
    st.markdown("## 📊 Deep Dive Analytics (AI Explained)")
      
    gc1, gc2, gc3 = st.columns(3)

    # 1. Income Source Breakdown (Donut)
    with gc1:
        st.markdown("#### 1. Income Source Breakdown")
        income_data = df[df['type'] == 'income'].groupby('category')['amount'].sum().reset_index()
        fig1 = px.pie(income_data, values='amount', names='category', hole=0.6, title='Income Sources')
        fig1.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown('<div class="callout-box-vfa"><span class="animated-arrow-vfa">➡️</span> **AI Explanation:** This shows your **income diversification**.')

    # 2. Expense vs Income (Area Chart)
    with gc2:
        st.markdown("#### 2. Monthly Expense vs. Income")
        # Ensure we use 'tmp' which has the 'period' column based on the grouping selection
        if not tmp.empty:
            # We use df here to ensure the period aggregation covers all available data, not just the filtered view, for a better monthly flow perspective
            monthly_flow_df = add_period(df, group_period="Monthly")
            monthly_net = monthly_flow_df.groupby(["period", "type"])["amount"].sum().unstack(fill_value=0)
            if "income" not in monthly_net.columns: monthly_net["income"] = 0
            if "expense" not in monthly_net.columns: monthly_net["expense"] = 0
            
            fig2 = go.Figure()
            # Use go.Bar for stacked bar that looks similar to an area chart for net flow
            fig2.add_trace(go.Bar(x=monthly_net.index, y=monthly_net['income'], name='Income', marker_color='#0ea5e9'))
            fig2.add_trace(go.Bar(x=monthly_net.index, y=-monthly_net['expense'], name='Expense', marker_color='#ff79b0'))
            fig2.update_layout(
                barmode='relative', 
                template="plotly_dark", 
                title="Monthly Flow (Income up, Expense down)", 
                height=400,
                yaxis_title="Net Flow (Income/Expense)",
            )
        else:
            fig2 = go.Figure()
            fig2.update_layout(title="No Data for Monthly Flow", height=400, template="plotly_dark")

        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('<div class="callout-box-vfa"><span class="animated-arrow-vfa">➡️</span> **AI Explanation:** Visualizes monthly **cash flow** stability. You want the blue (Income) bar to stay large compared to the pink (Expense) bar.')
      
    # 3. Transaction Count Distribution (Bar Chart)
    with gc3:
        st.markdown("#### 3. Transaction Count Distribution")
        count_data = view.groupby('type').size().reset_index(name='count')
        fig3 = px.bar(count_data, x='type', y='count', title='Transaction Count (Income vs. Expense)')
        fig3.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown('<div class="callout-box-vfa"><span class="animated-arrow-vfa">➡️</span> **AI Explanation:** Shows if you make many small purchases or few large ones. Useful for **behavioral tracking**.')


with tab_stock:
    # Stock tab logic
    st.header("📈 Real-time Stock Data (AlphaVantage)")
    st.info("This feature uses a simulated stock API key to fetch real-time stock quotes. Historical charts are generated from simulated data.")
      
    col_sym, col_button = st.columns([2, 1])
      
    with col_sym:
        symbol = st.text_input("Enter Stock Symbol (e.g., TCS.BSE, RELIANCE.NSE)", value="ITC.BSE", key="stock_symbol_input").upper()
      
    with col_button:
        st.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
        # Use a consistent name for the stock data cache trigger
        if st.button("Fetch Quote & Charts", use_container_width=True, key="fetch_quote_charts_btn_2"):
            st.session_state['last_quote'] = fetch_stock_quote(symbol)
            # Re-generate simulated data for the *new* symbol
            st.session_state['daily_data'] = generate_simulated_daily_data(symbol)
            st.rerun() # Rerun to display the new data

    if 'last_quote' in st.session_state and isinstance(st.session_state['last_quote'], dict):
        quote = st.session_state['last_quote']
        daily_df = st.session_state.get('daily_data')
          
        st.markdown("---")
        st.subheader(f"Quote for {quote.get('symbol', symbol)}")
          
        m_p, m_c, m_v = st.columns(3)
          
        m_p.metric("Current Price (₹)", f"₹{quote.get('price', 'N/A')}")
        m_c.metric("Change", quote.get('change', 'N/A'), delta_color="normal")
        m_v.metric("Volume", quote.get('volume', 'N/A'))
          
        st.caption(f"Last updated: {quote.get('last_updated', 'N/A')}")
          
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


# --- MAIN APPLICATION ENTRY POINT ---
if __name__ == '__main__':
    # Save database changes automatically
    st.session_state["DB"].save()