
# app.py — Personal Finance & Spending Analyzer (Gemini Smart V10 - Final Configuration)
# Login: username = prakriti11, password = ujjwal11

from __future__ import annotations

import os
import base64
import joblib
import json
import requests 
from io import BytesIO
from pathlib import Path
from datetime import datetime, timedelta

# Import the actual Gemini SDK client (requires: pip install google-genai)
try:
    from google import genai
    HAS_GEMINI_SDK = True
except ImportError:
    HAS_GEMINI_SDK = False

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import qrcode 

# --- API Keys and Configuration ---

def _read_gemini_key_from_file() -> str:
    """Reads the Gemini key directly from gemini_key.txt, bypassing the problematic secrets.toml"""
    key_file = Path("gemini_key.txt")
    if key_file.exists():
        return key_file.read_text(encoding="utf-8").strip()
    return "KEY_FILE_NOT_FOUND" 

# Read Gemini key using the custom function
GEMINI_API_KEY = _read_gemini_key_from_file() 
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

# --- UPI Details (Your Real QR Data) ---
UPI_ID = 'jaiswalprakriti26@okaxis'
UPI_PAYMENT_STRING = f'upi://pay?pa={UPI_ID}&pn=PRAKRITI&cu=INR'

# --- Personalized Information (New Chatbot Context) ---
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
    import qrcode  # noqa: F401
    HAS_QR = True
except Exception:
    HAS_QR = False

# ---------- Utility (Modified for QR Generation) ----------
def generate_placeholder_image(path: Path, size: int = 300, color: str = "pink", text: str = "QR Placeholder") -> None:
    if path.exists():
        return
    try:
        img = Image.new("RGB", (size, size), color=color)
        d = ImageDraw.Draw(img)
        d.text((size // 4, size // 2), text, fill=(0, 0, 0))
        img.save(path)
    except Exception:
        pass

def money(val: float) -> str:
    return f"₹{val:,.2f}"

def _img64(path: Path | None) -> str:
    """Return base64 string for an image file, or empty string if not available."""
    try:
        if not path or not path.exists():
            return ""
        with open(path, "rb") as fh:
            return base64.b64encode(fh.read()).decode("utf-8")
    except Exception:
        return ""

def _pick_qr_path() -> Path | None:
    """Prefer user-updated PNG, else JPG, else None."""
    if UPI_QR_IMG.exists():
        return UPI_QR_IMG
    if UPI_QR_IMG_JPG.exists():
        return UPI_QR_IMG_JPG
    return None

def _generate_default_upi_qr(upi_string: str, path: Path):
    """Generates the QR image for the user's UPI ID."""
    if not HAS_QR: return False
    try:
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=2,
        )
        qr.add_data(upi_string)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        img.save(path)
        return True
    except Exception as e:
        print(f"Error generating default QR: {e}")
        return False
        
def _save_uploaded_qr(file) -> str:
    """Save uploaded QR as PNG (more portable). This overrides the default QR."""
    try:
        img = Image.open(file).convert("RGB")
        img.save(UPI_QR_IMG) # overwrite/create upi_qr.png
        return "QR updated. If not visible, press 'Rerun' or refresh."
    except Exception as e:
        return f"Failed to save QR: {e}"

# Gemini API Utility (Integration) - FIX APPLIED
def gemini_query(prompt: str, history: list[tuple[str, str]], context: str) -> str:
    """Handles the intelligent response using the Gemini API."""
    
    if GEMINI_API_KEY == "KEY_FILE_NOT_FOUND":
         return "❌ **GEMINI KEY MISSING:** Please create 'gemini_key.txt' in your root folder and paste your key inside."

    if not HAS_GEMINI_SDK:
        return "⚠️ **GEMINI SDK Missing:** Cannot connect to the intelligent chatbot. Please run `pip install google-genai`."
        
    try:
        # Initialize client with API key
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        system_instruction = (
            "You are a versatile, professional AI financial advisor named PRAKRITI AI. "
            "Your persona is based on the following: " + context +
            "You must be able to answer finance questions, but also handle casual conversation, greetings, and nonsense questions gracefully. "
            "For finance queries, be concise (3-5 sentences). For casual queries, respond like a friendly assistant. "
            "If the user asks a casual question (like 'hi' or 'how are you'), use a simple, friendly response (e.g., 'I am fine, how are you?')."
        )
        
        # FIX: The system instruction is now prepended to the user prompt.
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
    # ... (Stock quote simulation remains the same) ...
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

# Simulation for Daily Stock Data (for charts) 
@st.cache_data
def generate_simulated_daily_data(symbol: str, days: int = 60) -> pd.DataFrame:
    # ... (Daily stock data simulation remains the same) ...
    symbol_upper = symbol.upper()
    
    if "TCS" in symbol_upper:
        base_price = 4000
    elif "RELIANCE" in symbol_upper:
        base_price = 2800
    elif "ITC" in symbol_upper:
        base_price = 420
    else:
        base_price = 450 + len(symbol_upper) * 10 

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


# ---------- KB helpers (omitted for brevity) ----------
def ensure_kb_exists(default_kb: list[str] | None = None) -> None:
    default_kb = default_kb or [
        "help - Type questions about expenses, income, trends (e.g., 'total expense', 'top categories')",
        "overview - Show project overview and advantages",
        "trend groceries - Show spending trend for groceries",
        "plot - Explain the current plot and data",
        "streak - Show current and longest saving streak",
    ]
    if not KB_FILE.exists():
        try:
            KB_FILE.write_text("\n".join(default_kb), encoding="utf-8")
        except Exception:
            pass

def kb_texts_from_file() -> list[str]:
    try:
        if not KB_FILE.exists():
            return []
        return [
            line.strip()
            for line in KB_FILE.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except Exception:
        return []

def build_vectorizer(texts: list[str]):
    vect = TfidfVectorizer(
        strip_accents="ascii",
        lowercase=True,
        analyzer="word",
        stop_words="english",
        token_pattern=r"\b\w+\b",
        max_df=1.0,
        min_df=1,
        binary=False,
    )
    try:
        if not texts:
            return None, None
        kb_mat = vect.fit_transform(texts)
        joblib.dump(vect, KB_VECT)
        return vect, kb_mat
    except Exception:
        return None, None

def load_vectorizer():
    try:
        if KB_VECT.exists():
            return joblib.load(KB_VECT)
    except Exception:
        pass
    return None

def tfidf_answer(query: str, vect, kb_texts: list[str], kb_mat, threshold: float = 0.1) -> str | None:
    try:
        q = vect.transform([query])
        sims = cosine_similarity(q, kb_mat)[0]
        best = int(np.argmax(sims))
        if sims[best] >= threshold:
            return kb_texts[best]
    except Exception:
        pass
    return None

# ---------- Data / Plot helpers (omitted for brevity) ----------
def to_excel_bytes(df: pd.DataFrame) -> bytes:
    out = BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="data")
    return out.getvalue()

def generate_sample(months: int = 6) -> pd.DataFrame:
    rng = pd.date_range(end=pd.Timestamp.today(), periods=months * 30)
    cats = [
        "groceries", "rent", "utilities", "entertainment", "transport",
        "health", "salary", "investment", "subscriptions", "dining",
        "gifts", "shopping",
    ]
    rows = []
    for d in rng:
        for _ in range(np.random.poisson(1.2)):
            cat = np.random.choice(
                cats,
                p=[0.14, 0.10, 0.08, 0.08, 0.12, 0.06, 0.05, 0.05, 0.10, 0.13, 0.05, 0.04],
            )
            if cat in ("salary", "investment"):
                t = "income"
                amt = abs(round(np.random.normal(1200 if cat == "salary" else 300, 90), 2))
            else:
                t = "expense"
                amt = abs(round(np.random.normal(50, 35), 2))
            rows.append(
                {"date": d.date(), "amount": amt, "category": cat, "description": f"{cat}", "type": t}
            )
    firsts = pd.date_range(end=pd.Timestamp.today(), periods=months, freq="MS")
    for d in firsts:
        rows.append(
            {"date": d.date(), "amount": 1500.0, "category": "salary", "description": "monthly salary", "type": "income"}
        )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

def read_file(file):
    if isinstance(file, (str, Path)):
        if str(file).endswith(".csv"):
            return pd.read_csv(file)
        return pd.read_excel(file)
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
        full = pd.date_range(g.index.min(), g.index.max(), freq="D").date
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
        "This app is an interactive *Personal Finance Dashboard* that visualizes expenses and income, computes saving streaks, and provides quick actionable insights.\n\n"
        "- *Interactive visualizations* help you spot trends and top spending categories quickly.\n"
        "- *Smart chatbot (powered by Gemini) and KB* allow generative financial advice and semantic lookups without exposing data externally.\n"
        "- Built-in *UPI/QR* and form workflow for easy logging.\n"
        "- *Lightweight* and runs locally — your data stays with you.\n"
    )

# ---------- Initial Setup (QR generation added) ----------
# Generate default QR if it doesn't exist
if not UPI_QR_IMG.exists():
    if not _generate_default_upi_qr(UPI_PAYMENT_STRING, UPI_QR_IMG):
        # Fallback to placeholder if qrcode generation fails
        generate_placeholder_image(UPI_QR_IMG, text="UPI QR (Error)") 

generate_placeholder_image(PROFILE_IMG, size=70, color="#25D366", text="Money Icon")
ensure_kb_exists()

PROFILE64 = _img64(PROFILE_IMG)

# ---------- CSS (Modified for Flicker and Motion) ----------
st.markdown(
    f"""
<style>
html, body, [data-testid="stAppViewContainer"] {{
  background: #0f1117; color: #eaeef6;
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
}}
section[data-testid="stSidebar"] {{ display: none !important; }}
.navbar {{ position: sticky; top: 0; z-index: 1000; padding: 12px 18px; margin: 0 0 18px 0;
  border-radius: 14px; background: radial-gradient(120% 120% at 0% 0%, #ffd9ea 0%, #ffcfe3 30%, rgba(255,255,255,0.08) 70%);
  box-shadow: 0 12px 30px rgba(255, 105, 180, 0.25), inset 0 0 60px rgba(255,255,255,0.25);
  border: 1px solid rgba(255,255,255,0.35); }}
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
@keyframes drop {{ 0%{{ transform: translateY(-60px) rotate(0deg); opacity:0 }}
  10%{{ opacity:1 }} 100%{{ transform: translateY(120px) rotate(360deg); opacity:0 }} }}
.card {{border-radius:16px; background:linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
  padding:16px; box-shadow: 0 12px 30px rgba(0,0,0,0.35); border: 1px solid rgba(255,255,255,0.12);}}
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
@keyframes fall {{ 0%{{ transform: translateY(-30px) rotate(0deg); opacity:0 }}
  15%{{ opacity:1 }} 100%{{ transform: translateY(85px) rotate(360deg); opacity:0 }} }}
.streak-metric{{ font-weight:800; font-size:26px }}
.badge-ok{{ background:#0ea5e9; color:white; padding:4px 10px; border-radius:999px; font-size:12px }}
.profile-wrap{{display:flex;align-items:center;justify-content:flex-end}}
.profile-pic{{
  width:70px;height:70px;border-radius:50%;object-fit:cover;
  box-shadow:0 6px 20px rgba(0,0,0,.35); border:2px solid #25D366;
}}
/* MODIFIED: Custom pink flicker and motion for QR */
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
@keyframes qr-flicker {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.9; }}
}}
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
  0%{{ transform:translateY(0) }}
  15%{{ transform:translateY(-8px) }}
  30%{{ transform:translateY(0) }}
  45%{{ transform:translateY(-5px) }}
  60%,100%{{ transform:translateY(0) }}
}}
.coin-rain {{
  position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: 10000;
  pointer-events: none; overflow: hidden;
  animation: fade-out 5s forwards;
}}
.coin-rain span{{
  position:absolute; top:-50px; font-size:22px; filter:drop-shadow(0 6px 8px rgba(0,0,0,.35));
  animation: rain 2.2s linear infinite;
}}
.coin-rain span:nth-child(1){{ left:8% ; animation-delay:.0s}}
.coin-rain span:nth-child(2){{ left:20%; animation-delay:.3s}}
.coin-rain span:nth-child(3){{ left:35%; animation-delay:.6s}}
.coin-rain span:nth-child(4){{ left:50%; animation-delay:.1s}}
.coin-rain span:nth-child(5){{ left:65%; animation-delay:.5s}}
.coin-rain span:nth-child(6){{ left:78%; animation-delay:.8s}}
.coin-rain span:nth-child(7){{ left:90%; animation-delay:.2s}}
@keyframes rain{{
  0%{{ transform:translateY(-60px) rotate(0deg); opacity:0 }}
  15%{{ opacity:1 }} 100%{{ transform:translateY(120vh) rotate(360deg); opacity:0 }}
}}
@keyframes fade-out {{ 0% {{ opacity: 1; visibility: visible; }} 90% {{ opacity: 1; visibility: visible; }} 100% {{ opacity: 0; visibility: hidden; }} }}
.device {{ border-radius: 18px; overflow: hidden; border: 1px solid rgba(255,255,255,.15);
  box-shadow: 0 16px 40px rgba(0,0,0,.55); background: #0b0f1a; }}
.device-top {{ height: 44px; background: linear-gradient(180deg,#141826,#0b0f1a);
  display:flex; align-items:center; justify-content:center; color:#cbd5e1; font-weight:700; letter-spacing:.5px;
  border-bottom: 1px solid rgba(255,255,255,.08); }}
.device-iframe {{ width: 100%; height: 720px; border: 0; }}
.robot-wrap{{display:flex;align-items:center;gap:12px;margin-bottom:8px}}
.robot{{font-size:36px;display:inline-block;filter:drop-shadow(0 8px 12px rgba(0,0,0,.45)); animation:robot-glow 3s ease-in-out infinite, robot-jump 3s ease-in-out infinite}}
.hi{{font-weight:900;color:#ffd9ea;padding:6px 10px;border-radius:10px;background:linear-gradient(90deg,#ff79b0,#ffb3d6);box-shadow:0 8px 30px rgba(255,90,150,0.12); animation:hi-flicker 2.6s linear infinite}}
@keyframes robot-glow{{0%{{text-shadow:0 0 6px #ffd9ea}}50%{{text-shadow:0 0 18px #ff79b0}}100%{{text-shadow:0 0 6px #ffd9ea}}}}
@keyframes robot-jump{{0%{{transform:translateY(0)}}15%{{transform:translateY(-8px)}}30%{{transform:translateY(0)}}45%{{transform:translateY(-5px)}}100%{{transform:translateY(0)}}}}
@keyframes hi-flicker{{0%{{opacity:1}}20%{{opacity:.3}}40%{{opacity:1}}60%{{opacity:.5}}80%{{opacity:1}}100%{{opacity:1}}}}
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Login (omitted for brevity) ----------
VALID_USER = "prakriti11"
VALID_PASS = "ujjwal11"

def _login_view() -> None:
    st.markdown(
        """
    <div class="navbar">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <div>
          <div class="nav-title">🔐 Finance Analyzer — Login</div>
          <div class="nav-sub">Enter your credentials to continue</div>
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
    with st.form("login_form", clear_on_submit=False):
        c1, c2 = st.columns([2, 1])
        with c1:
            u = st.text_input("Username", "")
            p = st.text_input("Password", "", type="password")
        with c2:
            st.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
            submit = st.form_submit_button("Login", use_container_width=True)
        if submit:
            if u == VALID_USER and p == VALID_PASS:
                st.session_state["auth_ok"] = True
                st.session_state["auth_user"] = u
                st.success("Login successful. Rerunning...")
                st.rerun()
            else:
                st.error("Invalid username or password.")

if "auth_ok" not in st.session_state:
    st.session_state["auth_ok"] = False
    st.session_state["auth_user"] = None

if not st.session_state["auth_ok"]:
    _login_view()
    st.stop()

# ---------- Post-Login (omitted for brevity) ----------
try:
    if STREAK_FILE.exists():
        _d = json.loads(STREAK_FILE.read_text(encoding="utf-8"))
        st.session_state.setdefault("longest_streak_ever", int(_d.get("longest_streak", 0)))
    else:
        st.session_state.setdefault("longest_streak_ever", 0)
except Exception:
    st.session_state.setdefault("longest_streak_ever", 0)

if "coin_rain_start" not in st.session_state:
    st.session_state["coin_rain_start"] = None
    st.session_state["coin_rain_show"] = False

if st.session_state["coin_rain_show"]:
    if datetime.now() > st.session_state["coin_rain_start"] + timedelta(seconds=RAIN_DURATION_SEC):
        st.session_state["coin_rain_show"] = False
        st.session_state["coin_rain_start"] = None

params = st.query_params if hasattr(st, "query_params") else st.experimental_get_query_params()
if params.get("rain", ["0"])[0] == "1":
    if not st.session_state["coin_rain_show"]:
        st.session_state["coin_rain_show"] = True
        st.session_state["coin_rain_start"] = datetime.now()

# ---------- Navbar (omitted for brevity) ----------
colA, colB = st.columns([4, 0.6])
with colA:
    st.markdown(
        """
    <div class="navbar">
      <div class="nav-title">💎 Personal Finance Dashboard</div>
      <div class="nav-sub">Visualize expenses, savings & investments — premium, Power BI–style UI</div>
      <div class="coin-wrap">
        <span class="coin">🪙</span><span class="coin">💰</span><span class="coin">🪙</span>
        <span class="coin">💰</span><span class="coin">🪙</span><span class="coin">💰</span><span class="coin">🪙</span>
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
with colB:
    if PROFILE64:
        st.markdown(
            f"""<div class="profile-wrap" title="Profile">
            <img class="profile-pic" src="data:image/jpg;base64,{PROFILE64}" />
        </div>""",
            unsafe_allow_html=True,
        )

# Check and warn if Gemini SDK is missing
if HAS_GEMINI_SDK:
    st.success("**Now integrated with GEMINI!** Access intelligent financial guidance via the Smart Chatbot.")
else:
    st.error("⚠️ **GEMINI SDK Missing:** Chatbot intelligence is disabled. Please run `pip install google-genai`.")

# ---------- Promise (omitted for brevity) ----------
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
    # ---------- Toolbar (omitted for brevity) ----------
    tb1, tb2, tb3, tb4, tb5, tb6 = st.columns([1.6, 1.4, 1.4, 1.8, 1.2, 1])
    with tb1:
        data_source = st.radio("Data source", ["Generate sample", "Upload CSV/Excel"], index=0, horizontal=True)
    with tb2:
        plot_type = st.selectbox(
            "Plot type",
            ["Line plot (trend)", "Bar plot (aggregate)", "Count plot (category counts)", "Scatter plot", "Distribution (KDE)", "Histogram"],
        )
    with tb3:
        group_period = st.selectbox("Group period", ["Monthly", "Weekly", "Daily"], index=0)
    with tb4:
        bar_mode = st.selectbox("Bar mode", ["By Category", "By Period (stacked by type)"], index=1 if plot_type.startswith("Bar") else 0)
    with tb5:
        numeric_col = st.selectbox("Numeric (scatter/hist)", ["amount"], index=0)
    with tb6:
        if st.button("Logout", key="logout_1"):
            for k in ("auth_ok", "auth_user", "chat_history", "virtual_transactions", "coin_rain_show", "coin_rain_start", "longest_streak_ever"):
                st.session_state.pop(k, None)
            st.rerun()

    # ---------- Load data (omitted for brevity) ----------
    uploaded = None
    if data_source.startswith("Upload"):
        uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

    if st.button("Generate sample dataset", key="generate_sample_1"):
        uploaded = None
        st.success("Sample will be generated on load (approx 6 months).")

    raw_df = None
    if data_source == "Generate sample" and uploaded is None:
        raw_df = generate_sample(6)
    elif uploaded is not None:
        try:
            raw_df = read_file(uploaded)
        except Exception as e:
            st.error(f"Error reading file: {e}. Ensure it's a valid CSV/Excel format.")
            raw_df = generate_sample(1)
    else:
        raw_df = generate_sample(6)

    if raw_df is None:
        st.stop()

    try:
        df = normalize(raw_df)
    except Exception as e:
        st.error(f"Error normalizing data: {e}. Please check column names.")
        st.stop()

    # ---------- Virtual deposits (omitted for brevity) ----------
    if "virtual_transactions" not in st.session_state:
        st.session_state["virtual_transactions"] = pd.DataFrame(columns=["date", "amount", "category", "description", "type"])
    vt = st.session_state["virtual_transactions"]
    if not vt.empty:
        vt2 = vt.copy()
        vt2["date"] = pd.to_datetime(vt2["date"]).dt.date
        df = pd.concat([df, vt2], ignore_index=True).sort_values("date").reset_index(drop=True)

    # ---------- Filters (omitted for brevity) ----------
    f1, f2, f3 = st.columns([1.3, 1.6, 1.1])
    if df.empty:
        st.info("No data available after loading/generation.")
        st.stop()

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

    # ---------- KPIs (omitted for brevity) ----------
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

    # ---------- Saving Streak (omitted for brevity) ----------
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
    if not dn.empty:
        dn_last = dn.tail(lookback).copy()
        thresh = target_daily if strict else max(1, target_daily * 0.6)
        hit = dn_last["net_saving"] >= thresh
        hit.index = pd.to_datetime(dn_last["day"])
        curr_streak, longest_streak = compute_streak(hit)

        try:
            prev_long = int(st.session_state.get("longest_streak_ever", 0))
            if longest_streak > prev_long:
                st.session_state["longest_streak_ever"] = int(longest_streak)
                with open(STREAK_FILE, "w", encoding="utf-8") as fh:
                    json.dump({"longest_streak": int(longest_streak), "updated_at": datetime.now().isoformat()}, fh)
        except Exception:
            pass

        pig_col, s1, s2, s3 = st.columns([1.1, 1, 1, 1.6])
        today_hit = bool(hit.iloc[-1]) if len(hit) > 0 and pd.to_datetime(hit.index[-1]).date() == pd.to_datetime("today").date() else False
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
            val_today = dn_last["net_saving"].iloc[-1] if len(dn_last) > 0 else 0
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

    # ---------- UPI QR + Coin Rain (Updated) ----------
    qr1, qr2 = st.columns([1, 2])

    with qr1:
        # UPDATED QR Logic with Flicker CSS
        st.markdown('<div class="upi-qr-wrap">', unsafe_allow_html=True)

        # Allow upload to override the default QR
        qr_upload = st.file_uploader("Replace QR (optional)", type=["png", "jpg", "jpeg"], key="qr_up")
        if qr_upload is not None:
            msg = _save_uploaded_qr(qr_upload)
            st.success(msg)

        qr_path = _pick_qr_path()
        if qr_path:
            # Display the UPI QR code image
            st.image(str(qr_path), caption=f"Scan & add ₹100 per week to make you smart! 🧠\nUPI ID: {UPI_ID}", use_container_width=True)
        else:
            st.info(f"QR not found. Using UPI ID: {UPI_ID}")

        st.markdown("</div>", unsafe_allow_html=True)
        
        # --- Arrow and Pop-up Message (Visual Enhancement) ---
        st.markdown("""
            <style>
                @keyframes pulsing_arrow {
                    0% { transform: scale(1) translateX(0px); opacity: 1; }
                    50% { transform: scale(1.1) translateX(10px); opacity: 0.8; }
                    100% { transform: scale(1) translateX(0px); opacity: 1; }
                }
                .callout-box {
                    background: #ff57a6; /* Pink */
                    color: white;
                    padding: 8px 12px;
                    border-radius: 8px;
                    font-weight: 600;
                    margin-top: 15px;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    animation: qr-glow 1.5s infinite alternate; /* Use existing glow for flicker/pulse */
                }
                .animated-arrow {
                    font-size: 24px;
                    animation: pulsing_arrow 1.5s infinite;
                    display: inline-block;
                }
            </style>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="callout-box">
                <span class="animated-arrow">➡️</span> 
                <span>Pop-up: Scan to achieve your savings goal!</span>
            </div>
        """, unsafe_allow_html=True)


        scan_amt = st.number_input("Amount scanned (₹)", min_value=1, value=100, step=1, key="scan_amount")
        if st.button(f"I scanned ₹{scan_amt} — Add to bucket", key="add_bucket_1"):
            new_row = {
                "date": pd.to_datetime("today").date(),
                "amount": float(scan_amt),
                "category": "collection",
                "description": "Scanned UPI payment",
                "type": "income",
            }
            st.session_state["virtual_transactions"] = pd.concat(
                [st.session_state["virtual_transactions"], pd.DataFrame([new_row])],
                ignore_index=True,
            )
            st.success(f"Added ₹{scan_amt} to bucket and triggered rain.")
            st.session_state["coin_rain_show"] = True
            st.session_state["coin_rain_start"] = datetime.now()
            components.html(f"<script>window.open('{FORM_URL}','_blank');</script>", height=0)
            st.rerun()

    with qr2:
        st.markdown("*Coin animation controls*")
        if st.button("Start coins (non-blocking)", key="start_coins_1"):
            st.session_state["coin_rain_show"] = True
            st.session_state["coin_rain_start"] = datetime.now()
            st.rerun()

        bucket_total = st.session_state["virtual_transactions"]["amount"].astype(float).sum()
        st.markdown(f"*Bucket total:* <span style='font-weight:700'>{money(bucket_total)}</span>", unsafe_allow_html=True)

        st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
        st.markdown(
            f"<a href='{FORM_URL}' target='_blank' style='text-decoration:none'>"
            f"<button style='background:#ff4da6;color:#fff;border:none;padding:10px 14px;border-radius:8px;font-weight:700;cursor:pointer'>"
            f"Google form Money Collection</button></a>",
            unsafe_allow_html=True,
        )

    # Show coin rain overlay (no infinite reruns)
    if st.session_state["coin_rain_show"]:
        st.markdown(
            """
        <div class="coin-rain">
          <span>🪙</span><span>💰</span><span>🪙</span>
          <span>💰</span><span>🪙</span><span>💰</span><span>🪙</span>
        </div>
        """,
            unsafe_allow_html=True,
        )
        if datetime.now() < st.session_state["coin_rain_start"] + timedelta(seconds=RAIN_DURATION_SEC + 0.5):
            st.rerun()

    # ---------- Main charts & table (omitted for brevity) ----------
    left, right = st.columns([3, 1])

    with left:
        st.subheader("Interactive chart")
        if tmp.shape[0] == 0:
            st.info("No data in current selection — adjust filters.")
        else:
            if plot_type.startswith("Line"):
                agg = tmp.groupby(["period", "type"])["amount"].sum().reset_index()
                fig = px.area(agg, x="period", y="amount", color="type", line_group="type", title=f"Trend by {group_period}")
            elif plot_type.startswith("Bar plot"):
                if bar_mode == "By Category":
                    bar = tmp.groupby("category")["amount"].sum().reset_index().sort_values("amount", ascending=False)
                    fig = px.bar(bar, x="category", y="amount", title=f"Spending by category ({group_period} selection)")
                else:
                    bar = tmp.groupby(["period", "type"])["amount"].sum().reset_index()
                    fig = px.bar(bar, x="period", y="amount", color="type", barmode="stack", title=f"Amount by {group_period} (stacked by type)")
            elif plot_type.startswith("Count plot"):
                cnt = tmp.groupby("category").size().reset_index(name="count").sort_values("count", ascending=False)
                fig = px.bar(cnt, x="category", y="count", title="Transaction counts by category")
            elif plot_type.startswith("Scatter"):
                fig = px.scatter(tmp, x="date", y="amount", color="category", hover_data=["description", "type"], title="Amount scatter over time")
            elif plot_type.startswith("Distribution"):
                data_kde = tmp[tmp["type"] == "expense"]["amount"]
                fig = px.histogram(data_kde, x="amount", nbins=40, histnorm="density", marginal="rug", title="Expense distribution (KDE approximation)")
            elif plot_type.startswith("Histogram"):
                fig = px.histogram(tmp, x="amount", nbins=40, title="Amount histogram")
            else:
                fig = px.scatter(tmp, x="date", y="amount", color="category", title="Chart")
            fig.update_layout(height=520, template="plotly_dark", legend_title="")
            st.plotly_chart(fig, use_container_width=True, key="main_chart_1")

        st.subheader("Transactions (filtered)")
        st.dataframe(view.sort_values("date", ascending=False).reset_index(drop=True), height=300)

    with right:
        st.subheader("Insights & Recommendations")
        top5 = view[view["type"] == "expense"].groupby("category")["amount"].sum().sort_values(ascending=False).head(5)
        if top5.shape[0] > 0:
            st.markdown("Top categories")
            for cat, val in top5.items():
                st.write(f"- {cat}: {money(val)}")

        st.markdown("### Quick budget diagnostics")
        if total_income > 0:
            disc = view[view["category"].isin(["entertainment", "dining", "subscriptions", "transport", "groceries", "shopping", "gifts"])]["amount"].sum()
            pct = disc / total_income if total_income > 0 else 0
            st.write(f"Discretionary / Income: *{pct:.0%}*")
            if pct > 0.3:
                st.warning("High discretionary spending — consider trimming subs or dining out.")
            else:
                st.success("Discretionary spending seems reasonable.")
        else:
            st.info("Add income data for better diagnostics.")

        st.markdown("---")
        st.subheader("Export")
        st.caption("Excel export temporarily disabled in Smart V3.")

        # ---- Smart Chatbot (Gemini Integrated) ----
        st.markdown("---")
        st.subheader("Smart Chatbot")
        
        # --- NEW Revolving Brain CSS ---
        st.markdown("""
            <style>
                @keyframes revolve {
                    0% { transform: rotate(0deg) scale(1); }
                    50% { transform: rotate(180deg) scale(1.05); }
                    100% { transform: rotate(360deg) scale(1); }
                }
                .revolving-brain {
                    font-size: 32px;
                    display: inline-block;
                    animation: revolve 3s linear infinite, qr-glow 2s infinite alternate;
                    color: #ffb3d6;
                    margin-left: 10px;
                }
            </style>
        """, unsafe_allow_html=True)
        
        if "thinking" in st.session_state and st.session_state["thinking"]:
             st.markdown('<div style="display:flex; align-items:center;">Thinking... <span class="revolving-brain">🧠</span></div>', unsafe_allow_html=True)
        # --- END Revolving Brain ---
        
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        with st.expander("Edit / Rebuild Knowledge Base (KB)"):
            kb_current = KB_FILE.read_text(encoding="utf-8") if KB_FILE.exists() else ""
            kb_edit = st.text_area("KB (one line = one entry)", value=kb_current, height=180, key="kb_edit_1")
            cA, cB = st.columns(2)
            if cA.button("Save KB", key="save_kb_1"):
                KB_FILE.write_text(kb_edit.strip(), encoding="utf-8")
                st.success("Saved KB.")
            if cB.button("Rebuild vectorizer", key="rebuild_vect_1"):
                kb_lines = [l.strip() for l in kb_edit.splitlines() if l.strip()]
                if len(kb_lines) == 0:
                    st.warning("Add KB entries first.")
                else:
                    build_vectorizer(kb_lines)
                    st.success("KB rebuilt.")

        kb_texts = kb_texts_from_file()
        vect = load_vectorizer()
        kb_mat = None
        if vect is None and len(kb_texts) > 0:
            vect, kb_mat = build_vectorizer(kb_texts)
        elif vect is not None and len(kb_texts) > 0:
            kb_mat = vect.transform(kb_texts)

        st.markdown('<div class="robot-wrap"><div class="robot">🤖</div><div class="hi">HI</div><div style="flex:1"></div></div>', unsafe_allow_html=True)

        user_q = st.text_input(
            "Ask (e.g., 'top categories this month', 'trend of groceries', 'help', 'invest advice')",
            key="chat_input",
        )

        if st.button("Send", key="send_1") and user_q:
            # Set thinking flag ON
            st.session_state["thinking"] = True
            
            ql = user_q.lower()
            ans = None
            
            # --- Personal Context for Gemini ---
            personal_context = (
                f"Team Name: {TEAM_INFO['Team Name']}. Leader: {TEAM_INFO['Team Leader']} (Idea 💡 behind this project). "
                f"Leader's Expertise: {TEAM_INFO['Leader Expertise']}. Frontend Developer: {TEAM_INFO['Frontend']}. "
                f"Guided by: {TEAM_INFO['Guidance']}. Contact: {TEAM_INFO['Contact']}. Email: {TEAM_INFO['Email']}. "
                f"Financial support UPI: {TEAM_INFO['Donate UPI']}."
            )
            
            # 1. Smart API / Gemini Call (Triggered by financial keywords)
            if HAS_GEMINI_SDK and any(k in ql for k in ["invest", "advice", "market", "recommend", "gemini"]):
                chat_history = st.session_state.get("chat_history", [])
                net_saving_proxy = float(view["amount"].sum())
                top_expenses_str = ", ".join([f"{k}: {money(v)}" for k, v in top5.items()])
                
                gemini_prompt = (
                    f"User Query: {user_q}\n"
                    f"Current Filtered Net Savings: {money(net_saving_proxy)}\n"
                    f"Top 5 Expenses in view: {top_expenses_str}\n"
                    "Based on the user's question, the current financial context (Net Savings, Top Expenses), and the last few chat turns, provide concise, generative financial advice. If the question is about stock/market trends, synthesize a response focusing on general Indian market sentiment, or prioritize savings advice if the net savings are low. Keep it under 5 sentences."
                )
                
                ans = gemini_query(gemini_prompt, chat_history, personal_context)

            # 2. Local Data Insights (Triggered by data/reporting keywords)
            if ans is None:
                def local_data_insight_answer(q: str, dfx: pd.DataFrame) -> str | None:
                    ql = q.lower()
                    if dfx is None or dfx.shape[0] == 0:
                        return None

                    if "top" in ql and ("categor" in ql or "spend" in ql):
                        s = dfx[dfx["type"] == "expense"].groupby("category")["amount"].sum().sort_values(ascending=False).head(7)
                        if s.empty:
                            return "No expense data in the selection."
                        return "*Top expense categories (Local Data):*\n" + "\n".join([f"- {k}: {money(v)}" for k, v in s.items()])
                    
                    if ("total" in ql or "what is" in ql) and "expense" in ql:
                        s = dfx[dfx["type"] == "expense"]["amount"].sum()
                        return f"*Total expense (Local Data):* {money(s)}"
                    
                    return None
                    
                ans = local_data_insight_answer(user_q, view)

            # 3. Special handlers (Plot, Overview, Advantages)
            if ans is None:
                if any(k in ql for k in ["plot", "graph", "explain", "describe", "trend", "visual", "chart"]):
                    ans = explain_plot_and_data(user_q, view, tmp, plot_type, group_period)
                elif any(k in ql for k in ["overview", "project overview", "explain project", "advantage", "why should i", "why use", "what can you do"]):
                    ans = project_overview_and_advantages()

            # 4. KB TF-IDF semantic match
            if ans is None and kb_mat is not None and vect is not None:
                ans = tfidf_answer(user_q, vect, kb_texts, kb_mat)

            # 5. Final Fallback / Smart AI Catch-all (FIX FOR IDIOT BOT)
            if ans is None:
                if HAS_GEMINI_SDK:
                    # If ALL local/KB checks fail, send the general query to Gemini for a generic, smart response
                    chat_history = st.session_state.get("chat_history", [])
                    
                    # Enhanced prompt to handle casual/nonsense/personal questions
                    if any(c in ql for c in ["hi", "hello", "hey", "how are you", "rowydee", "who are you", "tell me about yourself", "money"]):
                         gemini_prompt = f"User asked a casual question: '{user_q}'. You MUST answer politely and casually. If they ask who you are, provide the personalized team information provided in the context."
                    else:
                         gemini_prompt = f"User Query: {user_q}. The query did not match any finance topics. Answer the question as a helpful, knowledgeable assistant, linking keywords like 'teacher' or 'team' back to your context."
                    
                    ans = gemini_query(gemini_prompt, chat_history, personal_context)
                    
                    if ans.startswith("❌ GEMINI API Error"):
                        ans = "I couldn't find a local answer, and the Gemini AI is currently unavailable. Please check the API key setup."
                else:
                    # Non-smart fallback if SDK is missing
                    ans = "I couldn't find a direct answer. Try rephrasing or check the KB with *'help'*."
            
            # Set thinking flag OFF
            st.session_state["thinking"] = False

            st.session_state.chat_history.append(("You", user_q))
            st.session_state.chat_history.append(("Bot", ans))

        for speaker, msg in st.session_state.get("chat_history", [])[-12:]:
            if speaker == "You":
                st.markdown(f"*You:* {msg}")
            else:
                st.markdown(f"<div class='bot'>{msg}</div>", unsafe_allow_html=True)

    # ---------- Daily Savings Google Form (omitted for brevity) ----------
    st.markdown("### Daily Savings Form")
    fL, fR = st.columns([1.6, 1])

    with fL:
        components.html(
            f"""
        <div class="device">
          <div class="device-top">Daily Savings — Google Form</div>
          <iframe class="device-iframe" src="{FORM_URL}" allowtransparency="true"></iframe>
        </div>
        """,
            height=780,
            scrolling=True,
        )

    with fR:
        st.markdown("*Open on phone (scan)*")
        if HAS_QR:
            try:
                form_qr = qrcode.make(FORM_URL)
                b = BytesIO()
                form_qr.save(b, format="PNG")
                st.image(b.getvalue(), caption="Scan to open Google Form", width=220)
            except Exception:
                st.warning("QR code image generation failed.")

        st.markdown(f"[Open form in a new tab]({FORM_URL})")
        st.markdown("---")
        if st.button("I submitted today’s form — Celebrate 🎉", key="celebrate_1"):
            st.session_state["coin_rain_show"] = True
            st.session_state["coin_rain_start"] = datetime.now()
            st.rerun()

    st.caption(
        "Tip: run with LAN access for phone scanning → "
        "streamlit run app.py --server.address 0.0.0.0 --server.port 8501 and set APP_BASE_URL to your IP."
    )

with tab_stock:
    # --- New Stock Data Tab ---
    st.header("📈 Real-time Stock Data (AlphaVantage)")
    st.info("This feature uses the AlphaVantage API key to fetch real-time stock quotes. Historical charts are generated from simulated data.")
    
    col_sym, col_button = st.columns([2, 1])
    
    with col_sym:
        symbol = st.text_input("Enter Stock Symbol (e.g., TCS.BSE, RELIANCE.NSE)", value="ITC.BSE", key="stock_symbol_input").upper()
        
    with col_button:
        st.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
        if st.button("Fetch Quote & Charts", use_container_width=True, key="fetch_quote_charts_btn_2"):
            st.session_state['last_quote'] = fetch_stock_quote(symbol)
            st.session_state['daily_data'] = generate_simulated_daily_data(symbol)


    # Check if data exists in session state before trying to display it
    if 'last_quote' in st.session_state and isinstance(st.session_state['last_quote'], dict):
        quote = st.session_state['last_quote']
        daily_df = st.session_state.get('daily_data')
        
        # --- Metrics Row ---
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

        # --- Row 1: Line Chart & Portfolio Donut ---
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
            # Sample data for Donut/Pie Chart (simulating portfolio)
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
            
        # --- Row 2: Bar Chart ---
        st.markdown("---")
        st.markdown("#### Bar Chart: Last 60 Days Daily Volume")
        if daily_df is not None:
            fig_bar = px.bar(daily_df, x=daily_df.index, y='Volume',
                             title=f"Daily Volume for {symbol}",
                             labels={'Volume': 'Volume', 'Date': 'Date'})
            fig_bar.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
        
    else:
        # Initial display when no data has been fetched or an error occurred
        st.info("Enter a stock symbol and click 'Fetch Quote & Charts'.")
        # Placeholder for Donut/Pie 
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