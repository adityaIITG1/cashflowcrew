
# ui_patches.py

from __future__ import annotations
import streamlit as st
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# ==============================================================================
# BASE UTILITIES AND CSS ENFORCEMENT
# ==============================================================================

def ensure_once_css(key: str, css: str) -> None:
    """Inject CSS only once per session (avoid duplicates)."""
    flag = f"_css_{key}"
    if not st.session_state.get(flag):
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
        st.session_state[flag] = True

# --- Helper for Money Formatting ---
def money(v: float | int | str) -> str:
    """Formats a number as an Indian Rupee string (₹)."""
    
    try:
        # 1. Cast input to float to handle various numeric types (int, numpy scalars, etc.)
        v_float = float(v)
        
        # 2. Check if the float is mathematically equivalent to an integer
        is_integer_like = (v_float == round(v_float))
        
        # 3. Apply the formatting rule
        # Rule: If value is >= 1000 AND integer-like, format with 0 decimals.
        if abs(v_float) >= 1000 and is_integer_like:
            return f"₹{v_float:,.0f}"
            
        # Default: Format with 2 decimal places.
        return f"₹{v_float:,.2f}"
        
    except (ValueError, TypeError):
        # 4. Defensive Fallback: Return a safe string if conversion fails
        return "₹N/A"

# --- Consolidated CSS ---
BADGE_CSS = """
.badge {
    display: inline-block; padding: 4px 8px; margin: 2px; border-radius: 999px;
    font-size: 11px; font-weight: 700; color: #0f1117; text-shadow: 0 0 1px rgba(0,0,0,0.2);
}
.badge-diamond { background: linear-gradient(45deg, #0ea5e9, #ff79b0); color: white; border: 1px solid white; box-shadow: 0 0 8px #ff79b0; }
.badge-gold { background-color: #ffd700; }
.badge-silver { background-color: #c0c0c0; }
.badge-bronze { background-color: #cd7f32; }
.health-avatar { display: inline-block; animation: sparkle 1.5s infinite alternate; }
@keyframes sparkle { 0% { transform: scale(1); text-shadow: none; } 100% { transform: scale(1.1); text-shadow: 0 0 8px rgba(255, 255, 255, 0.7); } }
"""
OCR_BOX_CSS = """
.ocr-upload-box {
    padding: 10px; border-radius: 12px; margin-bottom: 10px;
    background: rgba(255, 105, 180, 0.1);
    border: 2px solid rgba(255, 105, 180, 0.5);
    box-shadow: 0 0 10px rgba(255, 105, 180, 0.7), inset 0 0 10px rgba(255, 105, 180, 0.5);
    /* Animation removed from here to avoid redundancy with app.py CSS, 
       but left the glow effect via box-shadow. */
}
"""

# ==============================================================================
# MAIN UI PATCH FUNCTIONS
# ==============================================================================

# --- Health Score and Mood Avatar ---
def display_health_score(health_data: Dict[str, Any]) -> None:
    """Renders the Financial Health Score meter and mood avatar with enhanced visuals."""
    ensure_once_css('badges', BADGE_CSS) # Ensure sparkle CSS is loaded

    score = health_data.get('score', 50)
    factors = health_data.get('factors', {})

    # Determine mood avatar
    if score >= 80:
        avatar = "😊💰"
        mood_text = "Excellent Health! Strong control over finances."
        color = "#22c55e" # Green
    # ... (rest of avatar logic is kept same)
    elif score >= 60:
        avatar = "🙂👍"
        mood_text = "Good Health, keep it up. Monitor volatile expenses."
        color = "#0ea5e9" # Blue
    elif score >= 40:
        avatar = "😐🤔"
        mood_text = "Okay, but check your spending. Budget adherence needs work."
        color = "#f97316" # Orange
    else:
        avatar = "😟💸"
        mood_text = "Urgent attention needed! Focus on savings rate."
        color = "#ef4444" # Red

    st.markdown(f"#### <span class='health-avatar'>{avatar}</span> Financial Health Index", unsafe_allow_html=True)
    st.caption(f"Score: **{score}/100** ({mood_text})")

    # Circular progress bar (Plotly Gauge is already high-quality, only layout tweak)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': "Fin. Health"},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': color},
            'bgcolor': "rgba(255,255,255,0.05)",
            'steps': [
                {'range': [0, 40], 'color': "rgba(239, 68, 68, 0.4)"},
                {'range': [40, 60], 'color': "rgba(249, 115, 22, 0.4)"},
                {'range': [60, 80], 'color': "rgba(14, 165, 233, 0.4)"},
                {'range': [80, 100], 'color': "rgba(34, 197, 94, 0.4)"}
            ],
            'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': 70}
        }
    ))
    fig.update_layout(height=200, margin={'t': 10, 'b': 10, 'l': 20, 'r': 20}, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # Breakdown metrics (using rupee formatter for a bit more detail)
    st.markdown("##### Factor Breakdown")
    f1, f2 = st.columns(2)
    f1.metric("Savings Rate", f"{factors.get('savings_rate', 0):.1f}%")
    f2.metric("Income Consistency", f"{factors.get('income_consistency', 0):.1f}%")
    f1.metric("Expense Volatility", f"{factors.get('expense_volatility', 0):.1f}%")
    f2.metric("Budget Adherence", f"{factors.get('budget_adherence', 0):.1f}%")
    st.caption(f"No-Spend Streak: {factors.get('no_spend_streak', 0)} days.")


# --- Badges ---
def display_badges(no_spend_streak_days: int) -> None:
    """Renders badges based on no-spend streak."""
    ensure_once_css('badges', BADGE_CSS)
    st.markdown("##### Streak Badges")

    badge_html = ""
    # Ensure only the highest applicable badge is shown for simplicity
    if no_spend_streak_days >= 30:
        badge_html += '<span class="badge badge-diamond">💎 30+ Day Saver</span>'
    elif no_spend_streak_days >= 20:
        badge_html += '<span class="badge badge-gold">🥇 20+ Day Saver</span>'
    elif no_spend_streak_days >= 10:
        badge_html += '<span class="badge badge-silver">🥈 10+ Day Saver</span>'
    elif no_spend_streak_days >= 5:
        badge_html += '<span class="badge badge-bronze">🥉 5+ Day Saver</span>'

    if badge_html:
        st.markdown(f"<div>{badge_html}</div>", unsafe_allow_html=True)
    else:
        st.caption("No-spend badges start at 5 days. Keep saving! 🎯")


# --- Budget Bot Mini Card ---
def budget_bot_minicard(budget_allocation: Dict[str, float]) -> Tuple[Dict[str, float], bool]:
    """
    Renders the Auto-Allocate Budget Mini Card with an editable grid.
    Returns: (Updated budget dict, whether to apply/save).
    """
    st.markdown("#### 🤖 Budget Bot: Auto-Allocate")

    budget_copy = budget_allocation.copy()
    
    savings_target = budget_copy.pop('Savings', budget_copy.pop('Savings Target', 0))

    df_budget = pd.DataFrame(list(budget_copy.items()), columns=['Category', 'Budget (₹)'])
    df_budget['Category'] = df_budget['Category'].str.capitalize()
    df_budget['Budget (₹)'] = df_budget['Budget (₹)'].astype(int)
    df_budget = df_budget[df_budget['Budget (₹)'] > 0] 

    st.markdown(f"**Target Savings:** {money(savings_target)} / month 🎯")

    # Editable data editor
    edited_df = st.data_editor(
        df_budget,
        column_config={
            "Budget (₹)": st.column_config.NumberColumn(
                "Budget (₹)",
                help="Recommended monthly expense budget per category.",
                format="₹%d",
                min_value=0,
                step=100
            ),
            "Category": st.column_config.TextColumn(disabled=True)
        },
        height=200,
        hide_index=True,
        key="budget_editor_ui"
    )

    # Re-assemble the budget dict (keys are converted back to lowercase)
    updated_budget = {row['Category'].lower(): float(row['Budget (₹)']) for _, row in edited_df.iterrows()}
    updated_budget['savings'] = savings_target

    if st.button("✅ Apply Budgets to Profile", use_container_width=True, help="This saves the new monthly budgets to your profile (session state)"):
        return updated_budget, True

    return updated_budget, False


# --- OCR Upload Box ---
def glowing_ocr_uploader() -> Tuple[Any, Optional[Dict[str, Any]]]:
    """
    Renders a glowing OCR upload box with enhanced feedback.
    Returns: (File uploader object, Extracted data).
    """
    try:
        from ocr import extract_bill_fields, HAS_TESSERACT
    except ImportError:
        class MockOCR:
            HAS_TESSERACT = False
            def extract_bill_fields(self, *args): return None
        MockOCR = MockOCR()
        extract_bill_fields = MockOCR.extract_bill_fields
        HAS_TESSERACT = MockOCR.HAS_TESSERACT

    ensure_once_css('ocr_box', OCR_BOX_CSS)
    extracted_data = None
    uploaded_file = None

    st.markdown('<div class="ocr-upload-box">', unsafe_allow_html=True)
    st.markdown("##### 📸 Scan Bill (OCR Powered)")

    if not HAS_TESSERACT:
        st.error("❌ OCR (Tesseract) is not configured. Install Tesseract to enable bill scanning.")
    else:
        uploaded_file = st.file_uploader("Upload Receipt Image (PNG/JPG)", type=['png', 'jpg', 'jpeg'], key="ocr_uploader")
        st.caption("Upload a receipt/bill image to auto-fill the transaction form.")

        if uploaded_file is not None:
            # Display file name or processing message
            st.info(f"File selected: **{uploaded_file.name}**. Processing...")
            
            with st.spinner("Analyzing image via Tesseract..."):
                file_bytes = uploaded_file.read()
                extracted_data = extract_bill_fields(file_bytes)

            if extracted_data is None:
                st.error("❌ OCR failed. Could not extract meaningful data. Try a clearer image.")
            else:
                st.success(f"✅ OCR Success! Found **{money(extracted_data.get('amount', 0))}** for **{extracted_data.get('merchant', 'N/A')}**.")

    st.markdown('</div>', unsafe_allow_html=True)

    return uploaded_file, extracted_data

# ==============================================================================
# ADD-ON HELPERS (Kept for completeness)
# ==============================================================================

def rupee(n: float) -> str:
    """Indian numbering format: e.g. 12,34,567.89 with ₹."""
    try:
        n = float(n)
        s = f"{n:.2f}"
        if "." in s:
            whole, frac = s.split(".")
        else:
            whole, frac = s, "00"
        neg = whole.startswith("-")
        if neg:
            whole = whole[1:]
        
        if len(whole) > 3:
            first = whole[-3:]
            rest = whole[:-3]
            groups = []
            while rest:
                groups.insert(0, rest[-2:])
                rest = rest[:-2]
            whole = ",".join(groups + [first])
        
        sign = "-" if neg else ""
        return f"{sign}₹{whole}.{frac}"
    except Exception:
        return f"₹{n}"

def metric_pill(label: str, value: str, *, help: Optional[str] = None) -> None:
    """Compact, pretty chip for KPIs."""
    ensure_once_css('pill', """
     .pill{display:inline-flex;gap:.5rem;align-items:center;padding:.25rem .6rem;border-radius:999px;
            background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.15);font-weight:600}
     .pill .k{opacity:.8}
    """)
    h = f' title="{help}"' if help else ""
    st.markdown(f'<span class="pill"{h}><span class="k">{label}</span><span>{value}</span></span>', unsafe_allow_html=True)

def display_health_score_compact(health_data: Dict[str, Any]) -> None:
    """Smaller progress-style version of the health score (use when space is tight)."""
    score = int(health_data.get('score', 50))
    st.progress(min(max(score, 0), 100), text=f"Health {score}/100")

def show_ocr_preview(extracted: Optional[Dict[str, Any]]) -> None:
    """Optional: pretty preview for OCR dict."""
    if not extracted:
        return
    df = pd.DataFrame([extracted]).T.reset_index()
    df.columns = ["Field", "Value"]
    st.dataframe(df, hide_index=True, use_container_width=True)