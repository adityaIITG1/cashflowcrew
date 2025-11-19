# ui_patches.py

from __future__ import annotations
import streamlit as st
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# --- AI/OCR Imports (Moved to the top for robust access) ---
# Assuming these are available in your environment (ocr.py module)
try:
    from ocr import extract_bill_fields, HAS_TESSERACT
except ImportError:
    HAS_TESSERACT = False
    def extract_bill_fields(*args, **kwargs):
        return None
# ------------------------------------------------------------


# --- Helper for Money Formatting ---
def money(v: float) -> str:
    """Formats a float as an Indian Rupee string."""
    # Ensure v is treated as an absolute value for the thousands comma check if it's an integer
    abs_v = abs(v)
    if abs_v >= 1000 and v.is_integer():
        return f"₹{v:,.0f}"
    return f"₹{v:,.2f}"

# --- Health Score and Mood Avatar ---

def display_health_score(health_data: Dict[str, Any]) -> None:
    """Renders the Financial Health Score meter and mood avatar with safe access."""
    
    # Use .get() with a safe default value to prevent KeyError
    score = health_data.get('score', 50)
    factors = health_data.get('factors', {})

    # Determine mood avatar and theme color based on score
    if score >= 80:
        avatar = "😊💰" # Happy money face
        mood_text = "Excellent Health!"
        color = "#22c55e" # Green
    elif score >= 60:
        avatar = "🙂👍" # Neutral/Good
        mood_text = "Good Health, keep it up."
        color = "#0ea5e9" # Blue
    elif score >= 40:
        avatar = "😐🤔" # Neutral/Concerned
        mood_text = "Okay, but check your spending."
        color = "#f97316" # Orange
    else:
        avatar = "😟💸" # Worried wallet
        mood_text = "Urgent attention needed!"
        color = "#ef4444" # Red
        
    st.markdown(f"#### {avatar} Health Score: {score}/100")
    st.caption(mood_text)

    # Circular progress bar (using Plotly for better aesthetics)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': "Fin. Health"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': color},
            'bgcolor': "rgba(255,255,255,0.05)",
            'steps': [
                {'range': [0, 40], 'color': "rgba(239, 68, 68, 0.4)"}, # Red
                {'range': [40, 60], 'color': "rgba(249, 115, 22, 0.4)"}, # Orange
                {'range': [60, 80], 'color': "rgba(14, 165, 233, 0.4)"}, # Blue
                {'range': [80, 100], 'color': "rgba(34, 197, 94, 0.4)"} # Green
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    # Using 'plotly_dark' is good for the dark theme you are building.
    fig.update_layout(height=200, margin={'t':10, 'b':10, 'l':20, 'r':20}, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # Breakdown (using two columns for space)
    st.markdown("##### Factor Breakdown")
    f1, f2 = st.columns(2)
    # Applied .get() for robust access
    f1.metric("Savings Rate", f"{factors.get('savings_rate', 0)}%")
    f2.metric("Income Consistency", f"{factors.get('income_consistency', 0)}%")
    f1.metric("Expense Volatility", f"{factors.get('expense_volatility', 0)}%")
    f2.metric("Budget Adherence", f"{factors.get('budget_adherence', 0)}%")
    st.caption(f"No-Spend Streak: {factors.get('no_spend_streak', 0)} days. (Longest: {factors.get('longest_no_spend', 0)} days)")


# --- Badges ---
def display_badges(no_spend_streak_days: int) -> None:
    """Renders professional gradient badges based on no-spend streak."""
    
    st.markdown("##### Streak Achievements")
    
    badge_html = ""
    if no_spend_streak_days >= 30:
        badge_html += '<span class="badge badge-diamond">💎 30+ Day Saver</span>'
    elif no_spend_streak_days >= 20:
        badge_html += '<span class="badge badge-gold">🥇 20+ Day Saver</span>'
    elif no_spend_streak_days >= 10:
        badge_html += '<span class="badge badge-silver">🥈 10+ Day Saver</span>'
    elif no_spend_streak_days >= 5:
        badge_html += '<span class="badge badge-bronze">🥉 5+ Day Saver</span>'
    
    # Enhanced CSS for a more professional, darker theme look
    st.markdown("""
    <style>
    .badge {
        display: inline-block; padding: 4px 8px; margin: 4px 2px; border-radius: 999px; 
        font-size: 12px; font-weight: 700; color: #f9fafb; /* Light text */
        box-shadow: 0 2px 5px rgba(0,0,0,0.4);
    }
    .badge-diamond { 
        background: linear-gradient(45deg, #0ea5e9, #8a2be2); /* Blue-Purple Gradient */
        color: white; 
        border: 1px solid #c7d2fe; 
    }
    .badge-gold { 
        background-color: #FFD700; 
        color: #1f2a40; /* Dark text on gold */
    }
    .badge-silver { background-color: #A9A9A9; }
    .badge-bronze { background-color: #CD7F32; }
    </style>
    """, unsafe_allow_html=True)

    if badge_html:
        st.markdown(f"<div>{badge_html}</div>", unsafe_allow_html=True)
    else:
        st.caption("No-spend badges start at 5 days. Keep saving! 🎯")


# --- Budget Bot Mini Card ---
def budget_bot_minicard(budget_allocation: Dict[str, float]) -> Tuple[Dict[str, float], bool]:
    """
    Renders the Auto-Allocate Budget Mini Card with an editable grid.
    
    Returns:
        Tuple[Dict[str, float], bool]: (Updated budget dict, whether to apply/save).
    """
    st.markdown("#### 🤖 Budget Bot: Auto-Allocate")
    
    # Create a mutable copy and handle savings target
    budget_copy = budget_allocation.copy()
    # Use .get() for safe access to 'Savings Target'
    savings_target = budget_copy.pop('Savings Target', 0)
    
    # Prepare DataFrame for editing
    df_budget = pd.DataFrame(list(budget_copy.items()), columns=['Category', 'Budget (₹)'])
    df_budget['Budget (₹)'] = df_budget['Budget (₹)'].astype(int)
    df_budget = df_budget[df_budget['Budget (₹)'] > 0] # Filter out 0 allocations

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
        key="budget_editor"
    )

    # Re-assemble the budget dict
    updated_budget = {row['Category'].lower(): float(row['Budget (₹)']) for _, row in edited_df.iterrows()}
    updated_budget['Savings Target'] = savings_target
    
    # Button to confirm and save
    if st.button("✅ Apply Budgets to Profile", use_container_width=True, help="This saves the new monthly budgets to your profile (session state)"):
        return updated_budget, True
        
    return updated_budget, False


# --- OCR Upload Box ---
def glowing_ocr_uploader() -> Tuple[Any, Optional[Dict[str, Any]]]:
    """
    Renders a glowing OCR upload box and a pre-filled transaction form on success.
    
    Returns:
        Tuple[st.file_uploader_result, Dict|None]: (File uploader object, Extracted data).
    """
    # Assuming extract_bill_fields and HAS_TESSERACT are now available via the top-level imports
    global extract_bill_fields, HAS_TESSERACT
    
    # CSS for the glowing box - Adjusted for a darker theme's visibility
    st.markdown("""
    <style>
    .ocr-upload-box {
        padding: 15px; border-radius: 12px; margin-bottom: 15px;
        background: rgba(138, 43, 226, 0.1); /* Purple background (matches buttons) */
        border: 2px solid rgba(138, 43, 226, 0.5);
        box-shadow: 0 0 15px rgba(138, 43, 226, 0.7), inset 0 0 10px rgba(138, 43, 226, 0.5);
        animation: qr-glow 2s infinite alternate; 
    }
    /* Define/ensure qr-glow is present if used */
    @keyframes qr-glow {
        0% { box-shadow: 0 0 10px rgba(138, 43, 226, 0.4); }
        100% { box-shadow: 0 0 20px rgba(138, 43, 226, 0.8); }
    }
    </style>
    """, unsafe_allow_html=True)
    
    extracted_data = None
    
    st.markdown('<div class="ocr-upload-box">', unsafe_allow_html=True)
    st.markdown("##### 📸 Scan Bill (OCR Powered)")
    
    if not HAS_TESSERACT:
        st.warning("⚠️ OCR (Tesseract) is not installed/configured. Feature disabled.")
        uploaded_file = st.file_uploader("Upload Image (PNG/JPG)", type=['png', 'jpg', 'jpeg'], key="ocr_uploader", disabled=True)
    else:
        uploaded_file = st.file_uploader("Upload Image (PNG/JPG)", type=['png', 'jpg', 'jpeg'], key="ocr_uploader")
        st.caption("Upload a receipt/bill image to auto-fill the transaction form.")

        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            
            # Call extract_bill_fields safely
            try:
                extracted_data = extract_bill_fields(file_bytes)
            except Exception as e:
                 # Catching any general exception during OCR processing
                 st.error(f"Error during OCR processing: {e}")
                 uploaded_file = None
                 return uploaded_file, None
            
            if extracted_data is None:
                st.error("❌ OCR failed. Tesseract failed to extract data. Try a clearer image.")
            else:
                st.success("✅ OCR Success! Fields pre-filled below.")
                
    st.markdown('</div>', unsafe_allow_html=True)
    
    return uploaded_file, extracted_data
