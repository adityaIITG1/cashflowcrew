
# ai_helpers.py
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

# Why: provide a consistent fallback viz even when Gemini is off
import plotly.graph_objects as go

# ===== Gemini SDK detection =====
try:
    from google import genai  # pip install -U google-genai
    _HAS_GEMINI_SDK = True
    _SDK_IMPORT_ERR: Optional[Exception] = None
except Exception as _e:
    _HAS_GEMINI_SDK = False
    _SDK_IMPORT_ERR = _e  # keep for diagnostics

_DEFAULT_MODELS = [
    "gemini-2.5-flash",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
]

def _read_key() -> str:
    return (os.getenv("GEMINI_API_KEY") or "").strip()

def gemini_enabled() -> Tuple[bool, str]:
    """Quick check to show a clear reason when disabled."""
    if not _HAS_GEMINI_SDK:
        return False, f"google-genai not installed: {_SDK_IMPORT_ERR!r}"
    key = _read_key()
    if not key:
        return False, "GEMINI_API_KEY not set"
    return True, "OK"

def _mk_client() -> "genai.Client":
    ok, reason = gemini_enabled()
    if not ok:
        raise RuntimeError(f"Gemini disabled: {reason}")
    return genai.Client(api_key=_read_key())

# ===== Chat / text generation =====
def smart_gemini_generate(
    prompt: str,
    *,
    system_instruction: Optional[str] = None,
    model_order: Optional[List[str]] = None,
    json_response: bool = False,
) -> Dict[str, str]:
    """
    Returns: {"ok": bool, "text": str, "model": str, "error": str}
    Why: stable wrapper with model fallback + actionable errors.
    """
    model_order = model_order or _DEFAULT_MODELS
    try:
        client = _mk_client()
    except Exception as e:
        return {"ok": False, "text": "", "model": "", "error": str(e)}

    sys_prompt = (system_instruction or "").strip()
    user_prompt = prompt.strip()
    if json_response:
        # Enforce JSON-only outputs (helps when you parse)
        user_prompt = (
            "Return ONLY a valid JSON object. No prose, no code fences.\n\n"
            + user_prompt
        )

    contents = []
    if sys_prompt:
        contents.append({"role": "user", "parts": [{"text": sys_prompt}]})
    contents.append({"role": "user", "parts": [{"text": user_prompt}]})

    last_err: Optional[Exception] = None
    for m in model_order:
        try:
            resp = client.models.generate_content(model=m, contents=contents)
            text = (getattr(resp, "text", None) or "").strip()
            if not text:
                raise RuntimeError(f"Model '{m}' returned empty text")
            return {"ok": True, "text": text, "model": m, "error": ""}
        except Exception as e:
            last_err = e
            continue

    return {
        "ok": False,
        "text": "",
        "model": "",
        "error": f"All models failed ({', '.join(model_order)}). Last error: {last_err}",
    }

def chat_reply(prompt: str, context: str = "", history: Optional[List[Tuple[str, str]]] = None) -> str:
    """
    Drop-in replacement for your gemini_query(); formats a friendly response.
    """
    system_instruction = (
        "You are a concise, proactive AI financial assistant with a friendly tone and emojis. "
        + (context or "")
    )
    out = smart_gemini_generate(prompt, system_instruction=system_instruction)
    if out["ok"]:
        return f"🧠 *Gemini Smart AI ({out['model']}):* {out['text']}"
    return f"❌ **GEMINI Error:** {out['error']}"

# ===== Generative viz (with automatic fallback) =====
def _mk_fallback_figure(summary: Dict[str, Any]) -> go.Figure:
    """Why: keep UI working without Gemini; simple, readable donut + net bar."""
    income = float(summary.get("total_income", 0.0))
    expense = float(summary.get("total_expense", 0.0))
    net = float(summary.get("net_savings", income - expense))

    fig = go.Figure()
    fig.add_trace(
        go.Pie(
            labels=["Income", "Expense"],
            values=[max(income, 0.0), max(expense, 0.0)],
            hole=0.55,
            name="Flow",
            textinfo="percent+label",
            hovertemplate="%{label}: ₹%{value:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(go.Bar(x=["Net"], y=[net], name="Net", hovertemplate="Net: ₹%{y:,.0f}<extra></extra>"))
    fig.update_layout(title="Total Financial Flow (Fallback)", barmode="overlay", legend_title="", height=420)
    return fig

def gen_viz_spec(summary: Dict[str, Any], spikes: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Returns always:
      {"description": str, "spec": <plotly_figure_dict>, "fallback": bool}
    If Gemini is unavailable or fails, returns a clean fallback (no error dicts).
    """
    spikes = spikes or []
    ok, reason = gemini_enabled()
    if not ok:
        fig = _mk_fallback_figure(summary)
        return {"description": "Visualizing key financial metrics.", "spec": fig.to_dict(), "fallback": True}

    # Try to get a Plotly figure spec JSON from Gemini
    prompt = (
        "You are a data viz model. Output ONLY a valid JSON object that matches a Plotly Figure.to_dict(). "
        "No prose, no code fences. Create an insightful figure comparing income vs expense and optionally top categories.\n\n"
        f"SUMMARY: {json.dumps(summary, ensure_ascii=False)}\n"
        f"SPIKES: {json.dumps(spikes, ensure_ascii=False)}\n"
    )
    out = smart_gemini_generate(prompt, json_response=True)
    if not out["ok"]:
        fig = _mk_fallback_figure(summary)
        return {"description": "Visualizing key financial metrics.", "spec": fig.to_dict(), "fallback": True}

    text = out["text"]
    # Remove accidental fences like ```json
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[text.find("\n") + 1 :].strip()

    try:
        spec = json.loads(text)
        if not isinstance(spec, dict) or not spec:
            raise ValueError("non-dict/empty spec")
        return {
            "description": "AI-generated infographic based on your current data.",
            "spec": spec,
            "fallback": False,
        }
    except Exception:
        fig = _mk_fallback_figure(summary)
        return {"description": "Visualizing key financial metrics.", "spec": fig.to_dict(), "fallback": True}

# ===== Optional: quick diagnostics you can print in a panel =====
def diagnostics() -> Dict[str, Any]:
    ok, reason = gemini_enabled()
    return {
        "sdk_imported": _HAS_GEMINI_SDK,
        "sdk_import_error": repr(_SDK_IMPORT_ERR) if _SDK_IMPORT_ERR else "",
        "api_key_present": bool(_read_key()),
        "enabled": ok,
        "reason": reason,
        "default_models": _DEFAULT_MODELS,
    }
