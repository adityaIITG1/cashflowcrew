
# File: telegram_utils.py
from __future__ import annotations
import os
from typing import Tuple, Optional
import requests

# --- Fallbacks (you provided these; rotate if exposed) ---
FALLBACK_TELEGRAM_BOT_TOKEN = "8553931141:AAETBKCN1jCYub3Hf7BZ1ylS3izMB5EDzII"
FALLBACK_TELEGRAM_CHAT_ID = "6242960424"

ENV_TOKEN_KEY = "TELEGRAM_BOT_TOKEN"
ENV_CHAT_KEY = "TELEGRAM_CHAT_ID"


def _read_from_streamlit_secrets() -> tuple[Optional[str], Optional[str]]:
    """Why: allow Streamlit Cloud/locals via secrets.toml without code changes."""
    try:
        import streamlit as st  # local import to avoid hard dependency
    except Exception:
        return None, None

    token = st.secrets.get(ENV_TOKEN_KEY) if hasattr(st, "secrets") else None
    chat_id = st.secrets.get(ENV_CHAT_KEY) if hasattr(st, "secrets") else None

    tg = st.secrets.get("telegram") if hasattr(st, "secrets") else None
    if isinstance(tg, dict):
        token = token or tg.get("bot_token")
        chat_id = chat_id or tg.get("chat_id")

    return token, chat_id


def get_telegram_cfg() -> tuple[str, str]:
    """Resolve token/chat id with precedence: env → secrets → fallbacks."""
    token = os.getenv(ENV_TOKEN_KEY)
    chat_id = os.getenv(ENV_CHAT_KEY)

    if not token or not chat_id:
        s_token, s_chat = _read_from_streamlit_secrets()
        token = token or s_token
        chat_id = chat_id or s_chat

    token = token or FALLBACK_TELEGRAM_BOT_TOKEN
    chat_id = chat_id or FALLBACK_TELEGRAM_CHAT_ID
    return str(token), str(chat_id)


def _error_hint(status: int, description: str) -> str:
    if status == 401:
        return "Invalid/revoked bot token."
    if status == 403:
        return "Bot not allowed in chat or wrong chat id."
    if "chat not found" in description.lower():
        return "Chat not found; add bot to chat or use numeric id."
    return "See logs for details."


def send_report_png(img_bytes: bytes, caption: str) -> Tuple[bool, str]:
    """
    Sends a PNG image report to Telegram using sendPhoto.
    Falls back to provided token/id if env/secrets are not set.
    """
    bot_token, chat_id = get_telegram_cfg()
    if not bot_token or not chat_id:
        return False, f"Missing Telegram config: {ENV_TOKEN_KEY}/{ENV_CHAT_KEY}"

    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    files = {"photo": ("report.png", img_bytes, "image/png")}
    data = {"chat_id": chat_id, "caption": caption, "parse_mode": "Markdown"}

    try:
        resp = requests.post(url, files=files, data=data, timeout=15)
        # Telegram may return 200 with ok:false; parse JSON
        payload = {}
        try:
            payload = resp.json()
        except Exception:
            pass

        if resp.status_code != 200 or not payload.get("ok", False):
            status = payload.get("error_code", resp.status_code)
            desc = payload.get("description", resp.reason)
            return False, f"Telegram error ({status}): {desc}. {_error_hint(status, desc)}"

        msg_id = payload.get("result", {}).get("message_id")
        return True, f"Report sent to Telegram (message_id={msg_id})."

    except requests.exceptions.Timeout:
        return False, "Telegram API timeout. Check network."
    except requests.exceptions.RequestException as e:
        return False, f"Request error: {e}"
    except Exception as e:
        return False, f"Unexpected error: {e}"
