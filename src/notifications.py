"""
DataMind AI — Notification & Tracking Module
Sends email alerts when someone downloads a report or views a shared link.
Uses Gmail SMTP (free) — no paid services.

Setup: set these in Streamlit secrets or environment variables:
  NOTIFY_EMAIL       = your Gmail address (recipient)
  GMAIL_SENDER       = sender Gmail address
  GMAIL_APP_PASSWORD = Gmail App Password (not your real password)
                       → Google Account → Security → 2-Step → App Passwords

Note on Share URLs:
  We generate a unique token per report and encode everything in the URL itself
  (no database needed). When someone opens the link, the Streamlit app decodes
  the token, reconstructs the report, and fires an email to you.
"""

import os
import smtplib
import hashlib
import base64
import json
import urllib.parse
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from html import escape as _html_escape
from typing import Optional, Dict


def _safe_url(url: Optional[str]) -> str:
    """Allow only http(s) absolute URLs or in-app relative ?report= links."""
    if not url:
        return ""
    u = str(url).strip()
    if u.startswith(("http://", "https://", "?")):
        return _html_escape(u, quote=True)
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# EMAIL
# ─────────────────────────────────────────────────────────────────────────────

def _get_email_config():
    """Read email config from Streamlit secrets or env vars."""
    try:
        import streamlit as st
        notify  = st.secrets.get("NOTIFY_EMAIL", os.getenv("NOTIFY_EMAIL", ""))
        sender  = st.secrets.get("GMAIL_SENDER", os.getenv("GMAIL_SENDER", ""))
        app_pwd = st.secrets.get("GMAIL_APP_PASSWORD", os.getenv("GMAIL_APP_PASSWORD", ""))
    except Exception:
        notify  = os.getenv("NOTIFY_EMAIL", "")
        sender  = os.getenv("GMAIL_SENDER", "")
        app_pwd = os.getenv("GMAIL_APP_PASSWORD", "")
    return notify, sender, app_pwd


def send_notification_email(
    event_type: str,          # "download" | "view_link"
    dataset_name: str,
    format_name: str,         # "PDF" | "Word" | "Markdown"
    user_info: Dict,          # {"name": ..., "email": ..., "company": ...}
    share_url: Optional[str] = None,
    extra_notes: str = "",
) -> bool:
    """
    Send a notification email to the report owner.
    Returns True if sent successfully.
    """
    notify_email, sender_email, app_password = _get_email_config()

    if not all([notify_email, sender_email, app_password]):
        return False  # Config not set — silent fail

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    event_label = {"download": "Report Downloaded", "view_link": "Shared Link Viewed"}.get(
        event_type, event_type.title()
    )

    user_name    = _html_escape(str(user_info.get("name", "Anonymous")))
    user_email   = _html_escape(str(user_info.get("email", "Not provided")))
    user_company = _html_escape(str(user_info.get("company", "Not provided")))
    user_role    = _html_escape(str(user_info.get("role", "Not provided")))
    dataset_safe = _html_escape(str(dataset_name))
    format_safe  = _html_escape(str(format_name))
    event_safe   = _html_escape(event_label)
    extra_safe   = _html_escape(str(extra_notes)) if extra_notes else ""
    share_safe   = _safe_url(share_url)

    subject = f"🧠 DataMind AI — {event_label}: {dataset_name}"

    html_body = f"""
<html><body style="font-family: Arial, sans-serif; background: #f9fafb; padding: 24px;">
<div style="max-width: 560px; margin: 0 auto; background: white; border-radius: 12px;
     border: 1px solid #e5e7eb; overflow: hidden;">

  <div style="background: #4F46E5; padding: 24px 28px;">
    <h1 style="color: white; margin: 0; font-size: 22px;">🧠 DataMind AI</h1>
    <p style="color: #c4b5fd; margin: 4px 0 0;">{event_safe}</p>
  </div>

  <div style="padding: 24px 28px;">
    <table style="width: 100%; border-collapse: collapse;">
      <tr><td style="padding: 8px 0; color: #6b7280; font-size: 13px; width: 40%;">Dataset</td>
          <td style="padding: 8px 0; font-weight: 600; font-size: 13px; color: #1f2937;">{dataset_safe}</td></tr>
      <tr style="background: #f9fafb;"><td style="padding: 8px 0; color: #6b7280; font-size: 13px;">Format</td>
          <td style="padding: 8px 0; font-size: 13px; color: #1f2937;">{format_safe}</td></tr>
      <tr><td style="padding: 8px 0; color: #6b7280; font-size: 13px;">Time</td>
          <td style="padding: 8px 0; font-size: 13px; color: #1f2937;">{now}</td></tr>
    </table>

    <div style="margin: 20px 0; padding: 16px; background: #EEF2FF; border-radius: 8px;
         border-left: 4px solid #4F46E5;">
      <p style="margin: 0 0 4px; font-weight: 700; color: #3730A3; font-size: 14px;">User details</p>
      <p style="margin: 0; font-size: 13px; color: #374151;">
        <b>Name:</b> {user_name}<br>
        <b>Email:</b> {user_email}<br>
        <b>Company:</b> {user_company}<br>
        <b>Role:</b> {user_role}
      </p>
    </div>

    {"<div style='margin: 12px 0; padding: 12px; background: #f0fdf4; border-radius: 8px;'><p style='margin: 0; font-size: 12px; color: #15803d;'><b>Share URL:</b> <a href='" + share_safe + "'>" + share_safe + "</a></p></div>" if share_safe else ""}
    {"<p style='font-size: 12px; color: #6b7280;'>" + extra_safe + "</p>" if extra_safe else ""}
  </div>

  <div style="padding: 16px 28px; background: #f9fafb; border-top: 1px solid #e5e7eb;
       text-align: center; font-size: 11px; color: #9ca3af;">
    DataMind AI · Automated notification
  </div>
</div>
</body></html>"""

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = sender_email
        msg["To"]      = notify_email
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, app_password)
            server.sendmail(sender_email, notify_email, msg.as_string())
        return True
    except Exception as e:
        print(f"[notify] Email send failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# SHARE URL  (token-based, no database needed)
# ─────────────────────────────────────────────────────────────────────────────

def generate_share_token(metadata: Dict) -> str:
    """Encode report metadata into a URL-safe token."""
    payload = json.dumps(metadata, default=str)
    token = base64.urlsafe_b64encode(payload.encode("utf-8")).decode("utf-8")
    return token


def decode_share_token(token: str) -> Optional[Dict]:
    """Decode a share token back to metadata. Returns None if invalid."""
    try:
        payload = base64.urlsafe_b64decode(token.encode("utf-8")).decode("utf-8")
        return json.loads(payload)
    except Exception:
        return None


def build_share_url(base_url: str, metadata: Dict) -> str:
    """
    Build a shareable URL that encodes the report metadata.
    base_url: the deployed Streamlit app URL, e.g. https://yourapp.streamlit.app
    """
    token = generate_share_token(metadata)
    return f"{base_url}?report={urllib.parse.quote(token)}"


def get_base_url() -> str:
    """Detect the base URL of the running Streamlit app."""
    try:
        import streamlit as st
        # Check if running on Streamlit Cloud
        cloud_url = os.getenv("STREAMLIT_SHARING_MODE", "")
        if cloud_url:
            return os.getenv("APP_URL", "https://your-datamind-app.streamlit.app")
        # Local fallback
        return "http://localhost:8501"
    except Exception:
        return "http://localhost:8501"
