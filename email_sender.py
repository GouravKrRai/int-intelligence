"""Gmail SMTP email sender for INT Intelligence reports.

Reads credentials from st.secrets first (production), env vars second
(local dev), .env third (local dev fallback). Sends a single MIME multipart
email with the PDF attached.

Designed to FAIL SOFT: any error during send is caught, returns (False, msg)
so the calling code can record failure without crashing the user's session.
"""
from __future__ import annotations
import os
import smtplib
import ssl
from email.message import EmailMessage


SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465  # SMTPS — TLS from the start, simpler than STARTTLS


def _get_credentials() -> tuple[str | None, str | None, str]:
    """Returns (gmail_user, app_password, from_name)."""
    user = os.environ.get("GMAIL_USER")
    pw = os.environ.get("GMAIL_APP_PASSWORD")
    name = os.environ.get("FROM_NAME") or "INT Intelligence"
    if user and pw:
        return user, pw, name

    # .env fallback
    try:
        from dotenv import load_dotenv
        load_dotenv()
        user = os.environ.get("GMAIL_USER")
        pw = os.environ.get("GMAIL_APP_PASSWORD")
        name = os.environ.get("FROM_NAME") or "INT Intelligence"
        if user and pw:
            return user, pw, name
    except ImportError:
        pass

    # st.secrets fallback
    try:
        import streamlit as st
        user = st.secrets.get("GMAIL_USER")
        pw = st.secrets.get("GMAIL_APP_PASSWORD")
        name = st.secrets.get("FROM_NAME") or "INT Intelligence"
        if user and pw:
            return user, pw, name
    except Exception:
        pass

    return None, None, "INT Intelligence"


def is_configured() -> bool:
    user, pw, _ = _get_credentials()
    return bool(user and pw)


def send_report(to_email: str,
                pdf_bytes: bytes,
                pdf_filename: str = "INT_Intelligence_Report.pdf",
                subject: str | None = None) -> tuple[bool, str]:
    """Send the user's PDF report. Returns (ok, message).

    On success returns (True, "...delivered to <email>")
    On failure returns (False, "<reason>") — never raises.
    """
    user, pw, from_name = _get_credentials()
    if not user or not pw:
        return False, "Email service not configured."

    to_email = (to_email or "").strip()
    if "@" not in to_email:
        return False, "Invalid recipient address."

    msg = EmailMessage()
    msg["From"] = f"{from_name} <{user}>"
    msg["To"] = to_email
    msg["Subject"] = subject or "Your INT Intelligence report"

    # plain-text body — Gmail will show this if the recipient blocks HTML
    msg.set_content(
        "Hi,\n\n"
        "Your INT Intelligence report is attached as a PDF.\n\n"
        "It shows your eight-intelligence profile and the top 20 careers "
        "whose day-to-day work matches what you wrote about.\n\n"
        "If you have questions or want to share what you found, just reply "
        "to this email.\n\n"
        "— INT Intelligence\n"
    )

    # HTML alternative — slightly nicer for most modern clients
    msg.add_alternative(
        f"""\
<!doctype html>
<html><body style="font-family: Georgia, serif; color: #1f1d1a; line-height: 1.55;">
  <p>Hi,</p>
  <p>Your <b>INT Intelligence</b> report is attached as a PDF.</p>
  <p>It shows your eight-intelligence profile and the top 20 careers
     whose day-to-day work matches what you wrote about.</p>
  <p>If you have questions or want to share what you found, just reply
     to this email.</p>
  <p style="color: #5e5a55;">— INT Intelligence</p>
</body></html>
""", subtype="html")

    msg.add_attachment(pdf_bytes,
                       maintype="application", subtype="pdf",
                       filename=pdf_filename)

    try:
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=ctx, timeout=30) as s:
            s.login(user, pw)
            s.send_message(msg)
        return True, f"Report sent to {to_email}."
    except smtplib.SMTPAuthenticationError:
        return False, "Email service auth failed. Check Gmail App Password."
    except smtplib.SMTPRecipientsRefused:
        return False, f"Recipient {to_email} was refused by Gmail."
    except (smtplib.SMTPException, OSError) as e:
        return False, f"Email send failed: {e}"
