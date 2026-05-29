"""Supabase persistence layer for INT Intelligence sessions.

Saves each completed user session to the `sessions` table. Reads credentials
from st.secrets first (production), then env vars (local dev), then a .env
file (local dev fallback).

Designed to FAIL SOFT: if Supabase isn't reachable, the app still works —
we just don't persist the session. This keeps the user experience intact
even if there's a temporary database issue.
"""
from __future__ import annotations
import os
import json
from typing import Any

# ---------------- credential resolution ----------------

_client = None  # lazy singleton


def _get_credentials() -> tuple[str | None, str | None]:
    """Resolve Supabase URL + service key from (in order):
       1. environment variables (works locally + on Streamlit Cloud secrets)
       2. .env file (local dev only)
       3. st.secrets (Streamlit Cloud)
    Returns (url, key) — either may be None if not configured.
    """
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if url and key:
        return url, key

    # try .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_KEY")
        if url and key:
            return url, key
    except ImportError:
        pass

    # try st.secrets (only available inside Streamlit)
    try:
        import streamlit as st
        url = st.secrets.get("SUPABASE_URL")
        key = st.secrets.get("SUPABASE_SERVICE_KEY")
        if url and key:
            return url, key
    except Exception:
        pass

    return None, None


def get_client():
    """Return a memoized Supabase client, or None if creds missing."""
    global _client
    if _client is not None:
        return _client

    url, key = _get_credentials()
    if not url or not key:
        return None

    try:
        from supabase import create_client
        _client = create_client(url, key)
        return _client
    except Exception as e:
        # any failure here means we silently disable persistence
        print(f"[db] supabase client init failed: {e}")
        return None


# ---------------- progressive save (refresh persistence) ----------------

# mapping from QID (Q1..Q7) to the column name used in the sessions table
_QID_TO_COL = {
    "Q1": "q1_place",
    "Q2": "q2_habit",
    "Q3": "q3_explain",
    "Q4": "q4_conflict",
    "Q5": "q5_sound",
    "Q6": "q6_body",
    "Q7": "q7_unworded",
}
_COL_TO_QID = {v: k for k, v in _QID_TO_COL.items()}


def init_pending_session(user_agent: str | None = None) -> str | None:
    """Create an empty session row when the user clicks Begin. Returns the
    new row id (UUID) which we then put in the URL as ?s=<id> so that page
    refreshes can hydrate from this row.
    """
    client = get_client()
    if client is None:
        return None
    try:
        result = client.table("sessions").insert({
            "user_agent": user_agent,
            "metadata": {"current_step": "q0", "pending": True},
        }).execute()
        if result.data:
            return result.data[0].get("id")
    except Exception as e:
        print(f"[db] init_pending_session failed: {e}")
    return None


def update_progress(session_id: str, step: str, answers: dict[str, str]) -> None:
    """Update an existing session row with the latest step + answers. Called
    on every Continue/Back click. Silent on failure so the app keeps working
    even if the DB is briefly unreachable.
    """
    client = get_client()
    if client is None or not session_id:
        return
    try:
        update: dict[str, Any] = {}
        for qid, text in (answers or {}).items():
            col = _QID_TO_COL.get(qid)
            if col and text:
                update[col] = text
        update["metadata"] = {"current_step": step, "pending": True}
        client.table("sessions").update(update).eq("id", session_id).execute()
    except Exception as e:
        print(f"[db] update_progress failed: {e}")


def load_progress(session_id: str) -> dict | None:
    """Fetch a session by id and return its state in a form that app.py can
    use to rehydrate st.session_state. Returns None if not found.
    """
    client = get_client()
    if client is None or not session_id:
        return None
    try:
        result = client.table("sessions").select("*").eq("id", session_id).limit(1).execute()
        if not result.data:
            return None
        row = result.data[0]
        # reconstruct the QID-keyed answers dict
        answers: dict[str, str] = {}
        for col, qid in _COL_TO_QID.items():
            v = row.get(col)
            if v:
                answers[qid] = v
        meta = row.get("metadata") or {}
        step = meta.get("current_step", "welcome")
        return {
            "step": step,
            "answers": answers,
            "saved_id": row["id"],
            "has_results": bool(row.get("top_matches_json")),
        }
    except Exception as e:
        print(f"[db] load_progress failed: {e}")
        return None


# ---------------- session save ----------------

def save_session(
    answers: dict[str, str],
    profile: dict[str, float],
    scored: dict[str, dict],
    matches: list[dict],
    email: str | None = None,
    user_agent: str | None = None,
    duration_seconds: int | None = None,
    metadata: dict | None = None,
    existing_id: str | None = None,
) -> str | None:
    """Persist a completed session. If existing_id is given (the typical case
    now, since init_pending_session is called when user clicks Begin), UPDATE
    that row with final results. Otherwise INSERT a new row.

    Returns the row id on success, None on failure.

    answers: keyed by QID ("Q1", "Q2", ...) → user's essay text
    profile: 8-dim percent breakdown summing to 100
    scored: {intel: {"score": int, "evidence": str}} from the LLM
    matches: top-N list of {soc, title, match_pct, ...} dicts
    email: optional user email (usually None at save time, added later)
    user_agent: optional browser UA string
    duration_seconds: optional total time spent
    metadata: optional dict of analytics (word counts, per-question timing, etc.)
    existing_id: id of a pre-created pending row to UPDATE in place
    """
    client = get_client()
    if client is None:
        return None

    # extract evidence-only mapping (drop the LLM score numbers; keep evidence)
    evidence = {k: v.get("evidence", "") for k, v in scored.items()}

    # keep only fields needed for analytics in top_matches (saves space)
    matches_slim = [
        {
            "rank": i + 1,
            "soc": m.get("soc"),
            "title": m.get("title"),
            "match_pct": round(m.get("match_pct", 0.0), 2),
            "gardner_cos": round(m.get("gardner_cos", 0.0), 4),
            "content_cos": round(m.get("content_cos", 0.0), 4) if m.get("content_cos") is not None else None,
        }
        for i, m in enumerate(matches[:10])
    ]

    # normalize email to lowercase (so john@x.com and JOHN@X.COM count as same)
    email_clean = email.strip().lower() if email and email.strip() else None

    # merge any pending metadata flag with the analytics metadata
    final_metadata = dict(metadata or {})
    final_metadata["current_step"] = "results"
    final_metadata["pending"] = False

    row: dict[str, Any] = {
        "email": email_clean,
        "q1_place": answers.get("Q1") or None,
        "q2_habit": answers.get("Q2") or None,
        "q3_explain": answers.get("Q3") or None,
        "q4_conflict": answers.get("Q4") or None,
        "q5_sound": answers.get("Q5") or None,
        "q6_body": answers.get("Q6") or None,
        "q7_unworded": answers.get("Q7") or None,
        "profile_json": profile,
        "evidence_json": evidence,
        "top_matches_json": matches_slim,
        "user_agent": user_agent,
        "duration_seconds": duration_seconds,
        "metadata": final_metadata,
    }

    try:
        if existing_id:
            result = client.table("sessions").update(row).eq("id", existing_id).execute()
            if result.data:
                return existing_id
            # if update returned nothing, fall through to insert (row may not exist)
        result = client.table("sessions").insert(row).execute()
        if result.data:
            return result.data[0].get("id")
    except Exception as e:
        print(f"[db] save_session failed: {e}")
    return None


#: Admin/owner emails that bypass the one-report-per-email limitation.
#: These can request reports as many times as they want (useful for
#: testing, debugging, and personal repeat use by the project owner).
ADMIN_EMAILS = {
    "gkrai890@gmail.com",
}


def request_report_email(session_id: str, email: str) -> tuple[bool, str]:
    """Called from the results screen when the user types an email to receive
    their report. Three outcomes:
       (True, ok-message)  — email accepted, recorded on session, ready to send
       (False, error-msg)  — email already used to receive a report
       (False, error-msg)  — DB error / invalid input

    The actual email-sending is handled separately. This just records intent
    so the same email can't be re-used. Admin emails in ADMIN_EMAILS bypass
    the one-time limit entirely.
    """
    email_clean = (email or "").strip().lower()
    if "@" not in email_clean or "." not in email_clean.split("@")[-1]:
        return False, "Please enter a valid email address."

    client = get_client()
    if client is None:
        return False, "Database not available right now. Try again in a minute."

    try:
        # admin emails skip the duplicate-check entirely
        if email_clean not in ADMIN_EMAILS:
            existing = (client.table("sessions")
                        .select("id")
                        .eq("email", email_clean)
                        .not_.is_("report_sent_at", "null")
                        .limit(1)
                        .execute())
            if existing.data:
                return False, (
                    "This email has already received a report. Each email can only "
                    "be used once."
                )

        # Pull the session data needed to build the PDF
        sess = (client.table("sessions")
                .select("profile_json,evidence_json,top_matches_json")
                .eq("id", session_id)
                .limit(1)
                .execute())
        if not sess.data:
            return False, "Couldn't find that session. Please refresh."
        row = sess.data[0]
        profile = row.get("profile_json") or {}
        evidence = row.get("evidence_json") or {}
        top_matches = row.get("top_matches_json") or []
        # evidence_json is {intel: "string"} — reshape to scorer format
        scored = {k: {"score": 0, "evidence": v} for k, v in evidence.items()}

        # Build the PDF
        try:
            from pdf_gen import build_pdf
            pdf_bytes = build_pdf(profile, scored, top_matches,
                                  all_matches=None,
                                  user_email=email_clean)
        except Exception as e:
            print(f"[db] PDF build failed: {e}")
            return False, "Couldn't build the PDF. Please try again."

        # Send the email via Gmail SMTP
        try:
            from email_sender import send_report, is_configured
            if not is_configured():
                # Don't claim it was sent if SMTP isn't wired up.
                return False, ("Email service isn't configured yet. "
                               "Your results stay on this page.")
            ok, msg = send_report(email_clean, pdf_bytes)
            if not ok:
                return False, msg
        except Exception as e:
            print(f"[db] email send failed: {e}")
            return False, "Couldn't send the email right now. Please try again."

        # Only mark sent AFTER the email actually went out, so failures
        # don't permanently lock the recipient out of getting their report.
        from datetime import datetime, timezone
        now_iso = datetime.now(timezone.utc).isoformat()
        (client.table("sessions")
         .update({
             "email": email_clean,
             "report_sent_to": email_clean,
             "report_sent_at": now_iso,
         })
         .eq("id", session_id)
         .execute())

        return True, f"Report sent to {email_clean}. Check your inbox."
    except Exception as e:
        print(f"[db] request_report_email failed: {e}")
        return False, "Something went wrong. Please try again."


def is_enabled() -> bool:
    """Quick check used by the UI: do we have working persistence?"""
    return get_client() is not None


def email_already_used(email: str) -> bool:
    """Return True if this email has already taken the test. Used to enforce
    one-session-per-email at the welcome screen. Empty/None emails always
    return False (anonymous sessions don't deduplicate). Admin emails in
    ADMIN_EMAILS always return False so the owner can re-run freely.
    """
    if not email or not email.strip():
        return False
    email_clean = email.strip().lower()
    # admin emails never count as "already used"
    if email_clean in ADMIN_EMAILS:
        return False
    client = get_client()
    if client is None:
        # if DB is down, don't block users
        return False
    try:
        result = (client.table("sessions")
                  .select("id")
                  .eq("email", email_clean)
                  .limit(1)
                  .execute())
        return bool(result.data)
    except Exception as e:
        print(f"[db] email_already_used check failed: {e}")
        return False   # fail-open: don't block on DB error
