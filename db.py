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


# ---------------- session save ----------------

def save_session(
    answers: dict[str, str],
    profile: dict[str, float],
    scored: dict[str, dict],
    matches: list[dict],
    email: str | None = None,
    user_agent: str | None = None,
    duration_seconds: int | None = None,
) -> str | None:
    """Insert one completed session. Returns the inserted row's id, or None
    if persistence is disabled / failed. Never raises — failure is silent
    to keep the user-facing app robust.

    answers: keyed by QID ("Q1", "Q2", ...) → user's essay text
    profile: 8-dim percent breakdown summing to 100
    scored: {intel: {"score": int, "evidence": str}} from the LLM
    matches: top-N list of {soc, title, match_pct, ...} dicts
    email: optional user email
    user_agent: optional browser UA string
    duration_seconds: optional time spent on the questionnaire
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
    }

    try:
        result = client.table("sessions").insert(row).execute()
        if result.data:
            return result.data[0].get("id")
    except Exception as e:
        print(f"[db] save_session failed: {e}")
    return None


def is_enabled() -> bool:
    """Quick check used by the UI: do we have working persistence?"""
    return get_client() is not None


def email_already_used(email: str) -> bool:
    """Return True if this email has already taken the test. Used to enforce
    one-session-per-email at the welcome screen. Empty/None emails always
    return False (anonymous sessions don't deduplicate).
    """
    if not email or not email.strip():
        return False
    client = get_client()
    if client is None:
        # if DB is down, don't block users
        return False
    try:
        result = (client.table("sessions")
                  .select("id")
                  .eq("email", email.strip().lower())
                  .limit(1)
                  .execute())
        return bool(result.data)
    except Exception as e:
        print(f"[db] email_already_used check failed: {e}")
        return False   # fail-open: don't block on DB error
