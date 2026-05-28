"""
Streamlit UI for the Gardner intelligence career matcher.

Flow:
    welcome -> Q1..Q7 (one question per screen) -> loading -> results

Run:
    cd /Users/gouravkumarrai/Downloads/onet
    streamlit run app.py

Reads ANTHROPIC_API_KEY from st.secrets first, then env var as fallback.
"""
from __future__ import annotations
import os
import time

# defensive: HF Hub's new xet protocol stalls on anonymous downloads — use plain HTTP
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

import streamlit as st

# local modules
from scorer import score as llm_score, to_percentages, embed_essays, INTEL
from match import match as do_match, EMB_NPY
import db

# ---------------- config ----------------

PRODUCT_NAME = "INT Intelligence"
TAGLINE = "see the kind of mind you are."

QUESTIONS = [
    ("Q1", "A place in memory",
     "Describe a place you keep returning to in your memory — not necessarily "
     "anywhere you still live. What's there? Take me through it."),
    ("Q2", "Something you do",
     "Tell me about something you do that other people find strange or don't "
     "quite understand. Not a hobby — a small habit, ritual, or quirk."),
    ("Q3", "How does it work",
     "Pick any everyday object or process — a zipper, rain, a key in a lock, "
     "how a fly lands on a wall. Explain how it actually works, the way *you* "
     "think about it. Don't worry about being scientifically right."),
    ("Q4", "A fight you remember",
     "Describe a fight or disagreement you remember vividly — even a small one. "
     "What was the story *behind* the surface story?"),
    ("Q5", "A sound that stays",
     "There's a sound from your life — not a song — that has stayed with you. "
     "Describe it. When did you first hear it? Why does it stick?"),
    ("Q6", "Body before mind",
     "Tell me about a moment when your body knew something before your mind did."),
    ("Q7", "The unworded thing",
     "Describe a feeling or experience that you've never quite found the right "
     "words for. Try to describe it anyway."),
]

PRETTY_LABEL = {
    "linguistic":    "Linguistic",
    "logical":       "Logical-Mathematical",
    "spatial":       "Spatial",
    "kinesthetic":   "Bodily-Kinesthetic",
    "musical":       "Musical",
    "interpersonal": "Interpersonal",
    "intrapersonal": "Intrapersonal",
    "naturalistic":  "Naturalistic",
}


# ---------------- request metadata helpers ----------------

def _get_user_agent() -> str | None:
    """Best-effort extraction of the user-agent string. Streamlit exposes
    this via st.context.headers in 1.37+. Returns None on older versions
    or if header is missing.
    """
    try:
        headers = st.context.headers
        if headers:
            return headers.get("User-Agent")
    except Exception:
        pass
    return None


# ---------------- API key plumbing ----------------

def setup_api_key() -> bool:
    """Pulls ANTHROPIC_API_KEY from st.secrets or env. Returns True if found."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return True
    try:
        k = st.secrets["ANTHROPIC_API_KEY"]
        if k:
            os.environ["ANTHROPIC_API_KEY"] = k
            return True
    except (FileNotFoundError, KeyError, Exception):
        pass
    return False


# ---------------- styling ----------------

st.set_page_config(
    page_title=PRODUCT_NAME,
    layout="centered",
    initial_sidebar_state="collapsed",
)

# JS to kill Streamlit's bare-C-key shortcut that pops "Clear caches" dialog.
# Streamlit's keydown handler doesn't check modifiers properly, so Cmd+C
# (which the OS handles for copy) ALSO triggers the dev menu. We intercept
# in capture phase before Streamlit's handler sees it. The browser still
# copies text because copy is handled at the selection layer, not the
# keydown event.
st.markdown("""
<script>
document.addEventListener('keydown', function(e) {
  // bare C key — kill the Streamlit dev shortcut entirely
  if ((e.key === 'c' || e.key === 'C') && !e.altKey && !e.shiftKey) {
    if (e.metaKey || e.ctrlKey) {
      // Cmd+C / Ctrl+C: let the OS-level copy fire, but stop Streamlit
      e.stopPropagation();
    } else {
      // bare C: block both the dev menu AND prevent default
      e.stopPropagation();
      e.preventDefault();
    }
  }
  // also block bare R (rerun shortcut)
  if ((e.key === 'r' || e.key === 'R') && !e.altKey && !e.shiftKey
      && !e.metaKey && !e.ctrlKey) {
    e.stopPropagation();
    e.preventDefault();
  }
}, true);  // true = capture phase, before Streamlit's bubble-phase handler
</script>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* page-wide */
.block-container {max-width: 720px; padding-top: 4rem; padding-bottom: 4rem;}
[data-testid="stHeader"] {display: none;}
[data-testid="stSidebar"] {display: none;}
[data-testid="stToolbar"] {display: none;}
footer {display: none;}

/* typography */
h1, h2, h3 {font-family: 'Georgia', serif; letter-spacing: -0.01em;}
body, .stMarkdown, .stTextArea textarea {font-family: 'Georgia', serif;}

/* question heading */
.q-tag {color: #888; font-size: 0.85rem; letter-spacing: 0.15em;
        text-transform: uppercase; margin-bottom: 0.5rem;}
.q-title {font-size: 1.5rem; font-weight: 500; line-height: 1.4;
          margin-bottom: 1.5rem; color: #222;}
.q-prompt {font-size: 1.05rem; line-height: 1.6; color: #444;
           margin-bottom: 1.5rem;}

/* textarea */
.stTextArea textarea {
    min-height: 280px !important;
    font-size: 1.05rem !important;
    line-height: 1.6 !important;
    border-radius: 4px !important;
    border-color: #ddd !important;
    padding: 1rem !important;
}

/* buttons */
.stButton button {
    background: #222 !important; color: white !important;
    border: none !important; border-radius: 4px !important;
    padding: 0.6rem 1.8rem !important; font-size: 1rem !important;
    font-family: 'Georgia', serif !important;
}
.stButton button:hover {background: #444 !important;}
.stButton button:disabled {background: #ccc !important; color: #fff !important;}

/* result-page profile bars */
.profile-row {display: flex; align-items: center; margin-bottom: 0.6rem;}
.profile-label {width: 220px; font-size: 1rem; color: #333;}
.profile-bar-bg {flex: 1; background: #eee; height: 18px; border-radius: 2px;
                 overflow: hidden;}
.profile-bar-fg {height: 100%; background: #222;}
.profile-pct {width: 60px; text-align: right; font-size: 0.95rem;
              color: #555; padding-left: 0.8rem;}

/* career list */
.career-row {padding: 0.9rem 0; border-bottom: 1px solid #eee;}
.career-rank {color: #999; font-size: 0.85rem; letter-spacing: 0.1em;}
.career-title {font-size: 1.15rem; color: #222; margin-top: 0.15rem;}
.career-match {color: #555; font-size: 0.95rem; margin-top: 0.2rem;}

/* evidence card */
.evidence-card {background: #f8f8f6; padding: 1rem 1.2rem; border-radius: 4px;
                margin-bottom: 0.7rem; border-left: 3px solid #222;}
.evidence-intel {font-weight: 500; color: #222; margin-bottom: 0.3rem;}
.evidence-text {font-size: 0.95rem; color: #555; line-height: 1.5;}
</style>
""", unsafe_allow_html=True)


# ---------------- session state + URL/Supabase persistence ----------------

# Refresh-persistence strategy:
#   - Each user session gets a row in Supabase from the moment they click Begin.
#   - The row id (uuid) is placed in the URL as ?s=<id>.
#   - On every step transition, db.update_progress() syncs current step + answers.
#   - On page refresh, the URL ?s=<id> is read first; the row is fetched from
#     Supabase and used to rehydrate the in-memory session_state.
#   - No localStorage, no async timing issues. URL is the source of truth.

# Default session_state values
if "step" not in st.session_state:
    st.session_state.step = "welcome"   # welcome | q0..q6 | loading | results
    st.session_state.answers = {}        # {qid: text}
    st.session_state.result = None       # full pipeline result
    st.session_state.started_at = None   # epoch seconds when "Begin" pressed
    st.session_state.q_timings = {}      # {qid: epoch_seconds when answered}
    st.session_state.q_visits = {}       # {qid: visit_count} — counts going back
    st.session_state.saved_id = None     # supabase row id after persist
    st.session_state.email_sent = False  # has the user requested email send?
    st.session_state.email_error = None  # last email submission error


def _hydrate_from_url() -> None:
    """If the URL contains ?s=<session_id>, fetch the row from Supabase and
    rehydrate the in-memory state. Only runs once per browser session."""
    if st.session_state.get("_url_hydrated"):
        return
    st.session_state._url_hydrated = True

    try:
        url_sid = st.query_params.get("s")
    except Exception:
        url_sid = None
    if not url_sid:
        return

    load_fn = getattr(db, "load_progress", None)
    if load_fn is None:
        return  # stale db.py without progress functions — skip silently
    try:
        saved = load_fn(url_sid)
    except Exception as e:
        print(f"[app] load_progress error: {e}")
        return
    if not saved:
        # row not found (maybe deleted) — clear stale URL param
        try:
            del st.query_params["s"]
        except Exception:
            pass
        return

    st.session_state.saved_id = saved.get("saved_id")
    st.session_state.answers = saved.get("answers") or {}

    # If results were already computed, route to results (where the pipeline
    # will be silently re-run from the saved answers to repopulate matches).
    # Otherwise route to whatever step they were last on.
    step = saved.get("step") or "welcome"
    if step in {"welcome", "loading", "results"}:
        st.session_state.step = step
    elif step.startswith("q"):
        st.session_state.step = step
    else:
        st.session_state.step = "welcome"


# Run URL hydration once at startup
_hydrate_from_url()


def go(step: str) -> None:
    """Transition to a new step. Persists step + answers to Supabase if we
    have a pending session id, so a page refresh can rehydrate from the DB."""
    st.session_state.step = step
    sid = st.session_state.get("saved_id")
    if sid:
        update_fn = getattr(db, "update_progress", None)
        if update_fn is not None:
            try:
                update_fn(sid, step, st.session_state.get("answers") or {})
            except Exception:
                pass
    st.rerun()


def _clear_storage() -> None:
    """Wipe URL ?s param when starting over."""
    try:
        if "s" in st.query_params:
            del st.query_params["s"]
    except Exception:
        pass


# ---------------- career-map chart ----------------

def render_career_map(all_matches: list[dict], top_matches: list[dict]) -> None:
    """Scatter of all careers in cosine space. Top 20 highlighted with:
      - one distinct color per career (tab20 colormap)
      - TRIANGLE marker for top 5 (your strongest matches)
      - CIRCLE marker for ranks 6-20
      - no numbers/labels inside dots — legend on the right maps each
        color+symbol to a career name
    """
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib.lines import Line2D

    rcParams["font.family"] = "Georgia, serif"

    xs_all = [m.get("gardner_cos", 0.0) for m in all_matches]
    ys_all = [m.get("content_cos", 0.0) for m in all_matches]

    top_pts = [(i, m["title"],
                m.get("gardner_cos", 0.0),
                m.get("content_cos", 0.0))
               for i, m in enumerate(top_matches, 1)]

    fig, ax = plt.subplots(figsize=(12.5, 7.5), dpi=140)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # background cloud — every other career, very subtle
    ax.scatter(xs_all, ys_all, s=8, c="#e6e6e2", alpha=0.45,
               edgecolors="none", zorder=1)

    # set axis limits with padding
    x_min, x_max = min(xs_all), max(xs_all)
    y_min, y_max = min(ys_all), max(ys_all)
    x_pad = (x_max - x_min) * 0.08
    y_pad = (y_max - y_min) * 0.12
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    # 20 distinct colors from tab20 — designed for max distinguishability
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i / 20.0) for i in range(20)]

    legend_handles = []
    # paint in reverse so rank 1 sits on top of any overlaps
    for rank, title, x, y in reversed(top_pts):
        color = colors[rank - 1]
        if rank <= 5:
            marker = "^"   # triangle for top 5 — your strongest matches
            size = 280
            edge_w = 2.2
        else:
            marker = "o"   # circle for ranks 6-20
            size = 130
            edge_w = 1.4
        ax.scatter([x], [y], s=size, c=[color], marker=marker,
                   edgecolors="white", linewidths=edge_w,
                   zorder=3 + (21 - rank))

    # build legend in rank order (#1 first)
    for rank, title, _, _ in top_pts:
        color = colors[rank - 1]
        marker = "^" if rank <= 5 else "o"
        # truncate long titles so legend doesn't blow out
        label = title if len(title) <= 42 else title[:39] + "…"
        label = f"{rank:>2}. {label}"
        legend_handles.append(
            Line2D([0], [0], marker=marker, color="w",
                   markerfacecolor=color, markeredgecolor="white",
                   markersize=11 if rank <= 5 else 8,
                   markeredgewidth=1.2, label=label)
        )

    # legend on the right side, anchored outside the axes
    ax.legend(handles=legend_handles,
              loc="upper left", bbox_to_anchor=(1.02, 1.0),
              fontsize=8.5, frameon=False, labelspacing=0.7,
              handletextpad=0.6, borderpad=0.4,
              title="top 20 careers", title_fontsize=9)

    # clean minimal axes
    ax.set_xlabel("cognitive shape match  →", fontsize=10,
                  color="#666", labelpad=10)
    ax.set_ylabel("content / interest match  →", fontsize=10,
                  color="#666", labelpad=10)
    ax.tick_params(axis="both", colors="#aaa", labelsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#ddd")
    ax.grid(True, which="major", linestyle="-", linewidth=0.5,
            color="#f3f3f0", zorder=0)

    # reserve right ~36% of figure for the 20-row legend
    fig.tight_layout(rect=[0, 0, 0.64, 1])
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ---------------- screens ----------------

def screen_welcome() -> None:
    st.markdown(f"<h1 style='margin-bottom:0.3rem;'>{PRODUCT_NAME}</h1>",
                unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:1.2rem; color:#666; margin-top:0;'>"
                f"{TAGLINE}</p>", unsafe_allow_html=True)
    st.markdown(
        "<div style='margin: 2.5rem 0; color:#444; line-height:1.7; font-size:1.05rem;'>"
        "You will see seven short prompts. Answer each one honestly and "
        "specifically — there are no right answers, and length is not the point. "
        "Write the way you'd talk to a friend who's curious about you."
        "<br><br>"
        "It takes about <b>20–30 minutes</b>. Don't think too hard. The whole "
        "thing only works if you don't try to sound smart."
        "</div>",
        unsafe_allow_html=True,
    )
    if st.button("Begin", key="begin_btn"):
        st.session_state.started_at = time.time()
        st.session_state.q_timings = {}
        st.session_state.q_visits = {}
        # Create the pending Supabase row right now and stash its id in the
        # URL. This lets refresh-during-test recover the session by reading
        # ?s=<id> back out of the URL on next page load.
        # Defensive: if a stale deploy doesn't have the function yet, skip
        # gracefully — the app still works, just without refresh-persistence.
        if not st.session_state.get("saved_id"):
            init_fn = getattr(db, "init_pending_session", None)
            if init_fn is not None:
                try:
                    sid = init_fn(user_agent=_get_user_agent())
                    if sid:
                        st.session_state.saved_id = sid
                        try:
                            st.query_params["s"] = sid
                        except Exception:
                            pass
                except Exception as e:
                    print(f"[app] init_pending_session error: {e}")
        go("q0")


def screen_question(idx: int) -> None:
    qid, label, prompt = QUESTIONS[idx]

    # count visits to this question (so we know if they went back/forth)
    st.session_state.q_visits[qid] = st.session_state.q_visits.get(qid, 0) + 1

    st.markdown(f"<div class='q-tag'>{qid} · {label}</div>",
                unsafe_allow_html=True)
    st.markdown(f"<div class='q-prompt'>{prompt}</div>",
                unsafe_allow_html=True)

    answer = st.text_area(
        label="answer",
        value=st.session_state.answers.get(qid, ""),
        key=f"text_{qid}",
        label_visibility="collapsed",
        placeholder="Take your time. Even half a page is enough.",
    )

    cols = st.columns([1, 1, 1])
    with cols[2]:
        next_label = "Continue" if idx < len(QUESTIONS) - 1 else "Finish"
        disabled = len(answer.strip()) < 30   # soft minimum
        if st.button(next_label, key=f"next_{qid}", disabled=disabled):
            st.session_state.answers[qid] = answer.strip()
            st.session_state.q_timings[qid] = time.time()
            if idx < len(QUESTIONS) - 1:
                go(f"q{idx+1}")
            else:
                go("loading")
    if idx > 0:
        with cols[0]:
            if st.button("Back", key=f"back_{qid}"):
                st.session_state.answers[qid] = answer.strip()
                go(f"q{idx-1}")


def screen_loading() -> None:
    first_run = "_embedder_warm" not in st.session_state
    st.markdown(
        "<div style='text-align:center; padding:6rem 0; color:#555;'>"
        "<h2 style='color:#222;'>Reading you...</h2>"
        "<p style='font-size:1.05rem; color:#777; margin-top:1.5rem;'>"
        + ("This is your first read of the day — give it a minute while "
           "the language model loads. Subsequent reads will be fast."
           if first_run else
           "This usually takes about 15 seconds.")
        + "</p></div>",
        unsafe_allow_html=True,
    )
    st.session_state._embedder_warm = True

    if not setup_api_key():
        st.error("ANTHROPIC_API_KEY not configured. Set it in `.streamlit/secrets.toml` "
                 "or as an env var and reload.")
        if st.button("Back to start"):
            st.session_state.step = "welcome"
            st.rerun()
        return

    # build essays dict in the format scorer.py expects: {label: text}
    essays = {}
    for (qid, label, prompt) in QUESTIONS:
        ans = st.session_state.answers.get(qid, "").strip()
        if ans:
            essays[f"{qid} — {label}"] = ans

    try:
        # 1. LLM scoring (~10s)
        scored = llm_score(essays)
        profile = to_percentages(scored)

        # 2. content embedding (only if career embeddings exist)
        user_emb = None
        used_content = False
        if EMB_NPY.exists():
            user_emb = embed_essays(essays)        # cached after first run
            used_content = True

        # 3. combined matching — fetch FULL ranking (for the career-map plot),
        #    then slice top 10 for the primary list.
        all_matches = do_match(profile, top_n=2000, user_embedding=user_emb)

        st.session_state.result = {
            "scored": scored,
            "profile": profile,
            "matches": all_matches[:20],
            "all_matches": all_matches,            # used by the career-map chart
            "used_content": used_content,
        }

        # 4. persist to Supabase (fail-soft — never blocks the user)
        try:
            duration = None
            if st.session_state.get("started_at"):
                duration = int(time.time() - st.session_state.started_at)

            # build per-question analytics
            word_counts = {
                qid: len(text.split())
                for qid, text in st.session_state.answers.items()
            }
            char_counts = {
                qid: len(text)
                for qid, text in st.session_state.answers.items()
            }
            # compute time spent on each question (gap between submits)
            time_per_q = {}
            prev_t = st.session_state.get("started_at")
            for qid, _, _ in QUESTIONS:
                t = st.session_state.q_timings.get(qid)
                if t and prev_t:
                    time_per_q[qid] = int(t - prev_t)
                    prev_t = t

            metadata = {
                "word_counts": word_counts,
                "char_counts": char_counts,
                "time_per_question_seconds": time_per_q,
                "question_visits": dict(st.session_state.q_visits),  # how often they went back
                "used_content_embedding": used_content,
                "top_intelligence": sorted(
                    profile.items(), key=lambda kv: -kv[1]
                )[0][0],
                "started_at_epoch": st.session_state.get("started_at"),
                "finished_at_epoch": time.time(),
            }

            # Pass existing_id so we UPDATE the pending row created on Begin
            # rather than inserting a duplicate. Falls back to insert if the
            # pending row is somehow missing OR if db.save_session is a stale
            # version that doesn't accept existing_id.
            save_kwargs = dict(
                answers=st.session_state.answers,
                profile=profile,
                scored=scored,
                matches=all_matches[:20],
                email=None,             # captured later on results screen
                user_agent=_get_user_agent(),
                duration_seconds=duration,
                metadata=metadata,
            )
            try:
                saved_id = db.save_session(
                    **save_kwargs,
                    existing_id=st.session_state.get("saved_id"),
                )
            except TypeError:
                # stale db.py — fall back to old signature
                saved_id = db.save_session(**save_kwargs)
            if saved_id:
                st.session_state.saved_id = saved_id
                # ensure URL still has the id (in case it was an insert fallback)
                try:
                    st.query_params["s"] = saved_id
                except Exception:
                    pass
        except Exception as e:
            # explicitly silent — the user must not see DB errors
            print(f"[app] save_session error: {e}")

        go("results")
    except Exception as e:
        st.error(f"Something went wrong while reading your answers:\n\n```\n{e}\n```")
        if st.button("Try again"):
            go("loading")


def screen_results() -> None:
    r = st.session_state.result
    if not r:
        go("welcome")
        return

    st.markdown(f"<h2 style='margin-bottom:0.2rem;'>your profile</h2>",
                unsafe_allow_html=True)
    st.markdown("<p style='color:#666; margin-bottom:2rem;'>"
                "how your eight intelligences balance, based on what you wrote."
                "</p>", unsafe_allow_html=True)

    # sort intelligences by score (descending) for the chart
    sorted_intel = sorted(INTEL, key=lambda k: r["profile"][k], reverse=True)
    max_pct = max(r["profile"].values()) or 1.0
    for k in sorted_intel:
        pct = r["profile"][k]
        width = (pct / max_pct) * 100
        st.markdown(
            f"<div class='profile-row'>"
            f"  <div class='profile-label'>{PRETTY_LABEL[k]}</div>"
            f"  <div class='profile-bar-bg'>"
            f"    <div class='profile-bar-fg' style='width:{width:.1f}%;'></div>"
            f"  </div>"
            f"  <div class='profile-pct'>{pct:.1f}%</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='margin:3rem 0; border:none; border-top:1px solid #ddd;'>",
                unsafe_allow_html=True)
    st.markdown(f"<h2 style='margin-bottom:0.2rem;'>what we saw in you</h2>",
                unsafe_allow_html=True)
    st.markdown("<p style='color:#666; margin-bottom:2rem;'>"
                "the specific signals from your answers."
                "</p>", unsafe_allow_html=True)

    for k in sorted_intel:
        s = r["scored"][k]
        if s["score"] >= 4:   # only show evidence for intelligences that scored meaningfully
            st.markdown(
                f"<div class='evidence-card'>"
                f"  <div class='evidence-intel'>{PRETTY_LABEL[k]} · {s['score']}/10</div>"
                f"  <div class='evidence-text'>{s['evidence']}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ---- career map (2D scatter with labeled top-10) ----
    if r.get("all_matches"):
        st.markdown("<hr style='margin:3rem 0; border:none; border-top:1px solid #ddd;'>",
                    unsafe_allow_html=True)
        st.markdown(f"<h2 style='margin-bottom:0.2rem;'>your career map</h2>",
                    unsafe_allow_html=True)
        st.markdown("<p style='color:#666; margin-bottom:1.5rem;'>"
                    "every career we know, plotted on two axes — how well it "
                    "fits the shape of your mind (horizontal), and how close "
                    "its day-to-day matches what you wrote about (vertical). "
                    "your top 20 are labeled. upper-right = best fit. "
                    "triangles are your top 5, circles are 6-20."
                    "</p>", unsafe_allow_html=True)
        render_career_map(r["all_matches"], r["matches"])

    st.markdown("<hr style='margin:3rem 0; border:none; border-top:1px solid #ddd;'>",
                unsafe_allow_html=True)
    st.markdown(f"<h2 style='margin-bottom:0.2rem;'>careers that match the shape of your mind</h2>",
                unsafe_allow_html=True)
    st.markdown("<p style='color:#666; margin-bottom:2rem;'>"
                "people whose profession asks for the same blend of intelligences you have."
                "</p>", unsafe_allow_html=True)

    for i, m in enumerate(r["matches"], 1):
        st.markdown(
            f"<div class='career-row'>"
            f"  <div class='career-rank'>#{i:02d}</div>"
            f"  <div class='career-title'>{m['title']}</div>"
            f"  <div class='career-match'>{m['match_pct']:.1f}% match</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ---- email capture for report sending ----
    st.markdown("<hr style='margin:3rem 0; border:none; border-top:1px solid #ddd;'>",
                unsafe_allow_html=True)
    st.markdown(f"<h2 style='margin-bottom:0.2rem;'>get this report by email</h2>",
                unsafe_allow_html=True)
    st.markdown("<p style='color:#666; margin-bottom:1.5rem;'>"
                "Want this report sent to you as a PDF? Enter your email below. "
                "Each email can only be used once."
                "</p>", unsafe_allow_html=True)

    if st.session_state.get("email_sent"):
        st.markdown(
            "<div style='padding:1rem 1.2rem; background:#eaf5ea; "
            "border-left:3px solid #2e8b57; color:#1a4f1a; "
            "border-radius:4px; line-height:1.5;'>"
            "<b>✓ Got it.</b> Your report request has been recorded. "
            "We'll email it to you shortly."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        email_input = st.text_input(
            label="email",
            key="results_email_input",
            label_visibility="collapsed",
            placeholder="you@example.com",
        )
        if st.session_state.get("email_error"):
            st.markdown(
                f"<div style='margin-top:0.5rem; padding:0.8rem 1rem; "
                f"background:#fbeaea; border-left:3px solid #c5443d; "
                f"color:#7a2a25; border-radius:4px; line-height:1.5;'>"
                f"{st.session_state.email_error}"
                f"</div>",
                unsafe_allow_html=True,
            )
        if st.button("Send report to my email", key="send_report_btn"):
            sid = st.session_state.get("saved_id")
            if not sid:
                st.session_state.email_error = (
                    "Couldn't record this — your session wasn't saved. "
                    "Please try again."
                )
                st.rerun()
            else:
                ok, msg = db.request_report_email(sid, email_input)
                if ok:
                    st.session_state.email_sent = True
                    st.session_state.email_error = None
                else:
                    st.session_state.email_error = msg
                st.rerun()

    st.markdown("<br><br>", unsafe_allow_html=True)
    if st.button("Start over", key="restart_btn"):
        st.session_state.step = "welcome"
        st.session_state.answers = {}
        st.session_state.result = None
        st.session_state.saved_id = None
        st.session_state.email_sent = False
        st.session_state.email_error = None
        st.session_state.q_timings = {}
        st.session_state.q_visits = {}
        _clear_storage()
        st.rerun()


# ---------------- router ----------------

step = st.session_state.step
if step == "welcome":
    screen_welcome()
elif step.startswith("q"):
    try:
        idx = int(step[1:])
        if 0 <= idx < len(QUESTIONS):
            screen_question(idx)
        else:
            go("welcome")
    except ValueError:
        go("welcome")
elif step == "loading":
    screen_loading()
elif step == "results":
    # if user refreshed on results, result/all_matches are gone from memory
    # because we don't persist big payloads. re-run the pipeline silently
    # by routing back to loading, which uses the saved answers.
    if not st.session_state.get("result"):
        if st.session_state.get("answers"):
            go("loading")
        else:
            go("welcome")
    else:
        screen_results()
else:
    go("welcome")
