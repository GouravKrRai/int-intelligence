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


# ---------------- session state + browser persistence ----------------

# Browser localStorage bridge so refresh doesn't blow away progress.
# Falls back to in-memory-only if the package isn't available.
try:
    from streamlit_local_storage import LocalStorage
    _LS = LocalStorage()
    _LS_OK = True
except Exception:
    _LS = None
    _LS_OK = False

_LS_KEY = "int_intelligence_state_v1"

# Fields we persist across refresh. NOT result/all_matches — those are big
# and can be re-rendered from saved_id if needed.
_PERSIST_FIELDS = ("step", "answers", "saved_id", "started_at",
                   "q_timings", "q_visits", "email_sent")


def _hydrate_from_storage() -> None:
    """Restore state from browser localStorage on first render of a session.
    Only runs once per browser session. Safe to call repeatedly."""
    if st.session_state.get("_hydrated"):
        return
    st.session_state._hydrated = True   # mark even if hydration fails

    if not _LS_OK:
        return
    try:
        import json
        raw = _LS.getItem(_LS_KEY)
        if not raw:
            return
        data = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(data, dict):
            return
        for k in _PERSIST_FIELDS:
            if k in data and data[k] is not None:
                st.session_state[k] = data[k]
    except Exception:
        pass


def _save_to_storage() -> None:
    """Snapshot the lightweight state fields to localStorage. Called after
    every state mutation (step transition or answer save)."""
    if not _LS_OK:
        return
    try:
        import json
        snapshot = {k: st.session_state.get(k) for k in _PERSIST_FIELDS}
        _LS.setItem(_LS_KEY, json.dumps(snapshot))
    except Exception:
        pass


def _clear_storage() -> None:
    """Wipe the localStorage entry — called on Start Over."""
    if not _LS_OK:
        return
    try:
        _LS.deleteItem(_LS_KEY)
    except Exception:
        pass


# Default session_state values (used only if localStorage has no saved entry)
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

# Try to hydrate from browser localStorage. If found, this overrides the
# defaults above. Must run after the defaults are set so missing keys exist.
_hydrate_from_storage()


def go(step: str) -> None:
    st.session_state.step = step
    _save_to_storage()
    st.rerun()


# ---------------- career-map chart ----------------

def render_career_map(all_matches: list[dict], top_matches: list[dict]) -> None:
    """Scatter of all careers in cosine space. Top 10 highlighted with a warm
    heat gradient (deep red = #1, fading to gold = #10). No leader lines, no
    side labels — the career list below the chart serves as the legend.
    """
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams["font.family"] = "Georgia, serif"

    xs_all = [m.get("gardner_cos", 0.0) for m in all_matches]
    ys_all = [m.get("content_cos", 0.0) for m in all_matches]

    top_pts = [(i, m["title"],
                m.get("gardner_cos", 0.0),
                m.get("content_cos", 0.0))
               for i, m in enumerate(top_matches, 1)]

    fig, ax = plt.subplots(figsize=(9.5, 6.0), dpi=140)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # soft background cloud — every other career
    ax.scatter(xs_all, ys_all, s=10, c="#e6e6e2", alpha=0.55,
               edgecolors="none", zorder=1)

    x_min, x_max = min(xs_all), max(xs_all)
    y_min, y_max = min(ys_all), max(ys_all)
    x_pad = (x_max - x_min) * 0.08
    y_pad = (y_max - y_min) * 0.12
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    # warm heat gradient: deep red (#1) → orange → gold (#10)
    heat_colors = [
        "#b8312b",  # 1 — deep red
        "#cb4a30",  # 2
        "#dc6034",  # 3
        "#e9763a",  # 4
        "#f08c42",  # 5
        "#f3a049",  # 6
        "#f3b455",  # 7
        "#efc665",  # 8
        "#e8d27c",  # 9
        "#dfd793",  # 10 — soft gold
    ]

    # dot sizes shrink slightly with rank for visual hierarchy
    sizes = [340, 290, 260, 235, 215, 200, 185, 175, 165, 155]

    # soft halo behind #1 so it pops even when buried in the cluster
    _, _, x1, y1 = top_pts[0]
    ax.scatter([x1], [y1], s=900, c="#b8312b", alpha=0.12,
               edgecolors="none", zorder=2)

    # paint from rank 10 down to 1 so #1 sits on top of any overlaps
    for (rank, _, x, y), color, size in reversed(list(zip(top_pts, heat_colors, sizes))):
        ax.scatter([x], [y], s=size, c=color, edgecolors="white",
                   linewidths=2.0, zorder=3 + (10 - rank))
        ax.annotate(str(rank), (x, y), ha="center", va="center",
                    fontsize=11, color="white", weight="bold",
                    zorder=20 + (10 - rank))

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

    # subtle "best match" pointer in upper-right corner
    ax.annotate("best match",
                xy=(0.99, 0.97), xycoords="axes fraction",
                ha="right", va="top",
                fontsize=8, color="#b8312b", style="italic",
                weight="bold")

    fig.tight_layout()
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
            "matches": all_matches[:10],
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

            saved_id = db.save_session(
                answers=st.session_state.answers,
                profile=profile,
                scored=scored,
                matches=all_matches[:10],
                email=None,             # captured later on results screen
                user_agent=_get_user_agent(),
                duration_seconds=duration,
                metadata=metadata,
            )
            st.session_state.saved_id = saved_id
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
                    "your top 10 are labeled. upper-right = best fit."
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
