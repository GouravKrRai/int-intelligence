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
// Ensure a mobile-friendly viewport tag is in <head>. Streamlit usually
// sets this, but we guarantee it so the layout doesn't fall back to
// desktop-width emulation on phones.
(function () {
  if (window.parent && window.parent.document) {
    var doc = window.parent.document;
    if (!doc.querySelector('meta[name="viewport"]')) {
      var m = doc.createElement('meta');
      m.name = 'viewport';
      m.content = 'width=device-width, initial-scale=1, viewport-fit=cover';
      doc.head.appendChild(m);
    }
  }
})();
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

/* ---------- mobile (≤640px wide) ---------- */
@media (max-width: 640px) {
  /* tighter page margins so we don't waste 20% of viewport on whitespace */
  .block-container {padding-top: 1.5rem !important;
                    padding-bottom: 2rem !important;
                    padding-left: 1rem !important;
                    padding-right: 1rem !important;}

  /* welcome / section headings — were too dominant on small screens */
  h1 {font-size: 1.85rem !important; line-height: 1.25 !important;}
  h2 {font-size: 1.35rem !important; line-height: 1.3 !important;}

  /* question screen — keep title prominent but smaller */
  .q-tag {font-size: 0.75rem !important; margin-bottom: 0.4rem !important;}
  .q-title {font-size: 1.15rem !important; line-height: 1.35 !important;
            margin-bottom: 1rem !important;}
  .q-prompt {font-size: 0.98rem !important; line-height: 1.55 !important;
             margin-bottom: 1.2rem !important;}

  /* textarea — slightly shorter so the keyboard doesn't push everything
     offscreen on small phones. font-size: 16px prevents iOS auto-zoom. */
  .stTextArea textarea {
      min-height: 180px !important;
      font-size: 16px !important;   /* 16px = no iOS keyboard zoom */
      line-height: 1.5 !important;
      padding: 0.8rem !important;
  }

  /* email input — same 16px trick to disable iOS zoom on focus */
  .stTextInput input {font-size: 16px !important;}

  /* button row — make sure they're full-width and tap-friendly */
  .stButton button {
      padding: 0.7rem 1.4rem !important;
      font-size: 0.95rem !important;
      width: 100% !important;
      min-height: 44px !important;   /* Apple's tap-target minimum */
  }

  /* profile bar layout — labels were 220px fixed, which broke at small width.
     Stack the label above the bar on narrow screens. */
  .profile-row {flex-direction: column !important; align-items: stretch !important;
                margin-bottom: 1rem !important;}
  .profile-label {width: auto !important; margin-bottom: 0.3rem !important;
                  font-size: 0.95rem !important;}
  .profile-pct {width: auto !important; text-align: right !important;
                padding-left: 0 !important; margin-top: 0.2rem !important;
                font-size: 0.85rem !important;}

  /* career row */
  .career-title {font-size: 1.02rem !important;}
  .career-match {font-size: 0.88rem !important;}
}
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
    """Mobile-friendly career-map chart. Specs:
      - PORTRAIT aspect (taller than wide) so it doesn't shrink horribly on phones
      - chart on top, legend BELOW in 2 columns (was 5 columns above)
      - 20 distinct colors (tab20), one per career
      - all circles, dot size = match %
      - cloud of every career in light grey background
      - zoom adapts to the top-20 bounding box
      - axis ticks scaled ×100, no decimals
      - larger fonts than before so they remain legible on small screens
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gs
    from matplotlib import rcParams
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter

    rcParams["font.family"] = "Georgia, serif"

    xs_all = [m.get("gardner_cos", 0.0) for m in all_matches]
    ys_all = [m.get("content_cos", 0.0) for m in all_matches]

    top_pts = [(i, m["title"], m.get("match_pct", 0.0),
                m.get("gardner_cos", 0.0),
                m.get("content_cos", 0.0))
               for i, m in enumerate(top_matches, 1)]

    cmap = plt.get_cmap("tab20")
    colors = [cmap(i / 20.0) for i in range(20)]

    pcts = [pct for _, _, pct, _, _ in top_pts]
    p_min, p_max = min(pcts), max(pcts)
    size_range = p_max - p_min if p_max > p_min else 1.0
    # slightly smaller base size so dots don't dominate the smaller chart area
    sizes = [280 + 1100 * (pct - p_min) / size_range
             for _, _, pct, _, _ in top_pts]

    # PORTRAIT figure: chart on top, legend below.
    # hspace generous so x-axis label doesn't collide with legend title.
    fig = plt.figure(figsize=(9, 11), dpi=140)
    fig.patch.set_facecolor("white")
    grid = gs.GridSpec(2, 1, height_ratios=[1.0, 0.55], figure=fig, hspace=0.18)
    ax = fig.add_subplot(grid[0])
    ax_legend = fig.add_subplot(grid[1])

    # ---- chart (top) ----
    # background cloud
    ax.scatter(xs_all, ys_all, s=10, c="#e6e6e2", alpha=0.55,
               edgecolors="none", zorder=1)
    # top 20 — real cosine positions
    for (rank, _, _, x, y), color, size in zip(top_pts, colors, sizes):
        ax.scatter([x], [y], s=size, c=[color],
                   edgecolors="white", linewidths=1.6,
                   alpha=0.95, zorder=3 + (21 - rank))

    # zoom into top-20 region
    top_xs = [x for _, _, _, x, _ in top_pts]
    top_ys = [y for _, _, _, _, y in top_pts]
    x_min, x_max = min(top_xs), max(top_xs)
    y_min, y_max = min(top_ys), max(top_ys)
    x_pad = max((x_max - x_min) * 0.18, 0.04)
    y_pad = max((y_max - y_min) * 0.25, 0.04)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:.0f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:.0f}"))
    ax.set_xlabel("cognitive shape match  →", fontsize=11,
                  color="#777", labelpad=8)
    ax.set_ylabel("content / interest match  →", fontsize=11,
                  color="#777", labelpad=8)
    ax.tick_params(axis="both", colors="#aaa", labelsize=10)
    ax.set_facecolor("#fafafa")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#e0e0e0")
        ax.spines[spine].set_linewidth(0.8)
    ax.grid(True, linestyle="-", linewidth=0.4, color="#eeeeee", zorder=0)

    # ---- legend (below) ----
    legend_handles = []
    for (rank, title, pct, _, _), color in zip(top_pts, colors):
        # titles can be longer here because we have 2 columns instead of 5
        label = title if len(title) <= 48 else title[:45] + "…"
        legend_handles.append(
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=color, markeredgecolor="white",
                   markersize=11, markeredgewidth=1.0,
                   label=f"{rank:>2}. {label}  ({pct:.1f}%)")
        )
    ax_legend.legend(handles=legend_handles, loc="center", ncol=2,
                     fontsize=10, frameon=False, handletextpad=0.6,
                     columnspacing=1.8, labelspacing=0.9,
                     title="top 20 careers   (dot size = match %)",
                     title_fontsize=11)
    ax_legend.axis("off")

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
                    "your top 20 are colored; each dot's size is its match %. "
                    "upper-right = best fit."
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
        sent_to = st.session_state.get("email_sent_to", "your inbox")
        st.markdown(
            f"<div style='padding:1rem 1.2rem; background:#eaf5ea; "
            f"border-left:3px solid #2e8b57; color:#1a4f1a; "
            f"border-radius:4px; line-height:1.5;'>"
            f"<b>✓ Sent.</b> Your PDF report is on its way to "
            f"<b>{sent_to}</b>. Check your inbox in the next minute or two "
            f"(don't forget the spam folder if it doesn't appear)."
            f"</div>",
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
                    st.session_state.email_sent_to = email_input.strip().lower()
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
