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


# ---------------- session state ----------------

if "step" not in st.session_state:
    st.session_state.step = "welcome"   # welcome | q0..q6 | loading | results
    st.session_state.answers = {}        # {qid: text}
    st.session_state.result = None       # full pipeline result


def go(step: str) -> None:
    st.session_state.step = step
    st.rerun()


# ---------------- career-map chart ----------------

def render_career_map(all_matches: list[dict], top_matches: list[dict]) -> None:
    """Scatter plot of all 1016 careers on (cognitive-shape, content-meaning) axes.
    Top 10 highlighted with labels. The user is at (1.0, 1.0) — perfect match w/ self.
    """
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    # we need the rank-normalized coords from each match
    # (gardner_rank, content_rank) are added when match() runs with normalize="rank"
    xs_all, ys_all = [], []
    for m in all_matches:
        xs_all.append(m.get("gardner_rank", 0.0))
        ys_all.append(m.get("content_rank", 0.0))

    # ordered list of top 10 with their rank numbers
    top_ranked = [(i, m) for i, m in enumerate(top_matches, 1)]

    rcParams["font.family"] = "Georgia, serif"
    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=140)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # all careers — light cloud
    ax.scatter(xs_all, ys_all, s=10, c="#d8d8d4", alpha=0.55,
               edgecolors="none", zorder=1)

    # top matches — dark, larger
    xs_top = [m.get("gardner_rank", 0.0) for _, m in top_ranked]
    ys_top = [m.get("content_rank", 0.0) for _, m in top_ranked]
    ax.scatter(xs_top, ys_top, s=140, c="#222", edgecolors="white",
               linewidths=1.5, zorder=3)

    # rank number INSIDE each top dot (clean, no overlap)
    for (i, m), x, y in zip(top_ranked, xs_top, ys_top):
        ax.annotate(str(i), (x, y), ha="center", va="center",
                    fontsize=9, color="white", weight="bold", zorder=4)

    # user is the implicit (1, 1) — show the "you" anchor
    ax.scatter([1.0], [1.0], s=200, marker="*", c="#c5443d",
               edgecolors="white", linewidths=1.5, zorder=5)
    ax.annotate("you", (1.0, 1.0), xytext=(10, -3), textcoords="offset points",
                fontsize=11, color="#c5443d", weight="bold", zorder=5)

    # axis cosmetics
    ax.set_xlim(-0.02, 1.08)
    ax.set_ylim(-0.02, 1.08)
    ax.set_xlabel("cognitive shape match  →", fontsize=10, color="#444",
                  labelpad=8)
    ax.set_ylabel("content / interest match  →", fontsize=10, color="#444",
                  labelpad=8)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis="both", colors="#888", labelsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#bbb")

    # subtle grid
    ax.grid(True, which="major", linestyle="-", linewidth=0.5,
            color="#eee", zorder=0)

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
        go("q0")


def screen_question(idx: int) -> None:
    qid, label, prompt = QUESTIONS[idx]
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

    # ---- career map (2D scatter of all 1016 careers) ----
    if r.get("all_matches"):
        st.markdown("<hr style='margin:3rem 0; border:none; border-top:1px solid #ddd;'>",
                    unsafe_allow_html=True)
        st.markdown(f"<h2 style='margin-bottom:0.2rem;'>your career map</h2>",
                    unsafe_allow_html=True)
        st.markdown("<p style='color:#666; margin-bottom:1.5rem;'>"
                    "every career we know, plotted on two axes: how well it fits "
                    "the shape of your mind (horizontal), and how close its day-to-day "
                    "matches what you wrote about (vertical). the 10 nearest careers "
                    "are highlighted."
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

    st.markdown("<br><br>", unsafe_allow_html=True)
    if st.button("Start over", key="restart_btn"):
        st.session_state.step = "welcome"
        st.session_state.answers = {}
        st.session_state.result = None
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
    screen_results()
else:
    go("welcome")
