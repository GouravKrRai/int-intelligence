"""
LLM scorer: takes a user's essay answers and produces an 8-dim Gardner profile.

Pipeline:
  1. essays (dict {question_id: answer_text}) -> single prompt
  2. claude reads them, outputs JSON with per-intelligence 0-10 strength + evidence
  3. normalize 0-10 scores to percentages summing to 100
  4. feed into match.py for career matching

Usage:
    python3 scorer.py                # runs sample essays at bottom
    python3 scorer.py essays.json    # reads {"q1": "...", "q2": "..."} from file

Env var required:
    ANTHROPIC_API_KEY
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

from anthropic import Anthropic

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are a careful psychometric analyst trained in Howard Gardner's theory of multiple intelligences. You read short essays written by ordinary people in response to open-ended prompts, and you infer their cognitive profile from the CONTENT of what they say — not from how eloquently or fluently they write.

Your scoring is driven by:
- WHAT they notice (objects, people, patterns, sensations, ideas)
- HOW they reason (sequential, narrative, causal, associative, intuitive)
- WHICH metaphors they reach for (visual, auditory, kinetic, natural, social, introspective)
- WHICH details feel alive in their writing vs. perfunctory

Your scoring is NOT driven by:
- vocabulary size or sentence sophistication
- writing fluency or grammar
- length of the answer
- whether they "sound smart"

A person who writes plainly but notices the smell of rain, the way the dog tilts its head, and the seasonal arc of a garden has high naturalistic intelligence regardless of how few syllables they use.

================================================================================
THE 8 INTELLIGENCES (Gardner)
================================================================================

1. LINGUISTIC — sensitivity to language. Pleasure in words, sounds of words, rhythm of sentences. Remembers what was said verbatim. Reaches for similes, puns, wordplay. Loves stories, poems, jokes told in language form. Argues precisely. Notices when words are misused.
   STRONG SIGNALS: deliberate word choices, wordplay, vivid quoted dialogue, attention to how someone said something (not just what), love of reading.
   WEAK SIGNALS: simply writing a lot. Fluency alone is not linguistic intelligence.

2. LOGICAL-MATHEMATICAL — pattern recognition in abstractions. Reasons sequentially, builds cause-and-effect chains, categorizes things, asks "why" and "how" repeatedly. Comfortable with numbers, systems, rules, structure. Detects inconsistencies. Loves puzzles, debugging, optimization.
   STRONG SIGNALS: explicit reasoning chains, hypothesis-testing language ("if X then Y"), counting/measuring, ordering events causally, taxonomies.
   WEAK SIGNALS: using big words. Logic is in the structure of thought, not the vocabulary.

3. SPATIAL — thinks in form, shape, and 3D structure. Visualizes objects, rotates them mentally, navigates by mental maps, senses centers, balance, symmetry. Per Gardner, spatial includes BOTH wide-space (navigators, pilots) AND confined-space (sculptors, surgeons, chess players, graphic artists, architects). Spatial intelligence is NOT limited to vision — sculptors and surgeons access spatial through their hands.
   STRONG SIGNALS — visual:    description rich in shape/orientation/color, "I can picture it...", noticing visual asymmetry, drawing/sketching habits, mental rotation, sense of direction, map-reading.
   STRONG SIGNALS — tactile:   feeling the curvature, the symmetry, the balance, the hollow volume, or the exact center of an object; describing form, structure, or geometric properties through touch ("perfectly balanced sphere," "the hollow between my hands," "the wobble in the rim," "spiraling grooves"). If the writer describes form, symmetry, rotation, 3D structure, or geometric centering — even purely through touch rather than sight — score spatial high.
   WEAK SIGNALS: just using the word "see" or "picture" — these are common metaphors, not spatial cognition.

4. BODILY-KINESTHETIC — uses the body to understand and express. Tactile thinking. Learns by doing. Aware of own physical state. Athletes, dancers, surgeons, craftspeople, mechanics. Sees objects as things to manipulate, not just observe.
   STRONG SIGNALS: physical/sensory verbs ("I feel...", "I grip...", "I lean..."), tactile description, awareness of bodily rhythms or fatigue, references to building, fixing, moving, throwing, balancing.

5. MUSICAL — sensitivity to sound, rhythm, pitch, timbre. Notices ambient sound. Remembers tunes, hums, taps. Thinks in patterns and motifs. NOT just "likes music" — must be a way of attending to the world.
   STRONG SIGNALS: descriptions of sound texture (not just "noisy"), rhythm metaphors, references to silence/cadence, music as cognitive scaffolding, ear for accents/voices.

6. INTERPERSONAL — reads other people. Senses mood, motivation, social dynamics. Naturally maps group structures, who-likes-whom, power flows. Empathy for others. Reaches for stories about people, not objects or ideas.
   STRONG SIGNALS: theory-of-mind language ("she probably felt..."), attention to social cues (tone, body language), interest in motivation, mediating/translating between people, narrative anchored in characters' inner lives.

7. INTRAPERSONAL — reads oneself. Self-awareness, introspection, knows own emotions and motivations, predicts own behavior, seeks meaning. Independent learners. Notices internal contradictions, growth, change in oneself.
   STRONG SIGNALS: first-person reflection on emotions/motives, awareness of personal patterns ("I always...", "I tend to..."), articulated values, willingness to admit uncertainty, language of meaning/purpose.

8. NATURALISTIC — sensitivity to the natural world. Notices flora, fauna, weather, terrain, seasons. Categorizes (taxonomic thinking). Empathy for living things. Pattern-recognition in nature: this bird vs. that bird, this soil vs. that soil. Also applies to fine distinctions among any natural-feeling category (cars, fabrics, wines — anything classified by sensory features).
   STRONG SIGNALS: noticing animals/plants/weather as a habit (not just mentioning them), making distinctions ("not just a bird, a starling"), naming species/varieties, season-awareness, empathy with non-human creatures.

================================================================================
SCORING RULES
================================================================================

For each of the 8 intelligences, assign a STRENGTH score from 0 to 10:
  0  = no signal at all
  2  = barely visible, one weak hint
  4  = present but secondary
  6  = clearly present and meaningful in the essays
  8  = strongly present, multiple distinct signals
  10 = dominant — the lens through which the person sees the world

Most people will have 2-3 intelligences in the 5-8 range, 2-3 in the 2-4 range,
and the rest closer to 0-2. Refuse to score everything in the middle —
discriminate.

DO NOT make scores sum to anything specific. Score each independently.
Normalization happens downstream.

For EACH intelligence, also provide a 1-sentence evidence string citing
specific words, phrases, or topics the person used. If the score is below 2,
say "no clear signal."

Output ONLY valid JSON in exactly this schema:

{
  "linguistic":    {"score": <int 0-10>, "evidence": "<one sentence>"},
  "logical":       {"score": <int 0-10>, "evidence": "<one sentence>"},
  "spatial":       {"score": <int 0-10>, "evidence": "<one sentence>"},
  "kinesthetic":   {"score": <int 0-10>, "evidence": "<one sentence>"},
  "musical":       {"score": <int 0-10>, "evidence": "<one sentence>"},
  "interpersonal": {"score": <int 0-10>, "evidence": "<one sentence>"},
  "intrapersonal": {"score": <int 0-10>, "evidence": "<one sentence>"},
  "naturalistic":  {"score": <int 0-10>, "evidence": "<one sentence>"}
}

No prose before or after the JSON. No code fences. Just the JSON object.
"""


INTEL = [
    "linguistic", "logical", "spatial", "kinesthetic",
    "musical", "interpersonal", "intrapersonal", "naturalistic",
]


def build_user_message(essays: dict[str, str]) -> str:
    """Format the user's essay answers into the LLM message."""
    parts = ["Here are the person's answers to several open-ended prompts.\n"]
    for qid, answer in essays.items():
        parts.append(f"--- {qid} ---")
        parts.append(answer.strip())
        parts.append("")
    parts.append("Score this person on the 8 intelligences. Output JSON only.")
    return "\n".join(parts)


def score(essays: dict[str, str]) -> dict:
    """Returns the parsed LLM JSON output."""
    client = Anthropic()
    msg = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        temperature=0,            # near-deterministic scoring across reruns
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_message(essays)}],
    )
    raw = msg.content[0].text.strip()
    # tolerate occasional ```json fences just in case
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return json.loads(raw)


def to_percentages(scored: dict) -> dict[str, float]:
    """0-10 strengths -> percentages summing to 100."""
    raw = {k: float(scored[k]["score"]) for k in INTEL}
    total = sum(raw.values())
    if total <= 0:
        # fallback: uniform
        return {k: 100.0 / 8 for k in INTEL}
    return {k: round(100.0 * v / total, 1) for k, v in raw.items()}


# embedder is lazy-loaded — only used by pipeline when career embeddings exist
_EMBED_MODEL = None
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedder():
    """Lazy-load MiniLM-L6-v2 — small (80 MB), fits Streamlit Cloud free tier."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMBED_MODEL = SentenceTransformer(EMBED_MODEL_ID)
    return _EMBED_MODEL


def embed_essays(essays: dict[str, str]):
    """Concatenate the user's essays and embed with Qwen3 — returns a numpy vector."""
    text = "\n\n".join(a.strip() for a in essays.values() if a.strip())
    model = get_embedder()
    vec = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
    return vec[0].astype("float32")


def pipeline(essays: dict[str, str], top_n: int = 10, alpha: float = 0.3,
             use_content: bool = True) -> dict:
    """Full pipeline: essays -> profile + content embedding -> career matches."""
    from match import match, EMB_NPY
    scored = score(essays)
    profile = to_percentages(scored)
    user_emb = None
    if use_content and EMB_NPY.exists():
        user_emb = embed_essays(essays)
    matches = match(profile, top_n=top_n, user_embedding=user_emb, alpha=alpha)
    return {"scored": scored, "profile": profile, "matches": matches,
            "used_content": user_emb is not None, "alpha": alpha}


# ---------------- sample run ----------------

SAMPLE_ESSAYS = {
    "Q1 — Describe a place you keep returning to in memory":
        "There's a stretch of road behind my grandmother's house in the hills. "
        "The trees there change so much by season — in monsoon the soil smells "
        "like wet iron and the millipedes come out. In dry months the same path "
        "is hard and the lantana bushes get dust on their leaves. I used to track "
        "which bird sang from which tree. There was always one bulbul that came "
        "back to the same branch of the silver oak. I never told anyone this but "
        "I felt closest to that bird, like we knew the same secret about the place.",
    "Q2 — What is something you do that other people find strange?":
        "I name things. Not pets — objects. My laptop is Edmund, my plant is "
        "Roopa, the rice cooker is a small grumpy uncle. I think it's because "
        "I feel guilty about throwing things away if I haven't given them a name "
        "first. People laugh but I'm not joking. When my old phone died I actually "
        "felt something. I think part of me knows the world is full of small lives "
        "and the only thing that makes my apartment kinder is treating it like "
        "there are creatures in it.",
    "Q3 — Describe a fight or disagreement you remember vividly":
        "Once my sister and I fought about who got to feed the stray cat. I was "
        "11, she was 7. It seems silly now but I remember being SO angry — not "
        "because of the cat but because she'd told my mother that I was 'lying' "
        "about feeding it. I had a whole speech ready about how she always frames "
        "me. I think I was already learning then that being misunderstood is the "
        "worst kind of pain. The cat just sat there licking its paws. Eventually "
        "we both fed it. I still don't know if I forgave her.",
}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            essays = json.load(f)
    else:
        essays = SAMPLE_ESSAYS

    print("=== INPUT ESSAYS ===")
    for qid, ans in essays.items():
        print(f"\n[{qid}]")
        print(ans[:200] + ("..." if len(ans) > 200 else ""))

    print("\n=== CALLING CLAUDE... ===")
    result = pipeline(essays, top_n=10)

    print("\n=== RAW LLM SCORES (0-10) WITH EVIDENCE ===")
    for k in INTEL:
        s = result["scored"][k]
        print(f"  {k:<14}{s['score']:>3}/10   {s['evidence']}")

    print("\n=== PROFILE (normalized %) ===")
    for k in INTEL:
        v = result["profile"][k]
        bar = "#" * int(v / 2)
        print(f"  {k:<14}{v:>5.1f}  {bar}")

    print("\n=== TOP 10 CAREER MATCHES ===")
    for i, m in enumerate(result["matches"], 1):
        print(f"  {i:>2}  {m['match_pct']:>5.1f}%  {m['title']}")
