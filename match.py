"""
Career matcher. Given a user's Gardner spectrum (and optionally their essay
content embedding), returns the top N most-similar O*NET occupations.

Two-signal scoring (both deterministic, both pure math):

  1. GARDNER COSINE — cosine similarity between
     - user's own-centered profile (deviation from personal mean of 12.5%)
     - each job's baseline-adjusted profile (deviation from POPULATION mean
       per intelligence)
     Captures "do my COGNITIVE STRENGTHS match what this job actually needs
     above the universal workplace baseline?"

  2. CONTENT COSINE — cosine similarity between
     - user's essays embedded into a 384-dim semantic vector
     - each career's text (title + description + tasks + work activities)
       embedded by the same fixed model
     Captures "is what I'm writing ABOUT close in meaning to what this career
     actually does day-to-day?"

  final_score = alpha * gardner_cosine + (1 - alpha) * content_cosine

Both cosines are deterministic — same input always produces the same output.
Combined score is a fixed weighted sum, no model judgment involved.

Usage:
    python3 match.py                  # runs with sample profile at bottom
    python3 match.py user.json        # reads {"linguistic": 25, ...} from file
"""
from __future__ import annotations
import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean

import numpy as np

INTEL = [
    "linguistic", "logical", "spatial", "kinesthetic",
    "musical", "interpersonal", "intrapersonal", "naturalistic",
]
HERE = Path(__file__).parent
CSV_PATH = HERE / "gardner_profiles.csv"
EMB_NPY = HERE / "career_embeddings.npy"
EMB_IDX = HERE / "career_embeddings_index.csv"
# every 8-dim profile that sums to 100 has mean = 12.5
PROFILE_MEAN = 100.0 / len(INTEL)
# default weight: content embedding gets 70% (better at surfacing latent callings),
# gardner cosine gets 30% (filters out semantic-only noise like "Parking Enforcement").
# tuned across potter/geology/env-law test profiles with rank normalization.
DEFAULT_ALPHA = 0.3


def load_population() -> tuple[list[dict], dict[str, float]]:
    """Returns (rows, baseline) — list of jobs and the population mean per intel."""
    rows = []
    with open(CSV_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            for k in INTEL:
                r[k] = float(r[k])
            rows.append(r)
    baseline = {k: mean(r[k] for r in rows) for k in INTEL}
    return rows, baseline


def user_centered_vec(profile: dict) -> list[float]:
    """User: subtract own mean (12.5) — preserves the 'compare to self' philosophy."""
    return [profile[k] - PROFILE_MEAN for k in INTEL]


def job_baseline_adjusted_vec(profile: dict, baseline: dict[str, float]) -> list[float]:
    """Job: subtract POPULATION mean per intelligence — strips universal workplace
    baseline (every job has ~14% linguistic by default) so the job's distinctive
    requirements become visible.
    """
    return [profile[k] - baseline[k] for k in INTEL]


def cosine(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def load_career_embeddings() -> tuple[np.ndarray, dict[str, int]] | None:
    """Returns (embeddings_matrix, soc->row_index) or None if not built yet."""
    if not (EMB_NPY.exists() and EMB_IDX.exists()):
        return None
    emb = np.load(EMB_NPY)
    idx = {}
    with open(EMB_IDX, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            idx[r["soc"]] = int(r["row_index"])
    return emb, idx


def cosine_np(user_emb: np.ndarray, career_embs: np.ndarray) -> np.ndarray:
    """Cosine similarity between one user vector and a matrix of career vectors.
    Assumes both already L2-normalized (Qwen3 embeds are normalized at encode time).

    np.errstate suppresses spurious "divide by zero / overflow" warnings that
    numpy on macOS (Accelerate BLAS) emits from internal SIMD edge-case checks.
    Verified: the actual output is finite and correct — no NaN, no inf.
    """
    user_n = user_emb / (np.linalg.norm(user_emb) + 1e-12)
    norms = np.linalg.norm(career_embs, axis=1, keepdims=True) + 1e-12
    careers_n = career_embs / norms
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return careers_n @ user_n


def normalize_user(profile: dict) -> dict:
    """Accepts loose input (any sum); rescales to sum to 100."""
    out = {k: float(profile.get(k, 0.0)) for k in INTEL}
    total = sum(out.values())
    if total <= 0:
        raise ValueError("profile sums to 0; need at least one positive value")
    return {k: 100.0 * v / total for k, v in out.items()}


# kept as alias for any external callers that imported the old name
centered_vec = user_centered_vec


def _rankpct(values: list[float]) -> list[float]:
    """Convert raw scores to rank-percentile in [0, 1]. Ties get average rank.
    1.0 = highest in the population, 0.0 = lowest. Makes scales comparable
    when combining gardner cosine (range ~0.5-0.9) with content cosine (~0.1-0.35).
    """
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    pct = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[indexed[j + 1]] == values[indexed[i]]:
            j += 1
        avg_rank = (i + j) / 2.0
        for k in range(i, j + 1):
            pct[indexed[k]] = avg_rank / (n - 1) if n > 1 else 0.5
        i = j + 1
    return pct


def match(user_profile: dict, top_n: int = 10,
          user_embedding: np.ndarray | None = None,
          alpha: float = DEFAULT_ALPHA,
          normalize: str = "rank") -> list[dict]:
    """
    user_profile     : dict mapping intel -> percentage
    user_embedding   : optional Qwen3 vector of the user's essays
    alpha            : weight on gardner vs content. 1.0=gardner, 0.0=content
    normalize        : "rank"   -> convert each cosine to percentile rank
                                   in [0,1] before combining (default; fair).
                       "raw"    -> use raw cosines (gardner ends up dominating
                                   because its values are larger in absolute scale).
    """
    user_profile = normalize_user(user_profile)
    rows, baseline = load_population()
    user_vec = user_centered_vec(user_profile)

    # gardner cosine — always computed
    gardner_cos = [cosine(user_vec, job_baseline_adjusted_vec(r, baseline)) for r in rows]

    # content cosine — only if user_embedding provided AND career embeddings exist
    content_cos = None
    emb_data = load_career_embeddings() if user_embedding is not None else None
    if emb_data is not None:
        career_embs, soc_to_row = emb_data
        career_cosines = cosine_np(np.asarray(user_embedding, dtype=np.float32), career_embs)
        content_cos = []
        for r in rows:
            row_i = soc_to_row.get(r["soc"])
            content_cos.append(float(career_cosines[row_i]) if row_i is not None else 0.0)

    # rank-normalize per axis if requested (default)
    if normalize == "rank":
        g_norm = _rankpct(gardner_cos)
        c_norm = _rankpct(content_cos) if content_cos is not None else None
    else:
        g_norm = gardner_cos
        c_norm = content_cos

    scored = []
    for i, r in enumerate(rows):
        g_raw = gardner_cos[i]
        g_n = g_norm[i]
        if content_cos is not None:
            c_raw = content_cos[i]
            c_n = c_norm[i]
            combined = alpha * g_n + (1 - alpha) * c_n
        else:
            c_raw = c_n = None
            combined = g_n
        # combined is in [0,1] when normalize=rank — rescale to 0-100 directly
        if normalize == "rank":
            match_pct = round(100 * combined, 1)
        else:
            match_pct = round(50 * (combined + 1), 1)
        entry = {
            "soc": r["soc"],
            "title": r["title"],
            "match_pct": match_pct,
            "cos": round(combined, 4),
            "gardner_cos": round(g_raw, 4),
            "gardner_rank": round(g_n, 3),
            "profile_raw": {k: round(r[k], 1) for k in INTEL},
        }
        if c_raw is not None:
            entry["content_cos"] = round(c_raw, 4)
            entry["content_rank"] = round(c_n, 3)
        scored.append(entry)
    scored.sort(key=lambda x: x["cos"], reverse=True)
    return scored[:top_n]


def explain(user_profile: dict, top_n: int = 10) -> None:
    user_profile = normalize_user(user_profile)
    print("=== YOUR PROFILE (raw %) ===")
    for k in INTEL:
        bar = "#" * int(user_profile[k] / 2)
        print(f"  {k:<14}{user_profile[k]:>5.1f}  {bar}")
    print()
    results = match(user_profile, top_n)
    print(f"=== TOP {top_n} CAREER MATCHES ===")
    print(f"{'#':>2}  {'match%':>7}  {'title':<55} soc")
    print("-" * 90)
    for i, r in enumerate(results, 1):
        print(f"{i:>2}  {r['match_pct']:>6.1f}%  {r['title']:<55} {r['soc']}")


# ---------- entry point ----------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            user = json.load(f)
    else:
        # sample: a creative, people-oriented person, light on math
        user = {
            "linguistic":    25,
            "logical":        8,
            "spatial":       15,
            "kinesthetic":    7,
            "musical":       12,
            "interpersonal": 18,
            "intrapersonal": 12,
            "naturalistic":   3,
        }
    explain(user, top_n=15)
