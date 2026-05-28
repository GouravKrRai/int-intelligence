"""
Build content embeddings for all 894 O*NET occupations.

For each occupation, we concatenate:
  - title
  - description
  - top 10 tasks (by importance)
  - top 10 work activities (by importance)

into a single text. We embed that text with sentence-transformers' MiniLM-L6-v2
(384-dim, 512-token context) and save the embeddings matrix + a soc->row index.

Chosen for deployment: 80 MB model, ~120 MB RAM at runtime, fits in 1 GB
Streamlit Cloud free tier. MTEB ~64 vs Qwen3-0.6B's ~69 — ~5% quality drop.

Run once after gardner_map.py. Outputs:
  career_embeddings.npy        — (n_occupations, 384) float32 matrix
  career_embeddings_index.csv  — soc, title, row_index
"""
from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DB = Path(__file__).parent / "db_30_3_text"
OUT_NPY = Path(__file__).parent / "career_embeddings.npy"
OUT_IDX = Path(__file__).parent / "career_embeddings_index.csv"


def load_occupations() -> dict[str, dict]:
    """soc -> {title, description}"""
    out = {}
    with open(DB / "Occupation Data.txt", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            out[row[0]] = {"title": row[1], "description": row[2] if len(row) > 2 else ""}
    return out


def load_tasks() -> dict[str, list[str]]:
    """soc -> list of task statement strings, sorted by frequency/importance desc, top 10."""
    # Task Ratings has IM scores per task; Task Statements has the text.
    # Join them to get top tasks per occupation.
    statements = {}  # task_id -> text
    with open(DB / "Task Statements.txt", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        h = next(r)
        c_soc = h.index("O*NET-SOC Code")
        c_id = h.index("Task ID")
        c_text = h.index("Task")
        for row in r:
            statements[(row[c_soc], row[c_id])] = row[c_text]

    # Task Ratings — pick rows where Scale ID == "IM"
    ratings = defaultdict(list)  # soc -> [(im, task_id)]
    with open(DB / "Task Ratings.txt", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        h = next(r)
        c_soc = h.index("O*NET-SOC Code")
        c_id = h.index("Task ID")
        c_scale = h.index("Scale ID")
        c_val = h.index("Data Value")
        for row in r:
            if row[c_scale] != "IM":
                continue
            try:
                im = float(row[c_val])
            except ValueError:
                continue
            ratings[row[c_soc]].append((im, row[c_id]))

    top_tasks = {}
    for soc, lst in ratings.items():
        lst.sort(reverse=True)
        texts = []
        for im, tid in lst[:10]:
            t = statements.get((soc, tid))
            if t:
                texts.append(t)
        top_tasks[soc] = texts
    return top_tasks


def load_work_activities() -> dict[str, list[str]]:
    """soc -> list of work activity names with high IM (top 10)."""
    ratings = defaultdict(list)  # soc -> [(im, name)]
    with open(DB / "Work Activities.txt", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        h = next(r)
        c_soc = h.index("O*NET-SOC Code")
        c_name = h.index("Element Name")
        c_scale = h.index("Scale ID")
        c_val = h.index("Data Value")
        for row in r:
            if row[c_scale] != "IM":
                continue
            try:
                im = float(row[c_val])
            except ValueError:
                continue
            ratings[row[c_soc]].append((im, row[c_name]))

    out = {}
    for soc, lst in ratings.items():
        lst.sort(reverse=True)
        out[soc] = [name for _, name in lst[:10]]
    return out


def main():
    print("loading O*NET tables...")
    occs = load_occupations()
    tasks = load_tasks()
    activities = load_work_activities()

    # build text per soc
    soc_text = {}
    for soc, meta in occs.items():
        parts = [meta["title"], meta.get("description", "")]
        if soc in tasks:
            parts.append("Key tasks: " + "; ".join(tasks[soc]))
        if soc in activities:
            parts.append("Work activities: " + ", ".join(activities[soc]))
        text = " ".join(p for p in parts if p)
        soc_text[soc] = text

    print(f"prepared text for {len(soc_text)} occupations")

    # device + dtype selection for M-series Macs
    if torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
        print("using MPS (Apple Silicon Metal) with float16")
    elif torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
        print("using CUDA with float16")
    else:
        device = "cpu"
        dtype = torch.float32
        print("using CPU with float32 (slow)")

    print(f"loading {EMBED_MODEL} (downloads ~80 MB on first run)...")
    # MiniLM is a small encoder; no need for fp16 or special tokenizer config
    model = SentenceTransformer(EMBED_MODEL, device=device)

    socs = sorted(soc_text.keys())
    texts = [soc_text[s] for s in socs]
    print(f"embedding {len(texts)} career descriptions...")
    embeddings = model.encode(
        texts,
        batch_size=64,          # small model, big batches
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embeddings = embeddings.astype(np.float32)

    np.save(OUT_NPY, embeddings)
    with open(OUT_IDX, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row_index", "soc", "title"])
        for i, soc in enumerate(socs):
            w.writerow([i, soc, occs[soc]["title"]])

    print(f"wrote {OUT_NPY}  (shape {embeddings.shape})")
    print(f"wrote {OUT_IDX}")


if __name__ == "__main__":
    main()
