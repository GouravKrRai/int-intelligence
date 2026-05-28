"""
Gardner 8 intelligences -> O*NET 30.3 rosetta stone.

Reads O*NET text files, computes a normalized 8-dim Gardner profile for every
occupation, writes CSV: occupation_code, title, linguistic, logical, spatial,
kinesthetic, musical, interpersonal, intrapersonal, naturalistic (each as %
summing to 100).

Usage:
    python3 gardner_map.py
"""
import csv
from collections import defaultdict
from pathlib import Path

DB = Path(__file__).parent / "db_30_3_text"

# (file, element_name, weight). All scores read as IM (importance, 1-5 scale).
# Default weight 1.0; partial-belonging items use < 1.0.
MAPPING = {
    # Expanded mapping. Each intelligence draws on a broader set of O*NET items,
    # with cross-sharing at reduced weights when an item legitimately serves
    # multiple intelligences. Musical and linguistic remain TIGHT (no generic
    # creativity/memory items) so they don't pollute non-musical/non-verbal
    # craft profiles. Naturalistic is broad to capture taxonomic thinking and
    # observation skills, not just biology/geography knowledge.
    "linguistic": [
        # Strictly word/language-specific (purged of generic creativity).
        ("Abilities.txt",            "Oral Comprehension",                1.0),
        ("Abilities.txt",            "Written Comprehension",             1.0),
        ("Abilities.txt",            "Oral Expression",                   1.0),
        ("Abilities.txt",            "Written Expression",                1.0),
        ("Abilities.txt",            "Speech Clarity",                    0.8),
        ("Abilities.txt",            "Speech Recognition",                0.6),
        ("Abilities.txt",            "Fluency of Ideas",                  0.5),
        ("Essential Skills.txt",     "Reading Comprehension",             1.0),
        ("Essential Skills.txt",     "Active Listening",                  0.8),
        ("Essential Skills.txt",     "Writing",                           1.0),
        ("Essential Skills.txt",     "Speaking",                          1.0),
        ("Knowledge.txt",            "English Language",                  1.0),
        ("Knowledge.txt",            "Foreign Language",                  0.7),
        ("Knowledge.txt",            "Communications and Media",          0.7),
    ],
    "logical": [
        ("Abilities.txt",            "Mathematical Reasoning",            1.0),
        ("Abilities.txt",            "Number Facility",                   1.0),
        ("Abilities.txt",            "Deductive Reasoning",               1.0),
        ("Abilities.txt",            "Inductive Reasoning",               1.0),
        ("Abilities.txt",            "Information Ordering",              0.9),
        ("Abilities.txt",            "Category Flexibility",              0.9),  # categorization is core (Gardner)
        ("Abilities.txt",            "Problem Sensitivity",               0.8),
        ("Essential Skills.txt",     "Mathematics",                       1.0),
        ("Essential Skills.txt",     "Science",                           1.0),
        ("Essential Skills.txt",     "Critical Thinking",                 1.0),
        ("Transferable Skills.txt",  "Complex Problem Solving",           1.0),
        ("Transferable Skills.txt",  "Judgment and Decision Making",      0.9),
        ("Transferable Skills.txt",  "Systems Analysis",                  0.9),
        ("Transferable Skills.txt",  "Systems Evaluation",                0.8),
        ("Transferable Skills.txt",  "Operations Analysis",               0.7),
        ("Knowledge.txt",            "Mathematics",                       1.0),
        ("Knowledge.txt",            "Physics",                           1.0),
        ("Knowledge.txt",            "Chemistry",                         1.0),
        ("Knowledge.txt",            "Computers and Electronics",         0.7),
        ("Knowledge.txt",            "Engineering and Technology",        0.6),
    ],
    "spatial": [
        ("Abilities.txt",            "Spatial Orientation",               1.0),
        ("Abilities.txt",            "Visualization",                     1.0),
        ("Abilities.txt",            "Depth Perception",                  0.9),
        ("Abilities.txt",            "Visual Color Discrimination",       0.9),
        ("Abilities.txt",            "Near Vision",                       0.5),
        ("Abilities.txt",            "Far Vision",                        0.5),
        ("Abilities.txt",            "Originality",                       0.6),
        ("Abilities.txt",            "Perceptual Speed",                  0.5),
        ("Abilities.txt",            "Flexibility of Closure",            0.6),
        ("Abilities.txt",            "Memorization",                      0.4),  # recall visual detail
        ("Abilities.txt",            "Inductive Reasoning",               0.3),  # pattern recognition
        ("Abilities.txt",            "Manual Dexterity",                  0.3),  # sculptors/surgeons
        ("Abilities.txt",            "Finger Dexterity",                  0.3),  # confined-space spatial
        ("Knowledge.txt",            "Design",                            1.0),
        ("Knowledge.txt",            "Fine Arts",                         0.5),  # shared with musical
        ("Transferable Skills.txt",  "Technology Design",                 0.5),
    ],
    "kinesthetic": [
        # all psychomotor (1.A.2.*)
        ("Abilities.txt",            "Arm-Hand Steadiness",               1.0),
        ("Abilities.txt",            "Manual Dexterity",                  1.0),
        ("Abilities.txt",            "Finger Dexterity",                  1.0),
        ("Abilities.txt",            "Control Precision",                 1.0),
        ("Abilities.txt",            "Multilimb Coordination",            1.0),
        ("Abilities.txt",            "Response Orientation",              0.9),
        ("Abilities.txt",            "Rate Control",                      0.9),
        ("Abilities.txt",            "Reaction Time",                     0.9),
        ("Abilities.txt",            "Wrist-Finger Speed",                0.9),
        ("Abilities.txt",            "Speed of Limb Movement",            0.9),
        # all physical (1.A.3.*)
        ("Abilities.txt",            "Static Strength",                   1.0),
        ("Abilities.txt",            "Explosive Strength",                1.0),
        ("Abilities.txt",            "Dynamic Strength",                  1.0),
        ("Abilities.txt",            "Trunk Strength",                    0.9),
        ("Abilities.txt",            "Stamina",                           1.0),
        ("Abilities.txt",            "Extent Flexibility",                0.9),
        ("Abilities.txt",            "Dynamic Flexibility",               0.9),
        ("Abilities.txt",            "Gross Body Coordination",           1.0),
        ("Abilities.txt",            "Gross Body Equilibrium",            1.0),
        ("Transferable Skills.txt",  "Operation and Control",             0.8),
        ("Transferable Skills.txt",  "Equipment Maintenance",             0.6),
        ("Transferable Skills.txt",  "Installation",                      0.6),
        ("Transferable Skills.txt",  "Repairing",                         0.7),
        ("Abilities.txt",            "Spatial Orientation",               0.4),  # dancers
        ("Abilities.txt",            "Originality",                       0.4),  # choreographers
    ],
    "musical": [
        # Strictly auditory + music-specific knowledge only.
        # (Originality / Fluency / Memorization / Inductive / Speech are kept OUT
        # because they are generic abilities that exist in jobs for non-musical
        # reasons. Musicians still dominate this dimension via their unique
        # auditory IM scores.)
        ("Abilities.txt",            "Auditory Attention",                1.0),
        ("Abilities.txt",            "Hearing Sensitivity",               1.0),
        ("Abilities.txt",            "Sound Localization",                0.8),
        ("Knowledge.txt",            "Fine Arts",                         0.3),  # diluted: shared w/ spatial
    ],
    "interpersonal": [
        ("Transferable Skills.txt",  "Social Perceptiveness",             1.0),
        ("Transferable Skills.txt",  "Coordination",                      0.9),
        ("Transferable Skills.txt",  "Persuasion",                        1.0),
        ("Transferable Skills.txt",  "Negotiation",                       1.0),
        ("Transferable Skills.txt",  "Instructing",                       0.9),
        ("Transferable Skills.txt",  "Service Orientation",               0.9),
        ("Transferable Skills.txt",  "Management of Personnel Resources", 1.0),
        ("Work Styles.txt",          "Cooperation",                       0.9),
        ("Work Styles.txt",          "Empathy",                           1.0),
        ("Work Styles.txt",          "Social Orientation",                1.0),
        ("Work Styles.txt",          "Leadership Orientation",            0.9),
        ("Work Styles.txt",          "Sincerity",                         0.6),
        ("Work Styles.txt",          "Humility",                          0.5),
        ("Knowledge.txt",            "Customer and Personal Service",     1.0),
        ("Knowledge.txt",            "Personnel and Human Resources",     0.9),
        ("Knowledge.txt",            "Sociology and Anthropology",        0.9),
        ("Knowledge.txt",            "Psychology",                        0.7),
        ("Knowledge.txt",            "Education and Training",            0.8),
        ("Knowledge.txt",            "Therapy and Counseling",            0.9),
        ("Abilities.txt",            "Oral Comprehension",                0.5),  # listening to people
        ("Abilities.txt",            "Speech Recognition",                0.4),  # tone/affect
        ("Abilities.txt",            "Inductive Reasoning",               0.4),  # reading patterns in people
        ("Work Styles.txt",          "Optimism",                          0.5),
    ],
    "intrapersonal": [
        ("Work Styles.txt",          "Self-Control",                      1.0),
        ("Work Styles.txt",          "Stress Tolerance",                  1.0),
        ("Work Styles.txt",          "Self-Confidence",                   0.9),
        ("Work Styles.txt",          "Adaptability",                      0.9),
        ("Work Styles.txt",          "Perseverance",                      0.9),
        ("Work Styles.txt",          "Initiative",                        0.8),
        ("Work Styles.txt",          "Achievement Orientation",           0.8),
        ("Work Styles.txt",          "Intellectual Curiosity",            0.7),
        ("Work Styles.txt",          "Integrity",                         0.6),
        ("Work Styles.txt",          "Dependability",                     0.6),
        ("Work Styles.txt",          "Tolerance for Ambiguity",           0.8),
        ("Work Styles.txt",          "Cautiousness",                      0.6),
        ("Work Styles.txt",          "Attention to Detail",               0.5),
        ("Essential Skills.txt",     "Active Learning",                   0.8),
        ("Essential Skills.txt",     "Learning Strategies",               0.7),
        ("Essential Skills.txt",     "Monitoring",                        0.6),
        ("Knowledge.txt",            "Psychology",                        0.5),
        ("Knowledge.txt",            "Philosophy and Theology",           0.6),
        ("Work Styles.txt",          "Innovation",                        0.7),  # self-directed
        ("Essential Skills.txt",     "Critical Thinking",                 0.5),  # self-reflection
    ],
    "naturalistic": [
        # core knowledge of the natural world
        ("Knowledge.txt",            "Biology",                           1.0),
        ("Knowledge.txt",            "Geography",                         0.8),
        ("Knowledge.txt",            "Food Production",                   0.9),
        ("Knowledge.txt",            "Chemistry",                         0.4),  # natural chemistry
        ("Knowledge.txt",            "Production and Processing",         0.4),  # agricultural processing
        # taxonomic thinking — the core gardner cognitive skill
        ("Abilities.txt",            "Category Flexibility",              0.9),  # taxonomy itself
        ("Abilities.txt",            "Inductive Reasoning",               0.5),  # patterns in observations
        ("Abilities.txt",            "Memorization",                      0.6),  # species/varieties
        # observation / pattern detection
        ("Abilities.txt",            "Perceptual Speed",                  0.6),  # spotting differences
        ("Abilities.txt",            "Flexibility of Closure",            0.6),  # partial-cue recognition
        ("Abilities.txt",            "Visual Color Discrimination",       0.6),  # species/soil/plant
        ("Abilities.txt",            "Near Vision",                       0.3),
        ("Abilities.txt",            "Far Vision",                        0.3),
        # methodology
        ("Essential Skills.txt",     "Science",                           0.6),
    ],
}


def load_scores(filename: str):
    """Returns {(soc_code, element_name): (combined_score, IM)}.

    combined_score = IM * (LV / 7), giving a 0-5 number that is high only when
    the ability is both required AND demanded at a high level.
    IM is returned separately so the main aggregation can weight items by their
    actual importance to each job (so abilities a job doesn't need barely count).
    Work Styles has only IM (no LV) -> combined falls back to IM alone.
    """
    by_scale = {}
    path = DB / filename
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        c_soc = header.index("O*NET-SOC Code")
        c_name = header.index("Element Name")
        c_scale = header.index("Scale ID")
        c_value = header.index("Data Value")
        for row in reader:
            if row[c_scale] not in ("IM", "LV"):
                continue
            try:
                v = float(row[c_value])
            except ValueError:
                continue
            by_scale.setdefault((row[c_soc], row[c_name]), {})[row[c_scale]] = v

    out = {}
    for key, scales in by_scale.items():
        im = scales.get("IM")
        lv = scales.get("LV")
        if im is None:
            continue
        combined = im if lv is None else im * (lv / 7.0)
        out[key] = (combined, im)
    return out


def load_occupations():
    """Returns {soc_code: title}."""
    out = {}
    with open(DB / "Occupation Data.txt", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            out[row[0]] = row[1]
    return out


def main():
    # load all needed score tables once
    needed_files = set(src[0] for items in MAPPING.values() for src in items)
    tables = {fn: load_scores(fn) for fn in needed_files}
    occupations = load_occupations()

    # IM-weighted average: items the job doesn't need (low IM) barely count;
    # items core to the job (high IM) dominate the average.
    # numerator   = sum(combined_score * IM * mapping_weight)
    # denominator = sum(IM * mapping_weight)
    raw = defaultdict(lambda: defaultdict(float))
    weights_used = defaultdict(lambda: defaultdict(float))
    for intel, items in MAPPING.items():
        for fn, elem, w in items:
            table = tables[fn]
            for (soc, name), (val, im) in table.items():
                if name == elem:
                    raw[soc][intel] += val * im * w
                    weights_used[soc][intel] += im * w

    # normalize: divide by IM-weighted total
    avg = {}
    for soc, intel_scores in raw.items():
        avg[soc] = {}
        for intel in MAPPING:
            w = weights_used[soc][intel]
            avg[soc][intel] = intel_scores[intel] / w if w > 0 else 0.0

    # for each occupation: normalize across 8 intelligences so they sum to 100%
    out_rows = []
    for soc, title in occupations.items():
        if soc not in avg:
            continue
        scores = avg[soc]
        total = sum(scores.values())
        if total == 0:
            continue
        pct = {k: round(100 * v / total, 2) for k, v in scores.items()}
        row = {"soc": soc, "title": title, **pct}
        out_rows.append(row)

    # z-score each intelligence column across the population of 894 occupations
    # so the math + display can reflect "distinctive" vs "typical" profiles.
    from statistics import mean, stdev
    intel_cols = list(MAPPING.keys())
    for intel in intel_cols:
        vals = [r[intel] for r in out_rows]
        mu = mean(vals)
        sigma = stdev(vals) if len(vals) > 1 else 1.0
        sigma = sigma if sigma > 0 else 1.0
        for r in out_rows:
            r[f"{intel}_z"] = round((r[intel] - mu) / sigma, 3)

    # percentile rank per column (0-100), useful for user-facing display
    for intel in intel_cols:
        sorted_vals = sorted([r[intel] for r in out_rows])
        n = len(sorted_vals)
        for r in out_rows:
            # percentile = % of population at or below this value
            v = r[intel]
            rank = sum(1 for x in sorted_vals if x <= v)
            r[f"{intel}_pct"] = round(100 * rank / n, 1)

    out_rows.sort(key=lambda r: r["title"])

    out_path = Path(__file__).parent / "gardner_profiles.csv"
    cols = (
        ["soc", "title"]
        + intel_cols
        + [f"{c}_z" for c in intel_cols]
        + [f"{c}_pct" for c in intel_cols]
    )
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out_rows)

    print(f"wrote {len(out_rows)} occupation profiles -> {out_path}")
    return out_rows


if __name__ == "__main__":
    rows = main()
    # quick sanity check: chef, software dev, registered nurse, musician
    sanity_codes = {
        "35-1011.00": "Chef",
        "27-2011.00": "Actor (~standup)",
        "15-1252.00": "Software Dev",
        "29-1141.00": "Reg Nurse",
        "27-2042.00": "Musician",
        "27-1024.00": "Graphic Designer",
        "19-1031.00": "Conservation Sci",
    }
    intel_cols = list(MAPPING.keys())
    by_soc = {r["soc"]: r for r in rows}

    print("\n=== RAW % (sums to 100 per row) ===")
    print(f"{'occupation':<22}{'ling':>6}{'logc':>6}{'spat':>6}{'kine':>6}{'musi':>6}{'intr':>6}{'intp':>6}{'natr':>6}")
    print("-" * 70)
    for soc, short in sanity_codes.items():
        r = by_soc.get(soc)
        if r is None:
            print(f"{short:<22}  (not found: {soc})")
            continue
        print(f"{short:<22}"
              f"{r['linguistic']:>6.1f}{r['logical']:>6.1f}{r['spatial']:>6.1f}"
              f"{r['kinesthetic']:>6.1f}{r['musical']:>6.1f}{r['interpersonal']:>6.1f}"
              f"{r['intrapersonal']:>6.1f}{r['naturalistic']:>6.1f}")

    print("\n=== Z-SCORES (deviations from population mean) ===")
    print(f"{'occupation':<22}{'ling':>6}{'logc':>6}{'spat':>6}{'kine':>6}{'musi':>6}{'intr':>6}{'intp':>6}{'natr':>6}")
    print("-" * 70)
    for soc, short in sanity_codes.items():
        r = by_soc.get(soc)
        if r is None:
            continue
        print(f"{short:<22}"
              f"{r['linguistic_z']:>+6.2f}{r['logical_z']:>+6.2f}{r['spatial_z']:>+6.2f}"
              f"{r['kinesthetic_z']:>+6.2f}{r['musical_z']:>+6.2f}{r['interpersonal_z']:>+6.2f}"
              f"{r['intrapersonal_z']:>+6.2f}{r['naturalistic_z']:>+6.2f}")

    print("\n=== PERCENTILE RANKS (0-100) ===")
    print(f"{'occupation':<22}{'ling':>6}{'logc':>6}{'spat':>6}{'kine':>6}{'musi':>6}{'intr':>6}{'intp':>6}{'natr':>6}")
    print("-" * 70)
    for soc, short in sanity_codes.items():
        r = by_soc.get(soc)
        if r is None:
            continue
        print(f"{short:<22}"
              f"{r['linguistic_pct']:>6.0f}{r['logical_pct']:>6.0f}{r['spatial_pct']:>6.0f}"
              f"{r['kinesthetic_pct']:>6.0f}{r['musical_pct']:>6.0f}{r['interpersonal_pct']:>6.0f}"
              f"{r['intrapersonal_pct']:>6.0f}{r['naturalistic_pct']:>6.0f}")
