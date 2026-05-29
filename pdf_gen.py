"""PDF report generator for INT Intelligence.

Builds a clean, branded multi-page PDF that mirrors what the user sees on
the results screen:

  page 1 — profile bars + dominant intelligence callout
  page 2 — evidence (what we saw in you) per intelligence
  page 3 — top 20 careers list
  page 4 — career-map chart (rendered fresh into the PDF, no external img)

Pure-python via reportlab — no LibreOffice, wkhtmltopdf, or browser needed.
Returns bytes; caller writes them to disk or attaches to email.
"""
from __future__ import annotations
import io
from typing import Any

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image,
    Table, TableStyle, KeepTogether,
)

INTEL_PRETTY = {
    "linguistic":    "Linguistic",
    "logical":       "Logical-Mathematical",
    "spatial":       "Spatial",
    "kinesthetic":   "Bodily-Kinesthetic",
    "musical":       "Musical",
    "interpersonal": "Interpersonal",
    "intrapersonal": "Intrapersonal",
    "naturalistic":  "Naturalistic",
}

# Colors aligned with the chart's tab20 palette
PRIMARY_INK = HexColor("#1f1d1a")
MUTED_INK = HexColor("#5e5a55")
LIGHT_INK = HexColor("#928c84")
ACCENT = HexColor("#b8312b")
RULE_GREY = HexColor("#e0ddd8")
BAR_BG = HexColor("#f3f0eb")
BAR_FILL = HexColor("#3a3530")


def _styles():
    """Build all paragraph styles in one place."""
    base = getSampleStyleSheet()
    return {
        "h1": ParagraphStyle("h1", parent=base["Heading1"],
                             fontName="Times-Bold", fontSize=24,
                             leading=28, textColor=PRIMARY_INK, spaceAfter=4),
        "h2": ParagraphStyle("h2", parent=base["Heading2"],
                             fontName="Times-Bold", fontSize=15,
                             leading=20, textColor=PRIMARY_INK,
                             spaceAfter=8, spaceBefore=14),
        "intel_label": ParagraphStyle("intel_label", parent=base["BodyText"],
                                      fontName="Times-Bold", fontSize=10,
                                      leading=13, textColor=PRIMARY_INK),
        "intel_pct": ParagraphStyle("intel_pct", parent=base["BodyText"],
                                    fontName="Times-Roman", fontSize=10,
                                    leading=13, textColor=MUTED_INK,
                                    alignment=2),  # right
        "body": ParagraphStyle("body", parent=base["BodyText"],
                               fontName="Times-Roman", fontSize=10.5,
                               leading=15, textColor=MUTED_INK,
                               spaceAfter=6, alignment=TA_LEFT),
        "evidence": ParagraphStyle("evidence", parent=base["BodyText"],
                                   fontName="Times-Italic", fontSize=9.5,
                                   leading=14, textColor=MUTED_INK,
                                   leftIndent=12, spaceAfter=10),
        "rank": ParagraphStyle("rank", parent=base["BodyText"],
                               fontName="Times-Roman", fontSize=9,
                               leading=12, textColor=LIGHT_INK),
        "career": ParagraphStyle("career", parent=base["BodyText"],
                                 fontName="Times-Bold", fontSize=11,
                                 leading=14, textColor=PRIMARY_INK),
        "match": ParagraphStyle("match", parent=base["BodyText"],
                                fontName="Times-Roman", fontSize=10,
                                leading=14, textColor=MUTED_INK,
                                alignment=2),
        "caption": ParagraphStyle("caption", parent=base["BodyText"],
                                  fontName="Times-Italic", fontSize=9,
                                  leading=12, textColor=LIGHT_INK),
        "footer": ParagraphStyle("footer", parent=base["BodyText"],
                                 fontName="Times-Roman", fontSize=8,
                                 textColor=LIGHT_INK, alignment=1),
    }


def _profile_bar_table(profile: dict[str, float], styles: dict) -> Table:
    """Renders the 8 intelligences as a horizontal bar table."""
    # sort by % descending
    rows = sorted(profile.items(), key=lambda kv: -kv[1])
    max_pct = max(profile.values()) if profile else 1.0
    # widths: label | bar+pct
    data = []
    for intel, pct in rows:
        # bar drawn as a 1-row, 2-cell sub-table with colored left cell
        # representing the % filled
        bar_w_total = 110 * mm
        fill_frac = (pct / max_pct) if max_pct > 0 else 0
        fill_w = max(0.5 * mm, bar_w_total * fill_frac)
        empty_w = bar_w_total - fill_w
        bar = Table(
            [[" ", " "]],
            colWidths=[fill_w, empty_w], rowHeights=[5 * mm]
        )
        bar.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, 0), BAR_FILL),
            ("BACKGROUND", (1, 0), (1, 0), BAR_BG),
            ("LINEBELOW", (0, 0), (-1, -1), 0, white),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))
        data.append([
            Paragraph(INTEL_PRETTY[intel], styles["intel_label"]),
            bar,
            Paragraph(f"{pct:.1f}%", styles["intel_pct"]),
        ])

    t = Table(data, colWidths=[45 * mm, 110 * mm, 18 * mm])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    return t


def _evidence_block(intel: str, score: int, evidence: str,
                    pct: float, styles: dict) -> list:
    """One intelligence's heading + evidence paragraph."""
    title = Paragraph(
        f"{INTEL_PRETTY[intel]}"
        f" &nbsp;&nbsp;<font color='#928c84'>· {score}/10 &nbsp;· "
        f"{pct:.1f}% of profile</font>",
        styles["intel_label"]
    )
    body = Paragraph(evidence or "—", styles["evidence"])
    return [title, body]


def _career_row(rank: int, title: str, pct: float, styles: dict) -> Table:
    rank_p = Paragraph(f"#{rank:02d}", styles["rank"])
    title_p = Paragraph(title, styles["career"])
    pct_p = Paragraph(f"{pct:.1f}%", styles["match"])
    t = Table([[rank_p, title_p, pct_p]],
              colWidths=[15 * mm, 130 * mm, 25 * mm])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, RULE_GREY),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


def _render_chart_image(all_matches: list[dict],
                        top_matches: list[dict]) -> bytes | None:
    """Render the same chart that appears on screen, as a PNG byte stream.
    Returns None if matplotlib is unavailable or no top matches.
    """
    if not top_matches:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gs
        from matplotlib.lines import Line2D
        from matplotlib.ticker import FuncFormatter
        from matplotlib import rcParams
    except ImportError:
        return None

    rcParams["font.family"] = "Times New Roman, Times, serif"

    xs_all = [m.get("gardner_cos", 0.0) for m in all_matches]
    ys_all = [m.get("content_cos", 0.0) for m in all_matches]
    top_pts = [(i, m["title"], m.get("match_pct", 0.0),
                m.get("gardner_cos", 0.0), m.get("content_cos", 0.0))
               for i, m in enumerate(top_matches, 1)]
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i / 20.0) for i in range(20)]

    pcts = [pct for _, _, pct, _, _ in top_pts]
    p_min, p_max = min(pcts), max(pcts)
    p_range = p_max - p_min if p_max > p_min else 1.0
    sizes = [200 + 800 * (pct - p_min) / p_range
             for _, _, pct, _, _ in top_pts]

    fig = plt.figure(figsize=(11, 8), dpi=130)
    fig.patch.set_facecolor("white")
    grid = gs.GridSpec(2, 1, height_ratios=[0.45, 1.0], figure=fig, hspace=0.02)
    ax_legend = fig.add_subplot(grid[0])
    ax = fig.add_subplot(grid[1])

    handles = []
    for (rank, title, pct, _, _), color in zip(top_pts, colors):
        label = title if len(title) <= 36 else title[:33] + "…"
        handles.append(
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=color, markeredgecolor="white",
                   markersize=9, markeredgewidth=1.0,
                   label=f"{rank:>2}. {label}  ({pct:.1f}%)")
        )
    ax_legend.legend(handles=handles, loc="center", ncol=5,
                     fontsize=7, frameon=False, handletextpad=0.4,
                     columnspacing=1.2, labelspacing=0.7,
                     title="top 20 careers   (dot size = match %)",
                     title_fontsize=8.5)
    ax_legend.axis("off")

    ax.scatter(xs_all, ys_all, s=8, c="#e6e6e2",
               alpha=0.55, edgecolors="none", zorder=1)
    for (rank, _, _, x, y), color, size in zip(top_pts, colors, sizes):
        ax.scatter([x], [y], s=size, c=[color],
                   edgecolors="white", linewidths=1.4,
                   alpha=0.95, zorder=3 + (21 - rank))

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
    ax.set_xlabel("cognitive shape match  →", fontsize=8.5,
                  color="#777", labelpad=6)
    ax.set_ylabel("content / interest match  →", fontsize=8.5,
                  color="#777", labelpad=6)
    ax.tick_params(axis="both", colors="#aaa", labelsize=7)
    ax.set_facecolor("#fafafa")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#e0e0e0")
        ax.spines[spine].set_linewidth(0.8)
    ax.grid(True, linestyle="-", linewidth=0.4, color="#eeeeee", zorder=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def build_pdf(profile: dict[str, float],
              scored: dict[str, dict],
              top_matches: list[dict],
              all_matches: list[dict] | None = None,
              user_email: str | None = None) -> bytes:
    """Render the full report and return PDF bytes.

    profile      : 8-dim percentage breakdown
    scored       : {intel: {"score": int, "evidence": str}}
    top_matches  : list of top 20 career dicts (rank order)
    all_matches  : full ranking (used to render the scatter chart).
                   If None, chart page is skipped.
    user_email   : embedded as caption on the cover/footer if provided
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm,
        title="INT Intelligence — Your Profile",
        author="INT Intelligence",
    )
    styles = _styles()
    story: list[Any] = []

    # ---- Cover / Profile page ----
    story.append(Paragraph("INT Intelligence", styles["h1"]))
    story.append(Paragraph(
        "Your eight intelligences, the way they balance in your writing.",
        styles["body"]))
    if user_email:
        story.append(Paragraph(
            f"Prepared for &nbsp;<b>{user_email}</b>",
            styles["caption"]))
    story.append(Spacer(1, 8 * mm))

    story.append(Paragraph("Your profile", styles["h2"]))
    story.append(_profile_bar_table(profile, styles))
    story.append(Spacer(1, 6 * mm))

    # ---- Evidence page ----
    story.append(PageBreak())
    story.append(Paragraph("What we saw in you", styles["h2"]))
    story.append(Paragraph(
        "Each score is anchored in specific signals from your answers.",
        styles["body"]))
    story.append(Spacer(1, 4 * mm))

    # iterate evidence in profile-percent order so the strongest shows first
    rows_by_pct = sorted(profile.items(), key=lambda kv: -kv[1])
    for intel, pct in rows_by_pct:
        s = scored.get(intel, {})
        evidence = s.get("evidence", "—")
        score = s.get("score", 0)
        block = _evidence_block(intel, score, evidence, pct, styles)
        story.extend(block)

    # ---- Top 20 careers page ----
    story.append(PageBreak())
    story.append(Paragraph("Careers that match the shape of your mind",
                           styles["h2"]))
    story.append(Paragraph(
        "People whose profession asks for the same blend of intelligences "
        "you have. Top 20 by combined match.",
        styles["body"]))
    story.append(Spacer(1, 4 * mm))
    for i, m in enumerate(top_matches[:20], 1):
        story.append(_career_row(i, m.get("title", "—"),
                                 m.get("match_pct", 0.0), styles))

    # ---- Career-map chart page ----
    if all_matches:
        chart_bytes = _render_chart_image(all_matches, top_matches[:20])
        if chart_bytes:
            story.append(PageBreak())
            story.append(Paragraph("Your career map", styles["h2"]))
            story.append(Paragraph(
                "Every career we know, plotted on two axes — how well it "
                "fits the shape of your mind (horizontal), and how close "
                "its day-to-day matches what you wrote about (vertical). "
                "Your top 20 are colored; dot size is the match %.",
                styles["body"]))
            story.append(Spacer(1, 3 * mm))
            img = Image(io.BytesIO(chart_bytes),
                        width=170 * mm, height=125 * mm)
            story.append(img)

    def _footer(canvas, _doc):
        canvas.saveState()
        canvas.setFont("Times-Italic", 8)
        canvas.setFillColor(LIGHT_INK)
        canvas.drawCentredString(
            A4[0] / 2, 10 * mm,
            "INT Intelligence  ·  generated from your seven answers")
        canvas.restoreState()

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()
