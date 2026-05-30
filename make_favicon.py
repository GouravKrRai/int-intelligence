"""Generate a favicon for INT Intelligence.

Design: bold serif "II" monogram on a warm cream square, with a small accent
dot — feels analog/literary (matches the app's Georgia serif typography),
not yet-another-startup-AI-logo. Produces both PNG sizes Streamlit needs:

  - favicon_512.png : master, used by st.set_page_config(page_icon=...)
  - favicon_64.png  : small fallback
  - favicon_32.png  : browser tab
  - favicon_16.png  : legacy

All saved into the project root. No external service. Deterministic.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle


HERE = Path(__file__).parent


def render_favicon(size_px: int, out_path: Path) -> None:
    """Render at a fixed 8x8 inch canvas regardless of final size; let
    savefig's dpi scale it down. This avoids matplotlib's font metrics
    breaking at very small canvas dimensions (the earlier 1x1 canvas
    caused the letters to render unbounded relative to the patch)."""
    canvas_in = 8.0
    target_dpi = size_px / canvas_in

    fig, ax = plt.subplots(figsize=(canvas_in, canvas_in), dpi=target_dpi)
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    # remove all axis padding so the patch fills the canvas
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.margins(0)

    # Dark rounded tile — fills the canvas with breathing room at edges
    bg = FancyBboxPatch(
        (0.04, 0.04), 0.92, 0.92,
        boxstyle="round,pad=0,rounding_size=0.16",
        linewidth=0,
        facecolor="#1d1b18",   # near-black, slightly warm
        zorder=1,
    )
    ax.add_patch(bg)

    # Thin inner keyline for subtle depth
    inner = FancyBboxPatch(
        (0.08, 0.08), 0.84, 0.84,
        boxstyle="round,pad=0,rounding_size=0.13",
        linewidth=max(0.8, canvas_in * 0.4),
        edgecolor="#3b362f",
        facecolor="none",
        zorder=2,
    )
    ax.add_patch(inner)

    # "II" monogram — bold Georgia serif, cream on near-black
    # Font size in points; at 8" canvas and 72pt/inch, ~3.5" of letterheight
    # = 252pt. Scales correctly because we're at 8" canvas regardless of dpi.
    ax.text(
        0.5, 0.49, "II",
        ha="center", va="center",
        fontsize=canvas_in * 30,           # consistent point size
        fontweight="bold",
        family="Georgia",
        color="#f3ede1",                   # warm cream
        zorder=3,
    )

    # Tiny accent dot lower-right — ties into the career-map chart's colors
    accent = Circle(
        (0.78, 0.24), 0.038,
        facecolor="#d97a4a",
        edgecolor="none",
        zorder=4,
    )
    ax.add_patch(accent)

    fig.savefig(out_path, dpi=target_dpi, bbox_inches=None, pad_inches=0,
                facecolor="none", transparent=True)
    plt.close(fig)
    print(f"  wrote {out_path.name} ({size_px}x{size_px})")


def main() -> None:
    for size in (512, 256, 128, 64, 32, 16):
        render_favicon(size, HERE / f"favicon_{size}.png")
    print("done.")


if __name__ == "__main__":
    main()
