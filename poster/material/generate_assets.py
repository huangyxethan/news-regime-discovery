"""Generate all PNG assets for the poster, with consistent Swiss Modern theme.

Output: PNG files under poster/material/
Each filename matches the reference in poster/plan.md.

Run:
    python poster/material/generate_assets.py
"""

from __future__ import annotations
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ── Theme constants (Swiss Modern) ──────────────────────────────────────
BG          = "#ffffff"
BG_CARD     = "#fafaf9"
BG_ALT      = "#f4f4f2"
INK         = "#0a0a0a"
INK2        = "#3a3a3a"
INK3        = "#737373"
RULE        = "#d4d4d4"
RULE_DARK   = "#1a1a1a"
ACCENT      = "#1856ff"   # electric blue
ACCENT_LITE = "#e6edff"   # tinted accent for cells
ACCENT_RED  = "#d62828"
ACCENT_GRN  = "#16a34a"
ACCENT_AMB  = "#ca8a04"

# ── matplotlib defaults ─────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Inter", "Helvetica", "DejaVu Sans", "Arial"],
    "font.size":          11,
    "axes.edgecolor":     INK,
    "axes.labelcolor":    INK,
    "xtick.color":        INK,
    "ytick.color":        INK,
    "savefig.facecolor":  "white",
    "savefig.edgecolor":  "none",
    "savefig.bbox":       "tight",
    "savefig.dpi":        200,
    "figure.dpi":         100,
    "mathtext.fontset":   "stix",
})

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {name}")


def fmt_axes(ax):
    """Blank axes with no ticks or spines."""
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# ════════════════════════════════════════════════════════════════════════
#   01 — Problem framing diagram (Price-only vs News-Augmented)
# ════════════════════════════════════════════════════════════════════════
def gen_01_problem_framing():
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.set_xlim(0, 11); ax.set_ylim(0, 4); ax.axis("off")

    # Left box — Conventional
    ax.add_patch(patches.Rectangle((0.2, 0.4), 4.5, 3.2,
                                    facecolor=BG_CARD, edgecolor=RULE, linewidth=0.8))
    ax.text(0.4, 3.25, "CONVENTIONAL", color=INK3, fontsize=10,
            weight="bold", family="monospace")
    ax.text(0.4, 2.75, "Price-only HMM", color=INK, fontsize=17, weight="bold")
    ax.text(0.4, 2.15, "·  ret, vol, vix, term/credit spreads",
            color=INK2, fontsize=11.5)
    ax.text(0.4, 1.75, "·  reactive — reflects past info only",
            color=INK2, fontsize=11.5)
    ax.text(0.4, 1.35, "·  5-feature input · Hamilton 1989",
            color=INK2, fontsize=11.5)

    # Right box — This work (accent)
    ax.add_patch(patches.Rectangle((6.3, 0.4), 4.5, 3.2,
                                    facecolor=ACCENT_LITE, edgecolor=ACCENT, linewidth=1.5))
    ax.text(6.5, 3.25, "THIS WORK", color=ACCENT, fontsize=10,
            weight="bold", family="monospace")
    ax.text(6.5, 2.75, "News-Augmented Fusion", color=INK, fontsize=17, weight="bold")
    ax.text(6.5, 2.15, "·  + scalar sentiment (DJPR · Macro)",
            color=INK2, fontsize=11.5)
    ax.text(6.5, 1.75, "·  + topic representation (LDA / BERTopic)",
            color=INK2, fontsize=11.5)
    ax.text(6.5, 1.35, "·  30–40 feature input · this study",
            color=INK2, fontsize=11.5)

    # Arrow
    ax.annotate("", xy=(6.2, 2.0), xytext=(4.8, 2.0),
                arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=2.8,
                                mutation_scale=22))

    save(fig, "01_problem_framing.png")


# ════════════════════════════════════════════════════════════════════════
#   02 — "By the numbers" panel (dark)
# ════════════════════════════════════════════════════════════════════════
def gen_02_data_numbers():
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.set_xlim(0, 11); ax.set_ylim(0, 3.2); ax.axis("off")

    # Dark background
    ax.add_patch(patches.Rectangle((0, 0), 11, 3.2, facecolor=INK, edgecolor="none"))

    items = [
        ("6,262",  "TRADING DAYS"),
        ("24 yr",  "2000 — 2024"),
        ("2",      "NEWS CORPORA"),
        ("4",      "TEXT REPRESENTATIONS"),
        ("18",     "MAIN PIPELINES"),
        ("180",    "GRID-SEARCH CONFIGS"),
    ]

    cell_w = 11 / 3
    for i, (big, small) in enumerate(items):
        col = i % 3
        row = i // 3
        x = 0.25 + col * cell_w
        y = 2.4 if row == 0 else 1.0
        # Left accent rule
        ax.plot([x, x], [y - 0.35, y + 0.45], color=ACCENT, lw=3.5)
        ax.text(x + 0.18, y + 0.05, big, color="white",
                fontsize=30, weight="bold", va="center")
        ax.text(x + 0.18, y - 0.5, small, color="#bcbcbc",
                fontsize=9, family="monospace", weight="bold", va="center")

    save(fig, "02_data_numbers.png")


# ════════════════════════════════════════════════════════════════════════
#   03 — Pipeline flow (6 stages)
# ════════════════════════════════════════════════════════════════════════
def gen_03_pipeline_flow():
    # 6 boxes, centered. box_w + gap chosen so total span = fig_w - 2*pad
    fig_w = 13.2; fig_h = 3.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w); ax.set_ylim(0, fig_h); ax.axis("off")

    stages = [
        ("INPUT",     "News + Market"),
        ("SMOOTH",    "21 — 35d rolling"),
        ("REPRESENT", "Scalar · LDA\nBERTopic · FinBERT"),
        ("REDUCE",    "PCA-95% · UMAP-10"),
        ("CLUSTER",   "GMM · HMM · KMeans"),
        ("REGIMES",   "K = 3 — 5 labels"),
    ]

    n     = len(stages)
    box_w = 1.92
    box_h = 1.9
    gap   = 0.22
    total = n * box_w + (n - 1) * gap
    x0    = (fig_w - total) / 2                  # ←  centred
    y_box = (fig_h - box_h) / 2

    for i, (title, desc) in enumerate(stages):
        x = x0 + i * (box_w + gap)
        ax.add_patch(patches.Rectangle((x, y_box), box_w, box_h,
                                        facecolor=BG_CARD, edgecolor=RULE, linewidth=0.8))
        ax.text(x + box_w/2, y_box + box_h - 0.32, title,
                ha="center", color=INK, fontsize=13, weight="bold",
                family="monospace")
        ax.text(x + box_w/2, y_box + box_h/2 - 0.18, desc,
                ha="center", va="center", color=INK2, fontsize=11,
                linespacing=1.3)
        if i < n - 1:
            arr_x_start = x + box_w + 0.02
            arr_x_end   = x + box_w + gap - 0.02
            ax.annotate("", xy=(arr_x_end, y_box + box_h/2),
                        xytext=(arr_x_start, y_box + box_h/2),
                        arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=2.0,
                                        mutation_scale=20))

    save(fig, "03_pipeline_flow.png")


# ════════════════════════════════════════════════════════════════════════
#   04 — Method space matrix (3 × 4)
# ════════════════════════════════════════════════════════════════════════
def gen_04_method_matrix():
    # Geometry: header row + 3 body rows fully fill the canvas — no dead space.
    label_w = 2.4
    cell_w  = 2.45
    cell_h  = 1.10
    n_rows  = 3      # body rows
    fig_w   = label_w + 4 * cell_w        # = 2.4 + 9.8 = 12.2
    fig_h   = cell_h * (n_rows + 1)       # = 1.10 * 4 = 4.4
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w); ax.set_ylim(0, fig_h); ax.axis("off")

    col_labels = ["SCALAR", "LDA", "BERTOPIC", "FINBERT"]
    row_labels = ["PRICE-ONLY", "NEWS-ONLY", "HYBRID"]

    cell_values = [
        [("—", "off"),  ("—", "off"),  ("—", "off"),  ("—", "off")],
        [("2 corpora", "on"), ("2 corpora", "on"), ("2 corpora", "on"), ("retrieval", "on")],
        [("5 fusions", "on"), ("3 fusions", "on"), ("News + Full ★", "star"), ("PCA · UMAP", "on")],
    ]

    top_y = fig_h - cell_h     # header row starts here

    # Header row (column labels)
    for j, col in enumerate(col_labels):
        x = label_w + j * cell_w
        ax.add_patch(patches.Rectangle((x, top_y), cell_w, cell_h,
                                        facecolor=BG_ALT, edgecolor=INK, linewidth=1))
        ax.text(x + cell_w/2, top_y + cell_h/2, col,
                ha="center", va="center", fontsize=12,
                weight="bold", family="monospace")
    # Corner cell
    ax.add_patch(patches.Rectangle((0, top_y), label_w, cell_h,
                                    facecolor=BG_ALT, edgecolor=INK, linewidth=1))

    # Body rows
    for i, row in enumerate(row_labels):
        y = top_y - (i + 1) * cell_h
        # Row label
        ax.add_patch(patches.Rectangle((0, y), label_w, cell_h,
                                        facecolor=BG_ALT, edgecolor=INK, linewidth=1))
        ax.text(0.18, y + cell_h/2, row, ha="left", va="center",
                fontsize=12, weight="bold", family="monospace")
        for j, (val, kind) in enumerate(cell_values[i]):
            x = label_w + j * cell_w
            if kind == "off":
                ax.add_patch(patches.Rectangle((x, y), cell_w, cell_h,
                                                facecolor=RULE, edgecolor=INK, linewidth=0.5))
                # diagonal stripes
                for k in range(-3, 10):
                    ax.plot([x + k*0.35, x + k*0.35 + cell_h],
                            [y, y + cell_h], color="#bbb", lw=0.5)
                ax.text(x + cell_w/2, y + cell_h/2, "—", ha="center", va="center",
                        color=INK3, fontsize=14)
            elif kind == "star":
                ax.add_patch(patches.Rectangle((x, y), cell_w, cell_h,
                                                facecolor=ACCENT, edgecolor=INK, linewidth=0.5))
                ax.text(x + cell_w/2, y + cell_h/2, val,
                        ha="center", va="center", color="white",
                        fontsize=12, weight="bold", family="monospace")
            else:
                ax.add_patch(patches.Rectangle((x, y), cell_w, cell_h,
                                                facecolor=ACCENT_LITE, edgecolor=RULE, linewidth=0.5))
                ax.text(x + cell_w/2, y + cell_h/2, val,
                        ha="center", va="center", color=ACCENT,
                        fontsize=12, weight="bold", family="monospace")

    save(fig, "04_method_matrix.png")


# ════════════════════════════════════════════════════════════════════════
#   05 — Model formulas (math)
# ════════════════════════════════════════════════════════════════════════
def gen_05_models_formulas():
    # Compact layout: title + equation card only.  Prose context for these
    # formulas lives in the HTML caption directly below the image, so the
    # inline italic notes are removed to save vertical space and avoid
    # overlap with the next cell's title.
    formulas = [
        ("STICKY-INIT HMM TRANSITION MATRIX",
         r"$T_{ii} = \mathbf{0.85}, \quad T_{ij} = \dfrac{0.15}{K-1}$ for $i \neq j$"),
        ("ADAPTIVE HMM COVARIANCE   (innovation)",
         r"$\Sigma \;=\; \mathrm{diag}$  if  $\dim \geq 10$,   else   $\mathrm{full}$"),
        ("SILHOUETTE   (cluster quality)",
         r"$s(i) \;=\; \dfrac{\,b_i - a_i\,}{\max(a_i,\, b_i)}$"),
        ("BAYESIAN INFORMATION CRITERION   (model selection)",
         r"$\mathrm{BIC} \;=\; -2 \log L \;+\; k \log N$"),
    ]

    # Geometry: each cell stacks title-gap-card-gap.  Cell height = step.
    n        = len(formulas)
    title_h  = 0.30
    card_h   = 0.95
    gap_tc   = 0.10        # title → card
    gap_cc   = 0.20        # card → next title
    step     = title_h + gap_tc + card_h + gap_cc   # = 1.55
    pad_v    = 0.10
    fig_w    = 11
    fig_h    = n * step + 2 * pad_v
    fig, ax  = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w); ax.set_ylim(0, fig_h); ax.axis("off")

    card_x = 0.25
    card_w = fig_w - 0.5    # 10.5

    for i, (title, eq) in enumerate(formulas):
        cell_top = fig_h - pad_v - i * step
        # title flush to top of cell
        ax.text(card_x, cell_top - 0.05, title,
                color=INK3, fontsize=12, weight="bold",
                family="monospace", va="top")
        # card below title
        card_y = cell_top - title_h - gap_tc - card_h
        ax.add_patch(patches.Rectangle((card_x, card_y), card_w, card_h,
                                        facecolor=BG_CARD, edgecolor="none"))
        ax.plot([card_x, card_x], [card_y, card_y + card_h],
                color=INK, lw=2.5)
        ax.text(card_x + 0.28, card_y + card_h/2, eq,
                color=INK, fontsize=18, va="center")

    save(fig, "05_models_formulas.png")


# ════════════════════════════════════════════════════════════════════════
#   06 — Innovation cards (3 before/after)
# ════════════════════════════════════════════════════════════════════════
def gen_06_innovations():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")

    items = [
        ("1", "Per-feature smoothing windows",
         "Scalar 35d  ·  LDA 30d  ·  BERTopic 21d  ·  FinBERT 10d.\n"
         "Topic distributions are already daily aggregates of\n"
         "many headlines, so they need less smoothing.",
         "21d uniform", "per-rep tuned"),
        ("2", "Diag-Σ threshold for high-dim HMM",
         "Auto-switch to diagonal covariance when feature\n"
         "dim ≥ 10. Eliminates absorbing-state failure mode\n"
         "on news-fusion pipelines.",
         "894 days", "200 days"),
        ("3", "Two-stage grid search",
         "text × dim-red × cluster × K, run twice\n"
         "(news-only / news + price). Pareto-optimal under\n"
         "silhouette and duration constraints.",
         "manual tuning", "180 configs"),
    ]

    card_h = 1.45
    for i, (idx, title, desc, before, after) in enumerate(items):
        y = 4.6 - i * (card_h + 0.05)
        # Index block (black)
        ax.add_patch(patches.Rectangle((0.2, y - card_h), 0.85, card_h,
                                        facecolor=INK, edgecolor="none"))
        ax.text(0.625, y - card_h/2, idx, ha="center", va="center",
                color="white", fontsize=34, weight="bold")
        # Body card
        ax.add_patch(patches.Rectangle((1.05, y - card_h), 8.6, card_h,
                                        facecolor=BG_CARD, edgecolor=RULE, linewidth=0.6))
        ax.text(1.25, y - 0.3, title, color=INK, fontsize=11.5, weight="bold")
        ax.text(1.25, y - 0.7, desc, color=INK2, fontsize=9.5,
                linespacing=1.4, va="top")
        # Delta column (right)
        ax.add_patch(patches.Rectangle((9.65, y - card_h), 2.15, card_h,
                                        facecolor=ACCENT_LITE, edgecolor=RULE, linewidth=0.5))
        ax.text(10.725, y - 0.4, before, ha="center", va="center",
                color=INK3, fontsize=10, family="monospace")
        ax.plot([10.725, 10.725], [y - 0.85, y - 0.55], color=INK3, lw=0.5)  # invisible spacer
        ax.text(10.725, y - 0.75, "↓", ha="center", va="center",
                color=ACCENT, fontsize=14, weight="bold")
        ax.text(10.725, y - 1.1, after, ha="center", va="center",
                color=ACCENT, fontsize=11, weight="bold", family="monospace")

    save(fig, "06_innovations.png")


# ════════════════════════════════════════════════════════════════════════
#   07 — Evaluation framework (4 dimensions in 2×2 grid)
# ════════════════════════════════════════════════════════════════════════
def gen_07_eval_framework():
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.5); ax.axis("off")

    dims = [
        ("STATISTICAL FIT",         "BIC  ·  AIC  ·  log-likelihood"),
        ("CLUSTERING QUALITY",      "silhouette ↑  ·  Davies-Bouldin ↓"),
        ("REGIME STABILITY",        "rolling ARI (1250-d)  ·  mean duration"),
        ("FINANCIAL INTERPRETABILITY", "per-regime Sharpe  ·  max DD  ·  avg VIX"),
    ]

    cell_w = 4.7
    cell_h = 1.5
    for i, (title, metrics) in enumerate(dims):
        col = i % 2
        row = i // 2
        x = 0.2 + col * (cell_w + 0.2)
        y = 1.75 - row * (cell_h + 0.15)
        # Card
        ax.add_patch(patches.Rectangle((x, y), cell_w, cell_h,
                                        facecolor=BG_CARD, edgecolor=RULE, linewidth=0.5))
        # Top accent strip
        ax.add_patch(patches.Rectangle((x, y + cell_h - 0.1), cell_w, 0.1,
                                        facecolor=ACCENT, edgecolor="none"))
        ax.text(x + 0.25, y + cell_h - 0.45, title, color=INK,
                fontsize=11, weight="bold", family="sans-serif")
        ax.text(x + 0.25, y + 0.45, metrics, color=INK2,
                fontsize=10, family="monospace")

    save(fig, "07_eval_framework.png")


# ════════════════════════════════════════════════════════════════════════
#   08 — Headline figure (copy existing ft_best_comparison.png)
# ════════════════════════════════════════════════════════════════════════
def gen_08_headline():
    src = os.path.normpath(os.path.join(OUT_DIR, "..", "..", "_images",
                                         "ft_best_comparison.png"))
    dst = os.path.join(OUT_DIR, "08_regime_timelines.png")
    if not os.path.exists(src):
        print(f"  WARN: {src} not found — skipped 08")
        return
    shutil.copy(src, dst)
    print(f"  Copied: 08_regime_timelines.png  (from {os.path.basename(src)})")


# ════════════════════════════════════════════════════════════════════════
#   09 — Master evaluation table
# ════════════════════════════════════════════════════════════════════════
def gen_09_master_table():
    rows = [
        # (pipeline, sil, dbi, dur, ari, sharpe, kind)
        ("Price-only HMM (baseline)",      "0.202", "1.48", "95",  "0.447", "2.02", "baseline"),
        ("DJPR Sentiment HMM",             "0.284", "1.26", "44",  "0.383", "0.80", "normal"),
        ("Macro Sentiment HMM",            "0.270", "1.27", "174", "0.239", "0.65", "highlight"),
        ("DJPR Topic-Vector HMM",          "0.261", "1.49", "70",  "0.278", "0.64", "normal"),
        ("Dual-Corpus Sentiment Fusion",   "0.245", "1.42", "49",  "0.268", "1.00", "normal"),
        ("Tuned News Fusion",              "0.279", "0.95", "202", "0.31",  "0.78", "highlight"),
        ("Tuned Full Stack (+ market)",    "0.272", "0.92", "224", "0.33",  "1.21", "highlight"),
    ]
    headers = ["PIPELINE", "SIL ↑", "DBI ↓", "DUR (D)", "ARI ↑", "SHARPE"]
    col_widths = [5.3, 1.1, 1.1, 1.1, 1.1, 1.1]
    col_x = [0]
    for w in col_widths:
        col_x.append(col_x[-1] + w)
    total_w = sum(col_widths)
    n_rows = len(rows)
    row_h = 0.5
    head_h = 0.55
    total_h = head_h + n_rows * row_h + 0.45

    fig, ax = plt.subplots(figsize=(total_w + 0.3, total_h + 0.3))
    ax.set_xlim(0, total_w); ax.set_ylim(0, total_h); ax.axis("off")

    # Header row
    y_head = total_h - head_h
    for j, h in enumerate(headers):
        ax.add_patch(patches.Rectangle((col_x[j], y_head), col_widths[j], head_h,
                                        facecolor=INK, edgecolor=INK))
        align = "left" if j == 0 else "right"
        text_x = col_x[j] + 0.15 if j == 0 else col_x[j] + col_widths[j] - 0.15
        ax.text(text_x, y_head + head_h/2, h, ha=align, va="center",
                color="white", fontsize=9.5, weight="bold", family="monospace")

    # Body rows
    for i, (label, sil, dbi, dur, ari, shp, kind) in enumerate(rows):
        y = y_head - (i + 1) * row_h
        face = "white"
        text_main = INK
        label_color = INK
        star_text = ""
        if kind == "highlight":
            face = ACCENT_LITE
            text_main = ACCENT
            label_color = ACCENT
            star_text = " ★"
        elif kind == "baseline":
            face = BG_ALT
            text_main = INK3
            label_color = INK3
        for j in range(len(headers)):
            ax.add_patch(patches.Rectangle((col_x[j], y), col_widths[j], row_h,
                                            facecolor=face, edgecolor=RULE, linewidth=0.4))
        # Label
        ax.text(col_x[0] + 0.15, y + row_h/2, label + star_text,
                ha="left", va="center", color=label_color,
                fontsize=10, weight="bold" if kind == "highlight" else "normal")
        # Numbers
        values = [sil, dbi, dur, ari, shp]
        for j, val in enumerate(values):
            color = ACCENT_GRN if (j == 0 and kind != "baseline") else (INK3 if kind == "baseline" else INK2)
            weight = "bold" if (j == 0 and kind != "baseline") else "normal"
            ax.text(col_x[j+1] + col_widths[j+1] - 0.15, y + row_h/2, val,
                    ha="right", va="center", color=color,
                    fontsize=10, weight=weight, family="monospace")

    # Footer caption
    ax.text(0, 0.1,
            "★ = grid-search winner   ·   ↑ = higher better, ↓ = lower better   ·   "
            "ARI / Sharpe for tuned configs are post-07 re-run estimates",
            color=INK3, fontsize=8.5, style="italic", va="bottom")

    save(fig, "09_master_table.png")


# ════════════════════════════════════════════════════════════════════════
#   10 — Key findings (4 big numerals + text)
# ════════════════════════════════════════════════════════════════════════
def gen_10_findings():
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5.5); ax.axis("off")

    items = [
        ("+35%", 26,
         "News augmentation lifts silhouette over price baseline.",
         "Macro Sentiment HMM reaches 0.27 vs 0.20 — robust across\n"
         "all three scalar / LDA / BERTopic representations."),
        ("Macro\n> Firm", 18,
         "Aggregate news outperforms firm-level press releases.",
         "Macro corpus (Fed, inflation, policy) is better aligned with\n"
         "unsupervised regime detection than firm-specific PR noise."),
        ("202d", 26,
         "Grid-search sweet spot.",
         "BERTopic + PCA-95% + KMeans (K=5) achieves silhouette 0.279\n"
         "with 202-day mean duration — within the canonical regime band."),
        ("+market", 20,
         "Price stabilizes news-driven regimes.",
         "Adding market features bumps duration 202 → 224 d at negligible\n"
         "silhouette cost; price acts as a low-frequency anchor."),
    ]

    for i, (big, fsize, headline, detail) in enumerate(items):
        y = 5.0 - i * 1.3
        if i > 0:
            ax.plot([0.1, 10.9], [y + 0.55, y + 0.55], color=RULE, lw=0.5)
        # Big numeral (left)
        ax.text(1.3, y - 0.05, big, color=ACCENT, fontsize=fsize,
                weight="bold", ha="center", va="center",
                linespacing=0.9)
        # Headline + detail (right)
        ax.text(2.6, y + 0.25, headline, color=INK,
                fontsize=11.5, weight="bold", va="center")
        ax.text(2.6, y - 0.3, detail, color=INK2,
                fontsize=10, linespacing=1.4, va="center")

    save(fig, "10_findings.png")


# ════════════════════════════════════════════════════════════════════════
#   11 — Discussion (3 cards: what worked / gotchas / limitations)
# ════════════════════════════════════════════════════════════════════════
def gen_11_discussion():
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5); ax.axis("off")

    cards = [
        (ACCENT,     "WHAT WORKED",
         "News provides a signal complementary to price.\n"
         "Aggregate macro news at the right granularity is more aligned\n"
         "with regime-state detection than firm-specific PR."),
        (ACCENT_AMB, "METHODOLOGICAL GOTCHAS",
         "Smoothing windows tuned per feature type — topic distributions\n"
         "are already daily aggregates; double-smoothing produces year-long\n"
         "“era” regimes.  High silhouette ≠ good regimes:\n"
         "FinBERT + UMAP hits sil = 0.66 but each cluster spans years."),
        (INK3,       "LIMITATIONS",
         "In-sample evaluation only; OOS check uses simplified GMM refit\n"
         "rather than walk-forward HMM. No bootstrap CI on silhouette.\n"
         "Macro corpus post-2000 only."),
    ]

    y = 4.6
    for color, title, text in cards:
        # Card box
        card_h = 1.4
        ax.add_patch(patches.Rectangle((0.2, y - card_h), 10.6, card_h,
                                        facecolor=BG_CARD, edgecolor=RULE, linewidth=0.5))
        # Left accent rule
        ax.plot([0.2, 0.2], [y - card_h, y], color=color, lw=4)
        # Title
        ax.text(0.5, y - 0.3, title, color=INK,
                fontsize=11, weight="bold", family="sans-serif")
        # Body
        ax.text(0.5, y - card_h + 0.15, text, color=INK2,
                fontsize=10, linespacing=1.45, va="bottom")
        y -= card_h + 0.05

    save(fig, "11_discussion.png")


# ════════════════════════════════════════════════════════════════════════
#   MAIN
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Generating poster assets in: {OUT_DIR}\n")
    gen_01_problem_framing()
    gen_02_data_numbers()
    gen_03_pipeline_flow()
    gen_04_method_matrix()
    gen_05_models_formulas()
    gen_06_innovations()
    gen_07_eval_framework()
    gen_08_headline()
    gen_09_master_table()
    gen_10_findings()
    gen_11_discussion()
    print("\nAll assets generated.")
