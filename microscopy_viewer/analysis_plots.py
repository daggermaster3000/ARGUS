"""The report folder an analysis run leaves behind: the workbook and its figures.

One folder per run, named after the experiment and the time it was run, so a
second run never overwrites the first and the figures always sit beside the
numbers they were drawn from::

    <experiment>_analysis_20260916_143012/
        <experiment>_analysis_20260916_143012.xlsx
        violin_cell_count.png
        violin_region_area.png
        pca_cells.png
        cell_outlines.npz          every segmented cell's outline, for the explorer app

Drawn with matplotlib's object API on the Agg canvas, never through ``pyplot``:
the report is written on a worker thread, and pyplot's global state and GUI
backend do not belong there.

In the violins colour follows the genotype, never its position in a particular
plot. In the cell PCA colour follows the sample and marker shape the genotype,
in the same shape order the violins use. The four hues were checked for
colour-vision-deficiency separation across every pair (scatter plots compare all
of them, not just neighbours); every genotype also gets its own marker shape and
a legend entry, so identity never rests on colour alone.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .utils import get_logger

logger = get_logger("analysis_plots")

#: Cell outlines, beside the workbook in every report folder.
CELL_OUTLINES_FILE = "cell_outlines.npz"

#: Genotype colours, in assignment order. Blue, orange, aqua, violet: validated
#: as a set for all-pairs separation on a white ground.
GENOTYPE_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7")
#: Beyond four genotypes: the remaining reference hues, with the marker shape
#: carrying identity where the colours get close.
EXTRA_COLORS = ("#e87ba4", "#008300", "#e34948", "#eda100")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")

#: Genotypes that are the reference group, drawn first and in the first colour.
REFERENCE_GENOTYPES = ("wt", "ctl", "sib")
UNKNOWN_GENOTYPE = "(unknown)"

INK = "#1f1f1e"
INK_MUTED = "#6b6a65"
GRID = "#e4e3de"
SURFACE = "#ffffff"

#: Cells drawn in the PCA scatter. The PCA is fitted on every cell; only the
#: drawing is thinned, because a few hundred thousand dots is a black rectangle.
MAX_SCATTER_POINTS = 20000

#: Per-cell measurements the cell PCA is fitted on.
CELL_FEATURES = (
    ("volume_um3", "size"),
    ("equivalent_diameter_um", "diameter"),
    ("mean", "mean intensity"),
    ("median", "median intensity"),
    ("std", "intensity SD"),
    ("maximum", "max intensity"),
    ("integrated", "integrated intensity"),
)


# ---------------------------------------------------------------------------
# Genotype identity
# ---------------------------------------------------------------------------


def genotype_label(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return text if text and text.lower() != "nan" else UNKNOWN_GENOTYPE


def genotype_order(values: Sequence[Any]) -> list[str]:
    """Reference genotypes first, then the rest alphabetically, unknown last."""
    found = {genotype_label(value) for value in values}
    reference = [g for g in REFERENCE_GENOTYPES if g in found]
    rest = sorted(g for g in found if g not in reference and g != UNKNOWN_GENOTYPE)
    tail = [UNKNOWN_GENOTYPE] if UNKNOWN_GENOTYPE in found else []
    return reference + rest + tail


def genotype_styles(order: Sequence[str]) -> dict[str, tuple[str, str]]:
    """``genotype -> (colour, marker)``, fixed by position in *order*."""
    colors = GENOTYPE_COLORS + EXTRA_COLORS
    styles = {}
    for index, genotype in enumerate(order):
        color = "#8a8983" if genotype == UNKNOWN_GENOTYPE else colors[index % len(colors)]
        styles[genotype] = (color, MARKERS[index % len(MARKERS)])
    return styles


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------


def pca(matrix: np.ndarray, components: int = 2):
    """Principal components of a standardised matrix, by SVD.

    Returns ``(scores, explained_ratio, kept_columns)``. Columns that do not vary
    carry no information and would divide by zero when standardised, so they are
    dropped; rows with any missing value are expected to have been removed.
    """
    data = np.asarray(matrix, dtype=float)
    if data.ndim != 2 or data.shape[0] < 2:
        return np.zeros((0, components)), np.zeros(components), []
    spread = data.std(axis=0)
    kept = [i for i in range(data.shape[1]) if np.isfinite(spread[i]) and spread[i] > 0]
    if not kept:
        return np.zeros((0, components)), np.zeros(components), []
    scaled = (data[:, kept] - data[:, kept].mean(axis=0)) / spread[kept]
    _u, singular, vt = np.linalg.svd(scaled, full_matrices=False)
    scores = scaled @ vt.T
    variance = singular**2
    ratio = variance / variance.sum() if variance.sum() > 0 else variance
    n = min(components, scores.shape[1])
    out = np.zeros((scores.shape[0], components))
    out[:, :n] = scores[:, :n]
    explained = np.zeros(components)
    explained[:n] = ratio[:n]
    return out, explained, kept


def cell_matrix(outcomes) -> tuple[np.ndarray, list[str], list[str], list[str]]:
    """Every cell of every sample as a feature row, with its sample and genotype."""
    rows: list[list[float]] = []
    genotypes: list[str] = []
    samples: list[str] = []
    for outcome in outcomes:
        if getattr(outcome, "error", ""):
            continue
        genotype = genotype_label(outcome.genotype)
        for stat in outcome.stats:
            rows.append([float(getattr(stat, name)) for name, _label in CELL_FEATURES])
            genotypes.append(genotype)
            samples.append(str(outcome.name))
    matrix = np.asarray(rows, dtype=float).reshape(-1, len(CELL_FEATURES))
    finite = np.isfinite(matrix).all(axis=1)
    keep = lambda values: [v for v, ok in zip(values, finite) if ok]  # noqa: E731
    return (
        matrix[finite],
        keep(samples),
        keep(genotypes),
        [label for _name, label in CELL_FEATURES],
    )


def _mix(color: str, other: str, amount: float) -> str:
    """*color* moved *amount* (0-1) of the way towards *other*."""
    from matplotlib.colors import to_hex, to_rgb

    a, b = np.asarray(to_rgb(color)), np.asarray(to_rgb(other))
    return to_hex(a + (b - a) * float(amount))


def sample_colors(samples: Sequence[str], genotype_of: dict[str, str]) -> dict[str, str]:
    """One colour per sample, grouped so a genotype's samples read as a family.

    Up to eight samples each take a reference hue, in order, genotype by genotype.
    Past that, eight hues cannot stay distinguishable and inventing more makes it
    worse, so each genotype keeps its own hue and its samples are shades of it,
    dark to light — identity then rests on the shade, the marker shape and the
    sample name drawn at each cloud's centre together.
    """
    genotypes = genotype_order(list(genotype_of.values()))
    ordered = [
        sample
        for genotype in genotypes
        for sample in samples
        if genotype_of.get(sample) == genotype
    ]
    palette = GENOTYPE_COLORS + EXTRA_COLORS
    if len(ordered) <= len(palette):
        return {sample: palette[i] for i, sample in enumerate(ordered)}

    family = genotype_styles(genotypes)
    colors: dict[str, str] = {}
    for genotype in genotypes:
        members = [sample for sample in ordered if genotype_of.get(sample) == genotype]
        base = family[genotype][0]
        for i, sample in enumerate(members):
            # From 45 % towards black to 45 % towards white across the group.
            t = 0.0 if len(members) == 1 else i / (len(members) - 1)
            shift = -0.45 + 0.9 * t
            colors[sample] = _mix(base, "#000000", -shift) if shift < 0 else _mix(base, "#ffffff", shift)
    return colors


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def _figure(width: float, height: float):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(width, height), dpi=150, facecolor=SURFACE)
    FigureCanvasAgg(figure)
    return figure


def _style_axes(axes) -> None:
    axes.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(GRID)
    axes.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    axes.grid(True, axis="y", color=GRID, linewidth=0.6)
    axes.set_axisbelow(True)


def _legend(axes, styles, order) -> None:
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], linestyle="", marker=styles[g][1], markersize=6,
               markerfacecolor=styles[g][0], markeredgecolor=SURFACE, label=g)
        for g in order
    ]
    legend = axes.legend(
        handles=handles, title="Genotype", frameon=False, fontsize=8, title_fontsize=8,
        loc="upper left", bbox_to_anchor=(1.0, 1.0),
    )
    for text in legend.get_texts():
        text.set_color(INK)
    legend.get_title().set_color(INK_MUTED)


def violin_plot(features, column: str, title: str, ylabel: str, path: str | Path) -> bool:
    """Violins with every sample as a dot, grouped by region, split by genotype.

    A violin needs at least two distinct values to have a shape; a group with
    fewer shows its dots and a median tick only, rather than a misleading blob.
    Returns ``False`` (and draws nothing) when the column has no values at all.
    """
    if features is None or features.empty or column not in features.columns:
        return False
    frame = features[["Region", "Genotype", column]].copy()
    frame[column] = frame[column].astype(float)
    frame = frame[np.isfinite(frame[column])]
    if frame.empty:
        return False
    frame["Genotype"] = [genotype_label(value) for value in frame["Genotype"]]

    regions = list(dict.fromkeys(frame["Region"]))
    order = genotype_order(frame["Genotype"])
    styles = genotype_styles(order)
    rng = np.random.default_rng(0)

    figure = _figure(max(4.5, 1.1 * len(regions) * max(1, len(order)) ** 0.5 + 2.0), 4.2)
    axes = figure.add_subplot(1, 1, 1)
    _style_axes(axes)

    slot = 0.8 / max(len(order), 1)
    for r, region in enumerate(regions):
        for g, genotype in enumerate(order):
            values = frame.loc[
                (frame["Region"] == region) & (frame["Genotype"] == genotype), column
            ].to_numpy(dtype=float)
            if values.size == 0:
                continue
            color, marker = styles[genotype]
            center = r - 0.4 + slot * (g + 0.5)
            if np.unique(values).size >= 2:
                body = axes.violinplot(
                    [values], positions=[center], widths=slot * 0.9,
                    showextrema=False, showmedians=False,
                )
                for patch in body["bodies"]:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.22)
                    patch.set_edgecolor(color)
                    patch.set_linewidth(1.0)
            median = float(np.median(values))
            axes.hlines(median, center - slot * 0.3, center + slot * 0.3,
                        color=INK, linewidth=1.5, zorder=4)
            jitter = rng.uniform(-slot * 0.18, slot * 0.18, size=values.size)
            axes.scatter(
                center + jitter, values, s=26, marker=marker, color=color,
                edgecolors=SURFACE, linewidths=0.8, zorder=5,
            )

    axes.set_xticks(range(len(regions)))
    axes.set_xticklabels(regions, rotation=30 if len(regions) > 4 else 0,
                         ha="right" if len(regions) > 4 else "center", color=INK)
    axes.set_xlim(-0.6, len(regions) - 0.4)
    axes.set_ylabel(ylabel, color=INK_MUTED, fontsize=9)
    axes.set_title(title, color=INK, fontsize=11, loc="left")
    _legend(axes, styles, order)
    figure.tight_layout()
    figure.savefig(str(path), facecolor=SURFACE)
    return True


def cell_pca_plot(outcomes, path: str | Path) -> str:
    """PCA scatter of every segmented cell, coloured by sample.

    Marker shape carries the genotype and each sample's name sits at the median
    of its cells, so a cloud can be told from its neighbours without matching
    shades against the legend. Returns an empty string when drawn, or why not.
    """
    matrix, samples, genotypes, labels = cell_matrix(outcomes)
    if matrix.shape[0] < 3:
        return "fewer than three cells to project"
    scores, explained, kept = pca(matrix)
    if len(kept) < 2:
        return "the cell measurements do not vary enough for a PCA"

    genotype_of = dict(zip(samples, genotypes))
    sample_names = list(dict.fromkeys(samples))
    colors = sample_colors(sample_names, genotype_of)
    sample_names = list(colors)  # grouped by genotype
    genotypes_in = genotype_order(genotypes)
    markers = {g: style[1] for g, style in genotype_styles(genotypes_in).items()}
    sample_array = np.asarray(samples)

    shown = np.arange(scores.shape[0])
    if shown.size > MAX_SCATTER_POINTS:
        shown = np.sort(np.random.default_rng(0).choice(shown, MAX_SCATTER_POINTS, replace=False))
    size = 14 if shown.size < 2000 else 5

    many = len(sample_names) > 12
    figure = _figure(7.4 if many else 6.6, 5.0)
    axes = figure.add_subplot(1, 1, 1)
    _style_axes(axes)
    axes.grid(True, axis="x", color=GRID, linewidth=0.6)

    from matplotlib.lines import Line2D

    handles = []
    # Largest sample first, so the smaller ones are not buried underneath it.
    for sample in sorted(sample_names, key=lambda n: -int((sample_array[shown] == n).sum())):
        picked = shown[sample_array[shown] == sample]
        if picked.size == 0:
            continue
        axes.scatter(
            scores[picked, 0], scores[picked, 1], s=size, marker=markers[genotype_of[sample]],
            color=colors[sample], alpha=0.55 if shown.size >= 2000 else 0.85,
            edgecolors="none", rasterized=True,
        )
    centres = []
    for number, sample in enumerate(sample_names, start=1):
        own = sample_array == sample
        if not own.any():
            continue
        cx, cy = float(np.median(scores[own, 0])), float(np.median(scores[own, 1]))
        axes.scatter([cx], [cy], s=70, marker=markers[genotype_of[sample]], color=colors[sample],
                     edgecolors=SURFACE, linewidths=1.5, zorder=6)
        centres.append((number, cx, cy))
        handles.append(Line2D(
            [], [], linestyle="", marker=markers[genotype_of[sample]], markersize=5,
            markerfacecolor=colors[sample], markeredgecolor=SURFACE,
            label=f"{number:>2} · {sample} ({genotype_of[sample]}, n={int(own.sum())})",
        ))

    legend = axes.legend(
        handles=handles, title="Sample", frameon=False, fontsize=6.5, title_fontsize=8,
        loc="upper left", bbox_to_anchor=(1.0, 1.0), ncol=2 if len(handles) > 24 else 1,
    )
    for text in legend.get_texts():
        text.set_color(INK)
    legend.get_title().set_color(INK_MUTED)

    axes.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)", color=INK_MUTED, fontsize=9)
    axes.set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)", color=INK_MUTED, fontsize=9)
    title = f"PCA of {scores.shape[0]:,} cells from {len(sample_names)} samples"
    if shown.size < scores.shape[0]:
        title += f" ({shown.size:,} drawn)"
    axes.set_title(title, color=INK, fontsize=11, loc="left")
    shapes = ", ".join(f"{g} {_MARKER_NAMES.get(markers[g], markers[g])}" for g in genotypes_in)
    used = ", ".join(labels[i] for i in kept)
    axes.text(0.0, -0.16,
              f"Colour: sample · shape: genotype ({shapes}) · large marker + number: sample median\n"
              f"Features (standardised): {used}",
              transform=axes.transAxes, fontsize=7, color=INK_MUTED, va="top")
    figure.tight_layout()
    # After the layout: the overlap test is in screen space.
    _place_labels(figure, axes, centres)
    figure.savefig(str(path), facecolor=SURFACE)
    return ""


def _place_labels(figure, axes, centres) -> None:
    """Number each sample median, nudging labels until none overlap.

    Medians of similar samples sit on top of each other, and names drawn there
    are unreadable. Each label tries rings of positions around its point and
    takes the first whose box is clear of every label already placed, with a
    thin leader line back when it had to move.
    """
    renderer = figure.canvas.get_renderer()
    placed = []
    rings = [(6, 4)] + [
        (radius * np.cos(angle), radius * np.sin(angle))
        for radius in (12, 20, 28, 36, 46, 58)
        for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False)
    ]
    for number, x, y in centres:
        chosen = None
        for dx, dy in rings:
            label = axes.annotate(
                str(number), (x, y), xytext=(dx, dy), textcoords="offset points",
                fontsize=7, color=INK, ha="center", va="center", zorder=7,
            )
            box = label.get_window_extent(renderer).expanded(1.15, 1.2)
            if not any(box.overlaps(other) for other in placed):
                chosen = (label, box, dx, dy)
                break
            label.remove()
        if chosen is None:  # crowded beyond every ring: draw it anyway
            label = axes.annotate(str(number), (x, y), xytext=rings[-1], textcoords="offset points",
                                  fontsize=7, color=INK, ha="center", va="center", zorder=7)
            chosen = (label, label.get_window_extent(renderer), *rings[-1])
        label, box, dx, dy = chosen
        placed.append(box)
        if (dx, dy) != rings[0]:
            label.remove()
            axes.annotate(
                str(number), (x, y), xytext=(dx, dy), textcoords="offset points",
                fontsize=7, color=INK, ha="center", va="center", zorder=7,
                arrowprops={"arrowstyle": "-", "color": INK_MUTED, "linewidth": 0.5,
                            "shrinkA": 3, "shrinkB": 4},
            )


_MARKER_NAMES = {"o": "●", "s": "■", "^": "▲", "D": "◆", "v": "▼", "P": "plus", "X": "cross", "*": "star"}


# ---------------------------------------------------------------------------
# The folder
# ---------------------------------------------------------------------------


def report_folder_name(name: str, when: _dt.datetime | None = None) -> str:
    moment = when or _dt.datetime.now()
    from .ims_store import sanitise_key

    stem = sanitise_key(name).replace(" ", "_") if name else "experiment"
    return f"{stem}_analysis_{moment:%Y%m%d_%H%M%S}"


def write_report(outcomes, parent: str | Path, name: str = "") -> tuple[Path, list[str]]:
    """Write the workbook and the figures into a new dated folder under *parent*.

    Returns the folder and a note for every figure that could not be drawn. The
    workbook failing is an error; a figure failing is a note, because the numbers
    are the result and the figures are a view of them.
    """
    from . import analysis as an
    from .exports import export_sheets

    base = Path(parent) / report_folder_name(name or Path(parent).name)
    folder, suffix = base, 1
    while folder.exists():  # two runs in the same second
        suffix += 1
        folder = base.with_name(f"{base.name}-{suffix}")
    folder.mkdir(parents=True)

    sheets = an.workbook_sheets(outcomes)
    export_sheets(sheets, folder / f"{folder.name}.xlsx")
    an.save_cell_outlines(outcomes, folder / CELL_OUTLINES_FILE)
    features = sheets["Region features"]

    notes: list[str] = []
    for column, filename, title, ylabel in (
        ("Objects", "violin_cell_count.png", "Detected cells per region", "Cells"),
        ("Region area (µm²)", "violin_region_area.png", "Region area", "Area (µm²)"),
    ):
        try:
            if not violin_plot(features, column, title, ylabel, folder / filename):
                notes.append(f"{filename}: no values to plot")
        except Exception as exc:
            logger.exception("could not draw %s", filename)
            notes.append(f"{filename}: {exc}")
    try:
        reason = cell_pca_plot(outcomes, folder / "pca_cells.png")
        if reason:
            notes.append(f"pca_cells.png: {reason}")
    except Exception as exc:
        logger.exception("could not draw the cell PCA")
        notes.append(f"pca_cells.png: {exc}")
    logger.info("analysis report written to %s", folder)
    return folder, notes
